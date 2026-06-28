"""POC sandbox: execute prompt-style chain stages.

T-85 Phase A.1 — implements the LLM dispatch + simplified placeholder
substitution for production prompts loaded from ai_prompt_section.

Pragmatic scope:
- Load prompt section text (concatenated) as the system message
- Append structured context blocks (## Customer profile, ## Recent dialogue,
  ## Anchors, ## Previous chain outputs) as a user message
- Dispatch to llm_provider via POC's existing Anthropic + Gemini clients
- Parse response per response_format

Does NOT implement:
- Full 22-source-type placeholder resolver (production has this in
  assistant.py via deeply-integrated db_unit + monitoring)
- Parallel group dispatch (sequential only)
- exit_fn / transform_fn callbacks (logged, not executed)

Production parity is partial — about 80% coverage on common cases.
For Phase A POC purposes, the chain runner produces multi-stage execution
where each stage SEES the previous stages' outputs, mirroring the production
chain's data-flow semantics even if the prompt format isn't pixel-perfect.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import anthropic
from google import genai
from google.genai import types as genai_types

from db import open_conn
from chain_runner import ChainStage, ChainContext

log = logging.getLogger(__name__)


# ── LLM clients (re-uses POC's existing API keys) ───────────────────────────

_anthropic = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
_gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY_1") or os.environ.get("GEMINI_API_KEY"))


# Map production llm_provider strings → (provider_kind, model_name)
_PROVIDER_MAP: dict[str, tuple[str, str]] = {
    "gemini": ("gemini", "gemini-2.5-pro"),
    "gemini_light": ("gemini", "gemini-2.5-flash"),
    "gemini-flash": ("gemini", "gemini-2.5-flash"),
    "gemini-flash-latest": ("gemini", "gemini-2.5-flash"),
    "anthropic": ("anthropic", "claude-sonnet-4-5-20250929"),
    "anthropic_haiku": ("anthropic", "claude-haiku-4-5-20251001"),
    # programmatic stages don't dispatch through here
}


# POC latency optimization: stages where Pro reasoning isn't load-bearing get
# downgraded to Flash. Saves 2-4s per call. ONLY applied when the stage's
# declared provider is gemini (Pro). Production keeps Pro for these — we
# downgrade in the POC because the demo UX tradeoff (lower latency) outweighs
# the marginal quality difference on these specific stages.
#
# DO NOT add prompt_manager, prompt_build_answer, or prompt_signal_analysis to
# this list — they are the load-bearing reasoning stages and need Pro.
_FAST_STAGE_OVERRIDES: set[str] = {
    "prompt_validate_urls",          # url validity check
    "prompt_validate_calculation",   # arithmetic check
    "prompt_price_tracker",          # historical pricing lookup
    "prompt_price_adapter",          # already gemini-flash-latest
}



# ── Prompt loading + caching ────────────────────────────────────────────────

_PROMPT_TEXT_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 60  # match production's prompt_loader TTL


def load_prompt_text(prompt_name: str, prompt_id: int | None = None) -> str | None:
    """Load and concatenate prompt sections from luna.ai_prompt_section.
    Returns None if not found. Caches with 60s TTL (matching production).

    T-86 fix: prefers prompt_id when provided. Multiple tenants have rows
    with the same prompt_name in luna.ai_prompt; resolving by name picks
    the wrong one. The chain definition carries the correct prompt_id
    via the ai_agent_prompt join, so callers should always pass it.
    """
    now = time.time()
    # Cache key includes prompt_id to disambiguate tenant variants
    cache_key = f"id:{prompt_id}" if prompt_id is not None else f"name:{prompt_name}"
    cached = _PROMPT_TEXT_CACHE.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]

    conn = open_conn()
    try:
        with conn.cursor() as cur:
            if prompt_id is None:
                # Legacy fallback: by name (will pick first match — risky)
                cur.execute(
                    "SELECT prompt_id FROM luna.ai_prompt WHERE prompt_name = %s",
                    (prompt_name,))
                row = cur.fetchone()
                if not row:
                    _PROMPT_TEXT_CACHE[cache_key] = (now + _CACHE_TTL, None)
                    return None
                prompt_id = row["prompt_id"]
            cur.execute("""
                SELECT text FROM luna.ai_prompt_section
                WHERE prompt_id = %s AND is_draft = 0
                ORDER BY section_id
            """, (prompt_id,))
            sections = cur.fetchall()
    finally:
        conn.close()

    if not sections:
        _PROMPT_TEXT_CACHE[cache_key] = (now + _CACHE_TTL, None)
        return None

    text = "\n\n".join(s["text"] or "" for s in sections)
    _PROMPT_TEXT_CACHE[cache_key] = (now + _CACHE_TTL, text)
    return text


# ── Context-block builder ───────────────────────────────────────────────────

def _render_directive_wrapper(directive: dict) -> str:
    """Phase 4.1 — supervisor directive wrapper (2026-05-10).

    Render the supervisor's structured directive with explicit authoritative
    framing, so the actor reads it as a turn-specific instruction set rather
    than as just-another-stage-output buried in `previous_chain_outputs`.

    Returns empty string if the directive is empty/missing."""
    if not isinstance(directive, dict):
        return ""

    strategy = directive.get("strategy") or {}
    knowledge = directive.get("knowledge") or {}
    customer_state = directive.get("customer_state") or {}
    signal_analysis = directive.get("signal_analysis") or {}
    audit = directive.get("audit") or {}

    must_say = knowledge.get("must_say") or []
    must_not_say = knowledge.get("must_not_say") or []

    primary_strategy = strategy.get("primary") or "?"
    tone = strategy.get("tone") or "?"
    rationale = (audit.get("rationale_summary") or "")[:280]

    primary_signal = signal_analysis.get("primary_signal") or "?"
    response_template = (signal_analysis.get("response_template") or "")[:280]

    objection = customer_state.get("objection_category") or "none"
    sentiment = customer_state.get("sentiment") or "?"
    commitment_level = customer_state.get("commitment_level")

    lines = [
        "## SUPERVISOR DIRECTIVE — AUTHORITATIVE",
        "(Follow these turn-specific rules. They override your default judgment.)",
        "",
        "### Strategy this turn",
        f"- Primary: {primary_strategy}",
        f"- Tone: {tone}",
    ]
    if rationale:
        lines.append(f"- Rationale: {rationale}")

    if must_say:
        lines.append("")
        lines.append("### MUST say (incorporate explicitly in your reply)")
        for item in (must_say if isinstance(must_say, list) else [must_say]):
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content") or json.dumps(item, default=str)[:200]
            else:
                txt = str(item)[:300]
            lines.append(f"- {txt}")

    if must_not_say:
        lines.append("")
        lines.append("### MUST NOT say")
        for item in (must_not_say if isinstance(must_not_say, list) else [must_not_say]):
            if isinstance(item, dict):
                txt = item.get("text") or item.get("rule") or item.get("content") or json.dumps(item, default=str)[:200]
            else:
                txt = str(item)[:300]
            lines.append(f"- {txt}")

    lines.append("")
    lines.append("### Customer state to address")
    lines.append(f"- Objection: {objection}")
    lines.append(f"- Sentiment: {sentiment}")
    if commitment_level is not None:
        lines.append(f"- Commitment level: {commitment_level}/5")

    lines.append("")
    lines.append("### Signal analysis")
    lines.append(f"- Primary signal: {primary_signal}")
    if response_template:
        lines.append(f"- Recommended response shape: {response_template}")

    lines.append("")
    lines.append("If the directive conflicts with your default behavior, the directive wins.")

    return "\n".join(lines)


# Per-stage system-prompt suffixes (Option 3 — Phase 4.1, 2026-05-10).
# Appended to the prod-loaded prompt at dispatch time so the actor's instruction
# set EXPECTS the supervisor directive in its user message. The prod prompt was
# authored for vanilla Luna and knows nothing about a supervisor concept; this
# closes the architectural gap without modifying the prod prompt itself.
_SYSTEM_SUFFIX_BUILD_ANSWER = """

---

You operate within a multi-stage chain that includes a Strategic Supervisor.
The user message may contain a "## SUPERVISOR DIRECTIVE — AUTHORITATIVE" block
at the top. When that block is present, treat its strategy choice, MUST say
rules, and MUST NOT say rules as overriding your default judgment for THIS turn.
The supervisor has analyzed the customer's profile, conversation state, and
historical patterns to produce the directive. Honor it. If a directive rule
conflicts with what you would otherwise say, follow the directive.

LANGUAGE MATCHING (mandatory):
Always respond in the SAME language as the customer's MOST RECENT message.
If the customer writes in English, respond in English. If the customer writes
in Hebrew, respond in Hebrew. Never switch languages unilaterally. Never mix
two languages within one reply. The supervisor's directive and any historical
exemplars do NOT override this rule — language tracking always follows the
customer's most recent turn.
"""


def build_context_blocks(ctx: ChainContext, max_dialog_turns: int = 12) -> str:
    """Compose named-section blocks for the user message. Mirrors production's
    practice of appending structured data as named sections so the LLM can
    reference them ("the current_chat below shows..." style instructions in
    the prompt sections then have actual data to point at).

    Phase 4.1 (2026-05-10): the supervisor's structured directive — when present
    in `prompt_signal_analysis_combined.directive` — is now rendered FIRST as
    a wrapped, authoritative block. Previously it sat as a JSON dump inside
    the generic `previous_chain_outputs` section, with no instruction-priority
    framing for the actor."""
    blocks = []

    # 2026-05-10 — Per-turn language directive at the very top of the user
    # message. The build_answer system suffix already carries a language rule,
    # but it's buried at the tail of a long DB-loaded prompt and gets
    # over-ridden when the dialog history contains turns in another language.
    # Pinning the rule in the FIRST user-message block (above even the
    # supervisor directive) gives it instruction-priority that the LLM can't
    # miss. The customer's most-recent message is authoritative.
    last_cust_text = ""
    for m in reversed(ctx.dialog or []):
        if m.get("role") == "customer" and m.get("text"):
            last_cust_text = m["text"]
            break
    if last_cust_text:
        # Hebrew block: U+0590..U+05FF
        cust_is_hebrew = any("֐" <= ch <= "׿" for ch in last_cust_text)
        cust_lang = "Hebrew" if cust_is_hebrew else "English"
        blocks.append(
            f"## ⚠ LANGUAGE — CRITICAL THIS TURN (overrides everything else)\n"
            f"\n"
            f"The customer's MOST RECENT message is in **{cust_lang}**. Your reply\n"
            f"text MUST be in {cust_lang}. This rule applies regardless of:\n"
            f"- Historical dialog turns being in another language\n"
            f"- The supervisor directive's must_say being in another language\n"
            f"- The tenant's typical-customer language (Libra typically serves\n"
            f"  Hebrew customers — irrelevant; the customer just typed in {cust_lang})\n"
            f"- Script-template phrasing in any other language\n"
            f"\n"
            f"If the directive's must_say is in a different language than {cust_lang},\n"
            f"TRANSLATE it preserving the strategic intent. Do NOT echo the directive\n"
            f"verbatim if it's in the wrong language. Do NOT mix two languages within\n"
            f"one reply.\n"
        )

    # Phase 4.1 — directive wrapper at top of context (Option 1).
    sa_combined_pre = ctx.previous_responses.get("prompt_signal_analysis_combined") or {}
    directive_pre = sa_combined_pre.get("directive") if isinstance(sa_combined_pre, dict) else None
    if isinstance(directive_pre, dict) and directive_pre:
        wrapper = _render_directive_wrapper(directive_pre)
        if wrapper:
            blocks.append(wrapper)

    # 2026-05-11 — Retrieval-augmented imitation (option B / RAG-of-exemplars).
    # When POC_EXEMPLAR_RAG=1, the stage_exemplar_retrieval stage has
    # populated ctx.previous_responses["prompt_exemplar_retrieval"] with
    # a rendered exemplars block — real won-deal Customer→Agent pairs
    # similar to the current customer turn. Inject as IMITATION GUIDES.
    # This complements (not replaces) the directive; eventually it may
    # replace the script_library scaffold.
    ex_retrieval = ctx.previous_responses.get("prompt_exemplar_retrieval") or {}
    if isinstance(ex_retrieval, dict) and ex_retrieval.get("rendered_block"):
        blocks.append(ex_retrieval["rendered_block"])

    p = ctx.opp_meta
    profile = (
        f"## customer_profile\n"
        f"- tenant: {p.get('company','?')} / {p.get('opp_type','?')}\n"
        f"- customer_name: {p.get('customer_name') or 'Customer'}\n"
        f"- motivator: {p.get('primary_motivator','?')}\n"
        f"- decision_logic: {p.get('decision_logic','?')}\n"
        f"- trust_level: {p.get('trust_level','?')}\n"
        f"- communication_style: {p.get('communication_style','?')}\n"
        f"- objection_pattern: {(p.get('objection_pattern') or '')[:200]}\n"
    )
    blocks.append(profile)

    if ctx.anchors:
        # Detect anchor pack flavor — T-81 Libra has economic-frame keys;
        # T-86 Heavys has product-catalog keys (different schema).
        if (ctx.anchors.get("_source") or "").startswith("CG workspace=Heavys"):
            from heavys_anchors import render_anchor_block
            heavys_block = render_anchor_block(ctx.anchors, max_per_field=900)
            if heavys_block:
                blocks.append(heavys_block)
        else:
            anchor_lines = ["## anchors (economic reference frame — T-81)"]
            for k in ("last_year_price_nis", "current_quoted_price_nis",
                       "claimed_increase_pct", "actual_market_yoy_change_pct",
                       "market_avg_for_segment_nis", "max_discount_pct_internal",
                       "coverage_summary", "loyalty_years", "claims_count",
                       "profile_appropriate_opening_nis",
                       "profile_appropriate_opening_reason"):
                if ctx.anchors.get(k) is not None:
                    anchor_lines.append(f"- {k}: {ctx.anchors[k]}")
            blocks.append("\n".join(anchor_lines))

    if ctx.business_rules:
        blocks.append(f"## business_rules\n{ctx.business_rules[:3000]}")

    if ctx.dialog:
        recent = ctx.dialog[-max_dialog_turns:]
        lines = ["## current_chat"]
        for m in recent:
            role = "Customer" if m.get("role") == "customer" else "Agent"
            text = (m.get("text") or "").replace("\n", " ").strip()[:300]
            if text:
                lines.append(f"{role}: {text}")
        blocks.append("\n".join(lines))

    if ctx.previous_responses:
        # Reference outputs from earlier stages (production prompts can refer
        # to "previous_answer" or specific stage outputs by name).
        #
        # Per-key char budgets — the supervisor's directive (signal_analysis,
        # binding rules, objection_handling.response_template) was being
        # truncated at 600 chars by the previous default, which silently
        # stripped the "don't retreat / stack non-price value" guidance from
        # build_answer's context. Bug found in repro of session 6bf083e8.
        # Larger budgets here are still well under build_answer's typical
        # context window (~30k tokens for Gemini 2.5 Pro).
        PREV_RESPONSE_BUDGETS = {
            "prompt_signal_analysis_combined": 4500,
            "prompt_manager":                  2500,
            "prompt_build_answer":             2500,
            "prompt_anchor_load":              1500,
            "prompt_anti_staircase_gate":      800,
            "prompt_retreat_passthrough_gate": 600,
        }
        DEFAULT_BUDGET = 1200
        prev_lines = ["## previous_chain_outputs"]
        for name, val in ctx.previous_responses.items():
            if name.startswith("_"):
                continue  # internal markers (e.g. _staircase_correction) aren't context
            if isinstance(val, dict) and val.get("_placeholder"):
                continue  # skip placeholder stubs
            budget = PREV_RESPONSE_BUDGETS.get(name, DEFAULT_BUDGET)
            if isinstance(val, (dict, list)):
                snippet = json.dumps(val, default=str)[:budget]
            else:
                snippet = str(val)[:budget]
            prev_lines.append(f"### {name}\n{snippet}")
        if len(prev_lines) > 1:
            blocks.append("\n".join(prev_lines))

    # T-83 staircase regenerate-loop hook — when chain_runner has fired the
    # anti-staircase gate and is asking us to regenerate prompt_build_answer,
    # it sets ctx.previous_responses["_staircase_correction"]. We append it
    # as a final, prominent block so the LLM sees it last.
    correction = ctx.previous_responses.get("_staircase_correction")
    if correction:
        blocks.append(correction)

    # R33 (2026-05-06) — Manager-escalation framing was originally injected
    # here, but this path only runs on Mode 1b (fresh LLM directive). Most
    # Libra c5 turns hit Mode 1a (cached directives) which skips the chain's
    # build_answer LLM call entirely — so R33 never activated. Moved to
    # luna_actor.generate (the actor's LLM call) where it fires on every turn.

    # CR-PS Phase 4 (2026-05-07) — cohort-conditioned won-deal precedents.
    # Pre-fetched once per turn in replayer._live_turn (right-panel only),
    # carried through opp_meta["_cohort_precedent_block"]. We append here so
    # the prod-chain prompt_build_answer path sees the block; the legacy
    # regenerate-loop path consumes the same block via the cohort_precedent_block
    # kwarg on luna_actor.generate. One fetch, two consumers — and because both
    # invocations land in the same SQLite-LRU 5min/256-entry cache, the
    # single fetch is the only network hop.
    cohort_block = (ctx.opp_meta or {}).get("_cohort_precedent_block")
    if cohort_block:
        blocks.append(cohort_block)

    # Tier 2 concrete-move execution block (Phase 1 of strategy-enum-extension,
    # 2026-05-03). When the supervisor's directive includes a concrete_move,
    # render the literal execution instruction so prompt_build_answer can
    # execute the named move directly instead of having to derive what to do
    # from the abstract Tier 1 primitive + must_not_say rules.
    sa_combined = ctx.previous_responses.get("prompt_signal_analysis_combined") or {}
    directive = sa_combined.get("directive") if isinstance(sa_combined, dict) else None
    if isinstance(directive, dict):
        try:
            # Phase 2: tenant-aware loader (replaces the Phase 1 hardcoded
            # `concrete_moves.py` paths). Move definitions now live in
            # `data/concrete_moves/{tenant}.yaml` + `_base.yaml`.
            from concrete_moves_loader import render_move_for_build_answer
            tenant = ctx.opp_meta.get("company") if ctx.opp_meta else None
            move_block = render_move_for_build_answer(directive.get("strategy"),
                                                       tenant=tenant)
            if move_block:
                blocks.append(move_block)
        except ImportError:
            pass

    return "\n\n".join(blocks)


# ── LLM dispatch ────────────────────────────────────────────────────────────

async def dispatch_llm(provider_str: str, system_text: str, user_text: str,
                       response_format: str | None = None) -> str:
    """Call the correct LLM per the provider string. Returns raw text.
    Returns empty string on error."""
    kind_model = _PROVIDER_MAP.get(provider_str)
    if not kind_model:
        log.warning("chain_executor: unknown llm_provider=%s; falling back to gemini-2.5-pro",
                       provider_str)
        kind_model = ("gemini", "gemini-2.5-pro")
    kind, model = kind_model

    try:
        if kind == "anthropic":
            msg = await _anthropic.messages.create(
                model=model,
                max_tokens=2000,
                system=system_text,
                messages=[{"role": "user", "content": user_text}],
            )
            return msg.content[0].text if msg.content else ""
        elif kind == "gemini":
            full_prompt = f"{system_text}\n\n---\n\n{user_text}"
            import asyncio
            loop = asyncio.get_running_loop()
            def _sync():
                return _gemini.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=4096,
                    ),
                )
            resp = await loop.run_in_executor(None, _sync)
            return (resp.text or "").strip()
    except Exception as e:
        log.warning("chain_executor: LLM dispatch failed (provider=%s): %s",
                       provider_str, e)
        return ""

    return ""


# ── Response parsing ────────────────────────────────────────────────────────

def parse_response(raw: str, response_format: str | None) -> Any:
    """Parse LLM response per response_format hint."""
    if not raw:
        return None
    text = raw.strip()
    if response_format == "json":
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            log.warning("chain_executor: JSON parse failed: %s; returning raw text",
                          e)
            return {"_parse_error": str(e), "_raw": text[:500]}
    return text  # default: text


# ── Prompt-stage executor ───────────────────────────────────────────────────

async def execute_prompt_stage(stage: ChainStage, ctx: ChainContext) -> Any:
    """Execute a single prompt-style chain stage. Returns the parsed response.

    Steps:
      1. Load prompt text from ai_prompt_section
      2. Build context blocks from ChainContext
      3. Dispatch to LLM per stage.llm_provider
      4. Parse per stage.response_format
    """
    system_text = load_prompt_text(stage.prompt_name, prompt_id=stage.prompt_id)
    if system_text is None:
        log.warning("chain_executor: prompt %s (id=%s) not found in DB",
                       stage.prompt_name, stage.prompt_id)
        return None

    # Phase 4.1 (2026-05-10) — Option 3: append supervisor-awareness suffix
    # to the actor's system prompt so the LLM's instruction-set EXPECTS the
    # SUPERVISOR DIRECTIVE block in the user message. Only for prompt_build_answer
    # (the actor's main reply generator).
    if stage.prompt_name == "prompt_build_answer":
        system_text = system_text + _SYSTEM_SUFFIX_BUILD_ANSWER

    user_text = build_context_blocks(ctx)

    # POC latency optimization: downgrade declared Pro provider → Flash for
    # stages whose reasoning quality isn't load-bearing (see _FAST_STAGE_OVERRIDES
    # for the rationale and the explicit do-not-downgrade list).
    effective_provider = stage.llm_provider
    if stage.prompt_name in _FAST_STAGE_OVERRIDES and stage.llm_provider == "gemini":
        effective_provider = "gemini-flash-latest"
        log.info("chain_executor: downgrading %s gemini-pro → flash (POC speedup)",
                 stage.prompt_name)

    raw = await dispatch_llm(effective_provider, system_text, user_text,
                               stage.response_format)
    parsed = parse_response(raw, stage.response_format)
    return parsed
