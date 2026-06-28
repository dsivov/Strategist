"""Luna actor wrapper — generates the next agent message.

Same function signature for both POC panels:
  await generate(opp_meta, dialog_history, directive=None) -> str

When `directive` is None: behaves as "Original Luna" (no supervisor).
When `directive` is a Strategic Directive dict: behaves as supervisor-augmented Luna.

Implementation: Gemini 2.5 Pro — the SAME model Luna prod uses for
`prompt_build_answer` (per `ai_prompt.llm_provider='gemini'` mapped to
`_PRO_MODELS=['gemini-2.5-pro', 'gemini-3.1-pro-preview']` in
`luna/new_core/gemini_chat.py`).

Why Gemini, not Sonnet: Luna's production actor is more grounded — less likely
to invent specific prices/discounts that aren't in business rules. Matching the
production model means the POC's left-panel "Original Luna" behaves like the
real product.

Full Luna prompt-chain (prompt_context → prompt_manager → prompt_build_answer)
collapsed into single Gemini call for POC simplicity (documented in POC.md §7).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re

import anthropic
from google import genai
from google.genai import types as genai_types

log = logging.getLogger(__name__)

# Same model Luna prod uses for prompt_build_answer
ACTOR_MODEL = "gemini-2.5-pro"
# Anthropic fallback — used when Gemini returns 503 UNAVAILABLE or empty after
# the 16384-budget retry. 2026-05-06 added because Gemini 2.5 Pro had a
# multi-hour high-demand spike that killed 100% of R32 v2 batch sessions
# (`agent_refused_to_continue` from empty actor responses on both panels).
ACTOR_FALLBACK_MODEL = "claude-sonnet-4-5-20250929"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY_1") or os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY (or GEMINI_API_KEY_1) required for actor")

_client = genai.Client(api_key=GEMINI_API_KEY)
_anthropic = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# Luna-style actor system prompt — adapted from ai_assistant/assistant.py
ACTOR_SYSTEM_BASE = """You are Luna, an AI agent for {tenant}. You are messaging a real
customer over WhatsApp. Your job is to convert the customer to a sale (renewal,
purchase, or whichever conversion makes sense for this tenant).

# Tenant-specific context
- Tenant: {tenant} ({domain_desc})
- Opportunity type: {opp_type}{opp_type_behavioral_note}
- Brand voice: warm, concise, professional, never pushy.
- CRITICAL LANGUAGE RULE: Use the SAME language as the most recent messages in
  the conversation. If the prior turns are in English, respond in English. If
  Hebrew, respond in Hebrew. DO NOT switch languages mid-conversation. DO NOT
  translate the customer's words.
- # LEAD WITH SUBSTANCE — open with the most important thing (a price, a number,
  a direct question, an answer). NEVER open with "I appreciate", "I understand",
  "Of course", "That's a great question", "I've reviewed". These are bot
  preambles; real human salespeople just answer.
- # HARD LENGTH RULE — ≤35 words, max 2 sentences per message.
  Real won conversations from {tenant} prod data: median 12 words, p90 37 words.
  Long agent messages trigger "bot trying too hard" pattern → drop trust.
  If multiple facts apply, pick the ONE most important and save the rest
  for the next turn (chat is multi-turn — don't pack everything in one).
  Lists, multi-paragraph offers, and stacked value props are FORBIDDEN.

# Real won-conversation style examples (match this length and rhythm)
- "Together it's 4,816 NIS."  (4 words)
- "It's regular, the call center's phone."  (6 words)
- "An attempt of theft is not hit and run, but let's try help with the price."  (16 words)
- "Okay, I'll check this. If I match the price, can we move forward and renew?"  (15 words)
- "Hi, how are you? Can I renew the insurance? Happy to help 🙏"  (12 words)
NEVER write a 4-paragraph reply. NEVER recite multiple facts in one message.
Acknowledge briefly + ask ONE thing OR offer ONE thing. That's the pattern.
- Don't be obviously AI. Sign as a normal agent (e.g., "Nofar from Libra" if a name fits).
- IF the customer has clearly disengaged or said goodbye, your reply should be
  a brief warm closing only ("All the best, take care!") — do NOT keep pitching.

# CRITICAL: NO HALLUCINATION (most important rule)
- NEVER invent specific prices, discount amounts, discount codes, product
  features, return policies, shipping details, or any other factual claim that
  is not explicitly grounded in the business rules below.
- If you don't have a specific number/code/fact: say "let me check on that"
  or "I'll need to verify the exact details" or escalate. Never make it up.
- This is a hard rule. The customer relationship depends on it.

# Customer profile
- Motivator: {motivator}
- Decision logic: {decision_logic}
- Trust level: {trust_level}
- Communication style: {communication_style}
- Objection pattern (historical): {objection_pattern}

# Business rules (selected, relevant to current turn)
{business_rules_excerpt}

# Constraints (always)
- Never fabricate discounts, prices, or product details not in the rules.
- Never repeat the same offer at the same price (B44 duplicate-offer prevention).
- If customer signals "I want to think" or "I'll check elsewhere" → soft retention check, not pressure.
- If customer goes silent after a clear decline, do NOT spam reminders.
"""

# T-80 minimal directive injection (2026-05-02). Prior verbose template was
# directly displacing the system-prompt's "be short, warm, concise" rule
# with multi-section facts/rules/rationale prescriptions, causing R-side
# brochure-paragraph regression vs L-side natural prose. Hypothesis: the
# strategy primitive + tone is the only useful guidance; everything else
# (must_say facts, rules-by-id, rationale paragraph) was actively harmful.
# This minimal form re-asserts the brand voice rule on top of the hint.
DIRECTIVE_INSTRUCTION = """

# Strategist hint (this turn only)
Suggested strategy: {primary_strategy}. Suggested tone: {tone}.

The brand voice's HARD LENGTH RULE (≤35 words, max 2 sentences) is ABSOLUTE
and overrides everything else, including this hint. Don't list facts, don't
recite a rationale, don't repeat the strategy name in the message. If the
strategy implies multiple facts/parameters, pick THE ONE that creates the
strongest next-turn engagement; save the others for follow-up turns. The
hint shapes WHICH MOVE you make this turn, not HOW MUCH YOU PACK INTO IT.
"""


def _format_must_not_say(items: list) -> str:
    if not items:
        return "  (none)"
    out = []
    for item in items:
        if isinstance(item, dict):
            out.append(f"  - {item.get('text','')} (reason: {item.get('reason','')})")
        else:
            out.append(f"  - {item}")
    return "\n".join(out)


def _format_must_say(items: list) -> str:
    if not items:
        return "  (none)"
    return "\n".join(f"  - {item}" for item in items)


def _trim_business_rules(rules: str, max_chars: int = 6000) -> str:
    """Pull the most relevant subset of business rules for the actor.
    For POC: take first N chars (which contain the highest-priority rules in
    Luna's actual file ordering)."""
    if not rules:
        return "(no business rules loaded)"
    return rules[:max_chars] + ("\n…[truncated]" if len(rules) > max_chars else "")


def _domain_desc(tenant: str) -> str:
    return {
        "Libra": "B2C auto insurance renewal in Israel; conversion = customer agrees to renew + provides CC last-4 digits",
        "Heavys": "B2C e-commerce (premium headphones); conversion = customer completes the order via the cart link",
        "HoneyBook": "B2B SaaS subscription (creative-business platform); conversion = trial-to-paid plan",
        "Cleandot": "B2C e-commerce (cleaning products); conversion = customer completes the order",
        "Panda": "B2C e-commerce (mattresses); conversion = customer completes the order",
    }.get(tenant, "B2C messaging")


def _opp_type_behavioral_note(opp_type: str) -> str:
    """T-82 — opp-type-aware behavioral guidance for the agent. Mirrors the
    coherence-rules added to the customer simulator on 2026-05-01 so BOTH
    sides of the negotiation share the same framing of the opportunity
    state. Empirically the agent's text was inferring opp_type from dialog
    only; explicit declaration grounds it more reliably."""
    t = (opp_type or "").lower()
    if "renewal" in t or "renew" in t:
        return ("\n  → Customer is an EXISTING customer at renewal. They have a "
                "prior relationship, prior premium, and likely competitor quotes. "
                "Match their reference frame; don't pitch as if this is a cold sale.")
    if any(k in t for k in ("abandoned cart", "cart abandon", "abandon cart")):
        return ("\n  → Customer added a product to cart and walked away. They had "
                "buy intent. Address what changed since cart-add (price hesitation, "
                "second thoughts, distraction); don't re-pitch the product from scratch.")
    if "upsell" in t or "cross" in t or "expansion" in t:
        return ("\n  → Customer already has a base product; this is an upgrade "
                "conversation. Anchor on their current setup; surface incremental value.")
    if "trial" in t:
        return ("\n  → Customer is mid-trial → paid conversion. Lean on their "
                "actual experience with the trial; don't over-explain features.")
    if "purchasing assistance" in t or "search_catalog" in t or "browse" in t:
        return ("\n  → Customer reached out for help selecting. Probe intent "
                "(buy-ready vs browsing) before pushing a specific product.")
    if "review" in t:
        return ("\n  → Post-purchase review request, not a sales conversation. "
                "Don't pitch; gather feedback.")
    return ""


# 2026-05-05 — Post-generation length enforcement. Gemini 2.5 Pro requires
# thinking mode (8192 token budget) so max_output_tokens can't enforce
# brevity. Empirically the actor still produces 70+ word paragraphs
# despite explicit prompt rules. We deterministically truncate at
# sentence boundary to enforce the 35-word target from real won-conversation
# data. Sentence-boundary cut preserves grammar; word cap is the safety net.
LENGTH_CAP_WORDS = 35

# 2026-05-05 — preamble phrases. Sentences starting with these are FILLER
# (apology / acknowledgement / promise-to-elaborate) without substance.
# When truncating, we skip them so the budget goes to the actual breakdown
# instead of "Let me break it down. Shall we proceed?" — the bug case where
# the customer correctly noted "you keep saying break it down without doing it"
# (session c258b6b6, 2026-05-05).
_PREAMBLE_PATTERNS = [
    "you are absolutely right", "you're absolutely right",
    "let me break", "i appreciate", "i understand",
    "thank you for", "of course",
    "i apologize for", "i'm sorry for",
    "great question", "that's a great question",
    "let me clarify", "let me explain",
    "to give you", "to clarify",
    # 2026-05-05 — extended after b9be41bc session: vague meta-acknowledgments
    # that occupy budget without adding substance
    "it's a valid point", "that's a fair point", "that's a valid point",
    "i see your concern", "i see your point", "fair point",
    "absolutely", "certainly", "definitely",
    "i hear you", "i totally understand",
    "of course, i", "of course i",
    "you make a good point", "you make a great point",
    "happy to help", "i'm happy to",
]

def _is_preamble(sentence: str) -> bool:
    """Detect filler-preamble sentences (no substance — only acknowledgement
    or promise-to-elaborate). The truncator skips these so substance survives."""
    s = (sentence or "").lower().strip()
    if not s:
        return False
    return any(s.startswith(p) for p in _PREAMBLE_PATTERNS)


# 2026-05-05 — Substance detection (Bug fix b9be41bc).
# Truncator's preamble-skip handles obvious openers, but "It's a valid point"
# / "Does looking at it from this angle..." are vague middle-of-message
# meta. Substance detection is the COMPLEMENT: identify sentences that
# carry concrete facts (numbers, prices, specific value claims). When the
# budget is tight, substance-bearing sentences are preserved before
# abstract meta-questions.
_SUBSTANCE_PATTERNS = [
    r"\b\d{2,5}\s*(?:NIS|nis|₪|shekels?|USD|dollars?|\$)\b",  # price
    r"\$\s?\d{1,4}\b",                                         # dollar amount
    r"\b\d{1,3}\s*%",                                          # percentage
    r"\bcode\s*[:]?\s*[A-Z][A-Z0-9]{3,}\b",                    # promo code
    r"\b\d+\s*(?:installments?|payments?|months?)\b",          # payment plan
    r"\b\d{1,3}\s*(?:hours?|hrs?|days?)\b",                    # time-bound
    # Specific value-prop keywords commonly found in the won-deal corpus
    r"\b(?:cashback|warranty|coverage|deductible|loyalty|installment|"
    r"librot|bonus|free shipping|return policy|14-day|30-day)\b",
]
_SUBSTANCE_REGEX = re.compile("|".join(_SUBSTANCE_PATTERNS), re.IGNORECASE)


def _has_substance(sentence: str) -> bool:
    """True if a sentence carries concrete facts (price, percentage, promo,
    specific value-prop). Substance sentences are preserved over abstract
    meta during truncation."""
    if not sentence: return False
    return bool(_SUBSTANCE_REGEX.search(sentence))


def _enforce_length(text: str) -> tuple[str, bool]:
    """Truncate text to ≤LENGTH_CAP_WORDS preserving GRAMMATICAL boundaries
    AND, when possible, the LAST sentence (typically the close question or
    CTA — the substance most worth keeping). Returns (text, was_truncated).
    Real won-conversation Libra agents: median=12, p90=37 words; cap at 35.

    Strategy:
      1) If full text fits, no change.
      2) Walk sentences from the FRONT, keep what fits.
      3) If the LAST sentence is a question or close-pattern AND we have
         budget left, splice it in (after the front-fit sentences) so the
         CTA survives.
      4) If nothing fits, return empty + flag (caller can fall back).
    """
    if not text:
        return text, False
    words = re.split(r"\s+", text.strip())
    if len(words) <= LENGTH_CAP_WORDS:
        return text, False

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return text, False

    # 2026-05-05 — Priority-ordered front-fit (Bug fix b9be41bc).
    # Priority: SUBSTANCE > CTA > non-substance prose > preamble.
    # We DON'T pre-reserve CTA budget — that meant substance got starved
    # when CTA was long. Instead, take substance first (using full cap),
    # then add CTA if it still fits, then fill with non-preamble prose,
    # finally fall back to preamble. Sentence order preserved.
    last = sentences[-1].strip() if sentences else ""
    last_w = len(re.split(r"\s+", last)) if last else 0
    is_cta = bool(last) and (
        last.endswith("?") or any(
            kw in last.lower() for kw in
            ("renew", "go ahead", "proceed", "let me know", "let's do",
             "shall we", "send the link", "are you in")
        )
    )
    # All sentences (including last) are eligible for the substance pass.
    out_sentences: list[tuple[int, str]] = []
    out_words = 0
    used = set()

    # Pass 1: SUBSTANCE-bearing non-preamble sentences (highest priority)
    # Use FULL cap budget (no upfront CTA reservation).
    for i, s in enumerate(sentences):
        if _is_preamble(s): continue
        if not _has_substance(s): continue
        sw = len(re.split(r"\s+", s.strip()))
        if out_words + sw > LENGTH_CAP_WORDS: continue
        out_sentences.append((i, s.strip()))
        out_words += sw
        used.add(i)

    # Pass 2: CTA — reserve only AFTER substance is in. If CTA fits in
    # remaining budget, splice it. Otherwise drop it (substance is the
    # higher priority).
    if is_cta and (len(sentences) - 1) not in used:
        if out_words + last_w <= LENGTH_CAP_WORDS:
            out_sentences.append((len(sentences) - 1, last))
            out_words += last_w
            used.add(len(sentences) - 1)

    # Pass 3: other non-preamble, non-substance prose (soft fillers)
    for i, s in enumerate(sentences):
        if i in used: continue
        if _is_preamble(s): continue
        sw = len(re.split(r"\s+", s.strip()))
        if out_words + sw > LENGTH_CAP_WORDS: continue
        out_sentences.append((i, s.strip()))
        out_words += sw
        used.add(i)

    # Pass 4 (fallback): preamble — only if we'd otherwise return empty
    if not out_sentences:
        for i, s in enumerate(sentences):
            sw = len(re.split(r"\s+", s.strip()))
            if out_words + sw > LENGTH_CAP_WORDS: break
            out_sentences.append((i, s.strip()))
            out_words += sw
            used.add(i)

    # Restore original sentence order
    out_sentences.sort(key=lambda x: x[0])
    out_sentences = [s for _, s in out_sentences]

    if out_sentences:
        return " ".join(out_sentences).strip(), True
    # No full sentence fits — hard cut at word cap (rare)
    return " ".join(words[:LENGTH_CAP_WORDS]).strip(), True


def build_actor_system(opp_meta: dict, business_rules: str) -> str:
    opp_type = opp_meta.get("opp_type") or opp_meta.get("type") or "unknown"
    base = ACTOR_SYSTEM_BASE.format(
        tenant=opp_meta.get("company") or "?",
        domain_desc=_domain_desc(opp_meta.get("company") or "?"),
        opp_type=opp_type,
        opp_type_behavioral_note=_opp_type_behavioral_note(opp_type),
        motivator=opp_meta.get("primary_motivator") or "unknown",
        decision_logic=opp_meta.get("decision_logic") or "unknown",
        trust_level=opp_meta.get("trust_level") or "unknown",
        communication_style=opp_meta.get("communication_style") or "unknown",
        objection_pattern=(opp_meta.get("objection_pattern") or "")[:200] or "none recorded",
        business_rules_excerpt=_trim_business_rules(business_rules),
    )
    # T-81 anchor enrichment: surface real ly_price / market_avg /
    # max_discount to the agent so they negotiate with grounded context
    # instead of inventing a "48% market hike" that doesn't match reality.
    anchors = opp_meta.get("anchors") or {}
    if anchors:
        base += _build_anchor_section(anchors)
    return base


def _build_anchor_section(anchors: dict) -> str:
    lines = ["", "# Customer's economic reference frame (use this — don't fabricate market claims)"]
    if anchors.get("last_year_price_nis"):
        lines.append(f"- Last year's premium: {anchors['last_year_price_nis']} NIS")
    if anchors.get("current_quoted_price_nis"):
        lines.append(f"- Current quoted price: {anchors['current_quoted_price_nis']} NIS")
    if anchors.get("actual_market_yoy_change_pct") is not None:
        lines.append(f"- ACTUAL market YoY change in our prod data: {anchors['actual_market_yoy_change_pct']:+.1f}% (NOT a 48% hike — do not fabricate)")
    if anchors.get("market_avg_for_segment_nis"):
        lines.append(f"- Market avg for this customer's vehicle segment: {anchors['market_avg_for_segment_nis']} NIS")
    if anchors.get("max_discount_pct_internal") is not None:
        lines.append(f"- Max internal discretionary discount: {anchors['max_discount_pct_internal']}%")
    if anchors.get("loyalty_years"):
        lines.append(f"- Customer loyalty: {anchors['loyalty_years']} years")
    lines.append("")
    lines.append("# Anti-staircase rule")
    lines.append("- Once you have stated a price within 5% of market_avg or below a competitor offer the customer mentioned, HOLD that price. Do not drop again — multiple price drops damage trust.")
    lines.append("- If customer continues to push, address objections via coverage value, deductibles, or relationship — not further discounts.")
    lines.append("")
    return "\n".join(lines)


def build_directive_section(directive: dict) -> str:
    strat = directive.get("strategy") or {}
    return DIRECTIVE_INSTRUCTION.format(
        primary_strategy=(strat.get("primary") or directive.get("primary_strategy") or "objection_handling"),
        tone=(strat.get("tone") or directive.get("tone") or "professional"),
    )


def render_dialog_for_actor(dialog_history: list[dict], k: int = 12) -> str:
    """dialog_history items: {role: customer|agent, text: str, sequence_number: int}"""
    msgs = dialog_history[-k:]
    parts = []
    for m in msgs:
        role = "Customer" if m.get("role") == "customer" else "Agent"
        text = (m.get("text") or "").replace("\n", " ").strip()[:300]
        if text:
            parts.append(f"{role}: {text}")
    return "\n".join(parts) if parts else "(conversation hasn't started)"


def build_user_prompt(opp_meta: dict, dialog_history: list[dict]) -> str:
    return f"""## Conversation so far (chronological, last 12 turns)
{render_dialog_for_actor(dialog_history)}

## Your task
Write the agent's NEXT message. Customer just spoke last; you reply.

Rules:
- Plain text only. No JSON, no markdown.
- ≤35 words, max 2 sentences. Match the customer's language.
- Don't restate things you already said. Don't repeat the previous offer at same price.
- One thing per message. If you have more to say, save it for the next turn.
"""


async def generate(opp_meta: dict, dialog_history: list[dict],
                   business_rules: str, directive: dict | None = None,
                   forced_language: str | None = None,
                   system_suffix: str | None = None,
                   cohort_precedent_block: str | None = None,
                   ) -> tuple[str, dict]:
    """Generate the next agent message via Gemini 2.5 Pro (Luna prod actor).

    Returns:
      (text, meta) where meta = {input_tokens, output_tokens, latency_ms,
                                  used_directive: bool}
    """
    import time
    system = build_actor_system(opp_meta, business_rules)
    if directive is not None:
        system += build_directive_section(directive)

    # R33 (2026-05-06) — Manager-escalation framing.
    # ORIGINAL wiring went through chain_executor.build_context_blocks, but that
    # path only fires on Mode 1b (fresh LLM directive). Most Libra c5 turns hit
    # Mode 1a (cached directives) where the chain skips the LLM call, so R33
    # never activated. Re-wired here at the actor LLM call so it fires on every
    # turn that reaches the actor — gated on `directive is not None` to keep
    # L-panel (vanilla) clean for A/B comparison.
    escalation_meta = None
    if directive is not None:
        try:
            from staircase_gate import (
                should_inject_escalation_framing,
                render_escalation_framing_block,
                extract_libra_offered_prices,
            )
            # Synthesize panel_concessions from dialog_history's agent-side
            # prices. The chain_runner's tracked panel_concessions isn't
            # threaded down to luna_actor — and we don't need it perfectly
            # accurate, just enough to detect "has the agent quoted before?"
            shadow_concessions = []
            for m in (dialog_history or []):
                if m.get("role") != "agent":
                    continue
                for p in extract_libra_offered_prices(m.get("text") or ""):
                    shadow_concessions.append({"type": "price", "amount": p})
            esc_active, esc_meta = should_inject_escalation_framing(
                dialog=dialog_history,
                panel_concessions=shadow_concessions,
                opp_meta=opp_meta,
            )
            if esc_active:
                # Pull anchors from opp_meta if upstream packed them there
                # (otherwise the renderer falls back to default 15%).
                anchors = opp_meta.get("_anchors") if opp_meta else None
                system += "\n\n" + render_escalation_framing_block(esc_meta, anchors=anchors)
                escalation_meta = esc_meta
                log.info("actor: R33 escalation framing injected — trigger=%s",
                         esc_meta.get("trigger"))
        except Exception as e:
            log.warning("actor: R33 framing injection failed: %s", e)

    # CR-PS Phase 4 — cohort-conditioned won-deal precedents.
    # Injected here (not chain_executor.build_context_blocks) for the same
    # reason as R33: Mode 1a cache hits skip the chain's LLM call but still
    # reach this function. Right-panel-only by virtue of the caller (replayer
    # gates on panel.side == "right" before passing the block).
    if cohort_precedent_block:
        system += "\n" + cohort_precedent_block

    if forced_language == "en":
        system += "\n\nFORCED LANGUAGE: respond in English ONLY for this turn. Do NOT use Hebrew. Even if a prior message used Hebrew, you must reply in English."
    elif forced_language == "he":
        system += "\n\nFORCED LANGUAGE: respond in Hebrew ONLY for this turn. Do NOT use English."
    if system_suffix:
        # T-83 staircase-gate retry hook — caller appends a correction
        # instruction to enforce no-further-concession on regenerate.
        system += system_suffix

    user = build_user_prompt(opp_meta, dialog_history)
    # Gemini convention: system instruction goes via system_instruction; user
    # content is the user prompt.
    full_prompt = f"{system}\n\n---\n\n{user}"

    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    def _sync(budget: int):
        return _client.models.generate_content(
            model=ACTOR_MODEL,
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.7,
                # Gemini 2.5 Pro requires thinking mode. Directive-augmented
                # prompts are larger; thinking eats more budget. 8192 leaves
                # plenty of room for both thinking AND a 1-3 sentence reply.
                max_output_tokens=budget,
            ),
        )
    # First attempt with 8192. If Gemini still emits empty (used all on
    # thinking), retry once with 16384.
    # Bug fix 2026-05-01: pre-initialize resp=None so the post-try `getattr(resp, …)`
    # doesn't NameError when the API call itself raises before assignment.
    # Reproduced when Gemini returned 503 UNAVAILABLE (transient high-demand);
    # see session d4b6b591 / log 08:55:11.
    #
    # 2026-05-06 — Added Anthropic Sonnet 4.5 fallback. Gemini 2.5 Pro had a
    # multi-hour 503 storm during R32 v2 batch (~1 503/min) which produced
    # `agent_refused_to_continue` on 100% of completed sessions. Falling back
    # to Anthropic when Gemini returns 503 OR empty-after-retry rescues the
    # session. Both panels (L and R) benefit equally.
    resp = None
    provider_used = "gemini"
    model_used = ACTOR_MODEL
    fallback_reason = None
    try:
        resp = await loop.run_in_executor(None, _sync, 8192)
        text = (resp.text or "").strip()
        if not text:
            log.warning("Gemini actor returned empty on first attempt — retrying with bigger budget")
            resp = await loop.run_in_executor(None, _sync, 16384)
            text = (resp.text or "").strip()
            if not text:
                fallback_reason = "empty_after_retry"
    except Exception as e:
        emsg = str(e)
        log.warning("Gemini actor call failed: %s", emsg)
        text = ""
        if "503" in emsg or "UNAVAILABLE" in emsg or "RESOURCE_EXHAUSTED" in emsg:
            fallback_reason = "gemini_503"
        else:
            fallback_reason = "gemini_exception"

    # ── Anthropic Sonnet 4.5 fallback ──
    if not text and fallback_reason:
        log.info("actor: Gemini failed (%s) — falling back to Anthropic %s",
                 fallback_reason, ACTOR_FALLBACK_MODEL)
        try:
            anth_msg = await _anthropic.messages.create(
                model=ACTOR_FALLBACK_MODEL,
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = anth_msg.content[0].text if anth_msg.content else ""
            text = (text or "").strip()
            if text:
                provider_used = "anthropic"
                model_used = ACTOR_FALLBACK_MODEL
                # Synthetic resp shim so the rest of the function doesn't
                # NameError on usage_metadata access
                input_tokens_anth = anth_msg.usage.input_tokens if anth_msg.usage else 0
                output_tokens_anth = anth_msg.usage.output_tokens if anth_msg.usage else 0
                resp = type("AnthropicShim", (), {
                    "text": text,
                    "usage_metadata": type("U", (), {
                        "prompt_token_count": input_tokens_anth,
                        "candidates_token_count": output_tokens_anth,
                    })()
                })()
                log.info("actor: Anthropic fallback succeeded (%d tokens)", output_tokens_anth)
        except Exception as e:
            log.warning("actor: Anthropic fallback also failed: %s", e)

    latency_ms = int((time.monotonic() - t0) * 1000)
    usage = getattr(resp, "usage_metadata", None) if resp is not None else None
    if usage is None:
        usage = type("U", (), {})()
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0

    # T-86 trace logging — captures full prompt + response per call
    try:
        from trace_logger import TraceLogger
        trace = TraceLogger.current()
        if trace:
            trace.llm(
                stage="actor.generate",
                provider=provider_used,
                model=model_used,
                system=system,
                user=user,
                response=text,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                extra={"used_directive": directive is not None,
                       "forced_language": forced_language,
                       "has_system_suffix": bool(system_suffix),
                       "r33_escalation_active": escalation_meta is not None,
                       "r33_escalation_trigger": (escalation_meta or {}).get("trigger"),
                       "r33_stall_active": (escalation_meta or {}).get("stall_active"),
                       "r33_customer_target": (escalation_meta or {}).get("customer_target"),
                       "fallback_provider_used": provider_used == "anthropic",
                       "fallback_reason": fallback_reason,},
            )
    except Exception:
        pass  # never let trace logging break a session

    # 2026-05-05 — Hard length enforcement. Gemini 2.5 Pro thinking-mode
    # produces paragraph-style output despite explicit prompt rules. Strategy:
    #   1) If text is over LENGTH_CAP_WORDS, REGENERATE with stricter prompt
    #      that feeds the violation back to the model.
    #   2) Only if regen also overshoots, truncate as last resort (preserves
    #      first sentence which usually has the substance).
    # Regenerate-first preserves the close question / key fact better than
    # truncate-only.
    initial_words = len(re.split(r"\s+", text.strip()))
    regenerated = False
    if text and initial_words > LENGTH_CAP_WORDS:
        log.info("actor: initial reply %d words (>%d cap) — regenerating with "
                 "stricter constraint", initial_words, LENGTH_CAP_WORDS)
        retry_suffix = (
            f"\n\n# REGENERATION REQUIRED — your previous reply was "
            f"{initial_words} words, max allowed is {LENGTH_CAP_WORDS}. "
            f"Rewrite it in ≤25 words, leading with the most important "
            f"fact (price/answer/question). NO preamble like "
            f"'I appreciate' or 'I understand'."
        )
        try:
            retry_prompt = f"{system}{retry_suffix}\n\n---\n\n{user}"
            def _retry(budget):
                return _client.models.generate_content(
                    model=ACTOR_MODEL,
                    contents=retry_prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.5,  # lower temp on retry
                        max_output_tokens=budget,
                    ),
                )
            r2 = await loop.run_in_executor(None, _retry, 8192)
            t2 = (r2.text or "").strip()
            if t2:
                t2_words = len(re.split(r"\s+", t2))
                log.info("actor: regen produced %d words", t2_words)
                text = t2
                regenerated = True
        except Exception as e:
            log.warning("actor: regen failed (%s); falling through to truncation", e)

    truncated_text, was_truncated = _enforce_length(text)
    if was_truncated:
        log.info("actor: post-gen truncation applied (%d → %d words; regen=%s)",
                 len(re.split(r"\s+", text.strip())),
                 len(re.split(r"\s+", truncated_text.strip())),
                 regenerated)
        text = truncated_text

    return text, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "used_directive": directive is not None,
        "regenerated_for_length": regenerated,
        "truncated_for_length": was_truncated,
    }


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    async def smoke():
        opp_meta = {
            "company": "Libra",
            "primary_motivator": "Price/savings",
            "decision_logic": "Analytical",
            "trust_level": "Skeptical",
            "communication_style": "Terse",
            "objection_pattern": "Comparing competitor prices",
        }
        dialog = [
            {"role": "agent", "text": "Hey, this is Nofar from Libra. Your auto insurance renewal: comprehensive 2,818 NIS + mandatory 1,420 NIS.",
             "sequence_number": 0},
            {"role": "customer", "text": "Other insurer offered me 3,500 total, comprehensive only.",
             "sequence_number": 1},
        ]
        rules = "B29: When customer says 'I'll check elsewhere', do soft retention check.\nB34: Don't drop price twice without resistance."

        # Without directive (baseline)
        text, meta = await generate(opp_meta, dialog, rules, directive=None)
        print("=== BASELINE (no directive) ===")
        print(text)
        print(f"  meta: {meta}")
        print()

        # With directive
        directive = {
            "primary_strategy": "objection_handling",
            "tone": "professional",
            "cialdini_levers_to_activate": ["commitment", "reciprocity"],
            "cialdini_levers_to_avoid": ["scarcity"],
            "must_say": [
                "Acknowledge the competitor price they shared",
                "Ask for the breakdown (mandatory vs comprehensive) to structure a fair comparison",
            ],
            "must_not_say": [
                {"text": "wishing you a safe year on the roads", "reason": "premature give-up; cluster-4 anti-pattern"},
                {"text": "any new scarcity claim", "reason": "B34 toughness brake"},
            ],
            "rules_to_enforce": ["B29", "B34", "B43"],
            "objection_handling_template": "Got it — let me check if I can structure something competitive on the comprehensive side. What's the breakdown of their offer?",
            "rationale": "Analytical price-motivated customer with specific competitor number. Won deals in this segment ask for breakdown and offer one structured price-check.",
        }
        text, meta = await generate(opp_meta, dialog, rules, directive=directive)
        print("=== WITH SUPERVISOR DIRECTIVE ===")
        print(text)
        print(f"  meta: {meta}")

    asyncio.run(smoke())
