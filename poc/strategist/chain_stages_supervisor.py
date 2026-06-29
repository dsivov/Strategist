"""POC sandbox: programmatic chain stages from POC supervisor mechanisms.

T-85 Phase A.2 — wire our existing supervisor modules as registered
programmatic stages that the chain runner can splice in.

Each function here is an async callable taking ChainContext, returning a
dict to merge into ctx.previous_responses. Side effects on ctx (anchor
attachment, agent_concessions tracking) happen in-place.

Registered stages:
  prompt_anchor_load               — fetch_insurance_anchors → ctx.anchors
  prompt_signal_analysis_combined  — supervisor_full mode1b w/ signal-retry
  prompt_retreat_passthrough_gate  — suppress directive on retreat signals
  prompt_anti_staircase_gate       — staircase_gate over the candidate text
  prompt_decision_trace_emit       — emit_decision_trace at session end

Per the approved 2026-05-02 design:
- signal_analysis + retry are COMBINED into a single stage (latency concern)
- supervisor AUGMENTS prompt_manager (not replaces) — its output carries
  through previous_responses so prompt_manager can read it
- anti-staircase is sequential (not parallel)
"""
from __future__ import annotations

import logging
import os
from typing import Any

from chain_runner import ChainContext, register_programmatic_stage
from db import open_conn, fetch_insurance_anchors

log = logging.getLogger(__name__)


@register_programmatic_stage("prompt_anchor_load")
async def stage_anchor_load(ctx: ChainContext) -> dict:
    """T-81 (Insurance) + T-86 (Ecommerce) — fetch tenant-appropriate anchor pack.

    Insurance: per-opp economic reference frame (last-year price, market avg,
      claimed_increase_pct, etc.) from insurance_stage_data.
    Ecommerce: tenant-wide product anchor pack (bundle, features, warranty,
      payments, social proof, promotions, internal policy) from CG
      workspace 'Ecommerce'. Cached process-wide for the day.
    """
    company = (ctx.opp_meta.get("company") or "").lower()
    if company == "insurance":
        conn = open_conn()
        try:
            anchors = fetch_insurance_anchors(conn, ctx.opp_id, ctx.opp_meta)
        except Exception as e:
            log.warning("anchor_load (Insurance) failed: %s", e)
            return {"_error": str(e)}
        finally:
            conn.close()
        if not anchors:
            return {"_no_anchors_available": True}
        ctx.anchors = anchors
        log.info("chain.anchor_load: Insurance %d fields, synthetic=%s",
                 len(anchors), anchors.get("synthetic"))
        return {
            "anchors_loaded": True,
            "tenant": "insurance",
            "synthetic": anchors.get("synthetic"),
            "ly_price": anchors.get("last_year_price_usd"),
            "market_avg": anchors.get("market_avg_for_segment_usd"),
        }
    elif company == "ecommerce":
        from ecommerce_anchors import fetch_ecommerce_anchors
        try:
            anchors = await fetch_ecommerce_anchors()
        except Exception as e:
            log.warning("anchor_load (Ecommerce) failed: %s", e)
            return {"_error": str(e)}
        if not anchors or anchors.get("_cg_queries_returned_content", 0) == 0:
            return {"_no_anchors_available": True, "tenant": "ecommerce"}
        ctx.anchors = anchors
        log.info("chain.anchor_load: Ecommerce %d/%d CG fields populated, fetch_ms=%s",
                 anchors.get("_cg_queries_returned_content"),
                 anchors.get("_cg_queries_total"),
                 anchors.get("_fetch_ms"))
        return {
            "anchors_loaded": True,
            "tenant": "ecommerce",
            "cg_queries_returned": anchors.get("_cg_queries_returned_content"),
            "cg_queries_total": anchors.get("_cg_queries_total"),
            "max_discount_pct": anchors.get("max_authorized_discount_pct_internal"),
        }
    else:
        return {"_skipped": f"no anchor schema for tenant={company}"}


def _detect_late_phase_low_score(ctx: ChainContext) -> tuple[bool, str | None]:
    """Returns (is_late_phase_low_score, score_band).

    Definition (from 2026-05-05 expanded analysis on 1286 won-conversation
    moments): late phase = commit_level >= 2 (close_attempt or beyond);
    low score = persuasion_score < 0.4 on the previous customer reply.

    Score band is one of:
      - extremely_low (<0.2)
      - very_low (0.2-0.3)
      - low (0.3-0.4)
    None if not in late-phase low-score state.
    """
    p = getattr(ctx, "prev_persuasion", None)
    c = getattr(ctx, "prev_commitment", None)
    if p is None or c is None:
        return (False, None)
    if c < 2:           # not late phase
        return (False, None)
    if p >= 0.4:        # not low score
        return (False, None)
    if p < 0.2:    band = "extremely_low"
    elif p < 0.3:  band = "very_low"
    else:          band = "low"
    return (True, band)


# ── LLM tier router (Phase 4.1, 2026-05-10) ─────────────────────────────────
#
# Replaces (or A/B-augments) the static heuristic that decides whether a cache
# hit should be served as Mode 1a or whether the turn warrants live Mode 1b
# (Sonnet 4.5) reasoning. Today the heuristic is: if cache score >= threshold
# AND segment matches AND signal matches AND phase matches → use cache. The
# router instead has a fast LLM (Haiku 4.5) read the actual current dialog +
# the cached directive's strategy and judge "does this directive fit THIS turn?"
#
# Cost: ~$0.0001 + ~1-2s per cache hit. (Cache miss path is untouched — no
# router call when there's no hit to evaluate.)

_TIER_ROUTER_MODEL = "claude-haiku-4-5-20251001"
_TIER_ROUTER_SYSTEM = """You are a TIER ROUTER for an AI sales-agent supervisor.

Your job: given a cached strategic directive and the current conversation context,
decide whether to use the cached directive (tier 1a) for this turn or whether
this turn warrants fresh strategic reasoning by the heavyweight supervisor (tier 1b).

Choose 1a when:
- The cached directive's strategy and tone fit the customer's most recent message
- The conversation state matches what the directive was compiled for
- The turn is routine (acknowledgement, info exchange, recap, etc.)

Choose 1b when:
- The customer's most recent message introduces a new objection, signal, or pivot
- The directive's strategy seems mismatched against the conversation's phase
- The customer's emotional state shifted (escalation, capitulation, drop-off)
- There's high stakes (closing turn, price negotiation crisis, competitor mention)

Output EXACTLY this JSON, no prose:
{"tier": "1a"|"1b", "confidence": 0.0-1.0, "reason": "<one short sentence>"}
"""


async def _llm_router_judge(hit: dict, ctx: ChainContext,
                              segment: str, primary_signal: str,
                              current_phase: str) -> dict:
    """Haiku 4.5 confidence gate on a cache hit. Returns:
        {tier: "1a"|"1b", confidence: 0-1, reason: "..."}

    Fail-safe: any exception falls through to Mode 1b in the caller.
    """
    import os, json
    import anthropic

    cached = (hit or {}).get("directive") or {}
    cached_strategy = cached.get("primary_strategy") or "?"
    cached_tone = cached.get("tone") or "?"
    cached_must_say = (cached.get("must_say") or [])[:3]
    cached_must_not_say = (cached.get("must_not_say") or [])[:3]
    cached_score = (hit or {}).get("score", 0.0)
    cached_age_days = (hit or {}).get("age_days", 0.0)

    # Compact dialog tail — last 4 exchanges
    dialog_tail = (ctx.dialog or [])[-4:]
    dialog_lines = []
    for m in dialog_tail:
        role = "Customer" if m.get("role") == "customer" else "Agent"
        text = (m.get("text") or "").replace("\n", " ").strip()[:240]
        if text:
            dialog_lines.append(f"{role}: {text}")
    dialog_str = "\n".join(dialog_lines)

    customer_state = ctx.opp_meta or {}
    profile_str = (
        f"motivator={customer_state.get('primary_motivator','?')} "
        f"decision_logic={customer_state.get('decision_logic','?')} "
        f"trust_level={customer_state.get('trust_level','?')}"
    )

    user_msg = f"""## Customer profile
{profile_str}

## Conversation phase / signal
phase={current_phase} primary_signal={primary_signal}

## Recent dialog (last 4)
{dialog_str if dialog_str else "(no dialog yet)"}

## Cached directive being considered
strategy.primary: {cached_strategy}
strategy.tone:    {cached_tone}
must_say (first 3): {json.dumps(cached_must_say, default=str)[:240]}
must_not_say (first 3): {json.dumps(cached_must_not_say, default=str)[:240]}
cache_score: {cached_score:.2f}
cache_age_days: {cached_age_days:.1f}

JSON only."""

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = await client.messages.create(
        model=_TIER_ROUTER_MODEL,
        max_tokens=200,
        system=_TIER_ROUTER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = (msg.content[0].text if msg.content else "").strip()
    # Strip code fences
    import re
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    try:
        v = json.loads(raw)
    except Exception:
        return {"tier": "1b", "confidence": 0.0,
                "reason": f"router parse error; raw[:80]={raw[:80]!r}"}

    tier = (v.get("tier") or "").lower()
    if tier not in ("1a", "1b"):
        tier = "1b"
    try:
        confidence = float(v.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    # Confidence threshold — when router says 1a but is uncertain, escalate to 1b.
    threshold = float(os.environ.get("POC_TIER_ROUTER_THRESHOLD", "0.7"))
    if tier == "1a" and confidence < threshold:
        return {"tier": "1b", "confidence": confidence,
                "reason": f"router_low_confidence ({confidence:.2f} < {threshold:.2f}): "
                          + (v.get("reason") or "")[:120]}

    return {"tier": tier, "confidence": confidence,
            "reason": v.get("reason") or ""}


async def _try_cache_lookup(ctx: ChainContext) -> dict | None:
    """Cached-Directive Mode 1a (2026-05-04 PROPOSAL). Looks up a past
    directive at this (segment, signal, phase) tuple. Returns a chain-stage
    result dict on cache hit, or None on miss/error.

    Cache hit semantics:
      - Reconstructs a directive from the stored `directive_v1` blob
      - Returns the standard chain-stage result shape (downstream stages
        like move_validity / anti_staircase work unchanged)
      - tier_label = "Mode 1a · cached directive (...)"
      - mode = "1a" so the architecture banner Playbook chip increments

    Safety guards (M1+M2, 2026-05-04):
      - M1: skip lookup when ctx.consecutive_mode1a >= MAX_CONSECUTIVE_1A
        (hard cap = 2; legacy dfb34792 evidence)
      - M2: random POC_CACHE_FALLBACK_RATE chance to skip even on score-pass,
        keeping adaptation signal alive
    """
    import os, random
    from supervisor_full import (
        lookup_cached_directive, build_segment_key, quick_classify_signal,
    )
    from conversation_phase import classify_dialog
    import httpx

    # M1 — max consecutive Mode 1a hits per session
    MAX_CONSECUTIVE_1A = int(os.environ.get("POC_CACHE_MAX_CONSECUTIVE", "2"))
    if getattr(ctx, "consecutive_mode1a", 0) >= MAX_CONSECUTIVE_1A:
        log.info("chain.cache_skip: M1 hit (consecutive_mode1a=%d >= %d)",
                 getattr(ctx, "consecutive_mode1a", 0), MAX_CONSECUTIVE_1A)
        # Reset counter so we go through Mode 1b for adaptation, then can
        # hit cache again next turn.
        ctx.consecutive_mode1a = 0
        return None

    # Compute tuple key — phase + signal classified cheaply (regex, no LLM)
    phase_history = classify_dialog(ctx.dialog or [])
    current_phase = phase_history[-1] if phase_history else "greet"
    primary_signal = quick_classify_signal(ctx.dialog or [])
    segment = build_segment_key(ctx.opp_meta)
    workspace = ctx.opp_meta.get("company") or ""

    # Guard against degenerate keys
    if "?" in segment or not workspace:
        return None

    # Late-phase recovery: when prev customer reply was low-engagement in a
    # late phase, try the recovery-pattern lookup FIRST (filtered to
    # historical "what real human agents did when stuck" edges). On miss,
    # fall through to the normal lookup.
    is_recovery, score_band = _detect_late_phase_low_score(ctx)

    try:
        async with httpx.AsyncClient(timeout=15) as c:
            hit = None
            if is_recovery:
                hit = await lookup_cached_directive(
                    c, workspace, segment,
                    primary_signal=primary_signal,
                    phase=current_phase,
                    late_phase_recovery=True,
                    score_band=score_band,
                )
                if hit is not None:
                    log.info("chain.late_phase_recovery_hit: band=%s segment=%s phase=%s strat=%s",
                             score_band, segment, current_phase,
                             hit["directive"].get("primary_strategy"))
                    # Tag the result so downstream telemetry can distinguish
                    hit["_recovery_mode"] = True
                    hit["_score_band"] = score_band
            if hit is None:
                hit = await lookup_cached_directive(
                    c, workspace, segment,
                    primary_signal=primary_signal,
                    phase=current_phase,
                )
    except Exception as e:
        log.warning("cache lookup raised: %s", e)
        return None

    # Telemetry — record cache outcome to trace (rich detail for /logs UI)
    try:
        from trace_logger import TraceLogger
        _tr = TraceLogger.current()
        if _tr is not None:
            chosen_directive = (hit or {}).get("directive") or {}
            chosen_move = chosen_directive.get("concrete_move") or {}
            _tr.note(
                "cache_lookup",
                hit=(hit is not None),
                recovery_mode=is_recovery,
                score_band=score_band,
                prev_persuasion=getattr(ctx, "prev_persuasion", None),
                prev_commitment=getattr(ctx, "prev_commitment", None),
                segment=segment,
                primary_signal=primary_signal,
                phase=current_phase,
                chosen_strategy=chosen_directive.get("primary_strategy"),
                chosen_tone=chosen_directive.get("tone"),
                chosen_move=chosen_move.get("name") if isinstance(chosen_move, dict) else None,
                chosen_audit=(chosen_directive.get("audit") or {}).get("rationale_summary", "")[:200],
                score=(hit["score"] if hit else None),
                age_days=(hit["age_days"] if hit else None),
                n_candidates=(hit["n_candidates"] if hit else 0),
            )
    except Exception:
        pass

    if hit is None:
        log.info("chain.cache_miss: tier=1b (segment=%s signal=%s phase=%s)",
                 segment, primary_signal, current_phase)
        # Reset counter on miss
        ctx.consecutive_mode1a = 0
        return None

    # M2 — random fallback to Mode 1b even on cache hit. Keeps the adaptation
    # signal alive AND surfaces stale-cache problems quickly. Default 10%.
    fallback_rate = float(os.environ.get("POC_CACHE_FALLBACK_RATE", "0.1"))
    if fallback_rate > 0 and random.random() < fallback_rate:
        log.info("chain.cache_skip: M2 random fallback (rate=%.2f) — using Mode 1b",
                 fallback_rate)
        ctx.consecutive_mode1a = 0
        return None

    # M3 (Phase 4.1, 2026-05-10) — LLM tier router. When POC_TIER_ROUTER=llm,
    # route cache hits through a Haiku 4.5 confidence gate that judges whether
    # the cached directive fits the current turn's context. Cache miss path
    # (already returned None above) is unchanged. A/B-default: off.
    if os.environ.get("POC_TIER_ROUTER", "off").lower() == "llm":
        try:
            router_decision = await _llm_router_judge(
                hit, ctx, segment, primary_signal, current_phase)
        except Exception as e:
            log.warning("chain.tier_router: error %s — defaulting to Mode 1b for safety", e)
            ctx.consecutive_mode1a = 0
            return None
        if router_decision.get("tier") != "1a":
            log.info("chain.tier_router: overrode cache hit (tier=%s confidence=%.2f reason=%s)",
                     router_decision.get("tier"), router_decision.get("confidence", 0.0),
                     (router_decision.get("reason") or "")[:120])
            ctx.consecutive_mode1a = 0
            return None
        log.info("chain.tier_router: confirmed Mode 1a (confidence=%.2f reason=%s)",
                 router_decision.get("confidence", 0.0),
                 (router_decision.get("reason") or "")[:120])

    cached = hit["directive"]
    # M1 — increment the counter so consecutive cache hits eventually hit
    # the hard cap and force Mode 1b.
    ctx.consecutive_mode1a = getattr(ctx, "consecutive_mode1a", 0) + 1
    log.info("chain.cache_hit: tier=1a score=%.2f age=%.1fd n_cands=%d "
             "consec=%d fallback_level=%d (segment=%s signal=%s phase=%s strat=%s move=%s)",
             hit["score"], hit["age_days"], hit["n_candidates"],
             ctx.consecutive_mode1a, hit.get("fallback_level", 0),
             segment, primary_signal, current_phase,
             cached.get("primary_strategy"),
             (cached.get("concrete_move") or {}).get("name") if cached.get("concrete_move") else None)

    # Reconstruct a directive in the shape downstream stages expect.
    # 2026-05-05 — surface Tier 1+2 enrichment fields under `_cache_*`
    # keys so the UI can render them as chat-level badges below R messages.
    must_not_say_dicts = [{"text": t} for t in (cached.get("must_not_say") or [])]
    directive = {
        "strategy": {
            "primary": cached.get("primary_strategy"),
            "tone": cached.get("tone"),
            "concrete_move": cached.get("concrete_move"),
        },
        "knowledge": {
            "must_not_say": must_not_say_dicts,
            "facts_to_anchor": [],
        },
        "must_not_say": must_not_say_dicts,
        # Tier 1+2 enrichment fields — surfaced for the UI badges
        "_cache_tactical": cached.get("tactical") or {},
        "_cache_cialdini": cached.get("cialdini_activated") or [],
        "_cache_delta_after_move": cached.get("delta_after_move"),
        "_cache_secondary_strategy": cached.get("secondary_strategy"),
        "_cache_reason": cached.get("reason"),
        "audit": {
            "rationale_summary": (
                f"Cached directive (Mode 1a). score={hit['score']:.2f}, "
                f"age={hit['age_days']:.0f}d, n_candidates={hit['n_candidates']}. "
                f"Reused from prior won/successful turn at the same "
                f"(segment, signal, phase) tuple. No fresh LLM call this turn."
            ),
        },
        "confidence": {
            "overall": min(0.95, hit["score"] + 0.1),  # cache hits are high-confidence
            "band": "high" if hit["score"] >= 0.7 else "med",
        },
        "signal_analysis": {
            "primary_signal": primary_signal,
            "observed_signals": (
                [{"signal": primary_signal, "confidence": 0.9}]
                if primary_signal else []
            ),
            "strategy_implication": f"Cache hit reusing prior decision for {primary_signal}",
            "counterfactual": "Cache hit; no fresh reasoning fired this turn.",
            "plan_alignment": "plan_silent",
        },
        "customer_state": {},
        "rules_to_enforce": [],
        "plan": {},
    }

    # Build the chain-stage return shape (mirrors stage_signal_analysis_combined)
    return {
        "directive": directive,
        "signal_analysis": directive["signal_analysis"],
        "primary_signal": primary_signal,
        "primary_strategy": cached.get("primary_strategy"),
        "tone": cached.get("tone"),
        "adherence_retry_fired": False,
        "signal_adherence_retry_fired": False,
        "consistency_retry": None,
        "move_validity_retry": None,
        "cg_evidence": {"products": False, "rules": False, "patterns": False, "decisions": True},
        "cg_n_entities": 0,
        "cg_n_relations": 0,
        "cg_n_chunks": 0,
        "directive_contradictions_removed": [],
        "tier_label": (
            f"Mode 1a · cached directive (score {hit['score']:.2f}, "
            f"age {hit['age_days']:.0f}d)"
        ),
        "conversation_phase": {
            "current": current_phase,
            "turns_in_phase": _count_phase_dwell(phase_history, current_phase),
            "cluster_plan_phase": None,
        },
        "_cache_hit": True,
        "_cache_score": hit["score"],
        "_cache_age_days": hit["age_days"],
        "_cache_n_candidates": hit["n_candidates"],
    }


def _count_phase_dwell(phase_history: list, current_phase: str) -> int:
    """Count consecutive trailing turns matching `current_phase`."""
    n = 0
    for p in reversed(phase_history or []):
        if p == current_phase:
            n += 1
        else:
            break
    return n


@register_programmatic_stage("prompt_signal_analysis_combined")
async def stage_signal_analysis_combined(ctx: ChainContext) -> dict:
    """T-79 + T-80 combined per approved design (latency > observability).
    Runs full mode1b_directive (which already includes signal_analysis +
    signal-driven retry inline). Returns the structured directive that
    downstream prompt_manager AUGMENTS its decision with.

    Tier-dispatch (2026-05-04): when POC_CHAIN_TIER_ROUTING=cache, attempts
    a Mode 1a cache lookup first. Cache hits skip the LLM call entirely
    (~50ms vs ~10-30s). Cache misses fall through to Mode 1b."""
    import os
    routing_mode = os.environ.get("POC_CHAIN_TIER_ROUTING", "off")

    if routing_mode == "cache":
        cache_result = await _try_cache_lookup(ctx)
        if cache_result is not None:
            return cache_result

    from supervisor_full import mode1b_directive
    business_rules = ctx.business_rules or ""
    # M1 — Mode 1b path resets the consecutive counter (we're NOT serving
    # from cache this turn, so adaptation is happening).
    ctx.consecutive_mode1a = 0

    # 2026-05-10 — Profile-validator regenerate-loop: when chain_runner
    # has injected a corrective marker after a prior directive failed
    # the profile validator, pass it through to mode1b_directive so
    # Sonnet sees the violations and produces a clean re-roll.
    corrective_context = ctx.previous_responses.pop("_directive_profile_correction", None)

    try:
        # Q14 (2026-05-04) — pass cluster_plan + session-memory through so
        # the chain runner is measured at parity with the legacy replayer
        # path. Falls back to None when ChainContext doesn't carry them
        # (older callers / smoke tests).
        result = await mode1b_directive(
            ctx.opp_meta,
            ctx.dialog,
            business_rules,
            strategies_used=getattr(ctx, "strategies_used", None),
            plan_state=getattr(ctx, "plan_state", None),
            pre_rendered_plan_section=getattr(ctx, "pre_rendered_plan_section", None),
            preferred_actions=getattr(ctx, "preferred_actions", None),
            current_phase_label=None,  # phase_label is the legacy hint;
                                       # phase_block is now built from dialog
                                       # inside mode1b_directive itself.
            moves_used=getattr(ctx, "moves_used", None),
            corrective_context=corrective_context,
        )
    except Exception as e:
        # Add full stack trace so we can locate the offending code path
        import traceback
        log.warning("signal_analysis_combined failed: %s\n%s",
                    e, traceback.format_exc())
        return {"_error": str(e)}

    directive = result.get("directive") or {}
    sa = directive.get("signal_analysis") or {}

    # T-87 fix-2 — strip contradicting must_not_say rules. Day-4 case 3 +
    # session 30986515 (2026-05-03) revealed the LLM emitting
    # `primary_strategy=reciprocity` AND `must_not_say='no unprompted price
    # reduction'` in the same directive — internally inconsistent. When the
    # supervisor's chosen strategy is reciprocity / objection_handling /
    # price-bargaining, "no unprompted reduction" rules become contradictory
    # because the strategy itself authorizes a price move. Strip them.
    contradictions_removed = _normalize_directive_consistency(directive)
    if contradictions_removed:
        log.info("chain.signal_analysis: stripped %d contradicting must_not_say rule(s) "
                 "(strategy=%s)",
                 len(contradictions_removed),
                 (directive.get("strategy") or {}).get("primary"))

    return {
        "directive": directive,  # for downstream stages to consume
        "signal_analysis": sa,
        "primary_signal": sa.get("primary_signal"),
        "primary_strategy": (directive.get("strategy") or {}).get("primary"),
        "tone": (directive.get("strategy") or {}).get("tone"),
        "adherence_retry_fired": bool((result.get("adherence_retry") or {}).get("retried")),
        "signal_adherence_retry_fired": bool(
            (result.get("signal_adherence_retry") or {}).get("retried")),
        # R4 (Q12) — surface consistency retry for UI counterfactual chip.
        "consistency_retry": result.get("consistency_retry"),
        # R6 — surface move-validity retry for UI counterfactual chip.
        "move_validity_retry": result.get("move_validity_retry"),
        "cg_evidence": result.get("cg_evidence") or {},
        # CG-data audit (2026-05-04) — chain path was dropping these counts.
        # UI shows them in the architecture banner via directive.cg.{entities,
        # relations, chunks}. Without these, the chain_runner path always
        # showed 0/0/0 even though Mode 1b retrieved real data.
        "cg_n_entities": result.get("cg_n_entities", 0),
        "cg_n_relations": result.get("cg_n_relations", 0),
        "cg_n_chunks": result.get("cg_n_chunks", 0),
        "directive_contradictions_removed": contradictions_removed,
        "tier_label": "Mode 1b · live + Context Graph",
        # Multi-turn arc awareness — surface phase up so replayer can put it
        # in directive_meta and the UI can render the chip.
        "conversation_phase": result.get("conversation_phase"),
    }


# Patterns that indicate a "no price reduction" rule. When the directive's
# primary_strategy authorizes a price move, these rules are self-contradictory
# and we strip them.
import re as _re_norm
_NO_REDUCTION_PATTERNS = [
    _re_norm.compile(r"\bno\s+(?:further|additional|unprompted)?\s*price\s+(?:reduction|drop)", _re_norm.IGNORECASE),
    _re_norm.compile(r"\bdon'?t\s+(?:reduce|drop|lower)\s+(?:the\s+)?price", _re_norm.IGNORECASE),
    _re_norm.compile(r"\bno\s+further\s+(?:concession|discount)", _re_norm.IGNORECASE),
    _re_norm.compile(r"\bhold\s+(?:the\s+)?(?:line|price)\s+at\b", _re_norm.IGNORECASE),
    _re_norm.compile(r"\bcannot\s+(?:reduce|lower|drop)\s+(?:the\s+)?price\b", _re_norm.IGNORECASE),
]
_PRICE_AUTHORIZING_STRATEGIES = {"reciprocity", "objection_handling", "logistics"}


def _normalize_directive_consistency(directive: dict) -> list[str]:
    """Strip must_not_say rules that contradict the chosen strategy.

    Mutates `directive["knowledge"]["must_not_say"]` in place. Returns a
    list of the stripped rule texts (for logging / observability).

    Rationale: the supervisor LLM sometimes emits both 'do this' (via
    primary_strategy) and 'don't do this' (via must_not_say) for the same
    move. When that happens, prompt_build_answer reads the contradiction and
    defaults to the conservative 'don't' because the LLM's training-distribution
    bias favors safety. Strip the contradiction so the strategy directive
    actually binds.
    """
    primary = ((directive.get("strategy") or {}).get("primary") or "").lower()
    if primary not in _PRICE_AUTHORIZING_STRATEGIES:
        return []
    knowledge = directive.get("knowledge") or {}
    must_not_say = knowledge.get("must_not_say") or []
    if not must_not_say:
        return []
    removed: list[str] = []
    kept: list = []
    for entry in must_not_say:
        text = entry.get("text") if isinstance(entry, dict) else str(entry)
        if not text:
            kept.append(entry)
            continue
        if any(p.search(text) for p in _NO_REDUCTION_PATTERNS):
            removed.append(text[:120])
            continue
        kept.append(entry)
    if removed:
        knowledge["must_not_say"] = kept
        directive["knowledge"] = knowledge
    return removed


# Q13 — followup-commit discriminator. A pace_request in late-stage
# negotiation (commit_pending / commit_level≥4) is usually a payment-mechanics
# pause ("let me grab my wallet"), NOT a retreat — suppressing the directive
# there sends the customer back to objection-handling instead of finishing
# the close. These lexical cues, combined with stage signals, flip the verdict.
_FOLLOWUP_COMMIT_LEX = _re_norm.compile(
    r"\b(card|wallet|phone|details?|number|info(rmation)?|payment|cc|credit|debit|account|cvv)\b",
    _re_norm.IGNORECASE,
)
_FOLLOWUP_COMMIT_PHRASE = _re_norm.compile(
    r"\b(grab|find|fetch|pull up|look up|let me (find|grab|get|check|look))\b",
    _re_norm.IGNORECASE,
)


def _is_followup_commit_pace(sa_stage: dict, dialog: list[dict]) -> str | None:
    """Returns reason string when a pace_request should be treated as
    commit-followup (not retreat); None otherwise."""
    directive = sa_stage.get("directive") or {}
    cs = directive.get("customer_state") or {}
    commit_level = cs.get("current_commitment_level") or 0

    cp = sa_stage.get("conversation_phase") or {}
    phase = cp.get("current")

    if phase == "commit_pending":
        return f"phase=commit_pending"
    if commit_level >= 4:
        return f"commit_level={commit_level}"

    last_customer = ""
    for msg in reversed(dialog):
        if msg.get("role") == "customer":
            last_customer = msg.get("text", "") or ""
            break
    if last_customer:
        has_lex = bool(_FOLLOWUP_COMMIT_LEX.search(last_customer))
        has_phrase = bool(_FOLLOWUP_COMMIT_PHRASE.search(last_customer))
        if has_lex and has_phrase:
            return "lexical(grab/find + card/wallet/details)"
    return None


@register_programmatic_stage("prompt_retreat_passthrough_gate")
async def stage_retreat_passthrough(ctx: ChainContext) -> dict:
    """T-80 productionization: suppress the directive when customer signals
    pace_request or disengagement, so downstream prompt_build_answer falls
    back to natural-Agent soft-retention behavior.

    Q13 (2026-05-04): pace_request in commit_pending / high-buy-intent is
    NOT a retreat — it's a payment-mechanics pause. Discriminator flips the
    verdict in those cases so the supervisor's commit_pending directive
    survives to build_answer."""
    sa_stage = ctx.previous_responses.get("prompt_signal_analysis_combined") or {}
    primary_signal = sa_stage.get("primary_signal")
    RETREAT_SIGNALS = {"pace_request", "disengagement"}
    if primary_signal not in RETREAT_SIGNALS:
        return {"retreat_active": False, "primary_signal": primary_signal}

    # Q13 discriminator
    followup_reason = _is_followup_commit_pace(sa_stage, ctx.dialog)
    if followup_reason:
        log.info("chain.retreat_passthrough: pace-but-commit discriminator "
                 "fired (%s) — NOT suppressing directive (signal=%s)",
                 followup_reason, primary_signal)
        return {
            "retreat_active": False,
            "primary_signal": primary_signal,
            "discriminator_fired": True,
            "discriminator_reason": followup_reason,
        }

    # Suppress the directive in previous_responses so downstream stages
    # don't see it (mirror current replayer.py retreat passthrough behavior)
    if "directive" in sa_stage:
        sa_stage["_directive_suppressed_by_retreat"] = sa_stage.pop("directive")
    log.info("chain.retreat_passthrough: suppressing directive (signal=%s)",
             primary_signal)
    return {
        "retreat_active": True,
        "primary_signal": primary_signal,
        "suppressed_strategy": sa_stage.get("primary_strategy"),
    }


@register_programmatic_stage("prompt_anti_staircase_gate")
async def stage_anti_staircase(ctx: ChainContext) -> dict:
    """T-83 productionization: detect monotonic concession pattern in the
    candidate reply text. Runs after prompt_build_answer.
    Returns {staircase_detected, retry_recommended} — actual regenerate-
    loop semantics live in the chain runner (TODO: implement when we wire
    the chain runner into replayer)."""
    from staircase_gate import check_staircase
    candidate = ctx.previous_responses.get("prompt_build_answer")
    if not candidate or not isinstance(candidate, str):
        return {"_no_candidate_to_check": True}
    tenant = (ctx.opp_meta.get("company") or "").strip()
    is_sc, meta = check_staircase(
        panel_concessions=ctx.agent_concessions,
        new_text=candidate,
        tenant=tenant,
        dialog=ctx.dialog,
    )
    if is_sc:
        log.info("chain.anti_staircase: staircase detected — %s",
                 meta.get("reason"))
        # 2026-05-10 BUGFIX — was dropping is_critical / observed_drop_pct
        # from the gate's metadata. Without these, chain_runner's exhaustion
        # handler couldn't distinguish critical violations (≥25% drop, needs
        # safe-fallback substitution) from non-critical ones (15-25% drop,
        # original draft acceptable). Result: critical violations were being
        # passed through as if non-critical. Fix: propagate the full gate
        # metadata so the exhaustion handler can see is_critical=True.
        return {
            "staircase_detected": True,
            "reason": meta.get("reason"),
            "new_amount": meta.get("new_amount"),
            "prior_best": meta.get("prior_best"),
            "is_critical": meta.get("is_critical"),
            "observed_drop_pct": meta.get("observed_drop_pct"),
            "retry_recommended": True,
        }
    # No staircase → record new concessions
    for c in (meta.get("new_concessions") or []):
        ctx.agent_concessions.append({**c, "turn": len(ctx.dialog) + 1})
    return {
        "staircase_detected": False,
        "new_concessions": meta.get("new_concessions"),
    }


@register_programmatic_stage("prompt_move_validity_gate")
async def stage_move_validity(ctx: ChainContext) -> dict:
    """Phase 2 — move-validity gate (architect DP3, Class B).

    Validates the supervisor's emitted `directive.strategy.concrete_move`
    against the loaded catalog for the tenant:
      1. concrete_move is null OR a valid dict (not malformed type)
      2. has `name` (or alias `source_ref` per fix A — temporary; Phase 3 will tighten)
      3. name resolves to a move in this tenant's catalog (base + tenant.yaml)
      4. all required parameters present + non-empty
      5. (Phase 2.5: cg_entity_refs validated against live CG — deferred)

    Phase 2 scope: ADVISORY ONLY. Issues logged + surfaced in the result
    dict for observability. Regenerate-loop semantics (re-invoke Mode 1b
    with corrective suffix on invalid) is Phase 2.5 work — same pattern
    as T-79's _retry_for_signal_adherence but targeting the directive's
    Tier 2 selection.

    Splice position: in preprocessing alongside other supervisor stages,
    AFTER signal_analysis_combined produces the directive, BEFORE
    build_answer would consume it. This is architect's approved DP3
    position (before prompt_build_answer)."""
    from concrete_moves_loader import get_move

    sa = ctx.previous_responses.get("prompt_signal_analysis_combined") or {}
    directive = sa.get("directive") if isinstance(sa, dict) else None
    if not isinstance(directive, dict):
        return {"verdict": "no_directive_to_validate", "_skipped": True}

    strategy = directive.get("strategy") or {}
    cm = strategy.get("concrete_move")

    # Tier 2 is OPTIONAL per architect DP4 — null is valid
    if cm is None or cm == "null":
        return {"verdict": "no_move_picked", "valid": True,
                "note": "Tier 2 is optional; supervisor declined to pick a concrete move this turn"}

    if not isinstance(cm, dict):
        log.info("move_validity: concrete_move is %s, expected dict",
                 type(cm).__name__)
        return {"verdict": "malformed_type", "valid": False,
                "issue": f"concrete_move is {type(cm).__name__}, expected dict or null"}

    # Schema check: must have a name (or known alias per fix A)
    move_name = (
        cm.get("name") or cm.get("source_ref") or cm.get("move") or cm.get("id")
    )
    used_alias = bool(move_name and not cm.get("name"))
    if not move_name:
        log.info("move_validity: concrete_move missing name field; keys=%s",
                 list(cm.keys()))
        return {"verdict": "missing_name", "valid": False,
                "issue": f"concrete_move has no name; got keys: {list(cm.keys())}"}

    # Catalog lookup
    tenant = (ctx.opp_meta.get("company") or "").lower() or None
    move = get_move(move_name, tenant=tenant)
    if move is None:
        log.info("move_validity: '%s' not in %s catalog", move_name, tenant or "_base")
        return {"verdict": "unknown_move", "valid": False,
                "issue": f"move name '{move_name}' not in {tenant or '_base'} catalog",
                "move_name": move_name,
                "used_alias": used_alias}

    # Parameter completeness
    params = cm.get("parameters") or {}
    missing = []
    empty = []
    for p_name in move.parameters.keys():
        if p_name not in params:
            missing.append(p_name)
        elif params[p_name] in (None, "", []):
            empty.append(p_name)

    if missing or empty:
        issues = []
        if missing:
            issues.append(f"missing: {missing}")
        if empty:
            issues.append(f"empty: {empty}")
        log.info("move_validity: '%s' parameter issues — %s",
                 move_name, "; ".join(issues))
        return {"verdict": "param_incomplete", "valid": False,
                "issue": "; ".join(issues),
                "move_name": move_name,
                "missing_params": missing,
                "empty_params": empty,
                "used_alias": used_alias}

    log.info("move_validity: '%s' valid (alias=%s)", move_name, used_alias)

    # Move-usage telemetry — feeds 2026-05-10 cron `218b4144` for
    # telemetry-driven catalog pruning (Tier 2 catalog cap is currently 17 vs
    # soft cap of 15; usage data tells us which moves to drop).
    try:
        from trace_logger import TraceLogger
        _tr = TraceLogger.current()
        if _tr is not None:
            _tr.note(
                "move_used",
                tenant=tenant or "_base",
                move_name=move_name,
                primitive=move.primitive,
                used_alias=used_alias,
                gate_class=move.gate_class,
            )
    except Exception:
        pass

    return {"verdict": "valid", "valid": True,
            "move_name": move_name,
            "n_params": len(params),
            "used_alias": used_alias,
            "cg_entity_refs": move.cg_entity_refs}


@register_programmatic_stage("prompt_anti_capitulation_gate")
async def stage_anti_capitulation(ctx: ChainContext) -> dict:
    """T-87 — detect capitulation language in candidate reply (apologizing
    for our offer, endorsing competitor's deal as 'better', directing the
    customer to use the competitor's code, or admitting we 'can't compete').

    Runs after prompt_build_answer (and after prompt_anti_staircase_gate,
    so staircase fires first if both apply). Triggers regenerate-loop with
    a corrective system_suffix that pulls concrete value from the loaded
    anchor pack (T-86 Ecommerce / T-81 Insurance)."""
    from capitulation_gate import check_capitulation
    candidate = ctx.previous_responses.get("prompt_build_answer")
    if not candidate or not isinstance(candidate, str):
        return {"_no_candidate_to_check": True}
    tenant = (ctx.opp_meta.get("company") or "").strip()
    is_cap, meta = check_capitulation(
        candidate_text=candidate,
        dialog=ctx.dialog,
        tenant=tenant,
    )
    if is_cap:
        log.info("chain.anti_capitulation: capitulation detected — %s",
                 meta.get("reason"))
        return {
            "capitulation_detected": True,
            "reason": meta.get("reason"),
            "matched_patterns": meta.get("matched_patterns"),
            "matched_phrases": meta.get("matched_phrases"),
            "competitor_context": meta.get("competitor_context"),
            "retry_recommended": True,
        }
    return {
        "capitulation_detected": False,
        "matched_patterns": meta.get("matched_patterns") or [],
        "verdict": meta.get("verdict") or "clean",
    }


@register_programmatic_stage("prompt_premature_close_gate")
async def stage_premature_close_gate(ctx: ChainContext) -> dict:
    """2026-05-14 — Detect agent giving-up phrasing in the candidate response
    and trigger regenerate.

    Fires when the agent's draft matches the `agent_premature_close` intent
    AND the conversation is still in-play (< PREMATURE_TURN_THRESHOLD customer
    turns) AND the customer has not explicitly declined. On fire, returns
    `retry_recommended=True` and chain_runner re-runs prompt_build_answer with
    a corrective system_suffix that pushes the actor to propose a concrete
    alternative instead of a soft handoff.

    Triggered by session bca61ad8 / opp 28d73ce4 (Ecommerce, budget-objection
    customer that supervised side abandoned at turn 8).
    """
    from premature_close_gate import check_premature_close
    candidate = ctx.previous_responses.get("prompt_build_answer")
    if not candidate or not isinstance(candidate, str):
        return {"_no_candidate_to_check": True}
    tenant = (ctx.opp_meta.get("company") or "").strip()
    is_premature, meta = check_premature_close(
        candidate_text=candidate,
        dialog=ctx.dialog,
        tenant=tenant,
    )
    if is_premature:
        log.info("chain.premature_close: FIRED — %s (anchor=%r sim=%.3f, "
                 "n_customer_turns=%d)",
                 meta.get("reason"), meta.get("matched_anchor"),
                 meta.get("intent_sim", 0.0), meta.get("n_customer_turns", -1))
        return {
            "premature_close_detected": True,
            "reason": meta.get("reason"),
            "matched_anchor": meta.get("matched_anchor"),
            "intent_sim": meta.get("intent_sim"),
            "n_customer_turns": meta.get("n_customer_turns"),
            "retry_recommended": True,
        }
    return {
        "premature_close_detected": False,
        "reason": meta.get("reason"),
    }


@register_programmatic_stage("prompt_invariants_gate")
async def stage_invariants(ctx: ChainContext) -> dict:
    """R5 — additional hard-invariant gates (max-discount, no-disparagement,
    no-fabricated-statistics, regulatory-language). Runs after prompt_build_answer,
    after anti_staircase + anti_capitulation. Triggers regenerate-loop on
    violation with a corrective system_suffix that names the specific invariant."""
    from invariant_gates import check_invariants
    candidate = ctx.previous_responses.get("prompt_build_answer")
    if not candidate or not isinstance(candidate, str):
        return {"_no_candidate_to_check": True}
    violated, meta = check_invariants(
        candidate_text=candidate,
        dialog=ctx.dialog,
        opp_meta=ctx.opp_meta,
    )
    violations = meta.get("violations") or []
    if violated:
        types = [v.get("type") for v in violations]
        log.info("chain.invariants: %d violation(s) — %s",
                 len(violations), types)
        return {
            "invariants_violated": True,
            "violations": violations,
            "violation_types": types,
            "retry_recommended": True,
        }
    return {
        "invariants_violated": False,
        "verdict": "clean",
    }


@register_programmatic_stage("prompt_live_profile_classifier")
async def stage_live_profile_classifier(ctx: ChainContext) -> dict:
    """2026-05-10 — Live first-turn profile classifier.

    Replaces the missing piece from the original architecture: instead of
    relying solely on `research_profile_flash` (precomputed by labeler at
    2026-04-09 vintage), classify the customer's profile from their first
    1-3 live messages and update opp_meta in-place.

    Activates only when:
      - POC_LIVE_PROFILE_CLASSIFIER=1 (env-gated; default OFF)
      - Customer has emitted >= POC_LIVE_PROFILE_MIN_TURNS messages
      - Hasn't already classified this session

    Updates only fields that are missing OR where classifier confidence
    is high AND disagrees with precomputed. Defensive: existing
    high-confidence precomputed profiles are preserved.

    Cost: ~$0.0001 per session × ~1-2s latency. Runs ONCE per session.
    """
    try:
        from profile_classifier import maybe_classify_and_update_profile
        result = await maybe_classify_and_update_profile(ctx)
        return result
    except Exception as e:
        log.warning("live_profile_classifier stage failed (non-fatal): %s", e)
        return {"fired": False, "reason": f"error: {str(e)[:120]}"}


@register_programmatic_stage("prompt_directive_profile_gate")
async def stage_directive_profile_gate(ctx: ChainContext) -> dict:
    """2026-05-10 — Profile-aware directive validator (mechanical gate).

    Runs after `prompt_signal_analysis_combined` produces the directive.
    Validates the directive against profile-derived rules in
    `data/anchor_strategy/<Tenant>.yaml`'s `validators` block:

      - Skeptical customer + directive.strategy.primary == "direct_ask"
        → violation (Skeptical customers react badly to direct asks)
      - Skeptical customer + must_say contains "loyalty bonus" / "special
        discount" / "as a returning customer" → violation (manipulation tells)
      - Analytical customer + must_say contains "amazing value" / "perfect
        for you" → violation (Analytical wants numbers, not rhetoric)

    Phase 1 (this version): LOGGING ONLY. Records violations in metadata
    so the trace logger captures them. Does NOT yet trigger regenerate-loop —
    if the upstream profile_anchor_guidance (Step 1) works, this gate
    should rarely fire. If it fires frequently, that's a signal to wire
    the regenerate-loop in Phase 2.

    Failure mode: never blocks the chain. On any error, logs a warning
    and returns {valid: True, _error: ...} so downstream stages proceed.
    """
    sa_combined = ctx.previous_responses.get("prompt_signal_analysis_combined") or {}
    directive = sa_combined.get("directive") if isinstance(sa_combined, dict) else None
    if not isinstance(directive, dict) or not directive:
        return {"valid": True, "skipped": "no_directive"}

    try:
        from directive_profile_gate import validate_directive
        # Inject dialog into opp_meta so the validator can do language-mismatch detection
        opp_meta_for_validator = dict(ctx.opp_meta or {})
        opp_meta_for_validator["_dialog_for_validator"] = ctx.dialog or []
        is_valid, meta = validate_directive(directive, opp_meta_for_validator)
    except Exception as e:
        log.warning("directive_profile_gate: validator failed (non-fatal): %s", e)
        return {"valid": True, "_error": str(e)[:200]}

    if not is_valid:
        violations = meta.get("violations") or []
        log.info("chain.directive_profile_gate: %d violation(s) — %s",
                 len(violations),
                 [v.get("type") for v in violations])
        # Phase 2 (2026-05-10) — escalate to retry. The chain_runner reads
        # `retry_recommended` and re-runs `prompt_signal_analysis_combined`
        # with the corrective_suffix as additional context to mode1b_directive.
        return {
            "valid": False,
            "violations": violations,
            "rules_applied": meta.get("rules_applied", 0),
            "retry_recommended": True,
            "corrective_suffix": meta.get("corrective_suffix", ""),
        }

    return {"valid": True, "rules_applied": meta.get("rules_applied", 0)}


@register_programmatic_stage("prompt_decision_trace_emit")
async def stage_decision_trace_emit(ctx: ChainContext) -> dict:
    """S4 closed-loop decision-trace write. Per-turn postprocessing slot.

    Calls supervisor_full.emit_decision_trace which POSTs to
    /graph/decision/emit, building a decision-edge from the supervisor's
    directive + the current turn's commit/persuasion peaks. Each successful
    emit feeds the precedent corpus that future sessions read at start
    (S7 — Decision Precedent Lookup).

    Failure mode: if emit fails (CG unavailable, bad payload), we don't
    block the chain — the session continues, just no precedent is recorded
    for this turn. Logged for observability."""
    sa_combined = ctx.previous_responses.get("prompt_signal_analysis_combined") or {}
    directive = sa_combined.get("directive") if isinstance(sa_combined, dict) else None
    if not isinstance(directive, dict) or not directive:
        return {"emitted": False, "_skipped": "no directive available"}

    # Compute lightweight outcome for this turn (used as quantitative_data
    # in the emitted edge). Real win/loss outcome is unavailable per-turn —
    # use commit + strategy as the per-turn signal.
    customer_state = directive.get("customer_state") or {}
    commit_now = customer_state.get("current_commitment_level") or 0
    primary_strategy = (directive.get("strategy") or {}).get("primary") or "?"
    concrete_move = (directive.get("strategy") or {}).get("concrete_move") or {}
    move_name = concrete_move.get("name") if isinstance(concrete_move, dict) else None

    # Multi-turn arc awareness (§9.2 #7): phase metadata read from sa_combined.
    phase_meta = sa_combined.get("conversation_phase") if isinstance(sa_combined, dict) else None

    outcome = {
        "session_id": ctx.opp_id,  # not the real session_id but stable per-opp
        "outcome_label": "in_progress",
        "commit_peak": commit_now,
        "persuasion_peak": directive.get("confidence", {}).get("overall") or 0,
        "rules": [primary_strategy] + ([move_name] if move_name else []),
        "phase": phase_meta,
    }

    try:
        from supervisor_full import emit_decision_trace
        result = await emit_decision_trace(ctx.opp_meta, directive, outcome)
        emitted = bool(result.get("emitted"))
        log.info("chain.decision_trace_emit: emitted=%s strategy=%s move=%s",
                 emitted, primary_strategy, move_name)
        return {
            "emitted": emitted,
            "strategy": primary_strategy,
            "concrete_move": move_name,
            "phase": (phase_meta or {}).get("current") if phase_meta else None,
            "src": result.get("src"),
            "tgt": result.get("tgt"),
            "_error": result.get("_error"),
        }
    except Exception as e:
        log.warning("chain.decision_trace_emit: failed: %s", e)
        return {"emitted": False, "_error": str(e)}


# ── 2026-05-13 — Directive loop-breaker ─────────────────────────────────────
#
# Detects the "customer is asking the same question again because the agent
# didn't answer it" pattern. When triggered, mutates the directive in place
# to force a direct acknowledgement, overriding pivots to features.
#
# Signal:
#   (a) Semantic match against `question_repeated_unanswered` intent on the
#       customer's most recent turn (explicit call-out language like "you
#       still didn't answer", "you're not explaining"), OR
#   (b) High cosine similarity (≥ 0.70) between the last two customer turns
#       — same question rephrased, no explicit call-out yet.
#
# When either fires, the supervisor's directive is mutated:
#   - rules[] prepended with "Address the literal question. Do not pivot."
#   - must_not_say[] extended with feature-list / boilerplate phrasings
#   - knowledge.must_say[] extended with "acknowledge what changed" instruction
#
# Telemetry is returned so the trace logger captures when this fired.

@register_programmatic_stage("prompt_directive_loop_breaker")
async def stage_directive_loop_breaker(ctx: ChainContext) -> dict:
    """Detect customer-question repetition + mutate directive to force a
    direct acknowledgement instead of feature pivot."""
    # Need at least one customer turn to evaluate
    customer_turns = [m for m in (ctx.dialog or [])
                      if (m.get("role") or "") == "customer" and (m.get("text") or "").strip()]
    if not customer_turns:
        return {"fired": False, "skipped": "no_customer_turns"}

    last_cust = customer_turns[-1].get("text", "").strip()
    prev_cust = customer_turns[-2].get("text", "").strip() if len(customer_turns) >= 2 else ""

    fired = False
    reason_bits: list[str] = []
    intent_sim = 0.0
    pair_sim = 0.0

    # Signal A — explicit "you didn't answer me" semantic intent
    try:
        from intent_classifier import intent_score, is_available, _load_model, _normalize
        if is_available():
            decision, intent_sim, anchor = intent_score(
                last_cust, "question_repeated_unanswered")
            if decision:
                fired = True
                reason_bits.append(
                    f"intent:question_repeated_unanswered({intent_sim:.2f})→{anchor[:40]!r}")
            # Signal B — pair similarity between last two customer turns.
            # Catches loops where the customer rephrases without using
            # "you didn't answer" call-out language.
            if not fired and prev_cust and len(prev_cust) > 20 and len(last_cust) > 20:
                import numpy as np
                model = _load_model()
                if model is not None:
                    vecs = _normalize(model.encode([last_cust, prev_cust]))
                    pair_sim = float(vecs[0] @ vecs[1])
                    if pair_sim >= 0.70:
                        fired = True
                        reason_bits.append(
                            f"pair_similarity:{pair_sim:.2f} (customer rephrased)")
    except Exception as e:
        log.warning("loop_breaker: intent_classifier failed: %s", e)
        return {"fired": False, "_error": str(e)[:200]}

    if not fired:
        return {"fired": False, "intent_sim": intent_sim, "pair_sim": pair_sim}

    # Mutate the directive in place. The directive lives in
    # ctx.previous_responses["prompt_signal_analysis_combined"]["directive"].
    sa_combined = ctx.previous_responses.get("prompt_signal_analysis_combined") or {}
    directive = sa_combined.get("directive") if isinstance(sa_combined, dict) else None
    if not isinstance(directive, dict):
        log.info("loop_breaker: FIRED but no directive to mutate (reason=%s)",
                 "; ".join(reason_bits))
        return {"fired": True, "reason": "; ".join(reason_bits),
                "mutation": "skipped_no_directive",
                "intent_sim": intent_sim, "pair_sim": pair_sim}

    # Prepend corrective rules
    rules = directive.get("rules")
    if not isinstance(rules, list):
        rules = []
    corrective_rules = [
        "LOOP_BREAK: Address the customer's literal question directly. Do NOT pivot to features, benefits, or coverage details.",
        "LOOP_BREAK: If your prior message contradicted itself (e.g., you said 'can't match X' then offered X), explicitly acknowledge what changed (manager approval, new info checked, etc.).",
        "LOOP_BREAK: Start your reply with 'You're right' or 'Let me directly answer' — name the question, then answer it.",
    ]
    directive["rules"] = corrective_rules + rules

    # Extend must_not_say to block the pivots that caused the loop
    knowledge = directive.get("knowledge")
    if not isinstance(knowledge, dict):
        knowledge = {}
        directive["knowledge"] = knowledge
    mns = knowledge.get("must_not_say")
    if not isinstance(mns, list):
        mns = []
    pivot_blockers = [
        "list of policy benefits",
        "comprehensive coverage details unless directly asked",
        "headlights and mirrors coverage as a deflection",
        "competitor comparison as a deflection",
    ]
    for p in pivot_blockers:
        if p not in mns:
            mns.append(p)
    knowledge["must_not_say"] = mns

    # Tag the strategy as overridden for telemetry
    strat = directive.get("strategy") or {}
    if isinstance(strat, dict):
        strat["_loop_break_applied"] = True
        directive["strategy"] = strat
    directive["_loop_break_applied"] = True
    directive["_loop_break_reason"] = "; ".join(reason_bits)

    log.info("chain.loop_breaker: FIRED — reason=%s. Mutated directive.rules + must_not_say.",
             "; ".join(reason_bits))

    return {
        "fired": True,
        "reason": "; ".join(reason_bits),
        "intent_sim": intent_sim,
        "pair_sim": pair_sim,
        "n_corrective_rules": len(corrective_rules),
        "n_pivot_blockers": len(pivot_blockers),
    }


# ── 2026-05-11 — Retrieval-augmented imitation (option B from CTO discussion) ──

@register_programmatic_stage("prompt_exemplar_retrieval")
async def stage_exemplar_retrieval(ctx: ChainContext) -> dict:
    """Per-turn RAG of real won-deal exemplars from CG.

    Replaces hand-authored must_say_template + required_phrases (the
    whack-a-mole pattern) with retrieved exemplars: "in similar situations,
    real winning agents said X." Stored in ctx.previous_responses so the
    actor's prompt-build can inject the rendered exemplars block.

    Gated by POC_EXEMPLAR_RAG=1 for clean A/B vs the script-based pipeline.

    Failure mode: never blocks the chain. If CG query fails or returns
    nothing, the actor falls back to script-based scaffolding as before.
    """
    if os.environ.get("POC_EXEMPLAR_RAG", "0").strip().lower() not in ("1", "true", "yes"):
        return {"enabled": False, "_skipped": "POC_EXEMPLAR_RAG not set"}

    p = ctx.opp_meta or {}
    tenant = (p.get("company") or "").strip()
    dialog = ctx.dialog or []
    # Find customer's most recent message
    last_cust = ""
    for m in reversed(dialog):
        if m.get("role") == "customer" and m.get("text"):
            last_cust = m["text"]
            break
    if not tenant or not last_cust:
        return {"enabled": True, "_skipped": "no tenant or no customer turn yet"}

    try:
        from exemplar_retrieval import retrieve_exemplars, render_exemplars_block
        result = retrieve_exemplars(tenant=tenant, current_customer_turn=last_cust)
        rendered = render_exemplars_block(result)
        return {
            "enabled": True,
            "n_won": len(result.get("won") or []),
            "n_lost": len(result.get("lost") or []),
            "n_chunks_seen": result.get("n_chunks_seen", 0),
            "rendered_block": rendered,
            "raw_exemplars": result,  # for telemetry
        }
    except Exception as e:
        log.warning("exemplar_retrieval stage failed (non-fatal): %s", e)
        return {"enabled": True, "_error": str(e)[:200]}


# ── 2026-05-11 — Production escalation rule honoring ──────────────────────────

@register_programmatic_stage("prompt_escalation_router")
async def stage_escalation_router(ctx: ChainContext) -> dict:
    """Honor production's prompt_co_pilot_escalation rule.

    Production Agent already has security/escalation rules encoded in the prod
    chain's prompt_co_pilot_escalation stage. The stage runs (postprocessing,
    order 5) and returns:
        {"is_escalation_needed": bool, "reason": str}

    The supervisor pipeline previously stored this output in
    ctx.previous_responses["prompt_co_pilot_escalation"] and discarded it.
    This consumer reads it and, on positive escalation signal:

      1. Marks ctx.early_exit so downstream sales stages stop
      2. Replaces the actor's generated text with a hand-off message
         that surfaces the tenant's actual support channel from anchors/CG
      3. Emits structured metadata the panel-end logic uses to label the
         outcome as `escalated_to_human` rather than a sales loss

    Architectural rationale: prod escalation rules are security-vetted and
    cover cases sales scripts cannot handle (refund disputes, fraud signals,
    bot-confusion legal requirements). Ignoring them in the supervised
    pipeline regresses safety below vanilla Agent. This stage reuses prod's
    rule rather than reimplementing it.

    Failure mode: never blocks the chain. If escalation output is missing
    or malformed, the stage is a no-op.
    """
    co_pilot_out = ctx.previous_responses.get("prompt_co_pilot_escalation") or {}
    if not isinstance(co_pilot_out, dict):
        return {"escalation_consumed": False, "_skipped": "non-dict output"}

    is_escalation = co_pilot_out.get("is_escalation_needed")
    if not is_escalation:
        return {"escalation_consumed": True, "escalate": False,
                "reason": co_pilot_out.get("reason", "")}

    reason = (co_pilot_out.get("reason") or "user requested human")
    reason_lower = reason.lower()

    # 2026-05-11 — Verify trigger against the CUSTOMER's actual message, not
    # against Flash's narrative reasoning. Flash quotes rule names verbatim
    # when invoking them ("...falls under: 'You didn't answer my previous
    # question.'") — matching keywords in Flash's reason text matches the
    # rule paraphrase, not the customer's actual words. The discipline:
    # trust the prod rule's detection only when the customer's own message
    # contains the explicit trigger phrase.
    dialog = (ctx.opp_meta or {}).get("_dialog_for_validator") or ctx.dialog or []
    last_cust_text = ""
    for m in reversed(dialog):
        if m.get("role") == "customer" and m.get("text"):
            last_cust_text = m["text"]
            break
    last_cust_lower = last_cust_text.lower()

    # 2026-05-11 — Hybrid: model2vec for the "explicit human request" intent
    # (paraphrase-rich; regex misses synonyms like "get me a real person"),
    # narrow keyword list for the other categories where wording is
    # constrained ("code didn't work", "you didn't answer my previous
    # question"). Phrase-specific categories don't benefit from embeddings
    # because their false-positive risk on adjacent phrasings is high.
    NARROW_KEYWORD_TRIGGERS = (
        # Coupon failure
        "code didn't work", "code did not work", "coupon didn't work",
        "coupon failed", "discount wasn't applied", "discount was not applied",
        "the code threw an error", "code gave me an error",
        # Misunderstanding accusation (direct, per prod rule's own examples)
        "you didn't understand me", "you did not understand me",
        "that is not what i asked", "that's not what i asked",
        "you are ignoring my question", "you're ignoring my question",
        "you didn't answer my previous question",
        "you did not answer my previous question",
        "you keep ignoring",
        # Conflict-with-internet (only when customer makes a direct accusation)
        "your website says different", "the website contradicts",
        "you're lying", "that's a lie", "you are wrong",
    )
    # Loop guard: the rule may interpret our OWN prior handoff message as
    # "previous assistant has already initiated an escalation". Suppress.
    SELF_REFERENCE_PHRASES = (
        "i'm flagging this to a human",
        "support@ecommerce.com",
        "support@",
        "priority-flagged",
        "previous assistant has already initiated",
        "the assistant's last message",
        "previously initiated an escalation",
    )
    if any(s in reason_lower for s in SELF_REFERENCE_PHRASES):
        log.info("escalation_router: SUPPRESSED (self-reference loop) reason=%s",
                 reason[:120])
        return {"escalation_consumed": True, "escalate": False,
                "_filtered": "self_reference_loop",
                "reason": reason}

    # Narrow phrase-specific match (coupon failure / misunderstanding / conflict)
    narrow_matched = [t for t in NARROW_KEYWORD_TRIGGERS if t in last_cust_lower]

    # Semantic match for the human-request intent (model2vec)
    human_request_match = False
    h_score = 0.0
    h_anchor = ""
    try:
        from intent_classifier import intent_score, is_available
        if is_available() and last_cust_text:
            human_request_match, h_score, h_anchor = intent_score(
                last_cust_text, "human_request")
    except Exception as e:
        log.warning("escalation_router: intent_classifier failed: %s", e)

    if not (narrow_matched or human_request_match):
        # Prod rule fired but customer's actual message doesn't match either
        # the narrow-keyword categories or the semantic human-request anchors.
        # Discipline: verify against source evidence, not Flash paraphrase.
        log.info("escalation_router: SUPPRESSED (no customer-side trigger) "
                 "prod_reason=%s customer_text=%s human_score=%.3f",
                 reason[:120], last_cust_text[:120], h_score)
        return {"escalation_consumed": True, "escalate": False,
                "_filtered": "no_customer_side_evidence",
                "reason": reason,
                "human_request_score": h_score,
                "customer_text_checked": last_cust_text[:200]}
    matched_triggers = narrow_matched + (
        [f"semantic:human_request:{h_anchor}({h_score:.2f})"]
        if human_request_match else []
    )
    p = ctx.opp_meta or {}
    tenant = (p.get("company") or "").strip()

    # Build hand-off message. Prefer tenant-specific support channel info from
    # CG (looked up in anchors / business-rules at session start). Fall back
    # to a generic acknowledgment if no support channel surfaced.
    support_channel = (ctx.anchors or {}).get("support_channel") or \
                      (ctx.anchors or {}).get("escalation_contact") or ""
    if not support_channel:
        # Cheap heuristic — tenant-known channels (would be loaded from CG in
        # production via the 0d depth-of-content audit step's escalation
        # content requirement)
        if tenant == "Ecommerce":
            support_channel = "support@ecommerce.com (priority-flagged for human callback)"
        elif tenant == "Insurance":
            support_channel = "*5556 / human agent (priority-flagged)"
        else:
            support_channel = "human support team (priority-flagged)"

    handoff_text = (
        f"Understood. I'm flagging this to a human now — "
        f"{support_channel} will follow up directly. "
        f"You'll have priority routing based on this conversation. "
        f"I'll stop here so the human can take it from here. "
    )

    # Override actor's response with the hand-off text. The chain_executor
    # reads previous_responses[REGENERATE_LOOP_TARGET] (= prompt_build_answer)
    # so we replace that.
    target_name = "prompt_build_answer"
    prior = ctx.previous_responses.get(target_name) or {}
    if isinstance(prior, str):
        ctx.previous_responses[target_name] = handoff_text
    elif isinstance(prior, dict):
        # Preserve other fields, override the text
        prior_copy = dict(prior)
        prior_copy["_raw"] = handoff_text
        prior_copy["_escalation_handoff"] = True
        ctx.previous_responses[target_name] = prior_copy

    # Mark early_exit so any remaining stages in the chain skip
    ctx.early_exit = True
    ctx.early_exit_reason = "escalation_required"

    log.info("escalation_router: ESCALATION FIRED — reason=%s tenant=%s channel=%s",
             reason, tenant, support_channel[:40])

    return {
        "escalation_consumed": True,
        "escalate": True,
        "reason": reason,
        "handoff_channel": support_channel,
        "handoff_text": handoff_text,
        "early_exit_set": True,
    }
