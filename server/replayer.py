"""Parallel A/B replayer — both panels, same starting point, supervisor is the only variable.

Per session:
  Seed phase: stream historical messages to both panels (same content, same timing)
  Live phase: alternating agent + customer turns; LEFT no directive, RIGHT with Mode 1a v1 directive
  Stop: per panel — commitment_5, customer decline, or 25-turn timeout

Public API:
  await run_session(session_id, opp_id, ws_send, get_speed) -> None
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Awaitable, Callable

# sys.path setup — works in two layouts:
#   (1) original poc-supervisor-strategist: this file is at server/, runners at
#       <repo>/ai-agent-sales-improvement/runners. POC_RUNNERS_PATH points there.
#   (2) handoff POC package: this file is at server/, the substrate is at
#       <pkg>/poc/, strategist runners at <pkg>/poc/strategist/runners.
_SD = os.path.dirname(os.path.abspath(__file__))
_PR = os.path.dirname(_SD)  # package root
# Reverse order: last insertion ends up at position 0; server/ must win
# for db.py (JSON shim) to override the MySQL-talking poc/db.py.
for _p in (
    os.path.join(_PR, "poc", "strategist", "runners"),
    os.path.join(_PR, "poc", "strategist"),
    os.path.join(_PR, "poc"),
    _SD,
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
RUNNERS_PATH = os.environ.get(
    "POC_RUNNERS_PATH",
    os.path.join(_PR, "poc", "strategist", "runners"),
)
if os.path.isdir(RUNNERS_PATH) and RUNNERS_PATH not in sys.path:
    sys.path.insert(0, RUNNERS_PATH)

import anthropic
from db import (
    open_conn, fetch_opp_meta, fetch_messages, fetch_turn_states,
    fetch_persuasive_scores, fetch_business_rules, find_failure_mode_turn_index,
    find_supervisor_intervention_index, fetch_insurance_anchors,
)
from session_logger import SessionLogger
from actor import generate as actor_generate, build_actor_system
from staircase_gate import check_staircase, build_correction_prompt
from customer_simulator import (
    CustomerSimulator, detect_decline, detect_close_signal,
    detect_agent_refusal, detect_agent_graceful_close, detect_language,
    detect_customer_farewell, detect_saturation, detect_counter_offer,
)
from persuasion_scorer import score_turn

# v1 supervisor pieces — strategist-native path only. batch9_5_mode1a_v1
# extends a research module that is NOT bundled in the standalone package, so
# the import is guarded: without it the classifier stays None and the mode-1a
# tier is skipped (both call sites already handle that), while baseline /
# planner / plugin runs are unaffected.
try:
    from batch9_5_mode1a_v1 import load_classifier_v1, mode1a_directive_v1
except Exception as _mode1a_err:  # ModuleNotFoundError in the standalone package
    load_classifier_v1 = None
    mode1a_directive_v1 = None
    logging.getLogger(__name__).info(
        "mode-1a supervisor unavailable (%s); strategist mode-1a tier disabled",
        _mode1a_err)
try:
    from batch9_0a_attribution import features_for_opp, fetch_opp_data
except Exception as _attr_err:  # needs pymysql — vendor/prod path only
    features_for_opp = None
    fetch_opp_data = None
    logging.getLogger(__name__).info(
        "attribution features unavailable (%s); classifier feature "
        "pre-compute disabled", _attr_err)

# Full architecture: Mode 1a (compiled playbook), Mode 1b (live CG), Mode 2 (CGR3 multi-hop)
from supervisor_full import (
    mode1b_directive, mode2_directive, emit_decision_trace,
    fetch_precedent_decisions, QUERY_CACHE_STATS, USE_QUERY_AUTO,
    CG_ENDPOINT_CALLS, _directive_strategy,
)
from cluster_plan import load_plan, PlanState, advance_plan, record_strategy_adherence, can_graceful_close
import win_plan as wp_mod
import win_proximity
import httpx

POC_WIN_PLAN_ENABLED = os.environ.get("POC_WIN_PLAN_ENABLED", "0") == "1"

# POC_MODE: '1a' (compiled playbook only), '1b' (live Sonnet + CG only),
# 'auto' (full hybrid — tier router picks per turn). Default 'auto' shows the
# real architecture: cheap-when-routine, premium-when-stakes-high, plus
# closed-loop learning via decision-trace emission.
POC_MODE = os.environ.get("POC_MODE", "auto")


def select_tier(panel, classifier_confidence: float, has_playbook: bool) -> str:
    """Heuristic tier router. Returns 'mode_1a' | 'mode_1b' | 'mode_2'.

    Decision rules (matching the architecture spec):
      - High-stakes close: customer at commit >= 4 → Mode 2 (CGR3 multi-hop)
      - Recovery moment: commit dropped >=2 from peak → Mode 2
      - Mode 1a auto-escalation: 2+ consecutive Mode 1a turns → Mode 1b
      - Routine + classifier confident + playbook hit → Mode 1a (cheap)
      - Default novel turn → Mode 1b (live reasoning)
    """
    ch = panel.commitment_history
    # High-stakes close moment (customer ready to buy)
    if ch and ch[-1] >= 4:
        return "mode_2"
    # Recovery moment — sharp drop from peak
    if len(ch) >= 2:
        peak = max(ch)
        latest = ch[-1]
        if peak >= 3 and (peak - latest) >= 2:
            return "mode_2"
    # Mode 1a auto-escalation: Mode 1a returns the SAME static playbook
    # directive on every call, so >=2 consecutive Mode 1a turns guarantees
    # strategy repetition. The 3rd call would be the 3rd repetition — force
    # to Mode 1b so the live LLM with session-memory can diversify.
    # Empirical motivation: dfb34792 c80947a2 session 2026-05-01.
    if getattr(panel, 'consecutive_mode1a', 0) >= 2:
        return "mode_1b"
    # Plan-loaded override: when a ClusterPlan is loaded the supervisor needs
    # to honor the plan's phase context (qualify_need is different from
    # 'ask probing question' even though both are 'information' strategy).
    # Mode 1a's static playbook directive can't see the plan; Mode 1b can.
    # Empirical motivation: dfb34792 1faf5927 — Mode 1a ignored the plan's
    # phase 1, asked the wrong question, customer pushed back, abort fired.
    if getattr(panel, 'plan_state', None) is not None:
        return "mode_1b"
    # Routine cheap path — match Mode 1a's internal gate (0.35) so we don't
    # double-gate
    if classifier_confidence >= 0.35 and has_playbook:
        return "mode_1a"
    return "mode_1b"

log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
HAIKU_CLIENT = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

MAX_LIVE_TURNS = 8    # default max agent-turn iterations per panel after seed
                       # (≈ 4-5 min/session at instant speed with Gemini 2.5 Pro)
# Per-tenant override — Ecommerce customers ask multi-axis questions (drivers +
# price + warranty + shells), so 8 turns ceilings them at commit=4. Insurance is
# linear (price → renew), so 8 turns is enough. Empirically calibrated from
# the 5-scenario Ecommerce batch on 2026-04-30 (all 5 ceilinged at c4 in 8 turns).
MAX_LIVE_TURNS_BY_TENANT = {
    # 2026-05-12 — Bumped from 12 → 18 after session 13f9b2c6 / opp e4bbc864
    # timed out on an engaged customer asking the final-justification question
    # (post factual-error recovery at agent t30, customer t31 asked
    # "why pay $199 over DT177X $120?"). Conversation was clearly in closing
    # arc; cap killed it. Same pattern that forced the Insurance 12→20 bump on
    # 2026-05-10. Ecommerce multi-axis questions (drivers + price + warranty +
    # trust + factual-error recovery) need more runway.
    "Ecommerce": 18,
    "Insurance":  20,   # was 12 — engaged-but-frustrated customer trajectories need
                    # more turns to recover. Real Insurance won-deals in our 12,401-deal
                    # mining often run 30+ total turns. Engagement-override at the
                    # stall-gate keeps panels alive through frustrated questions, but
                    # the absolute turn cap was killing them at T30. Bumped to 20 live
                    # turns (~30+ total counting seed). 2026-05-10. Was: 12 — Insurance c5
                    # 'competed-away' needs ≥12 turns to recover
                    # under 8-turn ceiling, both panels timed out at commit=4)
}
STALL_THRESHOLD_TURNS = 3   # if N consecutive turns score < 0.3 AND commit ≤ 1, end as lost
COMMITMENT_WIN_THRESHOLD = 5

# Three-of-three close guard (2026-05-04). Defends against persuasion-scorer
# false positives — Gemini sometimes returns commitment_level=5 on
# price-objection questions like "Why the change? Last year was 12." because
# the message is short, contains a number, and follows agent CTA. To declare
# a win we now require all three: G1 commitment_level=5 (the existing check),
# G2 either close-language regex OR persuasion score >= threshold, and G3
# the previous agent message must have actually offered a close.
PERSUASION_CONFIRM_THRESHOLD = 0.85   # G2 — corroborating persuasion score
CLOSE_MOVE_NAMES = {
    "payment_plan_offer", "apply_promo_code", "match_competitor_offer",
    "direct_renewal_ask", "differentiation_and_close", "cta_to_renew",
    "link_to_renewal", "send_renewal_link", "surgical_third_party_reduction",
}
CLOSE_STRATEGIES = {"commitment", "direct_ask"}
_CLOSE_TEXT_REGEX_EN = re.compile(
    r"\brenew\?|\bsign up\b|\blet'?s go\b|\bgo ahead\b"
    r"|\bi'?ll send (?:you )?the link\b|\bhere'?s the link\b"
    r"|\bshall (?:i|we) (?:proceed|go ahead|send)\b"
    r"|\bcan i (?:send|share) the link\b",
    re.IGNORECASE,
)
_CLOSE_PHRASES_HE = ["תחדש", "אשלח לך", "מאשר", "סגור", "בוא נחדש", "שולח לך לינק"]

# 2026-05-05 — Explicit payment-info bypass for G3 (close-guard). When the
# customer provides CC last-4, full card digits, or "card ending in NNNN",
# that's a definitive close regardless of whether agent explicitly asked.
# Triggered case: a897d59b turn 31 — customer said "last 4 digits are 7834"
# after agent's concession statement; G3 was rejecting because agent didn't
# match the close-offer regex. Now this regex makes G3 irrelevant on payment.
_EXPLICIT_PAYMENT_REGEX = re.compile(
    r"\blast\s+(?:4|four)\s+(?:digits?|nums?)\s*(?:are\s+|is\s+|:\s*)?\d{4}\b"
    r"|\bcc[\s:#-]+\d{4}\b"
    r"|\bcredit\s+card\s+(?:is\s+|number\s+is\s+|:\s*)?\d{4}\b"
    r"|\b(?:my )?card\s+ending\s+(?:in\s+)?\d{4}\b"
    r"|\bcc[\s:]*[\d\s\-]{4,}\b",
    re.IGNORECASE,
)


def _agent_just_offered_close(prev_directive: dict | None,
                              prev_agent_text: str,
                              tenant: str | None = None) -> bool:
    """G3 of the three-of-three close guard. True if the previous agent
    message actually offered a close — CTA / payment / renewal / link.
    Customer can't accept what wasn't offered.

    Three signal paths:
      1. Structured directive: strategy.primary or primary_strategy in
         CLOSE_STRATEGIES (most reliable when supervisor authored the turn)
      2. Semantic match against mined agent close-CTA anchors (model2vec) —
         covers Mode 1a cached directives where the strategy label was lost
      3. Regex fallback on close-language phrases
    """
    if isinstance(prev_directive, dict):
        strat = prev_directive.get("strategy") or {}
        if isinstance(strat, dict):
            if strat.get("primary") in CLOSE_STRATEGIES:
                return True
            cm = strat.get("concrete_move")
            if isinstance(cm, dict) and cm.get("name") in CLOSE_MOVE_NAMES:
                return True
        if prev_directive.get("primary_strategy") in CLOSE_STRATEGIES:
            return True
    # 2026-05-13 — Semantic check against mined agent close-CTA anchors.
    # Catches Mode 1a directives whose strategy label was lost / mis-cached.
    if prev_agent_text:
        try:
            from intent_classifier import intent_score, is_available
            if is_available():
                decision, _s, _a = intent_score(
                    prev_agent_text, "agent_close_offer", tenant=tenant)
                if decision:
                    return True
        except Exception:
            pass
    if prev_agent_text:
        if _CLOSE_TEXT_REGEX_EN.search(prev_agent_text):
            return True
        for he in _CLOSE_PHRASES_HE:
            if he in prev_agent_text:
                return True
    return False


# 2026-05-17 (dcfbdb58 finding). A bare "7834" — the natural reply to "what
# are the last 4 digits?" — has NO semantic content, so model2vec cannot
# carry it (per the 2026-05-11 vectors-over-regex decision, the structural
# fallback owns cases the anchor set can't represent). This matches a
# message that is essentially just a 3-6 digit run with trivial punctuation.
# It is ONLY honored when an agent turn in the recent window actually asked
# for payment / offered close (see _agent_recently_offered_close), so a bare
# counter-offer like "3000" cannot trip it out of context.
_BARE_LAST4_RE = re.compile(r"^[\s#:*.\-]*\d{3,6}[\s#:*.\-]*$")


def _agent_recently_offered_close(panel, opp_meta: dict | None,
                                  lookback: int = 6) -> bool:
    """True if ANY agent turn in the last `lookback` dialog entries offered a
    close / asked for payment. Fixes the original bug where G3 only inspected
    the immediately-prior agent line (a "take your time" wait-ack) and missed
    the real "what are the last 4 digits?" ask two turns earlier."""
    tenant = (opp_meta or {}).get("company")
    seen = 0
    for m in reversed(panel.dialog):
        if m.get("role") != "agent":
            continue
        if _agent_just_offered_close(None, m.get("text") or "", tenant=tenant):
            return True
        seen += 1
        if seen >= lookback:
            break
    return False


def _close_already_occurred(panel, opp_meta: dict | None) -> bool:
    """True only if the CUSTOMER has affirmatively accepted earlier in this
    panel — so §4b never stamps a post-close pleasantry as a loss.

    2026-05-19 correction: the prior version returned True on (a) the AGENT
    merely asserting it sent documents/renewal, or (b) a single scorer
    commit==WIN reading. Both are false-positive sources: "agent sent the
    details" + customer "I'll review and get back to you if I decide" is the
    single most common NON-close, and the scorer assigns commit=5 to such
    deferrals. A real prior close requires hard CUSTOMER acceptance:
      1. payment_seen  (customer supplied card/payment), OR
      2. a prior CUSTOMER turn that is an explicit acceptance / close
         (close-phrase, explicit payment text, or model2vec
         customer_initiated_close).
    Agent claims and scorer commit alone are NOT sufficient.
    """
    if panel.payment_seen:
        return True
    tenant = (opp_meta or {}).get("company")
    for m in panel.dialog:
        if m.get("role") != "customer":
            continue
        txt = m.get("text") or ""
        if not txt.strip():
            continue
        if detect_close_signal(txt) or _EXPLICIT_PAYMENT_REGEX.search(txt):
            return True
        try:
            from intent_classifier import intent_score, is_available
            if is_available():
                decision, _s, _a = intent_score(
                    txt, "customer_initiated_close", tenant=tenant)
                if decision:
                    return True
        except Exception:
            pass
    return False


def max_live_turns_for(opp_meta: dict) -> int:
    return MAX_LIVE_TURNS_BY_TENANT.get(opp_meta.get("company"), MAX_LIVE_TURNS)

SEND_FN = Callable[[dict], Awaitable[None]]
SPEED_FN = Callable[[], str]


def _speed_to_delay_ms(speed: str, base_gap_ms: int = 800) -> int:
    """Map speed setting → delay between events. base_gap_ms is real-conversation
    typical pacing; 1x respects message timestamps; 5x compresses; instant snaps."""
    if speed == "instant":
        return 50
    if speed == "5x":
        return max(150, base_gap_ms // 5)
    return base_gap_ms  # 1x


@dataclass
class PanelState:
    side: str  # "left" or "right"
    dialog: list[dict] = field(default_factory=list)
    chat_for_scorer: list[dict] = field(default_factory=list)  # accumulated, with persuasive_score
    persuasion_history: list[float] = field(default_factory=list)   # seed + live (mixed)
    commitment_history: list[int] = field(default_factory=list)     # live only
    # Live-phase persuasion track aligned with commitment_history (same length,
    # same indices). Used by velocity metrics (time_to_persuasion_0_8, AUC).
    # We can't reuse persuasion_history because it includes seed-phase scores.
    live_persuasion_history: list[float] = field(default_factory=list)
    # Session memory: ordered sequence of primary_strategy choices the supervisor
    # has emitted in this session (right panel only). Fed back into next turn's
    # supervisor prompt so the LLM doesn't loop on the same move.
    # Empirical motivation: 6/6 sessions on 2026-04-30 showed 54% strategy
    # repetition rate without this signal.
    strategies_used: list[str] = field(default_factory=list)
    # Auto-escalation counter — number of consecutive Mode 1a turns this panel
    # has had. Mode 1a returns the SAME static playbook directive every call,
    # so >=2 consecutive Mode 1a turns guarantees strategy repetition. We
    # force-escalate to Mode 1b after 2 consecutive Mode 1a calls, letting
    # session-memory diversify. Reset to 0 when a non-1a tier is used.
    # Empirical motivation: dfb34792 UI session 2026-05-01 — 4 consecutive
    # Mode 1a 'information' turns drove customer persuasion 0.75 → 0.1.
    consecutive_mode1a: int = 0
    # ClusterPlan v0 — session-level goal + ordered phases. None when no plan
    # exists for the (cluster, motivator, decision_logic) cell. The plan_state
    # is mutated server-side per turn (deterministic state machine, not LLM).
    plan_state: object = None  # PlanState | None — typed as object to avoid forward-ref issues
    # Win-mode plan (T-76 Path B Day 4) — cell-keyed plan with conditional
    # branches, activated when engagement gate fires (positive trajectory).
    # Coexists with plan_state; only one is active at a time. Gated by
    # POC_WIN_PLAN_ENABLED env var (default off so behavior unchanged).
    win_plan_state: object = None
    plan_mode_active: str = "loss"  # "loss" (cluster_plan) or "win" (win_plan)
    won: bool = False
    lost: bool = False
    end_reason: str | None = None
    last_directive: dict | None = None
    seq_counter: int = 0
    # T-83 anti-staircase gate state — accumulates agent's prior offers/
    # discounts to detect staircase concessions. Each entry is a dict like
    # {"type": "price"|"discount"|"code", "amount": float, "currency": str,
    #  "turn": int}. See server/staircase_gate.py.
    agent_concessions: list = field(default_factory=list)
    # R7 (2026-05-04) — per-session concrete-move usage history. Drives
    # variation pressure: when a Tier-2 move has fired N times this session,
    # the next turn's supervisor prompt gets a "recently used" marker for it,
    # encouraging move variation. Empirical motivation: bcbe5f71 lift-batch
    # session showed apples_to_apples_probe firing 5x in a row.
    moves_used: dict = field(default_factory=dict)
    # 2026-05-17 (dcfbdb58 finding) — latched once the customer has provided
    # payment info (narrated or bare-digit). Lets a trailing polite "Thank
    # you" be classified post-close-won instead of customer_polite_close=lost.
    payment_seen: bool = False


@dataclass
class SessionCtx:
    """Mutable shared state — `stopped` is checked before each LLM call to support
    fast cancellation when the user clicks Stop. `scenarios` accumulates which
    INTEGRATION.md §4 scenarios this session has demonstrated."""
    stopped: bool = False
    # Set of scenario IDs demonstrated so far (e.g., "S1", "S2", "S4")
    scenarios: set[str] = field(default_factory=set)


# Map INTEGRATION.md §4 scenarios → display metadata.
GUIDE_SCENARIOS = [
    {"id": "S1", "title": "Product Knowledge",       "blurb": "kg_products via /query"},
    {"id": "S2", "title": "Business Rules Retrieval","blurb": "kg_rules via /query"},
    {"id": "S3", "title": "Conversation Intelligence","blurb": "Won-deal patterns via /query"},
    {"id": "S4", "title": "Decision Audit Trail",    "blurb": "/graph/decision/emit"},
    {"id": "S5", "title": "Prompt Improvement w/ KG","blurb": "Multi-hop reasoning (CGR3)"},
    {"id": "S6", "title": "Follow-Up Personalization","blurb": "(out of scope for live A/B)"},
    {"id": "S7", "title": "Decision Precedent Lookup","blurb": "/graph/decisions read-side"},
    {"id": "S8", "title": "Structured Data Retrieval","blurb": "/query/data (no LLM)"},
]


def _evidence_to_scenarios(evidence: dict, mode: str) -> set[str]:
    """Map a single Mode 1b/2 directive's CG evidence to scenarios. Mode 1b
    always demonstrates S8 (raw /query/data retrieval). The chunk-classification
    flags decide which content-type scenarios fired."""
    fired = set()
    if mode in ("1b", "2"):
        fired.add("S8")  # /query/data was used as the underlying retrieval path
    if mode == "2":
        fired.add("S5")  # CGR3 multi-hop = Scenario 5 (advanced reasoning)
    if evidence.get("products"):  fired.add("S1")
    if evidence.get("rules"):     fired.add("S2")
    if evidence.get("patterns"):  fired.add("S3")
    if evidence.get("decisions"): fired.add("S7")
    return fired


def _build_seed_dialog(messages_history: list[dict], failure_idx: int) -> list[dict]:
    """Take messages up to (and including) the customer's last meaningful inbound."""
    seed_msgs = messages_history[: failure_idx + 1]
    out = []
    for i, m in enumerate(seed_msgs):
        out.append({
            "role": "customer" if m.get("direction") == "inbound" else "agent",
            "text": (m.get("text") or "").strip(),
            "sequence_number": i,
        })
    return out


async def _stream_seed(panel: PanelState, seed_dialog: list[dict], opp_id: str,
                        send: SEND_FN, get_speed: SPEED_FN,
                        precomputed_scores: dict, turn_states: list[dict]) -> None:
    """Send each seed message to the panel; emit persuasion-score events for inbound
    turns using precomputed values where available, with live-Gemini fallback if not."""
    for m in seed_dialog:
        delay = _speed_to_delay_ms(get_speed())
        await asyncio.sleep(delay / 1000.0)
        panel.dialog.append(m)
        panel.seq_counter = m["sequence_number"]
        await send({
            "event": f"{panel.side}_msg",
            "role": m["role"],
            "text": m["text"],
            "turn": m["sequence_number"],
        })
        # For seed inbound turns: emit a score event
        if m["role"] == "customer":
            ts_match = next((t for t in turn_states
                             if t.get("sequence_number") == m["sequence_number"]), None)
            score = None
            commitment = None
            if ts_match:
                pscore = ts_match.get("persuasion_score")
                if pscore is not None:
                    score = float(pscore)
                cl = ts_match.get("commitment_level")
                if cl is not None:
                    commitment = int(cl)
            # Fallback: if no precomputed score, run live Gemini scoring on the seed
            if score is None:
                try:
                    panel.chat_for_scorer = _build_chat_for_scorer(panel.dialog)
                    s = await score_turn(panel.chat_for_scorer)
                    if s:
                        score = s["score"]
                        commitment = s["commitment_level"]
                except Exception as e:
                    log.warning("[%s] seed-score fallback failed: %s", panel.side, e)
            if score is not None:
                panel.persuasion_history.append(score)
                await send({
                    "event": f"{panel.side}_score",
                    "turn": m["sequence_number"],
                    "score": score,
                    "commitment": commitment,
                })


# Module-level scenarios.json index for cluster_id lookup. Loaded lazily.
_SCENARIOS_BY_OPP: dict | None = None


def _scenarios_cluster_lookup(opp_id: str | None) -> int | None:
    """Resolve cluster_id from data/scenarios.json by opp_id. Cached at module
    level (lazy load). Returns None when opp not in scenarios or cluster_id
    field missing."""
    global _SCENARIOS_BY_OPP
    if not opp_id:
        return None
    if _SCENARIOS_BY_OPP is None:
        try:
            import json
            from pathlib import Path
            path = Path(__file__).resolve().parent.parent / "data" / "scenarios.json"
            with open(path) as f:
                arr = json.load(f)
            _SCENARIOS_BY_OPP = {s.get("opp_id"): s for s in arr if isinstance(s, dict)}
        except Exception as e:
            log.warning("scenarios.json lookup unavailable: %s", e)
            _SCENARIOS_BY_OPP = {}
    rec = _SCENARIOS_BY_OPP.get(opp_id) or {}
    cid = rec.get("cluster_id")
    return cid if isinstance(cid, int) else None


async def _run_via_chain_runner(panel, opp_meta: dict, business_rules: str,
                                  send) -> tuple[str | None, dict, dict | None]:
    """T-85 Phase A.3: route the agent generation through the production-
    equivalent prompt chain.

    Loads the chain definition for (company, opp_type) from prod DB, splices
    supervisor stages into the RIGHT panel's chain, executes sequentially,
    returns the agent text from prompt_build_answer + a directive_meta dict.
    """
    from chain_runner import (
        fetch_chain_definition, splice_supervisor_stages,
        run_chain, ChainContext,
    )
    import chain_stages_supervisor  # registers programmatic stages (idempotent)

    company = opp_meta.get("company") or ""
    opp_type = opp_meta.get("opp_type") or opp_meta.get("type") or ""
    stages = fetch_chain_definition(company, opp_type)
    if not stages:
        log.warning("[%s] chain_runner: no chain found for (%s, %s)",
                       panel.side, company, opp_type)
        return None, {}, None

    # Splice supervisor stages on RIGHT panel only (LEFT runs plain prod chain)
    if panel.side == "right":
        stages = splice_supervisor_stages(stages)

    chain_ctx = ChainContext(
        opp_id=opp_meta.get("id") or "",
        opp_meta=opp_meta,
        dialog=panel.dialog,
        business_rules=business_rules or "",
        agent_concessions=panel.agent_concessions,
        anchors=opp_meta.get("anchors"),
        voice_profile=opp_meta.get("voice_profile"),
        # Q14 (2026-05-04) — measurement parity with legacy path. The
        # supervisor stage now sees cluster_plan + session-memory.
        plan_state=panel.plan_state,
        strategies_used=panel.strategies_used,
        # R7 — per-session move usage for variation pressure
        moves_used=panel.moves_used,
        # M1 (Option C) — bring forward the consecutive-Mode-1a counter so
        # the cache-lookup gate can enforce the max-2 hard cap.
        consecutive_mode1a=panel.consecutive_mode1a,
        # 2026-05-05 — Late-phase low-score detector input. Forward the
        # previous customer turn's persuasion + commitment so the chain
        # stage can detect "stuck in close_attempt with low engagement"
        # and pull recovery patterns from historical won conversations.
        prev_persuasion=(panel.persuasion_history[-1] if panel.persuasion_history else None),
        prev_commitment=(panel.commitment_history[-1] if panel.commitment_history else None),
    )
    log.info("[%s] chain_runner: executing %d stages (chain_type sequence)",
             panel.side, len(stages))
    result_ctx = await run_chain(stages, chain_ctx)

    # M1 — write the consecutive-Mode-1a counter back to PanelState so the
    # next turn's cache lookup sees the correct value.
    if hasattr(result_ctx, "consecutive_mode1a"):
        panel.consecutive_mode1a = result_ctx.consecutive_mode1a

    # Extract agent text from build_answer stage's output. In production this
    # is the FINAL agent text after dedup/validation/translate; in Phase A
    # POC the simplified prompt-stage executor returns the LLM raw text.
    build_answer = result_ctx.previous_responses.get("prompt_build_answer")
    if isinstance(build_answer, str) and build_answer.strip():
        agent_text = build_answer.strip()
    elif isinstance(build_answer, dict):
        agent_text = (build_answer.get("_raw") or "").strip()
    else:
        agent_text = ""

    if not agent_text:
        log.warning("[%s] chain_runner: build_answer produced no text", panel.side)
        return None, {}, None

    # 2026-05-05 — Length enforcement for R-side. The chain runner uses
    # chain_executor.dispatch_llm() which bypasses actor.generate()'s
    # post-gen length checks. Apply the same `_enforce_length` truncation
    # here so R-side messages also obey the 35-word cap. (Regen path stays
    # in actor.generate; here we only truncate as last resort.)
    try:
        from actor import _enforce_length as _enforce_length_fn
        truncated_text, was_truncated = _enforce_length_fn(agent_text)
        if was_truncated:
            import re as _re
            log.info("[%s] chain.build_answer truncated for length: %d → %d words",
                     panel.side,
                     len(_re.split(r"\s+", agent_text.strip())),
                     len(_re.split(r"\s+", truncated_text.strip())))
            agent_text = truncated_text
    except Exception as e:
        log.warning("[%s] chain.build_answer length-enforcement failed: %s",
                    panel.side, e)

    # Build a directive_meta-equivalent dict from chain results (so the
    # rest of the replayer's bookkeeping can pretend nothing changed)
    sa_combined = result_ctx.previous_responses.get("prompt_signal_analysis_combined") or {}
    gate_result = result_ctx.previous_responses.get("prompt_anti_staircase_gate") or {}
    retreat_result = result_ctx.previous_responses.get("prompt_retreat_passthrough_gate") or {}
    anchor_result = result_ctx.previous_responses.get("prompt_anchor_load") or {}

    # cg_evidence: chain_runner mode bypasses supervisor_full's mode-routing,
    # so we propagate the cg_evidence the splice stages produced themselves.
    # signal_analysis_combined invokes mode1b_directive internally, which
    # makes real /query/data + /query/auto CG calls and emits cg_evidence
    # keyed by flags/counts (bool / int), not list-of-strings. We normalize to
    # truthy values per scenario-mapping key so _evidence_to_scenarios fires.
    raw_cg = dict(sa_combined.get("cg_evidence") or {})
    cg_evidence: dict = {}
    for k, v in raw_cg.items():
        # Map bool/int/list/dict → simple truthy marker the scenario mapper
        # already accepts (any truthy value fires the corresponding S-id).
        if v:
            cg_evidence[k] = v if isinstance(v, list) else [v]
    # T-86 anchor_load → 'rules' scenario (S2) fires when anchor pack loads
    if anchor_result and anchor_result.get("anchors_loaded"):
        existing = cg_evidence.get("rules")
        marker = f"anchor_pack:{anchor_result.get('tenant', 'unknown')}"
        if isinstance(existing, list):
            existing.append(marker)
        else:
            cg_evidence["rules"] = [marker]
    # signal_analysis_combined classified a signal → patterns scenario (S3)
    if sa_combined.get("primary_signal") and not cg_evidence.get("patterns"):
        cg_evidence["patterns"] = [sa_combined.get("primary_signal")]

    # Per-turn decision_trace_emit result — surface so replayer can mark
    # S4 (Decision Audit Trail) demonstrated.
    dte_result = result_ctx.previous_responses.get("prompt_decision_trace_emit") or {}
    decision_trace_emitted_this_turn = bool(dte_result.get("emitted"))

    # 2026-05-11 — Capture escalation router's verdict; merged into
    # directive_meta below after that dict is constructed.
    esc_result = result_ctx.previous_responses.get("prompt_escalation_router") or {}

    # R5 invariants gate result — surface for UI / failure-mode taxonomy.
    invariants_result = result_ctx.previous_responses.get("prompt_invariants_gate") or {}

    # Detect cache-hit metadata (set by _try_cache_lookup when Mode 1a fires)
    cache_hit = bool(sa_combined.get("_cache_hit"))
    cache_recovery = bool(sa_combined.get("_recovery_mode"))
    cache_score = sa_combined.get("_cache_score")
    cache_score_band = sa_combined.get("_score_band")

    # Pull tactical/cialdini/delta from the cached directive's blob (when present)
    cached_dir = sa_combined.get("directive") or {}
    tactical_brief = []
    cialdini_brief = []
    historical_delta = None
    if cache_hit:
        # The cached directive was reconstructed from a directive_v1 blob —
        # surface a few key fields so chat-level chips can show "what kind
        # of move from history" without the user opening /logs.
        tactical_dict = cached_dir.get("_cache_tactical") or {}
        if isinstance(tactical_dict, dict):
            tactical_brief = [k for k, v in tactical_dict.items()
                              if v and k != "word_count"][:3]
        cialdini_brief = (cached_dir.get("_cache_cialdini") or [])[:3]
        historical_delta = cached_dir.get("_cache_delta_after_move")

    directive_meta = {
        # Mode label is "1b" or "1a" depending on whether the cache hit fired.
        # Drives both the architecture banner chips AND the chat-level badges.
        "mode": "1a" if cache_hit else "1b",
        "tier_label": (
            f"Mode 1a · cached (score {cache_score:.2f})"
            if cache_hit and cache_score
            else "Chain Runner (Phase A.3) · Mode 1b"
        ),
        # Tier 1+2 visibility — chat-level chips read these
        "cache_hit": cache_hit,
        "cache_recovery_mode": cache_recovery,
        "cache_score_band": cache_score_band,
        "cache_score": cache_score,
        "cache_tactical_brief": tactical_brief,
        "cache_cialdini_brief": cialdini_brief,
        "historical_delta_after_move": historical_delta,
        "chain_stages_run": len(stages),
        "signal_analysis": sa_combined.get("signal_analysis"),
        "primary_signal": sa_combined.get("primary_signal"),
        "primary_strategy": sa_combined.get("primary_strategy"),
        "adherence_retry_fired": sa_combined.get("adherence_retry_fired"),
        "signal_adherence_retry_fired": sa_combined.get("signal_adherence_retry_fired"),
        # R4 (Q12) — propagate consistency-retry meta to UI
        "consistency_retry": sa_combined.get("consistency_retry"),
        # R6 — propagate move-validity retry meta to UI
        "move_validity_retry": sa_combined.get("move_validity_retry"),
        "anti_staircase_verdict": gate_result.get("verdict"),
        "retreat_active": retreat_result.get("retreat_active"),
        "decision_trace_emitted_this_turn": decision_trace_emitted_this_turn,
        # CG-data audit fix (2026-05-04) — propagate Mode 1b CG counts so UI
        # architecture banner shows real numbers, not 0/0/0.
        "cg_entities": sa_combined.get("cg_n_entities", 0),
        "cg_relations": sa_combined.get("cg_n_relations", 0),
        "cg_chunks": sa_combined.get("cg_n_chunks", 0),
        # R5 — invariants gate (max-discount, disparagement, fabricated-stat, regulatory)
        "invariants_verdict": invariants_result.get("verdict") or invariants_result.get("invariants_violated"),
        "invariants_violation_types": invariants_result.get("violation_types") or [],
        "cg_evidence": cg_evidence,
        # Multi-turn arc awareness (§9.2 #7) — surface phase to the UI per turn.
        "conversation_phase": sa_combined.get("conversation_phase"),
        # 7-bonus — surface a brief anchor snapshot so the UI can show the
        # economic reference frame the supervisor is reasoning over.
        "anchors_brief": _brief_anchors(opp_meta.get("anchors")),
    }

    # Synthesize a directive dict so _summarize_directive produces a label
    # for the right_msg event; UI's chat-side architecture/strategy chip
    # depends on directive being non-None.
    rules_to_enforce = []
    if gate_result.get("verdict") == "regenerate":
        rules_to_enforce.append("anti_staircase_active")
    elif gate_result.get("verdict") == "clean":
        rules_to_enforce.append("anti_staircase_passed")
    if retreat_result.get("retreat_active"):
        rules_to_enforce.append("retreat_passthrough")
    # R5 — surface invariant violations as enforced rules so the UI sees them
    for vt in invariants_result.get("violation_types") or []:
        rules_to_enforce.append(f"invariant:{vt}")
    # Surface Tier 2 concrete_move from Mode 1b's directive (lives in
    # sa_combined['directive']['strategy']['concrete_move']) onto the
    # synthesized chain_directive so _summarize_directive can render it
    # in the per-turn UI chip.
    sa_full_directive = sa_combined.get("directive") if isinstance(sa_combined, dict) else None
    sa_strat = (sa_full_directive or {}).get("strategy") or {}
    concrete_move_from_sa = sa_strat.get("concrete_move")

    # Pull audit + confidence + must_not_say from the FULL Mode 1b directive
    # so _summarize_directive can surface them to the UI (rationale tooltip,
    # confidence dot/chart, must-not-say bullets). Chain_directive used to
    # be a thin synthetic — these fields were getting dropped.
    sa_audit = (sa_full_directive or {}).get("audit") or {}
    sa_confidence = (sa_full_directive or {}).get("confidence") or {}
    sa_knowledge = (sa_full_directive or {}).get("knowledge") or {}
    sa_must_not_say = sa_knowledge.get("must_not_say") or []
    sa_facts = sa_knowledge.get("facts_to_anchor") or anchor_result.get("anchor_facts") or []
    sa_customer_state = (sa_full_directive or {}).get("customer_state") or {}

    chain_directive = {
        "primary_strategy": sa_combined.get("primary_strategy"),
        "tone": sa_combined.get("tone"),
        "rules_to_enforce": rules_to_enforce,
        "signal_analysis": sa_combined.get("signal_analysis"),
        "knowledge": {
            "facts_to_anchor": sa_facts,
            "must_not_say": sa_must_not_say,
        },
        "must_not_say": sa_must_not_say,
        "audit": sa_audit,
        "confidence": sa_confidence,
        "customer_state": sa_customer_state,
        # Mirrors Mode 1b's directive.strategy.concrete_move so the
        # per-turn UI chip can show "value_stack_with_anchors" etc.
        "strategy": {
            "primary": sa_combined.get("primary_strategy"),
            "tone": sa_combined.get("tone"),
            "concrete_move": concrete_move_from_sa,
        },
    }
    # 2026-05-11 — Merge escalation router's verdict into directive_meta now
    # that the dict exists. _live_turn checks _escalation_required to short-
    # circuit the customer-simulator step and label panel-end correctly.
    if isinstance(esc_result, dict) and esc_result.get("escalate"):
        directive_meta["_escalation_required"] = True
        directive_meta["_escalation_reason"] = esc_result.get("reason", "")
        directive_meta["_escalation_handoff_channel"] = esc_result.get("handoff_channel", "")
    return agent_text, directive_meta, chain_directive


async def _live_turn(panel: PanelState, opp_meta: dict, business_rules: str,
                     classifier, simulator: CustomerSimulator, send: SEND_FN,
                     get_speed: SPEED_FN, agent_turn_index: int,
                     historical_messages: list[dict],
                     cached_features: dict | None = None,
                     ctx: SessionCtx | None = None,
                     http_client: httpx.AsyncClient | None = None) -> bool:
    """One full live turn for a panel: agent reply → customer reply → scoring.
    Returns True if panel should continue, False if it ended."""
    if panel.won or panel.lost:
        return False
    if ctx and ctx.stopped:
        panel.lost = True
        panel.end_reason = "stopped_by_user"
        return False

    # T-85 Phase A.3 — alternate path: when POC_USE_PROD_CHAIN=1, route the
    # agent generation through the production-equivalent multi-stage prompt
    # chain (with supervisor stages spliced in on the RIGHT panel only).
    # Bypasses the classic supervisor+actor flow; downstream (refusal
    # detection, customer reply, scoring) continues normally.
    # POC_USE_PROD_CHAIN=1 enables chain_runner ON THE RIGHT (SUPERVISED) PANEL
    # ONLY. LEFT panel always uses the fast single-call actor.generate
    # baseline so the side-by-side comparison stays usable: LEFT = ~5s/turn
    # (Original agent), RIGHT = ~30-60s/turn (production-equivalent chain +
    # supervisor splice). Running both through chain_runner used to balloon
    # per-turn latency to 150-200s, making live demos unusable.
    USE_PROD_CHAIN = (os.environ.get("POC_USE_PROD_CHAIN", "0") == "1"
                       and panel.side == "right")
    chain_agent_text: str | None = None
    chain_directive_meta: dict = {}
    chain_directive: dict | None = None

    # Pluggable-engine path. Any registered engine whose live_mode is "produce"
    # (the Planner and every third-party plugin) is driven generically here via
    # its Engine.produce() contract. The engine returns final customer-facing
    # text + a telemetry meta dict; the shared actor/customer-sim substrate
    # downstream is untouched (single-variable invariant preserved). The legacy
    # "strategist" (supervised) and "baseline" (vanilla) ids are live_mode
    # "native" and fall through to the classic flow below.
    # NOTE: ENGINE_MODE supersedes the old hardcoded PLANNER_MODE branch, which
    # imported a `planner_produce` bridge that never shipped (it silently fell
    # back). Routing through the registry both fixes live Planner and opens the
    # path to arbitrary engines.
    # Per-panel engine selection (any-vs-any A/B). Each panel's engine id rides
    # on the shared opp_meta as `_engine_<side>`; the legacy single `_engine`
    # key (R-side) is honored as a fallback. Defaults preserve the classic
    # pairing — LEFT = baseline (control), RIGHT = strategist (supervised) — so
    # an unconfigured session is byte-identical to the pre-registry flow.
    panel_engine = (opp_meta or {}).get(f"_engine_{panel.side}")
    if not panel_engine:
        panel_engine = (((opp_meta or {}).get("_engine") if panel.side == "right" else None)
                        or ("strategist" if panel.side == "right" else "baseline"))
    # The supervised ("strategist") native flow follows the ENGINE, not the
    # side. Re-keying the old `panel.side == "right"` supervisor gates on this
    # boolean keeps the default pairing unchanged while letting the supervisor
    # run on either panel.
    panel_is_supervised = (panel_engine == "strategist")

    ENGINE_MODE = False
    if panel_engine:
        try:
            from registry import get as _get_engine_spec
            _spec = _get_engine_spec(panel_engine)
        except Exception:
            _spec = None
        if _spec is not None and _spec.live_mode == "produce":
            ENGINE_MODE = True
            try:
                eng = getattr(panel, "_engine_instance", None)
                if eng is None:
                    eng = _spec.create(
                        **((opp_meta or {}).get(f"_engine_params_{panel.side}") or {}))
                    try:
                        panel._engine_instance = eng  # reuse across turns
                    except Exception:
                        pass
                chain_agent_text, chain_directive_meta = await eng.produce(
                    opp_meta, panel.dialog, business_rules or "")
                chain_directive = chain_directive_meta or {}
            except Exception as e:
                log.warning("[%s] engine %r failed: %s", panel.side, panel_engine, e)
                chain_agent_text = None

    if USE_PROD_CHAIN and not ENGINE_MODE:
        # Stash cluster_id in opp_meta so chain stages (cache lookup, decision-
        # trace emit) can build the segment key. Legacy branch sets this
        # after Manager classifies; the chain path skips that branch, so we
        # need to pre-populate. cluster_id lives in data/scenarios.json keyed
        # by opp_id; this lookup is cheap (file is read once via _scenarios_index).
        if "_cluster_id" not in opp_meta or opp_meta.get("_cluster_id") in (None, "?"):
            cid = opp_meta.get("cluster_id")
            if cid is None and isinstance(opp_meta.get("scenario"), dict):
                cid = opp_meta["scenario"].get("cluster_id")
            if cid is None:
                # Last resort: look up from scenarios.json by opp_id
                try:
                    cid = _scenarios_cluster_lookup(opp_meta.get("id") or opp_meta.get("opp_id"))
                except Exception:
                    cid = None
            if cid is not None:
                opp_meta["_cluster_id"] = cid
        try:
            chain_agent_text, chain_directive_meta, chain_directive = await _run_via_chain_runner(
                panel, opp_meta, business_rules, send,
            )
        except Exception as e:
            log.warning("[%s] chain_runner failed: %s; falling back to classic flow",
                          panel.side, e)
            chain_agent_text = None

    # 1. Compute directive (right panel only) — with tier router for 'auto' mode
    # T-85 Phase A.3: skip classic supervisor+actor flow if chain_runner produced text
    directive = None
    directive_meta = {}
    if (USE_PROD_CHAIN or ENGINE_MODE) and chain_agent_text is not None:
        # Chain runner succeeded; use synthesized directive + meta so the UI
        # still gets architecture/strategy labels and integration-scenario
        # updates per turn.
        directive = chain_directive
        directive_meta = chain_directive_meta
    elif panel_is_supervised:
        try:
            # Decide which tier this turn uses
            classifier_conf = 0.0
            has_playbook = False
            cluster_id_for_routing = None
            if classifier is not None:
                # Cheap classify just to drive tier decision (regardless of POC_MODE)
                features = cached_features or await asyncio.to_thread(
                    _compute_features_for_classifier, opp_meta, historical_messages)
                cluster_id_for_routing, classifier_conf, _ = classifier.classify(features)
                motivator = opp_meta.get("primary_motivator")
                decision_logic = opp_meta.get("decision_logic")
                pb, _ls = classifier.lookup_playbook(cluster_id_for_routing, motivator, decision_logic)
                has_playbook = pb is not None
                # ClusterPlan lazy-load: first time we know the cluster, try to
                # load the matching session-level plan. None if no plan exists.
                if panel.plan_state is None:
                    plan_dict = load_plan(opp_meta.get("company"),
                                            cluster_id_for_routing,
                                            motivator, decision_logic)
                    if plan_dict is not None:
                        panel.plan_state = PlanState(plan=plan_dict)
                        log.info("[%s] loaded ClusterPlan %s — starting at phase %d",
                                  panel.side, plan_dict.get("cluster_plan_id"),
                                  panel.plan_state.current_phase_id)
                        # Surface plan to UI as a one-time event
                        await send({
                            "event": "plan_loaded",
                            "side": panel.side,
                            "cluster_plan_id": plan_dict.get("cluster_plan_id"),
                            "n_phases": len(plan_dict.get("path") or []),
                            "phases": [{"id": ph["phase_id"], "name": ph["name"],
                                          "max_turns": ph.get("max_turns")}
                                         for ph in (plan_dict.get("path") or [])],
                            "goal": plan_dict.get("goal", {}).get("primary"),
                            "time_budget_turns": plan_dict.get("time_budget_turns"),
                        })

                # Win-mode plan lazy-load (T-76 Day 4) — cell-keyed, gated by
                # POC_WIN_PLAN_ENABLED. Loads alongside cluster_plan but stays
                # dormant until engagement gate fires.
                if (POC_WIN_PLAN_ENABLED and panel_is_supervised
                        and panel.win_plan_state is None):
                    win_plan_dict = wp_mod.load_win_plan(
                        opp_meta.get("company"),
                        opp_meta.get("opp_type") or opp_meta.get("type"),
                        motivator, decision_logic)
                    if win_plan_dict is not None:
                        panel.win_plan_state = wp_mod.WinPlanState(plan=win_plan_dict)
                        log.info("[%s] loaded WinPlan %s (dormant — awaiting engagement gate)",
                                  panel.side, win_plan_dict.get("win_plan_id"))
                        await send({
                            "event": "win_plan_loaded",
                            "side": panel.side,
                            "win_plan_id": win_plan_dict.get("win_plan_id"),
                            "n_phases": len(win_plan_dict.get("path") or []),
                            "branches": [
                                "branch_A_bargain_and_stretch",
                                "branch_B_counter_once_then_close",
                                "branch_C_long_tail_followup",
                            ],
                            "engagement_gate": win_plan_dict.get("engagement_gate", {}).get("load_when_all_of"),
                        })

            if POC_MODE == "1a":
                tier = "mode_1a"
            elif POC_MODE == "1b":
                tier = "mode_1b"
            elif POC_MODE == "2":
                tier = "mode_2"
            else:  # auto
                tier = select_tier(panel, classifier_conf, has_playbook)

            log.info("[%s] tier=%s (classifier_conf=%.2f playbook=%s commits=%s)",
                      panel.side, tier, classifier_conf, has_playbook,
                      panel.commitment_history)

            # Win-mode engagement gate — switch supervisor to win-plan when
            # trajectory looks like a plausible win path. Gate fires at turn
            # >= 4 with persuasion_avg >= 0.4 OR commit_max >= 2.
            pre_rendered_plan_section = None
            adherence_preferred_actions: list[str] | None = None
            adherence_phase_label: str | None = None
            if (POC_WIN_PLAN_ENABLED and panel_is_supervised
                    and panel.win_plan_state is not None):
                turn_idx_so_far = len(panel.commitment_history)
                panel_metrics = {
                    "persuasion_scores": panel.live_persuasion_history,
                    "commit_levels": panel.commitment_history,
                    "aborted": (panel.plan_state is not None and
                                  getattr(panel.plan_state, 'aborted', False)),
                }
                gate_ok = wp_mod.engagement_gate_met(
                    panel.win_plan_state.plan, panel_metrics, turn_idx_so_far)
                if gate_ok and panel.plan_mode_active != "win":
                    panel.plan_mode_active = "win"
                    log.info("[%s] WIN-MODE plan activated at turn %d (persuasion_avg + commit signal cleared gate)",
                              panel.side, turn_idx_so_far)
                    await send({
                        "event": "win_plan_activated",
                        "side": panel.side,
                        "turn": turn_idx_so_far,
                        "persuasion_avg": (
                            round(sum(panel.live_persuasion_history) /
                                  len(panel.live_persuasion_history), 3)
                            if panel.live_persuasion_history else None),
                        "commit_max": (max(panel.commitment_history)
                                          if panel.commitment_history else None),
                    })
                elif not gate_ok and panel.plan_mode_active == "win":
                    panel.plan_mode_active = "loss"
                    log.info("[%s] win-mode gate fell — reverting to loss-mode plan",
                              panel.side)
                if panel.plan_mode_active == "win":
                    pre_rendered_plan_section = wp_mod.render_win_plan_section(
                        panel.win_plan_state.plan, panel.win_plan_state)
                    cur_phase_dict = panel.win_plan_state.current_phase()
                    if cur_phase_dict:
                        adherence_preferred_actions = (
                            cur_phase_dict.get("agent_actions_preferred") or None)
                        adherence_phase_label = (
                            cur_phase_dict.get("name")
                            or str(panel.win_plan_state.current_phase_id))

            if tier == "mode_1a" and classifier is not None:
                state = _state_for_classifier(opp_meta, panel.dialog)
                turn_idx = max(0, len(state["messages"]) - 1)
                r1a = await mode1a_directive_v1(
                    opp_meta, state, turn_idx, features, classifier, HAIKU_CLIENT,
                )
                directive = r1a.get("directive")
                directive_meta = {"mode": "1a", "tier_label": "Mode 1a · playbook"}
                if directive is None:
                    log.info("[%s] mode1a fallback to 1b: %s",
                              panel.side, r1a.get("fallback_reason"))
                    tier = "mode_1b"  # downgrade
            if tier == "mode_2":
                r2 = await mode2_directive(opp_meta, panel.dialog, business_rules,
                                            http_client=http_client,
                                            strategies_used=panel.strategies_used,
                                            plan_state=panel.plan_state,
                                            pre_rendered_plan_section=pre_rendered_plan_section,
                                            preferred_actions=adherence_preferred_actions,
                                            current_phase_label=adherence_phase_label)
                directive = r2.get("directive")
                directive_meta = {
                    "mode": "2",
                    "tier_label": "Mode 2 · CGR3 multi-hop",
                    "cgr3_chars": r2.get("cgr3_response_chars", 0),
                    "cgr3_latency_ms": r2.get("cgr3_latency_ms", 0),
                    "sup_latency_ms": r2.get("latency_ms", 0),
                    "cg_evidence": r2.get("cg_evidence") or {},
                    "adherence_retry": r2.get("adherence_retry") or None,
                    "signal_adherence_retry": r2.get("signal_adherence_retry") or None,
                }
                if directive is None:
                    log.info("[%s] mode2 fallback to 1b: %s", panel.side,
                              r2.get("fallback_reason"))
                    tier = "mode_1b"
            if tier == "mode_1b" and directive is None:
                # Default tier OR downgraded from mode_1a / mode_2
                r1b = await mode1b_directive(opp_meta, panel.dialog, business_rules,
                                              http_client=http_client,
                                              strategies_used=panel.strategies_used,
                                              plan_state=panel.plan_state,
                                              pre_rendered_plan_section=pre_rendered_plan_section,
                                              preferred_actions=adherence_preferred_actions,
                                              current_phase_label=adherence_phase_label)
                directive = r1b.get("directive")
                directive_meta = {
                    "mode": "1b",
                    "tier_label": "Mode 1b · live + Context Graph",
                    "cg_entities": r1b.get("cg_n_entities", 0),
                    "cg_relations": r1b.get("cg_n_relations", 0),
                    "cg_chunks": r1b.get("cg_n_chunks", 0),
                    "cg_latency_ms": r1b.get("cg_latency_ms", 0),
                    "sup_latency_ms": r1b.get("latency_ms", 0),
                    "cg_evidence": r1b.get("cg_evidence") or {},
                    "adherence_retry": r1b.get("adherence_retry") or None,
                    "signal_adherence_retry": r1b.get("signal_adherence_retry") or None,
                }

            # T-79 phase stamping — when win-plan is engaged, stamp the active
            # phase / branch / preferred-actions onto directive_meta so the
            # right_msg event (and the session log) record which plan phase
            # this turn's strategy was chosen UNDER. Enables retrospective
            # adherence audits and per-phase strategy distribution analysis.
            if (POC_WIN_PLAN_ENABLED and panel_is_supervised
                    and panel.win_plan_state is not None
                    and panel.plan_mode_active == "win"):
                directive_meta["plan_phase_id"] = (
                    panel.win_plan_state.current_phase_id)
                directive_meta["plan_phase_label"] = adherence_phase_label
                directive_meta["plan_preferred_actions"] = (
                    adherence_preferred_actions)
                directive_meta["plan_active_branch"] = (
                    panel.win_plan_state.active_branch)
                directive_meta["plan_mode_active"] = panel.plan_mode_active

            # Track tier usage for session aggregate
            panel.last_directive = directive
            if not hasattr(panel, "tier_counts"):
                panel.tier_counts = {}
            mode_actually_used = directive_meta.get("mode", "?")
            panel.tier_counts[mode_actually_used] = (
                panel.tier_counts.get(mode_actually_used, 0) + 1)
            # Maintain consecutive-Mode-1a counter for next-turn auto-escalation.
            # Mode 1a returns a static playbook → if used twice in a row the
            # third call is force-routed to Mode 1b by select_tier(). Reset on
            # any other tier so we re-enable Mode 1a use after a break.
            if mode_actually_used == "1a":
                panel.consecutive_mode1a += 1
                if panel.consecutive_mode1a >= 2:
                    log.info("[%s] Mode 1a counter at %d — next call will auto-escalate to Mode 1b",
                              panel.side, panel.consecutive_mode1a)
            else:
                panel.consecutive_mode1a = 0
            # Record the chosen primary_strategy in session memory so the NEXT
            # turn's supervisor prompt knows what's already been used. This is
            # the cheapest possible session-memory implementation; full
            # ClusterPlan is the proper fix (see research-notes/2026-04-30-…).
            if directive:
                strat = (directive.get("strategy") or {}).get("primary") \
                    or directive.get("primary_strategy")
                if strat:
                    panel.strategies_used.append(strat)
                # R7 — track concrete-move usage for cooldown / variation pressure.
                cm = (directive.get("strategy") or {}).get("concrete_move")
                move_name = cm.get("name") if isinstance(cm, dict) else None
                if move_name:
                    panel.moves_used[move_name] = panel.moves_used.get(move_name, 0) + 1
            # ClusterPlan adherence: record whether supervisor stayed within the
            # current phase's preferred actions. Increments probe_attempts when
            # in-plan; bumps off_plan_count when deviating. Used downstream by
            # the graceful-close guard so supervisor can't bail before probing.
            if panel.plan_state is not None and directive:
                record_strategy_adherence(panel.plan_state, directive)
            # Stash cluster id for later decision-trace emission
            if cluster_id_for_routing is not None:
                opp_meta["_cluster_id"] = cluster_id_for_routing
        except Exception as e:
            log.warning("[%s] supervisor (mode=%s) failed: %s",
                        panel.side, POC_MODE, e)
            directive = None

    # T-80 retreat passthrough: when supervisor's signal_analysis indicates
    # the customer wants space (pace_request, disengagement), suppress the
    # directive entirely so the agent's natural soft-retention behavior
    # takes over. Empirical evidence (2026-05-02 d7bdba96 session, t11-t17):
    # supervisor's engagement strategies cause trust collapse when customer
    # has signaled "let me think". Original agent's system prompt rule
    # ("If customer signals 'I want to think' → soft retention check, not
    # pressure") handles retreat correctly only when no directive overrides it.
    # The supervisor's strategy enum has no "back off" primitive, so we get
    # the equivalent behavior by passing directive=None on retreat signals.
    RETREAT_SIGNALS = {"pace_request", "disengagement"}
    if directive and panel_is_supervised:
        sa = directive.get("signal_analysis") or {}
        if sa.get("primary_signal") in RETREAT_SIGNALS:
            log.info("[%s] retreat passthrough — primary_signal=%s, suppressing directive",
                      panel.side, sa.get("primary_signal"))
            await send({
                "event": "retreat_passthrough",
                "side": panel.side,
                "primary_signal": sa.get("primary_signal"),
                "suppressed_strategy": _directive_strategy(directive),
                "turn": panel.seq_counter + 1,
            })
            directive = None  # natural-Agent behavior takes over

    # 2. Agent reply (forced to seed language if simulator detected one)
    forced_lang = simulator.seed_language if simulator.seed_language in ("en", "he") else None

    # CR-PS Phase 4 — cohort-conditioned precedent retrieval. Fetched once per
    # turn for the right panel; consumed by either the chain path (via
    # opp_meta["_cohort_precedent_block"], read in chain_executor.build_context_blocks)
    # or the legacy regenerate-loop path (via cohort_precedent_block kwarg on
    # actor_generate). Same block, two consumers — opp_meta is the carrier.
    cohort_block = ""
    opp_meta.pop("_cohort_precedent_block", None)
    if panel_is_supervised and http_client is not None:
        try:
            from supervisor_full import (fetch_cohort_precedents,
                                         render_cohort_precedent_block,
                                         POC_COHORT_PRECEDENTS_ENABLED)
            if POC_COHORT_PRECEDENTS_ENABLED:
                commit_max = (max(panel.commitment_history)
                              if panel.commitment_history else None)
                primary_signal = None
                if directive:
                    sa = directive.get("signal_analysis") or {}
                    primary_signal = sa.get("primary_signal")
                cp_resp = await fetch_cohort_precedents(
                    http_client=http_client, opp_meta=opp_meta,
                    commit_max=commit_max, primary_signal=primary_signal,
                    top_k=5,
                )
                cohort_block = render_cohort_precedent_block(cp_resp)
                if cohort_block:
                    opp_meta["_cohort_precedent_block"] = cohort_block
                # Trace fire-or-not (per substrate doc §7 fired-vs-credited
                # discipline). Logged regardless of whether block is non-empty.
                await send({
                    "event": "cohort_precedents_fetched",
                    "side": panel.side,
                    "turn": panel.seq_counter + 1,
                    "tier": cp_resp.get("tier"),
                    "sqlite_count": cp_resp.get("sqlite_count"),
                    "cg_count": cp_resp.get("cg_count"),
                    "fallback_fired": cp_resp.get("fallback_fired"),
                    "filters": cp_resp.get("filters"),
                    "block_chars": len(cohort_block),
                })
        except Exception as e:
            log.warning("[%s] cohort precedent fetch failed: %s", panel.side, e)

    # T-85 Phase A.3: if chain_runner produced text, use it directly and skip
    # the classic regenerate-loop (the chain has its own staircase semantics).
    if (USE_PROD_CHAIN or ENGINE_MODE) and chain_agent_text is not None:
        agent_text = chain_agent_text
        _meta = {}
        directive_meta = chain_directive_meta
    else:
        # T-83 anti-staircase regenerate loop. Both panels run the gate so we
        # mechanically prevent staircase concessions on EITHER side (independent
        # of whether the supervisor is active). The agent's prior concession
        # history lives in panel.agent_concessions.
        tenant = (opp_meta.get("company") or "").strip()
        MAX_STAIRCASE_RETRIES = 2
        first_draft = None
        staircase_meta_history = []
        agent_text = None
        _meta = {}
        try:
            for attempt in range(MAX_STAIRCASE_RETRIES + 1):
                # Build correction suffix from prior staircase meta on retries
                sys_suffix = None
                if attempt > 0 and staircase_meta_history:
                    sys_suffix = build_correction_prompt(staircase_meta_history[-1])
                candidate, candidate_meta = await actor_generate(
                    opp_meta, panel.dialog, business_rules, directive=directive,
                    forced_language=forced_lang,
                    system_suffix=sys_suffix,
                    cohort_precedent_block=cohort_block or None,
                )
                if attempt == 0:
                    first_draft = (candidate, candidate_meta)
                # Run the gate
                is_staircase, sc_meta = check_staircase(
                    panel_concessions=panel.agent_concessions,
                    new_text=candidate,
                    tenant=tenant,
                )
                if not is_staircase:
                    agent_text = candidate
                    _meta = candidate_meta
                    # Accept new concessions into panel state (only on success)
                    for c in (sc_meta.get("new_concessions") or []):
                        panel.agent_concessions.append(
                            {**c, "turn": panel.seq_counter + 1})
                    if attempt > 0:
                        log.info("[%s] staircase gate: regenerate succeeded on attempt %d",
                                  panel.side, attempt + 1)
                        await send({
                            "event": "compliance_gate_fired",
                            "side": panel.side,
                            "turn": panel.seq_counter + 1,
                            "gate": "anti_staircase",
                            "verdict": "blocked_then_regenerated",
                            "retry_attempt": attempt,
                            "reason": staircase_meta_history[-1].get("reason"),
                            "outcome": "regenerated_successfully",
                        })
                    break
                # Staircase detected → log + retry (or fall back on max attempts)
                staircase_meta_history.append(sc_meta)
                log.info("[%s] staircase gate fired (attempt %d): %s",
                          panel.side, attempt + 1, sc_meta.get("reason"))
                if attempt >= MAX_STAIRCASE_RETRIES:
                    # Defensive fallback: accept the FIRST draft (not the worst-of-N)
                    log.warning("[%s] staircase gate exhausted retries — accepting first draft",
                                  panel.side)
                    agent_text, _meta = first_draft
                    await send({
                        "event": "compliance_gate_fired",
                        "side": panel.side,
                        "turn": panel.seq_counter + 1,
                        "gate": "anti_staircase",
                        "verdict": "blocked_fallback_to_original",
                        "retry_attempt": attempt + 1,
                        "reason": sc_meta.get("reason"),
                        "outcome": "fallback_to_original",
                    })
                    # Even on fallback, record any new concessions from the
                    # accepted (first) draft so panel state stays consistent
                    _, first_sc_meta = check_staircase(
                        panel_concessions=panel.agent_concessions,
                        new_text=agent_text, tenant=tenant)
                    for c in (first_sc_meta.get("new_concessions") or []):
                        panel.agent_concessions.append(
                            {**c, "turn": panel.seq_counter + 1})
                    break
        except Exception as e:
            log.warning("[%s] actor failed: %s", panel.side, e)
            agent_text = "(actor error)"

    # Agent self-refusal detection (boundary respect, empty msg, etc.)
    if detect_agent_refusal(agent_text):
        panel.lost = True
        panel.end_reason = "agent_refused_to_continue"
        await send({
            "event": f"{panel.side}_msg",
            "role": "system",
            "text": f"[Agent refused to continue: '{agent_text[:120]}']",
            "turn": panel.seq_counter + 1,
        })
        return False

    # Agent graceful-close detection — when agent says "all the best / take care
    # / good luck" AFTER customer has shown low commitment, end the panel
    # instead of looping into pleasantries.
    #
    # Bug fix (2026-04-30): the previous condition `recent_low or not history`
    # mis-fired on the FIRST live turn — supervisor would emit a soft sign-off
    # before any commitment signal existed, ending the panel prematurely
    # (caused the dfb34792 c3 regression in the Ecommerce batch). Now: only fire
    # graceful-close when we have actual evidence the customer is disengaged.
    #
    # Bug fix (2026-05-01): supervisor with plan loaded was choosing `empathy`
    # at probe phases and warm-closing before doing ANY probing (dfb34792
    # session 5852c1cc). Plan now has a probe-quota guard — close blocked if
    # current phase's preferred actions haven't been attempted ≥1 time.
    if detect_agent_graceful_close(agent_text):
        # Plan-aware graceful-close guard: refuse close if the current phase
        # hasn't been probed yet
        plan_close_allowed, plan_block_reason = can_graceful_close(
            getattr(panel, "plan_state", None))
        if not plan_close_allowed:
            log.info("[%s] graceful-close BLOCKED by plan-quota guard: %s",
                      panel.side, plan_block_reason)
            await send({
                "event": f"{panel.side}_msg",
                "role": "system",
                "text": f"[Plan blocks early close: {plan_block_reason}]",
                "turn": panel.seq_counter + 1,
            })
            # Fall through — agent's message still emitted but panel does NOT end
        else:
            # End the panel on agent_graceful_close when EITHER:
            #   1. Customer is unconverted (commitment_history[-1] <= 2) — original logic, OR
            #   2. Customer's last message is itself a farewell (the conversation is
            #      in mutual-pleasantry mode regardless of commitment level — this is
            #      the "Take care Sarah" / "Thanks have a great day" pattern that was
            #      causing 3-min pleasantry-loop tails before the fix on 2026-05-03)
            last_customer_msg = next(
                (m["text"] for m in reversed(panel.dialog)
                 if m.get("role") == "customer"),
                "",
            )
            customer_also_saying_goodbye = detect_customer_farewell(
                last_customer_msg, tenant=(opp_meta or {}).get("company"))
            commit_low = (panel.commitment_history
                          and panel.commitment_history[-1] <= 2)
            if commit_low or customer_also_saying_goodbye:
                panel.seq_counter += 1
                panel.dialog.append({"role": "agent", "text": agent_text,
                                     "sequence_number": panel.seq_counter})
                await send({
                    "event": f"{panel.side}_msg",
                    "role": "agent",
                    "text": agent_text,
                    "turn": panel.seq_counter,
                    "directive": _summarize_directive(directive) if directive else None,
                })
                end_reason_label = (
                    "agent_graceful_close" if commit_low
                    else "mutual_farewell"
                )
                await send({
                    "event": f"{panel.side}_msg",
                    "role": "system",
                    "text": f"[{end_reason_label} — ending panel]",
                    "turn": panel.seq_counter + 1,
                })
                panel.lost = True
                panel.end_reason = end_reason_label
                return False

    panel.seq_counter += 1
    agent_seq = panel.seq_counter
    panel.dialog.append({"role": "agent", "text": agent_text, "sequence_number": agent_seq})
    # Per-turn dialog text logging — debug aid so we can read the conversation
    # off the streaming server log without waiting for end-of-session JSON.
    _is_he = any("֐" <= ch <= "׿" for ch in (agent_text or ""))
    log.info("[%s] [%s] AGENT t%d (%s): %s",
             panel.session_id[:8] if hasattr(panel, "session_id") else "?",
             panel.side, agent_seq,
             "he" if _is_he else "en",
             (agent_text or "")[:300].replace("\n", " ⏎ "))
    # 2026-05-11 — Honor production escalation rule. When the chain's
    # prompt_co_pilot_escalation fires (consumed by stage_escalation_router),
    # the agent_text has already been replaced with a handoff message. Send
    # the message, log the escalation, label panel-end as escalated_to_human
    # (NOT a sales loss), and stop the panel cleanly.
    if isinstance(directive_meta, dict) and directive_meta.get("_escalation_required"):
        await send({
            "event": "escalation_fired",
            "side": panel.side,
            "turn": agent_seq,
            "reason": directive_meta.get("_escalation_reason"),
            "handoff_channel": directive_meta.get("_escalation_handoff_channel"),
        })
        log.info("[%s] [%s] ESCALATION FIRED at t%d — reason=%s channel=%s",
                 panel.session_id[:8] if hasattr(panel, "session_id") else "?",
                 panel.side, agent_seq,
                 directive_meta.get("_escalation_reason"),
                 (directive_meta.get("_escalation_handoff_channel") or "")[:40])
    await send({
        "event": f"{panel.side}_msg",
        "role": "agent",
        "text": agent_text,
        "turn": agent_seq,
        "directive": _summarize_directive(directive, directive_meta) if directive else None,
    })
    # Live tier-counter update — frontend can update arch banner per turn
    if directive_meta.get("mode"):
        await send({
            "event": "tier_used",
            "side": panel.side,
            "mode": directive_meta.get("mode"),
            "turn": agent_seq,
        })
        # Scenario coverage update — fold this turn's CG evidence into the
        # cumulative session set and emit if it grew. This populates the
        # "Integration scenarios demonstrated" panel in the UI live.
        if ctx is not None and panel_is_supervised:
            mode_id = directive_meta.get("mode")
            evidence = directive_meta.get("cg_evidence") or {}
            fired_this_turn = _evidence_to_scenarios(evidence, mode_id)
            # S4 — Decision Audit Trail fires per turn when chain_runner's
            # decision_trace_emit stage successfully POSTed to /graph/decision/emit.
            if directive_meta.get("decision_trace_emitted_this_turn"):
                fired_this_turn.add("S4")
            newly_added = fired_this_turn - ctx.scenarios
            if newly_added:
                ctx.scenarios |= fired_this_turn
                await send({
                    "event": "scenarios_update",
                    "demonstrated": sorted(ctx.scenarios),
                    "added_this_turn": sorted(newly_added),
                    "turn": agent_seq,
                })
    await asyncio.sleep(_speed_to_delay_ms(get_speed()) / 1000.0)

    # 2026-05-11 — Escalation: end the panel here (don't generate a customer
    # reply). The supervised agent has handed off to a human. Mark panel.lost
    # so the outer session loop sees it as terminated; end_reason
    # `escalated_to_human` distinguishes this from a sales loss in outcome
    # reporting (downstream accounting can filter on end_reason).
    if isinstance(directive_meta, dict) and directive_meta.get("_escalation_required"):
        panel.lost = True
        panel.won = False
        panel.end_reason = "escalated_to_human"
        await send({
            "event": f"{panel.side}_complete",
            "outcome": "escalated_to_human",
            "reason": directive_meta.get("_escalation_reason"),
            "turn": agent_seq,
        })
        return False

    # 3. Customer simulator reply
    try:
        cust_text, sim_mode = await simulator.reply(panel.dialog, agent_text, agent_turn_index)
    except Exception as e:
        log.warning("[%s] simulator failed: %s", panel.side, e)
        cust_text = "ok"
        sim_mode = "fallback"
    panel.seq_counter += 1
    cust_seq = panel.seq_counter
    panel.dialog.append({"role": "customer", "text": cust_text, "sequence_number": cust_seq})
    _is_he = any("֐" <= ch <= "׿" for ch in (cust_text or ""))
    log.info("[%s] [%s] CUSTOMER t%d (%s, sim=%s): %s",
             panel.session_id[:8] if hasattr(panel, "session_id") else "?",
             panel.side, cust_seq,
             "he" if _is_he else "en", sim_mode,
             (cust_text or "")[:300].replace("\n", " ⏎ "))
    await send({
        "event": f"{panel.side}_msg",
        "role": "customer",
        "text": cust_text,
        "turn": cust_seq,
        "sim_mode": sim_mode,
    })

    # 4. Detect decline / close signal
    if detect_decline(cust_text):
        panel.lost = True
        panel.end_reason = "customer_declined"
        await send({"event": f"{panel.side}_msg_meta", "decline_detected": True})

    # 4b. Soft farewell (customer winding down without decline/commitment).
    # End the panel to avoid pleasantry loops that dilute the persuasion curve.
    # Only fire when there's already engagement (commitment > 0 at some point);
    # a turn-1 "ok" shouldn't end the conversation.
    if not panel.lost and not panel.won and detect_customer_farewell(
            cust_text, tenant=(opp_meta or {}).get("company")):
        had_engagement = (panel.commitment_history and
                          max(panel.commitment_history) >= 2)
        if had_engagement:
            commit_now = panel.commitment_history[-1] if panel.commitment_history else 0
            # 2026-05-17 (dcfbdb58) — a polite "Thank you" AFTER a close
            # already happened is a post-close pleasantry, not a polite
            # decline. Latch won instead of mislabeling it lost.
            if _close_already_occurred(panel, opp_meta):
                panel.won = True
                panel.end_reason = "post_close_confirmation"
                await send({"event": f"{panel.side}_msg_meta",
                            "farewell_detected": True,
                            "post_close_won": True,
                            "end_reason": panel.end_reason})
                await send({
                    "event": "won",
                    "side": panel.side,
                    "turn": cust_seq,
                    "score": (panel.live_persuasion_history[-1]
                              if panel.live_persuasion_history else
                              (panel.persuasion_history[-1]
                               if panel.persuasion_history else 1.0)),
                    "commitment": commit_now,
                })
                return False
            panel.lost = True
            panel.end_reason = ("customer_polite_close" if commit_now >= 3
                                else "customer_dropped")
            await send({"event": f"{panel.side}_msg_meta",
                        "farewell_detected": True,
                        "end_reason": panel.end_reason})

    # 4c. Saturation (3 short low-info customer messages back-to-back)
    if not panel.lost and not panel.won:
        recent_cust = [m["text"] for m in panel.dialog if m.get("role") == "customer"]
        if detect_saturation(recent_cust, tenant=(opp_meta or {}).get("company")):
            panel.lost = True
            panel.end_reason = "saturated"
            await send({"event": f"{panel.side}_msg_meta",
                        "saturation_detected": True})

    # User-initiated stop check
    if ctx and ctx.stopped:
        panel.lost = True
        panel.end_reason = "stopped_by_user"
        return False

    # 5. Score
    panel.chat_for_scorer = _build_chat_for_scorer(panel.dialog)
    score_result = None
    try:
        score_result = await score_turn(panel.chat_for_scorer)
    except Exception as e:
        log.warning("[%s] scorer failed: %s", panel.side, e)
    if score_result:
        panel.persuasion_history.append(score_result["score"])
        panel.commitment_history.append(score_result["commitment_level"])
        panel.live_persuasion_history.append(score_result["score"])
        await send({
            "event": f"{panel.side}_score",
            "turn": cust_seq,
            "score": score_result["score"],
            "commitment": score_result["commitment_level"],
            "reason": score_result["reason"],
        })
        # Three-of-three close guard. G1 already true (commitment_level==5).
        # Compute G2 and G3 to decide whether to actually fire `won`.
        guard_close = False
        if score_result["commitment_level"] >= COMMITMENT_WIN_THRESHOLD:
            # G2 — close-language regex match OR strong persuasion score,
            # MINUS counter-offer block: a customer making a counter-offer
            # ("round to X", "if you can do Y", "what if 3000") is
            # negotiating, NOT accepting, even when Gemini scores commit=5.
            is_counter_offer = detect_counter_offer(
                cust_text, tenant=(opp_meta or {}).get("company"))
            g2 = (
                detect_close_signal(cust_text)
                or score_result["score"] >= PERSUASION_CONFIRM_THRESHOLD
            ) and not is_counter_offer
            prev_agent_text = ""
            if (len(panel.dialog) >= 2
                    and panel.dialog[-2].get("role") == "agent"):
                prev_agent_text = panel.dialog[-2].get("text") or ""
            g3 = _agent_just_offered_close(panel.last_directive, prev_agent_text,
                                            tenant=(opp_meta or {}).get("company"))
            # 2026-05-05 — explicit-payment bypass for G3. Triggered case:
            # a897d59b turn 31 — customer said "last 4 digits are 7834"
            # after agent made a concession statement (not a close offer).
            # G3 was rejecting because agent_just_offered_close=False, but
            # customer providing payment info is definitive close regardless
            # of whether agent explicitly asked. Detect via the customer's
            # text: if it contains CC last-4 or explicit payment info, G3
            # becomes irrelevant.
            # 2026-05-17 (dcfbdb58) — vectors-primary, structural-fallback per
            # the 2026-05-11 decision. _EXPLICIT_PAYMENT_REGEX is FROZEN (no
            # new alternations); narrated forms go through model2vec; the
            # bare-digit form ("7834") has no semantics so it uses a tiny
            # structural test gated on an agent payment/close ask anywhere in
            # the recent window (not just the immediately-prior line — that
            # was the original bug).
            _regex_pay = bool(_EXPLICIT_PAYMENT_REGEX.search(cust_text or ""))
            _narrated_pay = False
            try:
                from intent_classifier import intent_score as _isc, is_available as _iav
                if _iav():
                    _narrated_pay, _s, _a = _isc(
                        cust_text or "", "customer_payment_provided",
                        tenant=(opp_meta or {}).get("company"))
            except Exception:
                pass
            _bare_pay = (
                bool(_BARE_LAST4_RE.match((cust_text or "").strip()))
                and not detect_counter_offer(
                    cust_text, tenant=(opp_meta or {}).get("company"))
                and _agent_recently_offered_close(panel, opp_meta))
            explicit_payment_close = _regex_pay or _narrated_pay or _bare_pay
            if explicit_payment_close:
                panel.payment_seen = True
            # 2026-05-13 — customer-initiated close bypass for G3. Triggered:
            # session 329d36c3 turn 19 — customer said "I'll complete the
            # purchase now" after agent just said "You are welcome! 🤘"
            # (empathy directive, no close offer). G3 was rejecting because
            # agent_just_offered_close=False, but the customer is explicitly
            # driving the close themselves — agent's previous strategy is
            # irrelevant once the customer commits in words. Uses model2vec
            # semantic match at threshold 0.62 (tuned to keep "I'll complete
            # the purchase now" while rejecting vague gratitude).
            customer_initiated = False
            try:
                from intent_classifier import intent_score, is_available
                if is_available():
                    tenant = (opp_meta or {}).get("company")
                    customer_initiated, _s, _a = intent_score(
                        cust_text or "", "customer_initiated_close",
                        tenant=tenant)
            except Exception:
                pass
            # 2026-05-17 (dcfbdb58) — sustained-close latch. If commit==WIN
            # with confirming persuasion holds for >=2 consecutive live
            # turns, the close is real even when G3 keeps rejecting (the
            # agent's last line was a wait-ack, not a fresh offer). Without
            # this, a definitive close that the guard blocks gets converted
            # to a loss by the next polite turn.
            _ch, _ph = panel.commitment_history, panel.live_persuasion_history
            sustained_close = (
                len(_ch) >= 2 and len(_ph) >= 2
                and _ch[-1] >= COMMITMENT_WIN_THRESHOLD
                and _ch[-2] >= COMMITMENT_WIN_THRESHOLD
                and _ph[-1] >= PERSUASION_CONFIRM_THRESHOLD
                and _ph[-2] >= PERSUASION_CONFIRM_THRESHOLD)
            g3_or_bypass = (g3 or explicit_payment_close
                            or customer_initiated or sustained_close)
            if not panel.won and g2 and g3_or_bypass:
                guard_close = True
                panel.won = True
                panel.end_reason = "commitment_5"
                await send({
                    "event": "won",
                    "side": panel.side,
                    "turn": cust_seq,
                    "score": score_result["score"],
                    "commitment": score_result["commitment_level"],
                })
            else:
                # Guard rejected — telemetry-only event so we can measure
                # how often the gate saves us from false positives.
                await send({
                    "event": "close_guard_block",
                    "side": panel.side,
                    "turn": cust_seq,
                    "score": score_result["score"],
                    "commitment": score_result["commitment_level"],
                    "g2_passed": g2,
                    "g3_passed": g3,
                    "explicit_payment_close": explicit_payment_close,
                    "customer_initiated_close": customer_initiated,
                    "counter_offer_detected": is_counter_offer,
                })
                log.info("[%s] close_guard_block t=%d score=%.2f commit=%d g2=%s g3=%s payment_bypass=%s counter_offer=%s",
                         panel.side, cust_seq, score_result["score"],
                         score_result["commitment_level"], g2, g3,
                         explicit_payment_close, is_counter_offer)
        # Stall detection — applies whether guard fired or not, as long as
        # we didn't actually declare won.
        # 2026-05-10 — Engagement override. The persuasion+commit threshold
        # conflates "disengaged customer" with "engaged but frustrated by
        # agent quality." Real example (d62ad2ca): customer asked
        # "Why couldn't you offer 5,000 from the start?" — clearly engaged,
        # but persuasion was 0.1 because the agent was evading. Killing the
        # panel here gives up on a recoverable conversation. The right move
        # is to keep the agent in the seat and let it try again.
        cust_t_lower = (cust_text or "").lower().strip()
        # 2026-05-11 — Engagement override now uses model2vec semantic match
        # (primary) with the prior keyword set as a Hebrew-aware fallback.
        # Question-mark ending remains a fast deterministic signal.
        is_engaged_question = False
        if cust_t_lower and len(cust_t_lower) >= 25:
            if cust_t_lower.endswith("?"):
                is_engaged_question = True
            else:
                try:
                    from intent_classifier import intent_score, is_available
                    if is_available():
                        decision, _s, _a = intent_score(cust_t_lower, "engaged_question")
                        if decision:
                            is_engaged_question = True
                except Exception:
                    pass
                # Fallback (model unavailable OR Hebrew text — anchors are EN-only)
                if not is_engaged_question:
                    is_engaged_question = (
                        any(k in cust_t_lower for k in (
                            "why", "how", "what", "when", "where", "explain",
                            "tell me", "answer", "didn't you", "did you",
                            "can you", "will you", "could you",
                            "למה", "איך", "מה ", "מתי ", "תסביר", "תענה",
                        ))
                        or any(k in cust_t_lower for k in (
                            "doesn't prove", "doesn't tell me", "doesn't answer",
                            "doesn't address", "doesn't change", "doesn't solve",
                            "not enough", "still not", "still doesn't",
                            "i'll keep waiting", "i'll wait", "i need to know",
                            "i still need", "without that", "without knowing",
                            "before i decide", "before i commit",
                            "לא מספיק", "לא פותר", "לא עונה",
                        ))
                    )
        if (not guard_close
                and not is_engaged_question
                and len(panel.commitment_history) >= STALL_THRESHOLD_TURNS
                and all(c <= 1 for c in panel.commitment_history[-STALL_THRESHOLD_TURNS:])
                and all(s < 0.3 for s in panel.persuasion_history[-STALL_THRESHOLD_TURNS:])):
            panel.lost = True
            panel.end_reason = "stalled_low_engagement"
        elif (is_engaged_question
              and len(panel.commitment_history) >= STALL_THRESHOLD_TURNS
              and all(c <= 1 for c in panel.commitment_history[-STALL_THRESHOLD_TURNS:])
              and all(s < 0.3 for s in panel.persuasion_history[-STALL_THRESHOLD_TURNS:])):
            log.info("[%s] stall-condition met but customer is engaged-questioning — "
                     "deferring panel-end (text=%r)", panel.side, cust_t_lower[:80])
    elif detect_close_signal(cust_text):
        # Backup signal — close-language detected
        panel.won = True
        panel.end_reason = "close_signal"
        await send({"event": "won", "side": panel.side, "turn": cust_seq})

    # 6. ClusterPlan advancement (right panel only) — after customer reply lands
    # so we can use customer's last message as the advancement signal.
    if panel_is_supervised and panel.plan_state is not None:
        # Increment turns_in_current_phase BEFORE checking advance (we just had a turn)
        panel.plan_state.turns_in_current_phase += 1
        before_phase = panel.plan_state.current_phase_id
        advanced, reason = advance_plan(panel.plan_state, directive, cust_text)
        # If plan ABORTED (declared no-need, signed elsewhere, etc.) → end panel
        if panel.plan_state.aborted and not panel.lost and not panel.won:
            panel.lost = True
            panel.end_reason = f"plan_aborted:{panel.plan_state.abort_reason}"
            await send({
                "event": f"{panel.side}_msg",
                "role": "system",
                "text": f"[Plan aborted: {panel.plan_state.abort_reason}]",
                "turn": cust_seq + 1,
            })
        if advanced and not panel.plan_state.aborted:
            await send({
                "event": "plan_advanced",
                "side": panel.side,
                "from_phase": before_phase,
                "to_phase": panel.plan_state.current_phase_id,
                "reason": reason,
                "turn": cust_seq,
            })

    # Win-mode plan advancement — only when active (T-76 Day 4)
    if (panel_is_supervised and panel.win_plan_state is not None
            and panel.plan_mode_active == "win"):
        panel.win_plan_state.turns_in_current_phase += 1
        wp_before = panel.win_plan_state.current_phase_id
        wp_advanced, wp_reason = wp_mod.advance_win_plan(
            panel.win_plan_state, directive, cust_text)
        if wp_advanced:
            await send({
                "event": "win_plan_advanced",
                "side": panel.side,
                "from_phase": wp_before,
                "to_phase": panel.win_plan_state.current_phase_id,
                "active_branch": panel.win_plan_state.active_branch,
                "reason": wp_reason,
                "turn": cust_seq,
            })
        if directive:
            wp_mod.record_strategy_adherence(panel.win_plan_state, directive)

    return not (panel.won or panel.lost)


def _build_chat_for_scorer(dialog: list[dict]) -> list[dict]:
    """Convert our dialog history to the scorer's expected format."""
    out = []
    for m in dialog:
        role = "user" if m.get("role") == "customer" else "assistant"
        out.append({
            "sequence_number": m["sequence_number"],
            "role": role,
            "text": m["text"],
            "persusive_score": None,  # scorer fills in for the latest user msg
        })
    return out


def _brief_anchors(anchors: dict | None) -> dict | None:
    """7-bonus — extract the small, demo-friendly subset of the anchor pack
    so the UI hovercard can show the economic reference frame at a glance.
    Empty dict -> None so the UI hides the chip when no anchors loaded."""
    if not anchors:
        return None
    keys = (
        "last_year_price_usd", "current_quoted_price_usd",
        "market_avg_for_segment_usd", "max_discount_pct_internal",
        "claimed_increase_pct", "actual_market_yoy_change_pct",
        "loyalty_years", "synthetic", "provenance",
        # Ecommerce-side keys (T-86 anchor pack)
        "max_authorized_discount_pct_internal",
        "_cg_queries_returned_content", "_cg_queries_total",
    )
    brief = {k: anchors[k] for k in keys if k in anchors and anchors[k] is not None}
    return brief or None


def _summarize_directive(directive: dict, directive_meta: dict | None = None) -> dict:
    """Compact summary for UI directive-tag — includes CG entity counts and
    the top fact-to-anchor for visual proof of CG-grounding."""
    strat = directive.get("strategy") or {}
    knowledge = directive.get("knowledge") or {}
    facts = knowledge.get("facts_to_anchor") or []
    must_not = knowledge.get("must_not_say") or directive.get("must_not_say") or []
    # Tier 2 concrete_move (Phase 1 of strategy-enum-extension, 2026-05-03):
    # surface name + parameters compactly for the UI per-turn directive chip
    concrete_move = strat.get("concrete_move")
    concrete_move_summary = None
    if isinstance(concrete_move, dict) and concrete_move.get("name"):
        params = concrete_move.get("parameters") or {}
        # Extract a 1-line "what" — the most informative parameter, truncated
        primary_param = None
        for k in ("target_price", "new_amount", "items_to_stack",
                   "dimensions_to_probe", "request_artifact",
                   "features_to_anchor", "mechanism", "continuity_signal"):
            if k in params and params[k]:
                v = params[k]
                primary_param = (k, v if not isinstance(v, list) else ", ".join(str(x) for x in v[:3]))
                break
        concrete_move_summary = {
            "name": concrete_move.get("name"),
            "primary_param": (
                f"{primary_param[0]}: {str(primary_param[1])[:80]}" if primary_param else None
            ),
            "n_params": len(params),
        }

    summary = {
        "primary_strategy": strat.get("primary") or directive.get("primary_strategy"),
        "tone": strat.get("tone") or directive.get("tone"),
        "rules": directive.get("rules_to_enforce") or [],
        "concrete_move": concrete_move_summary,
        "facts_to_anchor": [
            {"text": (f.get("text") or "")[:140],
             "source": (f.get("source_ref") or "")[:80]}
            for f in facts[:2]
        ],
        "must_not_say_top": (
            (must_not[0].get("text") if isinstance(must_not[0], dict) else str(must_not[0]))
            if must_not else None
        ),
        # 7.1 — supervisor's audit trail rationale ("why this directive?").
        # The Sonnet supervisor already emits this in directive.audit.rationale_summary;
        # surface it for the per-turn UI tooltip.
        "rationale": ((directive.get("audit") or {}).get("rationale_summary")
                      or directive.get("rationale")),
        # 7.4 — confidence indicator. Strategist emits confidence as a dict
        # {overall, band}; other engines (Planner) may emit a bare float —
        # tolerate both so the summary never crashes the turn emit.
        "confidence": (
            directive["confidence"].get("overall")
            if isinstance(directive.get("confidence"), dict)
            else directive.get("confidence")),
        "confidence_band": (
            directive["confidence"].get("band")
            if isinstance(directive.get("confidence"), dict) else None),
    }
    if directive_meta:
        summary["mode"] = directive_meta.get("mode")
        # Forward engine-agnostic per-turn telemetry (Planner surfaces
        # tier/user_state/sop_valid/architecture here; harmless if absent).
        for _k in ("engine", "architecture", "tier", "user_state",
                   "sop_valid", "off_sop",
                   # envelope telemetry
                   "envelope_mode", "envelope_gate_fired",
                   "envelope_applied", "envelope",
                   # post-render gate telemetry
                   "gates_applied", "gates_fired",
                   "gates_regens", "gates_final",
                   # mined playbook telemetry (Planner positive-direction)
                   "playbooks_loaded", "playbook_ids",
                   "playbook_block_present"):
            if directive_meta.get(_k) is not None:
                summary[_k] = directive_meta.get(_k)
        # Multi-turn arc awareness — surface phase chip data to the UI.
        cp = directive_meta.get("conversation_phase")
        if isinstance(cp, dict) and cp.get("current"):
            summary["conversation_phase"] = {
                "current": cp.get("current"),
                "turns_in_phase": cp.get("turns_in_phase"),
                "cluster_plan_phase": cp.get("cluster_plan_phase"),
            }
        # 7-bonus — surface anchor snapshot for the hovercard chip.
        ab = directive_meta.get("anchors_brief")
        if ab:
            summary["anchors_brief"] = ab
        if directive_meta.get("mode") == "1b":
            summary["cg"] = {
                "entities": directive_meta.get("cg_entities", 0),
                "relations": directive_meta.get("cg_relations", 0),
                "chunks": directive_meta.get("cg_chunks", 0),
            }
        retry = directive_meta.get("adherence_retry")
        if retry:
            summary["adherence_retry"] = {
                "retried": retry.get("retried", False),
                "violation": retry.get("violation"),
                "second_strategy": retry.get("second_strategy"),
            }
        sig_retry = directive_meta.get("signal_adherence_retry")
        if sig_retry:
            summary["signal_adherence_retry"] = {
                "retried": sig_retry.get("retried", False),
                "violation": sig_retry.get("violation"),
                "second_strategy": sig_retry.get("second_strategy"),
                "primary_signal": sig_retry.get("primary_signal"),
                "allowed_strategies": sig_retry.get("allowed_strategies"),
            }
        # R4 — directive-consistency retry (Q12). Surface for UI counterfactual.
        cons_retry = directive_meta.get("consistency_retry")
        if cons_retry and cons_retry.get("retried"):
            summary["consistency_retry"] = {
                "retried": True,
                "violation_rules": cons_retry.get("violation_rules") or [],
                "fell_back_to_strip": bool(cons_retry.get("fell_back_to_strip")),
            }
        # R6 — move-validity retry. Surface for UI counterfactual.
        mv_retry = directive_meta.get("move_validity_retry")
        if mv_retry and mv_retry.get("retried"):
            summary["move_validity_retry"] = {
                "retried": True,
                "first_violation": mv_retry.get("first_violation"),
                "first_move_name": mv_retry.get("first_move_name"),
                "second_move_name": mv_retry.get("second_move_name"),
                "fell_back": bool(mv_retry.get("fell_back")),
            }
        if directive_meta.get("plan_phase_id") or directive_meta.get("plan_phase_label"):
            summary["plan_phase"] = {
                "phase_id": directive_meta.get("plan_phase_id"),
                "phase_label": directive_meta.get("plan_phase_label"),
                "preferred_actions": directive_meta.get("plan_preferred_actions"),
                "branch": directive_meta.get("plan_active_branch"),
                "mode": directive_meta.get("plan_mode_active"),
            }
    sa = directive.get("signal_analysis")
    if isinstance(sa, dict):
        primary = sa.get("primary_signal")
        signals = sa.get("observed_signals") or []
        primary_conf = next(
            (s.get("confidence") for s in signals
             if isinstance(s, dict) and s.get("signal") == primary),
            None)
        summary["signal_analysis"] = {
            "primary_signal": primary,
            "primary_confidence": primary_conf,
            "plan_alignment": sa.get("plan_alignment"),
            "n_signals": len(signals),
        }
    return summary


def _state_for_classifier(opp_meta: dict, dialog: list[dict]) -> dict:
    """Adapter: build minimal state dict the classifier expects."""
    return {
        "profile": {
            "primary_motivator": opp_meta.get("primary_motivator"),
            "decision_logic": opp_meta.get("decision_logic"),
            "trust_level": opp_meta.get("trust_level"),
        },
        "messages": [
            {"direction": "inbound" if m.get("role") == "customer" else "outbound",
             "text": m.get("text"),
             "timestamp": None,
             "is_reminder": 0, "is_followup": 0, "automatic_response": 0}
            for m in dialog
        ],
        "turn_states": [],
        "strategies": [],
    }


def _compute_features_for_classifier(opp_meta: dict, historical_messages: list[dict]) -> dict:
    """Use the historical opp's features (already known) — for POC we route based
    on the static historical features since the cluster assignment was made on those.
    This matches how Batch 10 v1 evaluation worked."""
    # Re-fetch via attribution.fetch_opp_data — but we'd need a connection.
    # Simpler: open a short-lived connection, pull data, compute.
    conn = open_conn()
    try:
        data = fetch_opp_data(conn, opp_meta["id"])
        opp_for_features = {
            "id": opp_meta["id"],
            "company": opp_meta["company"],
            "opp_type": opp_meta.get("opp_type"),
            "status": opp_meta.get("status"),
            "created_at": opp_meta.get("created_at"),
            "status_update_timestamp": opp_meta.get("status_update_timestamp"),
            "expiration_date": None,
            "total_inbounds": opp_meta.get("total_inbounds") or 0,
            "total_outbounds": opp_meta.get("total_outbounds") or 0,
            "total_reminders": opp_meta.get("total_reminders") or 0,
            "client_engaged": None,
            "primary_motivator": opp_meta.get("primary_motivator"),
            "objection_pattern": opp_meta.get("objection_pattern"),
            "decision_logic": opp_meta.get("decision_logic"),
            "trust_level": opp_meta.get("trust_level"),
            "regulatory_focus": opp_meta.get("regulatory_focus"),
            "budget_sensitivity": opp_meta.get("budget_sensitivity"),
            "purchase_urgency": opp_meta.get("purchase_urgency"),
            "primary_resistance": opp_meta.get("primary_resistance"),
        }
        return features_for_opp(opp_for_features, data)
    finally:
        conn.close()


# ── Main session orchestrator ───────────────────────────────────────────────
async def run_session(session_id: str, opp_id: str, send: SEND_FN, get_speed: SPEED_FN,
                       scenario_meta: dict | None = None,
                       hard_customer: bool = False,
                       seed_end_override: int = 0,
                       engine: str = "strategist",
                       planner_envelope: str = "off",
                       engine_left: str = "baseline",
                       engine_params: dict | None = None,
                       engine_params_left: dict | None = None) -> None:
    log.info("[%s] starting session for opp %s", session_id[:8], opp_id)
    logger = SessionLogger(session_id, opp_id, scenario=scenario_meta)

    # T-86 trace logger — captures every LLM/CG/gate event for end-to-end
    # diagnostic walkthrough; persists to test_logs/{date}/{sid}_trace.json
    from trace_logger import TraceLogger
    trace = TraceLogger(session_id, opp_id, scenario_meta=scenario_meta)
    trace_token = TraceLogger.set_current(trace)

    # Wrap send to also log
    async def send_and_log(ev: dict):
        logger.add_event(ev)
        await send(ev)

    ctx = SessionCtx()
    SESSION_CTXS[session_id] = ctx
    try:
        await _run_session_inner(session_id, opp_id, send_and_log, get_speed, logger, ctx,
                                  hard_customer=hard_customer,
                                  seed_end_override=seed_end_override,
                                  engine=engine,
                                  planner_envelope=planner_envelope,
                                  engine_left=engine_left,
                                  engine_params=engine_params,
                                  engine_params_left=engine_params_left)
    except asyncio.CancelledError:
        logger.add_note("session cancelled (likely WS disconnect)")
        raise
    except Exception as e:
        logger.add_error("run_session", str(e))
        log.exception("[%s] run_session failed", session_id[:8])
    finally:
        SESSION_CTXS.pop(session_id, None)
        try:
            log_path = logger.write()
            log.info("[%s] log written: %s", session_id[:8], log_path)
        except Exception as e:
            log.warning("[%s] log write failed: %s", session_id[:8], e)
        try:
            trace.write()
        except Exception as e:
            log.warning("[%s] trace write failed: %s", session_id[:8], e)
        try:
            TraceLogger.reset_current(trace_token)
        except Exception:
            pass


# Module-level registry so main.py's WS handler can flip ctx.stopped on Stop
SESSION_CTXS: dict[str, SessionCtx] = {}


def request_stop(session_id: str) -> bool:
    """Flip the ctx.stopped flag for a session — handler calls this on Stop."""
    ctx = SESSION_CTXS.get(session_id)
    if ctx is None:
        return False
    ctx.stopped = True
    return True


async def _run_session_inner(session_id: str, opp_id: str, send_and_log: SEND_FN,
                              get_speed: SPEED_FN, logger: SessionLogger,
                              ctx: SessionCtx | None = None,
                              hard_customer: bool = False,
                              seed_end_override: int = 0,
                              engine: str = "strategist",
                              planner_envelope: str = "off",
                              engine_left: str = "baseline",
                              engine_params: dict | None = None,
                              engine_params_left: dict | None = None) -> None:
    # Snapshot CG endpoint counters at session start; we'll compute the delta
    # at session end to surface "CG calls this session" in the architecture banner.
    cg_calls_at_start = dict(CG_ENDPOINT_CALLS)

    conn = open_conn()
    try:
        opp_meta = fetch_opp_meta(conn, opp_id)
        if opp_meta is None:
            await send_and_log({"event": "error", "message": "Opportunity not found"})
            return
        # R10 — propagate per-session hard-customer flag onto opp_meta so the
        # simulator's _hard_customer_enabled() check picks it up. Per-session
        # override takes precedence over the env flag.
        if hard_customer:
            opp_meta["_hard_customer"] = True
            log.info("[%s] hard_customer mode ENABLED for this session",
                     session_id[:8])
            logger.add_note("hard_customer_mode_enabled")
        # Per-panel engine selection (any-vs-any A/B). The engine ids ride on
        # opp_meta so _live_turn can route each panel independently. Defaults
        # (_engine_left=baseline, _engine_right=strategist) reproduce the
        # classic control-vs-supervisor pairing exactly. `_engine` is kept as a
        # back-compat alias for the R-side id.
        opp_meta["_engine_left"] = engine_left or "baseline"
        opp_meta["_engine_right"] = engine
        opp_meta["_engine"] = engine
        opp_meta["_engine_params_left"] = dict(engine_params_left or {})
        opp_meta["_engine_params_right"] = dict(engine_params or {})
        # Back-compat: PlannerEngine.produce reads _planner_envelope off opp_meta.
        # Honor an explicit per-panel value first, else the legacy top-level arg.
        opp_meta["_planner_envelope"] = (
            (engine_params or {}).get("planner_envelope") or planner_envelope or "off")
        log.info("[%s] engines: L=%s R=%s", session_id[:8],
                 opp_meta["_engine_left"], opp_meta["_engine_right"])
        logger.add_note(f"engines_L_{opp_meta['_engine_left']}_R_{opp_meta['_engine_right']}")
        # T-81 anchor enrichment — fetch economic reference frame so the
        # supervisor and customer simulator both negotiate from real anchors
        # (last year's price, market avg, max discretionary discount) instead
        # of the unanchored adversarial defaults that produced staircase
        # pricing in 2026-05-02 sessions.
        try:
            anchors = fetch_insurance_anchors(conn, opp_id, opp_meta)
            if anchors:
                opp_meta["anchors"] = anchors
                log.info("[%s] anchors loaded: %s", session_id[:8],
                          {k: v for k, v in anchors.items()
                           if k in ("last_year_price_usd", "current_quoted_price_usd",
                                    "market_avg_for_segment_usd", "synthetic", "provenance")})
        except Exception as e:
            log.warning("[%s] anchor fetch failed: %s", session_id[:8], e)
        messages = fetch_messages(conn, opp_id)

        # T-84 voice profile — extract customer's actual phrasing/register/
        # decisiveness markers from historical messages. Injected into the
        # customer simulator's system prompt so it speaks in this customer's
        # specific voice, not a generic LLM voice.
        try:
            from voice_profile import extract_voice_profile
            voice = await extract_voice_profile(messages)
            if voice:
                opp_meta["voice_profile"] = voice
                log.info("[%s] voice_profile loaded: register=%s, n_msgs=%d, summary=%r",
                          session_id[:8], voice.get("register"),
                          voice.get("n_messages", 0),
                          (voice.get("voice_summary") or "")[:120])
        except Exception as e:
            log.warning("[%s] voice_profile extraction failed: %s", session_id[:8], e)

        turn_states = fetch_turn_states(conn, opp_id)
        persuasive_scores = fetch_persuasive_scores(conn, opp_id)
        business_rules = fetch_business_rules(conn, opp_meta["company"])
        # Smarter seed-end: last engaged customer turn (commit≥2 or score≥0.4)
        # This gives the supervisor a fair chance — intervening BEFORE customer
        # has fully disengaged.
        # 2026-05-12 — UI override: if seed_end_override > 0, use it directly
        # (clamped to a valid range). Lets researchers test the supervisor
        # against different seed depths from the front-end slider.
        auto_failure_idx = find_supervisor_intervention_index(
            messages, turn_states, persuasive_scores,
        )
        if seed_end_override and seed_end_override > 0:
            max_valid = max(1, len(messages) - 1)
            failure_idx = min(seed_end_override, max_valid)
            seed_source = f"ui_override (requested={seed_end_override}, clamped={failure_idx}, auto_would_be={auto_failure_idx})"
        else:
            failure_idx = auto_failure_idx
            seed_source = "peak_engagement (auto)"
        # Capture for log
        n_inbound_raw = sum(1 for m in messages if m.get("direction") == "inbound")
        last_inbound_idx_raw = find_failure_mode_turn_index(messages)
        logger.add_note(
            f"seed_strategy: {seed_source}. failure_idx={failure_idx} "
            f"(last_inbound_idx={last_inbound_idx_raw}, n_inbound_total={n_inbound_raw})"
        )
        log.info("[%s] seed_end: %s → failure_idx=%d",
                 session_id[:8], seed_source, failure_idx)
    finally:
        conn.close()

    if failure_idx is None or failure_idx < 1:
        await send_and_log({"event": "error", "message": "No failure-mode turn detected; need at least 2 messages"})
        return

    # Load v1 classifier (right panel only)
    try:
        classifier = load_classifier_v1(opp_meta["company"])
    except Exception as e:
        log.warning("Failed to load v1 classifier for %s: %s", opp_meta["company"], e)
        classifier = None

    # Pre-compute features ONCE for the classifier (historical opp features are static)
    cached_features = None
    if classifier is not None:
        try:
            cached_features = await asyncio.to_thread(
                _compute_features_for_classifier, opp_meta, messages)
            logger.add_note(f"classifier_features_cached: {len(cached_features)} keys")
        except Exception as e:
            log.warning("Failed to pre-compute features: %s", e)
            logger.add_error("feature_precompute", str(e))

    # Build seed dialog + initialize panels
    seed_dialog = _build_seed_dialog(messages, failure_idx)
    log.info("[%s] seed dialog: %d msgs (failure_idx=%d)", session_id[:8], len(seed_dialog), failure_idx)

    left = PanelState(side="left", dialog=[], seq_counter=0)
    right = PanelState(side="right", dialog=[], seq_counter=0)
    # Stamp session_id on panels so per-turn dialog logs can prefix with it
    left.session_id = session_id
    right.session_id = session_id

    # Customer simulator (shared instance for both panels' historical alignment)
    simulator = CustomerSimulator(opp_meta, messages)

    precomputed_scores = {}  # could fetch from persuasive_score if desired

    # Closed-loop READ: pull precedent decisions from CG (proof the system
    # has a learning corpus that's been growing)
    try:
        async with httpx.AsyncClient(timeout=15) as pc:
            precedents = await fetch_precedent_decisions(pc, opp_meta["company"])
        logger.add_note(f"precedents_in_graph: {precedents.get('count', 0)}")
    except Exception as e:
        precedents = {"count": 0, "_error": str(e)}
        logger.add_error("precedent_fetch", str(e))

    # R1 (2026-05-04) — make precedents available to Mode 1b directive so
    # the supervisor can ground its decisions in past sessions, not just the
    # current dialog. Closed-loop READ side wired into the prompt path.
    opp_meta["precedents"] = precedents


    # Precedent fetch demonstrates Scenario 7 (Decision Precedent Lookup) if
    # the graph returned at least one decision edge for this workspace.
    if ctx is not None and (precedents.get("count") or 0) > 0:
        ctx.scenarios.add("S7")

    # 6.2 — surface a sanitized sample of precedent edges so the UI
    # closed-loop chip can show actual past decisions on hover (not just an
    # aggregate count). One CG call per session — reuses the fetch above.
    precedent_sample_brief = []
    for d in (precedents.get("sample") or [])[:5]:
        rc = d.get("relation_context") or {}
        precedent_sample_brief.append({
            "src": (d.get("src_id") or "")[:80],
            "tgt": (d.get("tgt_id") or "")[:60],
            "confidence": rc.get("confidence_score"),
            "outcome": (rc.get("quantitative_data") or "")[:120],
            "rationale": (rc.get("decision_trace") or "")[:200],
        })

    await send_and_log({
        "event": "session_started",
        "n_seed_msgs": len(seed_dialog),
        "tenant": opp_meta["company"],
        "opp_type": opp_meta.get("opp_type"),
        "precedents_in_graph": precedents.get("count", 0),
        "precedent_strategies": dict(sorted(
            (precedents.get("by_strategy") or {}).items(),
            key=lambda kv: -kv[1])[:6]),
        "precedent_sample": precedent_sample_brief,
        "guide_scenarios": GUIDE_SCENARIOS,
        "scenarios_demonstrated_initial": sorted(ctx.scenarios) if ctx else [],
    })

    # Stream seed dialog to BOTH panels in parallel
    await asyncio.gather(
        _stream_seed(left, [dict(m) for m in seed_dialog], opp_id, send_and_log, get_speed,
                     precomputed_scores, turn_states),
        _stream_seed(right, [dict(m) for m in seed_dialog], opp_id, send_and_log, get_speed,
                     precomputed_scores, turn_states),
    )

    await send_and_log({"event": "seed_complete", "seed_end_turn": failure_idx})
    log.info("[%s] seed complete; entering live A/B phase", session_id[:8])

    # Live A/B phase: alternating turns
    # Compute starting agent-turn-index in historical sequence
    agent_turn_in_history = sum(1 for m in messages[: failure_idx + 1] if m.get("direction") == "outbound")

    # Open one shared httpx client for Mode 1b CG calls (avoids per-turn setup)
    max_turns = max_live_turns_for(opp_meta)
    async with httpx.AsyncClient(timeout=30) as http_client:
        for live_iter in range(max_turns):
            # Stop-check at every iteration top
            if ctx and ctx.stopped:
                log.info("[%s] live loop: ctx.stopped — terminating", session_id[:8])
                break
            if (left.won or left.lost) and (right.won or right.lost):
                break

            # Run both panels in parallel — DON'T swallow CancelledError
            left_task = _live_turn(left, opp_meta, business_rules, None, simulator, send_and_log,
                                     get_speed, agent_turn_in_history + live_iter, messages,
                                     cached_features, ctx, http_client) \
                         if not (left.won or left.lost) else asyncio.sleep(0)
            right_task = _live_turn(right, opp_meta, business_rules, classifier, simulator, send_and_log,
                                     get_speed, agent_turn_in_history + live_iter, messages,
                                     cached_features, ctx, http_client) \
                          if not (right.won or right.lost) else asyncio.sleep(0)
            try:
                await asyncio.gather(left_task, right_task)
            except asyncio.CancelledError:
                # Propagate — this lets the outer try/finally write the session log
                raise
            except Exception as e:
                log.warning("[%s] live_turn iteration error: %s", session_id[:8], e)
                continue

    # Timeout if we hit MAX_LIVE_TURNS
    if not (left.won or left.lost):
        left.lost = True
        left.end_reason = "timeout"
    if not (right.won or right.lost):
        right.lost = True
        right.end_reason = "timeout"

    await send_and_log({"event": "end", "side": "left", "outcome": "won" if left.won else "lost",
                          "reason": left.end_reason})
    await send_and_log({"event": "end", "side": "right", "outcome": "won" if right.won else "lost",
                          "reason": right.end_reason})

    # ── Win-proximity scoring (T-77) ──
    # Continuous 0..1 score: how close was each panel to winning?
    # WIN → 1.00; otherwise α·trajectory_auc + β·semantic_sim + γ·payment_capture
    try:
        scenario_for_proximity = {
            "tenant": opp_meta.get("company") if opp_meta else None,
            "opp_type": opp_meta.get("type") if opp_meta else None,
        }
        proximity_centroids = win_proximity._load_all_win_centroids()
        left_proximity = win_proximity.score(left, scenario_for_proximity,
                                                centroids_cache=proximity_centroids)
        right_proximity = win_proximity.score(right, scenario_for_proximity,
                                                 centroids_cache=proximity_centroids)
        proximity_aggregate = win_proximity.aggregate_proximity(left_proximity, right_proximity)
        await send_and_log({
            "event": "win_proximity",
            "left": left_proximity,
            "right": right_proximity,
            "aggregate": proximity_aggregate,
        })
    except Exception as e:
        log.warning("[%s] win_proximity scoring failed: %s", session_id[:8], e)
        left_proximity = right_proximity = proximity_aggregate = None
    cache_total = QUERY_CACHE_STATS["hits"] + QUERY_CACHE_STATS["misses"]
    cache_hit_rate = (
        round(QUERY_CACHE_STATS["hits"] / cache_total, 3) if cache_total else 0.0
    )
    # Compute per-session CG endpoint deltas
    cg_calls_this_session = {
        k: CG_ENDPOINT_CALLS[k] - cg_calls_at_start.get(k, 0)
        for k in CG_ENDPOINT_CALLS
    }
    await send_and_log({"event": "session_complete",
                          "left_outcome": "won" if left.won else "lost",
                          "right_outcome": "won" if right.won else "lost",
                          # 6.1 — surface end-reason per panel so UI can render
                          # the failure-mode taxonomy badge.
                          "left_reason": left.end_reason,
                          "right_reason": right.end_reason,
                          "left_persuasion_final": (left.persuasion_history[-1] if left.persuasion_history else None),
                          "right_persuasion_final": (right.persuasion_history[-1] if right.persuasion_history else None),
                          "cg_cache": {**QUERY_CACHE_STATS,
                                        "hit_rate": cache_hit_rate},
                          "cg_endpoint_calls": cg_calls_this_session,
                          "use_query_auto": USE_QUERY_AUTO,
                          })
    log.info("[%s] session complete: left=%s right=%s", session_id[:8],
             left.end_reason, right.end_reason)

    # Closed-loop learning: emit decision-trace edge to CG so future
    # supervisor calls can retrieve this session as precedent.
    if right.last_directive:
        outcome = {
            "outcome": "won" if right.won else "lost",
            "commitment_peak": max(right.commitment_history) if right.commitment_history else 0,
            "persuasion_peak": max(right.persuasion_history) if right.persuasion_history else 0,
            "session_id": session_id,
        }
        try:
            async with httpx.AsyncClient(timeout=15) as ec:
                trace_result = await emit_decision_trace(opp_meta, right.last_directive,
                                                          outcome, http_client=ec)
            await send_and_log({"event": "decision_trace_emitted",
                                  "result": trace_result})
            logger.add_note(f"decision_trace_emit: {trace_result}")
            # S4 fires when the structured decision-edge actually lands in CG
            if ctx is not None and trace_result.get("emitted"):
                added = "S4" not in ctx.scenarios
                ctx.scenarios.add("S4")
                if added:
                    await send_and_log({
                        "event": "scenarios_update",
                        "demonstrated": sorted(ctx.scenarios),
                        "added_this_turn": ["S4"],
                        "turn": None,
                    })
        except Exception as e:
            log.warning("decision_trace_emit failed: %s", e)
            logger.add_error("decision_trace_emit", str(e))

    # Tier usage summary for this session
    tier_counts = getattr(right, "tier_counts", {})
    await send_and_log({"event": "session_tier_summary", "right_tier_counts": tier_counts})
    logger.add_note(f"tier_counts: {tier_counts}")
