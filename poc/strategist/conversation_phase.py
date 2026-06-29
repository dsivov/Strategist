"""Conversation-phase tracker (rule-based, deterministic).

Implements multi-turn arc awareness per design Q1–Q7 (locked 2026-05-03):

- 8 phases: greet, probe, present, objection_handling, close_attempt,
  commit_pending, retreat, won (+ lost terminal).
- SUGGEST soft binding (output is context for Mode 1b, not a filter).
- Pure rule-based — no LLM call (target <1ms per turn).
- Free transitions allowed, with N=2 dwell-time hysteresis on phase REVERSALS
  (retreat ↔ any is excluded from hysteresis — customer-explicit).

Validated against the 8-case Day-4 phase baseline
(`research-notes/2026-05-03-day4-phase-baseline.md`); target ≥80% agreement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

PHASE_GREET = "greet"
PHASE_PROBE = "probe"
PHASE_PRESENT = "present"
PHASE_OBJECTION_HANDLING = "objection_handling"
PHASE_CLOSE_ATTEMPT = "close_attempt"
PHASE_COMMIT_PENDING = "commit_pending"
PHASE_RETREAT = "retreat"
PHASE_WON = "won"
PHASE_LOST = "lost"

ALL_PHASES = [
    PHASE_GREET,
    PHASE_PROBE,
    PHASE_PRESENT,
    PHASE_OBJECTION_HANDLING,
    PHASE_CLOSE_ATTEMPT,
    PHASE_COMMIT_PENDING,
    PHASE_RETREAT,
    PHASE_WON,
    PHASE_LOST,
]

# Canonical forward order for "monotonic progress" detection.
# Index in this list determines forward (immediate-flip) vs backward (N=2 hysteresis).
# Off-track phases (probe, present, retreat, lost) are diagnostic/side-track and
# bypass hysteresis — going to them is never "backward flicker." Won is a forward
# terminal that always flips immediately.
_FORWARD_ORDER = [
    PHASE_GREET,
    PHASE_OBJECTION_HANDLING,
    PHASE_CLOSE_ATTEMPT,
    PHASE_COMMIT_PENDING,
    PHASE_WON,
]

_HYSTERESIS_THRESHOLD = 2  # turns of disagreement before backward flip


# ---------------------------------------------------------------------------
# Cue patterns (deterministic regex)
# ---------------------------------------------------------------------------

# Customer retreat — explicit pause / declination
_RETREAT_PATTERNS = [
    re.compile(r"\bcheck (the )?market\b", re.I),
    re.compile(r"\bcall (me )?(back )?(on )?(sunday|monday|tuesday|wednesday|thursday|friday|saturday|tomorrow|later|next week)\b", re.I),
    re.compile(r"\bget back to (you|me)\b", re.I),
    re.compile(r"\bhusband (will |would )?decide", re.I),
    re.compile(r"\bwife (will |would )?decide", re.I),
    re.compile(r"\b(need|let me) (to )?think", re.I),
    re.compile(r"\bnot (interested|now|today)\b", re.I),
    re.compile(r"\bdon'?t renew\b", re.I),
    re.compile(r"^\s*no\s*[.!]?\s*$", re.I),  # standalone "no"
    re.compile(r"\btake your time\b", re.I),  # agent honoring retreat
    re.compile(r"\bno worries\b.*\b(time|check)\b", re.I),
    # US English / Ecommerce vocabulary (added 2026-05-04 after live-session gap)
    re.compile(r"\bno thank you\b", re.I),
    re.compile(r"\bappreciate (the |your )?offer\b.*\b(but|however)\b", re.I),
    re.compile(r"\b(i'?ll |i would )?(have to )?pass\b", re.I),
    re.compile(r"\bnot (?:at|right) (?:this|the) time\b", re.I),
    re.compile(r"\bkeep me posted\b", re.I),
    re.compile(r"\b(let me know|ping me) (when|if)\b.*\b(future|later|next|drops?)\b", re.I),
    re.compile(r"\b(holding out|waiting for|hold(?:ing)? off)\b.*\b(better|next|future|deal|discount|sale|drop)\b", re.I),
    re.compile(r"\bmaybe (next|another|later) (time|month|week|year)\b", re.I),
]

# Customer objection — price / competitor / coverage concern
_OBJECTION_PATTERNS = [
    re.compile(r"\b(too |very |really )?expensive\b", re.I),
    re.compile(r"\btoo (high|much)\b", re.I),
    re.compile(r"\bis (there|this) (a )?discount\b", re.I),
    re.compile(r"\b(improve|lower|reduce) the price\b", re.I),
    re.compile(r"\bcheapest (offer|price|you)", re.I),
    re.compile(r"\b(comparing|compared) (other|to)\b", re.I),
    re.compile(r"\b(found|received|have|got) (a |an )?(quote|insurance|offer|price) (from|with|of|for)\b", re.I),
    re.compile(r"\b(phoenix|yashir|wesure|direct|menorah|harel|clal|migdal|ayalon)\b", re.I),
    re.compile(r"\b\d{2,4}\s*(dollars?|usd|ש[\"״]?ח)\s*(cheaper|less|lower)\b", re.I),
    re.compile(r"\b(loyal|long(time|-time)) customer\b", re.I),
    re.compile(r"\brefund\b", re.I),
    re.compile(r"\badditional discount\b", re.I),
    # US English / Ecommerce vocabulary (added 2026-05-04 after live-session gap)
    re.compile(r"\bhow much (?:of |is )(?:a |the )?discount\b", re.I),
    re.compile(r"\b(any|got any|got a|is there) (a )?(discount|promo|coupon|deal)\b", re.I),
    re.compile(r"\bwhat'?s the (discount|deal|lowest|best price)\b", re.I),
    re.compile(r"\b(best|lowest) (price|you can (do|go))\b", re.I),
    re.compile(r"\bfiscally (responsible|justifiable)\b", re.I),
    re.compile(r"\b(can'?t|cannot) (justify|afford)\b", re.I),
    re.compile(r"\b(out of|over) (my )?budget\b", re.I),
    re.compile(r"\bstretch(ing)? (my |the )?budget\b", re.I),
    re.compile(r"\b\$\d{2,4}\s*(off|discount|cheaper|less)\b", re.I),  # "$80 off discount"
    re.compile(r"\b(holiday|black\s*friday|cyber\s*monday|christmas|spring|summer|holiday) (deal|sale|discount|offer)\b", re.I),
]

# Probe — diagnostic question (agent or customer)
_PROBE_PATTERNS = [
    re.compile(r"\b(can|may) i ask\b.*\?", re.I),
    re.compile(r"\b(why not|why)\??$", re.I),
    re.compile(r"\bcompared to what\??", re.I),
    re.compile(r"\bdid you check\b", re.I),
    re.compile(r"\bcheck (the )?(exact )?coverage\b", re.I),
    re.compile(r"\bhow much (did|do|was|were) (we|i|you)\b", re.I),
    re.compile(r"\bdetail\b.*\b(last year|previous|prior)", re.I),
    re.compile(r"\bwhich (company|insurer)\b", re.I),
]

# Close attempt — agent commitment ask
_CLOSE_ATTEMPT_PATTERNS = [
    re.compile(r"\bif (i |we |they )?(can |manage|reach|match|reduce|lower|approve|find)\b.*\b(can|could|will|would) (we|you)\b.*\?", re.I),
    re.compile(r"\bcan we (proceed|close|move forward|renew|finalize|secure)\b", re.I),
    re.compile(r"\bshall we (proceed|go ahead|close|renew|finalize|sort)\b", re.I),
    re.compile(r"\b(will|can) we be able to renew\b", re.I),
    re.compile(r"\bclose the deal\b", re.I),
    re.compile(r"\b(would|will) that be relevant for renewal\b", re.I),
    re.compile(r"\bproceed with (this|the) (price|offer)\b", re.I),
    re.compile(r"\b(secure|finalize|close) the (renewal|deal|policy|sale)\b", re.I),
    # Customer counter-price acceptance
    re.compile(r"^\s*\d{3,4}\s*,?\s*and (we can|let'?s) (proceed|renew|close)\b", re.I),
    re.compile(r"^\s*(yes|yeah|ok|okay|sure|great|let'?s go|sounds (like a plan|good))\s*[.!]?\s*$", re.I),
]

# Commit pending — payment-mechanics
_COMMIT_PENDING_PATTERNS = [
    re.compile(r"\bhow many (installments?|payments?)\b", re.I),
    re.compile(r"\blast 4 (digits?|of the credit card)", re.I),
    re.compile(r"\b\d{1,2} (interest-?free )?(installments?|payments?)\b", re.I),
    re.compile(r"\b\d{4}\s*$"),  # bare 4-digit card-tail
    re.compile(r"\bsame (terms|coverage|policy) (as|that)\b", re.I),
    re.compile(r"\b(send|please send) (the |me )?(terms|policy|details)\b", re.I),
    re.compile(r"\bcard (number|ending)\b", re.I),
]

# Won — renewal confirmation
_WON_PATTERNS = [
    re.compile(r"\b(we|i) will proceed with (the )?renewal\b", re.I),
    re.compile(r"\b(policy|insurance) (is )?(valid|continues|starting|renewed)\b.*\b(year|01/|05/|2026|2027)\b", re.I),
    re.compile(r"\barrive (via )?(by )?email\b", re.I),
    re.compile(r"\bdocuments? will arrive\b", re.I),
    re.compile(r"\bwishing you (a )?safe\b", re.I),
    re.compile(r"\b(have a |wishing you a )(great |nice |safe |pleasant )?(day|drive|year|weekend)\b", re.I),
    re.compile(r"\bthank you (very much|so much)\b", re.I),  # customer post-close
]

# Greet — opening
_GREET_PATTERNS = [
    re.compile(r"\bhi\b", re.I),
    re.compile(r"\bhello\b", re.I),
    re.compile(r"\bhey\b", re.I),
    re.compile(r"\bgood (morning|afternoon|evening)\b", re.I),
    re.compile(r"^היי\b", re.I),
    re.compile(r"\bthank you for contacting\b", re.I),
    re.compile(r"\bi will get back\b", re.I),  # customer auto-reply
]


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class PhaseState:
    """Per-conversation state — caller passes this back in each turn."""
    current_phase: str = PHASE_GREET
    turns_in_phase: int = 0
    candidate_phase: Optional[str] = None
    candidate_streak: int = 0
    history: List[str] = field(default_factory=list)  # phase per turn

    def to_dict(self) -> dict:
        return {
            "current_phase": self.current_phase,
            "turns_in_phase": self.turns_in_phase,
            "candidate_phase": self.candidate_phase,
            "candidate_streak": self.candidate_streak,
            "history": list(self.history),
        }


# ---------------------------------------------------------------------------
# Single-turn cue classifier (no hysteresis yet — pure cue → phase)
# ---------------------------------------------------------------------------

def _matches_any(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def _classify_cue(text: str, role: str, prev_phase: str, turn_idx: int) -> Optional[str]:
    """Return the phase suggested by this single turn's cues, or None to hold."""
    if turn_idx == 0:
        return PHASE_GREET

    t = (text or "").strip()
    if not t:
        return None

    # Order matters: check most specific cues first.

    # Customer auto-reply ("thank you for contacting... will get back") = greet,
    # NOT retreat. Special-case before retreat patterns.
    if role == "customer" and re.search(r"\bthank you for contacting\b", t, re.I) and prev_phase == PHASE_GREET:
        return PHASE_GREET

    # Won cues — terminal renewal confirmation (agent or customer post-close).
    if _matches_any(t, _WON_PATTERNS):
        if prev_phase in (PHASE_COMMIT_PENDING, PHASE_CLOSE_ATTEMPT, PHASE_WON):
            return PHASE_WON

    # Agent compound "no worries...take your time...BUT if I manage discount can we renew?"
    # is dominantly a close attempt despite the lead-in retreat-honor.
    if role == "agent":
        has_retreat_honor = bool(re.search(r"\b(no worries|take your time)\b", t, re.I))
        has_conditional_close = bool(re.search(r"\bif (i |we )?(can |manage|reach|find)\b.*\b(can|could|will|would) we\b.*\?", t, re.I))
        if has_retreat_honor and has_conditional_close:
            return PHASE_CLOSE_ATTEMPT

    # Retreat cues — customer-explicit pause; agent honoring retreat.
    if role == "customer" and _matches_any(t, _RETREAT_PATTERNS):
        return PHASE_RETREAT
    if role == "agent" and re.search(r"\bno worries.*(time|check)\b", t, re.I):
        return PHASE_RETREAT

    # Strong commit_pending cues — explicit "how many installments + last 4" or
    # customer-providing-card-tail pattern. Require prev phase be close-zone OR
    # commit-zone to avoid false-positives from value-prop mentions of
    # "10 interest-free payments" inside an objection-handling rebuttal.
    in_commit_zone = prev_phase in (PHASE_CLOSE_ATTEMPT, PHASE_COMMIT_PENDING)

    explicit_commit_ask = bool(re.search(r"\bhow many (installments?|payments?)\b", t, re.I) and re.search(r"\blast 4\b|\bdigits?\b", t, re.I))
    customer_card_tail = (
        role == "customer"
        and (
            re.search(r"\b\d{1,2}\s*(installments?|payments?)\b.*\b\d{3,4}\b", t, re.I)
            or re.search(r"\blast 4\b.*\b\d{3,4}\b", t, re.I)
        )
    )
    if explicit_commit_ask or customer_card_tail:
        return PHASE_COMMIT_PENDING

    if in_commit_zone and _matches_any(t, _COMMIT_PENDING_PATTERNS):
        return PHASE_COMMIT_PENDING

    # Probe BEFORE close_attempt and OH for agent diagnostic questions.
    # ("Expensive compared to what? Did you check other offers?" → probe, not OH.)
    if role == "agent" and _matches_any(t, _PROBE_PATTERNS):
        return PHASE_PROBE

    # Close attempt cues — agent-commitment-ask OR customer counter-price.
    if _matches_any(t, _CLOSE_ATTEMPT_PATTERNS):
        if re.match(r"^\s*(yes|yeah|ok|okay|sure|great|let'?s go|sounds)\s*[.!]?\s*$", t, re.I):
            if prev_phase in (PHASE_CLOSE_ATTEMPT, PHASE_OBJECTION_HANDLING):
                return PHASE_CLOSE_ATTEMPT
            return None
        return PHASE_CLOSE_ATTEMPT

    # Probe answer continuation: SHORT customer reply (≤4 words) when prev was
    # probe → stay in probe. Long messages — even if they mention a competitor
    # name — are not probe-answers; they're OH (mid-message objection).
    if role == "customer" and prev_phase == PHASE_PROBE:
        word_count = len(t.split())
        if word_count <= 4:
            return PHASE_PROBE

    # Probe — customer-side diagnostic question.
    if role == "customer" and _matches_any(t, _PROBE_PATTERNS):
        return PHASE_PROBE

    # Objection handling — customer concern. Restrict "expensive"-style cues to
    # customer role (agent quoting "expensive compared to what?" is a probe, not OH).
    if role == "customer" and _matches_any(t, _OBJECTION_PATTERNS):
        return PHASE_OBJECTION_HANDLING

    # Agent rebuttal heuristic — defending price/competitor stance after a
    # customer objection.
    if role == "agent" and prev_phase == PHASE_OBJECTION_HANDLING:
        if re.search(r"\b(no|cannot|can'?t)\b.*\b(refund|discount|lower|reduce)\b", t, re.I):
            return PHASE_OBJECTION_HANDLING
        if re.search(r"\b(market check|cheap|excellent price|won'?t find|competitive|significant difference)\b", t, re.I):
            return PHASE_OBJECTION_HANDLING
        # Agent challenge-coverage / value-differentiation rebuttal.
        if re.search(r"\b(includes|automatically receive|insurance (you|automatically))\b.*\b(coverage|headlights|hit and run|mirrors)\b", t, re.I):
            return PHASE_OBJECTION_HANDLING

    # Greet — first agent intro / customer auto-reply.
    if _matches_any(t, _GREET_PATTERNS) and prev_phase == PHASE_GREET:
        return PHASE_GREET

    # Agent factual answer to a probe (present).
    if role == "agent" and prev_phase == PHASE_PROBE:
        if re.search(r"^\s*\d{3,4}\s*(USD|usd|ש[\"״]?ח|total|dollars?)?\s*\.?\s*$", t):
            return PHASE_PRESENT
        if re.search(r"^(third party|mandatory|comp(rehensive)?|tp)\b", t, re.I) and re.search(r"\b\d{3,4}\b", t) and not _matches_any(t, _CLOSE_ATTEMPT_PATTERNS):
            return PHASE_PRESENT

    return None  # hold previous phase


# ---------------------------------------------------------------------------
# Hysteresis-aware update
# ---------------------------------------------------------------------------

def _is_forward(prev: str, new: str) -> bool:
    """True iff `new` is later than `prev` along the canonical sequence."""
    if prev not in _FORWARD_ORDER or new not in _FORWARD_ORDER:
        return True  # off-track phases (retreat/lost) bypass hysteresis
    return _FORWARD_ORDER.index(new) >= _FORWARD_ORDER.index(prev)


def update_phase(state: PhaseState, text: str, role: str, turn_idx: int) -> PhaseState:
    """Advance the phase tracker by one turn.

    Args:
        state: prior PhaseState (mutated and returned).
        text: this turn's message text.
        role: "agent" or "customer".
        turn_idx: 0-based turn index in the conversation.

    Returns:
        The same `state`, updated.
    """
    cue = _classify_cue(text, role, state.current_phase, turn_idx)

    if cue is None or cue == state.current_phase:
        # Hold — reset any pending candidate flip.
        state.turns_in_phase += 1
        state.candidate_phase = None
        state.candidate_streak = 0
        state.history.append(state.current_phase)
        return state

    # New cue suggests a different phase.
    forward = _is_forward(state.current_phase, cue)
    is_explicit_terminal = cue in (PHASE_RETREAT, PHASE_LOST, PHASE_WON)

    if forward or is_explicit_terminal:
        # Immediate flip.
        state.current_phase = cue
        state.turns_in_phase = 1
        state.candidate_phase = None
        state.candidate_streak = 0
    else:
        # Backward — apply hysteresis.
        if state.candidate_phase == cue:
            state.candidate_streak += 1
        else:
            state.candidate_phase = cue
            state.candidate_streak = 1

        if state.candidate_streak >= _HYSTERESIS_THRESHOLD:
            state.current_phase = cue
            state.turns_in_phase = 1
            state.candidate_phase = None
            state.candidate_streak = 0
        else:
            state.turns_in_phase += 1

    state.history.append(state.current_phase)
    return state


# ---------------------------------------------------------------------------
# Convenience: classify whole dialog at once
# ---------------------------------------------------------------------------

def classify_dialog(turns: List[dict]) -> List[str]:
    """Return a phase per turn for an entire dialog.

    Args:
        turns: list of {"role": "agent"|"customer", "text": str}.

    Returns:
        List of phase labels, same length as `turns`.
    """
    state = PhaseState(current_phase=PHASE_GREET)
    for i, turn in enumerate(turns):
        update_phase(state, turn.get("text", ""), turn.get("role", ""), i)
    return list(state.history)


def render_phase_block(state: PhaseState, cluster_plan_phase: Optional[str] = None) -> str:
    """Render the phase context block for Mode 1b prompt injection (Step 3)."""
    lines = ["## Conversation Arc"]
    lines.append(f"- dynamic_phase: {state.current_phase}")
    lines.append(f"- turns_in_phase: {state.turns_in_phase}")
    if cluster_plan_phase:
        agree = cluster_plan_phase == state.current_phase
        lines.append(f"- cluster_plan_phase: {cluster_plan_phase} ({'agrees' if agree else 'DISAGREES'})")
    lines.append("- semantics: SUGGEST — context for strategy, not a hard filter")
    return "\n".join(lines)
