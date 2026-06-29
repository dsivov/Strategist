"""Shared post-render gate chain — harness substrate (not engine code).

Currently wired only on the Planner path (the Strategist has its own
in-chain gates; double-gating would over-constrain). Two crisp mechanical
gates per §8.5 of ENGINE-COMPARISON.md, targeting roughly 27–29% of real
Ecommerce losses with deterministic detection:

  1. Anti-staircase — block a further price concession without an earned
     customer signal since the last offer.
  2. Premature-close — block ending the engagement without a CTA or
     scheduled follow-up, when the customer has not explicitly declined.

Engine-agnostic. Disabled via POC_PLANNER_GATES=off for control runs.
"""
from __future__ import annotations

import os
import re
from typing import Awaitable, Callable, Optional

# An agent OFFER (distinct from restating a customer number or anchor reference)
_OFFER_VERB_RE = re.compile(
    r"\b(i can (offer|do|give|drop to|come down to|go as low as)|"
    r"we can (offer|do|give|drop to|come down to)|"
    r"the (price|cost|best)( i can do)? (is|would be)|"
    r"i'?ll (give|offer|drop|do|come down to)|"
    r"how about|let me offer|i'?m able to (offer|do)|"
    r"approved (for|to)|i can get (you|that) (to|down to)|"
    r"final (price|offer)( is)?|absolute lowest|"
    r"set up for|renew(ed)? for|policy for)\b",
    re.IGNORECASE)

# Currency-number extractor
_NUM_RE = re.compile(
    r"\$\s*(\d[\d,]*(?:\.\d{1,2})?)|"
    r"(\d[\d,]*(?:\.\d{1,2})?)\s*(?:USD|USD|dollar|dollar)",
    re.IGNORECASE)

# Earned customer signal: competitor disclosure or explicit lower target.
_EARNED_RE = re.compile(
    r"competitor|another (company|quote|offer)|cheaper|got a quote|"
    r"\b\d[\d,]{2,}\b.*?(elsewhere|other)|"
    r"match (the|that|my) (\$?\d|price|quote|offer)|"
    r"i (got|found) (a|an) (better|lower) (offer|quote|price)",
    re.IGNORECASE)

# Close-ask / CTA / explicit question in the SAME agent message.
_CTA_RE = re.compile(
    r"\b(shall (i|we)|would you like|are you ready|ready to|let me send|"
    r"can i (send|share|process|set up|help you (complete|finalize))|"
    r"send (you )?the (link|code|details|invoice)|"
    r"finalize|check ?out|complete (your|the) (order|purchase|renewal)|"
    r"go ahead|use code|enter code|here'?s (the|your) (code|link))\b|"
    r"\?",
    re.IGNORECASE)

# Scheduled / time-bound follow-up commitment in the SAME agent message.
_FOLLOWUP_RE = re.compile(
    r"\b(i'?ll (follow up|check back|reach out|circle back|message|text|"
    r"call|email)( with you)? (on|in|by|next|tomorrow|tonight|this week)|"
    r"i'?ll be back (in|by|on)|"
    r"following up (on|in|by) (tomorrow|tonight|next \w+|\w+day))\b",
    re.IGNORECASE)

# Customer explicit decline — don't penalize the agent for accepting a hard no.
_DECLINE_RE = re.compile(
    r"\b(no (thanks|thank you)|not interested|i'?m (going to )?pass|"
    r"not (for|going to) (buy|order|get|do this)|i'?ll (have to )?(walk away|pass)|"
    r"(definite|absolute)ly not|never mind|stop (messaging|texting|contacting))\b",
    re.IGNORECASE)


def _extract_agent_offers(dialog) -> list[tuple[int, float]]:
    """[(turn_idx, price)] for each agent turn that contains an OFFER verb
    AND a price number. Restated-customer-numbers / anchor refs are skipped."""
    out: list[tuple[int, float]] = []
    for i, m in enumerate(dialog or []):
        if m.get("role") != "agent":
            continue
        txt = m.get("text") or ""
        if not _OFFER_VERB_RE.search(txt):
            continue
        nums: list[float] = []
        for g1, g2 in _NUM_RE.findall(txt):
            raw = (g1 or g2).replace(",", "")
            try:
                nums.append(float(raw))
            except ValueError:
                pass
        if nums:
            out.append((i, max(nums)))
    return out


def _earned_signal_after(dialog, after_idx: int) -> bool:
    """True if any customer turn strictly after `after_idx` contains an
    earned signal."""
    for i, m in enumerate(dialog or []):
        if i <= after_idx:
            continue
        if m.get("role") != "customer":
            continue
        if _EARNED_RE.search(m.get("text") or ""):
            return True
    return False


def _live_agent_turn_count(dialog) -> int:
    return sum(1 for m in (dialog or []) if m.get("role") == "agent")


def _last_customer_declined(dialog) -> bool:
    for m in reversed(dialog or []):
        if m.get("role") == "customer":
            return bool(_DECLINE_RE.search(m.get("text") or ""))
    return False


def _premature_close_anchor(text: str, tenant: Optional[str]) -> bool:
    if not text:
        return False
    try:
        from intent_classifier import intent_score, is_available
        if is_available():
            decision, _s, _a = intent_score(text, "agent_premature_close",
                                            tenant=tenant)
            return bool(decision)
    except Exception:
        pass
    return bool(re.search(
        r"\b(we'?ll be here|feel free to (reach out|come back)|"
        r"no (rush|pressure)|whenever you'?re ready|when you'?re ready|"
        r"just (reach out|let us know)|take your time|happy to wait)\b",
        text, re.IGNORECASE))


# ── individual gates ──────────────────────────────────────────────────────
def gate_staircase(dialog, agent_text, opp_meta) -> Optional[str]:
    """Returns a corrective string if the agent text is a staircase
    concession without an earned signal; else None."""
    if not _OFFER_VERB_RE.search(agent_text or ""):
        return None  # the agent isn't offering a price → not a staircase
    new_nums: list[float] = []
    for g1, g2 in _NUM_RE.findall(agent_text or ""):
        raw = (g1 or g2).replace(",", "")
        try:
            new_nums.append(float(raw))
        except ValueError:
            pass
    if not new_nums:
        return None
    new_offer = max(new_nums)
    prior = _extract_agent_offers(dialog)
    if not prior:
        return None
    last_offer_idx, _ = prior[-1]
    prior_max = max(p for _, p in prior)
    if new_offer >= prior_max:
        return None  # holding or raising; not a staircase
    if _earned_signal_after(dialog, last_offer_idx):
        return None
    return (f"Do not concede below {prior_max:g} — you have already offered "
            f"{prior_max:g} and the customer has not disclosed a competitor "
            f"number or an explicit lower target since. Hold that price and "
            f"probe for the customer's specific number before any further "
            f"drop.")


def gate_premature_close(dialog, agent_text, opp_meta) -> Optional[str]:
    """Returns a corrective string if the agent text ends the engagement
    without a CTA / scheduled follow-up at a still-recoverable point."""
    if not _premature_close_anchor(agent_text,
                                   (opp_meta or {}).get("company")):
        return None
    if _live_agent_turn_count(dialog) < 3:
        return None
    if _last_customer_declined(dialog):
        return None
    if _CTA_RE.search(agent_text or ""):
        return None
    if _FOLLOWUP_RE.search(agent_text or ""):
        return None
    return ("Do not end the engagement here. The customer has not declined; "
            "instead, do exactly one of: (a) make a close-ask now (specific "
            "CTA + a question), (b) commit to a specific follow-up time "
            "(e.g., 'I'll check back tomorrow morning'), or (c) probe a "
            "remaining objection. Do not say 'no rush' / 'we'll be here' / "
            "'feel free to reach out' without one of these.")


_GATES = [("staircase", gate_staircase),
          ("premature_close", gate_premature_close)]
MAX_RETRIES = 2


def enabled() -> bool:
    return os.environ.get("POC_PLANNER_GATES", "on").lower() in (
        "on", "1", "true", "yes")


async def apply(dialog, opp_meta, agent_text,
                regen_fn: Callable[[str], Awaitable[str]]):
    """Run the gate chain on `agent_text`. On any gate violation, call
    `regen_fn(corrective)` and re-evaluate. Cap = MAX_RETRIES; then
    templated fallback. Returns (final_text, gates_meta).

    When POC_PLANNER_GATES is off, returns text unchanged (control mode)."""
    if not enabled():
        return agent_text, {"gates_applied": False}
    fired: list[dict] = []
    regens = 0
    text = agent_text
    for attempt in range(MAX_RETRIES + 1):
        violated: Optional[tuple[str, str]] = None
        for name, gate in _GATES:
            corrective = gate(dialog, text, opp_meta)
            if corrective:
                violated = (name, corrective)
                fired.append({"gate": name, "attempt": attempt,
                              "will_regen": attempt < MAX_RETRIES})
                break
        if not violated:
            return text, {"gates_applied": True, "gates_fired": fired,
                          "gates_regens": regens, "gates_final": "passed"}
        if attempt < MAX_RETRIES:
            try:
                text = await regen_fn(violated[1])
                regens += 1
            except Exception as e:
                return text, {"gates_applied": True, "gates_fired": fired,
                              "gates_regens": regens,
                              "gates_final": f"regen_err:{type(e).__name__}"}
            continue
        # exhausted retries → templated fallback for the still-violating gate
        if violated[0] == "staircase":
            prior = _extract_agent_offers(dialog)
            prior_max = max(p for _, p in prior) if prior else None
            tail = (f" I can hold {prior_max:g} for you. "
                    if prior_max else " ")
            text = (f"That's actually the best I can do right now.{tail}"
                    f"What number would work on your side?")
        else:  # premature_close
            text = (text.rstrip(".!? ") +
                    " — would you like me to go ahead and send the link "
                    "now, or shall I check back with you tomorrow?")
        fired.append({"gate": violated[0], "attempt": attempt,
                      "fallback": True})
        return text, {"gates_applied": True, "gates_fired": fired,
                      "gates_regens": regens, "gates_final": "fallback"}
    return text, {"gates_applied": True, "gates_fired": fired,
                  "gates_regens": regens, "gates_final": "exhausted"}
