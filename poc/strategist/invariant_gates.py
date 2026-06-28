"""R5 — Additional hard-invariant gates (2026-05-04).

Today the supervisor pipeline has TWO mechanical regenerate-loop guards:
  T-83 anti-staircase  — blocks successive price drops
  T-87 anti-capitulation — blocks endorsing competitor / apologizing for our offer

R5 adds four more invariants observed empirically (Day-4 cases + live sessions):

  R5.1 max-discount guard       — agent went below tenant's max_discount_pct cap
  R5.2 no-disparagement guard   — agent spoke ill of a competitor
  R5.3 no-fabricated-stats guard — agent cited a number not grounded in anchors
  R5.4 regulatory-language guard — agent used a forbidden insurance/financial term

All four follow the same architecture as staircase / capitulation:
  1. detect violation in candidate text (post build_answer, pre dedup)
  2. emit a corrective system_suffix
  3. trigger regenerate-loop in the chain runner

Aggregated entry point: `check_invariants(candidate_text, dialog, opp_meta)`
returns `(violated_bool, meta)`. The chain runner consumes meta to decide
whether to fire the regenerate-loop and to build the correction prompt.

Empirical motivation per sub-check is in the docstrings.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


# ── R5.2 — no-disparagement patterns ────────────────────────────────────────
# Day-4 evidence: cases 5/8 had real Libra agents saying "Phoenix is missing
# headlights coverage" — that's NOT disparagement (it's factual differentiation).
# Disparagement is opinion-laden: "Phoenix is unreliable / shady / a bad choice".
# These patterns lean conservative — only flag clearly negative judgment language.
_DISPARAGEMENT_PATTERNS = [
    re.compile(
        r"\b(?:phoenix|yashir|wesure|direct|menorah|harel|clal|migdal|ayalon)\b"
        r".{0,40}\b(?:is|are|sounds|seems)\b.{0,20}"
        r"\b(?:bad|terrible|awful|unreliable|untrustworthy|shady|sketchy|dishonest|crooked|cheap(?:ly)?\s*made)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:avoid|stay\s+away\s+from|don'?t\s+trust)\s+(?:phoenix|yashir|wesure|direct|menorah|harel|clal|migdal|ayalon)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:they|their\s+(?:company|service|claims))\s+(?:cut\s+corners|don'?t\s+pay|won'?t\s+honor|will\s+(?:reject|deny)|always\s+(?:reject|deny))",
        re.IGNORECASE,
    ),
]


def _check_disparagement(text: str) -> dict | None:
    for pat in _DISPARAGEMENT_PATTERNS:
        m = pat.search(text)
        if m:
            return {
                "type": "disparagement",
                "matched": m.group(0)[:120],
            }
    return None


# ── R5.3 — no-fabricated-statistics patterns ────────────────────────────────
# Day-4 evidence: case 7 (abda27f1) had real Libra agent saying "Prices have
# gone up across the entire insurance industry, there was a tariff update,
# raising prices by up to 48 percent." The anchor's actual_market_yoy_change_pct
# was much lower (~10-15%). Customer reaction: "Hahaha" — credibility blown.
#
# Heuristic: if candidate contains a specific %-claim about industry-wide
# tariff/increase/inflation AND the anchor's actual market change doesn't
# support it (or anchors empty), flag. Threshold: claim > anchor + 10pp.
_TARIFF_PCT_PATTERN = re.compile(
    r"\b(\d{1,3})\s*(?:%|percent)\b.{0,40}\b(?:tariff|industry-?wide|across\s+the\s+(?:entire\s+)?industry|inflation|market(?:-?wide)?\s+(?:increase|rise))",
    re.IGNORECASE,
)
# Order: keyword first ("industry-wide tariff update raising prices by N%"),
# then any increase/raise verb form, then the % number.
_GENERIC_PCT_INCREASE_PATTERN = re.compile(
    r"\b(?:industry|market|tariff)[\w\s,-]{0,40}"
    r"\b(?:increase|increasing|increased|rise|rising|raise|raising|raised|jump|jumping|hike|hiking|going\s+up|gone\s+up)\b"
    r"[\w\s,]{0,40}\b(\d{1,3})\s*(?:%|percent)",
    re.IGNORECASE,
)
# Reverse-order pattern: "[verb] [prices] by N% ... [industry/tariff/market]"
_REVERSE_PCT_INCREASE_PATTERN = re.compile(
    r"\b(?:increase|increasing|increased|rising|raise|raising|raised|jump|hike|going\s+up|gone\s+up)\b"
    r"[\w\s,]{0,30}\b(\d{1,3})\s*(?:%|percent)\b"
    r"[\w\s,]{0,80}\b(?:tariff|industry|market|inflation)\b",
    re.IGNORECASE,
)


def _check_fabricated_statistic(text: str, anchors: dict | None) -> dict | None:
    actual = (anchors or {}).get("actual_market_yoy_change_pct")
    threshold_pp = 10  # claims more than 10pp above actual = fabrication
    for pat in (_TARIFF_PCT_PATTERN, _GENERIC_PCT_INCREASE_PATTERN, _REVERSE_PCT_INCREASE_PATTERN):
        m = pat.search(text)
        if not m:
            continue
        try:
            claimed = int(m.group(1))
        except (ValueError, IndexError):
            continue
        # If we have an actual market figure, check whether the claim exceeds it
        if isinstance(actual, (int, float)):
            if claimed > actual + threshold_pp:
                return {
                    "type": "fabricated_statistic",
                    "claimed_pct": claimed,
                    "actual_pct": actual,
                    "matched": m.group(0)[:120],
                }
        else:
            # No actual figure available — flag any specific industry-wide
            # tariff claim above 20% as suspicious
            if claimed > 20:
                return {
                    "type": "fabricated_statistic",
                    "claimed_pct": claimed,
                    "actual_pct": None,
                    "matched": m.group(0)[:120],
                    "reason": "no anchor for verification; claim > 20%",
                }
    return None


# ── R5.4 — regulatory-language guard ────────────────────────────────────────
# Israeli insurance regulations (and most insurance regs globally) forbid
# unqualified absolute claims. POC-scope: a small starter set; productionization
# would source from each tenant's compliance team.
_REGULATORY_FORBIDDEN_PATTERNS = [
    re.compile(r"\b(?:guaranteed|100%\s+guarantee|guaranteed\s+(?:claim|payout|return))", re.IGNORECASE),
    re.compile(r"\blifetime\s+(?:guarantee|warranty|coverage)\b", re.IGNORECASE),
    re.compile(r"\b(?:free|no-?cost|zero-?cost)\s+(?:insurance|coverage|protection)\b", re.IGNORECASE),
    re.compile(r"\bbest\s+(?:in\s+the\s+)?(?:industry|market|country)\b", re.IGNORECASE),
    re.compile(r"\b(?:cheapest|lowest)\s+(?:in\s+the\s+)?(?:market|country|industry)\b", re.IGNORECASE),
]


def _check_regulatory(text: str) -> dict | None:
    for pat in _REGULATORY_FORBIDDEN_PATTERNS:
        m = pat.search(text)
        if m:
            return {
                "type": "regulatory_language",
                "matched": m.group(0)[:120],
            }
    return None


# ── R5.1 — max-discount-pct guard ───────────────────────────────────────────
# Anchor pack carries `max_discount_pct_internal` (Libra) or
# `max_authorized_discount_pct_internal` (Heavys). If the candidate's quoted
# price implies a discount > cap from `current_quoted_price_nis` (Libra) or
# from the catalog price (Heavys), that's a violation.
#
# Conservative pattern: extract the largest 4-digit price-like number from
# candidate, compare to current_quote * (1 - max_discount). False positives
# (numbers that aren't prices) handled by the threshold tolerance.
_PRICE_NUMBER_PATTERN = re.compile(r"\b(\d{3,4})\s*(?:NIS|nis|ש[\"״]?ח|shekels?|\$|USD)?", re.IGNORECASE)


def _check_max_discount(text: str, anchors: dict | None) -> dict | None:
    if not anchors:
        return None
    current_quote = (
        anchors.get("current_quoted_price_nis")
        or anchors.get("current_quoted_price")
        or anchors.get("catalog_price")
    )
    max_pct = (
        anchors.get("max_discount_pct_internal")
        or anchors.get("max_authorized_discount_pct_internal")
    )
    if not (isinstance(current_quote, (int, float)) and current_quote > 0):
        return None
    if not (isinstance(max_pct, (int, float)) and max_pct > 0):
        return None
    floor = current_quote * (1 - max_pct / 100.0)
    # Find lowest price-like number in candidate that's above some sanity floor
    sanity_min = current_quote * 0.4  # don't flag mentions of 100 NIS deductibles etc.
    candidates = []
    for m in _PRICE_NUMBER_PATTERN.finditer(text):
        try:
            v = int(m.group(1))
        except ValueError:
            continue
        if sanity_min <= v < current_quote:
            candidates.append(v)
    if not candidates:
        return None
    lowest = min(candidates)
    if lowest < floor:
        # Drop is steeper than the cap allows
        actual_drop_pct = 100.0 * (current_quote - lowest) / current_quote
        return {
            "type": "max_discount_exceeded",
            "lowest_quoted": lowest,
            "current_quote": current_quote,
            "max_pct_internal": max_pct,
            "actual_drop_pct": round(actual_drop_pct, 1),
            "floor": round(floor, 0),
        }
    return None


# ── Aggregator ──────────────────────────────────────────────────────────────


def check_invariants(candidate_text: str, dialog: list[dict], opp_meta: dict) -> tuple[bool, dict]:
    """Run all four R5 sub-checks against the candidate. Return whether any
    fired plus a meta dict with all violations + the canonical correction
    framing for the first violation (the loop will re-check after regen)."""
    if not candidate_text or not isinstance(candidate_text, str):
        return False, {"violations": []}
    anchors = opp_meta.get("anchors") if isinstance(opp_meta, dict) else None
    violations = []
    for fn, name in (
        (lambda t: _check_max_discount(t, anchors), "max_discount"),
        (_check_disparagement, "disparagement"),
        (lambda t: _check_fabricated_statistic(t, anchors), "fabricated_statistic"),
        (_check_regulatory, "regulatory"),
    ):
        try:
            v = fn(candidate_text)
        except Exception as e:
            log.warning("invariant gate %s raised %s — skipping", name, e)
            continue
        if v:
            violations.append(v)
    return bool(violations), {"violations": violations}


# ── Correction prompt builder ───────────────────────────────────────────────


def build_invariants_correction(meta: dict) -> str:
    """Build a system-suffix string for the regenerate-loop. Tells the
    supervisor's build_answer LLM exactly which invariant fired and how to
    avoid the violation on regeneration."""
    violations = meta.get("violations") or []
    if not violations:
        return ""
    lines = ["", "---", "## ⚠ INVARIANT GUARD CORRECTION"]
    for v in violations:
        t = v.get("type")
        if t == "max_discount_exceeded":
            lines.append(
                f"- The proposed price {v.get('lowest_quoted')} is below the floor "
                f"{v.get('floor')} (current quote {v.get('current_quote')}, "
                f"max internal discount {v.get('max_pct_internal')}%, "
                f"actual proposed drop {v.get('actual_drop_pct')}%). "
                f"Re-emit with a price ≥ {v.get('floor')}."
            )
        elif t == "disparagement":
            lines.append(
                f"- Disparaging language about a competitor detected: "
                f"`{v.get('matched')}`. Re-emit with FACTUAL coverage "
                f"differentiation only — no opinion adjectives about the competitor."
            )
        elif t == "fabricated_statistic":
            actual = v.get("actual_pct")
            actual_s = f"actual market YoY is {actual}%" if actual is not None else "no anchor data supports it"
            lines.append(
                f"- A specific {v.get('claimed_pct')}% industry / tariff / market "
                f"figure was cited ({v.get('matched')}). This is not grounded — "
                f"{actual_s}. Re-emit WITHOUT specific %-claims about industry trends; "
                f"either drop the claim or quote only what's in the anchor pack."
            )
        elif t == "regulatory_language":
            lines.append(
                f"- Forbidden regulatory phrasing: `{v.get('matched')}`. "
                f"Re-emit without absolute / superlative claims (avoid "
                f"`guaranteed`, `lifetime`, `free insurance`, `best/cheapest in market`)."
            )
        else:
            lines.append(f"- Unknown invariant violation: {v}")
    lines.append("")
    lines.append("Keep your tone, strategy, and Tier-2 move EXACTLY the same — "
                  "only fix the specific violation(s) above.")
    return "\n".join(lines)
