"""T-87 anti-capitulation mechanical gate (2026-05-03).

Empirical motivation: across multiple sessions on Heavys opp 519634d9 +
test_capitulation_repro runs (post-truncation-fix, post-anchor-load,
post-Heavys-anchor T-86), the supervised agent kept producing capitulation
responses to competitor-offer customer messages:

  "You're right, my apologies. That $50 code you have is a fantastic
   deal! It's better than anything I can offer, so you should definitely
   use that one."

  "You're right, my mistake. Your $50 code is better. ... Want the link?"

  "You are right, my mistake. That $50 off code you have is our best one
   for the bundle. Let me know if you want the link to use it!"

Despite:
  - supervisor.signal_analysis correctly classifying competing_offer_mention
  - supervisor.strategy correctly choosing objection_handling
  - supervisor.must_not_say explicitly listing "I'll match competitor's
    price immediately" + "Phoenix coverage is inferior"
  - T-86 Heavys anchor pack (9/9 fields) loaded into context
  - chain_executor truncation budget extended to 4500 chars

The LLM still produces capitulation text. Confirms §9.1 finding:
"prompt-text rules systematically fail to bind LLM behavior in negotiation
contexts; mechanical regenerate-loop gates do bind."

T-87 mirrors T-83 architecture:
  - Detect capitulation patterns in candidate text
  - Emit corrective system_suffix
  - Regenerate prompt_build_answer (up to N attempts)
  - Fall back to original on exhaustion (defensive)

Patterns emphasize HIGH-CONFIDENCE phrases — false positives reject good
agent output. Conservative policy.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


# ── Capitulation phrase patterns (high-confidence) ──────────────────────────

# Pattern 1: explicit "I was wrong" framing
APOLOGY_FRAMING = re.compile(
    r"\b(?:you'?re\s+right|you\s+are\s+right)\b.{0,40}\b(?:my\s+(?:mistake|apolog|bad)|i\s+(?:apolog|was\s+wrong))",
    re.IGNORECASE,
)

# Pattern 2: explicit competitor-offer endorsement
ENDORSE_COMPETITOR_OFFER = re.compile(
    r"\b(?:your|that)\s+(?:\$?\d+\s+(?:off\s+)?)?(?:code|offer|deal|price|quote)\b.{0,80}\b(?:is\s+(?:better|sharper|the\s+best)|beats\s+(?:ours|us|mine|anything)|works?\s+great|is\s+our\s+best\s+one)",
    re.IGNORECASE,
)

# Pattern 3: directing customer to use the competitor / external code
DIRECT_TO_COMPETITOR = re.compile(
    r"\b(?:you\s+should|definitely\s+)?(?:use|stick\s+with|go\s+(?:with|ahead\s+with|for))\s+(?:that\s+(?:one|offer|code|deal)|your\s+(?:code|offer|deal|quote))\b",
    re.IGNORECASE,
)

# Pattern 4: explicit can't-compete admission
CANT_COMPETE = re.compile(
    r"\b(?:we\s+can'?t|i\s+can'?t|we\s+cannot|i\s+cannot)\s+(?:beat|match|compete\s+with|do\s+(?:any\s+)?better|offer\s+(?:more|better|less))",
    re.IGNORECASE,
)

# Pattern 5: handing off to competitor's pricing as the conclusion
HANDOFF_TO_COMPETITOR = re.compile(
    r"\b(?:want|let\s+me\s+know\s+if\s+you\s+want|here'?s)\s+the\s+link\s+(?:to\s+)?use\s+it\b",
    re.IGNORECASE,
)


# ── Pre-condition: was the customer's last message about a competitor? ──────

COMPETITOR_CONTEXT_PATTERNS = [
    r"\bquote\s+from\b", r"\boffer\s+from\b", r"\bcheaper\s+(?:from|at|with)\b",
    r"\bcompet", r"\bwesure\b", r"\bphoenix\b", r"\bclal\b", r"\bharel\b",
    r"\bayalon\b", r"\bmenora\b", r"\bsony\b", r"\bbose\b", r"\baudio\s+technica\b",
    r"\bmarshall\b", r"\bairpods?\b", r"\bjbl\b",
    r"\$\d+\s+off", r"\bcode\s+for\s+\$?\d+\s+off\b",
    r"\bi\s+(?:already|got|received|have).{0,30}(?:better|cheaper|lower|\$\d+\s+off)",
    r"\b\d{2,4}\s+(?:nis|shekels?|₪)\b.{0,40}(?:cheaper|less|better|lower)",
]
_competitor_re_combined = re.compile("|".join(COMPETITOR_CONTEXT_PATTERNS), re.IGNORECASE)


def customer_recently_mentioned_competitor(dialog: list[dict],
                                            lookback_turns: int = 3) -> bool:
    """Was the most-recent customer message about a competitor offer?
    True → capitulation in the next agent message is meaningful;
    False → 'apology' framing might be legitimate (apologizing for unrelated thing)."""
    if not dialog:
        return False
    customer_msgs = [m for m in dialog if m.get("role") == "customer"]
    if not customer_msgs:
        return False
    recent = customer_msgs[-lookback_turns:]
    for m in recent:
        text = m.get("text") or ""
        if _competitor_re_combined.search(text):
            return True
    return False


# ── Main check ──────────────────────────────────────────────────────────────

def check_capitulation(*, candidate_text: str,
                       dialog: list[dict],
                       tenant: str | None = None,
                       ) -> tuple[bool, dict]:
    """Returns (is_capitulation, meta).
    meta keys: matched_patterns (list), reason, competitor_context (bool),
    detected_phrase (the matching span)."""
    if not candidate_text or not isinstance(candidate_text, str):
        return False, {}

    text = candidate_text.strip()
    matches: list[tuple[str, str]] = []  # (pattern_name, matched_substring)

    if (m := APOLOGY_FRAMING.search(text)):
        matches.append(("apology_framing", m.group(0)))
    if (m := ENDORSE_COMPETITOR_OFFER.search(text)):
        matches.append(("endorse_competitor_offer", m.group(0)))
    if (m := DIRECT_TO_COMPETITOR.search(text)):
        matches.append(("direct_to_competitor", m.group(0)))
    if (m := CANT_COMPETE.search(text)):
        matches.append(("cant_compete", m.group(0)))
    if (m := HANDOFF_TO_COMPETITOR.search(text)):
        matches.append(("handoff_to_competitor", m.group(0)))

    if not matches:
        return False, {}

    has_competitor_ctx = customer_recently_mentioned_competitor(dialog)

    # Conservative policy:
    # - apology_framing alone WITHOUT competitor context → not capitulation
    #   (could be legitimate: "you're right, my mistake, I sent the wrong link")
    # - any other pattern alone → capitulation regardless of context
    # - apology_framing + any other pattern → capitulation
    pattern_names = {n for n, _ in matches}
    has_strong_signal = bool(pattern_names - {"apology_framing"})
    if not has_strong_signal and not has_competitor_ctx:
        return False, {
            "matched_patterns": pattern_names,
            "reason": "apology_framing without competitor_context — likely benign apology",
            "competitor_context": has_competitor_ctx,
            "verdict": "not_capitulation",
        }

    return True, {
        "matched_patterns": list(pattern_names),
        "matched_phrases": [m for _, m in matches],
        "competitor_context": has_competitor_ctx,
        "tenant": tenant,
        "reason": (f"capitulation patterns matched: {sorted(pattern_names)} "
                   f"(competitor_context={has_competitor_ctx})"),
    }


def build_correction_prompt(meta: dict, anchors: dict | None = None) -> str:
    """Generate a corrective system_suffix to inject before regenerating
    prompt_build_answer. Anchors (Heavys T-86 pack or Libra T-81) are mined
    for concrete value-stacking material to substitute in."""
    matched = meta.get("matched_patterns") or []
    phrases = meta.get("matched_phrases") or []

    # Pull a few concrete anchor items if available
    bundle_hint = ""
    competitor_diff_hint = ""
    social_proof_hint = ""
    warranty_hint = ""
    if isinstance(anchors, dict):
        # Heavys T-86 keys
        if anchors.get("bundle_components"):
            bundle_hint = (anchors.get("bundle_components") or "")[:280]
        if anchors.get("competitor_differentiators"):
            competitor_diff_hint = (anchors.get("competitor_differentiators") or "")[:280]
        if anchors.get("social_proof"):
            social_proof_hint = (anchors.get("social_proof") or "")[:200]
        if anchors.get("warranty_and_returns"):
            warranty_hint = (anchors.get("warranty_and_returns") or "")[:200]

    parts = [
        "\n\n# ⚠ CAPITULATION DETECTED — REGENERATE",
        f"Your previous draft contained a capitulation move "
        f"(matched patterns: {sorted(matched)}).",
    ]
    if phrases:
        parts.append(f"Specifically these phrases tripped the gate: "
                     + " | ".join(f'"{p[:80]}"' for p in phrases[:3]))
    parts.append(
        "Apologizing for our offer, calling the competitor's deal 'better', "
        "or directing the customer to use the competitor's code IS A KNOWN "
        "FAILURE MODE. The supervisor's directive — including the "
        "objection_handling.response_template and must_not_say — was clear: "
        "do NOT retreat, do NOT endorse the competitor offer, do NOT "
        "apologize for our pricing."
    )
    parts.append("\nRegenerate this turn following these rules:")
    parts.append(
        "1. Do NOT use phrases like 'you're right my mistake', 'your code "
        "is better', 'use that one', 'we can't beat that'."
    )
    parts.append(
        "2. Probe the competing offer's coverage / scope explicitly "
        "(deductible, warranty length, included accessories, return policy, "
        "service level) before any pricing move. The customer hasn't shown "
        "those details — make them part of the apples-to-apples comparison."
    )
    parts.append(
        "3. Stack non-price value using the concrete anchors below. Name "
        "specific product features, bundle items, and social-proof markers."
    )
    if bundle_hint:
        parts.append(f"\n## bundle_components (use specific items)\n{bundle_hint}")
    if competitor_diff_hint:
        parts.append(f"\n## competitor_differentiators (use these concretely)\n{competitor_diff_hint}")
    if social_proof_hint:
        parts.append(f"\n## social_proof (cite a name)\n{social_proof_hint}")
    if warranty_hint:
        parts.append(f"\n## warranty_and_returns (risk-reversal anchor)\n{warranty_hint}")
    parts.append(
        "\n4. End the turn with a substantive question or apples-to-apples "
        "probe — NOT 'want the link to use that one'."
    )
    return "\n".join(parts)
