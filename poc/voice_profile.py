"""T-84 customer voice-profile extraction (2026-05-02 evening).

Empirical motivation: our 10-axis persona summary (motivator, decision_logic,
trust_level, communication_style, objection_pattern, ...) plus opp_type
coherence rules (T-82) give the customer simulator a STRUCTURED character but
not a TEXTURED one. Real customers in the historical transcript exhibit
specific phrasing patterns, interjections, hedging vocabulary, decisiveness
markers, and topic anchors that our flat persona summary loses.

This module extracts a structured voice profile from the customer's actual
historical messages. Pre-computed once per session at session-init (parallel
to fetch_libra_anchors); injected into customer_simulator's system prompt.

Approach: combine cheap statistical features (sentence length, word frequency,
common phrases) with one Haiku LLM call to characterize register, decisiveness
markers, hedge phrases, and topic anchors.

Cost: ~$0.0005 per session (one Haiku call at ~3000 input tokens).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import anthropic

log = logging.getLogger(__name__)

VOICE_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


VOICE_EXTRACTION_PROMPT = """Analyze this customer's voice from their historical messages in a sales conversation. Produce a JSON object characterizing how they actually talk.

Customer messages (in order):
{transcript}

Return ONLY a JSON object (no preamble, no markdown fences) with this shape:
{{
  "register": "casual" | "formal" | "mixed",
  "decisiveness_markers": [up to 5 phrases this customer uses when committing or declining decisively (verbatim from their messages where possible)],
  "hedge_phrases": [up to 5 phrases this customer uses when hesitating or deferring],
  "emotional_interjections": [up to 5 emotional/expressive markers ("Hahaha", "Look,", "Honestly,", "Wait,", etc.) from their messages],
  "topic_anchors": [up to 5 specific things this customer cares about (e.g. "price comparison", "deductible amount", "policy match for old car")],
  "voice_summary": "1-2 sentences in third person describing how this customer actually talks"
}}

Guidelines:
- Pull verbatim phrases where possible — these are TEXTUAL fingerprints, not paraphrases
- For "topic_anchors" use 2-5 word phrases capturing what the customer keeps returning to
- "register" should reflect actual word choice: contractions + idioms + interjections = casual; complete sentences + formal vocabulary = formal; both = mixed"""


def _statistical_features(customer_msgs: list[str]) -> dict[str, Any]:
    """Pure-Python statistical features that don't need an LLM call."""
    if not customer_msgs:
        return {}
    lengths = [len(m.strip()) for m in customer_msgs if m.strip()]
    if not lengths:
        return {}
    avg_len = sum(lengths) / len(lengths)
    # Variance / stddev
    var = sum((L - avg_len) ** 2 for L in lengths) / len(lengths) if len(lengths) > 1 else 0
    sd = var ** 0.5

    # Casual-register signals
    contraction_re = re.compile(r"\b(don't|can't|won't|I'm|I've|I'll|isn't|it's|that's|you're|they're|we're|wasn't|weren't|hasn't|haven't|wouldn't|shouldn't|couldn't|aren't|let's)\b", re.IGNORECASE)
    n_contractions = sum(len(contraction_re.findall(m)) for m in customer_msgs)
    casualness_density = n_contractions / max(1, len(customer_msgs))

    # Punctuation tells
    n_exclamations = sum(m.count("!") for m in customer_msgs)
    n_questions = sum(m.count("?") for m in customer_msgs)
    n_ellipses = sum(m.count("...") for m in customer_msgs)

    return {
        "n_messages": len(customer_msgs),
        "msg_length_avg": int(avg_len),
        "msg_length_stddev": int(sd),
        "casualness_density": round(casualness_density, 2),
        "exclamations": n_exclamations,
        "questions": n_questions,
        "ellipses": n_ellipses,
    }


async def extract_voice_profile(historical_messages: list[dict] | None) -> dict:
    """Return a structured voice profile derived from the customer's historical
    messages. Returns empty dict on any error (caller handles)."""
    if not historical_messages:
        return {}
    if _client is None:
        log.warning("voice_profile: no ANTHROPIC_API_KEY; skipping")
        return {}

    # Filter to customer (inbound) messages with text
    cust_msgs = [
        (m.get("text") or "").strip()
        for m in historical_messages
        if m.get("direction") == "inbound" and (m.get("text") or "").strip()
    ]
    if len(cust_msgs) < 2:
        return {}  # not enough signal

    stats = _statistical_features(cust_msgs)

    # Cap transcript at ~3000 chars for the LLM call (typical customer turn
    # is 50-150 chars; 3000 fits ~20-30 turns)
    transcript_lines = []
    total_chars = 0
    for i, m in enumerate(cust_msgs):
        line = f"  [{i+1}] {m[:300]}"
        if total_chars + len(line) > 3000:
            transcript_lines.append("  [... truncated]")
            break
        transcript_lines.append(line)
        total_chars += len(line)
    transcript = "\n".join(transcript_lines)

    user_prompt = VOICE_EXTRACTION_PROMPT.format(transcript=transcript)

    try:
        msg = await _client.messages.create(
            model=VOICE_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text if msg.content else ""
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        voice = json.loads(text)
    except Exception as e:
        log.warning("voice_profile extraction failed: %s", e)
        return {"_extraction_error": str(e)[:100], **stats}

    return {**voice, **stats, "_provenance": "extracted_from_historical_transcript"}


def render_voice_block(voice: dict | None) -> str:
    """Format a voice profile dict as a system-prompt block for the customer
    simulator. Empty string if no voice data available."""
    if not voice or voice.get("_extraction_error"):
        return ""
    lines = ["## Your speaking voice (pulled from your real conversation history — speak like THIS, not like a generic customer)"]
    if voice.get("voice_summary"):
        lines.append(f"- Summary: {voice['voice_summary']}")
    if voice.get("register"):
        lines.append(f"- Register: {voice['register']}")
    if voice.get("decisiveness_markers"):
        lines.append(f"- When deciding (committing or declining), you say things like: {voice['decisiveness_markers']}")
    if voice.get("hedge_phrases"):
        lines.append(f"- When hesitating, you say: {voice['hedge_phrases']}")
    if voice.get("emotional_interjections"):
        lines.append(f"- Your characteristic interjections / openings: {voice['emotional_interjections']}")
    if voice.get("topic_anchors"):
        lines.append(f"- Things you specifically care about: {voice['topic_anchors']}")
    if voice.get("msg_length_avg"):
        lines.append(f"- Your typical message length: ~{voice['msg_length_avg']} chars (stddev {voice.get('msg_length_stddev','?')}). Match this — don't write multi-paragraph essays.")
    if voice.get("casualness_density", 0) > 0.5:
        lines.append("- You use contractions naturally — write 'I'm' not 'I am', 'don't' not 'do not'.")
    return "\n" + "\n".join(lines) + "\n"


# ── Temperature jitter for variability ──────────────────────────────────────

def stable_temperature(opp_id: str, turn: int, base_lo: float = 0.5,
                       base_hi: float = 0.9) -> float:
    """T-84 controlled temperature jitter — varies the customer simulator's
    sampling temperature within [base_lo, base_hi] in a reproducible per-
    (opp, turn) way. Real customers exhibit within-conversation variability
    in decisiveness / wording; fixed temperature 0.7 collapses this."""
    import hashlib
    h = hashlib.md5(f"{opp_id}:{turn}".encode()).hexdigest()
    seed = int(h[:8], 16) / 0xffffffff  # 0.0-1.0
    return round(base_lo + (base_hi - base_lo) * seed, 3)
