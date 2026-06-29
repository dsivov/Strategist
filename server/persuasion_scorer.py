"""Live persuasion scorer — matches the production system model + prompt exactly.

the production system (as of 2026-04-30) uses **Gemini 2.5 Pro** in
`agent/ai_tools/persuasive_score/persuasive_score_generator.py` to score
customer messages 0.0-1.0. We use the same model + same prompt template +
extend it to also output commitment_level (0-5 scale) for the Won-detection
trigger.

Apples-to-apples comparison guarantee: this scorer's outputs are directly
comparable with the system's stored `persuasive_score` values.

Public API:
  await score_turn(chat_history) -> {score: float, reason: str, commitment_level: int}
    - chat_history: list of {"role": "user"|"assistant", "text": str,
                              "persusive_score": float|None, "sequence_number": int}
    - Returns score for the LATEST user message that has persusive_score=None
    - "user" = customer; "assistant" = agent (matches the system's chat-history shape)
"""
from __future__ import annotations

import json
import logging
import os
import re
import asyncio
from typing import Any

from google import genai
from google.genai import types as genai_types

log = logging.getLogger(__name__)

# Same model the production system uses — apples-to-apples comparison
SCORER_MODEL = "gemini-2.5-pro"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY_1") or os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY (or GEMINI_API_KEY_1) required for persuasion scorer")

_client = genai.Client(api_key=GEMINI_API_KEY)


# Extended from agent/ai_tools/persuasive_score/persuasive_score_generator.py
# (verbatim Agent section + a `commitment_level` output addition).
PROMPT_TEMPLATE = """
# Role
You are an expert Sales Psychologist. Your task is to analyze a JSON chat history,
calculate persuasion scores for specific user messages, and explain the reasoning
behind each score.

# Input Data
**Chat History (JSON):**

{{INSERT_JSON_CHAT_HERE}}

# Processing Instructions

1.  **Iterate** through the provided JSON array.
2.  **Filter** for messages that meet **both** of the following criteria:
    *   `role` is exactly `"user"`.
    *   The field `persusive_score` is `null` (or missing).
3.  **Analyze** the content of these specific messages to calculate a
    **Persuasion Score (0.0 to 1.0)**:
    *   **0.0 - 0.4 (Resistance/Negative):** "STOP", complaints, anger, or clear rejection.
    *   **0.5 - 0.7 (Inquiry/Neutral):** Answering questions (e.g., "Music"),
        providing info, or asking for details.
    *   **0.8 - 1.0 (High Interest/Conversion):** Enthusiastic agreement ("Yah!"),
        gratitude ("Thanks!"), or confirming a purchase.
4.  **Also estimate `commitment_level` (0-5 scale)** for each message:
    *   **0:** Hostile / hard refusal / "STOP".
    *   **1:** Disengaged / one-word reply / no interest signaled.
    *   **2:** Asked a question / providing context / casually exploring.
    *   **3:** Mid-engagement / weighing options / soft positive signals.
    *   **4:** High intent / explicit interest / asking for next steps.
    *   **5:** Closing or closed — providing payment info, saying yes to renewal,
        explicit closing language ("yes renew", "I'll take it", CC last-4 digits).
5.  **Generate Output:** Create a list of objects containing the
    `sequence_number`, the calculated `score`, the `commitment_level`,
    and a short text `reason` explaining the score.

# Output Format
Return **only** a valid JSON array of objects. Do not use markdown formatting.

**Structure:**
[
  {
    "sequence_number": 1,
    "score": 0.90,
    "commitment_level": 4,
    "reason": "User expressed enthusiastic agreement ('Yah!') to the initial hook."
  }
]
"""


async def _call_gemini(prompt: str) -> str:
    """Async wrapper around google-genai (which is sync).
    Runs in default executor."""
    loop = asyncio.get_running_loop()
    def _sync():
        resp = _client.models.generate_content(
            model=SCORER_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,  # low — we want consistent scores
            ),
        )
        return resp.text or ""
    return await loop.run_in_executor(None, _sync)


def _parse(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("Persuasion scorer parse failed: %s; raw[:200]=%s", e, text[:200])
        return []
    if isinstance(parsed, dict):
        if "result" in parsed and isinstance(parsed["result"], list):
            return parsed["result"]
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


async def score_turn(chat_history: list[dict]) -> dict | None:
    """Score the LATEST user message that has persusive_score=None.

    Args:
      chat_history: list of dicts with at least:
        - role: "user" | "assistant"
        - text: str
        - sequence_number: int
        - persusive_score: float | None  (existing scores; None for the one to compute)

    Returns:
      dict {sequence_number, score: 0-1, commitment_level: 0-5, reason: str}
      or None if scoring failed.
    """
    if not chat_history:
        return None

    # Find the unscored user message
    target_seq = None
    for m in reversed(chat_history):
        if m.get("role") == "user" and m.get("persusive_score") is None:
            target_seq = m.get("sequence_number")
            break
    if target_seq is None:
        log.debug("No unscored user message in chat history")
        return None

    chat_json = json.dumps(chat_history, ensure_ascii=False)
    prompt = PROMPT_TEMPLATE.replace("{{INSERT_JSON_CHAT_HERE}}", chat_json)

    try:
        raw = await _call_gemini(prompt)
    except Exception as e:
        log.warning("Gemini call failed: %s", e)
        return None

    rows = _parse(raw)
    if not rows:
        return None
    # Find the matching row for target_seq
    for r in rows:
        if r.get("sequence_number") == target_seq:
            return {
                "sequence_number": target_seq,
                "score": float(r.get("score", 0)),
                "commitment_level": int(r.get("commitment_level", 0)),
                "reason": r.get("reason", ""),
            }
    # Fallback — return the last row
    last = rows[-1]
    return {
        "sequence_number": target_seq,
        "score": float(last.get("score", 0)),
        "commitment_level": int(last.get("commitment_level", 0)),
        "reason": last.get("reason", ""),
    }


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio as _asyncio

    async def smoke():
        chat = [
            {"sequence_number": 0, "role": "assistant", "text":
             "Hi! Your auto insurance is up for renewal. Mandatory + comprehensive: 4,238 USD for the year.",
             "persusive_score": None},
            {"sequence_number": 1, "role": "user", "text":
             "That seems high. Other insurers offered me 3,800 USD.",
             "persusive_score": None},
            {"sequence_number": 2, "role": "assistant", "text":
             "Thanks for sharing the comparison. Could you share the breakdown of their offer (mandatory vs comprehensive) so I can structure something competitive?",
             "persusive_score": None},
            {"sequence_number": 3, "role": "user", "text":
             "Yes please, that would be helpful. Mandatory was about 1,400 and comprehensive 2,400.",
             "persusive_score": None},
        ]
        print("Scoring latest user message…")
        r = await score_turn(chat)
        print(json.dumps(r, indent=2, ensure_ascii=False))

    _asyncio.run(smoke())
