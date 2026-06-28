"""Semantic intent classifier — replaces brittle regex heuristics.

Uses minishlab's model2vec (potion-base-32M) for fast local embedding +
cosine-similarity against per-intent anchor sets. Replaces multiple
regex-based heuristics across the codebase with one consistent abstraction.

Design:
  • Singleton model loaded once at first use (~5s cold-start, then in-memory)
  • Anchor sets defined per intent (farewell, human_request, etc.)
  • intent_score(text, intent_name) → (decision, score, matched_anchor)
  • Per-intent default thresholds tuned against validation cases

Latency: ~0.2ms per call. No network, no API cost.

Fallback: if model2vec fails to import or load, intent_score returns
(False, 0.0, "") — callers should treat this as "no signal" and fall back
to their existing regex path. Logged once at module load.
"""
from __future__ import annotations

import logging
import threading
import numpy as np

log = logging.getLogger(__name__)

import json as _json
from pathlib import Path as _Path

_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_NAME = "minishlab/potion-base-32M"
_MODEL_AVAILABLE = True

# Anchor files live in data/intent_anchors/<intent>__<tenant>.json (mined)
# and data/intent_anchors/<intent>__synthetic.json (hand-authored bridge).
# Both are loaded and concatenated for the given tenant.
_ANCHOR_DIR = _Path(__file__).resolve().parent.parent / "data" / "intent_anchors"


def _load_model():
    """Load + cache the embedding model. Thread-safe singleton."""
    global _MODEL, _MODEL_AVAILABLE
    if _MODEL is not None or not _MODEL_AVAILABLE:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None or not _MODEL_AVAILABLE:
            return _MODEL
        try:
            from model2vec import StaticModel
            _MODEL = StaticModel.from_pretrained(_MODEL_NAME)
            log.info("intent_classifier: loaded model2vec model %s (dim=%s)",
                     _MODEL_NAME, getattr(_MODEL, "dim", "?"))
        except Exception as e:
            log.warning("intent_classifier: failed to load %s: %s — falling "
                        "back to regex-only mode", _MODEL_NAME, e)
            _MODEL_AVAILABLE = False
            _MODEL = None
    return _MODEL


# ── Per-intent anchor sets ─────────────────────────────────────────────────
#
# Each intent has 8-15 anchor phrases representing the "centroid" of the
# semantic neighborhood. Anchors are TEMPLATE phrases (no specific names /
# prices / codes) — model2vec compares the customer's actual message against
# these and we threshold on max cosine similarity.
#
# Thresholds are empirically tuned against validation cases. Defaults:
#   farewell:        0.55 — catches polite sign-offs without firing on
#                            substantive questions
#   human_request:   0.60 — slightly stricter; explicit asks only
#   engaged_question: 0.50 — looser; protective override against false stalls
#
# To extend a set, append phrases to the corresponding list and re-validate
# thresholds against the test corpus.

_ANCHORS = {
    "farewell": [
        "thanks, have a great day",
        "I'll wait to hear from you",
        "I'll be in touch",
        "I'll reach out if anything changes",
        "I'll let you know if I decide to move forward",
        "take care",
        "catch you later",
        "I appreciate it, talk soon",
        "thanks for your time",
        "I'll think about it and get back to you",
        "ok sounds good, talk later",
        "no thanks, not now",
        "I'll come back if I'm interested",
        "appreciate the help, that's all for now",
    ],
    "human_request": [
        "I want to talk to a human, not a bot",
        "give me a real person on the phone",
        "I need to speak with a manager",
        "transfer me to a supervisor",
        "can I have a phone number to a real agent",
        "I'd like to escalate this to your manager",
        "I need a live agent now",
        "stop with the bot answers, get me a human",
    ],
    "engaged_question": [
        "but what's actually the answer to my question",
        "you didn't really address my concern",
        "that doesn't tell me what I need to know",
        "what's actually been done about this",
        "I still need to understand the details",
        "without that data I can't decide",
        "what are the specifics on this",
        "but how does that actually work",
        "the answer didn't quite get there",
        "I'll keep waiting for a real answer",
        # Declarative rejection: customer is engaged but stating a position
        # rather than asking — needs to be detected as engagement, not stall.
        "that doesn't prove the durability over years",
        "two weeks isn't enough evidence of long-term reliability",
        "that's not enough to convince me",
        "I need real data before I commit",
        "I'm not ready until you can show me the numbers",
        "that still doesn't solve the actual problem",
    ],
    "decline": [
        "I'm going to pass on this",
        "no thanks, not interested",
        "not buying today",
        "this isn't going to work for me",
        "I'll have to walk away from this",
        "I'm out, sorry",
        "not for me at this point",
    ],
    # Agent prematurely surrenders to a still-engaged but constrained customer.
    # Detects "no pressure / take your time / when you're ready / if things
    # change" patterns the agent emits when it has no concrete alternative to
    # propose and folds into a soft handoff. Used by prompt_premature_close_gate
    # to trigger regenerate before the simulator picks the phrase up as
    # agent_graceful_close and ends the panel.
    "agent_premature_close": [
        "No pressure at all, take your time",
        "If things change, just reach out",
        "we'll be here when you're ready",
        "no rush, let us know when you're ready",
        "the code is good for 24 hours if you change your mind",
        "feel free to come back when you're ready",
        "no worries, the offer stands",
        "I completely understand, no pressure",
        "let us know if anything changes",
        "we'll be here whenever you're ready",
        "happy to wait until you're ready to move forward",
        "the offer's open for now, get back to us anytime",
        # Multi-clause variants — capture longer phrasings where the giving-up
        # signal sits inside a larger sympathetic message.
        "I completely understand. No pressure at all. The code is good for 24 hours if things change, and we'll be here when you're ready.",
        "I hear you, no rush. The discount code is valid for 24 hours if anything changes — we'll be here.",
        "Totally get it, take your time. If things change, the offer is open for you.",
    ],
    # Customer is calling out that the agent didn't actually answer the question.
    # Used by the directive-loop-breaker stage to override pivots-to-features
    # and force a direct acknowledgement. Hand-authored — these are short and
    # tightly bounded; mining wouldn't add much (and there's no clean label in
    # the data for "customer accusing of evasion").
    "question_repeated_unanswered": [
        "You still didn't answer my question",
        "You didn't address what I asked",
        "I asked why X — what changed?",
        "You're apologizing but you're not explaining",
        "Stop deflecting and answer me",
        "Why are you avoiding the question",
        "That's not what I asked",
        "You keep dodging my question",
        "I'm asking you a specific question",
        "You haven't actually answered",
        "Please answer the question I asked",
        "Why did you say X earlier and now Y",
        "What changed between then and now",
    ],
    # customer_initiated_close anchors moved to:
    #   data/intent_anchors/customer_initiated_close__<Tenant>.json (mined)
    #   data/intent_anchors/customer_initiated_close__synthetic.json (bridge)
    # See server/mine_intent_anchors.py for the mining pipeline.
    #
    # Customer narrating that they're providing / authorizing payment. The
    # *bare-digit* form ("7834") has no semantic content and is handled
    # structurally in replayer (_BARE_LAST4_RE gated on agent-context); this
    # anchor covers the paraphrase-variant NARRATED forms. (2026-05-17 —
    # dcfbdb58 outcome-mislabel finding.)
    "customer_payment_provided": [
        "you can use the card on file from last year",
        "charge it to the same card as before",
        "go ahead with the saved card",
        "use the card you already have for me",
        "the last four digits are seven eight three four",
        "my card ending is one two three four",
        "here are the last 4 digits of my card",
        "yes charge my usual card",
        "just put it on the existing card",
        "process it on the card you have on file",
        "use last year's payment details",
    ],
    # Agent asserting the renewal/policy is DONE — a post-close confirmation.
    # Used so a polite "Thank you" after this is latched as won, not
    # mislabeled customer_polite_close=lost. (2026-05-17 — dcfbdb58.)
    "agent_confirmed_renewal": [
        "your policy is renewed",
        "I've renewed the policy for you",
        "the renewal is complete",
        "your policy has been renewed",
        "the documents are on their way to your email",
        "you're all set, the policy is active",
        "renewal confirmed",
        "I've processed the renewal",
        "the policy is now renewed and confirmed",
        "all done, your insurance is renewed for the year",
    ],
}

_THRESHOLDS = {
    "farewell": 0.55,           # mined anchors per tenant + hardcoded fallback
    "human_request": 0.60,      # hand-authored only (no labeled signal in DB)
    "engaged_question": 0.50,   # protective override — looser is safer
    # decline: stricter than farewell because false positive here marks panel
    # `lost` prematurely. Mined Heavys data has "Thank you!" in both farewell
    # and decline sets, so we lean on threshold to disambiguate.
    "decline": 0.65,
    # customer_initiated_close: bug-session texts score 0.63-0.71, worst
    # negative sits at 0.515; 100mV margin. Bypass also gated by G1+G2.
    "customer_initiated_close": 0.60,
    # agent_close_offer: mined CTA/payment language from won-deal agent turns.
    # Used as G3 in close guard. Looser threshold — false positives still
    # gated by G1+G2 (customer commit + persuasion score).
    "agent_close_offer": 0.55,
    # counter_offer: false positive BLOCKS legitimate wins, so stricter.
    "counter_offer": 0.65,
    # objection_price: used for supervisor script selection, not panel state.
    # Looser is fine — wrong script is recoverable; missed objection isn't.
    "objection_price": 0.55,
    # agent_premature_close: balance between "regenerate the agent's draft"
    # cost (one extra LLM call per false positive) and "panel-ends-prematurely"
    # cost. The bug-session text scores ~0.78 on these anchors; legitimate
    # objection-handling agent text scores ~0.30. 0.60 separates cleanly.
    "agent_premature_close": 0.60,
    # question_repeated_unanswered: relatively strict because false positives
    # incorrectly override the supervisor's normal strategy selection.
    # Negatives like "what's the warranty?" must NOT fire.
    "question_repeated_unanswered": 0.58,
    # customer_chitchat_acknowledgment: short post-close pleasantries (thanks,
    # yes, ok, sure, תודה). Looser threshold 0.42 catches "great"/"awesome"
    # while keeping engaged turns (objections, questions, "I'll consider it")
    # safely below. Used by detect_saturation: all 3 of last 3 must fire.
    "customer_chitchat_acknowledgment": 0.42,
    # customer_payment_provided: narrated payment authorization. Gated by
    # G1+G2 in the close guard, so a moderate threshold is safe; false
    # negatives are caught by the structural bare-digit path.
    "customer_payment_provided": 0.58,
    # agent_confirmed_renewal: post-close confirmation. Used to reclassify a
    # trailing "Thank you" as won. Conjunctive with farewell detection +
    # other close-occurred signals, so a moderate threshold is safe.
    "agent_confirmed_renewal": 0.55,
    # customer_post_match: used for simulator overlay. The intent overlaps
    # semantically with neutral price-mentioning ("got a quote", "from
    # competitor X") which is hard to disambiguate from "I went elsewhere"
    # in static embeddings. Threshold 0.55 + `?` veto in caller + the
    # _post_match_detected function's other prerequisites (agent must have
    # made a price-match pivot first) provide multi-layer protection.
    "customer_post_match": 0.55,
}

# Cached normalized anchor matrices (lazy-built on first use per intent[/tenant])
_ANCHOR_VECS: dict[str, np.ndarray] = {}
_ANCHOR_PHRASES: dict[str, list[str]] = {}


def _load_anchor_file(path: _Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = _json.loads(path.read_text())
        anchors = data.get("anchors") or []
        return [a for a in anchors if isinstance(a, str) and a.strip()]
    except Exception as e:
        log.warning("intent_classifier: failed to load %s: %s", path, e)
        return []


def _resolve_anchors(intent: str, tenant: str | None = None) -> list[str]:
    """Compose the anchor set for an intent: mined (per-tenant) + synthetic
    (tenant-agnostic bridge) + hardcoded fallback. Cached.

    Cache key: <intent>:<tenant>. Different tenants get different anchor
    matrices so we don't conflate Hebrew Libra phrasings with English Heavys.
    """
    cache_key = f"{intent}:{tenant or ''}"
    if cache_key in _ANCHOR_PHRASES:
        return _ANCHOR_PHRASES[cache_key]

    composed: list[str] = []
    if tenant:
        composed.extend(_load_anchor_file(
            _ANCHOR_DIR / f"{intent}__{tenant}.json"))
    composed.extend(_load_anchor_file(
        _ANCHOR_DIR / f"{intent}__synthetic.json"))
    if not composed:
        # Fallback: hand-coded set in _ANCHORS (covers intents we haven't
        # mined yet — farewell, human_request, engaged_question, etc.)
        composed = list(_ANCHORS.get(intent, []))
    # De-duplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for a in composed:
        if a in seen:
            continue
        seen.add(a)
        deduped.append(a)
    _ANCHOR_PHRASES[cache_key] = deduped
    return deduped


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _get_anchor_vecs(intent: str, tenant: str | None = None):
    """Lazy-encode + cache anchor matrix for (intent, tenant)."""
    cache_key = f"{intent}:{tenant or ''}"
    if cache_key in _ANCHOR_VECS:
        return _ANCHOR_VECS[cache_key]
    model = _load_model()
    if model is None:
        return None
    anchors = _resolve_anchors(intent, tenant)
    if not anchors:
        return None
    A = _normalize(model.encode(anchors))
    _ANCHOR_VECS[cache_key] = A
    return A


def intent_score(text: str, intent: str,
                  tenant: str | None = None,
                  threshold: float | None = None) -> tuple[bool, float, str]:
    """Classify whether `text` expresses the given intent.

    Args:
      text: customer or agent message
      intent: name of a known intent (mined per-tenant or hardcoded fallback)
      tenant: optional tenant key — picks mined per-tenant anchors when present
      threshold: cosine threshold; defaults to _THRESHOLDS[intent]

    Returns:
      (decision_bool, max_cosine_similarity, matched_anchor_phrase)

    On model load failure or unknown intent, returns (False, 0.0, "").
    Callers should treat that as "no signal" and use their fallback path.
    """
    if not text or not text.strip():
        return (False, 0.0, "")
    model = _load_model()
    if model is None:
        return (False, 0.0, "")
    A = _get_anchor_vecs(intent, tenant)
    if A is None:
        return (False, 0.0, "")
    th = threshold if threshold is not None else _THRESHOLDS.get(intent, 0.55)
    try:
        v = _normalize(model.encode([text]))
        sims = (v @ A.T).flatten()
        max_idx = int(sims.argmax())
        max_sim = float(sims[max_idx])
        anchors = _resolve_anchors(intent, tenant)
        return (max_sim >= th, max_sim, anchors[max_idx])
    except Exception as e:
        log.warning("intent_classifier: encode failed: %s", e)
        return (False, 0.0, "")


def is_available() -> bool:
    """Whether the model is loaded and ready. Useful for callers deciding
    between embedding-based and regex-fallback paths."""
    _load_model()
    return _MODEL_AVAILABLE and _MODEL is not None
