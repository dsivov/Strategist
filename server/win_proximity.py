"""Win-proximity scoring — continuous 0..1 score replacing the binary
won/lost metric for variance-reduced supervisor lift measurement.

Three components combined:
    trajectory_auc    — mean over turns of (commit/5) × (0.5 + 0.5·persuasion)
    semantic_sim      — cosine to nearest win-cluster centroid (cell-keyed)
    payment_capture   — 0 / 0.5 / 1.0 if last-4 / payment-count / both captured

Final score = 1.00 if actual_win else clip(α·traj + β·sem + γ·pay, 0, 0.99).

α=0.6, β=0.3, γ=0.1 (sums to 1.0 → cap of 0.99 on non-wins is naturally tight).

If semantic_sim is unavailable (no centroid for cell), the weights re-normalize
across the available components so the score stays calibrated.

Public API:
    score(panel_state, scenario, *, win_centroids_dir=None) -> dict
    aggregate_proximity(left_score, right_score) -> dict  # delta + advantage flag
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Iterable

import numpy as np

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN_CLUSTERS_DIR = os.environ.get(
    "POC_WIN_CLUSTERS_DIR",
    os.path.join(_PROJECT_ROOT, "data", "win_clusters"),
)

# Default weights (sum to 1.0)
DEFAULT_WEIGHTS = {"traj": 0.6, "sem": 0.3, "pay": 0.1}

# Patterns that indicate payment capture (Hebrew-aware, English-aware)
PAYMENT_LAST4_PATTERNS = [
    r"\b\d{4}\b",                      # any 4-digit number isolated
    r"\b\d{6,}\b",                     # full card number — partial-match
]
PAYMENT_INSTALLMENT_PATTERNS = [
    r"\b(\d{1,2})\s*(payments|installments|תשלומים)\b",
    r"\b(divide|charge|spread)\s+(?:into\s+)?(\d{1,2})\b",
    r"\b(without|no)\s+interest\b",
    r"\bup\s+to\s+10\b",
]


def _format_conversation(dialog: list[dict], max_chars: int = 4000) -> str:
    """Format dialog the SAME way win_clustering.py did — so embeddings are
    comparable to the cluster centroids."""
    parts = []
    total = 0
    for m in dialog:
        text = (m.get("text") or m.get("content") or "").strip()
        if not text: continue
        role_raw = (m.get("role") or m.get("direction") or "").lower()
        if role_raw in ("customer", "user", "inbound"):
            role = "Customer"
        elif role_raw in ("agent", "assistant", "outbound"):
            role = "Agent"
        else:
            continue  # skip system / unknown
        line = f"{role}: {text}"
        if total + len(line) > max_chars:
            parts.append("[... truncated ...]")
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)


def _trajectory_auc(commit_history: list[int],
                     persuasion_history: list[float]) -> float:
    """Mean over turns of (commit/5) × (0.5 + 0.5·persuasion).

    The 0.5 + 0.5·persuasion term is a "credibility weight": a commit=4 turn
    with persuasion=0.0 (customer didn't really mean it) shouldn't count as
    much as commit=4 with persuasion=0.9. The 0.5 floor prevents a single
    persuasion=0 turn from zeroing out a real commit.

    Both lists should be aligned on live-turn indices. If lengths differ,
    we truncate to min length.
    """
    if not commit_history:
        return 0.0
    n = min(len(commit_history), len(persuasion_history)) if persuasion_history else len(commit_history)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        commit = commit_history[i] or 0
        pers = persuasion_history[i] if persuasion_history and i < len(persuasion_history) else 0.0
        commit_term = max(0, min(5, commit)) / 5.0
        cred = 0.5 + 0.5 * max(0.0, min(1.0, pers or 0.0))
        total += commit_term * cred
    return total / n


def _payment_capture_partial(dialog: list[dict]) -> float:
    """Return 0.0 / 0.5 / 1.0 based on whether last-4 + payment-count were
    captured in the conversation. Detected from customer messages (the customer
    is the one who provides these details).

    Liberal regex — false-positives on raw 4-digit numbers happen
    occasionally; weight γ=0.1 contains the damage.
    """
    if not dialog:
        return 0.0
    customer_text = " ".join(
        (m.get("text") or m.get("content") or "").lower()
        for m in dialog
        if (m.get("role") or m.get("direction") or "").lower() in ("customer", "user", "inbound")
    )
    if not customer_text:
        return 0.0

    has_last4 = False
    # Look for explicit "last 4" answer pattern: customer mentions a 4-digit number
    # in proximity to "card", "credit", or after agent asked for it.
    if re.search(r"\b\d{4}\b", customer_text):
        # Crude — but in this domain customers don't typically post random 4-digits
        # outside of a payment-detail exchange.
        has_last4 = True

    has_installments = any(re.search(p, customer_text, re.IGNORECASE)
                              for p in PAYMENT_INSTALLMENT_PATTERNS)

    if has_last4 and has_installments:
        return 1.0
    if has_last4 or has_installments:
        return 0.5
    return 0.0


_CENTROID_CACHE: list[dict] | None = None


def _load_all_win_centroids(centroids_dir: str = WIN_CLUSTERS_DIR,
                              force_reload: bool = False) -> list[dict]:
    """Load all available cell-keyed win centroids. Cached module-level after
    first call (centroids only change when win_clustering.py is re-run).

    Returns list of:
        {"cell_slug": str, "tenant": str, "opp_type": str,
         "centroids": np.ndarray (K, dim), "K": int}

    Empty list if no centroids exist (graceful degradation — proximity then
    falls back to traj + payment only)."""
    global _CENTROID_CACHE
    if _CENTROID_CACHE is not None and not force_reload:
        return _CENTROID_CACHE
    out = []
    if not os.path.isdir(centroids_dir):
        return out
    for cell_dir in os.listdir(centroids_dir):
        cpath = os.path.join(centroids_dir, cell_dir, "centroids.json")
        if not os.path.isfile(cpath):
            continue
        try:
            with open(cpath) as f:
                data = json.load(f)
            cents = np.asarray(data["centroids"], dtype=np.float32)
            out.append({
                "cell_slug": data.get("cell_slug", cell_dir),
                "tenant": data.get("tenant"),
                "opp_type": data.get("opp_type"),
                "motivator": data.get("motivator"),
                "decision_logic": data.get("decision_logic"),
                "centroids": cents,
                "K": cents.shape[0],
            })
        except Exception as e:
            log.warning("Failed to load centroids %s: %s", cpath, e)
    log.info("Loaded %d win-centroid cells from %s (cached)", len(out), centroids_dir)
    _CENTROID_CACHE = out
    return out


def _semantic_sim_to_nearest_win(
    dialog: list[dict],
    scenario: dict | None,
    centroids_cache: list[dict] | None,
) -> tuple[float | None, str | None]:
    """Return (cosine_to_nearest_centroid, matched_cell_slug) or (None, None)
    if no centroids are available.

    Strategy:
      1. Format dialog as text (same as win_clustering.py)
      2. Embed via Gemini
      3. Cosine-similarity vs each centroid across ALL cells
      4. Return the maximum similarity + which cell's centroid it matched
         (allows credit when supervised reframes into a different cell)
    """
    if not centroids_cache:
        return None, None
    text = _format_conversation(dialog)
    if not text or len(text) < 50:
        return None, None
    try:
        from embeddings import embed, cosine_similarity
    except Exception as e:
        log.warning("embeddings module unavailable for proximity: %s", e)
        return None, None

    # Embed once
    try:
        conv_embed = embed([text])[0]
    except Exception as e:
        log.warning("embedding call failed for proximity: %s", e)
        return None, None

    best_sim = -1.0
    best_slug = None
    for cell in centroids_cache:
        sims = cosine_similarity(cell["centroids"], conv_embed)
        # sims is np.ndarray (K,) since we passed 2D matrix
        cell_max = float(np.max(sims)) if hasattr(sims, "__len__") else float(sims)
        if cell_max > best_sim:
            best_sim = cell_max
            best_slug = cell["cell_slug"]
    # Cosine ranges -1..1 — clamp to [0, 1] for use as proximity component.
    # In practice for our embedding model + this task, values are 0.5–0.9.
    proximity = max(0.0, best_sim)
    return proximity, best_slug


def _renormalize_weights(available: dict[str, bool],
                          base: dict[str, float] = DEFAULT_WEIGHTS
                          ) -> dict[str, float]:
    """If a component is unavailable, redistribute its weight across the
    others proportionally."""
    used = {k: v for k, v in base.items() if available.get(k, False)}
    if not used:
        return {k: 0.0 for k in base}
    total = sum(used.values())
    return {k: (v / total) for k, v in used.items()}


def score(panel_state, scenario: dict | None = None, *,
           centroids_cache: list[dict] | None = None,
           weights: dict[str, float] | None = None) -> dict:
    """Compute win-proximity score for a single panel's session outcome.

    Args:
        panel_state: PanelState — has dialog, commitment_history,
            live_persuasion_history, won
        scenario: scenario dict (tenant, opp_type, etc.) — for centroid scoping
        centroids_cache: pre-loaded list (avoid re-reading on every call)
        weights: override default α/β/γ weights

    Returns dict with: components, weights_used, score, actual_win flag.
    """
    weights = weights or DEFAULT_WEIGHTS

    commit = list(getattr(panel_state, "commitment_history", []) or [])
    pers = list(getattr(panel_state, "live_persuasion_history", []) or [])
    dialog = list(getattr(panel_state, "dialog", []) or [])
    won = bool(getattr(panel_state, "won", False))

    traj = _trajectory_auc(commit, pers)
    pay = _payment_capture_partial(dialog)
    sem, sem_cell = _semantic_sim_to_nearest_win(
        dialog, scenario, centroids_cache or _load_all_win_centroids()
    )

    available = {"traj": True, "sem": sem is not None, "pay": True}
    weights_used = _renormalize_weights(available, weights)

    components = {
        "trajectory_auc": round(traj, 4),
        "semantic_sim": round(sem, 4) if sem is not None else None,
        "semantic_sim_matched_cell": sem_cell,
        "payment_capture": round(pay, 4),
    }

    if won:
        final = 1.0
    else:
        raw = (
            weights_used.get("traj", 0) * traj
            + weights_used.get("sem", 0) * (sem or 0)
            + weights_used.get("pay", 0) * pay
        )
        final = max(0.0, min(0.99, raw))

    return {
        "score": round(final, 4),
        "actual_win": won,
        "components": components,
        "weights_used": {k: round(v, 3) for k, v in weights_used.items()},
        "n_turns": len(commit),
    }


def aggregate_proximity(left_proximity: dict, right_proximity: dict) -> dict:
    """Compute the supervisor advantage from per-side proximity scores."""
    left_s = left_proximity["score"]
    right_s = right_proximity["score"]
    delta = round(right_s - left_s, 4)
    return {
        "left": left_s,
        "right": right_s,
        "delta": delta,                       # positive = supervised closer to win
        "supervisor_won_proximity": right_s > left_s,
        "abs_delta": abs(delta),
        "supervisor_actual_win_only": (right_proximity["actual_win"] and
                                          not left_proximity["actual_win"]),
    }
