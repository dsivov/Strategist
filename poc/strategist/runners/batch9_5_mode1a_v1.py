"""Mode 1a v1 — classifier + selector with persona-axis routing.

Lookup key changes from `(cluster, motivator)` to `(cluster, motivator, decision_logic)`.
Falls back to motivator-only if decision_logic unmatched.

Imports + extends batch9_5_mode1a.
"""
from __future__ import annotations

import json
import os
import re
import logging
from dataclasses import dataclass

import anthropic
import numpy as np

from batch9_5_mode1a import (
    Classifier, _slug, render_playbook_compact, render_dialog,
    SELECTOR_SYSTEM, build_selector_prompt, call_selector,
    parse_directive, mode1a_directive as _v0_mode1a,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PLAYBOOKS_V1_BASE = os.environ.get(
    "PLAYBOOKS_V1_BASE",
    "/home/dimas/research/development_team/research/ai-agent-sales-improvement/playbooks_v1",
)


@dataclass
class ClassifierV1:
    tenant: str
    feature_order: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    centroids: np.ndarray
    cluster_names: dict[int, str]
    # Three-axis lookup: (cluster_id, motivator, decision_logic) -> playbook
    playbooks: dict[tuple[int, str, str], dict]
    # Two-axis fallback: (cluster_id, motivator) -> any matching playbook (for when decision_logic missing)
    playbooks_2axis: dict[tuple[int, str], list[dict]]

    def classify(self, features: dict) -> tuple[int, float, dict]:
        x = np.array([float(features.get(f, 0) or 0) for f in self.feature_order])
        x_scaled = (x - self.scaler_mean) / np.where(self.scaler_scale > 0, self.scaler_scale, 1.0)
        dists = np.linalg.norm(self.centroids - x_scaled[None, :], axis=1)
        order = np.argsort(dists)
        nearest = int(order[0])
        median_spacing = float(np.median([np.linalg.norm(self.centroids[i] - self.centroids[j])
                                          for i in range(len(self.centroids))
                                          for j in range(i + 1, len(self.centroids))]) or 1.0)
        temperature = max(median_spacing / 4.0, 0.5)
        scaled = -dists / temperature
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        confidence = float(probs[nearest])
        return nearest, confidence, {
            "dist_nearest": float(dists[order[0]]),
            "dist_second": float(dists[order[1]]),
            "all_dists": dists.tolist(),
            "temperature": temperature,
            "softmax_probs": probs.tolist(),
        }

    def lookup_playbook(self, cluster_id: int, motivator: str,
                         decision_logic: str | None) -> tuple[dict | None, str]:
        """Returns (playbook, lookup_strategy_used). lookup_strategy_used in:
           'exact_3axis' / 'fallback_motivator_only' / 'none'."""
        if decision_logic is not None:
            pb = self.playbooks.get((cluster_id, motivator, decision_logic))
            if pb is not None:
                return pb, "exact_3axis"
        # Fallback: motivator-only — pick the first playbook for this (cluster, motivator)
        candidates = self.playbooks_2axis.get((cluster_id, motivator), [])
        if candidates:
            return candidates[0], "fallback_motivator_only"
        return None, "none"


def load_classifier_v1(tenant: str) -> ClassifierV1:
    base = f"{PLAYBOOKS_V1_BASE}/{tenant}"
    with open(f"{base}/centroids.json") as f:
        cd = json.load(f)
    cluster_names = {int(k): v for k, v in cd["cluster_names"].items()}

    playbooks: dict[tuple[int, str, str], dict] = {}
    playbooks_2axis: dict[tuple[int, str], list[dict]] = {}
    if os.path.isdir(base):
        for fname in os.listdir(base):
            if not fname.endswith(".json") or fname == "centroids.json":
                continue
            try:
                with open(f"{base}/{fname}") as f:
                    pb = json.load(f)
                cid = int(pb["cluster_id"])
                mot = pb["motivator"]
                dl = pb.get("decision_logic", "Mixed")
                playbooks[(cid, mot, dl)] = pb
                playbooks_2axis.setdefault((cid, mot), []).append(pb)
            except Exception as e:
                log.warning("Failed to load v1 playbook %s: %s", fname, e)

    return ClassifierV1(
        tenant=tenant,
        feature_order=cd["feature_order"],
        scaler_mean=np.array(cd["scaler_mean"]),
        scaler_scale=np.array(cd["scaler_scale"]),
        centroids=np.array(cd["centroids"]),
        cluster_names=cluster_names,
        playbooks=playbooks,
        playbooks_2axis=playbooks_2axis,
    )


CONFIDENCE_GATE_V1 = 0.35


async def mode1a_directive_v1(opp: dict, state: dict, turn_idx: int, features: dict,
                               classifier: ClassifierV1,
                               client: anthropic.AsyncAnthropic) -> dict:
    import time
    cluster_id, confidence, debug = classifier.classify(features)
    profile = state.get("profile") or {}
    motivator = profile.get("primary_motivator") or features.get("motivator")
    decision_logic = profile.get("decision_logic")

    result = {
        "mode": "1a_v1",
        "tenant": classifier.tenant,
        "classifier_cluster": cluster_id,
        "classifier_cluster_name": classifier.cluster_names.get(cluster_id, "?"),
        "classifier_confidence": round(confidence, 3),
        "classifier_debug": debug,
        "motivator": motivator,
        "decision_logic": decision_logic,
        "directive": None,
        "playbook_used": None,
        "lookup_strategy": None,
        "fallback_reason": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
    }

    if confidence < CONFIDENCE_GATE_V1:
        result["fallback_reason"] = f"classifier_low_confidence ({confidence:.2f})"
        return result

    playbook, lookup_strategy = classifier.lookup_playbook(cluster_id, motivator, decision_logic)
    result["lookup_strategy"] = lookup_strategy
    if playbook is None:
        result["fallback_reason"] = f"no_playbook_for_cluster_{cluster_id}_{motivator}_{decision_logic}"
        return result

    result["playbook_used"] = (
        f"{classifier.tenant}/cluster{playbook['cluster_id']}_"
        f"{_slug(playbook['motivator'])}_{_slug(playbook.get('decision_logic', 'Mixed'))}.json"
    )

    prompt = build_selector_prompt(opp, state, turn_idx, playbook)
    t0 = time.monotonic()
    raw, in_tok, out_tok = await call_selector(client, prompt)
    result["latency_ms"] = int((time.monotonic() - t0) * 1000)
    result["input_tokens"] = in_tok
    result["output_tokens"] = out_tok

    directive, err = parse_directive(raw)
    if directive is None:
        result["fallback_reason"] = f"selector_parse_error: {err}"
        return result

    result["directive"] = directive
    return result


if __name__ == "__main__":
    for tenant in ("Insurance", "Ecommerce"):
        try:
            c = load_classifier_v1(tenant)
            print(f"\n{tenant}: {len(c.centroids)} centroids, "
                  f"{len(c.playbooks)} v1 playbooks (3-axis), "
                  f"{len(c.playbooks_2axis)} 2-axis fallback groups")
            for (cid, mot, dl), pb in sorted(c.playbooks.items()):
                print(f"  cluster {cid:1d} ({c.cluster_names.get(cid, '?')[:40]:40s}) × "
                      f"{mot:20s} × {dl:12s} → {pb.get('primary_strategy')}")
        except Exception as e:
            print(f"\n{tenant}: load failed — {e}")
