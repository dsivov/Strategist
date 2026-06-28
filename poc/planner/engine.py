"""PlannerEngine — independent R-side engine (PCA-faithful).

M1: deterministic stub to prove the 3-arm plumbing (vanilla / Strategist /
Planner) end-to-end with the shared customer simulator + scorer. No LLM, no
external deps, no Strategist imports. The real CoT+SOP brain lands in M3
(SOP-graph artifact arrives in M2).

Interface (harness contract):
    reset(opp_meta, seed_dialog, anchors) -> None
    step(state: dict) -> {"agent_text", "directive", "directive_meta"}
        state = {dialogue:[{role,text}], opp_meta, anchors, business_rules}
"""
from __future__ import annotations

from . import ENGINE_NAME, ENGINE_VERSION


class PlannerEngine:
    def __init__(self) -> None:
        self._turn = 0

    def reset(self, opp_meta: dict | None = None,
              seed_dialog: list | None = None,
              anchors: dict | None = None) -> None:
        self._turn = 0

    def step(self, state: dict) -> dict:
        """M3: CoT+SOP planning. Returns a directive (NO agent_text) — the
        shared actor renders it downstream, preserving the single-variable
        invariant. On any failure, returns a safe SOP-valid fallback
        directive so a live demo never hard-crashes."""
        self._turn += 1
        from . import cot_sop
        try:
            out = cot_sop.plan(state)
            return {"directive": out["directive"],
                    "directive_meta": out["directive_meta"]}
        except Exception as e:
            directive = {
                "engine": ENGINE_NAME,
                "strategy": {"primary": "objection_handling",
                             "tone": "professional"},
                "primary_strategy": "objection_handling",
                "tone": "professional",
                "must_say": [], "must_not_say": [],
                "rationale": "CoT+SOP fallback (planner error)",
                "confidence": 0.0,
            }
            return {"directive": directive, "directive_meta": {
                "architecture": "Planner — PCA CoT+SOP (fallback)",
                "engine": ENGINE_NAME, "engine_version": ENGINE_VERSION,
                "tier": "fallback", "primary_strategy": "objection_handling",
                "error": str(e)[:160]}}
