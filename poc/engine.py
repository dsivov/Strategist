"""Engine protocol + the three reference engines.

The PI team implements this Protocol to plug their stack into the benchmark.
A single async method, simple in / simple out:

    text, meta = await engine.produce(opp_meta, dialog, business_rules)

`meta` is a free-form dict — PI can include strategy / tone / hint_confidence /
their own telemetry. The benchmark records it verbatim but doesn't interpret it.

Provided implementations:
  - BaselineEngine    — single-call Gemini 2.5 Pro production agent (no extras)
  - PlannerEngine     — PCA-derived planner + post-render gates (self-contained)
  - StrategistEngine  — mining/retrieval/supervisor (REQUIRES the production system DB + KG)
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

# Ensure the package directory is importable so the in-package modules
# (actor, post_render_gates, planner, strategist) can find each other
# with their original flat imports.
_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


# ── Protocol ────────────────────────────────────────────────────────────────

@runtime_checkable
class Engine(Protocol):
    """The contract.

    Implement this in your own class; the benchmark calls produce() once per
    agent turn. Side-effect-free per call (no hidden state in self).
    """

    async def produce(
        self,
        opp_meta: dict[str, Any],
        dialog_history: list[dict[str, Any]],
        business_rules: str = "",
    ) -> tuple[str, dict[str, Any]]:
        """Generate the agent's next message.

        Args:
          opp_meta:        customer profile + scenario metadata (see schema in README).
          dialog_history:  list of {"role": "agent"|"customer", "text": str} so far.
          business_rules:  tenant-specific rule text (may be empty).

        Returns:
          (text, meta) — text is the customer-facing reply; meta is any
          telemetry you want recorded per turn (strategy, tone, confidence,
          gates fired, etc.).
        """
        ...


# ── Baseline ────────────────────────────────────────────────────────────────

class BaselineEngine:
    """Single-call production agent. The "what you'd ship without thinking" arm.

    Calls the shared customer-facing LLM (Gemini 2.5 Pro) with the customer
    profile, the dialog history, and the business rules. No directive, no
    planning, no retrieval. ~5 s/turn.
    """

    async def produce(self, opp_meta, dialog_history, business_rules=""):
        import actor
        text, meta = await actor.generate(
            opp_meta, dialog_history, business_rules, directive=None
        )
        return text, {"arm": "baseline", **(meta or {})}


# ── Planner ─────────────────────────────────────────────────────────────────

class PlannerEngine:
    """PCA-derived state-graph planner + chain-of-thought + post-render gates.

    Self-contained — uses the bundled SOP graphs in `poc/planner/data/sop_graph/`
    and the mined playbook library in `data/script_library/`. No DB needed.

    Per-turn flow:
        1. Estimate customer state from the dialogue.
        2. Look up allowed actions from the SOP graph adjacency.
        3. Chain-of-thought reasoning to pick the best action.
        4. Render the chosen action through the shared customer-facing LLM.
        5. Post-render gate chain (anti-staircase + premature-close).

    Latency: ~25–30 s/turn (Anthropic Sonnet for planning + Gemini for render).
    """

    def __init__(self, planner_envelope: str = "off"):
        from planner.engine import PlannerEngine as _Planner
        self._eng = _Planner()
        self._envelope = planner_envelope

    async def produce(self, opp_meta, dialog_history, business_rules=""):
        # planner_produce() in the engines factory drives this — re-implementing
        # inline so the package has no cross-engine imports.
        import actor
        import post_render_gates

        # Anchors live on opp_meta under either key for legacy reasons
        anchors = opp_meta.get("anchors") or opp_meta.get("_anchors")
        state = {
            "dialogue":         [{"role": m.get("role"), "text": m.get("text")}
                                 for m in (dialog_history or [])],
            "opp_meta":         opp_meta,
            "anchors":          anchors,
            "planner_envelope": (opp_meta.get("_planner_envelope") or self._envelope),
            "business_rules":   business_rules,
        }
        out = self._eng.step(state)
        directive = out.get("directive", {}) or {}
        meta      = dict(out.get("directive_meta", {}) or {})

        try:
            text, _ = await actor.generate(
                opp_meta, dialog_history, business_rules or "", directive=directive
            )
        except Exception as e:
            log.warning("planner: actor render failed: %s", e)
            text = "Let me make sure I address that properly — could you confirm what matters most to you here?"
            meta["actor_error"] = str(e)[:160]

        # Post-render gate chain (engine-agnostic; reads POC_PLANNER_GATES env)
        async def _regen(corrective: str) -> str:
            t, _ = await actor.generate(
                opp_meta, dialog_history, business_rules or "",
                directive=directive, system_suffix=corrective,
            )
            return t

        try:
            text, gates_meta = await post_render_gates.apply(
                dialog_history, opp_meta, text, _regen
            )
            meta.update(gates_meta)
        except Exception as e:
            log.warning("planner: post-render gates failed: %s", e)
            meta["gates_error"] = str(e)[:160]

        meta["arm"] = "planner_gates"
        return text, meta


# ── Strategist ──────────────────────────────────────────────────────────────

class StrategistEngine:
    """Mining + cohort retrieval + multi-stage supervisor + safety gates.

    **NOT self-contained.** The Strategist runs through a stage-driven prompt
    chain (`poc/strategist/chain_runner.py`) that requires:
      - the production system MySQL access (cohort + business-rules retrieval)
      - Agent Knowledge Graph endpoint (LIGHTRAG_API_URL / _API_KEY)
      - Cohort precedent edges (~4,300 on the insurance tenant)

    The source is included verbatim for review and integration work, but the
    chain runner's entry point (`run_chain(stages, ctx)`) is driven through the
    websocket replayer in our own POC server — it's not a single-shot async
    function the benchmark can call without that scaffolding.

    For benchmark runs against an engine the PI team controls, use:
      - BaselineEngine + PlannerEngine (both fully self-contained), and
      - their own PI engine via the Engine protocol.

    Treat this class as a placeholder. Instantiating it raises with a pointer
    to the README's "Running the Strategist arm" section, which documents the
    Agent-side setup if a full three-arm comparison is needed.
    """

    def __init__(self):
        raise RuntimeError(
            "StrategistEngine is included for source review only. Running it "
            "requires the production system prod-DB + Knowledge Graph credentials and "
            "the websocket replayer scaffolding (see README §'Running the "
            "Strategist arm'). For benchmarks without prod access, use "
            "BaselineEngine + PlannerEngine + your own engine via the Engine "
            "protocol."
        )
