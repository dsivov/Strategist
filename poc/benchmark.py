"""Benchmark runner — paired n-arm comparison on the v1 scenario set.

Public API:
    scenarios = load_scenarios()                  # the 112 bundled scenarios
    bench     = Benchmark(scenarios, max_turns=12)
    results   = await bench.run_arm("baseline", BaselineEngine())
    results   = await bench.run_arm("example",        ExampleEngine())

Then aggregate paired (same scenario_id across arms) and analyze however you
like. The bundled `examples/run_benchmark.py` shows a full 2-arm paired run.

Design notes
------------
- Resumable: per-arm per-scenario result JSON is written to `results_dir`;
  re-running skips scenarios already complete for that arm.
- Per-scenario deterministic rng_seed is set before each run.
- v2 reference-aware customer simulator (default on; see POC_SIM_V2_REFERENCE).
- Outcome label: "won" when the customer's reply contains a close signal or
  payment; "lost" otherwise (timeout, refusal, graceful close, max_turns).
"""
from __future__ import annotations
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from .engine import Engine

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import customer_simulator as _cs
from customer_simulator import (
    CustomerSimulator,
    detect_close_signal,
    detect_decline,
    detect_agent_refusal,
    detect_agent_graceful_close,
)
from domain import active_domain

# canonical scenario set
_DATA = Path(os.environ.get("POC_DATA_ROOT",
                            str(_PKG.parent / "data")))
_DEFAULT_SCENARIOS = _DATA / "benchmark" / "v1_scenarios.json"

_EXPLICIT_PAYMENT_REGEX = re.compile(
    r"(?i)\b(card\s*last\s*4|last\s*four\s*digits|cvv|\b\d{4}\b\s*$)"
)
_DEFERRAL_RE = re.compile(
    r"\b(get back to you|if i decide|i'?ll (review|think|consider|let you know|"
    r"check)|think (about|it over)|review (it|the)|before i (decide|confirm|"
    r"commit|hear)|not sure( yet)?|decide later|need to (think|check|discuss|talk)|"
    r"maybe later|i'?ll see|circle back|hold off|"
    # 2026-07-05 — hedges/negations that co-occur with commit words in voice
    # transcripts; keep them from false-winning ("not to lock it in right now").
    r"not signing|not going to sign|won'?t sign|not ready|not proceeding|"
    r"not to lock|want to compare|still need to compare|compare with|"
    r"need to (compare|hear|see|speak)|on the phone before|"
    # echo/quote guard: customer quoting the agent's pitch ("you keep saying
    # 'let's lock in'…") is a rejection, not a commitment.
    r"you (keep|kept) (saying|repeating|telling)|you'?re just saying|"
    r"decided on|going with (klal|menora|phoenix|another|competitor))\b",
    re.IGNORECASE,
)


# ── Loading scenarios ──────────────────────────────────────────────────────

def load_scenarios(path: str | Path | None = None,
                   source: Any = None) -> list[dict]:
    """Load the benchmark scenario set.

    Resolution order:
      - explicit `source` (any ScenarioSource) → source.load_scenarios()
      - explicit `path` → read that JSON file directly (legacy behavior)
      - otherwise → the active scenario source (bundled JSON by default; swap
        it with `poc.set_scenario_source(...)`)

    Returns a list of dicts; see README §"Scenario schema" for the shape.
    """
    if source is not None:
        return source.load_scenarios()
    if path is not None:
        raw = json.loads(Path(path).read_text())
        return raw if isinstance(raw, list) else (
            raw.get("scenarios") or raw.get("rows") or [])
    from scenario_source import get_scenario_source  # flat: single shared instance
    return get_scenario_source().load_scenarios()


# ── Benchmark ──────────────────────────────────────────────────────────────

class Benchmark:
    """Run any Engine against the scenario set, get per-scenario results."""

    def __init__(
        self,
        scenarios: list[dict],
        results_dir: str | Path = "./benchmark_results",
        max_turns: int = 12,
    ):
        self.scenarios   = scenarios
        self.results_dir = Path(results_dir)
        self.max_turns   = max_turns
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ── registry-driven runner ──────────────────────────────────────────

    async def run_engine(
        self,
        engine_id: str,
        scenarios: list[dict] | None = None,
        on_scenario_done: Any = None,
        arm_name: str | None = None,
        **engine_params,
    ) -> list[dict]:
        """Run a registered engine by id — the pluggable entry point.

        Resolves `engine_id` through `poc.registry`, instantiates it with the
        given params (validated against the engine's ParamSpecs / defaults),
        and runs it. `arm_name` defaults to `engine_id` so results land under
        `{results_dir}/{engine_id}/`.

            results = await bench.run_engine("planner", planner_envelope="auto")
        """
        from registry import create  # flat import: single shared registry instance
        engine = create(engine_id, **engine_params)
        return await self.run_arm(arm_name or engine_id, engine,
                                  scenarios=scenarios,
                                  on_scenario_done=on_scenario_done)

    # ── per-arm runner ──────────────────────────────────────────────────

    async def run_arm(
        self,
        arm_name: str,
        engine: Engine,
        scenarios: list[dict] | None = None,
        on_scenario_done: Any = None,
    ) -> list[dict]:
        """Run `engine` on every scenario; return per-scenario result dicts.

        Args:
          arm_name:         label written to results (e.g. "baseline", "example").
          engine:           any object implementing the Engine protocol.
          scenarios:        subset to run (defaults to self.scenarios).
          on_scenario_done: optional callback(result_dict) for live progress.

        Resumable: scenarios already on disk under this arm_name are skipped.
        """
        scs = scenarios if scenarios is not None else self.scenarios
        arm_dir = self.results_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        for s in scs:
            sid = s.get("scenario_id") or f"idx_{len(results)}"
            out_path = arm_dir / f"{sid}.json"
            if out_path.exists():
                try:
                    r = json.loads(out_path.read_text())
                    results.append(r)
                    if on_scenario_done:
                        on_scenario_done(r)
                    continue
                except Exception:
                    pass

            r = await self._run_one(arm_name, engine, s)
            out_path.write_text(json.dumps(r, indent=2, ensure_ascii=False))
            results.append(r)
            if on_scenario_done:
                on_scenario_done(r)
        return results

    # ── per-scenario runner ─────────────────────────────────────────────

    async def _run_one(self, arm_name: str, engine: Engine, scenario: dict) -> dict:
        opp_meta = self._opp_meta(scenario)
        seed     = self._seed_slice(scenario)
        dialog   = self._to_dialog(seed)
        full_hist = scenario.get("seed_messages") or []
        random.seed(scenario.get("rng_seed", 0))

        sim = CustomerSimulator(opp_meta, full_hist)
        base_agent_count = sum(1 for m in dialog if m["role"] == "agent")
        t0 = time.monotonic()
        outcome, end_reason = None, None
        turns: list[dict] = []

        for live_turn in range(self.max_turns):
            try:
                agent_text, ameta = await engine.produce(opp_meta, dialog, "")
            except Exception as e:
                agent_text, ameta = "", {"error": f"{type(e).__name__}: {str(e)[:160]}"}

            if not agent_text:
                end_reason = "agent_empty"
                break
            if detect_agent_refusal(agent_text):
                end_reason = "agent_refused"
                break
            dialog.append({"role": "agent", "text": agent_text})
            if detect_agent_graceful_close(agent_text):
                outcome, end_reason = "lost", "agent_graceful_close"
                break

            agent_k = base_agent_count + live_turn
            try:
                cust_text, sim_mode = await sim.reply(dialog, agent_text, agent_k)
            except Exception as e:
                cust_text, sim_mode = "", f"sim_err:{type(e).__name__}"
            if not cust_text:
                end_reason = "customer_empty"
                break
            dialog.append({"role": "customer", "text": cust_text})

            check = self._check_customer_outcome(cust_text)
            turns.append({
                "live_turn":      live_turn,
                "agent_text":     agent_text[:400],
                "customer_text":  cust_text[:400],
                "sim_mode":       sim_mode,
                "outcome_check":  check,
                "agent_meta":     {k: v for k, v in (ameta or {}).items()
                                   if k not in ("input_tokens","output_tokens","latency_ms")},
            })
            if check is not None:
                outcome, end_reason = check, f"customer_{check}"
                break

        if outcome is None:
            outcome = "lost"
            end_reason = end_reason or "max_turns_no_close"

        return {
            "arm":          arm_name,
            "scenario_id":  scenario.get("scenario_id"),
            "opp_id":       scenario.get("opp_id"),
            "tenant":       scenario.get("tenant"),
            "real_outcome": scenario.get("real_outcome"),
            "is_sentinel":  scenario.get("is_sentinel", False),
            "rng_seed":     scenario.get("rng_seed"),
            "outcome":      outcome,
            "end_reason":   end_reason,
            "n_live_turns": len(turns),
            "elapsed_s":    round(time.monotonic() - t0, 1),
            "turns":        turns,
            "ts":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "env": {
                "v2_simulator":   os.environ.get("POC_SIM_V2_REFERENCE"),
                "planner_gates":  os.environ.get("POC_PLANNER_GATES"),
                "max_live_turns": self.max_turns,
            },
        }

    # ── scenario helpers ────────────────────────────────────────────────

    @staticmethod
    def _opp_meta(s: dict) -> dict:
        attrs = s.get("attributes") or {}
        meta = {"id": s["opp_id"], "company": s["tenant"], **attrs}
        if s.get("anchors"):
            meta["anchors"]  = s["anchors"]
            meta["_anchors"] = s["anchors"]
        if s.get("voice_profile"):
            meta["voice_profile"] = s["voice_profile"]
        return meta

    @staticmethod
    def _seed_slice(s: dict) -> list[dict]:
        msgs = s.get("seed_messages") or []
        cut = s.get("seed_dialog_cut_idx")
        if cut is None:
            cut = max(1, len(msgs) // 2)
        return msgs[:cut + 1]

    @staticmethod
    def _to_dialog(seed_slice: list[dict]) -> list[dict]:
        out = []
        for m in seed_slice:
            text = (m.get("text") or "").strip()
            if not text:
                continue
            role = "agent" if m.get("direction") == "outbound" else "customer"
            out.append({"role": role, "text": text})
        return out

    @staticmethod
    def _check_customer_outcome(cust_text: str) -> str | None:
        # Win/decline criteria are domain-specific and come from the active
        # domain pack (poc/domain.py). The sales pack delegates to the same
        # calibrated detectors, so behavior is unchanged for the reference set.
        if not cust_text:
            return None
        if _DEFERRAL_RE.search(cust_text):
            return None
        dom = active_domain()
        if dom.detect_close(cust_text) or _EXPLICIT_PAYMENT_REGEX.search(cust_text):
            return "won"
        if dom.detect_decline(cust_text):
            return "lost"
        return None


# ── Quick analysis helper ───────────────────────────────────────────────────

def paired_summary(arms_results: dict[str, list[dict]]) -> dict:
    """Build a paired summary across arms keyed by arm_name -> list of results.

    Returns aggregate counts and pairwise win/loss numbers.
    """
    arm_names = list(arms_results.keys())
    by_sid: dict[str, dict[str, str]] = {}
    for arm, rows in arms_results.items():
        for r in rows:
            sid = r.get("scenario_id") or ""
            by_sid.setdefault(sid, {})[arm] = r.get("outcome")

    summary: dict[str, Any] = {"n_scenarios": len(by_sid), "arms": {}}
    for arm in arm_names:
        won = sum(1 for sid, m in by_sid.items() if m.get(arm) == "won")
        n   = sum(1 for sid, m in by_sid.items() if arm in m)
        summary["arms"][arm] = {
            "n":       n,
            "won":     won,
            "win_rate": round(won / n, 3) if n else None,
        }

    if len(arm_names) == 2:
        a, b = arm_names
        a_better = sum(1 for m in by_sid.values()
                       if m.get(a) == "won" and m.get(b) == "lost")
        b_better = sum(1 for m in by_sid.values()
                       if m.get(b) == "won" and m.get(a) == "lost")
        ties     = sum(1 for m in by_sid.values()
                       if m.get(a) == m.get(b) and m.get(a) in ("won", "lost"))
        summary["pairwise"] = {
            f"{a}_better": a_better,
            f"{b}_better": b_better,
            "ties":        ties,
        }
    return summary
