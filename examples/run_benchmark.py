"""End-to-end benchmark driver — 2-arm paired comparison.

Runs Baseline vs an alternative arm across N scenarios, writes per-scenario
JSON to ./results/, prints a paired summary at the end.

Defaults to the bundled Planner+gates as the "other" arm so you can run this
out of the box. Replace `OTHER_ARM` with your PI engine for the head-to-head.

Usage:
    cd POC
    python examples/run_benchmark.py                 # 10-scenario smoke
    BENCHMARK_N=112 python examples/run_benchmark.py # full set
"""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

# Make `import poc` work when running this from POC/ directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poc import (
    Benchmark, BaselineEngine, PlannerEngine, load_scenarios,
)
from poc.benchmark import paired_summary


# ── Configure the two arms ────────────────────────────────────────────────
BASELINE_ARM = "baseline"
OTHER_ARM    = "planner_gates"   # change to "pi" + PIEngine() for the real run

N_SCENARIOS  = int(os.environ.get("BENCHMARK_N", "10"))


async def main():
    scenarios = load_scenarios()[:N_SCENARIOS]
    print(f"== POC Benchmark — {len(scenarios)} scenarios, paired ==\n")

    bench = Benchmark(scenarios, results_dir="./results", max_turns=12)

    def _on(r):
        print(f"  [{r['arm']:14}] {r['scenario_id']:18} "
              f"tenant={r['tenant']:6} real={r['real_outcome']:4} "
              f"-> {r['outcome']:4} ({r['end_reason']}, "
              f"{r['n_live_turns']:>2}t, {r['elapsed_s']:>5}s)", flush=True)

    a = await bench.run_arm(BASELINE_ARM, BaselineEngine(), on_scenario_done=_on)
    print()
    b = await bench.run_arm(OTHER_ARM,   PlannerEngine(),  on_scenario_done=_on)

    summary = paired_summary({BASELINE_ARM: a, OTHER_ARM: b})
    print("\n== Paired summary ==")
    for arm, info in summary["arms"].items():
        print(f"  {arm:14} n={info['n']:>3}  won={info['won']:>3}  win_rate={info['win_rate']}")
    if "pairwise" in summary:
        print(f"  pairwise: {summary['pairwise']}")


if __name__ == "__main__":
    asyncio.run(main())
