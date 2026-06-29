"""Example: run a head-to-head benchmark using the pluggable engine registry.

Engines are referenced by id and instantiated from the registry — including any
out-of-tree engine installed via a `strategist.engines` entry point (see
examples/external_engine_plugin). Contrast with examples/run_benchmark.py, which
constructs the reference engines directly.

Run a tiny smoke (first 5 scenarios):
    python examples/run_pluggable_benchmark.py

Note: this makes live LLM calls (Gemini for the actor; Anthropic for the
simulator/planner), so set ANTHROPIC_API_KEY and GEMINI_API_KEY_1 first.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
# Make `import poc` work when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poc import Benchmark, load_scenarios, all_engine_specs
from poc.benchmark import paired_summary


async def main():
    print("Registered engines:")
    for s in all_engine_specs():
        flag = "" if s.runnable else "  (not benchmark-runnable)"
        print(f"  - {s.id:12s} {s.name}{flag}")

    scenarios = load_scenarios()[:5]  # small smoke subset
    bench = Benchmark(scenarios, results_dir="./benchmark_results", max_turns=8)

    # Run two runnable engines by id. Swap "planner" for "echo" if you've
    # installed the example plugin, or for your own registered engine id.
    arms = {}
    for engine_id in ("baseline", "planner"):
        print(f"\n=== running arm: {engine_id} ===")
        arms[engine_id] = await bench.run_engine(engine_id)

    print("\nPaired summary:")
    print(paired_summary(arms))


if __name__ == "__main__":
    asyncio.run(main())
