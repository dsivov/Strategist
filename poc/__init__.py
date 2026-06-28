"""POC — sales-conversation AI benchmark package.

Public API:
    from poc import (
        Engine,            # the protocol PI implements
        BaselineEngine,    # single-call LLM (the "what you'd ship if you didn't think" arm)
        PlannerEngine,     # PCA-derived state-graph planner + post-render gates
        StrategistEngine,  # mining + retrieval + multi-stage supervisor (requires Luna KG access)
        Benchmark,         # the runner — pass an Engine, get per-scenario results
        load_scenarios,    # load v1_scenarios.json
        CustomerSimulator, # v2 reference-aware simulator
    )

See README.md for installation, integration, and the example PI adapter.
"""
from __future__ import annotations
import os
from pathlib import Path

# ── Locate bundled data, expose it to subpackages via env vars ──────────────
_PACKAGE_ROOT = Path(__file__).resolve().parent
_DATA_ROOT = _PACKAGE_ROOT.parent / "data"

# Used by poc.planner.playbook_reader and poc.strategist.concrete_moves_loader.
# Both packages default to file-relative paths; overriding via env keeps the
# upstream code unmodified.
os.environ.setdefault("POC_DATA_ROOT", str(_DATA_ROOT))
os.environ.setdefault("POC_SCRIPT_LIBRARY_DIR", str(_DATA_ROOT / "script_library"))
os.environ.setdefault("POC_CONCRETE_MOVES_DIR", str(_DATA_ROOT / "concrete_moves"))

# v2 reference-aware simulator + planner gates default ON for benchmark runs
os.environ.setdefault("POC_SIM_V2_REFERENCE", "on")
os.environ.setdefault("POC_PLANNER_GATES", "on")

# ── Public surface ──────────────────────────────────────────────────────────
from .engine import Engine, BaselineEngine, PlannerEngine, StrategistEngine
from .benchmark import Benchmark, load_scenarios
from .customer_simulator import CustomerSimulator

__all__ = [
    "Engine",
    "BaselineEngine",
    "PlannerEngine",
    "StrategistEngine",
    "Benchmark",
    "load_scenarios",
    "CustomerSimulator",
]

__version__ = "1.0.0"
