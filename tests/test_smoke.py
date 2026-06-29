"""Smoke test — verifies the package is wired correctly without making any
LLM calls. Run with: cd POC && python -m pytest tests/ -v

Or just run directly: cd POC && python tests/test_smoke.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Allow direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set placeholder env so module imports that read API keys don't crash
os.environ.setdefault("ANTHROPIC_API_KEY", "smoke-test-placeholder")
os.environ.setdefault("GEMINI_API_KEY_1",  "smoke-test-placeholder")


def test_package_imports():
    from poc import (
        Engine, BaselineEngine, PlannerEngine, StrategistEngine,
        Benchmark, load_scenarios, CustomerSimulator,
    )
    assert Engine is not None


def test_scenarios_load():
    from poc import load_scenarios
    scs = load_scenarios()
    assert len(scs) == 112, f"Expected 112 bundled scenarios, got {len(scs)}"
    s = scs[0]
    for key in ("scenario_id", "opp_id", "tenant", "attributes",
                "seed_messages", "rng_seed", "real_outcome"):
        assert key in s, f"missing key {key} in scenario {s.get('scenario_id')}"
    assert len(s["seed_messages"]) > 0


def test_playbooks_load():
    """Mined playbooks are bundled and reachable via env-pinned path."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "poc"))
    from planner.playbook_reader import load_tenant_playbooks
    for tenant in ("Insurance", "Ecommerce"):
        pbs = load_tenant_playbooks(tenant)
        assert len(pbs) > 0, f"no playbooks loaded for {tenant}"


def test_sop_graphs_exist():
    from pathlib import Path
    pkg = Path(__file__).resolve().parent.parent / "poc"
    sop = pkg / "planner" / "data" / "sop_graph"
    files = list(sop.glob("*.json"))
    assert len(files) >= 2, f"expected ≥2 SOP graphs, got {files}"


def test_engine_protocol_runtime_check():
    """A class with the right async method shape satisfies the Protocol."""
    from poc import Engine

    class _Stub:
        async def produce(self, opp_meta, dialog_history, business_rules=""):
            return "hi", {}

    # runtime_checkable Protocol — non-strict, checks method names
    assert isinstance(_Stub(), Engine)


def test_strategist_raises_with_hint():
    """Strategist is included but not runnable without prod-side scaffolding."""
    from poc import StrategistEngine
    import pytest
    with pytest.raises(RuntimeError) as exc:
        StrategistEngine()
    assert "README" in str(exc.value)


if __name__ == "__main__":
    test_package_imports();       print("✓ package imports")
    test_scenarios_load();         print("✓ scenarios load (112)")
    test_playbooks_load();         print("✓ playbooks load")
    test_sop_graphs_exist();       print("✓ SOP graphs exist")
    test_engine_protocol_runtime_check(); print("✓ engine protocol shape")
    try:
        test_strategist_raises_with_hint(); print("✓ strategist raises with README hint")
    except Exception as e:
        # pytest missing in env — do the check manually
        from poc import StrategistEngine
        try:
            StrategistEngine()
            raise AssertionError("StrategistEngine should have raised")
        except RuntimeError as re_:
            assert "README" in str(re_)
            print("✓ strategist raises with README hint")
    print("\nALL SMOKE TESTS PASSED")
