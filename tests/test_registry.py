"""Tests for the pluggable engine registry (poc.registry). No LLM calls."""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "poc"))
os.environ.setdefault("ANTHROPIC_API_KEY", "registry-test-placeholder")
os.environ.setdefault("GEMINI_API_KEY_1", "registry-test-placeholder")


def test_builtins_registered():
    from poc import all_engine_specs, has_engine
    ids = {s.id for s in all_engine_specs()}
    assert {"baseline", "planner", "strategist"} <= ids
    assert has_engine("planner")
    assert not has_engine("does-not-exist")


def test_runnable_ordering():
    from poc import all_engine_specs
    specs = all_engine_specs()
    # runnable engines sort before non-runnable ones
    runnable_flags = [s.runnable for s in specs]
    assert runnable_flags == sorted(runnable_flags, reverse=True)


def test_to_public_shape():
    from poc import get_engine_spec
    pub = get_engine_spec("planner").to_public()
    assert set(pub) == {"id", "name", "description", "runnable", "requires", "params"}
    pe = next(p for p in pub["params"] if p["name"] == "planner_envelope")
    assert pe["type"] == "enum"
    assert pe["default"] == "off"
    assert "auto" in pe["choices"]


def test_create_and_defaults():
    from poc import create_engine
    assert type(create_engine("baseline")).__name__ == "BaselineEngine"
    eng = create_engine("planner", planner_envelope="auto")
    assert type(eng).__name__ == "PlannerEngine"
    # unknown params are ignored, declared defaults applied
    eng2 = create_engine("planner", bogus="x")
    assert type(eng2).__name__ == "PlannerEngine"


def test_non_runnable_raises():
    from poc import create_engine
    try:
        create_engine("strategist")
    except RuntimeError as e:
        assert "not runnable" in str(e)
    else:
        raise AssertionError("strategist should not be instantiable")


def test_unknown_engine_raises():
    from poc import get_engine_spec
    try:
        get_engine_spec("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown engine should raise KeyError")


def test_duplicate_registration_guarded():
    from poc import EngineSpec, register_engine
    spec = EngineSpec(id="dup-test-engine", name="Dup")
    register_engine(spec)
    try:
        register_engine(EngineSpec(id="dup-test-engine", name="Dup2"))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate id should raise unless replace=True")
    # replace=True overwrites
    register_engine(EngineSpec(id="dup-test-engine", name="Dup2"), replace=True)
    from poc import get_engine_spec
    assert get_engine_spec("dup-test-engine").name == "Dup2"


def test_benchmark_run_engine_uses_registry(monkeypatch):
    """Benchmark.run_engine resolves the id via the registry and delegates to
    run_arm with the instantiated engine — without running any scenarios."""
    from poc import EngineSpec, register_engine
    from poc.benchmark import Benchmark

    built = {}

    class _FakeEngine:
        async def produce(self, opp_meta, dialog, business_rules=""):
            return "hi", {}

    register_engine(EngineSpec(
        id="fake-test-engine", name="Fake",
        factory=lambda **kw: built.setdefault("engine", _FakeEngine()),
    ), replace=True)

    captured = {}

    async def _fake_run_arm(self, arm_name, engine, scenarios=None, on_scenario_done=None):
        captured["arm_name"] = arm_name
        captured["engine"] = engine
        return ["sentinel"]

    monkeypatch.setattr(Benchmark, "run_arm", _fake_run_arm, raising=True)

    bench = Benchmark(scenarios=[], results_dir="/tmp/poc_reg_test_results")
    out = asyncio.run(bench.run_engine("fake-test-engine"))
    assert out == ["sentinel"]
    assert captured["arm_name"] == "fake-test-engine"
    assert isinstance(captured["engine"], _FakeEngine)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
