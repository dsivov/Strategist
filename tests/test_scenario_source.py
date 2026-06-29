"""Tests for the pluggable scenario/data source (poc.scenario_source). No LLM."""
from __future__ import annotations
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "poc"))
os.environ.setdefault("ANTHROPIC_API_KEY", "src-test-placeholder")
os.environ.setdefault("GEMINI_API_KEY_1", "src-test-placeholder")


def test_default_source_loads_bundled_112():
    from poc import load_scenarios, get_scenario_source
    assert len(load_scenarios()) == 112
    assert len(get_scenario_source().load_scenarios()) == 112


def test_json_source_fetch_surface():
    from poc import JsonScenarioSource, load_scenarios
    src = JsonScenarioSource()
    scs = load_scenarios()
    opp_id = scs[0]["opp_id"]
    meta = src.fetch_opp_meta(opp_id)
    assert meta is not None
    assert meta["id"] == opp_id
    assert meta["company"] == scs[0]["tenant"]
    assert isinstance(src.fetch_messages(opp_id), list)
    assert src.fetch_opp_meta("no-such-opp") is None
    # business rules empty in the standalone JSON corpus
    assert src.fetch_business_rules("Insurance") == ""


def test_explicit_path_still_works():
    """Legacy load_scenarios(path=...) keeps reading a file directly."""
    from poc import load_scenarios
    path = _ROOT / "data" / "benchmark" / "v1_scenarios.json"
    assert len(load_scenarios(path=str(path))) == 112


def test_set_custom_source():
    """A custom in-memory source plugs in and is used by load_scenarios()."""
    from poc import load_scenarios, set_scenario_source, get_scenario_source

    class MemorySource:
        def load_scenarios(self):
            return [{"opp_id": "x1", "tenant": "Acme", "attributes": {}}]
        def fetch_opp_meta(self, opp_id):
            return {"id": opp_id, "company": "Acme"} if opp_id == "x1" else None
        def fetch_messages(self, opp_id):
            return []
        def fetch_business_rules(self, company):
            return "be nice"
        def fetch_anchors(self, opp_id, opp_meta=None):
            return {}

    prev = get_scenario_source()
    try:
        set_scenario_source(MemorySource())
        scs = load_scenarios()
        assert scs == [{"opp_id": "x1", "tenant": "Acme", "attributes": {}}]
        assert get_scenario_source().fetch_business_rules("Acme") == "be nice"
    finally:
        set_scenario_source(prev)

    assert len(load_scenarios()) == 112  # restored


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
