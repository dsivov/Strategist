"""Tests for benchmark packs (poc.benchmark_packs / benchmarks/). No LLM."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "poc"))
os.environ.setdefault("ANTHROPIC_API_KEY", "src-test-placeholder")
os.environ.setdefault("GEMINI_API_KEY_1", "src-test-placeholder")


def test_bundled_packs_discovered():
    from poc import all_benchmark_packs, has_benchmark_pack
    ids = [p["id"] for p in all_benchmark_packs()]
    assert "insurance-renewal" in ids
    assert "ecommerce-cart" in ids
    assert has_benchmark_pack("insurance-renewal")
    assert not has_benchmark_pack("_template")   # templates are skipped


def test_pack_manifest_fields():
    from poc import get_benchmark_pack
    p = get_benchmark_pack("insurance-renewal")
    for field in ("id", "name", "description", "goal", "domain"):
        assert p.get(field), f"missing {field}"
    assert p["domain"] == "sales"


def test_pack_scenarios_slice_the_shared_dataset():
    from poc import load_pack_scenarios, load_scenarios
    ins = load_pack_scenarios("insurance-renewal")
    eco = load_pack_scenarios("ecommerce-cart")
    assert all(s["tenant"] == "Insurance" for s in ins)
    assert all(s["tenant"] == "Ecommerce" for s in eco)
    # The two bundled packs partition the full 112-scenario dataset
    assert len(ins) + len(eco) == len(load_scenarios()) == 112
    assert {s["opp_id"] for s in ins}.isdisjoint(s["opp_id"] for s in eco)


def test_unknown_pack_raises():
    from poc import load_pack_scenarios
    import pytest
    with pytest.raises(KeyError):
        load_pack_scenarios("no-such-pack")


def test_custom_pack_with_own_dataset(tmp_path):
    """A pack directory with its own scenarios.json (the _template flow)."""
    import benchmark_packs as bp

    pack_dir = tmp_path / "my-benchmark"
    pack_dir.mkdir()
    (pack_dir / "pack.json").write_text(json.dumps({
        "id": "my-benchmark",
        "name": "My Benchmark",
        "scenario_source": {"type": "json", "path": "scenarios.json"},
    }))
    (pack_dir / "scenarios.json").write_text(json.dumps([
        {"scenario_id": "X1", "opp_id": "x-1", "tenant": "Acme"},
        {"scenario_id": "X2", "opp_id": "x-2", "tenant": "Acme"},
    ]))
    # Ignored: underscore dir and a dir without pack.json
    (tmp_path / "_draft").mkdir()
    (tmp_path / "not-a-pack").mkdir()

    old = os.environ.get("POC_BENCHMARKS_DIR")
    os.environ["POC_BENCHMARKS_DIR"] = str(tmp_path)
    bp.reset_cache()
    try:
        packs = bp.all_packs()
        assert [p["id"] for p in packs] == ["my-benchmark"]
        assert packs[0]["domain"] == "sales"          # defaulted
        scs = bp.load_pack_scenarios("my-benchmark")
        assert [s["opp_id"] for s in scs] == ["x-1", "x-2"]
    finally:
        if old is None:
            os.environ.pop("POC_BENCHMARKS_DIR", None)
        else:
            os.environ["POC_BENCHMARKS_DIR"] = old
        bp.reset_cache()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
