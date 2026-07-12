"""Benchmark packs — self-describing, goal-oriented scenario bundles.

A *pack* is a directory under `<repo>/benchmarks/` with a `pack.json` manifest:

    {
      "id":          "insurance-renewal",
      "name":        "Insurance Renewal",
      "description": "Auto-insurance renewal negotiations ...",
      "goal":        "Customer agrees to renew the policy.",
      "domain":      "sales",
      "scenario_source": {
        "type":   "json",
        "path":   "../../data/benchmark/v1_scenarios.json",
        "filter": { "tenant": "Insurance" }
      }
    }

Packs are how the platform ships its *example* benchmarks (`insurance-renewal`,
`ecommerce-cart`) and the template for adding a new goal-oriented test: copy
`benchmarks/_template/`, point the manifest at your dataset, and pair it with a
domain pack (see docs/PLUGIN_GUIDE.md) when the framing / win criteria differ.
Directory names starting with `_` are ignored (templates, drafts).

Manifest semantics:
  - `scenario_source.path` is resolved relative to the pack directory.
  - `scenario_source.filter` is an equality match on top-level scenario fields;
    omit it to take the whole file. Several packs may slice one shared dataset
    (the two bundled packs both slice `data/benchmark/v1_scenarios.json`), or a
    pack can bring its own file in the same schema (README.md § dataset).
  - `domain` names the domain pack whose framing + won/lost detection the
    scenarios are written for (registered via `poc.register_domain`).

This module is imported by its FLAT name (`import benchmark_packs`) — same
single-instance rule as `registry` / `domain` (see `poc/__init__.py`).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

_LOCK = threading.Lock()
_CACHE: dict[str, dict] | None = None       # pack_id -> manifest (+ _dir)
_DATA_CACHE: dict[str, tuple[float, list]] = {}   # dataset path -> (mtime, rows)


def _read_dataset(path: Path) -> list:
    """mtime-cached JSON read — packs often share one multi-MB dataset file
    and the live server hits this per request."""
    key = str(path)
    mtime = path.stat().st_mtime
    hit = _DATA_CACHE.get(key)
    if hit is None or hit[0] != mtime:
        raw = json.loads(path.read_text())
        rows = raw if isinstance(raw, list) else (raw.get("scenarios") or [])
        _DATA_CACHE[key] = (mtime, rows)
    return _DATA_CACHE[key][1]


def packs_root() -> Path:
    """Directory scanned for packs. Override with POC_BENCHMARKS_DIR."""
    return Path(os.environ.get("POC_BENCHMARKS_DIR",
                               str(_REPO_ROOT / "benchmarks")))


def _scan() -> dict[str, dict]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        found: dict[str, dict] = {}
        root = packs_root()
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if not d.is_dir() or d.name.startswith("_"):
                    continue
                manifest_path = d / "pack.json"
                if not manifest_path.exists():
                    continue
                try:
                    m = json.loads(manifest_path.read_text())
                    pid = m.get("id") or d.name
                    m["id"] = pid
                    m.setdefault("name", pid.replace("-", " ").title())
                    m.setdefault("description", "")
                    m.setdefault("goal", "")
                    m.setdefault("domain", "sales")
                    m["_dir"] = str(d)
                    if pid in found:
                        log.warning("benchmark_packs: duplicate pack id %r "
                                    "(%s); keeping first", pid, d)
                        continue
                    found[pid] = m
                except Exception as e:
                    log.warning("benchmark_packs: skipping %s: %s", d, e)
        _CACHE = found
        return _CACHE


def reset_cache() -> None:
    """Forget scanned packs (tests / after changing POC_BENCHMARKS_DIR)."""
    global _CACHE
    with _LOCK:
        _CACHE = None


def all_packs() -> list[dict]:
    """Every pack manifest, sorted by id. `_dir` carries the pack directory."""
    return [dict(m) for _, m in sorted(_scan().items())]


def has_pack(pack_id: str) -> bool:
    return pack_id in _scan()


def get_pack(pack_id: str) -> dict:
    packs = _scan()
    if pack_id not in packs:
        raise KeyError(f"unknown benchmark pack: {pack_id!r} "
                       f"(available: {sorted(packs)})")
    return dict(packs[pack_id])


def load_pack_scenarios(pack_id: str) -> list[dict]:
    """Scenarios for one pack: read `scenario_source.path` (relative to the
    pack dir) and apply the equality `filter`, if any."""
    pack = get_pack(pack_id)
    src = pack.get("scenario_source") or {}
    if src.get("type", "json") != "json":
        raise ValueError(f"pack {pack_id!r}: unsupported scenario_source.type "
                         f"{src.get('type')!r} (only 'json')")
    rel = src.get("path")
    if not rel:
        raise ValueError(f"pack {pack_id!r}: scenario_source.path missing")
    path = (Path(pack["_dir"]) / rel).resolve()
    scenarios = _read_dataset(path)
    flt = src.get("filter") or {}
    if flt:
        return [s for s in scenarios
                if all(s.get(k) == v for k, v in flt.items())]
    return list(scenarios)   # copy — don't hand out the cached list itself
