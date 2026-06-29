"""Scenario sources — the seam that makes data loading pluggable.

The benchmark and server read scenarios, opportunity metadata, transcripts,
business rules, and pricing anchors from *somewhere*. Historically that was
either the bundled `v1_scenarios.json` (standalone) or the production system MySQL
(`poc/db.py`), selected by the `POC_USE_MYSQL` env var inside `server/db.py`.

This module promotes that to a small, documented `ScenarioSource` Protocol so a
third party can supply scenarios from any backend (a different DB, an API, a
different file format) and plug it in without editing the core:

    from poc import set_scenario_source, JsonScenarioSource
    set_scenario_source(JsonScenarioSource("/path/to/my_scenarios.json"))

`JsonScenarioSource` is the default and reproduces the bundled-JSON behavior.
The function-style `server/db.py` shim remains the live server's data layer and
mirrors the same field shapes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class ScenarioSource(Protocol):
    """Read-only access to a benchmark's scenarios and per-opportunity data."""

    def load_scenarios(self) -> list[dict]:
        """Return the full scenario list (see README §'Scenario schema')."""
        ...

    def fetch_opp_meta(self, opp_id: str) -> dict | None:
        """Customer profile + scenario context for one opportunity, or None."""
        ...

    def fetch_messages(self, opp_id: str) -> list[dict]:
        """Historical transcript rows for one opportunity (may be empty)."""
        ...

    def fetch_business_rules(self, company: str) -> str:
        """Tenant compliance/business-rules text (may be empty)."""
        ...

    def fetch_anchors(self, opp_id: str, opp_meta: dict | None = None) -> dict:
        """Per-opportunity economic anchors (may be empty)."""
        ...


# ── JSON-file implementation (default) ───────────────────────────────────────

def _default_scenarios_path() -> Path:
    root = Path(os.environ.get("POC_DATA_ROOT",
                               str(Path(__file__).resolve().parent.parent / "data")))
    return root / "benchmark" / "v1_scenarios.json"


def _opp_meta_from_scenario(s: dict) -> dict:
    """Build opp_meta from a scenario row — shared shape with benchmark + db."""
    meta: dict[str, Any] = {"id": s["opp_id"], "company": s["tenant"]}
    meta.update(s.get("attributes") or {})
    if s.get("anchors"):
        meta["anchors"] = s["anchors"]
        meta["_anchors"] = s["anchors"]
    if s.get("voice_profile"):
        meta["voice_profile"] = s["voice_profile"]
    return meta


class JsonScenarioSource:
    """Serves scenarios from a JSON file (the bundled v1_scenarios.json by
    default). Lazily indexes by opp_id on first per-opportunity access."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _default_scenarios_path()
        self._scenarios: list[dict] | None = None
        self._index: dict[str, dict] | None = None

    def load_scenarios(self) -> list[dict]:
        if self._scenarios is None:
            raw = json.loads(self.path.read_text())
            self._scenarios = raw if isinstance(raw, list) else (
                raw.get("scenarios") or raw.get("rows") or [])
        return self._scenarios

    def _idx(self) -> dict[str, dict]:
        if self._index is None:
            self._index = {s["opp_id"]: s for s in self.load_scenarios()
                           if s.get("opp_id")}
        return self._index

    def fetch_opp_meta(self, opp_id: str) -> dict | None:
        s = self._idx().get(opp_id)
        return _opp_meta_from_scenario(s) if s else None

    def fetch_messages(self, opp_id: str) -> list[dict]:
        s = self._idx().get(opp_id)
        return [dict(m) for m in (s.get("seed_messages") or [])] if s else []

    def fetch_business_rules(self, company: str) -> str:
        # Not bundled with the standalone JSON corpus; engines work without it.
        return ""

    def fetch_anchors(self, opp_id: str, opp_meta: dict | None = None) -> dict:
        if opp_meta and (opp_meta.get("anchors") or opp_meta.get("_anchors")):
            return opp_meta.get("anchors") or opp_meta.get("_anchors") or {}
        s = self._idx().get(opp_id)
        return (s.get("anchors") or {}) if s else {}


# ── Active-source registry ───────────────────────────────────────────────────

_ACTIVE: ScenarioSource | None = None


def get_scenario_source() -> ScenarioSource:
    """The active scenario source (defaults to the bundled JSON file)."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = JsonScenarioSource()
    return _ACTIVE


def set_scenario_source(source: ScenarioSource) -> ScenarioSource:
    """Install a custom scenario source for subsequent loads."""
    global _ACTIVE
    _ACTIVE = source
    return source
