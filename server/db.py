"""JSON-backed db shim for the standalone POC server.

The upstream `db.py` (in `poc/db.py`) talks to the production system MySQL. This
shim provides the SAME public API but serves data from the bundled
`data/benchmark/v1_scenarios.json` file, so the server can run without
prod credentials.

Activation:
  - If `POC_USE_MYSQL=1` is set in the environment, this module proxies
    every call into the real MySQL-backed db module (must be importable).
  - Otherwise (the default for this handoff package), it serves from JSON.

Public API mirrored 1:1:
  open_conn() -> Conn-like sentinel
  fetch_opp_meta(conn, opp_id) -> dict | None
  fetch_messages(conn, opp_id) -> list[dict]
  fetch_turn_states(conn, opp_id) -> list[dict]
  fetch_persuasive_scores(conn, opp_id) -> dict[str, dict]
  fetch_business_rules(conn, company) -> str
  fetch_insurance_anchors(conn, opp_id, opp_meta=None) -> dict
  find_failure_mode_turn_index(messages) -> int | None
  find_supervisor_intervention_index(messages, turn_states, persuasive) -> int | None
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── MySQL passthrough mode ──────────────────────────────────────────────────

if os.environ.get("POC_USE_MYSQL") == "1":
    # Re-export the real MySQL-backed module.
    from poc.db import (  # type: ignore  # noqa: F401
        open_conn,
        fetch_opp_meta,
        fetch_messages,
        fetch_turn_states,
        fetch_persuasive_scores,
        fetch_business_rules,
        fetch_insurance_anchors,
        find_failure_mode_turn_index,
        find_supervisor_intervention_index,
    )

    log.info("db.py: MySQL passthrough mode (POC_USE_MYSQL=1)")

else:
    # ── JSON-backed mode (default) ──────────────────────────────────────────

    _DATA = Path(os.environ.get("POC_DATA_ROOT",
                                 str(Path(__file__).resolve().parent.parent / "data")))
    _SCENARIOS_FILE = _DATA / "benchmark" / "v1_scenarios.json"

    _LOCK   = threading.Lock()
    _INDEX: dict[str, dict] | None = None

    def _load_index() -> dict[str, dict]:
        """Build an opp_id -> scenario dict lookup on first call."""
        global _INDEX
        if _INDEX is not None:
            return _INDEX
        with _LOCK:
            if _INDEX is not None:
                return _INDEX
            index: dict[str, dict] = {}
            if _SCENARIOS_FILE.exists():
                raw = json.loads(_SCENARIOS_FILE.read_text())
                scs = raw if isinstance(raw, list) else (
                    raw.get("scenarios") or raw.get("rows") or [])
                index = {s["opp_id"]: s for s in scs if s.get("opp_id")}
                log.info("db.py: loaded %d scenarios from %s",
                          len(index), _SCENARIOS_FILE)
            else:
                log.error("db.py: scenarios file not found at %s", _SCENARIOS_FILE)
            # Merge benchmark-pack datasets (benchmarks/*/pack.json) so a pack
            # that brings its own scenario file also works in the live server.
            try:
                from benchmark_packs import all_packs, load_pack_scenarios
                for p in all_packs():
                    try:
                        added = 0
                        for s in load_pack_scenarios(p["id"]):
                            oid = s.get("opp_id")
                            if oid and oid not in index:
                                index[oid] = s
                                added += 1
                        if added:
                            log.info("db.py: merged %d scenarios from pack %s",
                                      added, p["id"])
                    except Exception as e:
                        log.warning("db.py: pack %s merge failed: %s",
                                     p["id"], e)
            except Exception:
                pass
            _INDEX = index
            return _INDEX

    # ── Connection sentinel ─────────────────────────────────────────────────

    class _JsonConn:
        """Sentinel passed where the real code expects a MySQL connection.
        All real fetch_* shims ignore it; they read from the JSON index."""
        def cursor(self, *a, **k):  # for callers that do .cursor()
            return _JsonCursor()
        def close(self):
            return None
        def commit(self):
            return None

    class _JsonCursor:
        def execute(self, *a, **k):     return self
        def fetchall(self):              return []
        def fetchone(self):              return None
        def close(self):                 return None

    def open_conn():
        return _JsonConn()

    # ── fetch_opp_meta ──────────────────────────────────────────────────────

    def fetch_opp_meta(conn, opp_id: str) -> dict | None:
        s = _load_index().get(opp_id)
        if not s:
            return None
        attrs = dict(s.get("attributes") or {})
        meta: dict[str, Any] = {
            "id":      s["opp_id"],
            "company": s["tenant"],
        }
        meta.update(attrs)
        if s.get("anchors"):
            meta["anchors"]  = s["anchors"]
            meta["_anchors"] = s["anchors"]
        if s.get("voice_profile"):
            meta["voice_profile"] = s["voice_profile"]
        return meta

    # ── fetch_messages ──────────────────────────────────────────────────────

    def fetch_messages(conn, opp_id: str) -> list[dict]:
        s = _load_index().get(opp_id)
        if not s:
            return []
        msgs = s.get("seed_messages") or []
        # Already in the upstream row shape (direction, text, message_id,
        # timestamp, is_reminder, is_followup). Pass through as-is.
        return [dict(m) for m in msgs]

    # ── stubs for prod-only signal sources ─────────────────────────────────

    def fetch_turn_states(conn, opp_id: str) -> list[dict]:
        """research_turn_state_flash data — not bundled with the standalone
        package. Returns empty list; consumers treat it as 'unknown'."""
        return []

    def fetch_persuasive_scores(conn, opp_id: str) -> dict[str, dict]:
        """persuasive_score table — not bundled. Returns empty dict."""
        return {}

    def fetch_business_rules(conn, company: str) -> str:
        """company_business_info.business_rules — not bundled. The engines
        work without it; tenant-specific compliance text would be added by
        a prod-connected deployment."""
        return ""

    def fetch_insurance_anchors(conn, opp_id: str, opp_meta: dict | None = None) -> dict:
        """The anchor pack is embedded in v1_scenarios.json. If opp_meta was
        passed we use what's already there; otherwise look it up."""
        if opp_meta and (opp_meta.get("anchors") or opp_meta.get("_anchors")):
            return opp_meta.get("anchors") or opp_meta.get("_anchors") or {}
        s = _load_index().get(opp_id)
        return (s.get("anchors") or {}) if s else {}

    # ── pure helpers (no DB access) ─────────────────────────────────────────

    _DEFERRAL_RE = re.compile(
        r"\b(get back to you|if i decide|i'?ll (review|think|consider|let you know|"
        r"check)|think (about|it over)|review (it|the)|before i (decide|confirm|"
        r"commit)|not sure( yet)?|decide later|need to (think|check|discuss|talk)|"
        r"maybe later|i'?ll see|circle back|hold off)\b",
        re.IGNORECASE,
    )

    def find_failure_mode_turn_index(messages: list[dict]) -> int | None:
        """Heuristic: index of the last inbound (customer) message in `messages`.
        Used by the replayer to find the cut point for live takeover.
        """
        last = None
        for i, m in enumerate(messages):
            if m.get("direction") == "inbound":
                last = i
        return last

    def find_supervisor_intervention_index(
        messages: list[dict],
        turn_states: list[dict] | None = None,
        persuasive: dict | None = None,
    ) -> int | None:
        """Same heuristic with optional signals. Without turn_states /
        persuasive (the JSON-mode default), falls back to last-inbound."""
        return find_failure_mode_turn_index(messages)

    log.info("db.py: JSON-backed mode (POC_USE_MYSQL not set); "
              "scenarios from %s", _SCENARIOS_FILE)
