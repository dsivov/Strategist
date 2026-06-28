"""End-to-end trace logger — captures every LLM call, CG call, gate event,
and chain stage I/O per session.

T-86 (2026-05-03). Per-session trace dumps to
`test_logs/{date}/{sid}_trace.json` — separate from the regular session log
so that file stays compact for normal analysis. Trace file contains full
raw prompts + responses + retrieval payloads; intended for end-to-end
diagnostic walkthrough of any session.

Design:
- A TraceLogger instance is created per session in replayer.run_session
- All call sites use TraceLogger.current() (contextvars-based) to find it
  without threading a parameter through every function signature
- Each event recorded as a flat dict with kind, timestamp (relative to
  session start), and structured fields per kind

Privacy / size notes:
- Trace files contain full prompts + responses including tenant business
  rules and customer messages. POC scope (historical lost-data) is fine;
  production deployment of this would require prompt redaction.
- Typical session: ~30-50 LLM calls × ~5KB raw text → ~150-250KB per
  session trace. 20-session batch ≈ 3-5 MB. Fine.
- Files added to .gitignore via test_logs/ pattern.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_BASE = os.environ.get(
    "POC_TEST_LOGS_DIR",
    os.path.join(_PROJECT_ROOT, "test_logs"),
)

# contextvars-based session-scoped trace lookup; call sites use
# TraceLogger.current() to find the active trace without threading params.
_current_trace: contextvars.ContextVar = contextvars.ContextVar("_current_trace", default=None)


class TraceLogger:
    """Per-session trace recorder.

    Usage at session start:
        trace = TraceLogger(session_id, opp_id, scenario_meta=...)
        token = TraceLogger.set_current(trace)
        try:
            ... run session ...
        finally:
            trace.write()
            TraceLogger.reset_current(token)

    Usage at call sites:
        trace = TraceLogger.current()
        if trace:
            trace.llm(...)
    """
    def __init__(self, session_id: str, opp_id: str,
                 scenario_meta: dict | None = None):
        self.session_id = session_id
        self.opp_id = opp_id
        self.scenario_meta = scenario_meta or {}
        self.started_at = time.time()
        self.events: list[dict] = []

    @staticmethod
    def current() -> "TraceLogger | None":
        return _current_trace.get()

    @staticmethod
    def set_current(trace: "TraceLogger") -> Any:
        return _current_trace.set(trace)

    @staticmethod
    def reset_current(token: Any) -> None:
        _current_trace.reset(token)

    # ── Event-recording methods (one per kind) ─────────────────────────────

    def _t(self) -> float:
        """Time elapsed since session start, rounded to 3 decimals."""
        return round(time.time() - self.started_at, 3)

    def llm(self, *, stage: str, provider: str, model: str | None = None,
            system: str = "", user: str = "", response: str = "",
            latency_ms: int = 0, input_tokens: int = 0,
            output_tokens: int = 0, panel_side: str | None = None,
            extra: dict | None = None) -> None:
        """Record an LLM call. `stage` is a free-form name for which call site
        produced this (e.g. 'supervisor.mode1b', 'actor.generate',
        'simulator.persona_reply', 'voice_profile.extract')."""
        ev = {
            "t": self._t(),
            "kind": "llm_call",
            "stage": stage,
            "panel_side": panel_side,
            "provider": provider,
            "model": model,
            "system_chars": len(system or ""),
            "user_chars": len(user or ""),
            "response_chars": len(response or ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "raw_system": system,
            "raw_user": user,
            "raw_response": response,
        }
        if extra:
            ev["extra"] = extra
        self.events.append(ev)

    def cg(self, *, endpoint: str, workspace: str, query: str = "",
           response: dict | None = None, latency_ms: int = 0,
           cache_hit: bool = False, panel_side: str | None = None,
           extra: dict | None = None) -> None:
        """Record a Context Graph call. `endpoint` is e.g. '/query/data',
        '/cgr3/query', '/graph/decisions/search', '/graph/decision/emit'."""
        # Truncate the response to keep size bounded (entities + chunks can
        # be large). Keep top-level structure, sample chunks.
        compact_response = response or {}
        if isinstance(compact_response, dict):
            entities = (compact_response.get("entities") or [])[:20]
            relations = (compact_response.get("relationships") or
                         compact_response.get("relations") or [])[:10]
            chunks = (compact_response.get("chunks") or [])[:8]
            # 2026-05-05 — preserve non-standard response fields too. /graph/
            # decisions returns n_decisions; /graph/decision/emit returns
            # emitted/src/tgt; /cgr3/query returns multi-hop fields. Without
            # this, trace UI shows "empty" cg_calls for working endpoints.
            extras_passthrough = {
                k: v for k, v in compact_response.items()
                if k not in ("entities", "relationships", "relations", "chunks")
            }
            compact_response = {
                "n_entities": len(compact_response.get("entities") or []),
                "n_relations": len(compact_response.get("relationships") or
                                    compact_response.get("relations") or []),
                "n_chunks": len(compact_response.get("chunks") or []),
                "sample_entities": entities,
                "sample_relations": relations,
                "sample_chunks": chunks,
                "_cache_hit": compact_response.get("_cache_hit"),
                "_error": compact_response.get("_error"),
                **extras_passthrough,  # preserves n_decisions, emitted, etc.
            }
        ev = {
            "t": self._t(),
            "kind": "cg_call",
            "endpoint": endpoint,
            "workspace": workspace,
            "panel_side": panel_side,
            "query": query,
            "response": compact_response,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
        }
        if extra:
            ev["extra"] = extra
        self.events.append(ev)

    def gate(self, *, name: str, panel_side: str, verdict: str,
             reason: str | None = None, extra: dict | None = None) -> None:
        """Record a gate firing (anti-staircase, retreat passthrough,
        signal-adherence retry, etc.)"""
        ev = {
            "t": self._t(),
            "kind": "gate_event",
            "gate": name,
            "panel_side": panel_side,
            "verdict": verdict,
            "reason": reason,
        }
        if extra:
            ev["extra"] = extra
        self.events.append(ev)

    def stage(self, *, name: str, panel_side: str | None = None,
              input_summary: dict | None = None,
              output_summary: dict | None = None,
              latency_ms: int = 0, extra: dict | None = None) -> None:
        """Record a chain stage execution (Phase A.3) — captures input
        snapshot (which previous_responses keys were available) and output
        summary."""
        ev = {
            "t": self._t(),
            "kind": "chain_stage",
            "stage": name,
            "panel_side": panel_side,
            "input_summary": input_summary or {},
            "output_summary": output_summary or {},
            "latency_ms": latency_ms,
        }
        if extra:
            ev["extra"] = extra
        self.events.append(ev)

    def note(self, msg: str, **kwargs) -> None:
        """Free-form note for diagnostic context."""
        ev = {"t": self._t(), "kind": "note", "msg": msg}
        if kwargs:
            ev.update(kwargs)
        self.events.append(ev)

    # ── Aggregation + persistence ──────────────────────────────────────────

    def summary(self) -> dict:
        """Compute a top-of-trace summary: per-stage call counts + latency
        totals, gate firings count by name, cg endpoint usage."""
        from collections import defaultdict
        llm_by_stage: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "latency_ms_total": 0, "input_tokens": 0,
                     "output_tokens": 0, "response_chars_total": 0})
        cg_by_endpoint: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "latency_ms_total": 0, "cache_hits": 0})
        gate_by_name: dict[str, dict] = defaultdict(
            lambda: {"firings": 0, "verdicts": defaultdict(int)})
        chain_stage_by_name: dict[str, dict] = defaultdict(
            lambda: {"executions": 0, "latency_ms_total": 0})

        for ev in self.events:
            kind = ev.get("kind")
            if kind == "llm_call":
                key = ev.get("stage") or "?"
                e = llm_by_stage[key]
                e["calls"] += 1
                e["latency_ms_total"] += ev.get("latency_ms") or 0
                e["input_tokens"] += ev.get("input_tokens") or 0
                e["output_tokens"] += ev.get("output_tokens") or 0
                e["response_chars_total"] += ev.get("response_chars") or 0
            elif kind == "cg_call":
                key = ev.get("endpoint") or "?"
                e = cg_by_endpoint[key]
                e["calls"] += 1
                e["latency_ms_total"] += ev.get("latency_ms") or 0
                if ev.get("cache_hit"):
                    e["cache_hits"] += 1
            elif kind == "gate_event":
                key = ev.get("gate") or "?"
                e = gate_by_name[key]
                e["firings"] += 1
                e["verdicts"][ev.get("verdict") or "?"] += 1
            elif kind == "chain_stage":
                key = ev.get("stage") or "?"
                e = chain_stage_by_name[key]
                e["executions"] += 1
                e["latency_ms_total"] += ev.get("latency_ms") or 0

        # Convert defaultdicts to regular dicts (for clean JSON output)
        return {
            "session_id": self.session_id,
            "opp_id": self.opp_id,
            "n_events": len(self.events),
            "duration_s": round(time.time() - self.started_at, 1),
            "llm_by_stage": {k: dict(v) for k, v in llm_by_stage.items()},
            "cg_by_endpoint": {k: dict(v) for k, v in cg_by_endpoint.items()},
            "gate_by_name": {
                k: {"firings": v["firings"],
                    "verdicts": dict(v["verdicts"])}
                for k, v in gate_by_name.items()},
            "chain_stage_by_name": {k: dict(v) for k, v in chain_stage_by_name.items()},
        }

    def write(self) -> str:
        """Dump trace to disk. Returns path written."""
        date_dir = datetime.fromtimestamp(self.started_at).strftime("%Y-%m-%d")
        out_dir = f"{LOGS_BASE}/{date_dir}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{self.session_id[:8]}_{self.opp_id[:8]}_trace.json"
        body = {
            "session_id": self.session_id,
            "opp_id": self.opp_id,
            "scenario_meta": self.scenario_meta,
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(),
            "summary": self.summary(),
            "events": self.events,
        }
        try:
            with open(out_path, "w") as f:
                json.dump(body, f, indent=2, ensure_ascii=False, default=str)
            log.info("Trace written: %s (%d events)", out_path, len(self.events))
        except Exception as e:
            log.warning("Trace write failed: %s", e)
        return out_path
