"""POC FastAPI server — Persuasion Agent Benchmark.

Endpoints:
  GET  /                       → serves the single-page app
  GET  /api/engines            → list registered engines (drives the UI selectors)
  GET  /api/domains            → list registered domain packs
  GET  /api/scenarios          → list benchmark scenarios (v1 dataset)
  GET  /api/scenarios/{opp_id} → full scenario detail + transcript metadata
  POST /api/run/{opp_id}       → start a replay session
  WS   /ws/{session_id}        → stream events
  POST /api/run/{session_id}/control → speed / pause / reset

Run: python3 server/main.py  (binds 0.0.0.0:8443 — only port open outside container)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Load .env from project root BEFORE any module that reads env vars.
# Critical: supervisor_full + cg_probe + chain stages read LIGHTRAG_API_KEY
# at import time; if .env isn't loaded first, those calls silently no-op.
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    pass

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── POC package layout: server/ is the FastAPI app, poc/ holds the engines
# and substrate (customer_simulator, actor, gates), poc/strategist/ holds
# the Strategist's prompt chain, poc/planner/ is the Planner. All four dirs go
# on sys.path so the modules can find each other with their original flat
# imports.
_SERVER_DIR  = os.path.dirname(os.path.abspath(__file__))
_POC_ROOT    = os.path.dirname(_SERVER_DIR)
_POC_PKG     = os.path.join(_POC_ROOT, "poc")
# Insert in REVERSE order — the last one inserted ends up at position 0,
# which means server/ MUST be inserted last so its db.py shim wins over the
# MySQL-talking copy at poc/db.py.
for _p in (
    os.path.join(_POC_PKG, "strategist", "runners"),
    os.path.join(_POC_PKG, "strategist"),
    _POC_PKG,
    _SERVER_DIR,
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Surface bundled-data paths to modules that read env at import time
os.environ.setdefault("POC_DATA_ROOT", os.path.join(_POC_ROOT, "data"))
os.environ.setdefault("POC_SCRIPT_LIBRARY_DIR",
                       os.path.join(_POC_ROOT, "data", "script_library"))
os.environ.setdefault("POC_CONCRETE_MOVES_DIR",
                       os.path.join(_POC_ROOT, "data", "concrete_moves"))
os.environ.setdefault("POC_RUNNERS_PATH",
                       os.path.join(_POC_PKG, "strategist", "runners"))
os.environ.setdefault("POC_SIM_V2_REFERENCE", "on")
os.environ.setdefault("POC_PLANNER_GATES", "on")

from db import (
    open_conn, fetch_opp_meta, fetch_messages, fetch_turn_states,
    fetch_persuasive_scores, find_failure_mode_turn_index,
    find_supervisor_intervention_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

POC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_DIR = f"{POC_ROOT}/client"

# In-memory session store (single-user demo)
SESSIONS: dict[str, dict] = {}


async def _prewarm_ecommerce_anchors():
    """T-86 — pre-fetch Ecommerce CG anchor pack at server startup so the first
    Ecommerce session doesn't pay the 6s cold-fetch cost."""
    try:
        from ecommerce_anchors import fetch_ecommerce_anchors
        a = await fetch_ecommerce_anchors()
        n_ok = a.get("_cg_queries_returned_content", 0)
        n_total = a.get("_cg_queries_total", 0)
        log.info("Ecommerce anchor cache pre-warmed: %d/%d fields in %sms",
                 n_ok, n_total, a.get("_fetch_ms"))
    except Exception as e:
        log.warning("Ecommerce anchor pre-warm failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("POC server starting on port 8443 — port binds NOW; "
             "Ecommerce anchors warm in background")
    ecommerce_task = asyncio.create_task(_prewarm_ecommerce_anchors())
    yield
    ecommerce_task.cancel()
    log.info("POC server shutting down")


app = FastAPI(title="Persuasion Agent Benchmark", lifespan=lifespan)


_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/")
async def root():
    return FileResponse(f"{CLIENT_DIR}/index.html", headers=_NO_CACHE_HEADERS)


@app.get("/logs")
async def logs_page():
    return FileResponse(f"{CLIENT_DIR}/logs.html", headers=_NO_CACHE_HEADERS)


# Serve client assets
app.mount("/static", StaticFiles(directory=CLIENT_DIR), name="static")


@app.get("/api/engines")
async def list_engines():
    """Available Strategy/Supervisor engines, from the pluggable registry.

    The UI renders L/R panel selectors and per-engine parameter controls from
    this response — engines are no longer hardcoded in the client. An engine
    added in-tree (poc.registry.register) or via a `strategist.engines` entry
    point appears here automatically.
    """
    from registry import all_specs
    return {"engines": [s.to_public() for s in all_specs()]}


@app.get("/api/domains")
async def list_domains():
    """Available domain packs (persona framing + win/lose detection), from the
    pluggable domain registry. The UI's domain selector is populated from this;
    selecting one scopes the scenario dataset + sets outcome detection. Only
    'sales' ships today — the list grows as domain packs are registered.
    """
    from domain import all_domains, active_domain
    active = active_domain()
    active_name = getattr(active, "name", None)
    out = []
    for d in all_domains():
        name = getattr(d, "name", "generic")
        out.append({
            "id": name,
            "name": getattr(d, "display_name", None) or name.replace("_", " ").title(),
            "description": getattr(d, "description", "") or "",
            "active": name == active_name,
        })
    return {"domains": out}


V1_SCENARIOS_FILE = f"{POC_ROOT}/data/benchmark/v1_scenarios.json"

# Fields safe + useful for the picker list. seed_messages (transcripts) are
# deliberately excluded — heavy, and the list only needs persona metadata.
_LIST_FIELDS = ("scenario_id", "opp_id", "tenant", "diversity_bucket",
                "attributes", "real_outcome", "is_sentinel", "anchor_real")

# mtime-cached benchmark dataset (a few MB — don't re-parse per request)
_V1_CACHE: tuple[float, list] | None = None


def _load_v1_scenarios() -> list[dict]:
    global _V1_CACHE
    mtime = os.path.getmtime(V1_SCENARIOS_FILE)
    if _V1_CACHE is None or _V1_CACHE[0] != mtime:
        with open(V1_SCENARIOS_FILE) as f:
            _V1_CACHE = (mtime, json.load(f))
    return _V1_CACHE[1]


def _benchmark_scenarios(pack: str | None = None) -> list[dict]:
    """Scenarios for one benchmark pack, or the union of all packs (deduped by
    opp_id). Falls back to the bundled v1 dataset if no packs are installed."""
    from benchmark_packs import all_packs, load_pack_scenarios
    if pack:
        return load_pack_scenarios(pack)        # KeyError → 404 at call site
    seen: dict[str, dict] = {}
    for p in all_packs():
        try:
            for s in load_pack_scenarios(p["id"]):
                oid = s.get("opp_id")
                if oid and oid not in seen:
                    seen[oid] = s
        except Exception as e:
            log.warning("pack %s: scenario load failed: %s", p["id"], e)
    return list(seen.values()) if seen else _load_v1_scenarios()


@app.get("/api/benchmarks")
async def list_benchmarks():
    """Benchmark packs — goal-oriented scenario bundles under benchmarks/
    (see benchmarks/_template/). Drives the UI's Benchmark selector; a new
    pack directory appears here with no core edits."""
    from benchmark_packs import all_packs, load_pack_scenarios
    out = []
    for p in all_packs():
        try:
            n = len(load_pack_scenarios(p["id"]))
        except Exception as e:
            log.warning("pack %s: scenario load failed: %s", p["id"], e)
            n = None
        out.append({"id": p["id"], "name": p["name"],
                    "description": p.get("description", ""),
                    "goal": p.get("goal", ""),
                    "domain": p.get("domain", "sales"),
                    "n_scenarios": n})
    return {"benchmarks": out}


@app.get("/api/scenarios")
async def list_scenarios(pack: str | None = None):
    """Scenario picker list (persona metadata only — seed_messages excluded).
    Optional ?pack=<id> scopes to one benchmark pack; default is the union of
    all installed packs."""
    try:
        scenarios = _benchmark_scenarios(pack)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown benchmark pack: {pack}")
    return [{k: s[k] for k in _LIST_FIELDS if k in s} for s in scenarios]


@app.get("/api/scenarios/{opp_id}")
async def get_scenario(opp_id: str):
    """Full scenario detail: the benchmark record (minus seed_messages —
    transcripts stream over /ws) + transcript metadata via the scenario-backed
    DB shim (same dataset)."""
    s = next((s for s in _benchmark_scenarios() if s["opp_id"] == opp_id), None)
    if s is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    s = {k: v for k, v in s.items() if k != "seed_messages"}

    # Enrich with transcript metadata (don't ship full transcript over /api — that's for /ws)
    conn = open_conn()
    try:
        meta = fetch_opp_meta(conn, opp_id)
        msgs = fetch_messages(conn, opp_id)
        ts = fetch_turn_states(conn, opp_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="Opportunity not found in DB")

        n_msgs = len(msgs)
        n_inbound = sum(1 for m in msgs if m.get("direction") == "inbound")
        n_outbound = sum(1 for m in msgs if m.get("direction") == "outbound")
        max_commit = max((t.get("commitment_level") or 0 for t in ts), default=0)
        failure_idx = find_failure_mode_turn_index(msgs)
        # 2026-05-12 — Surface the auto-computed supervisor-intervention index
        # so the UI seed slider can default to it (lets researchers adjust ±
        # from the auto choice instead of always starting at 0).
        try:
            persuasive = fetch_persuasive_scores(conn, opp_id)
            auto_failure_idx = find_supervisor_intervention_index(msgs, ts, persuasive)
        except Exception:
            auto_failure_idx = failure_idx
    finally:
        conn.close()

    return {
        **s,
        "n_msgs": n_msgs,
        "n_inbound": n_inbound,
        "n_outbound": n_outbound,
        "historical_max_commit": int(max_commit),
        "historical_outcome": s.get("real_outcome"),
        "failure_mode_turn_index": failure_idx,
        "auto_failure_idx": int(auto_failure_idx) if auto_failure_idx else 0,
        "profile": {
            "motivator": meta.get("primary_motivator"),
            "decision_logic": meta.get("decision_logic"),
            "trust_level": meta.get("trust_level"),
            "objection_pattern": meta.get("objection_pattern"),
            "communication_style": meta.get("communication_style"),
        },
    }


@app.post("/api/run/{opp_id}")
async def start_run(opp_id: str, body: dict | None = Body(default=None)):
    """Initialize a replay session. Returns session_id for WebSocket connection.

    Optional body fields:
      hard_customer: bool — if true, this session's customer simulator runs
        with the adversarial overlay (R10). Overrides the POC_HARD_CUSTOMER
        env var for THIS session only.
    """
    session_id = str(uuid.uuid4())
    hard_customer = bool((body or {}).get("hard_customer"))
    # 2026-05-12 — Optional override for seed-end turn (failure_idx). 0 / unset
    # means "use the peak-engagement auto-detector". Any positive int overrides
    # the detector to let researchers test the supervisor against different
    # seed depths (more seed = supervisor sees more history; less = harder
    # cold-start).
    seed_end_override = int((body or {}).get("seed_end_override") or 0)
    # Per-panel engine selection (any-vs-any A/B). Each panel's engine id is
    # validated against the pluggable registry — no hardcoded whitelist — so a
    # newly-registered engine is accepted with no edit here. Defaults reproduce
    # the classic pairing: LEFT = baseline (control), RIGHT = strategist.
    from registry import has as _engine_known, get as _engine_get

    def _resolve(engine_id: str, fallback: str) -> str:
        eid = (engine_id or fallback).lower()
        if not _engine_known(eid):
            log.warning("unknown engine %r requested; falling back to %s", eid, fallback)
            return fallback
        return eid

    engine = _resolve((body or {}).get("engine"), "strategist")            # R-side
    engine_left = _resolve((body or {}).get("engine_left"), "baseline")    # L-side

    def _coerce_params(engine_id: str, raw: dict | None) -> dict:
        """Keep only params the engine declares; coerce a bare planner_envelope."""
        raw = dict(raw or {})
        try:
            spec = _engine_get(engine_id)
        except Exception:
            return {}
        allowed = {p.name for p in spec.params}
        return {k: v for k, v in raw.items() if k in allowed}

    # Back-compat: a top-level `planner_envelope` (bare true/1 == "always") maps
    # onto the R-side engine's params.
    _pe_raw = (body or {}).get("planner_envelope")
    _pe = None
    if _pe_raw is not None:
        if _pe_raw is True:
            _pe = "always"
        else:
            _pe_s = str(_pe_raw).strip().lower()
            _pe = ("always" if _pe_s in ("always", "true", "1", "on", "yes")
                   else "auto" if _pe_s == "auto" else "off")

    engine_params = _coerce_params(engine, (body or {}).get("engine_params"))
    if _pe is not None and "planner_envelope" not in engine_params and engine == "planner":
        engine_params["planner_envelope"] = _pe
    engine_params_left = _coerce_params(engine_left, (body or {}).get("engine_params_left"))
    # Legacy field still surfaced for the existing UI/log lines.
    planner_envelope = engine_params.get("planner_envelope", "off")

    SESSIONS[session_id] = {
        "opp_id": opp_id,
        "status": "ready",
        "speed": "5x",
        "started_at": None,
        "hard_customer": hard_customer,
        "seed_end_override": seed_end_override,
        "engine": engine,
        "engine_left": engine_left,
        "engine_params": engine_params,
        "engine_params_left": engine_params_left,
        "planner_envelope": planner_envelope,
    }
    log.info("Session %s ready for opp %s [L=%s R=%s%s]%s%s",
             session_id[:8], opp_id, engine_left, engine,
             f"+params:{engine_params}" if engine_params else "",
             " [HARD CUSTOMER]" if hard_customer else "",
             f" [SEED_OVERRIDE={seed_end_override}]" if seed_end_override > 0 else "")
    return {"session_id": session_id, "opp_id": opp_id,
            "hard_customer": hard_customer,
            "seed_end_override": seed_end_override,
            "engine": engine,
            "engine_left": engine_left,
            "engine_params": engine_params,
            "engine_params_left": engine_params_left,
            "planner_envelope": planner_envelope}


@app.websocket("/ws/{session_id}")
async def websocket_session(ws: WebSocket, session_id: str):
    await ws.accept()
    if session_id not in SESSIONS:
        await ws.send_json({"event": "error", "message": "Unknown session"})
        await ws.close()
        return

    sess = SESSIONS[session_id]
    opp_id = sess["opp_id"]

    await ws.send_json({
        "event": "session_ready",
        "opp_id": opp_id,
        "session_id": session_id,
    })

    # Send-fn closure for the replayer
    async def send(event: dict):
        try:
            await ws.send_json(event)
        except Exception:
            pass

    def get_speed():
        return sess.get("speed", "5x")

    # Run the replayer + a control listener concurrently
    from replayer import run_session, request_stop

    async def control_listener():
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("action") == "speed":
                    sess["speed"] = msg.get("speed", "5x")
                elif msg.get("action") == "stop":
                    # Set the cooperative-stop flag so in-flight turn loops bail fast
                    request_stop(session_id)
                    log.info("Stop requested for session %s", session_id[:8])
                    return
                elif msg.get("action") == "ping":
                    await ws.send_json({"event": "pong"})
        except WebSocketDisconnect:
            request_stop(session_id)
            return

    # Pull scenario meta for the logger — benchmark record minus the heavy
    # seed transcript (the replayer fetches messages itself via the db shim).
    scenario_meta = None
    try:
        for s in _benchmark_scenarios():
            if s["opp_id"] == opp_id:
                scenario_meta = {k: v for k, v in s.items() if k != "seed_messages"}
                break
    except Exception:
        pass

    hard_customer = bool(sess.get("hard_customer", False))
    seed_end_override = int(sess.get("seed_end_override") or 0)
    engine = sess.get("engine", "strategist")
    engine_left = sess.get("engine_left", "baseline")
    planner_envelope = sess.get("planner_envelope", "off")
    engine_params = sess.get("engine_params") or {}
    engine_params_left = sess.get("engine_params_left") or {}
    replayer_task = asyncio.create_task(
        run_session(session_id, opp_id, send, get_speed, scenario_meta,
                    hard_customer=hard_customer,
                    seed_end_override=seed_end_override,
                    engine=engine,
                    planner_envelope=planner_envelope,
                    engine_left=engine_left,
                    engine_params=engine_params,
                    engine_params_left=engine_params_left))
    control_task = asyncio.create_task(control_listener())

    try:
        # Whichever finishes first ends the session
        done, pending = await asyncio.wait(
            [replayer_task, control_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
    except WebSocketDisconnect:
        log.info("WS disconnected: %s", session_id[:8])
        replayer_task.cancel()
        control_task.cancel()
    finally:
        SESSIONS.pop(session_id, None)
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/api/trace/list")
async def trace_list(limit: int = 200):
    """List recent trace JSON files across all date dirs. Most recent first.
    2026-05-06 — extended to pair each trace with its session-log metrics
    (persuasion L/R, outcome, message-length compliance, hard-mode flag) so
    the /logs UI can render a historical-trends chart across all sessions."""
    import glob, os, json, re
    from trace_logger import LOGS_BASE
    files = sorted(glob.glob(f"{LOGS_BASE}/*/*_trace.json"),
                   key=os.path.getmtime, reverse=True)[:limit]
    out = []
    for path in files:
        try:
            with open(path) as f:
                body = json.load(f)
            scen = body.get("scenario_meta") or {}
            summ = body.get("summary") or {}
            session_id = body.get("session_id") or ""
            # Pair with session log file (same directory, name without _trace)
            session_path = path.replace("_trace.json", ".json")
            session_metrics = {}
            r_msg_lens = []
            is_hard = False
            if os.path.exists(session_path):
                try:
                    s = json.load(open(session_path))
                    p = s.get("persuasion", {}) or {}
                    o = s.get("outcomes", {}) or {}
                    L = p.get("left", {}) or {}
                    R = p.get("right", {}) or {}
                    Lo = (o.get("left") or {})
                    Ro = (o.get("right") or {})
                    notes = s.get("notes", []) or []
                    is_hard = any("hard_customer" in str(n) for n in notes)
                    # R-side message lengths
                    for ev in s.get("events", []) or []:
                        if ev.get("event") == "right_msg" and ev.get("role") == "agent":
                            txt = ev.get("text", "") or ""
                            if txt:
                                r_msg_lens.append(len(re.split(r"\s+", txt.strip())))
                    session_metrics = {
                        "L_max": L.get("max"),
                        "R_max": R.get("max"),
                        "L_end": L.get("end"),
                        "R_end": R.get("end"),
                        "L_outcome": Lo.get("outcome"),
                        "R_outcome": Ro.get("outcome"),
                        "L_reason": Lo.get("reason"),
                        "R_reason": Ro.get("reason"),
                        "delta_max": (
                            (R.get("max") or 0) - (L.get("max") or 0)
                            if (R.get("max") is not None and L.get("max") is not None)
                            else None
                        ),
                        "r_msg_count": len(r_msg_lens),
                        "r_msg_median_len": (sorted(r_msg_lens)[len(r_msg_lens)//2]
                                             if r_msg_lens else None),
                        "r_msg_max_len": max(r_msg_lens) if r_msg_lens else None,
                        "r_msg_pct_over_35": (
                            round(sum(1 for L in r_msg_lens if L > 35) /
                                  max(1, len(r_msg_lens)) * 100, 0)
                            if r_msg_lens else None
                        ),
                    }
                except Exception:
                    pass
            out.append({
                "session_id": session_id,
                "opp_id": body.get("opp_id"),
                "started_at": body.get("started_at"),
                "duration_s": summ.get("duration_s"),
                "n_events": summ.get("n_events"),
                "n_llm_calls": sum(v.get("calls", 0) for v in summ.get("llm_by_stage", {}).values()),
                "n_cg_calls": sum(v.get("calls", 0) for v in summ.get("cg_by_endpoint", {}).values()),
                "n_gate_firings": sum(v.get("firings", 0) for v in summ.get("gate_by_name", {}).values()),
                "company": scen.get("company") or scen.get("tenant"),
                "cluster_id": scen.get("cluster_id"),
                "opp_type": scen.get("opp_type"),
                "demo_label": scen.get("demo_label") or scen.get("demo_short"),
                "is_hard": is_hard,
                **session_metrics,
                "path": os.path.basename(path),
            })
        except Exception as e:
            log.warning("trace_list: failed to parse %s: %s", path, e)
    return {"traces": out}


@app.get("/api/trace/{session_id}")
async def trace_get(session_id: str):
    """Return the full trace JSON for a session. Looks across date dirs."""
    import glob, json
    from trace_logger import LOGS_BASE
    matches = glob.glob(f"{LOGS_BASE}/*/{session_id[:8]}_*_trace.json")
    if not matches:
        raise HTTPException(status_code=404, detail=f"trace not found for session {session_id}")
    with open(matches[0]) as f:
        return json.load(f)


@app.get("/health")
async def health():
    """Liveness + benchmark-dataset sanity. `db` only does a real ping in
    MySQL passthrough mode (POC_USE_MYSQL=1); the default JSON shim serves
    from the same dataset counted here."""
    try:
        n_scenarios = len(_load_v1_scenarios())
    except Exception as e:
        log.warning("health: benchmark dataset unavailable: %s", e)
        n_scenarios = 0
    db_ok = True
    if os.environ.get("POC_USE_MYSQL") == "1":
        try:
            conn = open_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                cur.fetchone()
            conn.close()
        except Exception as e:
            db_ok = False
            log.warning("DB health failed: %s", e)
    return {"status": "ok", "scenarios": n_scenarios, "db": db_ok}


if __name__ == "__main__":
    _port = int(os.environ.get("POC_PORT", "8443"))
    uvicorn.run("main:app", host="0.0.0.0", port=_port, reload=False, log_level="info")
