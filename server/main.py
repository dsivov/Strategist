"""POC FastAPI server — Sales Strategist Benchmark demo.

Endpoints:
  GET  /                       → serves the single-page app
  GET  /api/engines            → list registered engines (drives the UI selectors)
  GET  /api/scenarios          → list curated scenarios
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

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
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
SCENARIOS_FILE = f"{POC_ROOT}/data/scenarios.json"
CLIENT_DIR = f"{POC_ROOT}/client"

# In-memory session store (single-user demo)
SESSIONS: dict[str, dict] = {}


CANDIDATE_CACHE: dict | None = None


async def _build_cache_in_background():
    global CANDIDATE_CACHE
    try:
        from random_match import load_or_build_candidate_cache
        result = await asyncio.to_thread(load_or_build_candidate_cache)
        CANDIDATE_CACHE = result
        log.info("random_match cache: %d candidates loaded",
                  len(CANDIDATE_CACHE) if CANDIDATE_CACHE else 0)
    except Exception as e:
        log.warning("random_match cache build failed: %s", e)
        CANDIDATE_CACHE = {}


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
             "random_match cache + Ecommerce anchors warm in background")
    cache_task = asyncio.create_task(_build_cache_in_background())
    ecommerce_task = asyncio.create_task(_prewarm_ecommerce_anchors())
    yield
    cache_task.cancel()
    ecommerce_task.cancel()
    log.info("POC server shutting down")


app = FastAPI(title="Sales Strategist Benchmark POC", lifespan=lifespan)


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


@app.get("/api/scenarios")
async def list_scenarios():
    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)
    # Strip win_rate (internal) from public response
    public = [{k: v for k, v in s.items() if k != "v1_win_rate"} for s in scenarios]
    return public


@app.get("/api/scenarios/{opp_id}")
async def get_scenario(opp_id: str):
    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)
    s = next((s for s in scenarios if s["opp_id"] == opp_id), None)
    if s is None:
        # Random-match path — opp not in scenarios.json; build minimal stub from cache
        if CANDIDATE_CACHE and opp_id in CANDIDATE_CACHE:
            cached = CANDIDATE_CACHE[opp_id]
            s = {
                "opp_id": opp_id,
                "tenant": cached.get("tenant"),
                "cluster_id": cached.get("cluster_id") or 0,
                "cluster_name": "random match",
                "motivator": cached.get("motivator"),
                "decision_logic": cached.get("decision_logic"),
                "expected_lift_label": "random match (no precomputed v1 data)",
                "v1_win_rate": None,
            }
        else:
            raise HTTPException(status_code=404, detail="Scenario not found")

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
        "historical_outcome": meta.get("status"),
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


@app.get("/api/random_match/criteria_options")
async def random_match_criteria_options():
    """Return available values for each criteria axis, so the UI can populate
    dropdowns from real cache data instead of hardcoding lists."""
    if not CANDIDATE_CACHE:
        return {"error": "candidate cache not built"}
    tenants = sorted({c["tenant"] for c in CANDIDATE_CACHE.values() if c.get("tenant")})
    opp_types = sorted({c["opp_type"] for c in CANDIDATE_CACHE.values() if c.get("opp_type")})
    motivators = sorted({c["motivator"] for c in CANDIDATE_CACHE.values() if c.get("motivator")})
    decision_logics = sorted({c["decision_logic"] for c in CANDIDATE_CACHE.values() if c.get("decision_logic")})
    trust_levels = sorted({c["trust_level"] for c in CANDIDATE_CACHE.values() if c.get("trust_level")})
    return {
        "tenants": tenants,
        "opp_types": opp_types,
        "motivators": motivators,
        "decision_logics": decision_logics,
        "trust_levels": trust_levels,
        "n_total_in_cache": len(CANDIDATE_CACHE),
        "n_with_plan": sum(1 for c in CANDIDATE_CACHE.values() if c["has_plan"]),
    }


@app.post("/api/random_match")
async def random_match_endpoint(criteria: dict):
    """Find best clean-loss candidate matching audience-defined criteria.
    Hierarchical relaxation on empty result; prefer-with-plan partition.
    Body: {tenant, opp_type, motivator, decision_logic, trust_level} — any subset.
    """
    if not CANDIDATE_CACHE:
        raise HTTPException(status_code=503, detail="candidate cache not built")
    from random_match import find_best_match
    return find_best_match(criteria or {}, CANDIDATE_CACHE,
                            prefer_with_plan=True)


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

    # Pull scenario meta for the logger
    scenario_meta = None
    try:
        with open(SCENARIOS_FILE) as f:
            for s in json.load(f):
                if s["opp_id"] == opp_id:
                    scenario_meta = s
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


@app.get("/api/cohort_weights")
async def cohort_weights():
    """Return the auto-generated cohort_weights.yaml as JSON. Research artifact;
    NOT consulted by the supervisor at runtime yet. Surfaced for the /logs UI
    so reviewers can inspect what the per-cohort tactical priors look like.
    Run `python3 server/build_cohort_weights.py` to regenerate."""
    import os, re
    weights_path = f"{POC_ROOT}/data/cohort_weights.yaml"
    if not os.path.exists(weights_path):
        return JSONResponse(
            status_code=404,
            content={"error": "cohort_weights.yaml not yet generated",
                     "hint": "run: python3 server/build_cohort_weights.py"})
    # Minimal YAML→dict parse (avoid PyYAML dep).
    # The file is structured + flat enough for a hand-rolled parser.
    txt = open(weights_path).read()
    try:
        import yaml
        return yaml.safe_load(txt)
    except ImportError:
        # Fallback: return raw text for client-side display
        return {"_raw_yaml": txt,
                "_note": "PyYAML not installed; client should display raw text"}


@app.get("/api/cache_status")
async def cache_status():
    """random_match cache build state. UI polls this to know when 'Find best
    match' is usable. Cache builds at server startup (~2-3 min), so on a
    fresh start the random tab shows a building indicator."""
    if CANDIDATE_CACHE is None:
        return {"ready": False, "n_candidates": 0, "state": "building"}
    return {"ready": True, "n_candidates": len(CANDIDATE_CACHE), "state": "ready"}


@app.get("/api/precedents")
async def get_precedents(request: Request):
    """Phase 2 — cohort-conditioned precedent retrieval (two-tier router).

    Query params (all optional unless noted):
      company                       Insurance | Ecommerce
      outcome                       ClosedWon (default) | ClosedLost
      decision_logic, primary_motivator, budget_sensitivity, communication_style,
        regulatory_focus, trust_level, purchase_urgency, primary_resistance,
        profile_tone, gender, age_range, engagement_level, responsiveness,
        emotional_volatility, authority, social_proof_susceptibility,
        tech_savviness, risk_tolerance, education_level, family_status,
        sentiment_trend, time_pressure        (raw cohort dims — equality)
      primary_strategy, secondary_strategy, strategy_tone   (strategy filters)
      objection_category, sentiment                         (phase signals)
      commitment_level_min, commitment_level_max            (range, integer)
      min_persuasion_score, min_p_conv                      (float [0,1])
      limit                                                 (default 20, max 100)

    Response carries `tier` ∈ {sqlite, cg, sqlite+cg, empty, error} so callers
    can attribute which retrieval substrate served (per substrate doc §3, §7).
    """
    from precedent_retrieval import fetch_precedents

    raw_filters = dict(request.query_params)
    async with httpx.AsyncClient() as http:
        result = await fetch_precedents(raw_filters, http_client=http)

    if result.get("tier") == "error":
        return JSONResponse(status_code=400, content=result)
    return result


@app.get("/api/precedents/meta")
async def precedents_meta():
    """Substrate freshness + cache stats. Used by Phase 3 validation harness
    and by the /logs UI to surface 'precedents.db built_at' for traceability."""
    from precedent_retrieval import db_meta, cache_stats
    return {
        "db": db_meta(),
        "cache": cache_stats(),
    }


@app.get("/api/historical_persuasion/{opp_id}")
async def historical_persuasion(opp_id: str):
    """Return per-turn persuasion_score from the REAL (historical) conversation
    for this opp_id. Used by the UI to overlay a 'real salesman' line on the
    persuasion-over-time chart so the user can compare our supervised agent
    against actual historical performance.

    Joins research_turn_state_flash (per-turn persuasion + commitment)
    with message_event (sequence_number → timestamp + direction). Returns
    points indexed by sequence_number. Read-only.
    """
    try:
        conn = open_conn()
        cur = conn.cursor()
        # Pull turn-state per message_id; join with event direction so the UI
        # can distinguish customer vs agent turns. Order by timestamp.
        cur.execute(
            """
            SELECT ts.sequence_number   AS seq,
                   ts.persuasion_score  AS persuasion,
                   ts.commitment_level  AS commit,
                   ts.p_conv            AS p_conv,
                   me.type              AS direction,
                   me.timestamp         AS ts_at
            FROM research_turn_state_flash ts
            JOIN message_event me
              ON me.message_id     = ts.message_id
             AND me.opportunity_id = ts.opportunity_id
            WHERE ts.opportunity_id = %s
            ORDER BY me.timestamp
            """,
            (opp_id,),
        )
        rows = cur.fetchall()
        conn.close()
        # De-dupe per seq (keep highest persuasion in case of duplicates from
        # the LLM-inferred labeling re-running).
        by_seq = {}
        for r in rows:
            seq = r.get("seq")
            if seq is None:
                continue
            cur_seq = by_seq.get(seq)
            if cur_seq is None or (r.get("persuasion") or 0) > (cur_seq.get("persuasion") or 0):
                by_seq[seq] = r
        points = []
        for seq in sorted(by_seq.keys()):
            r = by_seq[seq]
            points.append({
                "turn":       int(seq),
                "persuasion": float(r["persuasion"]) if r.get("persuasion") is not None else None,
                "commit":     int(r["commit"])      if r.get("commit")     is not None else None,
                "p_conv":     float(r["p_conv"])    if r.get("p_conv")     is not None else None,
                "role":       "agent" if r.get("direction") == "outbound" else "customer",
            })
        return {"opp_id": opp_id, "points": points, "n": len(points)}
    except Exception as e:
        log.warning("historical_persuasion(%s) failed: %s", opp_id, e)
        return {"opp_id": opp_id, "points": [], "n": 0, "error": str(e)[:200]}


@app.get("/health")
async def health():
    try:
        conn = open_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            cur.fetchone()
        conn.close()
        db_ok = True
    except Exception as e:
        db_ok = False
        log.warning("DB health failed: %s", e)
    return {"status": "ok", "db": db_ok}


if __name__ == "__main__":
    _port = int(os.environ.get("POC_PORT", "8443"))
    uvicorn.run("main:app", host="0.0.0.0", port=_port, reload=False, log_level="info")
