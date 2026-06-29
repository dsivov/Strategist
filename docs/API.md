# API reference

The public Python surface (`poc.*`) and the server's HTTP/WebSocket endpoints.

- [Python: engines & benchmark](#python-engines--benchmark)
- [Python: engine registry](#python-engine-registry)
- [Python: domain packs](#python-domain-packs)
- [Python: scenario sources](#python-scenario-sources)
- [HTTP: REST endpoints](#http-rest-endpoints)
- [HTTP: the run request](#http-the-run-request)
- [WebSocket](#websocket)

Everything below is importable from the top-level package:

```python
from poc import Benchmark, load_scenarios, all_engine_specs, set_active_domain  # etc.
```

---

## Python: engines & benchmark

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `Engine` | Protocol | `runtime_checkable`; implement `produce`. |
| `BaselineEngine` | `BaselineEngine()` | Single-call actor (no directive). |
| `PlannerEngine` | `PlannerEngine(planner_envelope="off")` | PCA planner + gates. |
| `StrategistEngine` | `StrategistEngine()` | **Raises** — needs the production system DB + KG; source for review. |
| `CustomerSimulator` | `CustomerSimulator(opp_meta, full_history)` | v2 reference-aware simulator. |
| `Benchmark` | `Benchmark(scenarios, results_dir="./benchmark_results", max_turns=12)` | The runner. |
| `load_scenarios` | `load_scenarios(path=None, source=None) -> list[dict]` | See resolution order below. |
| `paired_summary` | `paired_summary({arm: results}) -> dict` | (in `poc.benchmark`) aggregate + pairwise counts. |

**`Engine.produce`:**

```python
async def produce(self, opp_meta: dict, dialog_history: list[dict],
                  business_rules: str = "") -> tuple[str, dict]:
    ...  # returns (customer_facing_text, telemetry_meta)
```

**`Benchmark` methods:**

```python
# run a constructed engine instance under an arm label
await bench.run_arm(arm_name, engine, scenarios=None, on_scenario_done=None) -> list[dict]

# resolve + instantiate from the registry by id (the pluggable entry point)
await bench.run_engine(engine_id, scenarios=None, on_scenario_done=None,
                       arm_name=None, **engine_params) -> list[dict]
```

`load_scenarios` resolution order: explicit `source=` → explicit `path=`
(legacy direct file read) → the active scenario source (bundled JSON default).

Per-scenario results are written to `{results_dir}/{arm}/{scenario_id}.json` and
runs are **resumable** (existing files are skipped).

---

## Python: engine registry

```python
from poc import (EngineSpec, ParamSpec, register_engine, get_engine_spec,
                 all_engine_specs, create_engine, has_engine)
```

| Symbol | Signature | Returns |
|--------|-----------|---------|
| `register_engine` | `register_engine(spec, *, replace=False)` | the spec (raises on duplicate id unless `replace`) |
| `has_engine` | `has_engine(id) -> bool` | |
| `get_engine_spec` | `get_engine_spec(id) -> EngineSpec` | raises `KeyError` if unknown |
| `all_engine_specs` | `all_engine_specs() -> list[EngineSpec]` | runnable first, then alphabetical |
| `create_engine` | `create_engine(id, **params) -> Engine` | applies defaults; raises if `factory is None` |

**`EngineSpec`** fields: `id`, `name`, `description`, `runnable` (bool),
`requires` (tuple of tags), `params` (tuple of `ParamSpec`), `live_mode`
(`"produce"`|`"native"`), `factory` (`(**params) -> Engine`).
`spec.to_public()` returns the JSON-serializable view used by `/api/engines`.

**`ParamSpec`** fields: `name`, `label`, `type` (`"enum"`|`"bool"`|`"string"`),
`default`, `choices` (for enum), `help`.

Out-of-tree discovery: entry-point group **`strategist.engines`** (each entry
loads to an `EngineSpec`, or a callable returning one/a list).

---

## Python: domain packs

```python
from poc import (DomainPack, SalesDomainPack, register_domain, get_domain,
                 all_domains, set_active_domain, active_domain)
```

| Symbol | Signature |
|--------|-----------|
| `set_active_domain` | `set_active_domain(name_or_pack) -> DomainPack` |
| `active_domain` | `active_domain() -> DomainPack` (defaults via `POC_DOMAIN`, else `sales`) |
| `register_domain` | `register_domain(pack, *, replace=False)` |
| `get_domain` / `all_domains` | lookup / list |

**`DomainPack`** methods (override what differs):
`describe_tenant(tenant)`, `opp_type_note(opp_type)`,
`render_anchor_section(anchors)`, `detect_close(text)`, `detect_decline(text)`.

---

## Python: scenario sources

```python
from poc import (ScenarioSource, JsonScenarioSource,
                 get_scenario_source, set_scenario_source)
```

**`ScenarioSource`** protocol: `load_scenarios()`, `fetch_opp_meta(opp_id)`,
`fetch_messages(opp_id)`, `fetch_business_rules(company)`,
`fetch_anchors(opp_id, opp_meta=None)`.

`JsonScenarioSource(path=None)` serves the bundled `v1_scenarios.json` by default.

---

## HTTP: REST endpoints

Served by `server/main.py` (launch with `./bin/run-server.sh`, default
`http://localhost:8443/`).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | the single-page UI |
| GET | `/logs` | trace-logs UI |
| GET | **`/api/engines`** | registered engines (drives the UI selectors) |
| GET | `/api/scenarios` | list curated scenarios |
| GET | `/api/scenarios/{opp_id}` | scenario detail + transcript metadata |
| GET | `/api/random_match/criteria_options` | dropdown options for random match |
| POST | `/api/random_match` | find best clean-loss candidate by criteria |
| POST | **`/api/run/{opp_id}`** | start a replay session (see below) |
| WS | `/ws/{session_id}` | stream session events |
| GET | `/api/trace/list`, `/api/trace/{session_id}` | trace JSONs |
| GET | `/api/cohort_weights`, `/api/cache_status` | research artifacts / cache state |
| GET | `/api/precedents`, `/api/precedents/meta` | cohort precedent retrieval (prod) |
| GET | `/api/historical_persuasion/{opp_id}` | real-conversation persuasion overlay (prod) |
| GET | `/health` | DB health |

**`GET /api/engines`** response:

```json
{ "engines": [
  { "id": "baseline", "name": "Baseline (Original agent)", "description": "...",
    "runnable": true, "requires": [], "params": [] },
  { "id": "planner", "name": "Planner + Gates", "description": "...",
    "runnable": true, "requires": [],
    "params": [ { "name": "planner_envelope", "label": "Econ envelope",
                  "type": "enum", "default": "off",
                  "choices": ["off","auto","always"], "help": "..." } ] },
  { "id": "strategist", "name": "Strategist (Supervisor)", "description": "...",
    "runnable": false, "requires": ["mysql","lightrag"], "params": [] }
] }
```

---

## HTTP: the run request

**`POST /api/run/{opp_id}`** — per-panel engine selection. All fields optional;
defaults reproduce the classic LEFT=baseline / RIGHT=strategist pairing. Unknown
engine ids fall back to defaults; params not declared by the chosen engine are
dropped.

```json
{
  "engine":             "strategist",          // R-side engine id
  "engine_left":        "baseline",            // L-side engine id
  "engine_params":      { "...": "..." },       // R-side params (validated)
  "engine_params_left": { "...": "..." },       // L-side params (validated)
  "hard_customer":      false,                  // adversarial simulator overlay
  "seed_end_override":  0,                       // 0 = auto peak-engagement detector
  "planner_envelope":   "off"                   // back-compat: maps to R-side planner param
}
```

Response echoes the resolved `session_id`, `engine`, `engine_left`,
`engine_params`, `engine_params_left`. Connect to `/ws/{session_id}` to stream.

---

## WebSocket

`/ws/{session_id}` — after `POST /api/run`, open the socket; the server streams
events (`session_ready`, `left_msg`/`right_msg`, gate firings, plan updates,
outcomes). Client→server control messages:

```json
{ "action": "speed", "speed": "5x" }   // playback speed
{ "action": "stop" }                    // cooperative stop
{ "action": "ping" }                    // → {"event":"pong"}
```
