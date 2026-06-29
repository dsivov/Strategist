# Architecture

How the pluggable benchmark platform fits together, and why it's shaped this way.

- [The single contract](#the-single-contract)
- [Three pluggable seams](#three-pluggable-seams)
- [The per-turn run loop](#the-per-turn-run-loop)
- [Two run modes](#two-run-modes)
- [The engine registry](#the-engine-registry)
- [Live-replayer engine routing](#live-replayer-engine-routing)
- [Domain packs](#domain-packs)
- [Scenario sources](#scenario-sources)
- [Module-instance unification](#module-instance-unification)
- [Behavior preservation](#behavior-preservation)

---

## The single contract

Everything composes around one method:

```python
text, meta = await engine.produce(opp_meta, dialog_history, business_rules)
```

- `opp_meta` — customer profile + scenario context (a dict).
- `dialog_history` — `[{"role": "agent"|"customer", "text": str}, ...]`.
- `business_rules` — tenant compliance text (may be `""`).
- returns `(customer_facing_text, telemetry_dict)`. The benchmark records `meta`
  verbatim and never interprets it.

`Engine` is a `runtime_checkable` `Protocol` (`poc/engine.py`), so any object
with a matching `produce` satisfies it — no base class required.

---

## Three pluggable seams

| Seam | Module | Replaces (was inlined / hardcoded) | Default |
|------|--------|------------------------------------|---------|
| **Engine registry** | `poc/registry.py` | engine names hardcoded in 6 places | `baseline`, `planner`, `strategist` |
| **Domain pack** | `poc/domain.py` | sales/insurance strings across the core | `SalesDomainPack` |
| **Scenario source** | `poc/scenario_source.py` | hardcoded `v1_scenarios.json` path | `JsonScenarioSource` |

Each seam is read by the existing consumers, so extending the platform never
requires editing the runner, the API handler, or the UI.

---

## The per-turn run loop

Both run modes drive the same loop (the headless version lives in
`Benchmark._run_one`, `poc/benchmark.py`):

1. Build `opp_meta` + the seed dialogue from the scenario.
2. Instantiate the `CustomerSimulator` for this opportunity.
3. For each live turn, up to `max_turns`:
   - `agent_text, meta = await engine.produce(opp_meta, dialog, business_rules)`
   - bail on empty / refusal / graceful-close.
   - `cust_text = await simulator.reply(dialog, agent_text, k)`
   - `check = domain.detect_close / detect_decline` → `won` / `lost` / continue.
4. Emit a per-scenario result JSON (outcome, end-reason, per-turn trace, `meta`).

The customer simulator is **shared** across the A/B panels so the engine is the
only variable.

---

## Two run modes

```
                       ┌──────────────────────────┐
   in-tree register()  │      Engine Registry      │  entry-point plugins
   ───────────────────▶│      poc/registry.py      │◀───────────────────
                       └────────────┬─────────────┘
              ┌─────────────────────┴─────────────────────┐
              ▼                                            ▼
   Headless Benchmark                          Dual-panel server + UI
   run_engine("planner")                       GET /api/engines → selectors
              │                                            │
              └───────────────► Engine.produce() ◀─────────┘
                                      │
                          Customer Simulator (shared)
                                      │
                       Domain pack → won / lost + trace
```

- **Headless** (`poc/benchmark.py`): rigorous, resumable, paired A/B across all
  112 scenarios. Run engines by id via `Benchmark.run_engine(id, **params)`.
- **Live** (`server/`, `client/`): the dual-panel UI streams a session over
  WebSocket so you can watch two engines negotiate the same scenario turn by
  turn.

---

## The engine registry

`EngineSpec` is the registry record (`poc/registry.py`):

| Field | Purpose |
|-------|---------|
| `id` | stable slug used in API, results dir, UI value |
| `name` | display label |
| `description` | UI tooltip |
| `runnable` | benchmark-instantiable? (`strategist` is `False` — needs prod DB) |
| `requires` | capability tags → UI badges (`mysql`, `lightrag`) |
| `params` | `ParamSpec[]` (enum/bool/string + default) → UI controls, validated server-side |
| `live_mode` | `"produce"` (generic) or `"native"` (wired into the replayer) |
| `factory` | `(**params) -> Engine` for the benchmark + generic live path |

Lookup is by id; `all_specs()` returns runnable engines first. Out-of-tree
plugins are discovered lazily from the `strategist.engines` entry-point group on
first read (`_ensure_discovered`), and a failing plugin is logged and skipped so
one bad plugin can't break the rest.

---

## Live-replayer engine routing

The live replayer (`server/replayer.py`, `_live_turn`) historically assumed
**LEFT = control, RIGHT = supervised**, with ~9 branches gated on
`panel.side == "right"`. The refactor makes each panel carry its own engine id
(`opp_meta["_engine_left"]` / `_engine_right`) and routes by the engine's
`live_mode`:

- `live_mode == "produce"` → generic path: instantiate via the registry factory
  (cached on the panel) and call `engine.produce(...)`. Used by `planner` and
  every plugin, on **either** panel.
- `live_mode == "native"` → the wired-in flows keyed by id: `baseline`
  (vanilla actor) and `strategist` (the supervisor chain).

The supervised-only branches were re-keyed from `panel.side == "right"` to:

```python
panel_is_supervised = (panel_engine == "strategist")
```

Because the default pairing is LEFT=`baseline` / RIGHT=`strategist`, this
boolean evaluates exactly as the old side check — **the default comparison is
unchanged by construction** — while novel pairings (e.g. Planner on the left,
or Strategist on both) become possible.

> Removed dead code: the previous live "planner" path imported
> `from engines import planner_produce`, a bridge module that was never shipped;
> it silently fell back to the classic flow, so live Planner never actually ran.
> The generic `produce()` path both deletes that branch and makes live Planner work.

---

## Domain packs

`DomainPack` (`poc/domain.py`) supplies the domain-specific pieces the engines
and scorer need:

| Method | Consumed by |
|--------|-------------|
| `describe_tenant(tenant)` | `actor._domain_desc` |
| `opp_type_note(opp_type)` | `actor._opp_type_behavioral_note` |
| `render_anchor_section(anchors)` | `actor._build_anchor_section` |
| `detect_close(text)` / `detect_decline(text)` | `benchmark._check_customer_outcome` |

`SalesDomainPack` is the default and reproduces the original strings/behavior
byte-for-byte (pinned by `tests/test_characterization.py`). Selection is global
via `set_active_domain(...)` or the `POC_DOMAIN` env var.

The bundled benchmark **data** stays a sales/insurance corpus — you can't make
fixed historical conversations domain-neutral. What's decoupled is the *code's*
coupling to that domain.

---

## Scenario sources

`ScenarioSource` (`poc/scenario_source.py`) is the read surface for scenarios,
opportunity metadata, transcripts, business rules, and anchors.
`JsonScenarioSource` (default) serves the bundled `v1_scenarios.json`; MySQL is
served by `poc/db.py` (the live server's `server/db.py` shim switches between
them on `POC_USE_MYSQL`).

`load_scenarios()` resolution order:

1. explicit `source=` → `source.load_scenarios()`
2. explicit `path=` → read that JSON file (legacy behavior, unchanged)
3. otherwise → the active source (`get_scenario_source()`, bundled JSON default)

---

## Module-instance unification

The codebase mixes **flat** imports (`import registry`, used by the server with
the package dir on `sys.path`) and **package** imports (`poc.registry`). Left
alone these are two distinct module objects with separate registry / active
state — a runtime `register` on one wouldn't be visible to the other.

Fix: `poc/__init__.py` imports `registry`, `domain`, and `scenario_source` under
their **flat** names (after `.engine` puts the package dir on `sys.path`), and
the registry/domain factories use flat imports too. Every consumer therefore
shares one module instance.

---

## Behavior preservation

The refactor was kept safe without live LLM runs via three mechanisms:

1. **Golden safety net** — `tests/test_characterization.py` captured the
   domain-coupled behavior *offline* (actor-prompt SHA-256, anchor section,
   opp-type notes, outcome detectors, `paired_summary` math) before any
   extraction, so any drift fails loudly.
2. **Default-path identity** — supervised gates re-keyed from side to engine id;
   the default pairing maps 1:1 onto the old branches.
3. **Graceful degradation** — unknown engine ids fall back to defaults; a
   failing plugin is logged and skipped; the JSON data shim runs the server
   without a prod DB.

What is *not* validated: live multi-turn sessions (no API keys / cost). A novel
live pairing such as Strategist-on-left is enabled structurally but deserves one
keyed smoke run before being relied upon.
