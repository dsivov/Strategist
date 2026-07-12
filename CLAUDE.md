# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Persuasion Agent Benchmark** — a pluggable platform for A/B-comparing goal-oriented conversation agents. An "engine" plays the agent across multi-turn matches against a shared LLM-driven customer simulator, over **benchmark packs** (bundled: `insurance-renewal` 51 + `ecommerce-cart` 61 scenarios; `benchmarks/_template/` turns any goal-oriented task into a new pack). Ships reference arms (`baseline`, `planner`, `strategist` source-only) plus a local llama.cpp plugin (`examples/llamacpp_engine_plugin/`: `llamacpp`, `llamacpp-strategist`, `llamacpp-planner` — no cloud key for the agent side). Register your own engine and it shows up in the headless runner, the REST API, and the live dual-panel UI with no core edits.

This is a research snapshot, not a maintained product. README.md is the authoritative deep reference; `docs/ARCHITECTURE.md`, `docs/PLUGIN_GUIDE.md`, `docs/API.md`, and `INTEGRATION.md` cover specific seams in depth.

## Setup & commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set ANTHROPIC_API_KEY + GEMINI_API_KEY_1

# Tests (no LLM calls — safe with placeholder keys)
python -m pytest tests/ -v
python -m pytest tests/test_registry.py -v          # single file
python -m pytest tests/test_domain.py::test_name    # single test
python tests/test_smoke.py                           # also runnable directly

# Headless benchmark (makes real LLM calls — slow + costs money)
python examples/run_benchmark.py            # 2-arm paired driver
python examples/run_pluggable_benchmark.py  # registry-driven A/B by engine id

# Dual-panel web UI (real LLM calls)
./bin/run-server.sh         # serves http://localhost:8443/ (uvicorn server.main:app)
```

There is no build step and no linter configured. The smoke test prints `ALL SMOKE TESTS PASSED` in <2 s and makes no network calls.

## The one contract everything composes around

```python
text, meta = await engine.produce(opp_meta, dialog_history, business_rules)
```

`Engine` (`poc/engine.py`) is a `runtime_checkable` Protocol — any object with a matching async `produce` satisfies it, no base class. The benchmark records the returned `meta` dict verbatim and never interprets it. Both the headless runner (`Benchmark._run_one` in `poc/benchmark.py`) and the live server drive the *same* per-turn loop, and the customer simulator is shared across A/B panels so the engine is the only variable.

## Four pluggable seams (extend without editing the core)

| Seam | Module | Default | Register via |
|------|--------|---------|--------------|
| Engine registry | `poc/registry.py` | `baseline`, `planner`, `strategist` | `register_engine(EngineSpec(...))` in-tree, or a `strategist.engines` entry point out-of-tree |
| Domain pack | `poc/domain.py` | `SalesDomainPack` | `register_domain(...)` / `set_active_domain(...)` |
| Scenario source | `poc/scenario_source.py` | `JsonScenarioSource` over `data/benchmark/v1_scenarios.json` | `set_scenario_source(...)` |
| Benchmark packs | `poc/benchmark_packs.py` | `benchmarks/{insurance-renewal,ecommerce-cart}` | drop a `benchmarks/<id>/pack.json` (copy `_template/`) |

The registry is the single source of truth read by both `Benchmark` and the server's `GET /api/engines` (the UI selectors are populated from it — engines are not hardcoded in the client). See `examples/` for worked examples of each seam (`external_engine_plugin/` is an installable out-of-tree engine; `custom_domain_pack.py` adds a domain).

## Critical gotcha: flat imports + single module instance

`poc/__init__.py` imports `.engine` **first** because doing so puts the package directory on `sys.path`. After that, `registry`, `domain`, and `scenario_source` are imported by their **flat names** (`from registry import ...`, not `from .registry`). This is deliberate: the public API and the internal flat-import consumers (`actor`, `benchmark`, the server) must share **one** module instance, or `poc.registry` and `registry` become distinct objects with separate registry / active-domain state. Preserve this pattern — don't "fix" the imports to be relative. Several subpackages also locate bundled data through env vars (`POC_DATA_ROOT`, `POC_SCRIPT_LIBRARY_DIR`, `POC_CONCRETE_MOVES_DIR`) that `poc/__init__.py` `setdefault`s, keeping the upstream subpackage code unmodified.

## The reference arms

- **`baseline`** — single Gemini 2.5 Pro call per turn. No planning/retrieval.
- **`planner`** (`poc/planner/`) — PCA-derived state-graph planner: estimate customer state → look up allowed actions from the bundled SOP graph (`poc/planner/data/sop_graph/`) → Anthropic Sonnet chain-of-thought picks the action → Gemini renders it → post-render gate chain (anti-staircase + premature-close, `poc/post_render_gates.py`, gated by `POC_PLANNER_GATES`).
- **`strategist`** (`poc/strategist/`) — **source included for review, not runnable here.** Its `chain_runner.py` needs production MySQL (`poc/db.py`), the LightRAG Agent Knowledge Graph (`LIGHTRAG_API_*`), and un-bundled concrete-moves catalogs. `StrategistEngine.__init__` raises with a pointer to the README. Only `baseline` and `planner` run on bundled data alone.

## Behavior is pinned — change it deliberately

`SalesDomainPack` is byte-for-byte identical to the original inlined behavior, **pinned by `tests/test_characterization.py`** (golden tests). If you refactor anything touching domain framing, anchor rendering, or won/lost outcome detection, those golden tests are the contract — run them.

## Other things worth knowing

- **Server runs without production MySQL.** `server/db.py` is a JSON-backed drop-in shim serving everything from `data/benchmark/v1_scenarios.json`. Set `POC_USE_MYSQL=1` to proxy to the real `poc/db.py` (note: `poc/db.py` is the MySQL version, `server/db.py` is the shim — they are different files).
- **Customer simulator** (`poc/customer_simulator.py`, v2) sees only the historical customer reply *at the current turn* as a posture reference — never the future (prevents outcome leakage). `POC_SIM_V2_REFERENCE=on` is default; `off` falls back to v1.
- **Results** are written per-scenario to `{results_dir}/{arm}/{scenario_id}.json`. Runs are **resumable**: re-running an arm skips scenarios already on disk; delete a JSON to re-run it. `poc.benchmark.paired_summary(...)` aggregates arms into win rates + pairwise comparison.
- **Required keys:** `ANTHROPIC_API_KEY` (simulator + planner CoT), `GEMINI_API_KEY_1` (customer-facing actor). `VOYAGE_API_KEY` is optional (word-overlap fallback). Full env table in README.md § Configuration.
- **`client/`** is static HTML/JS/CSS mounted by `server/main.py`; **`docs/`** is the GitHub Pages site.
