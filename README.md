# POC — Sales-Conversation AI Benchmark Package

A self-contained benchmark harness for evaluating sales-conversation AI
engines, plus two reference engines (Baseline, Planner+Gates) and a v2
reference-aware customer simulator. Designed to be plugged into your own
engine via a single Python `Engine` protocol and run head-to-head against the
reference arms on a 112-scenario diversity-stratified benchmark.

> Built for the **Example team** so the example engine can be benchmarked
> against the Strategist/Planner work and the merged-system architecture can
> be validated empirically.

---

## Documentation

**▶ Read it online (rendered):** [Play the Match, Not the Exam (blog)](https://dsivov.github.io/Strategist/BLOG_PLAY_THE_MATCH.html)
· [Architecture Overview](https://dsivov.github.io/Strategist/STRATEGIST_OVERVIEW.html)
· [docs home](https://dsivov.github.io/Strategist/) — served via GitHub Pages.

The table below links to the source files in this repo (GitHub shows raw HTML there; use the rendered links above to view them).

| Doc | What it covers |
|-----|----------------|
| **[docs/STRATEGIST_OVERVIEW.html](docs/STRATEGIST_OVERVIEW.html)** | Illustrated, self-contained field guide (open in any browser) — the architecture at a glance. |
| **[docs/BLOG_PLAY_THE_MATCH.html](docs/BLOG_PLAY_THE_MATCH.html)** | "Play the Match, Not the Exam" — human-friendly blog on why/how we A/B-test goal-oriented persuasion agents, framed through Cialdini's *Influence*. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The pluggable architecture in depth: the three seams, request flow, and the live-replayer engine routing. |
| [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md) | How to author & register an **engine** (in-tree + out-of-tree entry point), a **domain pack**, and a **scenario source**. |
| [docs/API.md](docs/API.md) | Reference for the public Python API (registry / domain / source) and the server's REST + WebSocket endpoints. |
| [INTEGRATION.md](INTEGRATION.md) | Step-by-step guide to writing your engine adapter. |

> This package began as a two-engine POC and was refactored into a **pluggable
> benchmark platform**: register a Strategy/Supervisor algorithm once and it
> appears in the headless runner, the REST API, and the live dual-panel UI.
> See [§ Pluggable engines, domains & data sources](#pluggable-engines-domains--data-sources).

---

## Quickstart

```bash
# 1. Install
cd POC
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and GEMINI_API_KEY_1

# 3. Smoke test (no LLM calls)
python tests/test_smoke.py

# 4a. Headless 10-scenario benchmark (Baseline vs Planner+Gates)
python examples/run_benchmark.py

# 4b. Dual-panel web UI (interactive — see a session unfold)
./bin/run-server.sh
# Then open http://localhost:8443/ in a browser
```

The smoke test should print `ALL SMOKE TESTS PASSED` in <2 s.
The 10-scenario headless benchmark takes ~30 min and writes results to `./results/`.
The web UI lets you pick a scenario, choose engines for LEFT and RIGHT panels,
and watch the two agents negotiate against the same customer simulator side
by side.

---

## What's in the package

```
POC/
├── README.md                      ← this file
├── INTEGRATION.md                 ← step-by-step example adapter guide
├── requirements.txt
├── .env.example
│
├── bin/
│   └── run-server.sh              ← launch the dual-panel UI server
│
├── server/                        ← FastAPI + WebSocket replayer (the UI)
│   ├── main.py                    ← REST + WS endpoints
│   ├── replayer.py                ← dual-panel session orchestrator
│   ├── db.py                      ← JSON-backed shim (no MySQL needed)
│   ├── random_match.py            ← scenario picker by criteria
│   ├── session_logger.py
│   ├── persuasion_scorer.py
│   ├── supervisor_full.py
│   ├── cluster_plan.py
│   ├── win_plan.py
│   └── win_proximity.py
│
├── client/                        ← static HTML/JS/CSS (mounted at /static)
│   ├── index.html                 ← the dual-panel UI
│   ├── app.js
│   ├── logs.html / logs.js
│   ├── style.css
│   └── chart.umd.min.js
│
├── poc/                           ← the Python library (public API)
│   ├── __init__.py
│   ├── engine.py                  ← Engine protocol + 3 reference engines
│   ├── registry.py                ← pluggable engine registry (+ entry-point discovery)
│   ├── domain.py                  ← pluggable domain packs (SalesDomainPack default)
│   ├── scenario_source.py         ← pluggable scenario/data sources (JSON default)
│   ├── benchmark.py               ← headless benchmark runner (run_engine by id)
│   ├── customer_simulator.py      ← v2 reference-aware simulator
│   ├── actor.py              ← shared customer-facing LLM renderer
│   ├── post_render_gates.py       ← anti-staircase + premature-close
│   ├── voice_profile.py
│   ├── intent_classifier.py
│   ├── trace_logger.py
│   ├── db.py                      ← MySQL-talking version (Strategist arm)
│   │
│   ├── planner/                   ← PCA-derived state-graph planner
│   │   ├── engine.py
│   │   ├── cot_sop.py
│   │   ├── sop.py · sop_builder.py
│   │   ├── playbook_reader.py
│   │   └── data/sop_graph/        ← bundled won-deal SOP graphs
│   │
│   └── strategist/                ← supervisor chain + gates
│       ├── chain_runner.py
│       ├── chain_stages_supervisor.py
│       ├── concrete_moves_loader.py
│       ├── staircase_gate.py
│       ├── capitulation_gate.py
│       ├── invariant_gates.py
│       └── runners/               ← Mode-1a v1 + attribution
│
├── data/
│   ├── benchmark/
│   │   └── v1_scenarios.json      ← 112 scenarios (856 KB)
│   ├── scenarios.json             ← UI-friendly index (derived)
│   ├── precedents-sample.db       ← 1,200-row SQLite sample (Strategist arm)
│   └── script_library/            ← 11 mined playbook YAMLs
│       ├── Insurance/                 (6 playbooks)
│       └── Ecommerce/                (5 playbooks)
│
├── examples/
│   ├── example_engine_template.py      ← starter shim for the example adapter
│   ├── run_benchmark.py           ← full 2-arm paired driver
│   ├── run_pluggable_benchmark.py ← registry-driven A/B (run engines by id)
│   ├── custom_domain_pack.py      ← add a new domain without core edits
│   └── external_engine_plugin/    ← installable out-of-tree engine (entry point)
│
└── tests/
    ├── test_smoke.py              ← imports + bundled-data sanity (no LLM calls)
    ├── test_characterization.py   ← golden tests pinning domain behavior
    ├── test_registry.py           ← engine registry
    ├── test_domain.py             ← domain-pack pluggability
    └── test_scenario_source.py    ← pluggable data sources
```

## The dual-panel benchmark server

The same FastAPI server we use internally for inspection runs is bundled at
`server/main.py`. It hosts a web UI that:

- Lists the 112 scenarios with their diversity-bucket labels
- Lets you pick LEFT and RIGHT engines (Baseline / Planner+Gates / Strategist)
- Streams a live WebSocket session, showing each agent move and each customer
  reply side by side as they happen
- Records trace JSONs you can replay or diff

Launch with `./bin/run-server.sh`, open `http://localhost:8443/`.

### Why this server runs without the production system MySQL

The upstream server reads opportunity rows, message histories, turn-state
labels, and persuasion scores out of our production MySQL. For the handoff,
`server/db.py` is a **drop-in JSON-backed shim** that serves all the same
calls from the bundled `data/benchmark/v1_scenarios.json` file. Embedded
historical transcripts (per scenario) replace `fetch_messages`. The
attributes block replaces `fetch_opp_meta`. Turn-state and persuasion-score
sources are stubbed (return empty) — the engines work fine without them;
they're prod-only signal channels.

If you do have the production system credentials, set `POC_USE_MYSQL=1` in your
environment and `server/db.py` will proxy through to the real
MySQL-backed `poc/db.py`. Same calling code; live data.

## Supporting databases — what's bundled, what isn't

| Database                       | Bundled?  | Size  | Used by              | Notes |
|--------------------------------|-----------|-------|----------------------|-------|
| `v1_scenarios.json`            | ✅ yes    | 856 KB | Simulator + db shim | The 112-scenario benchmark, byte-identical to upstream. |
| Mined playbook YAMLs           | ✅ yes    | 56 KB  | Planner + Strategist | All 11 playbooks (6 Insurance + 5 Ecommerce). |
| SOP graphs (JSON)              | ✅ yes    | 14 KB  | Planner              | Both tenants, ~142 edges total. |
| `precedents-sample.db`         | ✅ yes    | 1.5 MB | Strategist           | **Subsample.** 1,200 rows balanced across tenant × outcome. The full database has 872k rows (~1 GB) — too large to bundle. The sample lets the Strategist's retrieval code exercise its query path; for production-grade precedent breadth you'd need to mount the full DB. |
| the production system MySQL          | ❌ no     | —      | Strategist           | Source of cohort + business-rules + turn-state data. Requires VPN + credentials. JSON-backed shim is the standalone substitute. |
| Agent Knowledge Graph (LightRAG)| ❌ no     | —      | Strategist           | Provides decision-trace precedents and KG entity lookups. Set `LIGHTRAG_API_URL` + `LIGHTRAG_API_KEY` to enable. |

The runnable arms (Baseline + Planner+Gates) need only the bundled data.
The Strategist arm gracefully degrades without LightRAG (logs warnings)
but is fully functional only against the full precedent corpus + live KG.

---

## The Engine protocol — what the example engine implements

The benchmark calls one async method per agent turn. That's it.

```python
from poc import Engine          # the Protocol
from poc import BaselineEngine  # reference impl
from poc import PlannerEngine   # reference impl

class ExampleEngine:
    """Your shim around the example stack."""

    async def produce(
        self,
        opp_meta:        dict,       # customer profile + scenario context
        dialog_history:  list[dict], # [{"role": "agent"|"customer", "text": str}]
        business_rules:  str = "",
    ) -> tuple[str, dict]:
        # ── call your stack however you do today ─────────
        result = await self.engine.predict(opp_meta, dialog_history)
        text   = self.engine.render(result)
        return text, {
            "strategy":        result.strategy,
            "tone":            result.tone,
            "hint_confidence": result.confidence,
            "escalated":       False,
        }
```

Drive the benchmark:

```python
from poc import Benchmark, load_scenarios

scenarios = load_scenarios()                                 # 112 scenarios
bench     = Benchmark(scenarios, results_dir="./results")
results   = await bench.run_arm("example", ExampleEngine())            # per-scenario JSONs
```

See `examples/example_engine_template.py` for a starter shim (HTTP and in-process
patterns shown). See `INTEGRATION.md` for the full integration walkthrough.

---

## The three reference arms

### 1. `BaselineEngine` (self-contained, ~5 s/turn)

The single-call production agent. Calls Gemini 2.5 Pro with the customer
profile + dialogue + business rules and asks for a reply. No directive,
no planning, no retrieval. This is the "what you'd ship without thinking
about it" baseline.

### 2. `PlannerEngine` (self-contained, ~25–30 s/turn)

The PCA-derived state-graph planner:

1. Estimate the customer's current state (engaged, price-objection, near-close, …).
2. Look up allowed agent actions from the SOP graph adjacency for that state.
3. Chain-of-thought reasoning (Anthropic Sonnet) to pick the best action.
4. Render the directive into customer-facing text (Gemini).
5. Post-render gate chain — anti-staircase + premature-close.

The bundled SOP graphs were built from won-deal transition tallies on the
insurance tenant (79 edges) and the e-commerce tenant (63 edges). The mined
playbook library is wired into the planner's prompt via `playbook_reader`.

### 3. `StrategistEngine` (source included, **not runnable in this package**)

The mining/cohort-retrieval/supervisor stack is included verbatim under
`poc/strategist/` for source review. It requires the production system database
access and the Agent Knowledge Graph endpoint to run. Instantiating it raises
with a pointer to this README. See **"Running the Strategist arm"** below.

---

## Pluggable engines, domains & data sources

The benchmark is a **platform**: bring your own Strategy/Supervisor algorithm,
register it, and A/B it against the reference arms — headless or in the live
dual-panel UI. Three seams make this work, all surfaced from `poc`:

### 1. The engine registry (`poc.registry`)

Every engine is declared once as an `EngineSpec` (id, display name, description,
capability requirements, tunable params, and a `factory`). The registry is the
single source of truth read by **both** the headless `Benchmark` and the live
server's `GET /api/engines` (which the UI uses to render its selectors — engines
are no longer hardcoded in the client).

```python
from poc import all_engine_specs, create_engine, Benchmark, load_scenarios

for spec in all_engine_specs():        # baseline, planner, strategist, + plugins
    print(spec.id, spec.name, spec.runnable, [p.name for p in spec.params])

bench = Benchmark(load_scenarios())
results = await bench.run_engine("planner", planner_envelope="auto")   # run by id
```

**In-tree** engines call `poc.register_engine(EngineSpec(...))`. **Out-of-tree**
engines (no core edits) ship a package with a `strategist.engines` entry point;
they're discovered automatically and appear in `all_engine_specs()`, the API,
and the UI. A complete worked example is in
[`examples/external_engine_plugin/`](examples/external_engine_plugin) — install
it with `pip install -e examples/external_engine_plugin` and the `echo` engine
shows up everywhere.

Your engine implements the same one-method `Engine` protocol the example adapter uses
(`async produce(opp_meta, dialog_history, business_rules) -> (text, meta)`), so
the integration contract is unchanged — see [INTEGRATION.md](INTEGRATION.md).

### 2. Any-vs-any A/B in the live UI

The dual-panel server now selects each panel's engine independently from the
registry (LEFT and RIGHT are both dropdowns populated from `/api/engines`, with
per-engine parameter controls). The default pairing — LEFT = baseline, RIGHT =
strategist — reproduces the classic comparison exactly; any other registered
engine can be slotted into either panel.

### 3. Domain packs (`poc.domain`) and scenario sources (`poc.scenario_source`)

The bundled data **stays** a sales/insurance corpus. What's now pluggable is the
*code's* coupling to that domain: the agent's tenant/opp-type framing, the
economic-anchor rendering, and the won/lost outcome signals come from the
**active domain pack** (`SalesDomainPack` is the default and is byte-for-byte
identical to the original behavior — pinned by `tests/test_characterization.py`).
Likewise, scenario/opportunity/transcript loading goes through a `ScenarioSource`
(default `JsonScenarioSource` over the bundled JSON; MySQL via `poc/db.py`).

```python
from poc import set_active_domain, set_scenario_source, JsonScenarioSource
set_active_domain("sales")                              # or a custom DomainPack
set_scenario_source(JsonScenarioSource("my.json"))     # or your own source
```

See [`examples/custom_domain_pack.py`](examples/custom_domain_pack.py) for adding
a new domain, and [`examples/run_pluggable_benchmark.py`](examples/run_pluggable_benchmark.py)
for a registry-driven A/B run.

---

## The benchmark dataset

112 scenarios across two industries (auto insurance renewal, e-commerce cart
abandonment). Each scenario is anchored in a real historical conversation,
characterized along 23 customer dimensions, and stratified into 15 diversity
cells. About 20% are real-wins (catches regression on easy cases).

### Scenario schema

```json
{
  "scenario_id":          "L_Pr_An_Sk_04",
  "opp_id":               "00733e8d-...",
  "tenant":               "Insurance",
  "diversity_bucket":     "Price/savings × Analytical × Skeptical",
  "real_outcome":         "won",
  "is_sentinel":          false,
  "rng_seed":             17004,
  "seed_dialog_cut_idx":  9,
  "n_messages":           18,
  "attributes": {
    "primary_motivator":  "Price/savings",
    "decision_logic":     "Analytical",
    "trust_level":        "Skeptical",
    "communication_style":"Terse",
    "objection_pattern":  "Comparing competitor prices",
    "emotional_volatility":"Low",
    "purchase_urgency":   "High",
    "primary_resistance": "Price",
    "opp_type":           "Insurance Renewal",
    ...                                      // 23 dims total
  },
  "anchors": {                                // per-opp pricing reference
    "last_year_price_usd":         4238,
    "current_quoted_price_usd":    4581,
    "market_avg_for_segment_usd":  4400,
    "max_discount_pct_internal":   15
  },
  "anchor_real": true,                        // false = cohort fallback
  "seed_messages": [
    {
      "message_id": "...",
      "direction":  "outbound",
      "timestamp":  "2024-12-01 10:16:39",
      "text":       "Hey there Lior, this is Sapir from Insurance...",
      "is_reminder":1,
      "is_followup":0
    },
    ...                                       // full historical transcript
  ]
}
```

**Why seed transcripts are embedded.** The customer simulator's v2 design
(see below) needs the historical customer reply *at the current turn* as a
posture/tone reference. Embedding the full transcript per scenario means the
benchmark is self-contained — no DB fetch at runtime.

**Diversity stratification.** 15 cells (insurance: 6, e-commerce: 9), each
sampled via farthest-point sampling on a 23-dim profile vector. Maximizes
spread of customer types rather than uniform sampling.

---

## The customer simulator (v2 reference-aware)

The simulator on the other side of every conversation. Calibrated two ways:

| Test                                          | Result                          |
|-----------------------------------------------|---------------------------------|
| Realism (held-out continuation, n=23)         | 78% outcome agreement; **−22 pp pessimism bias** (under-states wins) |
| Adaptivity (perturbed agent, n=28, 2 judges)  | 4.71 / 5 mean; 3.6–7.1% echo rate |

Two design choices matter:

1. **Same-turn-only reference.** At turn `k`, the simulator sees the
   historical customer reply *at turn k* as a posture/tone reference —
   never the future. This prevents outcome leakage (otherwise every
   real-won scenario would tilt toward "won" regardless of the agent under
   test).
2. **Adapt-to-live-agent prompting.** When the live agent's move diverges
   from the historical agent, the simulator adapts on-topic to the live
   move while keeping the reference's posture.

`POC_SIM_V2_REFERENCE=on` is the default. Set `off` to fall back to v1
(rephrase-vs-generate hybrid) — kept for sanity-check comparisons.

---

## Running the Strategist arm

The Strategist's `chain_runner.py` is built around our websocket replayer
(see source comments) and depends on:

- **the production system MySQL** — for `fetch_business_rules`, cohort precedent
  retrieval, classifier metadata
- **Agent Knowledge Graph endpoint** (`LIGHTRAG_API_URL`, `LIGHTRAG_API_KEY`)
  — for decision-trace precedents, mined entity lookups
- **Concrete-moves catalogs** under `data/concrete_moves/` (not bundled —
  these are tenant-specific and live in our research notebook)

To run a true three-arm comparison including the Strategist:

1. Mount the the production system DB credentials in `db.py` (or set `MYSQL_*` env vars
   per `db.py:open_conn()`).
2. Set `LIGHTRAG_API_URL` + `LIGHTRAG_API_KEY` env vars.
3. Copy `data/concrete_moves/` from the upstream POC into this package.
4. Replace the `StrategistEngine.__init__` raise with a real driver — the
   chain runner's `run_chain(stages, ctx)` API needs adapter code to map
   the benchmark's per-turn shape into a `ChainContext`. **This is not
   trivial; budget ~1 day of integration work.**

For most example benchmark use cases (Example vs Baseline, Example vs Planner+Gates), the
self-contained two arms are sufficient. The Strategist source is included
for architectural review.

---

## Results format

Per scenario per arm, a JSON written to `{results_dir}/{arm_name}/{scenario_id}.json`:

```json
{
  "arm":          "example",
  "scenario_id":  "L_Pr_An_Sk_04",
  "opp_id":       "00733e8d-...",
  "tenant":       "Insurance",
  "real_outcome": "won",
  "outcome":      "won",                  // or "lost"
  "end_reason":   "customer_won",         // or "agent_graceful_close", "max_turns_no_close", ...
  "n_live_turns": 3,
  "elapsed_s":    78.4,
  "turns": [
    {
      "live_turn":     0,
      "agent_text":    "...",
      "customer_text": "...",
      "sim_mode":      "generate_v2_ref",  // or "rephrase", "generate_v2_no_ref"
      "outcome_check": null,                // or "won"|"lost" at this turn
      "agent_meta": {                       // your meta dict, recorded verbatim
        "strategy":        "objection_reframe",
        "tone":            "warm",
        "hint_confidence": 0.82,
        ...
      }
    },
    ...
  ],
  "env": {
    "v2_simulator":   "on",
    "planner_gates":  "on",
    "max_live_turns": 12
  }
}
```

For a paired summary across arms:

```python
from poc.benchmark import paired_summary

summary = paired_summary({
    "baseline": baseline_results,
    "example":       example_results,
})
# {
#   "n_scenarios": 112,
#   "arms": {
#     "baseline": {"n": 112, "won": 56, "win_rate": 0.500},
#     "example":       {"n": 112, "won": 71, "win_rate": 0.634},
#   },
#   "pairwise": {"baseline_better": 4, "example_better": 19, "ties": 89}
# }
```

---

## Configuration

| Env var                       | Default | Notes |
|-------------------------------|---------|-------|
| `ANTHROPIC_API_KEY`           | —       | Required. Simulator + planner CoT use Anthropic Sonnet 4.5. |
| `GEMINI_API_KEY_1`            | —       | Required. Customer-facing actor uses Gemini 2.5 Pro. |
| `VOYAGE_API_KEY`              | —       | Optional. Simulator similarity check; word-overlap fallback if unset. |
| `POC_SIM_V2_REFERENCE`        | `on`    | Default-on v2 reference-aware simulator. `off` = v1 hybrid. |
| `POC_PLANNER_GATES`           | `on`    | Anti-staircase + premature-close post-render gates. |
| `POC_DATA_ROOT`               | `<pkg>/data` | Where scenarios + mined library live. |
| `POC_SCRIPT_LIBRARY_DIR`      | `<pkg>/data/script_library` | Override mined-playbook location. |
| `LIGHTRAG_API_URL` / `_KEY`   | —       | Strategist only; Agent Knowledge Graph endpoint. |

---

## Resumable runs

Per-scenario JSON is written under `{results_dir}/{arm_name}/`. Re-running
the same arm skips scenarios already on disk. To re-run a scenario,
delete its JSON file. To run a subset, pass `scenarios=` to `run_arm()`.

---

## Latency & cost rough budget

For the full 112-scenario set on the bundled arms:

| Arm              | Time     | API cost     | Notes |
|------------------|----------|--------------|-------|
| Baseline         | ~2.5 h   | ~$15 Gemini  | 1 LLM call/turn |
| Planner + Gates  | ~5 h     | ~$60 (Sonnet + Gemini) | Planner CoT + actor render + optional regen on gate fire |
| Strategist (if running) | ~10 h | ~$140 | 4-stage chain + retrieval + retries |

The simulator adds another ~$20 across the full run.

Both reference arms are sequential by default. Parallelism is possible but
respects per-vendor rate limits — see `Benchmark.run_arm` for the loop and
extend as needed.

---

## What we deliberately didn't bundle

- **the production system database credentials.** The `db.py` module is the same
  one we use, but `open_conn()` needs MySQL creds. Required only for the
  Strategist arm.
- **Concrete-moves catalogs.** The Strategist's per-tenant move catalogs
  live in our internal `data/concrete_moves/`. Tenant-specific; not bundled.
- **Operator-only test infrastructure.** The websocket UI, replayer, lift-
  batch runner, trace inspectors. The benchmark is the public surface; the
  rest stays in the research notebook.

---

## Companion documents

- `INTEGRATION.md` — step-by-step guide to writing your example adapter
- `../research-blog.html` — the full research diary
- `../papers-we-used.html` — plain-language reading list of the underlying papers
- `../research-notes/2026-05-05-comparison-with-example-team.md` — the cross-team
  architecture comparison this package was built to support empirically

---

## Questions, issues, contributions

The package is a snapshot — not a maintained product. For architectural
questions about the engines or the benchmark methodology, the research notes
under `research/poc-supervisor-strategist/research-notes/` are the canonical
reference. For integration help, see `INTEGRATION.md` first; if something is
genuinely missing or broken, message the research team directly.
