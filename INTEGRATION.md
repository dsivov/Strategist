# Integration guide — plugging your engine into the benchmark

A step-by-step walkthrough from a green smoke test to a full 112-scenario
paired benchmark of **your** conversation agent against the reference arms.
Estimated time to first real result: 1–2 hours of focused work.

---

## Step 0 — Verify the package is alive

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (and GEMINI_API_KEY_1 if you have one)

python tests/test_smoke.py
```

You should see `ALL SMOKE TESTS PASSED`. The smoke test makes zero LLM
calls; it just verifies imports, bundled data, and the Engine protocol shape.

If anything fails here, the problem is almost certainly a missing API key
in `.env` (some modules read env at import time) or a Python version below
3.10 (we use match/case and PEP 604 unions throughout).

---

## Step 1 — Write your engine adapter

Copy `examples/example_engine_template.py` to `my_adapter.py` (or wherever you
keep your code) and fill in `produce()`:

```python
class MyEngine:
    def __init__(self, **config):
        # Stash whatever connection info / clients / models you need.
        self.config = config

    async def produce(self, opp_meta, dialog_history, business_rules=""):
        # ── call your stack however you do today ─────────
        result = await self.engine.predict_and_render(
            opp_meta, dialog_history, business_rules,
        )
        return result.text, {
            "strategy":   result.strategy,
            "tone":       result.tone,
            "confidence": result.confidence,
        }
```

**The only contract is the method signature.** What you put in the `meta`
dict is recorded verbatim; the benchmark does not interpret it. Put anything
that helps your post-run analysis: strategy/tone, confidence numbers, which
sub-model fired, which gates triggered, latencies, model versions.

### Input you receive each turn

**`opp_meta` (dict)** — customer profile + scenario context. Keys:

| Key                       | Type    | Notes |
|---------------------------|---------|-------|
| `id`                      | str     | Opportunity ID (UUID). |
| `company`                 | str     | Tenant — `"Insurance"` or `"Ecommerce"` in the bundled packs. |
| `primary_motivator`       | str     | Cialdini-style classification. `"Price/savings"`, `"Necessity"`, `"Security/fear"`, etc. |
| `decision_logic`          | str     | `"Analytical"`, `"Emotional"`, `"Pragmatic"`. |
| `trust_level`             | str     | `"Skeptical"`, `"Trusting"`, `"Neutral"`. |
| `communication_style`     | str     | `"Terse"`, `"Detailed"`, `"Conversational"`. |
| `objection_pattern`       | str     | Most common objection for this customer. |
| `emotional_volatility`    | str     | `"Low"`, `"Med"`, `"High"`. |
| `regulatory_focus`        | str     | `"Prevention"` (loss-averse) / `"Promotion"` (gain-seeking). |
| `budget_sensitivity`      | str     | `"Low"`, `"Med"`, `"High"`. |
| `purchase_urgency`        | str/None| `"Low"`, `"High"`, or `None` for sentinel scenarios. |
| `primary_resistance`      | str     | Top blocker — `"Price"`, `"Authority"`, `"Time"`, etc. |
| `opp_type`                | str     | Use case — `"Insurance Renewal"`, `"Abandoned Cart"`. |
| `anchors`                 | dict    | Per-scenario numbers: `last_year_price_usd`, `current_quoted_price_usd`, `market_avg_for_segment_usd`, `max_discount_pct_internal`. |
| `voice_profile`           | dict?   | Optional voice cues mined from the historical transcript. |

(The persona attributes come straight from the scenario's `attributes` block —
23 dimensions total; the table lists the ones engines use most.)

**`dialog_history` (list[dict])** — `[{"role": "agent"|"customer", "text": str}]`,
ordered chronologically. The most recent customer message is the one your
engine should respond to. (When there is no customer message to respond
to — first turn, or two agent turns in a row — you can still emit; the
benchmark treats whatever you produce as the agent's move.)

**`business_rules` (str)** — tenant-specific compliance/voice rules. Empty in
the standalone package (the bundled data source returns `""`). Treat as
optional context.

### Output you return each turn

`tuple[str, dict]` — `(customer_facing_text, telemetry_meta)`.

- `customer_facing_text`: what gets sent to the simulator.
- `telemetry_meta`: whatever you want recorded. Suggested keys:
  - `strategy`, `tone` — your taxonomy
  - `confidence` — your stack's own confidence signal
  - `model_version` — for cross-run diff
  - `gates_fired`, `gates_regens` — if you ran post-render gates
  - `latency_ms` — per-stage timings

---

## Step 2 — Run a 3-scenario smoke against your adapter

```python
import asyncio
from poc import Benchmark, load_scenarios
from my_adapter import MyEngine

async def smoke():
    scenarios = load_scenarios()[:3]            # 3 scenarios, ~5 min
    bench = Benchmark(scenarios, results_dir="./my_smoke")
    results = await bench.run_arm(
        "mine",
        MyEngine(api_url="http://localhost:8080"),
        on_scenario_done=lambda r: print(
            f"  {r['scenario_id']:18} -> {r['outcome']:4} "
            f"({r['end_reason']}, {r['n_live_turns']}t, {r['elapsed_s']}s)"
        ),
    )
    print(f"\n{sum(1 for r in results if r['outcome']=='won')} wins / 3")

asyncio.run(smoke())
```

Expect: 3 scenarios complete without errors, output JSONs land in
`./my_smoke/mine/`. If any turn throws, the error is captured in
`agent_meta.error` for that turn rather than crashing the run — check the
output JSONs for the trace.

To run one **benchmark pack** instead of the full set:

```python
from poc import load_pack_scenarios
scenarios = load_pack_scenarios("insurance-renewal")   # or "ecommerce-cart"
```

---

## Step 3 — Run the full 2-arm paired benchmark

```python
from poc import Benchmark, BaselineEngine, load_scenarios
from poc.benchmark import paired_summary
from my_adapter import MyEngine

async def run_full():
    scenarios = load_scenarios()                # all 112
    bench = Benchmark(scenarios, results_dir="./results")

    baseline = await bench.run_arm("baseline", BaselineEngine())
    mine     = await bench.run_arm("mine",     MyEngine(...))

    summary = paired_summary({"baseline": baseline, "mine": mine})
    print(summary)

asyncio.run(run_full())
```

The runs are resumable — re-running skips already-complete `(arm, scenario)`
JSONs under `{results_dir}/{arm}/`. Crash recovery is just "re-run the
script."

Estimated wall time: ~2.5 h Baseline + however long your engine takes per
turn × 112 scenarios × ~6 turns avg. A fast local engine is negligible next
to the simulator.

---

## Step 4 — Three-arm paired (add Planner+Gates)

```python
from poc import PlannerEngine

planner = await bench.run_arm("planner_gates", PlannerEngine())

summary = paired_summary({
    "baseline":      baseline,
    "planner_gates": planner,
    "mine":          mine,
})
```

Note: `paired_summary` returns per-arm win-rates and a `pairwise` block for
two-arm comparisons. For three or more arms, compute the head-to-head pairs
you care about by passing them two at a time.

---

## What to look at in the results

Per-scenario JSON has the full dialog with both sides' text. Useful
things to slice:

```python
import json, glob
from collections import Counter

results = [json.loads(open(f).read()) for f in glob.glob("results/mine/*.json")]

# Win rate
won = sum(r["outcome"] == "won" for r in results)
print(f"win rate: {won}/{len(results)} = {won/len(results):.0%}")

# End-reason distribution
print(Counter(r["end_reason"] for r in results))

# Per-tenant breakdown
for tenant in ("Insurance", "Ecommerce"):
    sub = [r for r in results if r["tenant"] == tenant]
    if sub:
        wr = sum(r["outcome"] == "won" for r in sub) / len(sub)
        print(f"  {tenant}: {wr:.0%} (n={len(sub)})")

# Mean turns to outcome
avg_turns = sum(r["n_live_turns"] for r in results) / len(results)
print(f"avg turns/scenario: {avg_turns:.1f}")

# Telemetry slicing — example: average confidence on won vs lost
def avg_meta(rows, key):
    vals = [t["agent_meta"].get(key) for r in rows for t in r["turns"]
            if isinstance(t["agent_meta"].get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None

won_rows  = [r for r in results if r["outcome"] == "won"]
lost_rows = [r for r in results if r["outcome"] == "lost"]
print(f"avg confidence — won:  {avg_meta(won_rows, 'confidence')}")
print(f"avg confidence — lost: {avg_meta(lost_rows, 'confidence')}")
```

---

## Common integration gotchas

**1. Your engine thinks it should keep selling but the dialog ended already.**
The benchmark stops a scenario when the customer's reply signals close or
explicit decline (see `benchmark._check_customer_outcome`, which delegates to
the active domain pack). Your engine doesn't need to detect "we're done" —
just produce normally; the runner handles termination.

**2. Your engine wants per-customer history across scenarios.**
Each scenario is an independent customer; no cross-scenario memory in the
benchmark. Testing cross-conversation behavior would need a
multi-conversation driver around the benchmark — out of scope here.

**3. Latency budget for the simulator.**
The simulator runs Anthropic Sonnet per turn; that's ~3-5 seconds per
customer reply. If your engine is fast, most of the wall time per turn is
the simulator + rendering, not your code.

**4. `meta` keys vs strict schemas.**
Free-form. The benchmark stores whatever you return. Don't worry about
schema enforcement. If you want to validate, do it in your adapter before
returning.

**5. Reproducibility.**
Every scenario sets `random.seed(scenario.rng_seed)` before the engine's
first turn. LLM stochasticity in your stack is independent of this — if you
want fully deterministic runs, set your LLM temperature to 0.

**6. Post-render gate parity.**
If you want the same anti-staircase / premature-close guards the Planner arm
has, wrap your reply in `poc.post_render_gates.apply(...)` before returning.
Optional.

---

## Step 5 — (optional) Register the engine so it shows up everywhere

The adapter above runs headlessly via `Benchmark(...).run_arm("mine", MyEngine())`.
To also drive it by id (`run_engine("mine")`) and have it appear in the live
dual-panel UI's engine selectors automatically, register an `EngineSpec`:

```python
from poc import register_engine, EngineSpec, ParamSpec
from my_adapter import MyEngine

register_engine(EngineSpec(
    id="mine",
    name="My Engine",
    description="My stack via the adapter.",
    runnable=True,
    params=(ParamSpec(name="api_url", label="API URL", type="string"),),
    factory=lambda api_url=None: MyEngine(api_url=api_url),
))
```

In-tree, call that at import. **Out-of-tree** (no edits to this package), ship
your adapter as its own installable package exposing a `strategist.engines`
entry point — see [`examples/external_engine_plugin/`](examples/external_engine_plugin)
for a minimal one and [`examples/llamacpp_engine_plugin/`](examples/llamacpp_engine_plugin)
for a full local-LLM example. Once installed, `poc.all_engine_specs()`, the
server's `GET /api/engines`, and the UI all pick it up with no further
changes. The engine still implements the exact same `produce()` contract from
Step 1 — registration only adds discovery metadata.

---

## Where to go next

- [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md) — engines, domain packs,
  scenario sources, and benchmark packs in depth.
- [benchmarks/_template/](benchmarks/_template/) — turn your own
  goal-oriented task into a benchmark pack.
- Found something broken or confusing? Open an issue — benchmark results
  (win rates + the paired summary) alongside your adapter make the best
  reports.
