# Integration guide — plugging the example stack into the POC benchmark

A step-by-step walkthrough for the example integration team. Goes from a green smoke test to
a 112-scenario paired benchmark against the Baseline arm. Estimated time to
first real result: 1–2 hours of focused work.

---

## Step 0 — Verify the package is alive

```bash
cd POC
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and GEMINI_API_KEY_1

python tests/test_smoke.py
```

You should see `ALL SMOKE TESTS PASSED`. The smoke test makes zero LLM
calls; it just verifies imports, bundled data, and the Engine protocol shape.

If anything fails here, the problem is almost certainly a missing API key
in `.env` (some modules read env at import time) or a Python version below
3.10 (we use match/case and PEP 604 unions throughout).

---

## Step 1 — Write the example adapter

Copy `examples/example_engine_template.py` to `example_adapter.py` (or wherever you
keep your code) and fill in `produce()`:

```python
class ExampleEngine:
    def __init__(self, **config):
        # Stash whatever connection info / clients / models you need.
        self.config = config

    async def produce(self, opp_meta, dialog_history, business_rules=""):
        # ── call your stack however you do today ─────────
        result = await self.engine.predict_and_render(
            opp_meta, dialog_history, business_rules,
        )
        return result.text, {
            "strategy":        result.strategy,
            "tone":            result.tone,
            "hint_confidence": result.confidence,
            "escalated":       False,
        }
```

**The only contract is the method signature.** What you put in `meta` is
recorded verbatim; the benchmark does not interpret it. Put anything that
helps your post-run analysis: strategy/tone, confidence numbers, which
sub-model fired, which gates triggered, escalation flags, training-data
markers, anything.

### Input you receive each turn

**`opp_meta` (dict)** — customer profile + scenario context. Keys:

| Key                       | Type    | Notes |
|---------------------------|---------|-------|
| `id`                      | str     | Opportunity ID (UUID). |
| `company`                 | str     | Tenant — `"Insurance"` (insurance) or `"Ecommerce"` (e-commerce). |
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
| `anchors`                 | dict    | Per-opp pricing: `last_year_price_usd`, `current_quoted_price_usd`, `market_avg_for_segment_usd`, `max_discount_pct_internal`. |
| `voice_profile`           | dict?   | Optional voice cues mined from the historical transcript. |

**`dialog_history` (list[dict])** — `[{"role": "agent"|"customer", "text": str}]`,
ordered chronologically. The most recent customer message is the one your
engine should respond to. (When the example engine doesn't have a customer message to respond
to — first turn, or two agent turns in a row — you can still emit; the
benchmark treats whatever you produce as the agent's move.)

**`business_rules` (str)** — tenant-specific compliance/voice rules. Often
empty during the benchmark (these come from `db.fetch_business_rules` which
the standalone package doesn't wire up). Treat as optional context.

### Output you return each turn

`tuple[str, dict]` — `(customer_facing_text, telemetry_meta)`.

- `customer_facing_text`: what gets sent to the simulator.
- `telemetry_meta`: whatever you want recorded. Suggested keys:
  - `strategy`, `tone` — your taxonomy
  - `hint_confidence` — for the merge architecture's escalation trigger
  - `model_version` — for cross-run diff
  - `escalated` — if your Tier-1 deferred to a slower path
  - `gates_fired`, `gates_regens` — if you ran post-render gates
  - `params` — concrete-move parameters (in the layered-moves architecture)

---

## Step 2 — Run a 3-scenario smoke against the example adapter

```python
import asyncio
from poc import Benchmark, load_scenarios
from example_adapter import ExampleEngine

async def smoke():
    scenarios = load_scenarios()[:3]            # 3 scenarios, ~5 min
    bench = Benchmark(scenarios, results_dir="./example_smoke")
    results = await bench.run_arm(
        "example",
        ExampleEngine(api_url="http://localhost:8080"),
        on_scenario_done=lambda r: print(
            f"  {r['scenario_id']:18} -> {r['outcome']:4} "
            f"({r['end_reason']}, {r['n_live_turns']}t, {r['elapsed_s']}s)"
        ),
    )
    print(f"\n{sum(1 for r in results if r['outcome']=='won')} wins / 3")

asyncio.run(smoke())
```

Expect: 3 scenarios complete without errors, output JSONs land in
`./example_smoke/example/`. If any turn throws, the error is captured in
`agent_meta.error` for that turn rather than crashing the run — check the
output JSONs for the trace.

---

## Step 3 — Run the full 2-arm paired benchmark

```python
from poc import Benchmark, BaselineEngine, load_scenarios
from poc.benchmark import paired_summary
from example_adapter import ExampleEngine

async def run_full():
    scenarios = load_scenarios()                # all 112
    bench = Benchmark(scenarios, results_dir="./results")

    baseline = await bench.run_arm("baseline", BaselineEngine())
    example_arm   = await bench.run_arm("example",       ExampleEngine(...))

    summary = paired_summary({"baseline": baseline, "example": example_arm})
    print(summary)

asyncio.run(run_full())
```

The runs are resumable — re-running skips already-complete `(arm, scenario)`
JSONs. Crash recovery is just "re-run the script."

Estimated wall time: ~2.5 h Baseline + however long the example engine takes per turn × 112
scenarios × ~6 turns avg. the example engine at 35 ms/turn ≈ negligible.

---

## Step 4 — Three-arm paired (add Planner+Gates)

```python
from poc import PlannerEngine

planner = await bench.run_arm("planner_gates", PlannerEngine())

summary = paired_summary({
    "baseline":      baseline,
    "planner_gates": planner,
    "example":            example_arm,
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

results = [json.loads(open(f).read()) for f in glob.glob("results/example/*.json")]

# Win rate
won = sum(r["outcome"] == "won" for r in results)
print(f"Example win rate: {won}/{len(results)} = {won/len(results):.0%}")

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

# Telemetry slicing — example: average hint_confidence on won vs lost
def avg_meta(rows, key):
    vals = [t["agent_meta"].get(key) for r in rows for t in r["turns"]
            if isinstance(t["agent_meta"].get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None

won_rows  = [r for r in results if r["outcome"] == "won"]
lost_rows = [r for r in results if r["outcome"] == "lost"]
print(f"avg hint_confidence — won: {avg_meta(won_rows, 'hint_confidence'):.2f}")
print(f"avg hint_confidence — lost: {avg_meta(lost_rows, 'hint_confidence'):.2f}")
```

---

## Common integration gotchas

**1. the example engine thinks it should produce a strategy but the dialog ended already.**
The benchmark stops a scenario when the customer's reply signals close or
explicit decline (see `benchmark._check_customer_outcome`). Your engine
doesn't need to detect "we're done" — just produce normally; the runner
handles termination.

**2. the example engine wants per-customer history across scenarios.**
Each scenario is an independent customer; no cross-scenario memory in the
benchmark. If you want to test the example engine's per-customer novelty filter, you'd need
to wrap the benchmark in a multi-conversation driver — out of scope here.

**3. Latency budget for the simulator.**
The simulator runs Anthropic Sonnet per turn; that's ~3-5 seconds per
customer reply. If your engine is fast (35 ms), most of the wall time per
turn is the simulator + the customer-facing actor, not your code.

**4. `meta` keys vs strict schemas.**
Free-form. The benchmark stores whatever you return. Don't worry about
schema enforcement. If you want to validate, do it in your adapter before
returning.

**5. Reproducibility.**
Every scenario sets `random.seed(scenario.rng_seed)` before the engine's
first turn. LLM stochasticity in your stack is independent of this — if you
want fully deterministic runs, set your LLM temperature to 0.

---

## Step 5 — (optional) Register the engine so it shows up everywhere

The adapter above runs headlessly via `Benchmark(...).run_arm("example", ExampleEngine())`.
To also drive it by id (`run_engine("example")`) and have it appear in the live
dual-panel UI's engine selectors automatically, register an `EngineSpec`:

```python
from poc import register_engine, EngineSpec, ParamSpec
from example_adapter import ExampleEngine

register_engine(EngineSpec(
    id="example",
    name="Example stack",
    description="Example engine via the example adapter.",
    runnable=True,
    params=(ParamSpec(name="api_url", label="API URL", type="string"),),
    factory=lambda api_url=None: ExampleEngine(api_url=api_url),
))
```

In-tree, call that at import. **Out-of-tree** (no edits to this package), ship
your adapter as its own installable package exposing a `strategist.engines`
entry point — see [`examples/external_engine_plugin/`](examples/external_engine_plugin).
Once installed, `poc.all_engine_specs()`, the server's `GET /api/engines`, and
the UI all pick it up with no further changes. The engine still implements the
exact same `produce()` contract from Step 1 — registration only adds discovery
metadata.

---

## When you're done

Send back:
- The full per-scenario JSONs (or a tarball of `results/example/`)
- Your `example_adapter.py` (so we can re-run on our side if needed)
- A one-pager: what surprised you, what failed, which scenarios look most
  diagnostic

We'll diff against our Baseline + Planner+Gates results from the same
scenarios and write up the head-to-head against the cross-team comparison
in `research-notes/2026-05-05-comparison-with-example-team.md`.

---

## Open questions for the integration discussion

These are the merge-design questions that affect the benchmark's
interpretation but aren't blockers for running it:

1. **Confidence-gate threshold.** The merge architecture escalates to our
   Tier 2 when the example engine's `hint_confidence < 0.6`. The benchmark records
   `hint_confidence` in `meta` but doesn't act on it — you can test the
   escalation logic in a separate arm if useful.
2. **Schema alignment** between the example engine's `(strategy, tone, observed_lift)` and
   our concrete-move parameter blobs. Doesn't affect the per-turn run; it
   affects how we build the merged Tier-1+Tier-2 arm later.
3. **Post-render gate parity.** If you want the same anti-staircase /
   premature-close guard the Planner arm has, you can wrap your reply in a
   call to `poc.post_render_gates.apply(...)` before returning. Optional.
4. **Quality gate for training-data sharing.** If we want the example engine's training set
   to absorb our directives later, we need a measured pass-rate criterion.
   The benchmark gives us the per-turn agreement / disagreement data needed
   to specify it.

These are for the schema-alignment workshop, not for this benchmark run.
