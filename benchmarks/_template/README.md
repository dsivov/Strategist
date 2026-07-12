# Create your own benchmark pack

A benchmark pack turns any goal-oriented conversation task — support
resolution, retention saves, appointment booking, debt collection, donation
asks — into an A/B benchmark that works everywhere: the headless runner, the
REST API, and the live dual-panel UI.

The two bundled packs ([`insurance-renewal`](../insurance-renewal),
[`ecommerce-cart`](../ecommerce-cart)) are worked examples of this template.

## 1. Copy this directory

```bash
cp -r benchmarks/_template benchmarks/my-benchmark
cd benchmarks/my-benchmark
mv pack.json.example pack.json
```

Directories starting with `_` are ignored by the pack scanner, so nothing
shows up until you rename `pack.json.example`.

## 2. Fill in `pack.json`

```json
{
  "id": "my-benchmark",
  "name": "My Benchmark",
  "description": "One paragraph: what situation the agent is dropped into.",
  "goal": "What counts as won, in one sentence.",
  "domain": "sales",
  "scenario_source": {
    "type": "json",
    "path": "scenarios.json",
    "filter": {}
  }
}
```

- `scenario_source.path` is relative to the pack directory. You can bring your
  own file (as here) or slice a shared dataset with `filter` (equality match on
  top-level scenario fields) the way the bundled packs slice
  `data/benchmark/v1_scenarios.json` by `tenant`.
- `domain` names the domain pack the scenarios are written for (see step 4).

## 3. Author `scenarios.json`

`scenarios.sample.json` in this directory is a minimal two-scenario skeleton.
The full schema is documented in the [root README](../../README.md#the-benchmark-dataset);
the essentials per scenario:

| Field | Purpose |
|-------|---------|
| `scenario_id` | short stable id (shows in the UI + results filenames) |
| `opp_id` | unique conversation id (UUID) |
| `tenant` | the "company" the agent represents |
| `real_outcome` | `won`/`lost` in the anchoring conversation (if you have one) |
| `attributes` | persona dimensions — motivator, decision logic, trust level, … the simulator plays this person |
| `anchors` | numeric context the negotiation is about (prices, offers, limits) |
| `seed_messages` | opening exchange (and, if available, the full historical transcript the simulator uses as a posture reference) |
| `rng_seed` | per-scenario seed for reproducibility |

No historical transcript? Provide just the opening agent message (and
optionally the customer's first reply) as `seed_messages` — the simulator then
plays the persona from `attributes` alone.

## 4. Pair it with a domain pack (if the framing differs)

The domain pack supplies what "this kind of conversation" means: tenant
framing, how numeric anchors are rendered into the prompt, and — critically —
the **won/lost detectors** applied to customer replies. If the default `sales`
framing doesn't fit (e.g. support resolution has no prices and "that worked!"
means won), register your own:

```python
from poc import DomainPack, register_domain

class SupportDomain(DomainPack):
    name = "support"
    def describe_tenant(self, tenant): return "B2C technical support"
    def render_anchor_section(self, anchors): return ""
    def detect_close(self, text):   return "that worked" in text.lower()
    def detect_decline(self, text): return "cancel" in text.lower()

register_domain(SupportDomain())
```

Then set `"domain": "support"` in `pack.json`. Full guide:
[docs/PLUGIN_GUIDE.md](../../docs/PLUGIN_GUIDE.md).

## 5. Run it

```python
import asyncio
from poc import Benchmark
from benchmark_packs import load_pack_scenarios

async def main():
    scenarios = load_pack_scenarios("my-benchmark")
    bench = Benchmark(scenarios, results_dir="./results")
    await bench.run_engine("baseline")     # any registered engine id
    await bench.run_engine("planner")

asyncio.run(main())
```

The live UI picks the pack up automatically: restart the server and
"My Benchmark" appears in the Benchmark selector, its scenarios in the
sidebar, ready for single or batch A/B runs.
