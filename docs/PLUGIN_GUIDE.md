# Plugin guide

How to extend the platform: a new **engine**, a new **domain pack**, or a new
**scenario source**. None of these require editing the core.

- [Write an engine](#write-an-engine)
- [Register an engine (in-tree)](#register-an-engine-in-tree)
- [Ship an engine as a plugin (out-of-tree)](#ship-an-engine-as-a-plugin-out-of-tree)
- [Engine parameters](#engine-parameters)
- [Write a domain pack](#write-a-domain-pack)
- [Write a scenario source](#write-a-scenario-source)
- [Checklist](#checklist)

---

## Write an engine

An engine is any object implementing the `Engine` protocol — one async method:

```python
class MyEngine:
    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature

    async def produce(self, opp_meta, dialog_history, business_rules=""):
        # opp_meta:        dict — customer profile + scenario context
        # dialog_history:  [{"role": "agent"|"customer", "text": str}, ...]
        # business_rules:  str (may be "")
        last_customer = next(
            (m["text"] for m in reversed(dialog_history) if m["role"] == "customer"),
            "",
        )
        text = await my_model.respond(opp_meta, last_customer, temp=self.temperature)
        return text, {"strategy": "reframe", "tone": "warm"}   # meta is free-form
```

Rules of thumb:

- **Stateless across calls.** `produce` is invoked once per turn; keep per-session
  state out of `self` (the dialogue is passed in each time).
- **Return final, customer-facing text.** The benchmark feeds it straight to the
  simulator as the agent's message.
- **`meta` is free-form** and recorded verbatim — put strategy/tone/confidence/
  gates-fired/anything useful for post-run analysis.
- See `examples/pi_engine_template.py` for an HTTP-backed adapter pattern.

---

## Register an engine (in-tree)

Declare it once with an `EngineSpec`; everything else (runner, API, UI) picks it
up automatically.

```python
from poc import register_engine, EngineSpec, ParamSpec

register_engine(EngineSpec(
    id="myengine",                       # stable slug
    name="My Engine",                    # UI label
    description="One-line description shown in the UI.",
    runnable=True,                       # benchmark-instantiable
    params=(
        ParamSpec(name="temperature", label="Temperature", type="string",
                  default="0.7", help="Sampling temperature."),
    ),
    factory=lambda temperature="0.7": MyEngine(temperature=float(temperature)),
))
```

Then run it by id:

```python
from poc import Benchmark, load_scenarios
bench = Benchmark(load_scenarios())
results = await bench.run_engine("myengine", temperature="0.4")
```

`live_mode` defaults to `"produce"`, which is what you want for a custom engine —
the live server will drive it generically via `produce()`. (The built-in
`baseline` and `strategist` use `"native"` because their live behavior is wired
into the replayer scaffolding.)

---

## Ship an engine as a plugin (out-of-tree)

To distribute an engine as its own installable package — no edits to this repo —
expose a `strategist.engines` entry point. The registry discovers it on first
read. A complete, installable example lives in
[`../examples/external_engine_plugin/`](../examples/external_engine_plugin).

**`pyproject.toml`:**

```toml
[project]
name = "my-strategist-engine"
version = "0.1.0"

[project.entry-points."strategist.engines"]
# name = "module:attribute"
# attribute may be an EngineSpec, or a zero-arg callable returning one / a list.
myengine = "my_engine:get_engines"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

**`my_engine/__init__.py`:**

```python
class MyEngine:
    async def produce(self, opp_meta, dialog_history, business_rules=""):
        ...
        return text, {}

def get_engines():
    # Import EngineSpec lazily so this module imports regardless of how the host
    # has the package on sys.path (poc package vs. flat module dir).
    try:
        from poc import EngineSpec, ParamSpec
    except Exception:
        from registry import EngineSpec, ParamSpec
    return [EngineSpec(id="myengine", name="My Engine",
                       factory=lambda: MyEngine())]
```

Install into the same environment as the benchmark and it appears everywhere:

```bash
pip install -e my-strategist-engine
python -c "import poc; print(poc.has_engine('myengine'))"   # True
```

> Verified end-to-end with the bundled `echo` example: after
> `pip install -e examples/external_engine_plugin`, the `echo` engine shows up in
> `poc.all_engine_specs()`, the `GET /api/engines` response, and the UI selectors.

---

## Engine parameters

`ParamSpec` declares a tunable knob that becomes a UI control and a
`run_engine(...)` keyword:

| `type` | UI control | Notes |
|--------|------------|-------|
| `"enum"` | dropdown | provide `choices=(...)` |
| `"bool"` | checkbox | |
| `"string"` | text input | coerce in your factory if numeric |

- `default` is applied when the caller omits the param.
- The server keeps only params the engine declares (others are dropped), so the
  client can't inject unexpected kwargs.
- Per-panel params flow as `engine_params` (R) / `engine_params_left` (L) in the
  run request.

---

## Write a domain pack

A domain pack supplies the domain-specific framing and the won/lost criteria.
Subclass `DomainPack` and override what differs; unspecified methods fall back to
neutral defaults.

```python
from poc import DomainPack, set_active_domain

class SupportDomain(DomainPack):
    name = "support"
    tenants = ("Acme",)

    def describe_tenant(self, tenant):
        return "B2C technical support; resolution = customer confirms the fix"

    def opp_type_note(self, opp_type):
        return "\n  → Support contact, not a sale." if opp_type else ""

    def render_anchor_section(self, anchors):
        return ""                          # no pricing in a support domain

    def detect_close(self, text):
        return "resolved" in text.lower() or "that worked" in text.lower()

    def detect_decline(self, text):
        return "still broken" in text.lower() or "cancel" in text.lower()

set_active_domain(SupportDomain())         # actor + benchmark now use it
# or: POC_DOMAIN=support  (env var, resolved on first use)
```

`SalesDomainPack` is the default and is byte-for-byte identical to the original
behavior. A runnable demo is at
[`../examples/custom_domain_pack.py`](../examples/custom_domain_pack.py).

> The bundled dataset is sales/insurance; a genuinely different domain also needs
> a matching scenario source (below) so the data and the framing agree.

---

## Write a scenario source

Implement the `ScenarioSource` protocol to load scenarios from any backend:

```python
from poc import set_scenario_source

class MyScenarioSource:
    def load_scenarios(self):                 # -> list[dict]
        ...
    def fetch_opp_meta(self, opp_id):         # -> dict | None
        ...
    def fetch_messages(self, opp_id):         # -> list[dict]
        ...
    def fetch_business_rules(self, company):  # -> str
        ...
    def fetch_anchors(self, opp_id, opp_meta=None):  # -> dict
        ...

set_scenario_source(MyScenarioSource())
```

For JSON in the bundled shape, reuse the default directly:

```python
from poc import JsonScenarioSource, set_scenario_source
set_scenario_source(JsonScenarioSource("/path/to/my_scenarios.json"))
```

`load_scenarios()` (no args) then reads from your source. The scenario schema is
documented in the [root README](../README.md#the-benchmark-dataset).

---

## Checklist

- [ ] `produce()` returns `(text, meta)` and is stateless across turns.
- [ ] `EngineSpec.id` is unique; `params` cover every tunable knob.
- [ ] Out-of-tree: `strategist.engines` entry point added; `get_engines` imports
      `EngineSpec` lazily.
- [ ] `pip install -e .` then `poc.has_engine("<id>")` returns `True`.
- [ ] Domain pack: `detect_close` / `detect_decline` defined; default
      (`sales`) restored after temporary swaps in tests.
- [ ] New domain → paired with a matching scenario source.
