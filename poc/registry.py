"""Engine registry — the single source of truth for which Strategy/Supervisor
algorithms exist, what they're called, what knobs they expose, and how to build
one.

Both consumers read from here so an engine is declared in ONE place:
  - the headless `Benchmark` (run any engine by id), and
  - the live dual-panel server (`GET /api/engines` → dynamic UI selectors).

Adding an engine:
  - In-tree:   call `register(EngineSpec(...))` (see the builtins below).
  - Out-of-tree (no core edits): ship a package that exposes a setuptools
    entry point in group ``strategist.engines`` pointing at an `EngineSpec`
    (or a zero-arg callable returning one / a list of them). It is discovered
    automatically the first time the registry is read. See
    `examples/external_engine_plugin/` for a worked example.

An engine implements the `Engine` protocol (`poc.engine.Engine`):

    text, meta = await engine.produce(opp_meta, dialog_history, business_rules)

`live_mode` tells the live server how to drive the engine in a panel:
  - "produce" — generic: the replayer calls `engine.produce(...)` each turn.
    This is the path every pluggable / third-party engine uses.
  - "native"  — the engine's live behavior is wired into the replayer/server
    scaffolding under its id (the legacy "strategist" supervised flow and the
    "baseline" vanilla flow). Headless benchmarking still uses `factory`.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


# ── Specs ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParamSpec:
    """A user-tunable engine parameter, surfaced as a UI control."""
    name: str
    label: str
    type: str = "enum"               # "enum" | "bool" | "string"
    default: Any = None
    choices: tuple[str, ...] = ()    # for type == "enum"
    help: str = ""


@dataclass(frozen=True)
class EngineSpec:
    id: str                          # stable slug used in API + results ("planner")
    name: str                        # UI display label ("Planner + Gates")
    description: str = ""            # one-liner for the UI
    runnable: bool = True            # False → can't be benchmarked (e.g. needs prod DB)
    requires: tuple[str, ...] = ()   # capability tags shown as badges ("mysql","lightrag")
    params: tuple[ParamSpec, ...] = ()
    live_mode: str = "produce"       # "produce" | "native"
    factory: Optional[Callable[..., Any]] = None  # (**param_values) -> Engine

    def create(self, **param_values) -> Any:
        """Instantiate the engine, applying parameter defaults."""
        if self.factory is None:
            raise RuntimeError(
                f"engine {self.id!r} is not runnable in this package "
                f"({'requires ' + ', '.join(self.requires) if self.requires else 'no factory'})")
        values = {p.name: p.default for p in self.params}
        values.update({k: v for k, v in param_values.items() if v is not None})
        return self.factory(**values)

    def to_public(self) -> dict:
        """JSON-serializable view for the /api/engines endpoint."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "runnable": self.runnable,
            "requires": list(self.requires),
            "params": [asdict(p) for p in self.params],
        }


# ── Registry store ───────────────────────────────────────────────────────────

_REGISTRY: dict[str, EngineSpec] = {}
_DISCOVERED = False


def register(spec: EngineSpec, *, replace: bool = False) -> EngineSpec:
    """Add an engine to the registry. Raises on duplicate id unless replace=True."""
    if not isinstance(spec, EngineSpec):
        raise TypeError(f"register() expects EngineSpec, got {type(spec).__name__}")
    if spec.id in _REGISTRY and not replace:
        raise ValueError(f"engine id {spec.id!r} already registered")
    _REGISTRY[spec.id] = spec
    return spec


def has(engine_id: str) -> bool:
    _ensure_discovered()
    return engine_id in _REGISTRY


def get(engine_id: str) -> EngineSpec:
    _ensure_discovered()
    try:
        return _REGISTRY[engine_id]
    except KeyError:
        raise KeyError(
            f"unknown engine {engine_id!r}; registered: "
            f"{', '.join(sorted(_REGISTRY)) or '(none)'}") from None


def all_specs() -> list[EngineSpec]:
    """All registered engines, runnable ones first, then alphabetical."""
    _ensure_discovered()
    return sorted(_REGISTRY.values(), key=lambda s: (not s.runnable, s.id))


def create(engine_id: str, **param_values) -> Any:
    return get(engine_id).create(**param_values)


# ── Entry-point discovery (out-of-tree plugins) ──────────────────────────────

ENTRY_POINT_GROUP = "strategist.engines"


def _ensure_discovered() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True  # set first: a failing plugin must not retrigger discovery
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover - importlib.metadata always present on 3.8+
        return
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # Python <3.10 select-by-kwarg signature
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            obj = ep.load()
            specs = obj() if callable(obj) and not isinstance(obj, EngineSpec) else obj
            for spec in (specs if isinstance(specs, (list, tuple)) else [specs]):
                register(spec, replace=True)
                log.info("registry: discovered engine %r from plugin %s", spec.id, ep.name)
        except Exception as e:  # one bad plugin must not break the rest
            log.warning("registry: failed to load engine plugin %s: %s", ep.name, e)


# ── Built-in engines ─────────────────────────────────────────────────────────
# Factories import lazily so registry import stays cheap and side-effect-free.

def _make_baseline(**_):
    from engine import BaselineEngine  # flat import (package dir on sys.path)
    return BaselineEngine()


def _make_planner(planner_envelope: str = "off", **_):
    from engine import PlannerEngine  # flat import (package dir on sys.path)
    return PlannerEngine(planner_envelope=planner_envelope)


def _register_builtins() -> None:
    register(EngineSpec(
        id="baseline",
        name="Baseline (Original agent)",
        description="Single-call production agent — customer profile + dialogue, "
                    "no directive, planning, or retrieval. The 'ship without thinking' arm.",
        runnable=True,
        live_mode="native",
        factory=_make_baseline,
    ))
    register(EngineSpec(
        id="planner",
        name="Planner + Gates",
        description="PCA-derived state-graph planner + chain-of-thought action "
                    "selection + post-render safety gates. Self-contained.",
        runnable=True,
        live_mode="produce",
        params=(
            ParamSpec(
                name="planner_envelope",
                label="Econ envelope",
                type="enum",
                default="off",
                choices=("off", "auto", "always"),
                help="Pricing-envelope mode: off, difficulty-gated (auto), or always.",
            ),
        ),
        factory=_make_planner,
    ))
    register(EngineSpec(
        id="strategist",
        name="Strategist (Supervisor)",
        description="Mining + cohort retrieval + multi-stage supervisor + safety "
                    "gates. Requires the production system MySQL + Knowledge Graph; "
                    "source included for review.",
        runnable=False,
        requires=("mysql", "lightrag"),
        live_mode="native",
        factory=None,
    ))


_register_builtins()
