"""Phase 2 — YAML-driven Tier 2 concrete-moves catalog loader.

Architect-approved per `research-notes/2026-05-03-PROPOSAL-strategy-enum-extension.md`
+ OPEN-QUESTIONS.md Q11 sign-off (2026-05-03).

Architecture:
  data/concrete_moves/_base.yaml      ← cross-tenant moves (Tier 2 base)
  data/concrete_moves/{tenant}.yaml   ← per-tenant moves; OVERRIDE _base by name

Loader merges base + tenant. Pydantic-validates at load time (reject-on-invalid).
Cached per process; reload on file mtime change.

Schema requirements (per architect's DP2 conditions):
  - name, primitive, parameters (typed dict), use_when, execution_template,
    gate_class (`A` | `B` | `none`), cg_entity_refs (list), cg_entity_examples
    (per-parameter map of example CG entity names)
  - tenant cap: 10-15 moves per tenant
  - Visioner owns quarterly review; Architect signs off on additions
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

log = logging.getLogger(__name__)


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_DIR = os.environ.get(
    "POC_CONCRETE_MOVES_DIR",
    os.path.join(_PROJECT_ROOT, "data", "concrete_moves"),
)
_BASE_FILE = "_base.yaml"


# ── Pydantic schema (architect-required fields) ─────────────────────────────

GateClass = Literal["A", "B", "none"]


class MoveParameter(BaseModel):
    """Per-parameter spec inside a move's `parameters` dict."""
    type: Literal["number", "string", "list[string]", "list[number]", "bool"] = "string"
    description: str = Field(..., min_length=4)
    cg_entity_examples: list[str] = Field(default_factory=list)
    """Sample CG entity names that legitimate values for this parameter
    should resolve to. Used by move-validity gate (Phase 2) to verify
    supervisor-emitted values aren't fabricated. Empty list = unconstrained."""


class ConcreteMoveSpec(BaseModel):
    """One Tier 2 concrete sales move. All architect-required fields present."""
    name: str = Field(..., min_length=4, max_length=80)
    primitive: str = Field(..., description="Tier 1 primitive this move composes")
    description: str = Field(..., min_length=10)
    parameters: dict[str, MoveParameter] = Field(default_factory=dict)
    use_when: str = Field(..., min_length=20)
    when_not_to_use: str = Field(default="", description="negative criterion")
    execution_template: str = Field(..., min_length=20)
    gate_class: GateClass = Field(default="none")
    cg_entity_refs: list[str] = Field(default_factory=list)
    """CG canonical entity names this move references (via execution_template
    or parameters). Move-validity gate verifies these resolve in the workspace's
    CG before allowing the move."""
    tenant_specific: bool = Field(default=False)
    "Set True for tenant-specific moves; False for cross-tenant base moves."

    @field_validator("primitive")
    @classmethod
    def _check_primitive(cls, v: str) -> str:
        valid = {
            "information", "direct_ask", "scarcity", "objection_handling",
            "reciprocity", "re_engagement", "logistics", "empathy",
            "authority", "commitment", "social_proof",
        }
        if v not in valid:
            raise ValueError(
                f"primitive must be one of Tier 1 primitives: {sorted(valid)}; got '{v}'")
        return v


class ConcreteMovesCatalog(BaseModel):
    """A loaded catalog (base or tenant). Wrapper to support metadata."""
    version: str = Field(default="1.0")
    tenant: str = Field(default="_base")
    moves: list[ConcreteMoveSpec] = Field(..., min_length=1)
    """All moves declared in this catalog. Tenant catalog moves override
    _base catalog moves with the same name."""


# ── Loader with mtime-keyed cache ──────────────────────────────────────────

_CACHE: dict[str, tuple[float, ConcreteMovesCatalog]] = {}


def _load_one_yaml(path: str) -> ConcreteMovesCatalog:
    """Load + validate one YAML file. Reject-on-invalid (raises ValidationError)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected top-level dict, got {type(raw).__name__}")
    return ConcreteMovesCatalog(**raw)


def _is_cache_fresh(path: str, ts: float) -> bool:
    try:
        return os.path.getmtime(path) <= ts
    except FileNotFoundError:
        return False


def load_catalog(tenant: str | None = None) -> list[ConcreteMoveSpec]:
    """Load the merged catalog for `tenant`. Tenant override-by-name semantics:
    if a move named X exists in both _base and tenant.yaml, the tenant version
    wins (full replacement).

    Returns: ordered list[ConcreteMoveSpec] — base moves first, then tenant
    moves that don't override; tenant-overridden base moves replaced in place.
    """
    base_path = os.path.join(CATALOG_DIR, _BASE_FILE)
    tenant_norm = (tenant or "").strip().lower()
    tenant_path = (os.path.join(CATALOG_DIR, f"{tenant_norm}.yaml")
                    if tenant_norm and tenant_norm != "_base" else None)

    cache_key = f"{base_path}::{tenant_path or '-'}"
    cached = _CACHE.get(cache_key)
    if cached:
        ts, catalog = cached
        if (_is_cache_fresh(base_path, ts)
                and (tenant_path is None or _is_cache_fresh(tenant_path, ts))):
            return list(catalog.moves)

    # Load base
    if not os.path.exists(base_path):
        log.warning("concrete_moves: base catalog missing at %s", base_path)
        return []
    try:
        base = _load_one_yaml(base_path)
    except (ValidationError, ValueError, yaml.YAMLError) as e:
        log.error("concrete_moves: base catalog INVALID — refusing to load: %s", e)
        raise

    base_moves_by_name = {m.name: m for m in base.moves}

    # Load tenant overlay if present
    tenant_moves_by_name: dict[str, ConcreteMoveSpec] = {}
    if tenant_path and os.path.exists(tenant_path):
        try:
            t = _load_one_yaml(tenant_path)
        except (ValidationError, ValueError, yaml.YAMLError) as e:
            log.error("concrete_moves: tenant catalog %s INVALID — using base only: %s",
                      tenant_path, e)
            t = None
        if t:
            for m in t.moves:
                tenant_moves_by_name[m.name] = m
            n_total = len(base_moves_by_name) + sum(
                1 for n in tenant_moves_by_name if n not in base_moves_by_name)
            if n_total > 15:
                log.warning(
                    "concrete_moves: tenant=%s catalog has %d total moves "
                    "(base + tenant-only) — architect-set cap is 15; consider pruning",
                    tenant, n_total)

    # Merge: tenant overrides by name; tenant-only moves appended
    merged: list[ConcreteMoveSpec] = []
    seen_names = set()
    for m in base.moves:
        if m.name in tenant_moves_by_name:
            merged.append(tenant_moves_by_name[m.name])
        else:
            merged.append(m)
        seen_names.add(m.name)
    for m_name, m in tenant_moves_by_name.items():
        if m_name not in seen_names:
            merged.append(m)

    # Wrap merged result for cache (tenant field for diagnostics)
    out_catalog = ConcreteMovesCatalog(
        version="merged", tenant=tenant_norm or "_base", moves=merged)
    _CACHE[cache_key] = (time.time(), out_catalog)

    log.info("concrete_moves: loaded %d moves for tenant=%s "
             "(base=%d, tenant_only=%d, overridden=%d)",
             len(merged), tenant_norm or "_base",
             len(base_moves_by_name),
             sum(1 for n in tenant_moves_by_name if n not in base_moves_by_name),
             sum(1 for n in tenant_moves_by_name if n in base_moves_by_name))
    return merged


def render_moves_for_supervisor_prompt(tenant: str | None = None) -> str:
    """Format the loaded catalog as a markdown section for Mode 1b's system
    prompt. Replaces the hardcoded version in `concrete_moves.py`.

    The supervisor reads this when deciding `strategy.concrete_move`. Includes
    BOTH cross-tenant base moves AND tenant-specific moves (so the supervisor
    sees the full vocabulary available for this tenant)."""
    moves = load_catalog(tenant)
    if not moves:
        return ""
    lines = [
        "## CONCRETE MOVES (Tier 2 — what to actually DO)",
        "",
        "After picking strategy.primary (Tier 1 abstract primitive), OPTIONALLY pick",
        "a concrete_move from the list below. The concrete_move tells the conversation",
        "agent the LITERAL ACTION to take, with specific parameters.",
        "",
        "Pick a concrete_move when there's a *concrete decision to make* — a price,",
        "a scope, an escalation, a conditional close. Leave it null on simple turns",
        "(small talk, plain commitment confirmation, etc.).",
        "",
        "If you pick one, you MUST populate ALL its parameters with grounded values.",
        "Don't invent parameters — if you can't ground them, don't pick the move.",
        "Parameters that name CG entities should reference real entities in",
        f"workspace='{tenant or '?'}' (cg_entity_examples per parameter listed below).",
        "",
        "Available moves:",
        "",
    ]
    for m in moves:
        tag = "[tenant]" if m.tenant_specific else "[base]"
        lines.append(f"### `{m.name}` {tag} (composes: {m.primitive})")
        lines.append(f"  {m.description}")
        if m.parameters:
            lines.append("  parameters:")
            for pname, pspec in m.parameters.items():
                ex = (f"  e.g. {', '.join(pspec.cg_entity_examples[:3])}"
                      if pspec.cg_entity_examples else "")
                lines.append(f"    - {pname} ({pspec.type}): {pspec.description}{ex}")
        lines.append(f"  use when: {m.use_when}")
        if m.when_not_to_use:
            lines.append(f"  do NOT use when: {m.when_not_to_use}")
        if m.cg_entity_refs:
            lines.append(f"  cg_entity_refs: {', '.join(m.cg_entity_refs)}")
        lines.append("")
    lines.append(
        "## SCHEMA — concrete_move output (STRICT)"
    )
    lines.append("")
    lines.append("Output the chosen move as `strategy.concrete_move` with EXACTLY")
    lines.append("these two keys:")
    lines.append("  - `name`: string, MUST EXACTLY MATCH one of the move names listed above")
    lines.append("  - `parameters`: object, MUST contain ALL parameters listed for that move")
    lines.append("")
    lines.append("⚠ COMMON MISTAKES — DO NOT MAKE:")
    lines.append("  - Do NOT use `source_ref` as the key for the move identifier "
                 "(that key belongs in `knowledge.facts_to_anchor[].source_ref`,")
    lines.append("    which is a DIFFERENT and unrelated field).")
    lines.append("  - Do NOT use `move`, `id`, `concrete_move_name`, or any other "
                 "synonym — only the literal key `name`.")
    lines.append("  - Do NOT make up a move name not in the catalog above.")
    lines.append("  - Do NOT leave `parameters` empty if the move requires parameters.")
    lines.append("")
    lines.append("CORRECT shape:")
    lines.append('```json')
    lines.append('{')
    lines.append('  "strategy": {')
    lines.append('    "primary": "objection_handling",')
    lines.append('    "concrete_move": {')
    lines.append('      "name": "match_competitor_offer",')
    lines.append('      "parameters": {')
    lines.append('        "target_price": 2200,')
    lines.append('        "magnitude_cap": 350,')
    lines.append('        "rationale": "customer disclosed Wesure 2200 with same coverage"')
    lines.append('      }')
    lines.append('    }')
    lines.append('  }')
    lines.append('}')
    lines.append('```')
    lines.append("")
    lines.append("Or `concrete_move: null` if no concrete decision this turn.")
    return "\n".join(lines)


# ── Callers: get_move + render for build_answer ─────────────────────────────

def get_move(name: str, tenant: str | None = None) -> ConcreteMoveSpec | None:
    """Lookup a move by name from the merged catalog for `tenant`. Returns
    None if the move isn't in this tenant's catalog. Tenant-aware: a move
    that exists only in tenant.yaml will be visible only when called with
    that tenant."""
    moves = load_catalog(tenant)
    for m in moves:
        if m.name == name:
            return m
    return None


def validate_concrete_move(cm: object, tenant: str | None = None) -> tuple[bool, str | None, dict]:
    """R6 — Phase 2.5 validation. Used by both the chain stage (advisory)
    and the supervisor retry (regenerative). Returns
        (is_valid, reason_str_or_None, meta_dict)
    where meta has:
        - move_name (resolved from name/source_ref/move/id)
        - used_alias (True if non-canonical key was used)
        - missing_params, empty_params (lists)
        - available_moves (list of all valid move names for this tenant —
          fed back to the LLM correction prompt)
    """
    catalog = load_catalog(tenant)
    available_moves = [m.name for m in catalog]
    meta = {
        "move_name": None,
        "used_alias": False,
        "missing_params": [],
        "empty_params": [],
        "available_moves": available_moves,
    }
    # Tier 2 is OPTIONAL — null/None is valid
    if cm is None or cm == "null":
        return True, "no_move_picked", meta
    if not isinstance(cm, dict):
        return False, f"malformed_type:{type(cm).__name__}", meta
    move_name = (
        cm.get("name") or cm.get("source_ref")
        or cm.get("move") or cm.get("id")
    )
    used_alias = bool(move_name and not cm.get("name"))
    meta["move_name"] = move_name
    meta["used_alias"] = used_alias
    if not move_name:
        return False, "missing_name", meta
    move = get_move(move_name, tenant=tenant)
    if move is None:
        return False, "unknown_move", meta
    params = cm.get("parameters") or {}
    missing = []
    empty = []
    for p_name in move.parameters.keys():
        if p_name not in params:
            missing.append(p_name)
        elif params[p_name] in (None, "", []):
            empty.append(p_name)
    meta["missing_params"] = missing
    meta["empty_params"] = empty
    if missing or empty:
        return False, "param_incomplete", meta
    return True, "valid", meta


def render_move_for_build_answer(directive_strategy: dict | None,
                                  tenant: str | None = None) -> str:
    """When the supervisor's directive includes a concrete_move, format it as
    an `## Execute this concrete move` block for prompt_build_answer's user
    context. Returns empty string if no move was picked or unknown.

    Tenant-aware: the move name might exist only in this tenant's catalog
    (e.g. `bundle_artist_shell_upgrade` is Heavys-only).
    """
    import json as _json
    if not directive_strategy:
        return ""
    cm = directive_strategy.get("concrete_move")
    if not isinstance(cm, dict):
        return ""
    # Fix A (2026-05-03 smoke finding): the supervisor LLM sometimes emits
    # the move identifier under `source_ref` instead of `name` (mis-copying
    # the pattern from `knowledge.facts_to_anchor[].source_ref`). Treat them
    # as aliases so the move-rendering doesn't silently no-op.
    name = cm.get("name") or cm.get("source_ref") or cm.get("move") or cm.get("id")
    if not name:
        return ""
    params = cm.get("parameters") or {}
    move = get_move(name, tenant=tenant)
    if move is None:
        return (
            f"## Execute this concrete move\n"
            f"Move name: `{name}` (UNKNOWN — supervisor picked a move not in the "
            f"`{tenant or '_base'}` catalog; fall back to your default judgment)\n"
            f"Parameters: {_json.dumps(params, indent=2)}"
        )
    # Render the move's execution_template with the supervisor's parameters.
    # Each parameter we expect ({foo}) is substituted; missing parameters are
    # left as literal placeholder text so the LLM can see what was missing.
    try:
        rendered = move.execution_template.format(**params)
    except (KeyError, IndexError) as e:
        rendered = (
            f"(template render failed: {e}; raw parameters below)\n"
            f"{_json.dumps(params, indent=2)}"
        )
    lines = [
        "## Execute this concrete move",
        f"**Move:** `{move.name}` (composes Tier-1 primitive: {move.primitive}, "
        f"tenant_specific={move.tenant_specific})",
        f"**Parameters (supervisor-grounded):**",
    ]
    for k, v in params.items():
        lines.append(f"  - {k}: {v}")
    if move.cg_entity_refs:
        lines.append(f"**CG-canonical references:** "
                     f"{', '.join(move.cg_entity_refs)} "
                     f"(use these names verbatim where the template references them)")
    lines.append("")
    lines.append(f"**Execution instruction:** {rendered}")
    lines.append("")
    lines.append(
        "This move was chosen by the supervisor with full context. Execute it "
        "directly — don't preamble with 'let me check' or 'I understand'. The "
        "supervisor has already done the strategic reasoning."
    )
    return "\n".join(lines)


# ── CLI for validation ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                          format="%(levelname)s [%(name)s] %(message)s")
    tenant = sys.argv[1] if len(sys.argv) > 1 else None
    moves = load_catalog(tenant)
    print(f"Loaded {len(moves)} moves for tenant={tenant or '_base'}")
    for m in moves:
        tag = "[tenant]" if m.tenant_specific else "[base]"
        print(f"  {tag} {m.name} (primitive={m.primitive}, gate_class={m.gate_class}, "
              f"cg_refs={m.cg_entity_refs[:2] or '-'})")
