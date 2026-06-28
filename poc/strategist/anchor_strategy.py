"""Profile-aware anchor strategy loader (2026-05-10).

Loads `data/anchor_strategy/<Tenant>.yaml`, matches the customer's profile
fields against the rules in `profiles[]`, and returns the concatenated
supervisor guidance text. The supervisor's user prompt receives this text
as a "## profile_anchor_guidance" block, making profile-aware anchoring
authoritative rather than prose-only.

Key entry point:
    build_profile_guidance(tenant, opp_meta) -> str
        Returns the guidance markdown to inject into the supervisor's
        user prompt, or empty string if no rules match.

YAML schema (per tenant):
    profiles:
      - match:
          trust_level: Skeptical
          # all keys must match the customer's profile (AND semantics)
          # special key `_any: true` matches every profile
        supervisor_guidance: |
          ## CRITICAL ...
          ...

Match semantics: ALL keys in a rule's `match` block must equal the
corresponding fields in `opp_meta` (case-insensitive). Multiple rules can
match; their `supervisor_guidance` blocks concatenate in YAML order. The
catch-all `_any: true` rule should be last.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

log = logging.getLogger(__name__)


_CACHE: dict[str, dict[str, Any]] = {}
_DIR_CACHE: Path | None = None


def _config_dir() -> Path:
    global _DIR_CACHE
    if _DIR_CACHE is not None:
        return _DIR_CACHE
    override = os.environ.get("POC_ANCHOR_STRATEGY_DIR")
    if override:
        _DIR_CACHE = Path(override)
        return _DIR_CACHE
    project_root = Path(__file__).resolve().parent.parent
    _DIR_CACHE = project_root / "data" / "anchor_strategy"
    return _DIR_CACHE


def _load_yaml(path: Path) -> dict[str, Any]:
    if not _YAML_OK or not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as e:
        log.warning("anchor_strategy: failed to load %s: %s", path, e)
        return {}


def get_config(tenant: str | None) -> dict[str, Any]:
    """Return the anchor-strategy config for a tenant. Cached."""
    if not tenant:
        return {}
    key = tenant.strip().lower()
    if not key:
        return {}
    if key in _CACHE:
        return _CACHE[key]

    cfg_dir = _config_dir()
    base = _load_yaml(cfg_dir / "_base.yaml")
    tenant_yaml = {}
    for name in (tenant.strip(), tenant.strip().title(), tenant.strip().upper()):
        candidate = cfg_dir / f"{name}.yaml"
        if candidate.exists():
            tenant_yaml = _load_yaml(candidate)
            if tenant_yaml:
                break

    if base and tenant_yaml:
        merged = dict(base)
        # Merge: tenant-yaml's `profiles` REPLACES base.profiles
        # (each tenant defines its own list); other keys merge shallowly.
        for k, v in tenant_yaml.items():
            merged[k] = v
        cfg = merged
    elif tenant_yaml:
        cfg = dict(tenant_yaml)
    else:
        cfg = {}

    _CACHE[key] = cfg
    return cfg


def _rule_matches(rule_match: dict, opp_meta: dict) -> bool:
    """Return True if all keys in `rule_match` equal opp_meta values
    (case-insensitive string compare). Special key `_any: true` always
    matches."""
    if not isinstance(rule_match, dict):
        return False
    if rule_match.get("_any") is True:
        return True
    for k, v in rule_match.items():
        if k.startswith("_"):
            continue
        actual = opp_meta.get(k)
        if actual is None:
            return False
        if str(actual).strip().lower() != str(v).strip().lower():
            return False
    return True


def build_profile_guidance(tenant: str | None, opp_meta: dict | None) -> str:
    """Match the customer's profile against tenant's profile rules, return
    the concatenated supervisor_guidance markdown. Empty string if no rules
    match or tenant has no anchor_strategy config.

    The catch-all `_any: true` rule (if defined) is included only when no
    OTHER rule matches; this prevents the default text from drowning out
    specific guidance.
    """
    cfg = get_config(tenant)
    if not cfg:
        return ""
    profiles_rules = cfg.get("profiles") or []
    if not isinstance(profiles_rules, list) or not profiles_rules:
        return ""
    opp = opp_meta or {}

    matched_specific = []
    catchall = None
    for rule in profiles_rules:
        if not isinstance(rule, dict):
            continue
        match_block = rule.get("match") or {}
        is_catchall = match_block.get("_any") is True
        if is_catchall:
            catchall = rule
            continue
        if _rule_matches(match_block, opp):
            matched_specific.append(rule)

    rules_to_use = matched_specific if matched_specific else (
        [catchall] if catchall else [])
    if not rules_to_use:
        return ""

    blocks = []
    for rule in rules_to_use:
        guidance = (rule.get("supervisor_guidance") or "").strip()
        if guidance:
            blocks.append(guidance)
    if not blocks:
        return ""
    return "\n\n".join(blocks)


def reset_cache() -> None:
    """Clear the in-process config cache. Useful for tests / hot-reload."""
    global _DIR_CACHE
    _CACHE.clear()
    _DIR_CACHE = None
