"""Positive-direction mined-playbook reader — Planner-owned, treats the
mined YAML library as shared substrate (raw data, not Strategist code).

The library lives at `data/script_library/{tenant}/*.yaml` and was mined
from won-deal turn-by-turn lift attribution: high-lift agent phrases became
must_say_template, negative-lift phrases became must_not_say. The Strategist
consumes these via its state machine; the Planner consumes them here purely
as data, no engine-boundary coupling.

Public API:
    load_tenant_playbooks(tenant) -> list[dict]
    format_for_prompt(playbooks, state) -> str   # compact CoT prompt block
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Allow the handoff package to relocate the script library outside the
# planner subpackage. POC_SCRIPT_LIBRARY_DIR wins when set; otherwise we
# fall back to the in-tree default (planner/../data/script_library).
_ROOT = Path(__file__).resolve().parent.parent
_DIR = Path(os.environ["POC_SCRIPT_LIBRARY_DIR"]) if os.environ.get(
    "POC_SCRIPT_LIBRARY_DIR") else _ROOT / "data" / "script_library"


def _load_yaml(path: Path) -> dict | None:
    try:
        import yaml  # PyYAML; available in the poc env
        return yaml.safe_load(path.read_text())
    except Exception:
        return None


def load_tenant_playbooks(tenant: str) -> list[dict]:
    """All approved playbooks for the tenant. Empty list when none.
    Sorted by priority (desc) when present."""
    d = _DIR / tenant
    if not d.exists() or not d.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(d.glob("*.yaml")):
        y = _load_yaml(p)
        if not isinstance(y, dict):
            continue
        if (y.get("status") or "").lower() not in ("approved", "active", "", None):
            continue
        # `_path` is just metadata for telemetry/debugging. relative_to
        # only works when the data lives under the planner package; in the
        # handoff package data is sibling to the package, so fall back to
        # a path relative to the playbook root.
        try:
            y["_path"] = str(p.relative_to(_ROOT))
        except ValueError:
            try:
                y["_path"] = str(p.relative_to(_DIR.parent))
            except ValueError:
                y["_path"] = p.name
        out.append(y)
    out.sort(key=lambda x: int(x.get("priority") or 0), reverse=True)
    return out


def _filter_applies_when(pb: dict, opp_meta: dict, commit_est: int | None) -> bool:
    """Cheap structural fit-check against the playbook's applies_when block.
    Tenant + commitment-band only (phase guess is left to the LLM)."""
    aw = pb.get("applies_when") or {}
    t = aw.get("tenant")
    if t and t != opp_meta.get("company"):
        return False
    conv = aw.get("conversation") or {}
    if commit_est is not None:
        lo = conv.get("commitment_level_min")
        hi = conv.get("commitment_level_max")
        if lo is not None and commit_est < int(lo):
            return False
        if hi is not None and commit_est > int(hi):
            return False
    return True


def _estimate_commitment(state: dict) -> int | None:
    """Light heuristic — last available commit cue in the dialogue. If the
    state already carries an inferred user_state we let that override. The
    Planner's CoT will infer the real user_state anyway; this only filters
    which playbooks are even shown."""
    cl = state.get("commitment_level_hint")
    if isinstance(cl, int):
        return cl
    # very rough: scan customer turns for buying / decline / objection cues
    txt = " ".join((m.get("text") or "")
                   for m in (state.get("dialogue") or [])
                   if m.get("role") == "customer").lower()
    if any(k in txt for k in ("let's do it", "yes please", "sign me up",
                              "last 4", "send the link", "go ahead")):
        return 5
    if any(k in txt for k in ("not interested", "no thanks", "i'll pass",
                              "not for me")):
        return 1
    if any(k in txt for k in ("too expensive", "lower", "discount", "cheaper",
                              "competitor", "another quote")):
        return 3
    return None  # let all playbooks through


def _trim(s: Any, n: int) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def format_for_prompt(playbooks: list[dict], state: dict,
                      max_scripts: int = 3,
                      max_steps_per_script: int = 2) -> str:
    """Render a compact block suitable for the CoT+SOP prompt. Filters by
    tenant + commitment band, then takes the top `max_scripts` by priority,
    surfacing only the most informative arc step(s) per script."""
    if not playbooks:
        return ""
    opp = state.get("opp_meta") or {}
    commit_est = _estimate_commitment(state)
    matched = [pb for pb in playbooks
               if _filter_applies_when(pb, opp, commit_est)]
    if not matched:
        matched = playbooks  # no filter hit → show top-priority as reference
    matched = matched[:max_scripts]
    lines = ["POSITIVE-DIRECTION MINED PATTERNS (high-lift agent moves "
             "from won deals at this tenant; use as guidance, not as rigid "
             "templates):"]
    for pb in matched:
        sid = pb.get("script_id") or pb.get("_path", "?")
        desc = _trim((pb.get("description") or "").splitlines()[0]
                     if pb.get("description") else "", 120)
        lines.append(f"\n  [{sid}] {desc}")
        for step in (pb.get("arc") or [])[:max_steps_per_script]:
            intent = _trim(step.get("turn_intent"), 100)
            must_say = _trim(step.get("must_say_template"), 240)
            mns = step.get("must_not_say") or []
            mns_txt = "; ".join(_trim(x, 60) for x in mns[:3])
            lines.append(f"    • intent: {intent}")
            if must_say:
                lines.append(f"      must_say (template): {must_say}")
            if mns_txt:
                lines.append(f"      must_not_say: {mns_txt}")
    return "\n".join(lines)
