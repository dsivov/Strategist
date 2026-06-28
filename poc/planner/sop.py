"""SOP graph — PCA §3 representation. Planner-owned, Strategist-independent.

Directed graph over agent_actions ∪ user_states with typed edges
{none, fwd, back, bi} ≡ PCA's {–, →, ←, ↔}. Persisted per (tenant, opp_type)
as JSON under planner/data/sop_graph/.

Node vocabulary is derived from the production won-deal schema (the
primary_strategy taxonomy + objection_category + commitment bands) — read as
*data*, never via Strategist code.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# PCA edge directions {–, →, ←, ↔} as ascii enum (json/code-safe).
EDGE_DIRS = ("none", "fwd", "back", "bi")
_DIR_GLYPH = {"none": "–", "fwd": "→", "back": "←", "bi": "↔"}

# Agent actions A — the 11 production strategy primitives + 2 structural
# terminals the taxonomy doesn't name explicitly but every SOP needs.
AGENT_ACTIONS = [
    "information", "direct_ask", "re_engagement", "reciprocity",
    "objection_handling", "scarcity", "logistics", "authority",
    "commitment", "empathy", "social_proof",
    "finalize_close",   # explicit CTA / payment-info / lock-in
    "farewell",         # graceful wind-down
]

# User states S — objection_category ∪ commitment-band ∪ terminals.
USER_STATES = [
    "engaged",
    "price_objection", "timing_objection", "trust_objection",
    "competitor_objection", "need_objection", "authority_objection",
    "near_close",       # commitment_level >= 4, not yet closed
    "closed_won",        # SUCCESS terminal
    "disengaging",       # losing engagement, recoverable
    "lost",              # FAIL terminal
]
TERMINAL_SUCCESS = "closed_won"
TERMINAL_FAIL = "lost"

_DATA_DIR = Path(__file__).resolve().parent / "data" / "sop_graph"


def _artifact_path(tenant: str, opp_type: str) -> Path:
    return _DATA_DIR / f"{tenant}__{opp_type}.json"


def new_graph(tenant: str, opp_type: str) -> dict:
    return {
        "tenant": tenant,
        "opp_type": opp_type,
        "agent_actions": list(AGENT_ACTIONS),
        "user_states": list(USER_STATES),
        "edges": [],   # list of {"src","dst","dir"}
        "meta": {},
    }


def all_nodes(g: dict) -> set[str]:
    return set(g.get("agent_actions", [])) | set(g.get("user_states", []))


def validate(g: dict) -> tuple[bool, list[str]]:
    """Structural validation. Returns (ok, problems)."""
    problems: list[str] = []
    nodes = all_nodes(g)
    for e in g.get("edges", []):
        if e.get("src") not in nodes:
            problems.append(f"edge src not a node: {e.get('src')!r}")
        if e.get("dst") not in nodes:
            problems.append(f"edge dst not a node: {e.get('dst')!r}")
        if e.get("dir") not in EDGE_DIRS:
            problems.append(f"bad edge dir: {e.get('dir')!r}")
    if TERMINAL_SUCCESS not in g.get("user_states", []):
        problems.append("missing success terminal closed_won")
    # success terminal must be reachable from 'engaged' over fwd/bi edges
    adj: dict[str, set[str]] = {}
    for e in g.get("edges", []):
        if e.get("dir") in ("fwd", "bi"):
            adj.setdefault(e["src"], set()).add(e["dst"])
        if e.get("dir") in ("back", "bi"):
            adj.setdefault(e["dst"], set()).add(e["src"])
    seen, stack = set(), ["engaged"]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, ()))
    if TERMINAL_SUCCESS not in seen:
        problems.append("closed_won unreachable from 'engaged'")
    orphan = [a for a in g.get("agent_actions", [])
              if a not in adj and not any(a == e.get("dst")
                                          for e in g.get("edges", []))]
    if orphan:
        problems.append(f"orphan agent_actions (no edges): {orphan}")
    return (not problems, problems)


def children(g: dict, node: str) -> list[str]:
    """SOP-valid successors of `node` over fwd/bi edges (the §5.3 constraint
    set the online planner appends to the prompt)."""
    out = []
    for e in g.get("edges", []):
        if e.get("src") == node and e.get("dir") in ("fwd", "bi"):
            out.append(e["dst"])
        if e.get("dst") == node and e.get("dir") == "bi":
            out.append(e["src"])
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def save(g: dict) -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = _artifact_path(g["tenant"], g["opp_type"])
    g.setdefault("meta", {})["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                          time.gmtime())
    p.write_text(json.dumps(g, indent=2, ensure_ascii=False))
    return p


def load(tenant: str, opp_type: str) -> dict | None:
    p = _artifact_path(tenant, opp_type)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def ascii_summary(g: dict) -> str:
    lines = [f"SOP {g['tenant']}/{g['opp_type']} — "
             f"{len(g['agent_actions'])} actions, "
             f"{len(g['user_states'])} states, {len(g['edges'])} edges"]
    for e in g.get("edges", [])[:60]:
        lines.append(f"  {e['src']} {_DIR_GLYPH.get(e['dir'],'?')} {e['dst']}")
    return "\n".join(lines)
