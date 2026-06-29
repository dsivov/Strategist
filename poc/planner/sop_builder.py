"""Offline SOP builder — PCA §5.2 (TCoT), grounded on won-deal evidence.

Faithful to PCA: an LLM authors the typed adjacency graph via
Translation-CoT (reason about each node's successors in NL, then emit JSON).
Our only departure from PCA §4's pure role-play is that we *ground* the
prompt with real won-deal transition frequencies mined from precedents.db
(read as data — no Strategist imports).

Run:  python -m planner.sop_builder --tenant Insurance --opp-type renewal
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from . import sop

_ROOT = Path(__file__).resolve().parent.parent
_PRECEDENTS = _ROOT / "data" / "precedents.db"


def _load_env() -> None:
    """Independent env load (don't rely on the server process)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except Exception:
        pass


def _turn_user_state(objection: str | None, commit) -> str:
    o = (objection or "").strip().lower()
    if o in ("price", "timing", "trust", "competitor", "need", "authority"):
        return f"{o}_objection"
    try:
        c = int(commit) if commit is not None else None
    except (TypeError, ValueError):
        c = None
    if c == 5:
        return sop.TERMINAL_SUCCESS
    if c is not None and c >= 4:
        return "near_close"
    if c is not None and c <= 1:
        return "disengaging"
    return "engaged"


def mine_evidence(tenant: str, limit_opps: int = 250) -> dict:
    """Empirical transition frequencies from won deals: user_state→action and
    action→next_user_state bigrams. Grounding only; the LLM still authors."""
    if not _PRECEDENTS.exists():
        return {"state_to_action": [], "action_to_state": [], "n_opps": 0}
    con = sqlite3.connect(f"file:{_PRECEDENTS}?mode=ro", uri=True)
    opps = [r[0] for r in con.execute(
        "SELECT DISTINCT opp_id FROM precedent_turn "
        "WHERE company=? AND outcome='ClosedWon' LIMIT ?",
        (tenant, limit_opps))]
    s2a = collections.Counter()
    a2s = collections.Counter()
    for opp in opps:
        rows = con.execute(
            "SELECT primary_strategy, objection_category, commitment_level "
            "FROM precedent_turn WHERE opp_id=? AND outcome='ClosedWon' "
            "ORDER BY sequence_number ASC", (opp,)).fetchall()
        prev_state = "engaged"
        for strat, obj, commit in rows:
            if not strat:
                continue
            st = _turn_user_state(obj, commit)
            s2a[(prev_state, strat)] += 1
            nxt = _turn_user_state(obj, commit)
            a2s[(strat, nxt)] += 1
            prev_state = nxt
    con.close()
    return {
        "n_opps": len(opps),
        "state_to_action": s2a.most_common(45),
        "action_to_state": a2s.most_common(45),
    }


def _tcot_prompt(tenant: str, opp_type: str, ev: dict) -> str:
    s2a = "\n".join(f"  {s} → {a}: {n}" for (s, a), n in ev["state_to_action"])
    a2s = "\n".join(f"  {a} ⇒ {st}: {n}" for (a, st), n in ev["action_to_state"])
    return f"""You are designing a Standard Operating Procedure (SOP) graph \
for a {tenant} {opp_type} sales conversation, in the PCA formalism
(Li et al., arXiv:2407.03884).

The SOP is a DIRECTED graph over two node types:
- AGENT ACTIONS: {sop.AGENT_ACTIONS}
- USER STATES:   {sop.USER_STATES}

Edge direction types (PCA): "fwd" (src naturally precedes dst), "back" \
(dst can revert to src), "bi" (either order), "none" (no relation — do not \
emit). Terminals: success="{sop.TERMINAL_SUCCESS}", fail="{sop.TERMINAL_FAIL}".

EMPIRICAL GROUNDING — transition counts mined from {ev['n_opps']} real \
WON {tenant} deals (evidence, not gospel; use judgement):

user_state → agent_action (most frequent):
{s2a}

agent_action ⇒ resulting user_state (most frequent):
{a2s}

TASK (Translation-CoT):
1. REASONING: for each node, briefly state in natural language which nodes \
should follow it and why, consistent with the evidence and sound sales \
practice. Every objection user_state should route to objection_handling \
and back toward near_close; near_close should route to finalize_close; \
finalize_close should reach closed_won; persistent failure routes to \
disengaging→lost.
2. OUTPUT: then emit ONLY a fenced ```json block with:
{{"edges":[{{"src":<node>,"dst":<node>,"dir":<"fwd"|"back"|"bi">}}, ...]}}
Use only the listed node names. Ensure closed_won is reachable from \
engaged. Keep it to the ~30-60 most important edges (no full clique)."""


def build(tenant: str, opp_type: str, model: str = "claude-sonnet-4-5") -> dict:
    _load_env()
    import anthropic
    ev = mine_evidence(tenant)
    prompt = _tcot_prompt(tenant, opp_type, ev)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model, max_tokens=4000,
        messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    # extract the json block
    raw = text
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    edges = json.loads(raw.strip()).get("edges", [])
    g = sop.new_graph(tenant, opp_type)
    nodes = sop.all_nodes(g)
    g["edges"] = [e for e in edges
                  if e.get("src") in nodes and e.get("dst") in nodes
                  and e.get("dir") in sop.EDGE_DIRS and e["dir"] != "none"]
    g["meta"] = {
        "builder": "tcot", "model": model,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_won_opps_evidence": ev["n_opps"],
        "raw_edges_proposed": len(edges),
    }
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="Insurance")
    ap.add_argument("--opp-type", default="renewal")
    ap.add_argument("--model", default="claude-sonnet-4-5")
    a = ap.parse_args()
    g = build(a.tenant, a.opp_type, a.model)
    ok, probs = sop.validate(g)
    p = sop.save(g)
    print(sop.ascii_summary(g))
    print(f"\nvalidate: {'OK' if ok else 'PROBLEMS'}")
    for x in probs:
        print("  -", x)
    print(f"saved: {p}")
    sys.exit(0 if ok else 1)
