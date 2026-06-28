"""Win-mode plans — Path B Day 4.

Parallel to cluster_plan.py (loss-mode). Win-mode plans are cell-keyed
(tenant × opp_type × motivator × decision_logic), authored from the
win-clustering pipeline output. They differ from loss-mode plans in:

    - No cluster_id (the cell IS the strategy after win-clustering finding)
    - String phase_ids ("3a", "4b", "6_capture_payment", ...) for branch support
    - Phase 2 is a routing phase that picks a branch (A/B/C); the supervisor
      records `active_branch` and subsequent phase advancement follows the
      branch's path
    - Engagement-gate: the plan only loads when trajectory looks like a
      plausible win path (positive sentiment + commit ≥ 2 by turn 4)

Author: research-notes/2026-05-01-win-clustering-findings.md

Public API:
    load_win_plan(tenant, opp_type, motivator, decision_logic) -> dict | None
    engagement_gate_met(plan, panel_metrics, persona) -> bool
    WinPlanState (dataclass) — per-session mutable state
    advance_win_plan(state, supervisor_directive, last_customer_msg) -> (bool, str)
    render_win_plan_section(plan, state) -> str
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

WIN_PLANS_DIR = "/home/dev/development_team_luna/research/poc-supervisor-strategist/data/win_plans"


def _slug(s: str) -> str:
    return (s or "").replace("/", "_").replace(" ", "_")


def _filename(tenant: str, opp_type: str, motivator: str, decision_logic: str) -> str:
    return f"{tenant}__{_slug(opp_type)}__{_slug(motivator)}__{_slug(decision_logic)}.json"


def load_win_plan(tenant: str, opp_type: str | None, motivator: str | None,
                   decision_logic: str | None) -> dict | None:
    """Load matching win-mode plan or None if no plan exists for the cell."""
    if not (tenant and opp_type and motivator and decision_logic):
        return None
    fname = _filename(tenant, opp_type, motivator, decision_logic)
    path = os.path.join(WIN_PLANS_DIR, fname)
    if not os.path.isfile(path):
        log.debug("No win_plan for %s — fallback to plan-less or loss-mode", fname)
        return None
    try:
        with open(path) as f:
            plan = json.load(f)
        log.info("Loaded win_plan %s (%d phases)",
                  plan.get("win_plan_id"), len(plan.get("path") or []))
        return plan
    except Exception as e:
        log.warning("Failed to load win_plan %s: %s", fname, e)
        return None


def engagement_gate_met(plan: dict, panel_metrics: dict | None, turn_idx: int) -> bool:
    """Should the supervisor activate this win-mode plan for the current session?

    Reads the plan's `engagement_gate.load_when_all_of` semantically. We only
    enforce the trajectory-signal condition here — the cell-match conditions
    are already satisfied by the load_win_plan(...) lookup itself.

    Conditions checked here:
      - by turn >= 4: persuasion_score_avg >= 0.4 OR commit_level_max >= 2
      - no abort signal observed (panel_metrics.aborted falsy)

    Args:
        plan: loaded win plan dict
        panel_metrics: dict with keys like persuasion_scores, commit_levels, aborted
        turn_idx: current turn index (1-based)

    Returns: True if trajectory + abort gate cleared. False otherwise.
    """
    if not plan or not panel_metrics:
        # Be permissive at turn 1 — we don't have signal yet
        return turn_idx <= 1

    # Abort check
    if panel_metrics.get("aborted"):
        return False

    # Trajectory check kicks in at turn 4
    if turn_idx < 4:
        return True  # too early to gate on; let plan stay loaded

    persuasion_scores = panel_metrics.get("persuasion_scores") or []
    commit_levels = panel_metrics.get("commit_levels") or []

    persuasion_avg = (
        sum(persuasion_scores) / len(persuasion_scores)
        if persuasion_scores else 0.0
    )
    commit_max = max(commit_levels) if commit_levels else 0

    return persuasion_avg >= 0.4 or commit_max >= 2


@dataclass
class WinPlanState:
    """Per-session mutable state for a win-mode plan."""
    plan: dict
    current_phase_id: str = "1"  # string to handle "3a", "6_capture_payment" etc
    turns_in_current_phase: int = 0
    phase_history: list[str] = field(default_factory=lambda: ["1"])
    completed: bool = False
    aborted: bool = False
    abort_reason: str | None = None
    active_branch: str | None = None  # 'branch_A_bargain_and_stretch' etc
    captured: dict[str, Any] = field(default_factory=dict)
    probe_attempts_by_phase: dict[str, int] = field(default_factory=dict)
    off_plan_strategy_count: int = 0

    def current_phase(self) -> dict | None:
        for ph in (self.plan.get("path") or []):
            if str(ph.get("phase_id")) == str(self.current_phase_id):
                return ph
        return None

    def is_terminal_phase(self) -> bool:
        ph = self.current_phase()
        if ph is None:
            return True
        # Terminal if name contains close/exit
        name = (ph.get("name") or "").lower()
        return "close" in name or "exit" in name or "capture_payment" in name


def _normalize_target_phase(target: str | None, plan: dict) -> str | None:
    """Some plans may reference branches by symbolic names like
    'branch_A_bargain_and_stretch' instead of phase_id. Resolve those to
    the first phase in that branch."""
    if not target:
        return None
    target = str(target)
    # Direct phase_id match?
    for ph in (plan.get("path") or []):
        if str(ph.get("phase_id")) == target:
            return target
    # Branch symbolic name → first phase with matching branch_id
    for ph in (plan.get("path") or []):
        if ph.get("branch_id") == target:
            return str(ph.get("phase_id"))
    return None


def advance_win_plan(state: WinPlanState,
                      supervisor_directive: dict | None,
                      last_customer_msg: str) -> tuple[bool, str | None]:
    """Win-mode advancement supporting string phase_ids + branch routing.

    Decision precedence:
      1. Abort condition (customer signed elsewhere etc.) — phase 1-2 only
      2. Supervisor self-report: plan.phase_status='exit_met' OR
         plan.recommended_next_phase_id set
      3. Phase 2 routing: if current_phase==2 and supervisor specified a
         branch, pick the branch's first phase
      4. Phase budget exhausted → force advance (linear within branch)
      5. Stay
    """
    plan = state.plan
    current = state.current_phase()
    if current is None:
        return False, "no_current_phase"

    # 1. Abort scan — phase 1-2 only, with bargaining-marker guards
    abort_on = (plan.get("goal") or {}).get("abort_on") or []
    cust = (last_customer_msg or "").strip()
    cust_lower = cust.lower()
    abort_keywords = {
        "customer_states_signed_with_competitor": [
            "already signed", "went with another", "signed with another",
            "סגרתי עם אחר", "עברתי לחברה אחרת", "כבר חידשתי", "כבר שילמתי",
        ],
        "customer_states_no_longer_needs_insurance": [
            "no longer need", "i don't need insurance", "selling the car",
            "got rid of the car", "not insuring this car",
        ],
        "customer_explicitly_requests_stop": [
            "please stop messaging", "stop messaging me", "leave me alone",
            "remove me from", "unsubscribe me", "do not contact me",
        ],
    }

    is_early_phase = state.current_phase_id in ("1", "2")
    has_question = cust.rstrip("!.").endswith("?") or "?" in cust[-40:]
    has_bargaining = any(m in cust_lower for m in [
        " but ", " why ", "what would", "what if", "how would", "how could",
        "can you ", "could you ", "would you ", "what's the ", "what is the ",
        "if you ", " maybe ", "checking", "i'll get back",
    ])

    if is_early_phase and not has_question and not has_bargaining:
        for cond in abort_on:
            for kw in abort_keywords.get(cond, []):
                if kw in cust_lower:
                    state.aborted = True
                    state.abort_reason = cond
                    log.info("win_plan ABORTED: matched '%s' in customer msg", kw)
                    return True, f"aborted: {cond}"

    # 2. Supervisor self-report — preferred path
    plan_signal = (supervisor_directive or {}).get("plan") or {}
    sup_phase_status = plan_signal.get("phase_status")
    sup_next = plan_signal.get("recommended_next_phase_id")
    sup_branch = plan_signal.get("active_branch")

    if sup_branch and not state.active_branch:
        state.active_branch = sup_branch
        log.info("win_plan: supervisor committed to branch '%s'", sup_branch)

    if sup_phase_status == "exit_met" or sup_next:
        target = _normalize_target_phase(sup_next, plan) if sup_next else None
        if target is None and sup_phase_status == "exit_met":
            # Default: linear next-in-path, respecting active_branch
            target = _next_phase_id_for_branch(plan, state)
        if target and target != state.current_phase_id:
            state.current_phase_id = target
            state.turns_in_current_phase = 0
            state.phase_history.append(target)
            return True, f"supervisor_advanced_to_phase_{target}"

    # 3. Phase 2 special routing — if budget exhausted without a branch decision,
    # default to branch B (counter-once) as the safest middle option
    if state.current_phase_id == "2" and state.turns_in_current_phase >= 1:
        if not state.active_branch:
            state.active_branch = "branch_B_counter_once_then_close"
            log.info("win_plan phase 2 budget — defaulting to branch_B")
        target = _next_phase_id_for_branch(plan, state)
        if target:
            state.current_phase_id = target
            state.turns_in_current_phase = 0
            state.phase_history.append(target)
            return True, f"phase_2_routed_to_{target}"

    # 4. Force-advance if phase budget blown (within active branch)
    constraints = plan.get("session_constraints") or {}
    max_turns = current.get("max_turns") or 2
    hard_max = constraints.get("max_turns_per_phase_before_force_advance",
                                max_turns + 1)
    if state.turns_in_current_phase >= hard_max:
        target = _next_phase_id_for_branch(plan, state)
        if target is None:
            state.completed = True
            return False, "terminal_phase_budget_exhausted"
        state.current_phase_id = target
        state.turns_in_current_phase = 0
        state.phase_history.append(target)
        return True, f"force_advanced_to_{target}_budget_exhausted"

    # 5. Stay
    return False, "stay_in_phase"


def _next_phase_id_for_branch(plan: dict, state: WinPlanState) -> str | None:
    """Find the linear-next phase_id within the active branch (if set), or
    in the trunk path if not yet branched."""
    path = plan.get("path") or []
    # Build ordered list of phase_ids that belong to current trunk OR active branch
    relevant = []
    for ph in path:
        ph_branch = ph.get("branch_id")
        if state.active_branch:
            # In a branch: include trunk (no branch_id) + matching branch
            # + shared phase_6_*
            pid = str(ph.get("phase_id"))
            if (ph_branch is None) or (ph_branch == state.active_branch) \
                    or pid.startswith("6_"):
                relevant.append(pid)
        else:
            # Pre-branch: only trunk phases (1, 2)
            if ph_branch is None:
                relevant.append(str(ph.get("phase_id")))

    cur = state.current_phase_id
    if cur not in relevant:
        return None
    idx = relevant.index(cur)
    if idx + 1 >= len(relevant):
        return None
    return relevant[idx + 1]


def record_strategy_adherence(state: WinPlanState,
                                supervisor_directive: dict | None
                                ) -> tuple[bool, str | None]:
    """Same idea as cluster_plan.record_strategy_adherence — track whether the
    supervisor's chosen strategy matches the current phase's preferred actions."""
    if not supervisor_directive:
        return False, None
    strat = (supervisor_directive.get("strategy") or {}).get("primary") \
        or supervisor_directive.get("primary_strategy")
    if not strat:
        return False, None
    current = state.current_phase()
    if current is None:
        return False, strat
    preferred = set(current.get("agent_actions_preferred") or [])
    in_plan = strat in preferred
    if in_plan:
        cur_count = state.probe_attempts_by_phase.get(state.current_phase_id, 0)
        state.probe_attempts_by_phase[state.current_phase_id] = cur_count + 1
    else:
        state.off_plan_strategy_count += 1
    return in_plan, strat


def render_win_plan_section(plan: dict, state: WinPlanState) -> str:
    """Build the supervisor prompt block describing current win-plan state."""
    if not plan or not state:
        return ""
    current = state.current_phase()
    if current is None:
        return ""

    goal = plan.get("goal", {})
    constraints = plan.get("session_constraints", {})
    path = plan.get("path") or []

    # Path overview — group by branch
    trunk = [ph for ph in path if ph.get("branch_id") is None
              and not str(ph.get("phase_id")).startswith("6_")]
    branch_a = [ph for ph in path if ph.get("branch_id") == "branch_A_bargain_and_stretch"]
    branch_b = [ph for ph in path if ph.get("branch_id") == "branch_B_counter_once_then_close"]
    branch_c = [ph for ph in path if ph.get("branch_id") == "branch_C_long_tail_followup"]
    shared = [ph for ph in path if str(ph.get("phase_id")).startswith("6_")]

    def fmt_phase(ph):
        pid = str(ph["phase_id"])
        marker = " ← CURRENT" if pid == state.current_phase_id else ""
        completed = "✓" if pid in state.phase_history[:-1] else "·"
        return f"  {completed} {pid}. {ph['name']} (max {ph.get('max_turns','?')} turns){marker}"

    overview_lines = ["TRUNK (always visited):"]
    overview_lines.extend(fmt_phase(p) for p in trunk)
    overview_lines.append("")
    overview_lines.append(f"BRANCH A — bargain-and-stretch (45.6% of wins):")
    overview_lines.extend(fmt_phase(p) for p in branch_a)
    overview_lines.append("")
    overview_lines.append(f"BRANCH B — counter-once-then-close (37.4% of wins):")
    overview_lines.extend(fmt_phase(p) for p in branch_b)
    overview_lines.append("")
    overview_lines.append(f"BRANCH C — long-tail follow-up (17.0% of wins):")
    overview_lines.extend(fmt_phase(p) for p in branch_c)
    overview_lines.append("")
    overview_lines.append("SHARED close (after any branch):")
    overview_lines.extend(fmt_phase(p) for p in shared)

    # Current phase detail
    cd = []
    cd.append(f"**Current phase: {current['phase_id']} — {current['name']}**")
    cd.append(f"  Active branch: {state.active_branch or '(not yet committed)'}")
    cd.append(f"  Turns in this phase: {state.turns_in_current_phase}/{current.get('max_turns','?')}")
    if current.get("agent_actions_preferred"):
        cd.append(f"  Preferred actions: {', '.join(current['agent_actions_preferred'])}")
    if current.get("exit_signal"):
        cd.append(f"  Exit signal: {current['exit_signal']}")
    if current.get("tonal_constraint"):
        cd.append(f"  Tonal constraint: {current['tonal_constraint']}")
    if current.get("rules_to_enforce"):
        cd.append(f"  Rules: {', '.join(current['rules_to_enforce'])}")
    if current.get("branches"):
        cd.append("  Branch routing options:")
        for b in current["branches"]:
            cd.append(f"    - if {b.get('if','?')} → {b.get('then','?')}")
    if current.get("_routing_logic"):
        cd.append("  Routing definitions:")
        for k, v in current["_routing_logic"].items():
            cd.append(f"    {k}: {v}")

    # Special phase 2 instruction
    phase_2_instruction = ""
    if str(state.current_phase_id) == "2":
        phase_2_instruction = (
            "\n### ⚠ PHASE 2 IS A ROUTING DECISION\n"
            "You MUST commit to one of branch_A_bargain_and_stretch / "
            "branch_B_counter_once_then_close / branch_C_long_tail_followup "
            "in this turn by setting `plan.active_branch` in your directive. "
            "Read the customer's reply carefully against the routing definitions.\n"
        )

    constraint_lines = [f"  - {k}: {v}" for k, v in constraints.items()]

    return f"""## Win-Mode Plan ({plan.get('win_plan_id')})
This conversation has positive trajectory signal — supervisor is following a
WIN-MODE plan compiled from {plan.get('n_winning_conversations_analyzed','?')} winning conversations
in this cell. Three winning patterns exist as conditional branches.
{phase_2_instruction}
### Path overview
{chr(10).join(overview_lines)}

### Current phase
{chr(10).join(cd)}

### Session goal
  Goal: {goal.get('primary','?')}
  Success: {goal.get('success_criteria','?')}
  Fallback: {goal.get('fallback','?')}

### Session constraints
{chr(10).join(constraint_lines) if constraint_lines else '  (none)'}

### Anti-patterns to avoid
{chr(10).join('  - ' + a for a in (plan.get('anti_patterns') or [])[:5])}
"""
