"""ClusterPlan — session-level goal + ordered path of phases per (cluster,
motivator, decision_logic).

Provides the missing layer above the turn-level supervisor: a stateful plan
that the supervisor advances through one phase at a time, instead of
re-deriving the next move from scratch on every turn.

Architecture: research-notes/2026-04-30-session-plan-global-supervisor.md.

Public API:
    load_plan(tenant, cluster_id, motivator, decision_logic) -> dict | None
    PlanState (dataclass) — per-session mutable state machine
    advance_plan(plan, plan_state, supervisor_directive, last_customer_msg) -> bool
    render_plan_section(plan, plan_state) -> str  # for supervisor prompt
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANS_DIR = os.environ.get(
    "POC_PLANS_DIR",
    os.path.join(_PROJECT_ROOT, "data", "cluster_plans"),
)


def _slug_motivator(s: str) -> str:
    return (s or "").replace("/", "_")


def _plan_filename(tenant: str, cluster_id: int, motivator: str,
                    decision_logic: str) -> str:
    """Convention: {Tenant}__c{N}__{Motivator}__{DecisionLogic}.json
    motivator/decision_logic '/' replaced with '_'."""
    return (
        f"{tenant}__c{cluster_id}__"
        f"{_slug_motivator(motivator)}__{_slug_motivator(decision_logic)}.json"
    )


def load_plan(tenant: str, cluster_id: int | None, motivator: str | None,
              decision_logic: str | None) -> dict | None:
    """Load the matching ClusterPlan or return None if no plan exists for the cell."""
    if cluster_id is None or not motivator or not decision_logic:
        return None
    fname = _plan_filename(tenant, cluster_id, motivator, decision_logic)
    path = os.path.join(PLANS_DIR, fname)
    if not os.path.isfile(path):
        log.info("No cluster_plan for %s — fallback to plan-less supervision", fname)
        return None
    try:
        with open(path) as f:
            plan = json.load(f)
        log.info("Loaded cluster_plan %s (%d phases)",
                  plan.get("cluster_plan_id"), len(plan.get("path") or []))
        return plan
    except Exception as e:
        log.warning("Failed to load cluster_plan %s: %s", fname, e)
        return None


@dataclass
class PlanState:
    """Per-session mutable state machine. Owned by the right-panel only."""
    plan: dict
    current_phase_id: int = 1
    turns_in_current_phase: int = 0
    phase_history: list[int] = field(default_factory=lambda: [1])
    completed: bool = False
    aborted: bool = False
    abort_reason: str | None = None
    # When the supervisor enters a branch within a phase, record the branch_id
    active_branch: str | None = None
    # Captured signals (e.g., 'customer_competitor_offer_amount_NIS': '5000')
    captured: dict[str, Any] = field(default_factory=dict)
    # Per-phase probe-attempt count — prevents supervisor from skipping the
    # phase's expected action. Tracked when supervisor's strategy ∈ phase's
    # agent_actions_preferred. Used by replayer's graceful_close guard.
    probe_attempts_by_phase: dict[int, int] = field(default_factory=dict)
    # Track adherence — was the supervisor's chosen strategy in the phase's
    # preferred list this turn? Cumulative count of off-plan turns.
    off_plan_strategy_count: int = 0

    def current_phase(self) -> dict | None:
        for ph in (self.plan.get("path") or []):
            if ph.get("phase_id") == self.current_phase_id:
                return ph
        return None

    def is_terminal_phase(self) -> bool:
        path = self.plan.get("path") or []
        if not path:
            return True
        return self.current_phase_id >= path[-1].get("phase_id", 0)


def advance_plan(plan_state: PlanState,
                  supervisor_directive: dict | None,
                  last_customer_msg: str,
                  panel_metrics: dict | None = None) -> tuple[bool, str | None]:
    """Decide whether to advance to the next phase based on supervisor's
    self-reported phase status + simple heuristics.

    Returns (advanced: bool, reason: str | None).

    Decision precedence:
      1. Abort condition met (customer signed elsewhere, declared no-need, etc.) → mark aborted
      2. Supervisor's directive includes plan.phase_status='exit_met' or
         plan.recommended_next_phase != current → advance per supervisor
      3. Phase budget exhausted (turns_in_current_phase >= max_turns + 1) → force advance
      4. Otherwise → stay
    """
    plan = plan_state.plan
    current = plan_state.current_phase()
    if current is None:
        return False, "no_current_phase"

    # 1. Abort conditions — keyword scan on customer text
    #
    # Bug fix 2026-05-01 (session 1faf5927): the previous keyword set was too
    # broad. "works fine" and "not looking" matched on Karl's bargaining
    # response ("...Odioone works perfectly fine. Why would I spend money...?")
    # which was a NEGOTIATION not an abandonment. The customer's message ENDED
    # with a question — they were still engaged. New rules:
    #   (a) abort only fires in early phases (id <= 2 — qualify or probe)
    #   (b) skip abort if customer message ends with a question (still engaging)
    #   (c) skip abort if message contains bargaining markers ("but", "why",
    #       "what", "how", "can you", "could you", "would you", "if")
    #   (d) hard-no phrases tightened to unambiguous declarations only
    abort_on = (plan.get("goal") or {}).get("abort_on") or []
    cust = (last_customer_msg or "").strip()
    cust_lower = cust.lower()
    abort_keywords = {
        # Hard-no signals only — phrases ambiguous in bargaining contexts removed
        "customer_already_signed_with_competitor": [
            "already signed", "went with another", "signed with another",
            "סגרתי עם אחר", "עברתי לחברה אחרת", "כבר חידשתי", "כבר שילמתי",
        ],
        "customer_confirms_already_signed_with_competitor": [
            "already signed", "went with another", "i did it with another",
        ],
        "customer_says_already_paid_competitor": [
            "already paid", "i paid the policy", "i paid for the policy", "כבר שילמתי",
        ],
        "customer_explicitly_requests_stop": [
            "please stop messaging", "stop messaging me", "leave me alone",
            "remove me from", "unsubscribe me", "do not contact me",
        ],
        "customer_states_no_need_for_replacement": [
            # Tightened: must be unambiguous "ever" / "never" type phrasing
            "i don't want a new", "i'll never need", "i'm never going to upgrade",
            "i'm not interested in replacing", "no need for a replacement ever",
        ],
        "customer_states_current_product_works_fine": [
            # Tightened: removed "works fine" / "satisfied" alone (too broad).
            # Require explicit replacement-context "no need" phrasing.
            "i don't need a replacement", "no replacement needed",
            "won't be replacing", "have no plans to replace",
        ],
    }

    # Guard (a): only consider abort in phases 1-2
    if plan_state.current_phase_id > 2:
        # Past qualifying — never abort on customer push-back. Customer at this
        # phase has demonstrated engagement and is bargaining.
        pass
    # Guard (b): customer message ends with a question → still engaged
    elif cust.rstrip("!.").endswith("?") or "?" in cust[-40:]:
        log.info("plan abort skipped: customer message contains question — still engaging")
    # Guard (c): bargaining/negotiation markers
    elif any(m in cust_lower for m in [
        " but ", " why ", "what would", "what if", "how would", "how could",
        "can you ", "could you ", "would you ", "what's the ", "what is the ",
        "does it ", "do you ", "if you ", " maybe ",
    ]):
        log.info("plan abort skipped: customer message has bargaining markers")
    else:
        # Conditions cleared — check the actual abort phrases
        for cond in abort_on:
            for kw in abort_keywords.get(cond, []):
                if kw in cust_lower:
                    plan_state.aborted = True
                    plan_state.abort_reason = cond
                    log.info("plan ABORTED: matched '%s' in customer msg", kw)
                    return True, f"aborted: {cond}"

    # 2. Supervisor self-report (preferred path — LLM has the most context)
    plan_signal = (supervisor_directive or {}).get("plan") or {}
    sup_phase_status = plan_signal.get("phase_status")
    sup_next = plan_signal.get("recommended_next_phase_id")
    if sup_phase_status == "exit_met" or (sup_next is not None
                                            and isinstance(sup_next, int)
                                            and sup_next != plan_state.current_phase_id):
        # Validate: don't skip more than 1 phase forward, don't go backwards
        target = sup_next if isinstance(sup_next, int) else plan_state.current_phase_id + 1
        if target <= plan_state.current_phase_id:
            return False, f"supervisor_recommended_no_progress(target={target})"
        if target > plan_state.current_phase_id + 1:
            log.info("supervisor wanted to skip from phase %d → %d; clamping to +1",
                      plan_state.current_phase_id, target)
            target = plan_state.current_phase_id + 1
        plan_state.current_phase_id = target
        plan_state.turns_in_current_phase = 0
        plan_state.phase_history.append(target)
        plan_state.active_branch = plan_signal.get("active_branch")
        return True, f"supervisor_advanced_to_phase_{target}"

    # 3. Force-advance if phase budget blown
    max_turns = current.get("max_turns") or 2
    constraints = plan.get("session_constraints") or {}
    hard_max = constraints.get("max_turns_per_phase_before_force_advance", max_turns + 1)
    if plan_state.turns_in_current_phase >= hard_max:
        new_phase = plan_state.current_phase_id + 1
        if plan_state.is_terminal_phase():
            plan_state.completed = True
            return False, "terminal_phase_budget_exhausted"
        plan_state.current_phase_id = new_phase
        plan_state.turns_in_current_phase = 0
        plan_state.phase_history.append(new_phase)
        return True, f"force_advanced_to_phase_{new_phase}_budget_exhausted"

    # 4. Stay in phase
    return False, "stay_in_phase"


def record_strategy_adherence(plan_state: PlanState, supervisor_directive: dict | None) -> tuple[bool, str | None]:
    """Compare supervisor's chosen strategy.primary against the current phase's
    agent_actions_preferred. Track whether it adhered. Returns (in_plan, strategy).

    Used by:
      - probe_attempts_by_phase: increment when supervisor stayed in plan
      - graceful_close guard: refuse close if probe_attempts in current phase < 1
      - off_plan_strategy_count: diagnostic counter
    """
    if not supervisor_directive:
        return False, None
    strat = (supervisor_directive.get("strategy") or {}).get("primary") \
        or supervisor_directive.get("primary_strategy")
    if not strat:
        return False, None
    current = plan_state.current_phase()
    if current is None:
        return False, strat
    preferred = set(current.get("agent_actions_preferred") or [])
    in_plan = strat in preferred
    if in_plan:
        cur_count = plan_state.probe_attempts_by_phase.get(plan_state.current_phase_id, 0)
        plan_state.probe_attempts_by_phase[plan_state.current_phase_id] = cur_count + 1
    else:
        plan_state.off_plan_strategy_count += 1
        log.info("plan adherence: phase %d expects %s, supervisor chose '%s' (off-plan count=%d)",
                  plan_state.current_phase_id, sorted(preferred),
                  strat, plan_state.off_plan_strategy_count)
    return in_plan, strat


def can_graceful_close(plan_state: PlanState | None) -> tuple[bool, str | None]:
    """Should we honor the agent's graceful-close emission?

    NO if:
      - plan is loaded
      - current phase has agent_actions_preferred (i.e. it's a probe/active phase)
      - probe_attempts_by_phase[current_phase_id] < 1
    YES otherwise (no plan, or we've probed at least once already).

    Returns (allowed, block_reason).
    """
    if plan_state is None:
        return True, None
    current = plan_state.current_phase()
    if current is None:
        return True, None
    # Final phases (close_or_warm_exit etc) explicitly intend warm exit
    if "close" in (current.get("name") or "").lower() or "exit" in (current.get("name") or "").lower():
        return True, None
    # If phase has no preferred actions, no quota
    preferred = current.get("agent_actions_preferred") or []
    if not preferred:
        return True, None
    # Block close if we haven't tried the phase yet
    attempts = plan_state.probe_attempts_by_phase.get(plan_state.current_phase_id, 0)
    if attempts < 1:
        return False, (
            f"phase {plan_state.current_phase_id} ({current['name']}) "
            f"requires at least 1 attempt at {preferred} before graceful close; "
            f"attempts={attempts}"
        )
    return True, None


def render_plan_section(plan: dict, plan_state: PlanState) -> str:
    """Build the prompt block describing current plan state for the supervisor."""
    if not plan or not plan_state:
        return ""
    current = plan_state.current_phase()
    if current is None:
        return ""

    goal = plan.get("goal", {})
    constraints = plan.get("session_constraints", {})
    path = plan.get("path") or []
    n_phases = len(path)

    # Build the path overview as a compact list with current marker
    phase_lines = []
    for ph in path:
        marker = " ← CURRENT" if ph["phase_id"] == plan_state.current_phase_id else ""
        completed = "✓" if ph["phase_id"] < plan_state.current_phase_id else "·"
        phase_lines.append(
            f"  {completed} {ph['phase_id']}. {ph['name']} (max {ph.get('max_turns','?')} turns){marker}"
        )

    # Current phase detail
    current_detail = []
    current_detail.append(f"**Current phase: {current['phase_id']} — {current['name']}**")
    current_detail.append(f"  Turns spent in this phase: {plan_state.turns_in_current_phase}/{current.get('max_turns','?')}")
    if current.get("agent_actions_preferred"):
        current_detail.append(f"  Preferred actions for this phase: {', '.join(current['agent_actions_preferred'])}")
    if current.get("exit_signal"):
        current_detail.append(f"  Exit signal (advance when this happens): {current['exit_signal']}")
    if current.get("tonal_constraint"):
        current_detail.append(f"  Tonal constraint: {current['tonal_constraint']}")
    if current.get("must_capture"):
        captured_so_far = [k for k in current["must_capture"] if k in plan_state.captured]
        missing = [k for k in current["must_capture"] if k not in plan_state.captured]
        current_detail.append(f"  Must capture before advancing: {', '.join(missing) if missing else '(all captured)'}")
    if current.get("rules_to_enforce"):
        current_detail.append(f"  Rules to enforce: {', '.join(current['rules_to_enforce'])}")
    if current.get("branches"):
        current_detail.append("  Branch options:")
        for b in current["branches"]:
            cond = b.get("if") or b.get("branch_id", "?")
            then = b.get("then") or b.get("agent_actions") or "?"
            current_detail.append(f"    - if {cond} → {then}")

    # Session-level context
    session_lines = []
    if goal.get("primary"):
        session_lines.append(f"Goal: {goal['primary']}")
    if goal.get("success_criteria"):
        session_lines.append(f"Success: {goal['success_criteria']}")
    if goal.get("fallback"):
        session_lines.append(f"Fallback: {goal['fallback']}")
    captured_str = ", ".join(f"{k}={v}" for k, v in (plan_state.captured or {}).items())
    if captured_str:
        session_lines.append(f"Already captured: {captured_str}")

    # Constraints
    constraint_lines = []
    for k, v in constraints.items():
        if k.startswith("max_") or v is True:
            constraint_lines.append(f"  - {k}: {v}")

    preferred_str = ", ".join(current.get("agent_actions_preferred") or [])
    return f"""## Cluster Plan ({plan.get('cluster_plan_id')})
This conversation has a phase-progressive plan you MUST follow:
- Each phase has a specific exit signal — advance only when met OR budget blown.
- Set `plan.phase_status` to one of: `in_progress`, `exit_met`, `should_advance`, `aborting`.
- Set `plan.recommended_next_phase_id` if you want to advance to the next phase.

### ⚠ HARD CONSTRAINT — strategy choice must match the current phase
**For phase {current['phase_id']} ({current['name']}), `strategy.primary` MUST be one of:
{{ {preferred_str} }}**

DO NOT choose `empathy` from a probe phase — that's the warm-exit move.
DO NOT choose `commitment` before the customer has shown engagement.
DO NOT graceful-close from a probe phase. The customer hasn't been probed yet.

If you genuinely cannot execute the phase's preferred action (e.g., customer
explicitly demands "stop messaging me"), set `plan.phase_status='aborting'`
with a specific reason in `plan.rationale`. Otherwise execute the phase as
specified. Honest probing customers who say "I'm not in the market" is part
of phase 2's job — they often reveal a quality criterion when asked directly.

### Path overview ({n_phases} phases total)
{chr(10).join(phase_lines)}

### Current phase
{chr(10).join(current_detail)}

### Session goals
{chr(10).join('  - ' + s for s in session_lines) if session_lines else '  (none)'}

### Session constraints (must respect across all turns)
{chr(10).join(constraint_lines) if constraint_lines else '  (none)'}
"""
