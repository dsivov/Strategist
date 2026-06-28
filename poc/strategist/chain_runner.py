"""POC sandbox replication of production Luna's prompt-chain runner.

T-85 (Phase A of the playbook-compilation integration proposal,
2026-05-02-PROPOSAL-supervisor-via-playbook-compilation.md).

DOES NOT TOUCH PRODUCTION. This is a standalone re-implementation that
reads production's prompt-chain definitions (ai_agent + ai_agent_prompt +
ai_prompt + ai_prompt_section tables — read-only queries) and executes
them locally inside our POC server, splicing in our supervisor stages
at agreed positions.

Scope (Phase A):
- Read prod chain definition for a (company, opp_type) tuple
- Execute prompt-style stages (LLM calls per llm_provider)
- Execute programmatic stages (Python callables registered locally)
- Support our combined signal_analysis + retry stage (per approved
  design — single combined LLM call, AUGMENTS prompt_manager)
- Splice anti-staircase gate, retreat passthrough, anchor load,
  decision-trace emit at appropriate slots
- Parallel-batch execution by (chain_type, execution_order) — Phase A.3.1.
  Stages with the same (chain_type, execution_order) are run via
  asyncio.gather; programmatic stages within a batch run sequentially
  after the LLM batch (because supervisor splice stages depend on
  outputs of LLM stages at the same order — e.g. anti_staircase_gate
  reads prompt_build_answer; retreat_passthrough_gate reads
  prompt_signal_analysis_combined).
- Regenerate-loop semantics for anti-staircase gate (re-run
  prompt_build_answer with system_suffix on staircase detection)

Out of scope (Phase A):
- Modifying production runtime
- Per-tenant config UI / authoring tools
- LLM-judge fallback for unconfigured tenants
- Production observability (Loki structured logs, OTel spans) — POC
  uses standard Python logging
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import anthropic
from google import genai
from google.genai import types as genai_types

from db import open_conn

log = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ChainStage:
    """Mirrors a row from luna.ai_agent_prompt joined with luna.ai_prompt."""
    chain_type: str         # 'preprocessing' | 'processing' | 'postprocessing'
    execution_order: int
    prompt_name: str
    prompt_id: int | None   # T-86 fix: tenant-specific prompt_id from join.
                              # Multiple tenants have rows with the same
                              # prompt_name in luna.ai_prompt, so resolving
                              # by name picks the wrong one. The agent_prompt
                              # join carries the correct prompt_id.
    llm_provider: str       # 'gemini' | 'gemini_light' | 'gemini-flash' | etc.
    parallel_group: int | None
    exit_fn: str | None
    transform_fn: str | None
    response_format: str | None
    prompt_role: str | None
    flash_on_stages: list | None

    # Programmatic-stage extension (T-85)
    is_programmatic: bool = False
    python_callable: Callable | None = None


@dataclass
class ChainContext:
    """Passed through the chain. Each stage reads/writes here."""
    opp_id: str
    opp_meta: dict
    dialog: list[dict] = field(default_factory=list)
    previous_responses: dict[str, Any] = field(default_factory=dict)
    business_rules: str = ""
    early_exit: bool = False
    early_exit_reason: str | None = None

    # Supervisor-state we add (not in production schema)
    agent_concessions: list = field(default_factory=list)
    voice_profile: dict | None = None
    anchors: dict | None = None

    # Q14 (2026-05-04) — measurement parity with the legacy replayer path.
    # When the chain runner orchestrates supervisor stages, these flow into
    # mode1b_directive so cluster_plan + session-memory (strategies_used) are
    # available to the supervisor and its retry checks. Without this the chain
    # path was running supervisor "MINUS cluster_plan MINUS strategies_used"
    # vs the replayer path that had both — confounding Phase A.3 lift numbers.
    plan_state: Any | None = None
    strategies_used: list[str] | None = None
    pre_rendered_plan_section: str | None = None
    preferred_actions: list[str] | None = None
    # R7 — per-session concrete-move usage histogram for variation pressure
    moves_used: dict | None = None
    # M1 (Option C, 2026-05-04) — counter for consecutive Mode 1a cache hits
    # this session. Hard cap = 2 (legacy `consecutive_mode1a` pattern; dfb34792
    # showed 4× same-tier turns destroyed customer engagement). Read in
    # _try_cache_lookup; if >= 2, lookup is skipped → fall to Mode 1b.
    # Mutated by chain stage; replayer writes back to PanelState after turn.
    consecutive_mode1a: int = 0
    # 2026-05-05 — Late-phase low-score retrieval. Replayer writes the
    # previous customer turn's persuasion + commitment so the chain stage
    # can detect "stuck in close_attempt with low engagement" and pull
    # recovery patterns from historical won-conversations.
    prev_persuasion: float | None = None
    prev_commitment: int | None = None


# ── Chain definition loader ─────────────────────────────────────────────────

def fetch_chain_definition(company: str, opp_type: str,
                            is_followup: bool = False,
                            is_template_rephrase: bool = False) -> list[ChainStage]:
    """Read the production chain for (company, opp_type).
    Returns ordered list of ChainStage objects."""
    conn = open_conn()
    stages: list[ChainStage] = []
    try:
        with conn.cursor() as cur:
            # Find the matching agent
            cur.execute("""
                SELECT a.agent_id, a.agent_name
                FROM luna.ai_agent a
                WHERE a.is_enabled = 1
                  AND a.is_followup = %s
                  AND a.is_template_rephrase = %s
                  AND (a.agent_name LIKE %s OR a.agent_name LIKE %s)
                ORDER BY a.agent_id DESC
                LIMIT 1
            """, (1 if is_followup else 0,
                  1 if is_template_rephrase else 0,
                  f"%{company}%CarRenewal%V3%" if "Renewal" in (opp_type or "") else f"%{company}%",
                  f"%{company}%"))
            agent_row = cur.fetchone()
            if not agent_row:
                log.warning("chain_runner: no agent matched (company=%s, opp_type=%s)",
                              company, opp_type)
                return []
            agent_id = agent_row["agent_id"]
            agent_name = agent_row["agent_name"]
            log.info("chain_runner: matched agent %s (id=%s) for company=%s",
                     agent_name, agent_id, company)

            # Pull chain stages
            cur.execute("""
                SELECT aap.chain_type, aap.execution_order, aap.parallel_group,
                       aap.exit_fn, aap.transform_fn, aap.response_format,
                       aap.prompt_role, aap.flash_on_stages,
                       aap.prompt_id, p.prompt_name, p.llm_provider
                FROM luna.ai_agent_prompt aap
                JOIN luna.ai_prompt p ON aap.prompt_id = p.prompt_id
                WHERE aap.agent_id = %s
                  AND aap.execution_excluded = 0
                ORDER BY
                  FIELD(aap.chain_type, 'preprocessing', 'processing', 'postprocessing'),
                  aap.execution_order,
                  aap.parallel_group
            """, (agent_id,))
            for row in cur.fetchall():
                stages.append(ChainStage(
                    chain_type=row["chain_type"],
                    execution_order=row["execution_order"],
                    prompt_name=row["prompt_name"],
                    prompt_id=row["prompt_id"],
                    llm_provider=row["llm_provider"],
                    parallel_group=row["parallel_group"],
                    exit_fn=row["exit_fn"],
                    transform_fn=row["transform_fn"],
                    response_format=row["response_format"],
                    prompt_role=row["prompt_role"],
                    flash_on_stages=row.get("flash_on_stages"),
                ))
    finally:
        conn.close()
    log.info("chain_runner: loaded %d stages", len(stages))
    return stages


# ── Programmatic stage registry ─────────────────────────────────────────────

_PROGRAMMATIC_STAGES: dict[str, Callable] = {}


def register_programmatic_stage(name: str):
    """Decorator to register a Python callable as a programmatic chain stage.
    The callable signature: async (ctx: ChainContext) -> dict-to-merge-into-previous_responses."""
    def deco(fn):
        _PROGRAMMATIC_STAGES[name] = fn
        return fn
    return deco


def splice_supervisor_stages(stages: list[ChainStage]) -> list[ChainStage]:
    """T-85: insert our supervisor stages into the chain at agreed positions,
    with fallback positioning for agents that don't have the preferred anchor
    stages. Different production agents have different stage compositions:
    - LibraCarRenewalAssistantV3 has prompt_price_tracker, prompt_validate_gate
    - HeavysDemoAssistant has neither (simpler 6-stage chain)
    - HoneybookAssistant variants have yet another composition

    Splice rules (in order; first match wins per supervisor stage):
      anchor_load: after prompt_price_tracker, else end of preprocessing
      signal_analysis_combined: after anchor_load, else right before prompt_manager
      retreat_passthrough_gate: after signal_analysis_combined
      anti_staircase_gate: after prompt_validate_gate, else after prompt_build_answer
      decision_trace_emit: last postprocessing stage
    """
    out: list[ChainStage] = []
    inserted: set[str] = set()

    def _sup(name: str, chain_type: str, order: int) -> ChainStage:
        return ChainStage(
            chain_type=chain_type, execution_order=order, prompt_name=name,
            prompt_id=None,
            llm_provider="programmatic", parallel_group=None,
            exit_fn=None, transform_fn=None, response_format=None,
            prompt_role="programmatic", flash_on_stages=None,
            is_programmatic=True,
            python_callable=_PROGRAMMATIC_STAGES.get(name),
        )

    # Figure out anchor positions ahead of time so fallbacks know what's missing
    has_price_tracker = any(s.prompt_name == "prompt_price_tracker" for s in stages)
    has_validate_gate = any(s.prompt_name == "prompt_validate_gate" for s in stages)
    last_preproc = max((i for i, s in enumerate(stages)
                          if s.chain_type == "preprocessing"), default=-1)
    manager_idx = next((i for i, s in enumerate(stages)
                          if s.prompt_name == "prompt_manager"), -1)
    build_answer_idx = next((i for i, s in enumerate(stages)
                                if s.prompt_name == "prompt_build_answer"), -1)
    last_postproc = max((i for i, s in enumerate(stages)
                           if s.chain_type == "postprocessing"), default=-1)

    # Decide anchor index for each supervisor stage
    # anchor_load: after price_tracker if present, else at the very end of
    # preprocessing (so prompt_manager sees it)
    if has_price_tracker:
        anchor_load_after = next(i for i, s in enumerate(stages)
                                    if s.prompt_name == "prompt_price_tracker")
    else:
        anchor_load_after = last_preproc

    # 2026-05-10 — Live profile classifier: runs EARLIEST, before anchor_load
    # and signal_analysis, so any profile updates flow into the supervisor's
    # directive generation downstream. Same insertion site as anchor_load.
    live_profile_classifier_after = anchor_load_after

    # signal_analysis_combined: right after anchor_load (so it follows in same
    # preprocessing block), so we use the same target index
    sa_combined_after = anchor_load_after

    # 2026-05-10 — Profile-aware directive validator. Runs AFTER
    # signal_analysis_combined produces the directive. Same insertion site as
    # move_validity_gate (which also runs post-directive, pre-build_answer).
    directive_profile_gate_after = anchor_load_after

    # retreat_passthrough_gate: same insertion point as signal_analysis (will
    # be appended right after it)
    retreat_after = anchor_load_after

    # move_validity_gate (Phase 2 of strategy-enum-extension, architect DP3):
    # runs AFTER signal_analysis_combined produces the directive AND AFTER
    # retreat_passthrough_gate may have suppressed it, BEFORE build_answer
    # consumes the directive. Splice at the same anchor index — programmatic
    # batch ordering preserves: anchor_load → signal_analysis → retreat → move_validity
    move_validity_after = anchor_load_after

    # anti_staircase_gate: after validate_gate if present, else after build_answer
    if has_validate_gate:
        anti_staircase_after = next(i for i, s in enumerate(stages)
                                       if s.prompt_name == "prompt_validate_gate")
    elif build_answer_idx >= 0:
        anti_staircase_after = build_answer_idx
    else:
        anti_staircase_after = -1  # no insertion possible

    # anti_capitulation_gate (T-87): same insertion point as anti_staircase
    # so it sits right after build_answer's text — fires AFTER staircase so
    # staircase regenerate (if triggered) gets a chance first
    anti_capitulation_after = anti_staircase_after

    # premature_close_gate (2026-05-14): same insertion site. Runs AFTER
    # anti_capitulation so capitulation regenerate (if triggered) gets first
    # crack — premature_close is the broader budget-driven surrender pattern.
    premature_close_after = anti_staircase_after

    # R5 invariants gate: same insertion point — runs LAST among the
    # post-build gates so staircase + capitulation + premature_close get
    # first crack at retry.
    invariants_after = anti_staircase_after

    # decision_trace_emit: last postprocessing stage
    decision_emit_after = last_postproc

    # 2026-05-11 — Escalation router runs RIGHT AFTER prompt_co_pilot_escalation
    # so it can consume that stage's `is_escalation_needed` output before
    # decision_trace_emit and other downstream readers.
    escalation_router_after = next(
        (i for i, s in enumerate(stages) if s.prompt_name == "prompt_co_pilot_escalation"),
        last_postproc,
    )

    # Now build the output list with insertions tracked
    for i, s in enumerate(stages):
        out.append(s)
        # Multiple stages can hit the same anchor index — order matters within
        # an insertion site
        # 2026-05-10 — Live profile classifier FIRST so any profile updates
        # propagate through opp_meta to subsequent stages
        if (i == live_profile_classifier_after
                and "prompt_live_profile_classifier" not in inserted):
            out.append(_sup("prompt_live_profile_classifier", s.chain_type, s.execution_order))
            inserted.add("prompt_live_profile_classifier")
        if i == anchor_load_after and "prompt_anchor_load" not in inserted:
            out.append(_sup("prompt_anchor_load", s.chain_type, s.execution_order))
            inserted.add("prompt_anchor_load")
        # 2026-05-11 — Exemplar retrieval (option B / RAG-of-imitation)
        # Runs in preprocessing alongside anchor_load. Stage internally
        # short-circuits when POC_EXEMPLAR_RAG != "1" so it's safe to splice
        # by default; only active when the env flag is set.
        if i == anchor_load_after and "prompt_exemplar_retrieval" not in inserted:
            out.append(_sup("prompt_exemplar_retrieval", s.chain_type, s.execution_order))
            inserted.add("prompt_exemplar_retrieval")
        if i == sa_combined_after and "prompt_signal_analysis_combined" not in inserted:
            out.append(_sup("prompt_signal_analysis_combined", s.chain_type, s.execution_order))
            inserted.add("prompt_signal_analysis_combined")
        # 2026-05-10 — Directive profile gate runs RIGHT AFTER signal_analysis
        # so violations can trigger regenerate before downstream stages consume
        # the bad directive
        if (i == directive_profile_gate_after
                and "prompt_directive_profile_gate" not in inserted):
            out.append(_sup("prompt_directive_profile_gate", s.chain_type, s.execution_order))
            inserted.add("prompt_directive_profile_gate")
        # 2026-05-13 — Directive loop-breaker. Runs immediately after the
        # profile gate so any directive that passed validation is then
        # checked for "customer is repeating the same question" — if so,
        # corrective rules are injected before downstream stages consume
        # the directive (and before prompt_build_answer composes the prompt).
        if (i == directive_profile_gate_after
                and "prompt_directive_loop_breaker" not in inserted):
            out.append(_sup("prompt_directive_loop_breaker", s.chain_type, s.execution_order))
            inserted.add("prompt_directive_loop_breaker")
        if i == retreat_after and "prompt_retreat_passthrough_gate" not in inserted:
            out.append(_sup("prompt_retreat_passthrough_gate", s.chain_type, s.execution_order))
            inserted.add("prompt_retreat_passthrough_gate")
        if i == move_validity_after and "prompt_move_validity_gate" not in inserted:
            out.append(_sup("prompt_move_validity_gate", s.chain_type, s.execution_order))
            inserted.add("prompt_move_validity_gate")
        if i == anti_staircase_after and "prompt_anti_staircase_gate" not in inserted:
            out.append(_sup("prompt_anti_staircase_gate", s.chain_type, s.execution_order))
            inserted.add("prompt_anti_staircase_gate")
        if i == anti_capitulation_after and "prompt_anti_capitulation_gate" not in inserted:
            out.append(_sup("prompt_anti_capitulation_gate", s.chain_type, s.execution_order))
            inserted.add("prompt_anti_capitulation_gate")
        if i == premature_close_after and "prompt_premature_close_gate" not in inserted:
            out.append(_sup("prompt_premature_close_gate", s.chain_type, s.execution_order))
            inserted.add("prompt_premature_close_gate")
        if i == invariants_after and "prompt_invariants_gate" not in inserted:
            out.append(_sup("prompt_invariants_gate", s.chain_type, s.execution_order))
            inserted.add("prompt_invariants_gate")
        if i == decision_emit_after and "prompt_decision_trace_emit" not in inserted:
            out.append(_sup("prompt_decision_trace_emit", s.chain_type, s.execution_order))
            inserted.add("prompt_decision_trace_emit")
        # 2026-05-11 — Escalation router after co_pilot_escalation (consumer of
        # prod's security-vetted escalation rule). MUST run before any panel-end
        # logic so early_exit can be respected.
        if i == escalation_router_after and "prompt_escalation_router" not in inserted:
            out.append(_sup("prompt_escalation_router", s.chain_type, s.execution_order))
            inserted.add("prompt_escalation_router")

    # Diagnostic for whatever didn't get inserted (e.g., chain has no
    # postprocessing at all)
    expected = {"prompt_anchor_load", "prompt_signal_analysis_combined",
                "prompt_retreat_passthrough_gate", "prompt_move_validity_gate",
                "prompt_anti_staircase_gate", "prompt_anti_capitulation_gate",
                "prompt_premature_close_gate",
                "prompt_invariants_gate", "prompt_decision_trace_emit",
                "prompt_live_profile_classifier", "prompt_directive_profile_gate",
                "prompt_directive_loop_breaker",
                "prompt_exemplar_retrieval", "prompt_escalation_router"}
    missing = expected - inserted
    if missing:
        log.warning("chain_runner: could not splice these supervisor stages "
                       "(no compatible anchor in chain): %s", sorted(missing))

    log.info("chain_runner: spliced supervisor stages: %s", sorted(inserted))
    return out


# ── Chain runner ────────────────────────────────────────────────────────────

# Stages whose output the regenerate-loop targets when anti-staircase fires.
# In production the anti-staircase gate sits between prompt_validate_gate and
# prompt_dedup, so it operates on prompt_build_answer's output. When it fires,
# we re-run prompt_build_answer with a correction system_suffix.
_REGENERATE_LOOP_TARGET = "prompt_build_answer"
_MAX_STAIRCASE_RETRIES = 2


# Programmatic stages safe to run CONCURRENTLY with LLM stages at the same
# execution_order. These stages don't read any same-order LLM stage output
# AND don't write outputs that same-order LLM stages need to read. Adding a
# stage here moves it from "sequential after LLM batch" → "in the LLM batch's
# concurrency group." Saves ~10-25s per turn for slow programmatic stages
# (especially signal_analysis_combined which calls Mode 1b internally).
#
# DO NOT add stages that depend on LLM outputs at the same order:
#   - prompt_anti_staircase_gate    (reads prompt_build_answer)
#   - prompt_anti_capitulation_gate (reads prompt_build_answer)
#   - prompt_retreat_passthrough_gate (reads prompt_signal_analysis_combined)
_PARALLEL_SAFE_PROGRAMMATIC: set[str] = {
    "prompt_anchor_load",                # reads opp_meta only
    "prompt_signal_analysis_combined",   # internally calls Mode 1b; no chain-stage deps
}


def _batch_stages(stages: list[ChainStage]
                  ) -> list[tuple[str, list[ChainStage]]]:
    """Group consecutive stages by (chain_type, execution_order), then within
    each group split into:
      - 'llm' batch (LLM stages, may run concurrently via asyncio.gather)
      - 'prog' batch (programmatic stages, run sequentially after the LLM
        batch because supervisor splice stages depend on LLM outputs at the
        same order — e.g. anti_staircase_gate reads prompt_build_answer;
        retreat_passthrough_gate reads prompt_signal_analysis_combined)

    Returns a list of (kind, [stages]) tuples in execution order.
    """
    if not stages:
        return []
    # 1. Group by (chain_type, execution_order) preserving original order
    groups: list[list[ChainStage]] = []
    cur: list[ChainStage] = []
    cur_key: tuple[str, int] | None = None
    for s in stages:
        key = (s.chain_type, s.execution_order)
        if cur_key is None or key == cur_key:
            cur.append(s)
            cur_key = key
        else:
            groups.append(cur)
            cur = [s]
            cur_key = key
    if cur:
        groups.append(cur)
    # 2. Within each group, classify each stage:
    #    - LLM stages → parallel concurrency batch
    #    - Programmatic stages in _PARALLEL_SAFE_PROGRAMMATIC → join the
    #      concurrency batch (run via asyncio.gather alongside LLM)
    #    - Other programmatic → sequential batch AFTER the parallel batch
    out: list[tuple[str, list[ChainStage]]] = []
    for group in groups:
        parallel_batch = [
            s for s in group
            if not s.is_programmatic
            or s.prompt_name in _PARALLEL_SAFE_PROGRAMMATIC
        ]
        sequential_prog = [
            s for s in group
            if s.is_programmatic
            and s.prompt_name not in _PARALLEL_SAFE_PROGRAMMATIC
        ]
        if parallel_batch:
            out.append(("llm", parallel_batch))
        if sequential_prog:
            out.append(("prog", sequential_prog))
    return out


async def _execute_one_stage(s: ChainStage, ctx: ChainContext,
                              execute_prompt_stage_fn) -> tuple[ChainStage, Any, Exception | None]:
    """Run one stage and return (stage, result, exception_or_None).
    Never raises — caller can asyncio.gather without return_exceptions."""
    try:
        if s.is_programmatic:
            if s.python_callable is None:
                return (s, None, RuntimeError(f"programmatic stage {s.prompt_name} not registered"))
            result = await s.python_callable(ctx)
            return (s, result, None)
        else:
            response = await execute_prompt_stage_fn(s, ctx)
            return (s, response, None)
    except Exception as e:
        return (s, None, e)


async def run_chain(stages: list[ChainStage], ctx: ChainContext) -> ChainContext:
    """Execute the chain with parallel-batch semantics. Returns the updated context.

    Phase A.3.1 semantics:
    - Stages with the same (chain_type, execution_order) are batched.
    - LLM stages in a batch run concurrently via asyncio.gather (per-stage
      exception capture — one stage's 503/timeout doesn't nuke the others).
    - Programmatic stages in a batch run sequentially AFTER the LLM batch
      at the same order, since they read LLM outputs (anti_staircase reads
      build_answer; retreat_passthrough reads signal_analysis_combined).
    - Stops on early_exit between batches.
    - Anti-staircase regenerate-loop fires after the batch containing
      prompt_anti_staircase_gate; up to _MAX_STAIRCASE_RETRIES; falls back
      to the original draft on exhaustion.
    """
    # Lazy-import to avoid circular dependency (chain_executor imports
    # ChainStage / ChainContext from this module)
    from chain_executor import execute_prompt_stage, dispatch_llm, build_context_blocks, load_prompt_text
    from staircase_gate import build_correction_prompt
    from capitulation_gate import build_correction_prompt as build_capitulation_correction
    from premature_close_gate import build_correction_prompt as build_premature_close_correction
    from invariant_gates import build_invariants_correction

    batches = _batch_stages(stages)
    total_parallel_savings = sum(len(b) - 1 for kind, b in batches if kind == "llm")
    log.info("chain_runner: %d stages → %d batches (LLM parallelism saves ~%d sequential calls)",
             len(stages), len(batches), total_parallel_savings)

    for batch_idx, (kind, batch) in enumerate(batches):
        if ctx.early_exit:
            log.info("chain_runner: early-exit set (reason=%s); skipping %d remaining batches",
                     ctx.early_exit_reason, len(batches) - batch_idx)
            break

        if kind == "llm" and len(batch) > 1:
            # Parallel LLM batch
            t0 = time.time()
            log.info("chain_runner: parallel batch order=%d (%s) → %d LLM stages: %s",
                     batch[0].execution_order, batch[0].chain_type, len(batch),
                     [s.prompt_name for s in batch])
            tasks = [_execute_one_stage(s, ctx, execute_prompt_stage) for s in batch]
            results = await asyncio.gather(*tasks)
            for s, result, err in results:
                if err is not None:
                    log.warning("chain_runner: parallel stage %s failed: %s",
                                s.prompt_name, err)
                    ctx.previous_responses[s.prompt_name] = {"_error": str(err)}
                else:
                    ctx.previous_responses[s.prompt_name] = result
            log.info("chain_runner: parallel batch order=%d done in %dms",
                     batch[0].execution_order, int((time.time() - t0) * 1000))
            continue

        # Sequential path: single-LLM batch OR a programmatic batch
        for s in batch:
            if ctx.early_exit:
                log.info("chain_runner: early-exit set; skipping %s", s.prompt_name)
                break
            log.info("chain_runner: stage %s (%s) order=%d provider=%s",
                     s.prompt_name, s.chain_type, s.execution_order, s.llm_provider)
            t0 = time.time()
            stage, result, err = await _execute_one_stage(s, ctx, execute_prompt_stage)
            if err is not None:
                log.warning("chain_runner: stage %s failed: %s", s.prompt_name, err)
                ctx.previous_responses[s.prompt_name] = {"_error": str(err)}
                continue
            ctx.previous_responses[s.prompt_name] = result
            # Anti-staircase regenerate-loop hook (runs after the gate stage,
            # which is always programmatic and thus on the sequential path)
            if (s.is_programmatic and s.prompt_name == "prompt_anti_staircase_gate"
                    and isinstance(result, dict) and result.get("retry_recommended")):
                await _handle_staircase_retry(
                    stages, ctx, result,
                    execute_prompt_stage, build_correction_prompt,
                )
            # T-87 anti-capitulation regenerate-loop hook
            if (s.is_programmatic and s.prompt_name == "prompt_anti_capitulation_gate"
                    and isinstance(result, dict) and result.get("retry_recommended")):
                await _handle_capitulation_retry(
                    stages, ctx, result,
                    execute_prompt_stage, build_capitulation_correction,
                )
            # 2026-05-14 — Premature-close regenerate-loop hook
            if (s.is_programmatic and s.prompt_name == "prompt_premature_close_gate"
                    and isinstance(result, dict) and result.get("retry_recommended")):
                await _handle_premature_close_retry(
                    stages, ctx, result,
                    execute_prompt_stage, build_premature_close_correction,
                )
            # R5 invariants regenerate-loop hook
            if (s.is_programmatic and s.prompt_name == "prompt_invariants_gate"
                    and isinstance(result, dict) and result.get("retry_recommended")):
                await _handle_invariants_retry(
                    stages, ctx, result,
                    execute_prompt_stage, build_invariants_correction,
                )
            # 2026-05-10 — Profile validator regenerate-loop hook.
            # When the directive_profile_gate flags violations, re-run
            # prompt_signal_analysis_combined with the corrective_suffix as
            # additional context to mode1b_directive (Sonnet sees the
            # violations and produces a clean re-roll).
            if (s.is_programmatic and s.prompt_name == "prompt_directive_profile_gate"
                    and isinstance(result, dict) and result.get("retry_recommended")):
                await _handle_directive_profile_retry(stages, ctx, result, execute_prompt_stage)

    return ctx


async def _handle_staircase_retry(
    stages: list[ChainStage],
    ctx: ChainContext,
    gate_result: dict,
    execute_prompt_stage_fn,
    build_correction_prompt_fn,
) -> None:
    """T-83 regenerate-loop semantics in the chain runner.
    When anti-staircase gate detects a violation, re-run the build_answer
    stage with a corrective system_suffix. Up to _MAX_STAIRCASE_RETRIES;
    falls back to original draft on exhaustion.

    Mutates ctx.previous_responses[_REGENERATE_LOOP_TARGET] in place:
    - If retry succeeds (gate clears) → replace with corrected output
    - If retry exhausts → keep original draft (most permissive failure mode)
    """
    target_stage = next(
        (s for s in stages if s.prompt_name == _REGENERATE_LOOP_TARGET
         and not s.is_programmatic),
        None,
    )
    if target_stage is None:
        log.warning("chain_runner: regenerate-loop target %s not in chain; "
                       "anti-staircase verdict ignored", _REGENERATE_LOOP_TARGET)
        return

    original_output = ctx.previous_responses.get(_REGENERATE_LOOP_TARGET)
    correction = build_correction_prompt_fn(gate_result)

    # Attempt regenerate up to _MAX_STAIRCASE_RETRIES times
    from staircase_gate import check_staircase  # avoid top-level circular
    tenant = (ctx.opp_meta.get("company") or "").strip()
    for attempt in range(1, _MAX_STAIRCASE_RETRIES + 1):
        log.info("chain_runner: staircase retry attempt %d/%d",
                 attempt, _MAX_STAIRCASE_RETRIES)
        # Re-run build_answer with system_suffix injected via context (the
        # executor's build_context_blocks doesn't currently inject suffix;
        # quick path: extend ctx.previous_responses with a marker the
        # executor can read OR just append correction to user_text)
        ctx.previous_responses["_staircase_correction"] = correction
        try:
            new_output = await execute_prompt_stage_fn(target_stage, ctx)
        except Exception as e:
            log.warning("chain_runner: staircase regenerate attempt %d failed: %s",
                          attempt, e)
            ctx.previous_responses[_REGENERATE_LOOP_TARGET] = original_output
            ctx.previous_responses.pop("_staircase_correction", None)
            return

        # Check the new candidate
        if isinstance(new_output, str):
            is_sc, meta = check_staircase(
                panel_concessions=ctx.agent_concessions,
                new_text=new_output,
                tenant=tenant,
                dialog=ctx.dialog,
            )
            if not is_sc:
                # Success — replace the build_answer output, record concessions
                ctx.previous_responses[_REGENERATE_LOOP_TARGET] = new_output
                for c in (meta.get("new_concessions") or []):
                    ctx.agent_concessions.append({**c, "turn": len(ctx.dialog) + 1})
                ctx.previous_responses["prompt_anti_staircase_gate"] = {
                    "staircase_detected": True,
                    "verdict": "blocked_then_regenerated",
                    "retry_attempt": attempt,
                    "reason": gate_result.get("reason"),
                    "outcome": "regenerated_successfully",
                }
                ctx.previous_responses.pop("_staircase_correction", None)
                log.info("chain_runner: staircase regenerate succeeded on attempt %d",
                         attempt)
                return

        log.info("chain_runner: staircase regenerate attempt %d still violates", attempt)

    # All retries exhausted. Decision branches on criticality:
    # - is_critical (cumulative drop >= 25%): replace with safe fallback. The
    #   violating draft would create the inflated-baseline pattern that
    #   customers (and the simulator) perceive as overt manipulation, so
    #   passing it through is worse than emitting a safe holding message.
    # - non-critical: keep original draft (existing defensive behavior — at
    #   <25% the customer-visibility of the staircase is bounded, and a
    #   fallback message would be more disruptive than the violation).
    is_critical = bool(gate_result.get("is_critical"))
    if is_critical:
        # 2026-05-10 — Safe-fallback for critical staircase exhaustion. Picks
        # tenant + language from data/anti_staircase/<Tenant>.yaml via the
        # staircase_config helper, so adding a new tenant or changing the
        # message is a YAML edit (no code change).
        #
        # Pool-based + non-repeating: the YAML now defines a list of
        # question-shaped recovery messages. We track which have been used
        # in `ctx.opp_meta["_used_safe_fallbacks"]` and pick an unused one
        # each time, so the customer never sees the same fallback twice.
        # Question-shaped because real human agents in stuck negotiations
        # ask probing questions, not "let me check with manager."
        from staircase_config import get_safe_fallback_message
        last_customer_text = ""
        for m in reversed(ctx.dialog or []):
            if m.get("role") == "customer" and m.get("text"):
                last_customer_text = m["text"]
                break
        tenant = (ctx.opp_meta.get("company") or "").strip()
        used_fallbacks = list(ctx.opp_meta.get("_used_safe_fallbacks") or [])
        safe_fallback = get_safe_fallback_message(
            tenant, last_customer_text, already_used=used_fallbacks
        )
        # Track for future fires this session
        ctx.opp_meta["_used_safe_fallbacks"] = used_fallbacks + [safe_fallback]
        ctx.previous_responses[_REGENERATE_LOOP_TARGET] = safe_fallback
        ctx.previous_responses["prompt_anti_staircase_gate"] = {
            "staircase_detected": True,
            "verdict": "blocked_critical_safe_fallback",
            "retry_attempt": _MAX_STAIRCASE_RETRIES,
            "reason": gate_result.get("reason"),
            "outcome": "safe_fallback_substituted",
            "observed_drop_pct": gate_result.get("observed_drop_pct"),
            "is_critical": True,
        }
        ctx.previous_responses.pop("_staircase_correction", None)
        log.warning(
            "chain_runner: CRITICAL staircase retries exhausted (drop=%s%%) — "
            "substituting safe fallback message instead of passing violation through",
            gate_result.get("observed_drop_pct"))
        return

    # Non-critical exhaustion: keep original draft (defensive fallback).
    ctx.previous_responses[_REGENERATE_LOOP_TARGET] = original_output
    ctx.previous_responses["prompt_anti_staircase_gate"] = {
        "staircase_detected": True,
        "verdict": "blocked_fallback_to_original",
        "retry_attempt": _MAX_STAIRCASE_RETRIES,
        "reason": gate_result.get("reason"),
        "outcome": "fallback_to_original",
        "observed_drop_pct": gate_result.get("observed_drop_pct"),
        "is_critical": False,
    }
    ctx.previous_responses.pop("_staircase_correction", None)
    log.warning("chain_runner: staircase retries exhausted (non-critical) — accepting original draft")


_MAX_DIRECTIVE_PROFILE_RETRIES = 2


async def _handle_directive_profile_retry(
    stages: list[ChainStage],
    ctx: ChainContext,
    gate_result: dict,
    execute_prompt_stage_fn,
) -> None:
    """2026-05-10 — Profile validator regenerate-loop.

    When `prompt_directive_profile_gate` flags violations against the
    customer's profile (Skeptical + direct_ask, manipulation phrases in
    must_say, etc.), re-run `prompt_signal_analysis_combined` with the
    corrective_suffix passed through to mode1b_directive. Sonnet sees the
    violations and produces a clean directive re-roll.

    Up to _MAX_DIRECTIVE_PROFILE_RETRIES attempts. On exhaustion, keep the
    last directive (defensive: a profile violation isn't as severe as a
    staircase, so passing through is acceptable in the worst case).
    """
    target_name = "prompt_signal_analysis_combined"
    target_stage = next((s for s in stages if s.prompt_name == target_name), None)
    if target_stage is None:
        log.warning("chain_runner: directive_profile_retry — target stage %s not found", target_name)
        return

    corrective = gate_result.get("corrective_suffix") or ""
    if not corrective:
        log.info("chain_runner: directive_profile_retry — no corrective_suffix; skipping")
        return

    # Import directive validator for re-checking
    try:
        from directive_profile_gate import validate_directive
    except ImportError as e:
        log.warning("chain_runner: directive_profile_gate not importable: %s", e)
        return

    for attempt in range(1, _MAX_DIRECTIVE_PROFILE_RETRIES + 1):
        log.info("chain_runner: directive_profile retry attempt %d/%d (violations=%s)",
                 attempt, _MAX_DIRECTIVE_PROFILE_RETRIES,
                 [v.get("type") for v in (gate_result.get("violations") or [])])
        # Inject corrective marker for stage_signal_analysis_combined to pick up
        ctx.previous_responses["_directive_profile_correction"] = corrective
        try:
            new_output = await execute_prompt_stage_fn(target_stage, ctx)
        except Exception as e:
            log.warning("chain_runner: directive_profile regenerate attempt %d failed: %s",
                        attempt, e)
            return

        # Update ctx.previous_responses with the new output
        if isinstance(new_output, dict):
            ctx.previous_responses[target_name] = new_output
            new_directive = new_output.get("directive") if isinstance(new_output, dict) else None
            if isinstance(new_directive, dict):
                # Re-validate the new directive
                is_valid, meta = validate_directive(new_directive, ctx.opp_meta or {})
                if is_valid:
                    log.info("chain_runner: directive_profile regenerate succeeded on attempt %d",
                             attempt)
                    ctx.previous_responses["prompt_directive_profile_gate"] = {
                        "valid": True,
                        "verdict": "regenerated_successfully",
                        "retry_attempt": attempt,
                    }
                    return
                # Still violating — update gate_result so next loop sees latest
                gate_result["violations"] = meta.get("violations", [])
                corrective = meta.get("corrective_suffix") or corrective
        log.info("chain_runner: directive_profile regenerate attempt %d still violates", attempt)

    # Exhausted
    log.warning("chain_runner: directive_profile retries exhausted — accepting last directive")
    ctx.previous_responses["prompt_directive_profile_gate"] = {
        "valid": False,
        "verdict": "retries_exhausted",
        "retry_attempt": _MAX_DIRECTIVE_PROFILE_RETRIES,
        "violations": gate_result.get("violations") or [],
    }


_MAX_CAPITULATION_RETRIES = 2


async def _handle_capitulation_retry(
    stages: list[ChainStage],
    ctx: ChainContext,
    gate_result: dict,
    execute_prompt_stage_fn,
    build_capitulation_correction_fn,
) -> None:
    """T-87 regenerate-loop semantics. Mirrors _handle_staircase_retry.
    On capitulation detection, re-run prompt_build_answer with a corrective
    system_suffix that pulls concrete value from ctx.anchors. Up to N
    attempts; falls back to original on exhaustion."""
    target_stage = next(
        (s for s in stages if s.prompt_name == _REGENERATE_LOOP_TARGET
         and not s.is_programmatic),
        None,
    )
    if target_stage is None:
        log.warning("chain_runner: capitulation regenerate target %s not in chain",
                       _REGENERATE_LOOP_TARGET)
        return

    original_output = ctx.previous_responses.get(_REGENERATE_LOOP_TARGET)
    correction = build_capitulation_correction_fn(gate_result, ctx.anchors)
    tenant = (ctx.opp_meta.get("company") or "").strip()

    from capitulation_gate import check_capitulation

    for attempt in range(1, _MAX_CAPITULATION_RETRIES + 1):
        log.info("chain_runner: capitulation retry attempt %d/%d",
                 attempt, _MAX_CAPITULATION_RETRIES)
        # Use the same _staircase_correction marker key — chain_executor's
        # build_context_blocks reads it as a final correction block. Two gates
        # never fire on the same turn (staircase+capitulation are detected
        # sequentially) so the marker doesn't collide.
        ctx.previous_responses["_staircase_correction"] = correction
        try:
            new_output = await execute_prompt_stage_fn(target_stage, ctx)
        except Exception as e:
            log.warning("chain_runner: capitulation regenerate attempt %d failed: %s",
                          attempt, e)
            ctx.previous_responses[_REGENERATE_LOOP_TARGET] = original_output
            ctx.previous_responses.pop("_staircase_correction", None)
            return

        if isinstance(new_output, str):
            is_cap, meta = check_capitulation(
                candidate_text=new_output,
                dialog=ctx.dialog,
                tenant=tenant,
            )
            if not is_cap:
                ctx.previous_responses[_REGENERATE_LOOP_TARGET] = new_output
                ctx.previous_responses["prompt_anti_capitulation_gate"] = {
                    "capitulation_detected": True,
                    "verdict": "blocked_then_regenerated",
                    "retry_attempt": attempt,
                    "reason": gate_result.get("reason"),
                    "outcome": "regenerated_successfully",
                }
                ctx.previous_responses.pop("_staircase_correction", None)
                log.info("chain_runner: capitulation regenerate succeeded on attempt %d",
                         attempt)
                return

        log.info("chain_runner: capitulation regenerate attempt %d still violates",
                 attempt)

    # All retries exhausted; defensive fallback to original draft
    ctx.previous_responses[_REGENERATE_LOOP_TARGET] = original_output
    ctx.previous_responses["prompt_anti_capitulation_gate"] = {
        "capitulation_detected": True,
        "verdict": "blocked_fallback_to_original",
        "retry_attempt": _MAX_CAPITULATION_RETRIES,
        "reason": gate_result.get("reason"),
        "outcome": "fallback_to_original",
    }
    ctx.previous_responses.pop("_staircase_correction", None)
    log.warning("chain_runner: capitulation retries exhausted — accepting original draft")


_MAX_PREMATURE_CLOSE_RETRIES = 2


async def _handle_premature_close_retry(
    stages: list[ChainStage],
    ctx: ChainContext,
    gate_result: dict,
    execute_prompt_stage_fn,
    build_premature_close_correction_fn,
) -> None:
    """2026-05-14 regenerate-loop for premature-close. Mirrors the
    capitulation retry but checks via premature_close_gate.check_premature_close
    after each regenerate."""
    target_stage = next(
        (s for s in stages if s.prompt_name == _REGENERATE_LOOP_TARGET
         and not s.is_programmatic),
        None,
    )
    if target_stage is None:
        log.warning("chain_runner: premature_close regenerate target %s not in chain",
                       _REGENERATE_LOOP_TARGET)
        return

    original_output = ctx.previous_responses.get(_REGENERATE_LOOP_TARGET)
    correction = build_premature_close_correction_fn(gate_result, ctx.anchors)
    tenant = (ctx.opp_meta.get("company") or "").strip()

    from premature_close_gate import check_premature_close

    for attempt in range(1, _MAX_PREMATURE_CLOSE_RETRIES + 1):
        log.info("chain_runner: premature_close retry attempt %d/%d",
                 attempt, _MAX_PREMATURE_CLOSE_RETRIES)
        # Reuse the _staircase_correction marker — chain_executor reads it as
        # a generic post-build correction block. Different gates don't fire
        # on the same turn (capitulation → premature_close are evaluated
        # sequentially) so no collision.
        ctx.previous_responses["_staircase_correction"] = correction
        try:
            new_output = await execute_prompt_stage_fn(target_stage, ctx)
        except Exception as e:
            log.warning("chain_runner: premature_close regenerate attempt %d failed: %s",
                          attempt, e)
            ctx.previous_responses[_REGENERATE_LOOP_TARGET] = original_output
            ctx.previous_responses.pop("_staircase_correction", None)
            return

        if isinstance(new_output, str):
            is_premature, meta = check_premature_close(
                candidate_text=new_output,
                dialog=ctx.dialog,
                tenant=tenant,
            )
            if not is_premature:
                ctx.previous_responses[_REGENERATE_LOOP_TARGET] = new_output
                ctx.previous_responses["prompt_premature_close_gate"] = {
                    "premature_close_detected": True,
                    "verdict": "blocked_then_regenerated",
                    "retry_attempt": attempt,
                    "reason": gate_result.get("reason"),
                    "matched_anchor": gate_result.get("matched_anchor"),
                    "outcome": "regenerated_successfully",
                }
                ctx.previous_responses.pop("_staircase_correction", None)
                log.info("chain_runner: premature_close regenerate succeeded on attempt %d",
                         attempt)
                return

        log.info("chain_runner: premature_close regenerate attempt %d still violates",
                 attempt)

    # All retries exhausted — accept original draft (panel will end naturally)
    ctx.previous_responses[_REGENERATE_LOOP_TARGET] = original_output
    ctx.previous_responses["prompt_premature_close_gate"] = {
        "premature_close_detected": True,
        "verdict": "blocked_fallback_to_original",
        "retry_attempt": _MAX_PREMATURE_CLOSE_RETRIES,
        "reason": gate_result.get("reason"),
        "matched_anchor": gate_result.get("matched_anchor"),
        "outcome": "fallback_to_original",
    }
    ctx.previous_responses.pop("_staircase_correction", None)
    log.warning("chain_runner: premature_close retries exhausted — accepting original draft")


_MAX_INVARIANTS_RETRIES = 2


async def _handle_invariants_retry(
    stages: list[ChainStage],
    ctx: ChainContext,
    gate_result: dict,
    execute_prompt_stage_fn,
    build_invariants_correction_fn,
) -> None:
    """R5 regenerate-loop semantics. Mirrors _handle_capitulation_retry.
    On invariant violation (max-discount, disparagement, fabricated-stat,
    regulatory-language), re-run prompt_build_answer with a corrective
    system_suffix naming the specific violation. Up to N attempts; falls
    back to original on exhaustion."""
    target_stage = next(
        (s for s in stages if s.prompt_name == _REGENERATE_LOOP_TARGET
         and not s.is_programmatic),
        None,
    )
    if target_stage is None:
        log.warning("chain_runner: invariants regenerate target %s not in chain",
                       _REGENERATE_LOOP_TARGET)
        return

    original_output = ctx.previous_responses.get(_REGENERATE_LOOP_TARGET)
    correction = build_invariants_correction_fn(gate_result)

    from invariant_gates import check_invariants

    for attempt in range(1, _MAX_INVARIANTS_RETRIES + 1):
        log.info("chain_runner: invariants retry attempt %d/%d",
                 attempt, _MAX_INVARIANTS_RETRIES)
        ctx.previous_responses["_staircase_correction"] = correction
        try:
            new_output = await execute_prompt_stage_fn(target_stage, ctx)
        except Exception as e:
            log.warning("chain_runner: invariants regenerate attempt %d failed: %s",
                          attempt, e)
            ctx.previous_responses[_REGENERATE_LOOP_TARGET] = original_output
            ctx.previous_responses.pop("_staircase_correction", None)
            return

        if isinstance(new_output, str):
            still_bad, meta = check_invariants(
                candidate_text=new_output,
                dialog=ctx.dialog,
                opp_meta=ctx.opp_meta,
            )
            if not still_bad:
                ctx.previous_responses[_REGENERATE_LOOP_TARGET] = new_output
                ctx.previous_responses["prompt_invariants_gate"] = {
                    "invariants_violated": True,
                    "verdict": "blocked_then_regenerated",
                    "retry_attempt": attempt,
                    "violations": gate_result.get("violations") or [],
                    "violation_types": gate_result.get("violation_types") or [],
                    "outcome": "regenerated_successfully",
                }
                ctx.previous_responses.pop("_staircase_correction", None)
                log.info("chain_runner: invariants regenerate succeeded on attempt %d",
                         attempt)
                return

        log.info("chain_runner: invariants regenerate attempt %d still violates",
                 attempt)

    ctx.previous_responses[_REGENERATE_LOOP_TARGET] = original_output
    ctx.previous_responses["prompt_invariants_gate"] = {
        "invariants_violated": True,
        "verdict": "blocked_fallback_to_original",
        "retry_attempt": _MAX_INVARIANTS_RETRIES,
        "violations": gate_result.get("violations") or [],
        "violation_types": gate_result.get("violation_types") or [],
        "outcome": "fallback_to_original",
    }
    ctx.previous_responses.pop("_staircase_correction", None)
    log.warning("chain_runner: invariants retries exhausted — accepting original draft")


# ── Diagnostic / introspection helper ───────────────────────────────────────

def describe_chain(stages: list[ChainStage]) -> str:
    """Return a markdown-formatted description of the chain (for debugging
    + the demo UI)."""
    lines = ["| chain_type | order | prompt_name | provider | type |",
             "|---|---|---|---|---|"]
    for s in stages:
        kind = "🐍 programmatic (T-85)" if s.is_programmatic else "💬 LLM"
        lines.append(f"| {s.chain_type} | {s.execution_order} | "
                     f"{s.prompt_name} | {s.llm_provider} | {kind} |")
    return "\n".join(lines)
