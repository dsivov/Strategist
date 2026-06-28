"""Planner — independent, PCA-faithful R-side engine.

Hard rule: this package imports NOTHING from the Strategist codebase
(server/supervisor_full, chain_runner, script_library, precedent_retrieval,
gates, mining, etc.). It may consume raw data substrate (DB transcripts,
precedents.db as data) only via its own access code. The only bridge to the
harness is server/engines/__init__.py.

Reference: PCA — Planning with LLMs for Conversational Agents
(arXiv:2407.03884v1). §5.2 SOP Prediction (offline), §5.3 CoT+SOP (online),
§5.4 MCTS+SOP (Stage 2). SFT path intentionally skipped (vendor-only).
"""

ENGINE_NAME = "planner"
ENGINE_VERSION = "0.1-stub"  # M1 plumbing; M3 replaces .step() with CoT+SOP
