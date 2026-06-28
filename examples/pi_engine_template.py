"""Template: adapt the PI stack to the POC Engine protocol.

This is the file the PI team fills in. It's a thin shim that:
  - takes the (opp_meta, dialog_history, business_rules) input the benchmark
    sends each turn,
  - calls into the PI runtime however that's set up at your end,
  - returns (customer_facing_text, telemetry_dict).

The benchmark records `meta` verbatim; nothing in there is interpreted. Put
whatever you want there — strategy, tone, hint_confidence, gates fired,
training-data flags, anything that helps the post-run analysis.

Usage (see examples/run_benchmark.py for the full driver):

    from pi_engine_template import PIEngine

    engine = PIEngine(api_url="http://pi-internal:8080", api_key=...)
    results = await Benchmark(scenarios).run_arm("pi", engine)
"""
from __future__ import annotations

from typing import Any


class PIEngine:
    """Reference adapter. Replace the body of produce() with your call into
    the PI stack (HTTP, local Python, gRPC — whatever you ship)."""

    def __init__(self, **config):
        # Stash whatever connection info / clients / models you need.
        self.config = config

    async def produce(
        self,
        opp_meta: dict[str, Any],
        dialog_history: list[dict[str, Any]],
        business_rules: str = "",
    ) -> tuple[str, dict[str, Any]]:
        """One agent turn.

        Inputs
        ------
        opp_meta : dict
            Customer profile + scenario context. Keys typically present:
              id, company, primary_motivator, decision_logic, trust_level,
              communication_style, objection_pattern, emotional_volatility,
              regulatory_focus, budget_sensitivity, purchase_urgency,
              primary_resistance, opp_type, anchors (per-opp price/coverage),
              voice_profile (optional)
            Pass through unchanged to your stack.

        dialog_history : list[dict]
            Each item is {"role": "agent"|"customer", "text": str}.
            Ordered chronologically. The most recent customer message is the
            one your engine needs to respond to.

        business_rules : str
            Tenant-specific compliance text. May be empty.

        Returns
        -------
        (text, meta) : tuple[str, dict]
            text  — the customer-facing reply.
            meta  — free-form telemetry. Recorded per turn; not interpreted
                    by the benchmark. Suggested keys:
                      strategy, tone, hint_confidence, model_version,
                      escalated, gates_fired, gates_regens, ...
        """
        # ── REPLACE THIS BLOCK WITH YOUR CALL INTO THE PI STACK ─────────
        #
        # Example 1 — HTTP call to a PI service:
        #
        #   import httpx
        #   async with httpx.AsyncClient(timeout=10) as http:
        #       resp = await http.post(
        #           f"{self.config['api_url']}/produce",
        #           json={
        #               "opp_meta":       opp_meta,
        #               "dialog_history": dialog_history,
        #               "business_rules": business_rules,
        #           },
        #           headers={"Authorization": f"Bearer {self.config['api_key']}"},
        #       )
        #   data = resp.json()
        #   return data["text"], data.get("meta", {})
        #
        # Example 2 — direct Python call:
        #
        #   result = await self.pi_client.predict_and_render(
        #       opp_meta, dialog_history, business_rules,
        #   )
        #   return result.text, {
        #       "strategy":        result.strategy,
        #       "tone":            result.tone,
        #       "hint_confidence": result.confidence,
        #       "escalated":       False,
        #   }
        #
        # ────────────────────────────────────────────────────────────────
        raise NotImplementedError(
            "Replace this with your PI runtime call. See the comment block "
            "above for two reference patterns (HTTP or in-process)."
        )
