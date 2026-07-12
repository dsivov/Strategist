# E-commerce Cart Recovery — example benchmark pack

Cart-abandonment recovery. The agent re-engages a shopper who left items in
their cart and works to convert hesitation — shipping cost concerns, price
comparisons, "just looking", distraction — into a completed checkout.

| | |
|---|---|
| Scenarios | 61 (sliced from the shared dataset by `tenant == "Ecommerce"`) |
| Goal | Customer completes the purchase → **won**; explicit refusal or continued abandonment → **lost** |
| Domain pack | `sales` (default framing + won/lost detection) |
| Personas | 9 diversity cells over 23 customer dimensions (motivator × decision logic × trust, …) |
| Opp types | `Abandoned Cart US` and `Abandoned Cart US No Consent` (consent state changes what the agent may reference) |

Each scenario embeds the full historical transcript (`seed_messages`,
PII-scrubbed) — the customer simulator uses the historical reply at the
current turn as a posture reference, never the future, so real outcomes don't
leak into the match.

This pack is one of the two bundled **examples**. To create your own
goal-oriented benchmark, start from [`../_template/`](../_template/).
