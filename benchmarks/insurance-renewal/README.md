# Insurance Renewal — example benchmark pack

Auto-insurance renewal negotiations. The agent opens with a renewal quote
(comprehensive / third-party + mandatory premium) and negotiates against a
simulated customer who may push back on price, quote competitors, ask for
discounts, or simply go quiet.

| | |
|---|---|
| Scenarios | 51 (sliced from the shared dataset by `tenant == "Insurance"`) |
| Goal | Customer agrees to renew → **won**; explicit decline or drop-off → **lost** |
| Domain pack | `sales` (default framing + won/lost detection) |
| Personas | 6 diversity cells over 23 customer dimensions (motivator × decision logic × trust, …) |
| Anchors | Per-scenario pricing: last-year price, current quote, market average, max internal discount |

Each scenario embeds the full historical transcript (`seed_messages`,
PII-scrubbed) — the customer simulator uses the historical reply at the
current turn as a posture reference, never the future, so real outcomes don't
leak into the match.

This pack is one of the two bundled **examples**. To create your own
goal-oriented benchmark, start from [`../_template/`](../_template/).
