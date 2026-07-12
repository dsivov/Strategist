# Persuasion Agent Benchmark — documentation

Documentation for the pluggable platform for A/B-comparing goal-oriented
conversation agents.

## Start here

- **[STRATEGIST_OVERVIEW.html](STRATEGIST_OVERVIEW.html)** — illustrated,
  self-contained field guide. Open it in any browser (no server, no internet).
  Best first read for the big picture.
- **[BLOG_PLAY_THE_MATCH.html](BLOG_PLAY_THE_MATCH.html)** — "Play the Match,
  Not the Exam": a human-friendly blog post on *why* and *how* we evaluate
  goal-oriented persuasion agents as A/B matches, framed through Cialdini's
  *Influence*. Accents on the dataset, the metrics, and the A/B methodology.

## Reference

| Doc | For |
|-----|-----|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Understanding how the platform fits together — the four pluggable seams, the per-turn run loop, the live-replayer engine routing, and the behavior-preservation strategy. |
| [PLUGIN_GUIDE.md](PLUGIN_GUIDE.md) | Building extensions: a new **engine** (in-tree or as an installable plugin), a new **domain pack**, a new **scenario source**, or a new **benchmark pack**. |
| [API.md](API.md) | The exact public Python surface (`poc.*`) and the server's REST + WebSocket endpoints. |

## Project-level docs (repo root)

- [../README.md](../README.md) — install, quickstart, dataset, the three reference engines.
- [../INTEGRATION.md](../INTEGRATION.md) — step-by-step engine adapter walkthrough.

## The one-paragraph mental model

An **engine** produces one agent message per turn via a single async method,
`produce(opp_meta, dialog_history, business_rules) -> (text, meta)`. Engines are
declared once in the **registry** (`poc/registry.py`), which both the headless
`Benchmark` and the live dual-panel server read from — so a registered engine
shows up in the runner, the `/api/engines` endpoint, and the UI selectors with
no extra wiring. The customer it negotiates against (the **simulator**) and the
win/lost criteria (the **domain pack**) are likewise pluggable, where the
scenarios come from is a **scenario source**, and goal-oriented scenario
bundles ship as **benchmark packs** (`benchmarks/*/pack.json` — copy
`benchmarks/_template/` to add your own). The bundled packs are a
sales/insurance corpus; what's decoupled is the code's assumptions about it.
