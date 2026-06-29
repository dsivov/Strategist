# Recipe — Self-Contained "Field Guide" HTML Doc

Instructions for an AI agent asked to produce a comprehensive, illustrated, **single self-contained HTML** project overview in the style of `docs/PI_OVERVIEW.html`. Follow this to get the same look and rigor for a different project.

## 0. The goal in one line
One `.html` file, openable offline in any browser, blog-style but deeply technical, dark theme, color-coded sections, **hand-drawn inline SVG diagrams + tables**, and **zero external resources**.

## 1. Hard rules (non-negotiable)
- **Fully self-contained.** No CDNs, no `<script src>`, no `<link>`, no `@import`, no remote fonts/images. All CSS in one `<style>` block; all diagrams as inline `<svg>`; JS (if any) inline and optional.
- **Accuracy first.** Read the project's own source-of-truth files (READMEs, design docs, "current state" notes, key source files, real artifacts) *before* writing. Quote real numbers; never invent metrics, param counts, schemas, or latencies. If a figure isn't verifiable, phrase it as rationale, not fact. Mark **done vs. pending** honestly and preserve any "this is not causal / not proven" caveats verbatim in spirit.
- **One file.** No length limit — long is fine.

## 2. Process
1. Launch 2–3 parallel `Explore` agents to map the repo: structure & docs; core architecture/models; code/pipeline/tests. Collect quotable facts + file paths.
2. Re-read the canonical files directly to confirm exact numbers (don't trust summaries for figures).
3. Ask the user (only if unclear): audience (mixed/technical/exec), output path, diagram style. Default: mixed audience, `docs/<NAME>_OVERVIEW.html`, inline SVG.
4. Write the file in one pass. Verify (§6). Report path + size + section list.

## 3. Design system (copy these tokens)
Use a `:root` block with CSS variables; reuse verbatim, only rename accent meanings to fit the project's main "actors":
```
--bg:#0f1117; --bg2:#161922; --panel:#1b1f2a; --panel2:#222736;
--ink:#e7ebf3; --ink-soft:#aab3c5; --ink-faint:#7b8499;
--line:#2c3346; --line-soft:#232a3a;
/* give each major component/subsystem its own accent (blue/teal/amber/violet): */
--a:#5b8def; --b:#19b89a; --c:#f0a73c; --ctrl:#a974f0; --danger:#ef5b6e; --ok:#3ecf8e;
--shadow:0 10px 30px rgba(0,0,0,.35); --radius:14px; --maxw:1180px;
```
Typography: system sans for body (`"Segoe UI",system-ui,…`), a mono stack for code. Gradient-clipped `<h1>`. Background = subtle radial gradient.

## 4. Layout & component catalog (reuse the classes)
- **Hero header** (`.hero` / `.hero-inner`): eyebrow + big gradient `h1` + `.lede` + `.hero-tags` pill row.
- **Two-column layout** (`.layout`): sticky `nav.toc` (numbered, scrollspy-highlighted) + `main`. Collapses to one column under 900px (hide ToC).
- **Sections**: `<section id="…">` with `<h2><span class="sec-num">NN</span> Title</h2>` + `.sec-sub`.
- **Cards / grids**: `.card`, `.grid` + `.g2/.g3/.g4`.
- **Callouts**: `.callout` + variant `.info / .caveat / .gotcha / .ok`, each with a `.clabel` header. Use these for "the point to internalise", honesty caveats, gotchas, invariants.
- **Stat tiles**: `.stat` (+ `.a/.b/.c/.ctrl`) with `.v` (big number) + `.l` (label) for headline metrics.
- **Tables**: wrap in `.tablewrap`; sticky `thead`; `.pill` (`.done/.wip/.todo/.danger/.a/.b/.c`) for status chips.
- **"At a glance" block**: a `.card` with a `dl.kvs` (Role / Type / How built / In→Out / Feeds) at the top of each major subsystem — this is what makes "what is X and how is it built" instantly clear. Pair it with a **comparison table** ("N things, N jobs") early in the doc.
- **Model/subsystem band**: `.modelband` colored gradient header with a `.badge`.
- **Timeline**: `.timeline` / `.tl` (when · rail+dot · body); add `.next` for future items.
- **Dir tree**: a `.card .tree .mono` with `.d` (dir) and `.note` (comment) spans.
- **Collapsible deep-dives**: `<details class="fold"><summary>…</summary><div class="foldbody">…</div></details>` — put line-by-line code / heavy internals here so the main read stays light.
- **Code blocks**: `<pre><code>` with manual highlight spans `.kw .str .num .cm .fn` (no highlighter library).
- **Footer** + a fixed `.backtotop` button.

(The fastest path: copy the entire `<style>` block and the optional scrollspy `<script>` from `PI_OVERVIEW.html`, then swap content. Both are project-agnostic.)

## 5. Diagrams — inline SVG only
- Hand-author each `<svg viewBox="0 0 W H">`; let it scale via `svg{max-width:100%;height:auto}`.
- Define gradients + one arrowhead `<marker id="ar">` once (in the first SVG's `<defs>`); SVG `<defs>` and `<style>` are document-global, so later SVGs can reference `url(#ar)`, `url(#gA)`, etc.
- Patterns that work well: end-to-end pipeline flow; per-component architecture (inputs→trunk→outputs); a "live path" with off-path/reference branches drawn **dashed + greyed** and clearly labeled; archetype/composition (A × B × C =); fallback cascade; timeline.
- **Make data flow literal and correct.** Show the actual arrows (e.g., "this component's output IS that component's input"). Emphasize the real/live path (bold, colored); de-emphasize reference/offline branches (dashed grey, with an explicit "not used live" label). Caption every figure ("Figure N — …") and state what is *not* shown.

## 6. Verify before declaring done
```
grep -inE 'https?://|src=|cdn|<link |@import|url\(http' <file>   # must be NONE
```
Then a quick Python check: parse with `html.parser` (well-formed), confirm every `nav.toc` `href="#id"` has a matching `<section id>`, and that tag counts balance for `section/svg/table/details/pre/script/style`. Report file path, size, and the section list.

## 7. Suggested section skeleton (adapt to the project)
1. Hero + one-line summary + the key invariant/constraint
2. Executive summary (accessible) + headline stat tiles
3. **"N components, N jobs" comparison table** (role / input / output / type / how built / why)
4. Big-picture end-to-end diagram (+ latency/scale callout)
5. Data/scale (tables + stat tiles)
6…k. One section per major subsystem — each opens with an **at-a-glance card**, then architecture diagram, tables, design-rationale callouts, and a collapsible implementation fold
k+1. Cross-cutting concerns (safety / degradation / failure modes)
k+2. Roadmap (done vs pending), timeline
k+3. Repository map (annotated tree)
last. Glossary

Keep the tone "blog, but precise": short narrative paragraphs, generous diagrams/tables, honesty callouts, and folds for the deep internals.
