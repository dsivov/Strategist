// Persuasion Agent Benchmark — frontend controller
// (placeholder for v0 — fuller WS event handling lands when replayers are built)

const $ = (sel) => document.querySelector(sel);

const state = {
  scenarios: [],
  stressOnly: false,
  selected: null,
  speed: '5x',
  runMode: 'single',        // 'single' | 'batch'
  selectedIds: new Set(),   // opp_ids picked in the scenario list
  pack: '',                 // benchmark pack id ('' = all packs), from /api/benchmarks
  search: '',               // scenario-list text filter
  batch: null,              // { ids, i, total, results:[], stop } while a batch runs
  // Pluggable engines — loaded from /api/engines at boot; no hardcoded list.
  engines: { left: 'baseline', right: 'planner' },  // per-panel engine id
  engineParams: { left: {}, right: {} },               // per-panel param values
  engineSpecs: [],                                      // [{id,name,description,runnable,requires,params}]
  ws: null,
  sessionId: null,
  charts: { left: null, right: null },
  phase: 'idle',  // 'idle' | 'seed' | 'live'
  guideScenarios: null,  // [{id, title, blurb}, ...] — set on session_started
  demonstratedScenarios: new Set(),  // session-scoped set of scenario IDs fired so far
  // ClusterPlan state — set on plan_loaded WS event, updated on plan_advanced
  plan: null,             // {cluster_plan_id, n_phases, phases:[{id,name,max_turns}], goal, time_budget_turns}
  planCurrentPhase: 1,    // current phase id (server is source of truth)
  planAborted: false,
};

// ── Bootstrap ────────────────────────────────────────────────────────────────
async function init() {
  await initBenchmarks();
  await loadScenarios();
  await initEngines();
  initRunMode();
  initScenarioTools();
  initButtons();
  initSeedSlider();
  initCharts();
  // R20 — render persisted lift counter on page load + wire reset click
  renderLiftCounter();
  const liftEl = document.getElementById('liftCounter');
  if (liftEl) liftEl.addEventListener('click', resetLiftCounter);
  // R15 — render rolling stats on page load + wire clear button
  renderRollingStats();
  const rsClearBtn = document.getElementById('rsClearBtn');
  if (rsClearBtn) rsClearBtn.addEventListener('click', clearRollingStats);
  // R24 — restore directive-sidebar visibility from localStorage + wire toggle
  setDirectiveSidebarVisible(isDirectiveSidebarVisible());
  const dsToggle = document.getElementById('dsToggleBtn');
  if (dsToggle) dsToggle.addEventListener('click', toggleDirectiveSidebar);
  const dsClose = document.getElementById('dsCloseBtn');
  if (dsClose) dsClose.addEventListener('click', () => setDirectiveSidebarVisible(false));
}

async function loadScenarios() {
  const url = state.pack
    ? `/api/scenarios?pack=${encodeURIComponent(state.pack)}`
    : '/api/scenarios';
  const r = await fetch(url);
  state.scenarios = await r.json();
  renderScenarioList();
}

// ── Benchmark selector (packs from /api/benchmarks) ──────────────────────────
async function initBenchmarks() {
  const sel = document.getElementById('benchmarkSelect');
  if (!sel) return;
  try {
    const r = await fetch('/api/benchmarks');
    const data = await r.json();
    const packs = data.benchmarks || [];
    sel.innerHTML = '<option value="">All benchmarks</option>';
    packs.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.n_scenarios != null ? `${p.name} (${p.n_scenarios})` : p.name;
      opt.title = [p.description, p.goal && `Goal: ${p.goal}`].filter(Boolean).join('\n');
      sel.appendChild(opt);
    });
    sel.addEventListener('change', async () => {
      state.pack = sel.value;
      // Pack scopes the dataset — drop selections that may not exist in it.
      state.selectedIds.clear();
      state.selected = null;
      $('#scenarioMeta').hidden = true;
      await loadScenarios();
      updateRunButton();
    });
  } catch (e) {
    console.warn('failed to load /api/benchmarks', e);
    sel.innerHTML = '<option value="">All benchmarks</option>';
  }
}

// ── Run-model toggle (Single vs Batch) ───────────────────────────────────────
function initRunMode() {
  document.querySelectorAll('.runmode-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.runmode-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.runMode = btn.dataset.runmode;  // 'single' | 'batch'
      // Single mode keeps at most one selection.
      if (state.runMode === 'single' && state.selectedIds.size > 1) {
        const keep = [...state.selectedIds][0];
        state.selectedIds = new Set([keep]);
      }
      document.body.classList.toggle('mode-batch', state.runMode === 'batch');
      renderScenarioList();
      updateRunButton();
    });
  });
}

function initScenarioTools() {
  const search = document.getElementById('scenarioSearch');
  if (search) search.addEventListener('input', () => {
    state.search = search.value.trim().toLowerCase();
    renderScenarioList();
  });
  const randBtn = document.getElementById('selectRandomBtn');
  if (randBtn) randBtn.addEventListener('click', selectRandomN);
  const clearBtn = document.getElementById('clearSelBtn');
  if (clearBtn) clearBtn.addEventListener('click', () => {
    state.selectedIds.clear();
    state.selected = null;
    renderScenarioList();
    updateRunButton();
  });
}

function filteredScenarios() {
  let f = state.scenarios;
  if (state.search) {
    f = f.filter((s) => {
      // Searches the benchmark record: id, persona bucket + attributes, tenant.
      const hay = (`${s.scenario_id || ''} ${s.diversity_bucket || ''} ${s.cluster_name || ''} ` +
                   `${s.motivator || ''} ${s.decision_logic || ''} ${s.opp_id || ''} ${s.tenant || ''} ` +
                   `${JSON.stringify(s.attributes || {})}`).toLowerCase();
      return hay.includes(state.search);
    });
  }
  return f;
}

function scenarioLabel(s) {
  const a = s.attributes || {};
  const motivator = a.primary_motivator || s.motivator || '';
  const logic = a.decision_logic || s.decision_logic || '';
  const bucket = s.diversity_bucket || s.cluster_name || '';
  const dims = [motivator, logic].filter(Boolean).join(' · ');
  const title = s.scenario_id || (s.opp_id || '').slice(0, 8);
  const sub = [s.tenant, dims || bucket].filter(Boolean).join(' · ');
  return { title, sub };
}

function renderScenarioList() {
  const list = document.getElementById('scenarioList');
  if (!list) return;
  const filtered = filteredScenarios();
  if (!filtered.length) {
    list.innerHTML = '<div class="scenario-list-empty">No scenarios match.</div>';
    updateSelectedCount();
    return;
  }
  const multi = state.runMode === 'batch';
  list.innerHTML = '';
  filtered.forEach((s) => {
    const { title, sub } = scenarioLabel(s);
    const row = document.createElement('label');
    row.className = 'scenario-row';
    const checked = state.selectedIds.has(s.opp_id);
    if (checked) row.classList.add('selected');
    row.innerHTML =
      `<input type="${multi ? 'checkbox' : 'radio'}" name="scenarioSel" ${checked ? 'checked' : ''} />` +
      `<span class="sc-title">${title}</span><span class="sc-sub">${sub}</span>`;
    const input = row.querySelector('input');
    input.addEventListener('change', () => onScenarioToggle(s, input.checked));
    list.appendChild(row);
  });
  updateSelectedCount();
}

async function onScenarioToggle(s, checked) {
  if (state.runMode === 'single') {
    state.selectedIds = new Set(checked ? [s.opp_id] : []);
  } else {
    if (checked) state.selectedIds.add(s.opp_id);
    else state.selectedIds.delete(s.opp_id);
  }
  // Load detail for the (last) selected scenario so meta + seed slider populate.
  if (checked) {
    await loadScenarioDetail(s.opp_id);
  } else if (state.selectedIds.size === 0) {
    state.selected = null;
    $('#scenarioMeta').hidden = true;
  }
  renderScenarioList();
  updateRunButton();
}

async function loadScenarioDetail(oppId) {
  try {
    const r = await fetch(`/api/scenarios/${oppId}`);
    if (!r.ok) { console.warn('scenario detail failed', oppId); return null; }
    state.selected = await r.json();
    renderMeta(state.selected);
    return state.selected;
  } catch (e) { console.warn('scenario detail error', e); return null; }
}

function selectRandomN() {
  const n = Math.max(1, Number(document.getElementById('randomNInput')?.value) || 5);
  const pool = filteredScenarios().map((s) => s.opp_id);
  // Fisher-Yates partial shuffle
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const pick = pool.slice(0, state.runMode === 'single' ? 1 : n);
  state.selectedIds = new Set(pick);
  if (pick.length) loadScenarioDetail(pick[0]);
  renderScenarioList();
  updateRunButton();
}

function updateSelectedCount() {
  const el = document.getElementById('selectedCount');
  if (el) el.textContent = `${state.selectedIds.size} selected`;
}

function updateRunButton() {
  const btn = $('#runBtn');
  if (!btn) return;
  const n = state.selectedIds.size;
  if (state.runMode === 'batch') {
    btn.textContent = n > 1 ? `Run batch (${n})` : 'Run batch';
    btn.disabled = n < 1 || (state.batch && !state.batch.stop);
  } else {
    btn.textContent = 'Run scenario';
    btn.disabled = n !== 1;
  }
}

function renderMeta(s) {
  $('#scenarioMeta').hidden = false;
  $('#metaOppId').textContent = s.opp_id || '—';
  $('#metaTenant').textContent = s.tenant || '—';
  $('#metaCluster').textContent = s.diversity_bucket || '—';  // persona bucket (cohorts retired)
  const p = s.profile || {};
  const a = s.attributes || {};
  const motivator = p.motivator || a.primary_motivator || '—';
  const logic = p.decision_logic || a.decision_logic || '—';
  const trust = p.trust_level || a.trust_level || 'unknown trust';
  $('#metaProfile').textContent = `${motivator} · ${logic} · ${trust}`;
  $('#metaOutcome').textContent = s.historical_outcome || s.real_outcome || '—';
  if (s.n_msgs != null) {
    $('#metaMsgs').textContent = `${s.n_msgs} (${s.n_inbound || 0} customer / ${s.n_outbound || 0} agent)`;
  } else {
    $('#metaMsgs').textContent = `${(s.seed_messages || []).length} seed`;
  }

}

// Seed depth — Cold vs Warm only. The deep mid-transcript "warm K/N" seed was
// a replay-era artifact (irrelevant to the generic persona benchmark), so
// there is no numeric depth: Warm just preloads the scenario's opening
// exchange; Cold has the agent open the conversation itself.
function seedModeValue() {
  return document.getElementById('seedMode')?.value || 'warm';
}

function initSeedSlider() {
  // Cold/Warm select only — nothing to wire beyond the <select> itself.
}

// Resolve the seed_end_override int sent to /api/run:
//   cold → 1 (minimal; agent opens — backend needs ≥1 seed msg)
//   warm → 2 (opening exchange: first agent outreach + customer's first line)
function seedEndOverride() {
  return seedModeValue() === 'cold' ? 1 : 2;
}

function initButtons() {
  // Engine selectors are wired in initEngines() (dynamic, registry-driven).
  $('#runBtn').addEventListener('click', onRunClick);
  $('#stopBtn').addEventListener('click', stopSession);
  $('#resetBtn').addEventListener('click', resetUI);
  const batchStop = document.getElementById('batchStopBtn');
  if (batchStop) batchStop.addEventListener('click', () => { if (state.batch) state.batch.stop = true; });
}

// Run button dispatches on the current run model.
function onRunClick() {
  if (state.runMode === 'batch') return runBatch([...state.selectedIds]);
  return runScenario();
}

// ── Pluggable engines — populate L/R selectors from /api/engines ──────────────
async function initEngines() {
  let specs = [];
  try {
    const r = await fetch('/api/engines');
    const data = await r.json();
    specs = data.engines || [];
  } catch (e) {
    console.warn('failed to load /api/engines; engine selectors stay empty', e);
  }
  state.engineSpecs = specs;
  if (!specs.length) return;
  const byId = Object.fromEntries(specs.map((s) => [s.id, s]));
  // Fall back gracefully if the configured default isn't registered.
  if (!byId[state.engines.left]) state.engines.left = (byId.baseline ? 'baseline' : specs[0].id);
  if (!byId[state.engines.right]) state.engines.right = (byId.strategist ? 'strategist' : specs[specs.length - 1].id);

  ['left', 'right'].forEach((side) => {
    const sel = document.getElementById(side === 'left' ? 'engineSelectLeft' : 'engineSelectRight');
    if (!sel) return;
    sel.innerHTML = '';
    specs.forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = s.name;
      opt.title = (s.description || '') +
        (s.requires && s.requires.length ? `\nRequires: ${s.requires.join(', ')}` : '');
      sel.appendChild(opt);
    });
    sel.value = state.engines[side];
    sel.addEventListener('change', () => {
      state.engines[side] = sel.value;
      state.engineParams[side] = {};
      renderEngineParams(side);
      updatePanelTitles();
    });
    renderEngineParams(side);
  });
  updatePanelTitles();
}

function renderEngineParams(side) {
  const host = document.getElementById(side === 'left' ? 'engineParamsLeft' : 'engineParamsRight');
  if (!host) return;
  host.innerHTML = '';
  const spec = state.engineSpecs.find((s) => s.id === state.engines[side]);
  if (!spec || !spec.params || !spec.params.length) return;
  spec.params.forEach((p) => {
    if (state.engineParams[side][p.name] === undefined) {
      state.engineParams[side][p.name] = p.default;
    }
    const wrap = document.createElement('label');
    wrap.className = 'engine-param';
    wrap.title = p.help || '';
    if (p.type === 'bool') {
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = !!state.engineParams[side][p.name];
      cb.addEventListener('change', () => { state.engineParams[side][p.name] = cb.checked; });
      wrap.appendChild(cb);
      wrap.appendChild(document.createTextNode(' ' + p.label));
    } else if (p.type === 'enum') {
      wrap.appendChild(document.createTextNode(p.label + ' '));
      const psel = document.createElement('select');
      (p.choices || []).forEach((c) => {
        const o = document.createElement('option');
        o.value = c; o.textContent = c;
        psel.appendChild(o);
      });
      psel.value = state.engineParams[side][p.name];
      psel.addEventListener('change', () => { state.engineParams[side][p.name] = psel.value; });
      wrap.appendChild(psel);
    } else {
      wrap.appendChild(document.createTextNode(p.label + ' '));
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.value = state.engineParams[side][p.name] ?? '';
      inp.addEventListener('input', () => { state.engineParams[side][p.name] = inp.value; });
      wrap.appendChild(inp);
    }
    host.appendChild(wrap);
  });
}

function engineName(id) {
  const s = state.engineSpecs.find((x) => x.id === id);
  return s ? s.name : id;
}

function updatePanelTitles() {
  const lt = document.getElementById('leftPanelTitle');
  const rt = document.getElementById('rightPanelTitle');
  if (lt) lt.textContent = engineName(state.engines.left);
  if (rt) rt.textContent = engineName(state.engines.right);
}

let sessionTimerInterval = null;

function startSessionTimer() {
  const start = Date.now();
  $('#sessionTimer').hidden = false;
  if (sessionTimerInterval) clearInterval(sessionTimerInterval);
  sessionTimerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - start) / 1000);
    const m = Math.floor(elapsed / 60);
    const s = elapsed % 60;
    $('#sessionTimerText').textContent = m > 0 ? `${m}m ${s}s` : `${s}s`;
  }, 250);
}

function stopSessionTimer() {
  if (sessionTimerInterval) { clearInterval(sessionTimerInterval); sessionTimerInterval = null; }
  $('#sessionTimer').hidden = true;
}

function stopSession() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ action: 'stop' }));
  }
  setStatus('left', 'lost');
  setStatus('right', 'lost');
  stopSessionTimer();
  $('#stopBtn').disabled = true;
  $('#runBtn').disabled = false;
}

async function runScenario() {
  if (!state.selected) return;
  const oppId = state.selected.opp_id;
  const hardCustomer = !!document.getElementById('hardCustomerToggle')?.checked;
  // Seed depth → seed_end_override (cold=1, warm=K, 0=server per-scenario default)
  const seedEndOverrideVal = seedEndOverride();
  const r = await fetch(`/api/run/${oppId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      hard_customer: hardCustomer,
      seed_end_override: seedEndOverrideVal,
      engine: state.engines.right,         // R-side engine id (registry)
      engine_left: state.engines.left,     // L-side engine id (registry)
      engine_params: state.engineParams.right,
      engine_params_left: state.engineParams.left,
    }),
  });
  const { session_id } = await r.json();
  state.sessionId = session_id;
  if (hardCustomer) {
    document.body.classList.add('session-hard-customer');
  } else {
    document.body.classList.remove('session-hard-customer');
  }
  resetChats();
  resetCharts();
  setStatus('left', 'running');
  setStatus('right', 'running');
  $('#leftTurn').textContent = 'turn 0';
  $('#rightTurn').textContent = 'turn 0';
  $('#runBtn').disabled = true;
  $('#stopBtn').disabled = false;
  $('#wonBadge').hidden = true;
  // 6.1 — clear failure-mode badges from prior session
  $('#leftEndReason').hidden = true;
  $('#leftEndReason').className = 'end-reason-badge';
  $('#rightEndReason').hidden = true;
  $('#rightEndReason').className = 'end-reason-badge';
  // R21 — reset move-usage histogram for the new session
  resetMoveHistogram();
  // R17 — reset phase mini-sparklines for the new session
  resetPhaseSparkline();
  // R23 — reset emit visualizer for the new session
  resetEmitVisualizer();
  // Reset arch banner
  ['#tierChip1a','#tierChip1b','#tierChip2','#tierTrace'].forEach((s) => $(s)?.classList.remove('fired'));
  $('#tierCount1a').textContent = '0';
  $('#tierCount1b').textContent = '0';
  $('#tierCount2').textContent = '0';
  $('#traceStatus').textContent = '—';
  $('#cacheRate').textContent = '—';
  $('#tierCache').classList.remove('fired');
  $('#cgCallsBreakdown').textContent = '—';
  $('#tierCgCalls').hidden = true;
  $('#tierCgCalls').classList.remove('fired');
  $('#archBanner').hidden = false;
  // Reset guide-banner — server will re-send the scenario list on session_started
  state.demonstratedScenarios = new Set();
  if (state.guideScenarios) renderGuideBanner();
  // Reset ClusterPlan state — server will re-send plan_loaded if a plan exists
  state.plan = null;
  state.planCurrentPhase = 1;
  state.planAborted = false;
  $('#planProgress').hidden = true;
  // Reset win-proximity card — hide until next session_complete fires
  resetWinProximity();
  startSessionTimer();

  // Defensive: close any stale WS from a prior session before opening a new
  // one. Without this, late `won` events from the previous session leak into
  // the current panel and falsely show the WON badge (observed 2026-05-03).
  if (state.ws && state.ws.readyState !== WebSocket.CLOSED) {
    try { state.ws.close(); } catch (e) { /* ignore */ }
  }
  state.ws = new WebSocket(`ws://${location.host}/ws/${session_id}`);
  // Speed control retired (replay-era pacing); server uses its default.
  state.ws.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    // Ignore events from prior sessions whose WS happens to deliver after
    // this session started (e.g. buffered `won` events that fire the WON
    // badge spuriously on the new session's panel).
    if (ev.session_id && state.sessionId && ev.session_id !== state.sessionId) return;
    handleEvent(ev);
  };
  state.ws.onerror = (e) => console.warn('WS error', e);
  state.ws.onclose = () => {
    console.log('WS closed');
    stopSessionTimer();
    $('#stopBtn').disabled = true;
    // If a batch is awaiting this run's completion but session_complete never
    // arrived (error/abort), resolve so the batch can proceed.
    if (state._completeResolve) { const f = state._completeResolve; state._completeResolve = null; f(null); }
    else updateRunButton();
  };
}

// ── Batch A/B: run selected scenarios sequentially, aggregate paired summary ──
function runScenarioAwait() {
  return new Promise((resolve) => {
    state._completeResolve = resolve;
    runScenario();
  });
}

async function runBatch(oppIds) {
  if (!oppIds || !oppIds.length) return;
  state.batch = { ids: oppIds, i: 0, total: oppIds.length, results: [], stop: false };
  const summary = document.getElementById('batchSummary');
  if (summary) summary.hidden = false;
  document.getElementById('batchStopBtn')?.removeAttribute('hidden');
  $('#runBtn').disabled = true;
  const armL = engineName(state.engines.left), armR = engineName(state.engines.right);

  for (let i = 0; i < oppIds.length; i++) {
    if (state.batch.stop) break;
    state.batch.i = i;
    updateBatchProgress(`Running ${i + 1} / ${oppIds.length}…`);
    const detail = await loadScenarioDetail(oppIds[i]);
    if (!detail) { state.batch.results.push({ oppId: oppIds[i], left: null, right: null }); renderBatchSummary(); continue; }
    const res = await runScenarioAwait();  // resolves on session_complete
    state.batch.results.push({
      oppId: oppIds[i],
      scenario_id: detail.scenario_id || (detail.opp_id || '').slice(0, 8),
      left: res ? res.left_outcome : null,
      right: res ? res.right_outcome : null,
    });
    renderBatchSummary();
  }

  updateBatchProgress(state.batch.stop
    ? `Stopped — ${state.batch.results.length} / ${oppIds.length} run.`
    : `Done — ${oppIds.length} scenarios · L=${armL} vs R=${armR}.`);
  document.getElementById('batchStopBtn')?.setAttribute('hidden', '');
  state.batch.stop = true;  // mark finished so updateRunButton re-enables
  updateRunButton();
}

function updateBatchProgress(txt) {
  const el = document.getElementById('batchProgress');
  if (el) el.textContent = txt;
}

function renderBatchSummary() {
  const rows = (state.batch && state.batch.results) || [];
  const armL = engineName(state.engines.left), armR = engineName(state.engines.right);
  const isWon = (o) => o === 'won';
  const lWins = rows.filter((r) => isWon(r.left)).length;
  const rWins = rows.filter((r) => isWon(r.right)).length;
  const n = rows.length;
  const pct = (w) => n ? `${Math.round((w / n) * 100)}% (${w}/${n})` : '—';
  const lBetter = rows.filter((r) => isWon(r.left) && !isWon(r.right)).length;
  const rBetter = rows.filter((r) => isWon(r.right) && !isWon(r.left)).length;
  const ties = rows.filter((r) => (r.left != null && r.right != null) && isWon(r.left) === isWon(r.right)).length;

  const arms = document.getElementById('batchArmsTable');
  if (arms) arms.innerHTML =
    `<thead><tr><th>Arm</th><th>Win rate</th></tr></thead><tbody>` +
    `<tr><td>L · ${armL}</td><td>${pct(lWins)}</td></tr>` +
    `<tr><td>R · ${armR}</td><td>${pct(rWins)}</td></tr></tbody>`;
  const pw = document.getElementById('batchPairwiseTable');
  if (pw) pw.innerHTML =
    `<thead><tr><th>Pairwise</th><th>n</th></tr></thead><tbody>` +
    `<tr><td>L better</td><td>${lBetter}</td></tr>` +
    `<tr><td>R better</td><td>${rBetter}</td></tr>` +
    `<tr><td>Ties</td><td>${ties}</td></tr></tbody>`;
  const detail = document.getElementById('batchDetail');
  if (detail) detail.innerHTML = rows.map((r) => {
    const tag = (o) => `<span class="bs-oc ${o === 'won' ? 'won' : (o == null ? 'na' : 'lost')}">${o || '—'}</span>`;
    return `<div class="bs-row"><code>${r.scenario_id || ''}</code> L:${tag(r.left)} R:${tag(r.right)}</div>`;
  }).join('');
}

function resetUI() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ action: 'stop' }));
    state.ws.close();
  }
  state.ws = null;
  state.sessionId = null;
  resetChats();
  resetCharts();
  setStatus('left', 'idle');
  setStatus('right', 'idle');
  setPhase('idle');
  $('#leftTurn').textContent = '';
  $('#rightTurn').textContent = '';
  $('#wonBadge').hidden = true;
  resetWinProximity();
  stopSessionTimer();
  $('#runBtn').disabled = !state.selected;
  $('#stopBtn').disabled = true;
}

function resetWinProximity() {
  const wrap = $('#winProximity');
  if (!wrap) return;
  wrap.hidden = true;
  // Clear out card contents so old numbers don't briefly flash on re-show
  ['#proximityLeft', '#proximityRight'].forEach((sel, i) => {
    const el = $(sel);
    if (!el) return;
    el.classList.remove('won', 'partial', 'low');
    el.innerHTML = `<div class="prox-side-label">${i === 0 ? engineName(state.engines.left) : engineName(state.engines.right)}</div>`;
  });
  const deltaEl = $('#proximityDelta');
  if (deltaEl) {
    deltaEl.className = 'prox-delta';
    deltaEl.innerHTML = '';
  }
}

function resetChats() {
  $('#leftChat').innerHTML = '';
  $('#rightChat').innerHTML = '';
}

function setStatus(side, status) {
  const el = $(side === 'left' ? '#leftStatus' : '#rightStatus');
  el.className = `panel-status ${status}`;
  el.textContent = status === 'idle' ? 'Idle'
    : status === 'running' ? 'Live'
    : status === 'won' ? 'WON ✓'
    : status === 'lost' ? 'Lost' : status;
}

// ── Win-proximity rendering (T-77) ──────────────────────────────────────────
// Continuous 0..1 score per panel: actual WIN = 1.00, otherwise composite of
// trajectory_auc (commit × persuasion) + semantic_sim (cosine to win-cluster
// centroid) + payment_capture. Headline number lifts variance reduction.
function renderWinProximity(ev) {
  const wrap = $('#winProximity');
  if (!wrap) return;
  wrap.hidden = false;

  const fmtScore = v => v == null ? '—' : v.toFixed(2);
  const fmtPct   = v => v == null ? '—' : `${(v * 100).toFixed(0)}%`;

  const sides = [
    ['left',  ev.left,  $('#proximityLeft')],
    ['right', ev.right, $('#proximityRight')],
  ];
  for (const [side, p, el] of sides) {
    if (!el || !p) continue;
    const c = p.components || {};
    const won = p.actual_win;
    const score = p.score;
    el.classList.remove('won', 'partial', 'low');
    if (won)               el.classList.add('won');
    else if (score >= 0.5) el.classList.add('partial');
    else                   el.classList.add('low');

    el.innerHTML = `
      <div class="prox-headline">
        <span class="prox-score">${fmtScore(score)}</span>
        <span class="prox-label">${won ? 'WIN' : 'proximity to win'}</span>
      </div>
      <div class="prox-components">
        <div class="prox-row">
          <span class="prox-key">trajectory</span>
          <span class="prox-bar"><span class="prox-fill" style="width:${fmtPct(c.trajectory_auc)}"></span></span>
          <span class="prox-val">${fmtScore(c.trajectory_auc)}</span>
        </div>
        <div class="prox-row" ${c.semantic_sim == null ? 'style="opacity:0.5"' : ''}>
          <span class="prox-key">semantic</span>
          <span class="prox-bar"><span class="prox-fill" style="width:${fmtPct(c.semantic_sim)}"></span></span>
          <span class="prox-val">${c.semantic_sim == null ? 'n/a' : fmtScore(c.semantic_sim)}</span>
        </div>
        <div class="prox-row">
          <span class="prox-key">payment</span>
          <span class="prox-bar"><span class="prox-fill" style="width:${fmtPct(c.payment_capture)}"></span></span>
          <span class="prox-val">${fmtScore(c.payment_capture)}</span>
        </div>
      </div>
    `;
  }

  // Delta + advantage banner
  const deltaEl = $('#proximityDelta');
  if (deltaEl && ev.aggregate) {
    const d = ev.aggregate.delta;
    const sign = d > 0 ? '+' : '';
    const cls = d > 0.1 ? 'big-win' : d > 0 ? 'win' : d < -0.1 ? 'big-loss' : d < 0 ? 'loss' : 'tie';
    deltaEl.className = `prox-delta ${cls}`;
    deltaEl.innerHTML = `
      <span class="prox-delta-label">Supervisor lift (Δ proximity)</span>
      <span class="prox-delta-value">${sign}${d.toFixed(2)}</span>
    `;
  }
}

// ── Event handler (will expand as replayers come online) ─────────────────────
function handleEvent(ev) {
  switch (ev.event) {
    case 'session_ready':
      console.log('Session', ev.session_id, 'opp', ev.opp_id);
      break;
    case 'session_started':
      // Mark both panels as "Seed phase" until seed_complete fires
      setPhase('seed');
      // Surface precedent count from graph (closed-loop READ side).
      // 6.2 — Build a hovercard tooltip with a sample of actual precedent
      // edges so users can see "the system pulled these specific past
      // decisions before this session started" — not just a counter.
      if (ev.precedents_in_graph !== undefined) {
        $('#archBanner').hidden = false;
        $('#precedentCount').textContent = ev.precedents_in_graph.toLocaleString();
        if (ev.precedents_in_graph > 0) $('#tierPrecedents').classList.add('fired');
        // Hovercard content: histogram + sample edges
        const lines = [
          `Closed-loop READ — ${ev.precedents_in_graph} decision-trace edges in the knowledge graph for this tenant.`,
          '',
        ];
        if (ev.precedent_strategies && Object.keys(ev.precedent_strategies).length) {
          lines.push('By strategy:');
          for (const [tgt, cnt] of Object.entries(ev.precedent_strategies)) {
            lines.push(`  ${cnt}× ${tgt}`);
          }
          lines.push('');
        }
        if (Array.isArray(ev.precedent_sample) && ev.precedent_sample.length) {
          lines.push('Top precedents (by confidence):');
          ev.precedent_sample.forEach((p, i) => {
            const conf = p.confidence != null
              ? ` (conf=${(p.confidence * 100).toFixed(0)}%)` : '';
            lines.push(`  ${i + 1}. ${p.src} → ${p.tgt}${conf}`);
            if (p.outcome) lines.push(`     ${p.outcome}`);
          });
        } else if (ev.precedents_in_graph === 0) {
          lines.push('(no edges yet — graph is empty for this tenant)');
        }
        $('#tierPrecedents').title = lines.join('\n');
      }
      // Render the INTEGRATION scenario chips and seed any already-fired
      if (Array.isArray(ev.guide_scenarios)) {
        state.guideScenarios = ev.guide_scenarios;
        state.demonstratedScenarios = new Set(ev.scenarios_demonstrated_initial || []);
        renderGuideBanner();
      }
      // Surface opp_type in the integration panel
      if (ev.opp_type) {
        $('#oppTypeValue').textContent = ev.opp_type;
        $('#oppTypeTag').hidden = false;
      }
      break;
    case 'scenarios_update':
      if (Array.isArray(ev.demonstrated)) {
        state.demonstratedScenarios = new Set(ev.demonstrated);
        renderGuideBanner();
      }
      break;
    case 'plan_loaded':
      // ClusterPlan attached to this session — render the phase progress bar
      state.plan = {
        cluster_plan_id: ev.cluster_plan_id,
        n_phases: ev.n_phases,
        phases: ev.phases || [],
        goal: ev.goal,
        time_budget_turns: ev.time_budget_turns,
      };
      state.planCurrentPhase = 1;
      state.planAborted = false;
      renderPlanProgress();
      break;
    case 'plan_advanced':
      if (ev.to_phase != null) {
        state.planCurrentPhase = ev.to_phase;
        renderPlanProgress();
        // If reason includes 'aborted', visually mark the plan as aborted
        if (typeof ev.reason === 'string' && ev.reason.startsWith('aborted')) {
          state.planAborted = true;
          renderPlanProgress();
        }
      }
      break;
    case 'seed_complete':
      // Insert visual divider in BOTH chats and switch phase tag to LIVE
      insertPhaseDivider('left');
      insertPhaseDivider('right');
      setPhase('live');
      // Mark seed-end on the chart with a vertical line annotation
      if (ev.seed_end_turn != null) {
        state.seedEndTurn = ev.seed_end_turn;
        markSeedEndOnChart(ev.seed_end_turn);
      }
      break;
    case 'tier_used':
      // Live update of architecture-banner counters
      bumpTierCounter(ev.mode);
      break;
    case 'left_msg':
      addBubble('left', ev.role, ev.text, ev.directive);
      $('#leftTurn').textContent = `turn ${ev.turn}`;
      break;
    case 'right_msg':
      addBubble('right', ev.role, ev.text, ev.directive);
      $('#rightTurn').textContent = `turn ${ev.turn}`;
      // R22 — push supervisor confidence onto the chart's 3rd line
      if (ev.directive && typeof ev.directive.confidence === 'number') {
        pushConfidence(ev.turn, ev.directive.confidence);
      }
      // R24 — update directive-diff sidebar with the latest directive +
      // rendered text (only when this is an agent turn; customer turns
      // don't carry directives)
      if (ev.role === 'agent' && ev.directive) {
        renderDirectiveSidebar(ev.directive, ev.text || '', ev.turn);
      }
      break;
    case 'left_score':
      pushScore('left', ev.turn, ev.score, ev.commitment);
      break;
    case 'right_score':
      pushScore('right', ev.turn, ev.score, ev.commitment);
      break;
    case 'won':
      setStatus(ev.side, 'won');
      if (ev.side === 'right') showWonBadge();
      break;
    case 'end':
      setStatus(ev.side, ev.outcome === 'won' ? 'won' : 'lost');
      break;
    case 'win_proximity':
      renderWinProximity(ev);
      break;
    case 'session_tier_summary':
      updateArchBanner(ev.right_tier_counts || {});
      break;
    case 'decision_trace_emitted':
      $('#traceStatus').textContent = ev.result?.emitted ? '✓ saved to graph' : 'failed';
      $('#tierTrace').classList.add('fired');
      // R23 — bump live emit visualizer (counter + tile trail)
      bumpEmitVisualizer(!!ev.result?.emitted);
      break;
    case 'session_complete':
      stopSessionTimer();
      $('#stopBtn').disabled = true;
      if (!state.batch || state.batch.stop) $('#runBtn').disabled = false;
      // If a batch run is awaiting this scenario's outcome, hand it back.
      if (state._completeResolve) {
        const f = state._completeResolve; state._completeResolve = null;
        f({ left_outcome: ev.left_outcome, right_outcome: ev.right_outcome,
            left_reason: ev.left_reason, right_reason: ev.right_reason });
      }
      // 6.1 — Failure-mode taxonomy badges. End-reason carries WHY the
      // session ended (commitment_5, agent_graceful_close, customer_dropped,
      // saturated, stalled_low_engagement, customer_polite_close, incomplete...)
      renderEndReason('left',  ev.left_outcome,  ev.left_reason);
      renderEndReason('right', ev.right_outcome, ev.right_reason);
      // R20 — running lift counter across this device's session history
      bumpLiftCounter(ev.left_outcome, ev.right_outcome);
      // R15 — record session row for the rolling stats panel
      recordRollingSession(ev, _moveHistState.counts);
      // Surface CG-cache hit-rate for this session in the architecture banner.
      if (ev.cg_cache) {
        const c = ev.cg_cache;
        const rate = c.hit_rate != null ? Math.round(c.hit_rate * 100) : 0;
        $('#cacheRate').textContent = `${rate}% (${c.hits}/${c.hits + c.misses})`;
        if (c.hits > 0) $('#tierCache').classList.add('fired');
      }
      // Surface per-endpoint CG call breakdown for this session
      if (ev.cg_endpoint_calls) {
        const e = ev.cg_endpoint_calls;
        const parts = [];
        if (e.query_data || e.query_data_cached) {
          const cached = e.query_data_cached ? ` (${e.query_data_cached} cached)` : '';
          parts.push(`/query/data: ${(e.query_data || 0) + (e.query_data_cached || 0)}${cached}`);
        }
        if (e.cgr3) parts.push(`/cgr3: ${e.cgr3}`);
        if (e.query_auto) parts.push(`/query/auto: ${e.query_auto}`);
        if (e.decisions_read) parts.push(`/decisions: ${e.decisions_read}`);
        if (e.decision_emit) parts.push(`/emit: ${e.decision_emit}`);
        if (parts.length) {
          $('#cgCallsBreakdown').textContent = parts.join(' · ');
          $('#tierCgCalls').hidden = false;
          $('#tierCgCalls').classList.add('fired');
        }
      }
      break;
    case 'error':
      console.warn('Server error:', ev.message);
      break;
  }
}

function renderPlanProgress() {
  const wrap = $('#planProgress');
  if (!state.plan) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  $('#planId').textContent = state.plan.cluster_plan_id || '—';
  $('#planGoal').textContent = state.plan.goal
    ? `goal: ${state.plan.goal}` + (state.plan.time_budget_turns ? `  ·  budget: ${state.plan.time_budget_turns} turns` : '')
    : '';
  // Render phases
  const phasesEl = $('#planPhases');
  phasesEl.innerHTML = '';
  state.plan.phases.forEach((ph) => {
    const chip = document.createElement('div');
    let cls = 'phase-chip';
    if (state.planAborted && ph.id === state.planCurrentPhase) cls += ' aborted';
    else if (ph.id < state.planCurrentPhase) cls += ' done';
    else if (ph.id === state.planCurrentPhase) cls += ' current';
    chip.className = cls;
    chip.innerHTML = `<span class="phase-num">${ph.id}</span><span class="phase-name">${escapeHtml(ph.name)}</span><span class="phase-budget">${ph.max_turns ? '≤ ' + ph.max_turns + 't' : ''}</span>`;
    phasesEl.appendChild(chip);
  });
}

function renderGuideBanner() {
  const banner = $('#guideBanner');
  const grid = $('#guideGrid');
  if (!state.guideScenarios) return;
  banner.hidden = false;
  // Out-of-scope (S6 in our POC): render but visually dimmed, never lights up.
  const OUT_OF_SCOPE = new Set(['S6']);
  const inner = state.guideScenarios.map((s) => {
    const fired = state.demonstratedScenarios.has(s.id);
    const oos = OUT_OF_SCOPE.has(s.id);
    const cls = oos ? 'guide-chip oos' : (fired ? 'guide-chip fired' : 'guide-chip');
    const status = oos ? 'n/a' : (fired ? '✓' : '·');
    return `<div class="${cls}">
      <div class="guide-chip-head"><span class="chip-id">${s.id}</span><span class="chip-status">${status}</span></div>
      <div class="chip-title">${escapeHtml(s.title)}</div>
      <div class="chip-blurb">${escapeHtml(s.blurb)}</div>
    </div>`;
  }).join('');
  grid.innerHTML = inner;
}

function updateArchBanner(counts) {
  $('#archBanner').hidden = false;
  const c1a = counts['1a'] || 0;
  const c1b = counts['1b'] || 0;
  const c2  = counts['2'] || 0;
  $('#tierCount1a').textContent = c1a;
  $('#tierCount1b').textContent = c1b;
  $('#tierCount2').textContent = c2;
  if (c1a > 0) $('#tierChip1a').classList.add('fired');
  if (c1b > 0) $('#tierChip1b').classList.add('fired');
  if (c2  > 0) $('#tierChip2').classList.add('fired');
}

// Live per-turn counter bump as each tier_used event arrives
function bumpTierCounter(mode) {
  const map = { '1a': '#tierCount1a', '1b': '#tierCount1b', '2': '#tierCount2' };
  const chipMap = { '1a': '#tierChip1a', '1b': '#tierChip1b', '2': '#tierChip2' };
  const el = $(map[mode]);
  const chip = $(chipMap[mode]);
  if (el) {
    const cur = parseInt(el.textContent || '0', 10) || 0;
    el.textContent = cur + 1;
  }
  if (chip) chip.classList.add('fired');
}

// Vertical divider on the persuasion-score chart at seed-end turn.
// Uses a Chart.js plugin via afterDraw hook (no annotation library needed).
function markSeedEndOnChart(seedEndTurn) {
  if (!state.combinedChart) return;
  state.seedEndTurn = seedEndTurn;
  state.combinedChart.update();
}

function setPhase(phase) {
  state.phase = phase;
  ['left', 'right'].forEach((side) => {
    const tag = $(side === 'left' ? '#leftPhaseTag' : '#rightPhaseTag');
    if (!tag) return;
    if (phase === 'seed') {
      tag.className = 'phase-tag seed';
      tag.textContent = 'Seed (historical replay)';
    } else if (phase === 'live') {
      tag.className = 'phase-tag live';
      tag.textContent = 'Live A/B';
    } else {
      tag.className = 'phase-tag';
      tag.textContent = '—';
    }
  });
}

function insertPhaseDivider(side) {
  const chat = $(side === 'left' ? '#leftChat' : '#rightChat');
  const wrap = document.createElement('div');
  wrap.className = 'phase-divider';
  const label = document.createElement('span');
  label.className = 'divider-label';
  label.textContent = side === 'right'
    ? '— Supervisor takes over here —'
    : '— Original agent takes over here —';
  wrap.appendChild(label);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function addBubble(side, role, text, directive) {
  const chat = $(side === 'left' ? '#leftChat' : '#rightChat');
  const div = document.createElement('div');
  let cls = `bubble ${role === 'system' ? 'system' : role}`;
  if (state.phase === 'seed') cls += ' seed-phase';
  div.className = cls;
  const tag = document.createElement('div');
  tag.className = 'role-tag';
  tag.textContent = role === 'customer' ? 'Customer'
                  : role === 'system'   ? 'System'
                  : 'Agent';
  const content = document.createElement('div');
  content.textContent = text;
  div.appendChild(tag);
  div.appendChild(content);
  if (directive && side === 'right' && role === 'agent') {
    const tag2 = document.createElement('div');
    const mode = directive.mode || '1b';
    tag2.className = `directive-tag tier-${mode}`;
    const strat = directive.primary_strategy || '?';
    const tone = directive.tone || '';
    const rules = directive.rules || [];
    const cg = directive.cg;

    // Header line: tier badge + strategy + tone + CG retrieval count
    let modeLabel = (
      mode === '1a' ? 'Playbook'
      : mode === '2' ? 'Deep Reasoning'
      : 'Live Retrieval'
    );
    let header = `${modeLabel} · ${strat}`;
    if (tone) header += ` · ${tone}`;
    if (rules.length) header += ` · rules: ${rules.join(', ')}`;
    if (cg) header += ` · KG: ${cg.entities}E ${cg.relations}R ${cg.chunks}C`;
    const headerDiv = document.createElement('div');
    headerDiv.className = 'directive-header';
    headerDiv.textContent = header;
    // 7.4 — Confidence indicator. Colored dot inline in the header.
    const conf = directive.confidence;
    if (typeof conf === 'number') {
      const cdot = document.createElement('span');
      cdot.className = 'directive-conf-dot';
      const lvl = conf >= 0.85 ? 'high' : conf >= 0.6 ? 'med' : 'low';
      cdot.dataset.level = lvl;
      cdot.title = `Supervisor confidence: ${(conf * 100).toFixed(0)}% (${lvl})`;
      cdot.textContent = ' ●';
      headerDiv.appendChild(cdot);
    }
    // 7.1 — "Why this directive?" rationale shown on hover (and as small ?
    // affordance so users discover the tooltip exists).
    const rationale = directive.rationale;
    if (rationale && typeof rationale === 'string' && rationale.trim().length > 0) {
      headerDiv.classList.add('has-rationale');
      headerDiv.title = `Why: ${rationale}`;
      const why = document.createElement('span');
      why.className = 'directive-why-affordance';
      why.textContent = '  why?';
      why.title = `Why: ${rationale}`;
      headerDiv.appendChild(why);
    }
    tag2.appendChild(headerDiv);

    // 7-bonus — Anchors hovercard chip (Insurance: last-year price, market avg,
    // max discount; Ecommerce: KG fields populated).
    const anchorsBrief = directive.anchors_brief;
    if (anchorsBrief && Object.keys(anchorsBrief).length > 0) {
      const aEl = document.createElement('div');
      aEl.className = 'directive-anchors';
      aEl.innerHTML = '<span class="anchors-icon">📊</span> <span class="anchors-label">anchors</span>';
      // Build hovercard content
      const lines = [];
      const fmt = (k, v) => {
        if (v === null || v === undefined) return null;
        if (typeof v === 'boolean') return `${k}: ${v ? 'yes' : 'no'}`;
        return `${k}: ${v}`;
      };
      const labelMap = {
        last_year_price_usd: 'Last year', current_quoted_price_usd: 'Quote now',
        market_avg_for_segment_usd: 'Market avg', max_discount_pct_internal: 'Max discount %',
        claimed_increase_pct: 'YoY claimed %', actual_market_yoy_change_pct: 'YoY actual %',
        loyalty_years: 'Loyalty (yrs)', synthetic: 'Synthetic',
        provenance: 'Source',
        max_authorized_discount_pct_internal: 'Max discount %',
        _cg_queries_returned_content: 'KG fields populated',
        _cg_queries_total: 'KG queries total',
      };
      Object.entries(anchorsBrief).forEach(([k, v]) => {
        const ln = fmt(labelMap[k] || k, v);
        if (ln) lines.push(ln);
      });
      const hovercard = lines.join('\n');
      aEl.title = `Anchors / Reference Frame:\n\n${hovercard}`;
      tag2.appendChild(aEl);
    }

    // Conversation phase chip — multi-turn arc awareness (§9.2 #7).
    // Shows dynamic phase classifier output; if cluster_plan disagrees,
    // both labels surface so the supervisor's reconciliation is auditable.
    const cp = directive.conversation_phase;
    if (cp && cp.current) {
      const cpEl = document.createElement('div');
      cpEl.className = 'directive-conversation-phase';
      let cpText = `<span class="cp-icon">🌀</span> <span class="cp-name">${escapeHtml(cp.current)}</span>`;
      if (cp.turns_in_phase) {
        cpText += `<span class="cp-tip">·${cp.turns_in_phase}t</span>`;
      }
      if (cp.cluster_plan_phase && String(cp.cluster_plan_phase) !== String(cp.current)) {
        cpText += `<span class="cp-disagree" title="cluster_plan disagrees">⇄ plan:${escapeHtml(String(cp.cluster_plan_phase))}</span>`;
      }
      cpEl.innerHTML = cpText;
      tag2.appendChild(cpEl);
      // R17 — bump phase mini-sparkline in the panel header
      bumpPhaseSparkline(side, cp.current);
    }

    // Tier 2 concrete_move — show right after the header so it's the most
    // prominent piece of strategy information per turn (Phase 1 of
    // strategy-enum-extension, 2026-05-03).
    const cm = directive.concrete_move;
    if (cm && cm.name) {
      const cmEl = document.createElement('div');
      cmEl.className = 'directive-concrete-move';
      let cmText = `<span class="cm-icon">⚡</span> <span class="cm-name">${escapeHtml(cm.name)}</span>`;
      if (cm.primary_param) {
        cmText += `<span class="cm-param">${escapeHtml(cm.primary_param)}</span>`;
      }
      cmEl.innerHTML = cmText;
      tag2.appendChild(cmEl);
      // R21 — bump the move-usage histogram (right panel only)
      if (side === 'right') bumpMoveHistogram(cm.name);
    }

    // 7.5 — Counterfactual chip. When any retry fired, surface the path:
    // considered <rejected> → switched to <chosen>. Three retry sources:
    //   - adherence_retry (T-78 win-plan adherence)
    //   - signal_adherence_retry (T-79 signal-driven adherence binding)
    //   - consistency_retry (R4/Q12 directive-internal consistency)
    const retries = [];
    const ar = directive.adherence_retry;
    if (ar && ar.retried && ar.violation && ar.second_strategy && ar.violation !== ar.second_strategy) {
      retries.push({ from: ar.violation, to: ar.second_strategy, why: 'win-plan' });
    }
    const sar = directive.signal_adherence_retry;
    if (sar && sar.retried && sar.violation && sar.second_strategy && sar.violation !== sar.second_strategy) {
      const sig = sar.primary_signal ? ` for ${sar.primary_signal}` : '';
      retries.push({ from: sar.violation, to: sar.second_strategy, why: `binding rules${sig}` });
    }
    const cr = directive.consistency_retry;
    if (cr && cr.retried) {
      const violations = cr.violation_rules || [];
      const fellBack = cr.fell_back_to_strip ? ' (still inconsistent → stripped)' : '';
      retries.push({
        from: 'contradicting must_not_say rule',
        to: 'revised directive' + fellBack,
        why: `internal consistency · ${violations.length} rule(s) flagged: ${violations.slice(0,1)[0] || ''}`,
      });
    }
    const mr = directive.move_validity_retry;
    if (mr && mr.retried) {
      const firstName = mr.first_move_name || mr.first_violation || 'invalid move';
      const secondName = mr.second_move_name || '(null)';
      const fellBack = mr.fell_back ? ' (still invalid → kept original)' : '';
      retries.push({
        from: firstName,
        to: secondName + fellBack,
        why: `move validity · ${mr.first_violation || 'invalid'}`,
      });
    }
    retries.forEach((r) => {
      const cfEl = document.createElement('div');
      cfEl.className = 'directive-counterfactual';
      cfEl.innerHTML =
        `<span class="cf-icon">⤳</span> considered ` +
        `<span class="cf-from">${escapeHtml(r.from)}</span> → ` +
        `<span class="cf-to">${escapeHtml(r.to)}</span>`;
      cfEl.title = `Supervisor first proposed "${r.from}" but the ${r.why} check rejected it; supervisor revised to "${r.to}".`;
      tag2.appendChild(cfEl);
    });

    // Facts to anchor — show 1-2 with source attribution
    const facts = directive.facts_to_anchor || [];
    facts.forEach((f) => {
      const factEl = document.createElement('div');
      factEl.className = 'directive-fact';
      factEl.innerHTML = `<span class="fact-bullet">▸</span> ${escapeHtml(f.text)}`
                       + (f.source ? `<span class="fact-source">⇢ ${escapeHtml(f.source)}</span>` : '');
      tag2.appendChild(factEl);
    });
    // Must not say
    if (directive.must_not_say_top) {
      const mns = document.createElement('div');
      mns.className = 'directive-mns';
      mns.innerHTML = `<span class="mns-bullet">⊘</span> avoid: ${escapeHtml(directive.must_not_say_top)}`;
      tag2.appendChild(mns);
    }
    // 2026-05-05 — Tier 1+2 enrichment badges. Surface tactical features,
    // cialdini levers, and historical-delta on cache hits so the user can
    // verify the new system is firing live.
    const cacheHit = directive.cache_hit;
    const recoveryMode = directive.cache_recovery_mode;
    const scoreBand = directive.cache_score_band;
    const tactical = directive.cache_tactical_brief || [];
    const cialdini = directive.cache_cialdini_brief || [];
    const histDelta = directive.historical_delta_after_move;
    if (cacheHit || recoveryMode || tactical.length || cialdini.length
        || (typeof histDelta === 'number')) {
      const chips = document.createElement('div');
      chips.className = 'tier12-chips';
      const addChip = (text, cls) => {
        const c = document.createElement('span');
        c.className = `tier12-chip ${cls || ''}`;
        c.textContent = text;
        chips.appendChild(c);
      };
      if (cacheHit) addChip('Mode 1a · cached', 'chip-mode1a');
      if (recoveryMode) addChip(`🔥 recovery${scoreBand ? ' · ' + scoreBand : ''}`, 'chip-recovery');
      cialdini.forEach(c => addChip(c, 'chip-cialdini'));
      tactical.forEach(t => addChip(t.replace(/^_/,'').replace(/_/g,' '), 'chip-tactical'));
      if (typeof histDelta === 'number') {
        const sign = histDelta >= 0 ? '+' : '';
        const cls = histDelta >= 0.05 ? 'chip-delta-pos' : (histDelta <= -0.05 ? 'chip-delta-neg' : 'chip-delta-neutral');
        addChip(`hist Δ ${sign}${histDelta.toFixed(2)}`, cls);
      }
      tag2.appendChild(chips);
    }
    div.appendChild(tag2);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function showWonBadge() {
  $('#wonBadge').hidden = false;
}

// R24 — Directive-diff sidebar. Floating right-side panel that surfaces the
// supervisor's structured directive next to the actual rendered text. Toggle
// preference persists in localStorage. Updated on every right-panel agent
// turn with the latest directive.
const DS_KEY = 'directive_sidebar_visible_v1';

function isDirectiveSidebarVisible() {
  try { return localStorage.getItem(DS_KEY) === '1'; }
  catch (e) { return false; }
}

function setDirectiveSidebarVisible(v) {
  try { localStorage.setItem(DS_KEY, v ? '1' : '0'); } catch (e) {}
  const el = document.getElementById('directiveSidebar');
  if (el) el.hidden = !v;
  document.body.classList.toggle('ds-open', v);
  // Layout is handled by CSS grid (body.ds-open). No inline-style hacks.
}

function toggleDirectiveSidebar() {
  setDirectiveSidebarVisible(!isDirectiveSidebarVisible());
}

function renderDirectiveSidebar(directive, agentText, turn) {
  const body = document.getElementById('directiveSidebarBody');
  if (!body) return;
  if (!directive) {
    body.innerHTML = '<div class="ds-empty">No directive on this turn (left panel? legacy path?).</div>';
    return;
  }
  const strat = directive.primary_strategy || '?';
  const tone = directive.tone || '';
  const cm = directive.concrete_move;
  const cp = directive.conversation_phase;
  const conf = typeof directive.confidence === 'number' ? directive.confidence : null;
  const facts = directive.facts_to_anchor || [];
  const mns = directive.must_not_say_top;
  const ar = directive.adherence_retry;
  const sar = directive.signal_adherence_retry;
  const cr = directive.consistency_retry;
  const mr = directive.move_validity_retry;
  const rules = directive.rules || [];
  const cg = directive.cg;

  const sections = [];

  // Header — turn + mode + strategy + confidence
  let confDot = '';
  if (conf != null) {
    const lvl = conf >= 0.85 ? 'high' : conf >= 0.6 ? 'med' : 'low';
    confDot = `<span class="ds-conf ds-conf-${lvl}" title="confidence ${(conf*100).toFixed(0)}%">●</span>`;
  }
  sections.push(`
    <div class="ds-section ds-section-head">
      <div class="ds-turn">Turn ${turn || '?'}  ·  ${escapeHtml(directive.mode || '?')}</div>
      <div class="ds-strat-row">
        <span class="ds-strat">${escapeHtml(strat)}</span>
        ${tone ? `<span class="ds-tone">${escapeHtml(tone)}</span>` : ''}
        ${confDot}
      </div>
    </div>
  `);

  // Phase chip
  if (cp && cp.current) {
    const planNote = (cp.cluster_plan_phase && String(cp.cluster_plan_phase) !== String(cp.current))
      ? ` <span class="ds-plan-disagree">⇄ plan:${escapeHtml(String(cp.cluster_plan_phase))}</span>`
      : '';
    sections.push(`
      <div class="ds-section">
        <div class="ds-label">Conversation arc</div>
        <div class="ds-phase">🌀 ${escapeHtml(cp.current)} (${cp.turns_in_phase || 0}t)${planNote}</div>
      </div>
    `);
  }

  // Concrete move
  if (cm && cm.name) {
    sections.push(`
      <div class="ds-section">
        <div class="ds-label">Tactical move (Tier 2)</div>
        <div class="ds-move">⚡ ${escapeHtml(cm.name)}</div>
        ${cm.primary_param ? `<div class="ds-move-param">${escapeHtml(cm.primary_param)}</div>` : ''}
      </div>
    `);
  }

  // Facts to anchor
  if (facts.length) {
    sections.push(`
      <div class="ds-section">
        <div class="ds-label">Facts to anchor (${facts.length})</div>
        ${facts.map(f => `
          <div class="ds-fact">
            <span class="ds-fact-bullet">▸</span>
            ${escapeHtml(f.text)}
            ${f.source ? `<span class="ds-fact-source">⇢ ${escapeHtml(f.source)}</span>` : ''}
          </div>
        `).join('')}
      </div>
    `);
  }

  // Must-not-say
  if (mns) {
    sections.push(`
      <div class="ds-section">
        <div class="ds-label">Must not say</div>
        <div class="ds-mns">⊘ ${escapeHtml(mns)}</div>
      </div>
    `);
  }

  // Retries fired (counterfactual reasoning)
  const retryRows = [];
  if (ar && ar.retried) retryRows.push(`<div class="ds-retry">⤳ win-plan: <s>${escapeHtml(ar.violation || '?')}</s> → <b>${escapeHtml(ar.second_strategy || '?')}</b></div>`);
  if (sar && sar.retried) retryRows.push(`<div class="ds-retry">⤳ binding rules (${escapeHtml(sar.primary_signal || '?')}): <s>${escapeHtml(sar.violation || '?')}</s> → <b>${escapeHtml(sar.second_strategy || '?')}</b></div>`);
  if (cr && cr.retried) retryRows.push(`<div class="ds-retry">⤳ internal consistency: ${cr.violation_rules?.length || 0} contradicting rule(s)${cr.fell_back_to_strip ? ' (still bad → stripped)' : ' (revised)'}</div>`);
  if (mr && mr.retried) retryRows.push(`<div class="ds-retry">⤳ move validity: <s>${escapeHtml(mr.first_move_name || mr.first_violation || '?')}</s> → <b>${escapeHtml(mr.second_move_name || '(null)')}</b>${mr.fell_back ? ' (still invalid → kept)' : ''}</div>`);
  if (retryRows.length) {
    sections.push(`
      <div class="ds-section">
        <div class="ds-label">Counterfactual reasoning (retries fired)</div>
        ${retryRows.join('')}
      </div>
    `);
  }

  // CG retrieval counts
  if (cg) {
    sections.push(`
      <div class="ds-section ds-meta">
        <div class="ds-label">Knowledge-graph retrieval</div>
        <div class="ds-cg">${cg.entities||0}E · ${cg.relations||0}R · ${cg.chunks||0}C</div>
      </div>
    `);
  }

  // Rules enforced
  if (rules.length) {
    sections.push(`
      <div class="ds-section ds-meta">
        <div class="ds-label">Rules enforced</div>
        <div class="ds-rules">${rules.map(r => `<span class="ds-rule">${escapeHtml(r)}</span>`).join(' ')}</div>
      </div>
    `);
  }

  // The actual rendered answer
  if (agentText) {
    sections.push(`
      <div class="ds-section ds-rendered">
        <div class="ds-label">↓ Rendered as</div>
        <div class="ds-text">${escapeHtml(agentText)}</div>
      </div>
    `);
  }

  body.innerHTML = sections.join('');
}


// R23 — Closed-loop emit visualizer. Each successful emit adds a green tile
// to the architecture banner's "Decision audit trail" chip + bumps a counter.
// Failed emits add a red tile. Resets on new session.
const _emitVizState = { count: 0, fails: 0, tiles: [] };

function bumpEmitVisualizer(emitted) {
  _emitVizState.count += 1;
  if (!emitted) _emitVizState.fails += 1;
  _emitVizState.tiles.push(emitted);
  const counter = document.getElementById('emitCounter');
  const countEl = document.getElementById('emitCount');
  const trailEl = document.getElementById('emitTrail');
  if (counter) counter.hidden = false;
  if (countEl) countEl.textContent = _emitVizState.count;
  if (trailEl) {
    // Show last 25 tiles
    const visible = _emitVizState.tiles.slice(-25);
    trailEl.innerHTML = visible.map(ok =>
      `<span class="et-tile ${ok ? 'ok' : 'fail'}"></span>`
    ).join('');
  }
}

function resetEmitVisualizer() {
  _emitVizState.count = 0;
  _emitVizState.fails = 0;
  _emitVizState.tiles = [];
  const counter = document.getElementById('emitCounter');
  if (counter) counter.hidden = true;
  const trailEl = document.getElementById('emitTrail');
  if (trailEl) trailEl.innerHTML = '';
}

// R17 — Phase mini-sparkline (per panel). Each agent turn's phase is appended
// as a colored tile; the sequence at-a-glance shows the conversation arc.
const _phaseSparkState = { left: [], right: [] };
const PHASE_COLORS = {
  greet: '#5A6A7E',                  // grey
  probe: '#4A90E2',                  // blue
  present: '#78A0C8',                // light-blue
  objection_handling: '#C77B66',     // warm-red
  close_attempt: '#C8A04A',          // amber
  commit_pending: '#F5A623',         // orange
  retreat: '#9F7ACA',                // purple
  won: '#6FB87E',                    // green
  lost: '#E07060',                   // red
};

function bumpPhaseSparkline(side, phase) {
  if (!phase || !PHASE_COLORS[phase]) return;
  _phaseSparkState[side].push(phase);
  renderPhaseSparkline(side);
}

function renderPhaseSparkline(side) {
  const el = document.getElementById(side === 'left' ? 'leftPhaseSpark' : 'rightPhaseSpark');
  if (!el) return;
  const seq = _phaseSparkState[side];
  if (!seq.length) {
    el.innerHTML = '';
    return;
  }
  // Cap the visible sequence at last 20 to avoid overflow
  const visible = seq.slice(-20);
  el.innerHTML = visible.map(p =>
    `<span class="ps-tile" style="background:${PHASE_COLORS[p]}" title="${p}"></span>`
  ).join('');
}

function resetPhaseSparkline() {
  _phaseSparkState.left = [];
  _phaseSparkState.right = [];
  renderPhaseSparkline('left');
  renderPhaseSparkline('right');
}

// R21 — Move-usage live histogram per session. Bumped every time the right
// panel emits an agent message with a concrete_move; reset at session start.
const _moveHistState = { counts: {}, max: 0 };

function resetMoveHistogram() {
  _moveHistState.counts = {};
  _moveHistState.max = 0;
  const wrap = $('#moveHistogram');
  if (wrap) wrap.hidden = true;
  const bars = $('#moveHistogramBars');
  if (bars) bars.innerHTML = '';
}

function bumpMoveHistogram(moveName) {
  if (!moveName) return;
  const c = _moveHistState.counts;
  c[moveName] = (c[moveName] || 0) + 1;
  if (c[moveName] > _moveHistState.max) _moveHistState.max = c[moveName];
  const wrap = $('#moveHistogram');
  if (wrap) wrap.hidden = false;
  const bars = $('#moveHistogramBars');
  if (!bars) return;
  // Build sorted bar list (most-used first)
  const sorted = Object.entries(c).sort((a, b) => b[1] - a[1]);
  bars.innerHTML = '';
  sorted.forEach(([name, count]) => {
    const w = Math.max(8, Math.round(100 * count / Math.max(_moveHistState.max, 3)));
    // Highlight bars at >=3 fires (variation-pressure trigger threshold is 2;
    // 3+ is "this is the dominant move").
    const dom = count >= 3 ? ' dominant' : (count >= 2 ? ' warm' : '');
    const row = document.createElement('div');
    row.className = `mh-row${dom}`;
    row.innerHTML = `
      <div class="mh-name">${escapeHtml(name)}</div>
      <div class="mh-track">
        <div class="mh-fill" style="width:${w}%"></div>
        <div class="mh-count">${count}×</div>
      </div>`;
    bars.appendChild(row);
  });
}

// R20 — Lift counter persisted in localStorage. Tracks rolling supervisor
// lift across this device's session history. Reset by clicking the chip.
const LIFT_KEY = 'lift_counter_v1';

function loadLiftCounter() {
  try {
    const raw = localStorage.getItem(LIFT_KEY);
    if (!raw) return { l_wins: 0, r_wins: 0, n: 0 };
    const v = JSON.parse(raw);
    return {
      l_wins: v.l_wins | 0,
      r_wins: v.r_wins | 0,
      n: v.n | 0,
    };
  } catch (e) {
    return { l_wins: 0, r_wins: 0, n: 0 };
  }
}

function saveLiftCounter(state) {
  try {
    localStorage.setItem(LIFT_KEY, JSON.stringify(state));
  } catch (e) { /* ignore quota issues */ }
}

function renderLiftCounter() {
  const s = loadLiftCounter();
  const el = $('#liftCounter');
  if (!el) return;
  if (s.n === 0) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  $('#liftRWins').textContent = s.r_wins;
  $('#liftLWins').textContent = s.l_wins;
  $('#liftN').textContent = s.n;
  const lift_pp = s.n ? Math.round(100 * (s.r_wins - s.l_wins) / s.n) : 0;
  const sign = lift_pp >= 0 ? '+' : '';
  const ppEl = $('#liftPP');
  ppEl.textContent = `${sign}${lift_pp}pp`;
  ppEl.dataset.sentiment = lift_pp > 0 ? 'pos' : (lift_pp < 0 ? 'neg' : 'flat');
}

function bumpLiftCounter(left_outcome, right_outcome) {
  const s = loadLiftCounter();
  s.n += 1;
  if (left_outcome === 'won') s.l_wins += 1;
  if (right_outcome === 'won') s.r_wins += 1;
  saveLiftCounter(s);
  renderLiftCounter();
}

// R15 — Rolling stats across last N sessions. Persists session records to
// localStorage and renders a 4-card panel (win rate, persuasion delta, top
// moves, end-reasons).
const ROLLING_KEY = 'rolling_sessions_v1';
const ROLLING_CAP = 50;  // keep last 50 sessions

function loadRolling() {
  try {
    const raw = localStorage.getItem(ROLLING_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) { return []; }
}

function saveRolling(arr) {
  try {
    localStorage.setItem(ROLLING_KEY, JSON.stringify(arr.slice(-ROLLING_CAP)));
  } catch (e) { /* ignore */ }
}

function recordRollingSession(ev, sessionMoves) {
  const rec = {
    ts: Date.now(),
    l_outcome: ev.left_outcome || null,
    l_reason: ev.left_reason || null,
    r_outcome: ev.right_outcome || null,
    r_reason: ev.right_reason || null,
    l_persuasion_final: ev.left_persuasion_final ?? null,
    r_persuasion_final: ev.right_persuasion_final ?? null,
    moves: { ...sessionMoves },
  };
  const arr = loadRolling();
  arr.push(rec);
  saveRolling(arr);
  renderRollingStats();
}

function clearRollingStats() {
  if (!confirm('Clear rolling stats history? (keeps lift counter; clears moves/end-reasons aggregates)')) return;
  saveRolling([]);
  renderRollingStats();
}

function renderRollingStats() {
  const arr = loadRolling();
  const wrap = $('#rollingStats');
  if (!wrap) return;
  if (!arr.length) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const n = arr.length;
  $('#rsN').textContent = n;
  // Win rates
  const lWins = arr.filter(r => r.l_outcome === 'won').length;
  const rWins = arr.filter(r => r.r_outcome === 'won').length;
  $('#rsRWinRate').textContent = `${rWins}/${n} (${Math.round(100 * rWins / n)}%)`;
  $('#rsLWinRate').textContent = `${lWins}/${n} (${Math.round(100 * lWins / n)}%)`;
  // Persuasion deltas — Δ = final - 0.25 (start). Approximation; if final is null skip
  const lPers = arr.map(r => r.l_persuasion_final).filter(v => typeof v === 'number');
  const rPers = arr.map(r => r.r_persuasion_final).filter(v => typeof v === 'number');
  const avg = (xs) => xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : null;
  const lAvg = avg(lPers);
  const rAvg = avg(rPers);
  $('#rsLPersAvg').textContent = lAvg != null ? lAvg.toFixed(2) : '—';
  $('#rsRPersAvg').textContent = rAvg != null ? rAvg.toFixed(2) : '—';
  // Top moves across all sessions
  const moveTotals = {};
  arr.forEach(r => {
    Object.entries(r.moves || {}).forEach(([m, c]) => {
      moveTotals[m] = (moveTotals[m] || 0) + c;
    });
  });
  const sorted = Object.entries(moveTotals).sort((a,b) => b[1] - a[1]).slice(0, 5);
  const movesEl = $('#rsTopMoves');
  movesEl.innerHTML = sorted.length
    ? sorted.map(([m, c]) => `<div class="rs-move-row"><span>${escapeHtml(m)}</span><b>${c}×</b></div>`).join('')
    : '—';
  // End-reasons (right panel only — supervised side)
  const erTotals = {};
  arr.forEach(r => {
    if (!r.r_reason) return;
    erTotals[r.r_reason] = (erTotals[r.r_reason] || 0) + 1;
  });
  const erSorted = Object.entries(erTotals).sort((a,b) => b[1] - a[1]);
  const erEl = $('#rsEndReasons');
  erEl.innerHTML = erSorted.length
    ? erSorted.map(([r, c]) => `<div class="rs-er-row"><span>${escapeHtml(r)}</span><b>${c}×</b></div>`).join('')
    : '—';
}

function resetLiftCounter() {
  if (!confirm('Reset supervisor-lift counter? (clears all session history on this device)')) return;
  saveLiftCounter({ l_wins: 0, r_wins: 0, n: 0 });
  renderLiftCounter();
}

// 6.1 — Failure-mode taxonomy: render an end-reason badge in the panel
// header at session_complete. Each reason has a plain-English description
// and a sentiment class (won / partial / lost) that drives badge color.
const END_REASON_TAXONOMY = {
  commitment_5: {
    label: 'WON · commitment 5',
    sentiment: 'won',
    desc: 'Customer reached commitment level 5 (provided payment details). Clean close.',
  },
  agent_graceful_close: {
    label: 'lost · agent gave up',
    sentiment: 'lost',
    desc: 'Agent abandoned the negotiation and closed politely. Common when the agent runs out of plays.',
  },
  customer_dropped: {
    label: 'lost · customer dropped',
    sentiment: 'lost',
    desc: 'Customer disengaged before reaching a decision. Conversation died mid-flight.',
  },
  customer_polite_close: {
    label: 'lost · polite no',
    sentiment: 'partial',
    desc: 'Customer declined warmly without committing — relationship preserved, deal lost.',
  },
  stalled_low_engagement: {
    label: 'lost · stalled',
    sentiment: 'lost',
    desc: 'Engagement collapsed; customer stopped responding meaningfully. Partial loss — door may still be open.',
  },
  saturated: {
    label: 'lost · saturated',
    sentiment: 'partial',
    desc: 'Conversation reached natural endpoint; no further progress possible. Customer engaged longer than baseline.',
  },
  incomplete: {
    label: 'incomplete',
    sentiment: 'partial',
    desc: 'Session reached its turn budget without conclusive outcome. May indicate the supervisor was still moving the customer when time ran out.',
  },
  mutual_farewell: {
    label: 'mutual farewell',
    sentiment: 'partial',
    desc: 'Both sides closed the conversation gracefully without a transaction.',
  },
  unknown: {
    label: '?',
    sentiment: 'partial',
    desc: 'No specific end-reason recorded.',
  },
};

function renderEndReason(side, outcome, reason) {
  const elId = side === 'left' ? '#leftEndReason' : '#rightEndReason';
  const el = $(elId);
  if (!el) return;
  const key = (reason || '').toLowerCase().replace(/\s+/g, '_') || 'unknown';
  const taxo = END_REASON_TAXONOMY[key] || {
    label: outcome === 'won' ? `WON · ${reason || 'commitment'}`
                              : `lost · ${reason || 'unknown'}`,
    sentiment: outcome === 'won' ? 'won' : 'lost',
    desc: reason ? `End reason: ${reason}` : 'No specific end-reason recorded.',
  };
  el.textContent = taxo.label;
  el.className = `end-reason-badge sentiment-${taxo.sentiment}`;
  el.title = taxo.desc;
  el.hidden = false;
}

// ── Charts ──────────────────────────────────────────────────────────────────
// One combined chart with two lines: Original (blue) + Supervised (orange).
// X-axis = turn number; we collect score points keyed by turn so both lines
// stay aligned even if one panel emits more turns than the other.
function initCharts() {
  if (typeof Chart === 'undefined') {
    console.error('Chart.js failed to load — graph will not render');
    document.getElementById('combinedGraph').replaceWith(
      Object.assign(document.createElement('div'),
        { textContent: 'Chart.js failed to load',
          style: 'color:#C0392B; padding:20px;' }));
    return;
  }
  const ctx = document.getElementById('combinedGraph').getContext('2d');
  // Chart.js plugin: draws a vertical line at seed-end turn with a label.
  const seedEndPlugin = {
    id: 'seedEndLine',
    afterDraw(chart) {
      const seedEnd = state.seedEndTurn;
      if (seedEnd == null) return;
      // Find x-pixel for the first label that is >= seedEnd
      const labels = chart.data.labels || [];
      let xIndex = -1;
      for (let i = 0; i < labels.length; i++) {
        const t = parseInt(labels[i].replace(/^T/, ''), 10);
        if (t > seedEnd) { xIndex = i; break; }
      }
      if (xIndex === -1) return;
      const xScale = chart.scales.x;
      // Position line halfway between previous and current label (the boundary)
      const prev = xScale.getPixelForValue(xIndex - 1);
      const cur = xScale.getPixelForValue(xIndex);
      const x = (prev + cur) / 2;
      const top = chart.chartArea.top;
      const bot = chart.chartArea.bottom;
      const ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = '#F5A623';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bot);
      ctx.stroke();
      ctx.setLineDash([]);
      // Label at top
      ctx.fillStyle = '#F5A623';
      ctx.font = '600 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('SUPERVISOR TAKES OVER', x, top - 4);
      ctx.restore();
    },
  };
  state.combinedChart = new Chart(ctx, {
    type: 'line',
    plugins: [seedEndPlugin],
    data: {
      labels: [],
      datasets: [
        {
          label: 'Original',
          data: [],
          borderColor: '#4A90E2',
          backgroundColor: '#4A90E233',
          borderWidth: 2.5,
          tension: 0.3,
          pointStyle: 'circle',
          pointRadius: 5,
          pointHoverRadius: 7,
          pointBackgroundColor: '#4A90E2',
          fill: false,
          spanGaps: true,
        },
        {
          label: 'Supervised',
          data: [],
          borderColor: '#F5A623',
          backgroundColor: '#F5A62333',
          borderWidth: 3,
          borderDash: [8, 4],
          tension: 0.3,
          pointStyle: 'rectRot',  // diamond — distinct from circle
          pointRadius: 6,
          pointHoverRadius: 8,
          pointBackgroundColor: '#F5A623',
          fill: false,
          spanGaps: true,
        },
        // R22 — Supervisor confidence trajectory (per-turn directive.confidence)
        {
          label: 'Supervisor confidence',
          data: [],
          borderColor: '#9F7ACA',
          backgroundColor: '#9F7ACA22',
          borderWidth: 1.5,
          borderDash: [2, 3],
          tension: 0.25,
          pointStyle: 'triangle',
          pointRadius: 3,
          pointHoverRadius: 5,
          pointBackgroundColor: '#9F7ACA',
          fill: false,
          spanGaps: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 350 },
      scales: {
        y: {
          min: 0, max: 1,
          ticks: { color: '#8B98A5', stepSize: 0.2 },
          grid: { color: '#243044' },
          title: { display: true, text: 'Persuasion score', color: '#8B98A5' },
        },
        x: {
          ticks: { color: '#8B98A5' },
          grid: { color: '#243044' },
          title: { display: true, text: 'Turn', color: '#8B98A5' },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            color: '#8B98A5',
            font: { size: 10 },
            boxWidth: 12,
            padding: 8,
          },
        },
      },
    },
  });
  state.scoreByTurn = { left: {}, right: {} };
  // R22 — supervisor confidence per turn (right panel only)
  state.confByTurn = {};
}

// R22 — record supervisor confidence for the current right-panel turn.
// Called from addBubble when a directive carries directive.confidence.
function pushConfidence(turn, conf) {
  if (!state.combinedChart) return;
  if (typeof conf !== 'number' || isNaN(conf)) return;
  if (!state.confByTurn) state.confByTurn = {};
  state.confByTurn[turn] = conf;
  redrawCombined();
}

function pushScore(side, turn, score, commitment) {
  if (!state.combinedChart) return;
  state.scoreByTurn[side][turn] = score;
  redrawCombined();
}

function redrawCombined() {
  if (!state.combinedChart) return;
  // Union of all turns seen, sorted ascending
  const allTurns = new Set();
  Object.keys(state.scoreByTurn.left).forEach(t => allTurns.add(parseInt(t)));
  Object.keys(state.scoreByTurn.right).forEach(t => allTurns.add(parseInt(t)));
  const sorted = Array.from(allTurns).sort((a, b) => a - b);

  // Tiny Y-offset (+0.015) on the Supervised line for visual separation when
  // both lines have identical scores. Investors see two distinct lines instead
  // of an orange line hidden under a blue line.
  const SUPERVISED_OFFSET = 0.015;
  state.combinedChart.data.labels = sorted.map(t => `T${t}`);
  state.combinedChart.data.datasets[0].data = sorted.map(t =>
    state.scoreByTurn.left[t] !== undefined ? state.scoreByTurn.left[t] : null);
  state.combinedChart.data.datasets[1].data = sorted.map(t => {
    const v = state.scoreByTurn.right[t];
    return v !== undefined ? Math.min(1, v + SUPERVISED_OFFSET) : null;
  });
  // R22 — confidence dataset (3rd line)
  if (state.combinedChart.data.datasets[2]) {
    const cbT = state.confByTurn || {};
    // Add any confidence-only turns to the X axis if they're not already there
    Object.keys(cbT).forEach(t => allTurns.add(parseInt(t)));
    const sortedC = Array.from(allTurns).sort((a, b) => a - b);
    state.combinedChart.data.labels = sortedC.map(t => `T${t}`);
    state.combinedChart.data.datasets[2].data = sortedC.map(t =>
      cbT[t] !== undefined ? cbT[t] : null);
  }
  state.combinedChart.update();
}

function resetCharts() {
  if (!state.combinedChart) return;
  state.scoreByTurn = { left: {}, right: {} };
  state.confByTurn = {};
  state.seedEndTurn = null;
  state.combinedChart.data.labels = [];
  state.combinedChart.data.datasets[0].data = [];
  state.combinedChart.data.datasets[1].data = [];
  if (state.combinedChart.data.datasets[2]) state.combinedChart.data.datasets[2].data = [];
  state.combinedChart.update();
}

function escapeHtml(s) {
  if (!s) return '';
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}


document.addEventListener('DOMContentLoaded', init);
