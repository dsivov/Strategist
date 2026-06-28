// Trace Logs page — fetches and renders /api/trace/list and /api/trace/{sid}.
// Standalone (no dependency on app.js); served at /logs.

function escapeHtml(s) {
  if (!s) return '';
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

// Cached traces list — used by both the list rendering AND the chart
let _tracesCache = [];

async function loadTraceList() {
  const list = document.getElementById('tracesList');
  const meta = document.getElementById('tracesListMeta');
  list.innerHTML = 'Loading…';
  try {
    const r = await fetch('/api/trace/list?limit=200');
    const data = await r.json();
    const traces = data.traces || [];
    _tracesCache = traces;
    meta.textContent = `${traces.length} session${traces.length === 1 ? '' : 's'}`;
    if (!traces.length) {
      list.innerHTML = '<div class="traces-empty">No traces yet — run a session and look back here.</div>';
      return;
    }
    list.innerHTML = traces.map(t => {
      const startedShort = (t.started_at || '').slice(11, 19);
      const dateShort = (t.started_at || '').slice(0, 10);
      const tenant = t.company || '?';
      const cluster = (t.cluster_id !== null && t.cluster_id !== undefined) ? `c${t.cluster_id}` : '';
      const dl = t.demo_label || (t.opp_type || '').slice(0, 18) || '—';
      // Outcome badge
      const outcomeChip = (t.L_outcome || t.R_outcome) ? renderOutcomeChip(t) : '';
      // Δ persuasion
      const deltaChip = (typeof t.delta_max === 'number')
        ? `<span class="trace-delta ${t.delta_max>0?'pos':t.delta_max<0?'neg':'zero'}">Δ${t.delta_max>=0?'+':''}${t.delta_max.toFixed(2)}</span>`
        : '';
      const hardChip = t.is_hard ? `<span class="trace-hard">HARD</span>` : '';
      return `
        <div class="trace-row" data-sid="${escapeHtml(t.session_id)}">
          <div class="trace-row-top">
            <span class="trace-tenant">${escapeHtml(tenant)}${cluster ? ' '+cluster : ''}</span>
            <span class="trace-time">${escapeHtml(dateShort)} ${escapeHtml(startedShort)}</span>
          </div>
          <div class="trace-row-mid">${escapeHtml(dl)}</div>
          <div class="trace-row-bot">
            ${hardChip}
            ${outcomeChip}
            ${deltaChip}
            <span class="trace-stat">${t.n_llm_calls ?? 0} LLM</span>
            <span class="trace-stat">${t.duration_s ?? '?'}s</span>
          </div>
        </div>`;
    }).join('');
    list.querySelectorAll('.trace-row').forEach(row => {
      row.addEventListener('click', () => {
        list.querySelectorAll('.trace-row').forEach(r => r.classList.remove('active'));
        row.classList.add('active');
        loadTraceDetail(row.dataset.sid);
      });
    });
    // Render the historical chart
    renderHistoryChart(traces);
  } catch (err) {
    list.innerHTML = `<div class="traces-empty">Failed to load: ${escapeHtml(err.message)}</div>`;
  }
}

function renderOutcomeChip(t) {
  const L = t.L_outcome; const R = t.R_outcome;
  const lcls = L === 'won' ? 'won' : L === 'lost' ? 'lost' : '';
  const rcls = R === 'won' ? 'won' : R === 'lost' ? 'lost' : '';
  return `<span class="outcome-mini">L<span class="${lcls}">${L?L[0].toUpperCase():'?'}</span>R<span class="${rcls}">${R?R[0].toUpperCase():'?'}</span></span>`;
}

// 2026-05-06 — Historical metric trends chart. Plots per-session L_max,
// R_max, delta, and (optional) message-length-median across all sessions
// chronologically. Vertical milestone lines mark POC feature-ship times.
let _historyChart = null;

// Hand-curated milestones from today + yesterday's POC iteration
// (timestamps approximate; adjust as more events ship)
const POC_MILESTONES = [
  { ts: '2026-05-04T14:28:00', label: 'Option C (cache) activated' },
  { ts: '2026-05-04T15:00:00', label: '3-of-3 close-guard' },
  { ts: '2026-05-05T11:08:00', label: 'Length fix v1' },
  { ts: '2026-05-05T11:19:00', label: 'Late-phase recovery' },
  { ts: '2026-05-05T11:57:00', label: 'Length fix v2 (regen+truncator)' },
  { ts: '2026-05-05T12:06:00', label: 'Counter-offer detector' },
  { ts: '2026-05-05T12:45:00', label: 'Tier 1+2 cache enrichment' },
  { ts: '2026-05-05T13:10:00', label: 'Cohort weights endpoint' },
  { ts: '2026-05-05T13:43:00', label: 'Cumulative-concession + R27' },
  { ts: '2026-05-05T14:38:00', label: 'Length fix v4 (substance-aware)' },
];

function renderHistoryChart(traces) {
  const canvas = document.getElementById('historyChart');
  if (!canvas) return;
  // Sort chronologically (oldest → newest)
  const data = traces
    .filter(t => t.started_at && (t.delta_max !== undefined || t.L_max !== undefined))
    .slice()
    .sort((a, b) => new Date(a.started_at) - new Date(b.started_at));
  if (!data.length) return;

  const labels = data.map(t => (t.started_at || '').slice(5, 16).replace('T', ' '));
  const lPeak = data.map(t => t.L_max ?? null);
  const rPeak = data.map(t => t.R_max ?? null);
  const delta = data.map(t => t.delta_max ?? null);
  const lengthMed = data.map(t => (t.r_msg_median_len != null) ? t.r_msg_median_len/50 : null);

  const showL = document.getElementById('chartShowLPeak')?.checked;
  const showR = document.getElementById('chartShowRPeak')?.checked;
  const showD = document.getElementById('chartShowDelta')?.checked;
  const showLen = document.getElementById('chartShowLength')?.checked;
  const showMs = document.getElementById('chartShowMilestones')?.checked;

  const datasets = [];
  if (showL) datasets.push({
    label: 'L peak persuasion', data: lPeak,
    borderColor: '#888', backgroundColor: 'rgba(136,136,136,0.1)',
    borderWidth: 2, tension: 0.2, spanGaps: true, pointRadius: 2,
  });
  if (showR) datasets.push({
    label: 'R peak persuasion', data: rPeak,
    borderColor: '#5BA3F5', backgroundColor: 'rgba(91,163,245,0.1)',
    borderWidth: 2, tension: 0.2, spanGaps: true, pointRadius: 2,
  });
  if (showD) datasets.push({
    label: 'Δ peak (R−L)', data: delta,
    borderColor: '#88e0a0', backgroundColor: 'rgba(136,224,160,0.15)',
    borderWidth: 2, borderDash: [4, 3], tension: 0.2, spanGaps: true, pointRadius: 2,
  });
  if (showLen) datasets.push({
    label: 'R msg median length (÷50)', data: lengthMed,
    borderColor: '#f5b87a', backgroundColor: 'rgba(245,184,122,0.1)',
    borderWidth: 2, borderDash: [2, 4], tension: 0.2, spanGaps: true, pointRadius: 2,
  });

  // Milestone vertical lines via Chart.js plugin (annotation)
  const milestoneLines = showMs ? POC_MILESTONES.flatMap(m => {
    // Find the closest x-index by timestamp
    const t = new Date(m.ts).getTime();
    let best = -1, bestDiff = Infinity;
    data.forEach((row, i) => {
      const ts = new Date(row.started_at).getTime();
      if (Math.abs(ts - t) < bestDiff && ts >= t - 30*60*1000) {
        bestDiff = Math.abs(ts - t); best = i;
      }
    });
    if (best < 0) return [];
    return [{
      type: 'line', xMin: best, xMax: best,
      borderColor: 'rgba(201,122,58,0.6)', borderWidth: 1, borderDash: [4, 4],
      label: { display: false },
    }];
  }) : [];

  if (_historyChart) _historyChart.destroy();
  _historyChart = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#c4d0e3', boxWidth: 12 } },
        tooltip: {
          callbacks: {
            title: (items) => {
              const idx = items[0].dataIndex; const t = data[idx];
              return `${t.started_at} · ${t.company || '?'}c${t.cluster_id ?? '?'}${t.is_hard ? ' (HARD)' : ''}`;
            },
            afterBody: (items) => {
              const idx = items[0].dataIndex; const t = data[idx];
              return [`outcome: L=${t.L_outcome || '?'} R=${t.R_outcome || '?'}`,
                      `r-msg ${t.r_msg_count} median=${t.r_msg_median_len ?? '?'}w`];
            },
          },
        },
      },
      scales: {
        x: { ticks: { color: '#888', maxRotation: 60, minRotation: 30, autoSkip: true,
                      maxTicksLimit: 16 } },
        y: { min: -1, max: 1.2, ticks: { color: '#888' },
             grid: { color: 'rgba(136,136,136,0.15)' } },
      },
    },
  });

  // Render legend with milestones
  const legendEl = document.getElementById('chartLegend');
  if (legendEl && showMs) {
    legendEl.innerHTML = '<b>Milestones (vertical lines):</b> ' +
      POC_MILESTONES.map(m => {
        const t = (m.ts || '').slice(5, 16).replace('T', ' ');
        return `<span class="milestone-tag">${t} ${escapeHtml(m.label)}</span>`;
      }).join(' · ');
  } else if (legendEl) {
    legendEl.innerHTML = '';
  }
}

async function loadTraceDetail(sessionId) {
  const detail = document.getElementById('tracesDetail');
  detail.innerHTML = 'Loading trace…';
  try {
    const r = await fetch(`/api/trace/${encodeURIComponent(sessionId)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const trace = await r.json();
    detail.innerHTML = renderTraceDetail(trace);
    detail.querySelectorAll('.event-card').forEach(c => {
      c.querySelector('.event-card-head')?.addEventListener('click', () => {
        c.classList.toggle('open');
      });
    });
  } catch (err) {
    detail.innerHTML = `<div class="traces-empty">Failed to load trace: ${escapeHtml(err.message)}</div>`;
  }
}

function renderTraceDetail(trace) {
  const summ = trace.summary || {};
  const events = trace.events || [];
  const scen = trace.scenario_meta || {};
  const llmRows = Object.entries(summ.llm_by_stage || {}).map(([k, v]) =>
    `<tr><td>${escapeHtml(k)}</td><td>${v.calls}</td><td>${v.latency_ms_total}ms</td>`
    + `<td>${v.input_tokens}</td><td>${v.output_tokens}</td></tr>`).join('');
  const cgRows = Object.entries(summ.cg_by_endpoint || {}).map(([k, v]) =>
    `<tr><td>${escapeHtml(k)}</td><td>${v.calls}</td><td>${v.latency_ms_total}ms</td>`
    + `<td>${v.cache_hits || 0}</td></tr>`).join('');
  const gateRows = Object.entries(summ.gate_by_name || {}).map(([k, v]) => {
    const verd = Object.entries(v.verdicts || {}).map(([vv, c]) => `${vv}×${c}`).join(', ');
    return `<tr><td>${escapeHtml(k)}</td><td>${v.firings}</td><td>${escapeHtml(verd)}</td></tr>`;
  }).join('');
  const summaryHtml = `
    <div class="trace-summary">
      <div class="trace-summary-head">
        <h3>${escapeHtml(scen.company || '?')} · ${escapeHtml(scen.opp_type || '')}</h3>
        <span class="muted-sm">${escapeHtml(trace.session_id || '')}</span>
        <span class="muted-sm">opp ${escapeHtml((trace.opp_id || '').slice(0, 8))}</span>
        <span class="muted-sm">${summ.duration_s || '?'}s · ${summ.n_events || 0} events</span>
      </div>
      <div class="trace-summary-grid">
        <div class="trace-summary-block">
          <h4>LLM calls by stage</h4>
          <table><thead><tr><th>stage</th><th>calls</th><th>latency</th><th>in tok</th><th>out tok</th></tr></thead>
          <tbody>${llmRows || '<tr><td colspan="5" class="muted-sm">none</td></tr>'}</tbody></table>
        </div>
        <div class="trace-summary-block">
          <h4>Context Graph calls</h4>
          <table><thead><tr><th>endpoint</th><th>calls</th><th>latency</th><th>cache hits</th></tr></thead>
          <tbody>${cgRows || '<tr><td colspan="4" class="muted-sm">none</td></tr>'}</tbody></table>
        </div>
        <div class="trace-summary-block">
          <h4>Gate firings</h4>
          <table><thead><tr><th>gate</th><th>firings</th><th>verdicts</th></tr></thead>
          <tbody>${gateRows || '<tr><td colspan="3" class="muted-sm">none</td></tr>'}</tbody></table>
        </div>
      </div>
    </div>`;
  const eventsHtml = events.map((ev, i) => renderEventCard(ev, i)).join('');
  return summaryHtml + `
    <div class="trace-events">
      <h3>Timeline (${events.length} events) — click any event to expand</h3>
      <div class="event-list">${eventsHtml}</div>
    </div>`;
}

function renderEventCard(ev, i) {
  const t = (ev.t ?? 0).toFixed(2);
  const kind = ev.kind || '?';
  let head = '', body = '';
  if (kind === 'llm_call') {
    head = `<span class="ek llm">LLM</span> ${escapeHtml(ev.stage || '?')} · ${escapeHtml(ev.provider || '?')}` +
      ` <span class="muted-sm">${ev.latency_ms || 0}ms · ${ev.input_tokens || 0}/${ev.output_tokens || 0} tok</span>`;
    body = `
      <div class="event-section"><b>system</b><pre>${escapeHtml(ev.raw_system || '')}</pre></div>
      <div class="event-section"><b>user</b><pre>${escapeHtml(ev.raw_user || '')}</pre></div>
      <div class="event-section"><b>response</b><pre>${escapeHtml(ev.raw_response || '')}</pre></div>`;
  } else if (kind === 'cg_call') {
    head = `<span class="ek cg">CG</span> ${escapeHtml(ev.endpoint || '?')} · ${escapeHtml(ev.workspace || '?')}` +
      ` <span class="muted-sm">${ev.latency_ms || 0}ms${ev.cache_hit ? ' · cache' : ''}</span>`;
    body = `
      <div class="event-section"><b>query</b><pre>${escapeHtml(ev.query || '')}</pre></div>
      <div class="event-section"><b>response</b><pre>${escapeHtml(JSON.stringify(ev.response || {}, null, 2))}</pre></div>`;
  } else if (kind === 'gate_event') {
    head = `<span class="ek gate">GATE</span> ${escapeHtml(ev.gate || '?')} · ${escapeHtml(ev.verdict || '?')}` +
      (ev.panel_side ? ` <span class="muted-sm">${escapeHtml(ev.panel_side)}</span>` : '');
    body = `<div class="event-section"><pre>${escapeHtml(JSON.stringify(ev, null, 2))}</pre></div>`;
  } else if (kind === 'chain_stage') {
    head = `<span class="ek stage">STAGE</span> ${escapeHtml(ev.stage || '?')}` +
      ` <span class="muted-sm">${ev.latency_ms || 0}ms</span>`;
    body = `<div class="event-section"><pre>${escapeHtml(JSON.stringify(ev, null, 2))}</pre></div>`;
  } else {
    head = `<span class="ek note">${escapeHtml(kind)}</span> ${escapeHtml(ev.msg || '')}`;
    body = `<div class="event-section"><pre>${escapeHtml(JSON.stringify(ev, null, 2))}</pre></div>`;
  }
  return `<div class="event-card" data-i="${i}">
    <div class="event-card-head"><span class="event-t">${t}s</span>${head}</div>
    <div class="event-card-body">${body}</div>
  </div>`;
}

document.addEventListener('DOMContentLoaded', () => {
  loadTraceList();
  document.getElementById('tracesRefreshBtn')?.addEventListener('click', loadTraceList);
  // Chart toggle checkboxes — re-render when any flips
  ['chartShowLPeak', 'chartShowRPeak', 'chartShowDelta',
   'chartShowLength', 'chartShowMilestones'].forEach((id) => {
    document.getElementById(id)?.addEventListener('change', () => {
      if (_tracesCache.length) renderHistoryChart(_tracesCache);
    });
  });
});
