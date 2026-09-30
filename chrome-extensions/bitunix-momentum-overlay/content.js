(() => {
  window.__bisIntradayTeardown?.();
  document.querySelectorAll('#bis-panel').forEach(node => node.remove());
  const host = document.createElement('aside');
  host.id = 'bis-panel';
  host.setAttribute('aria-label', 'Bitunix intraday signals');
  const version = (() => {
    try { return chrome.runtime.getManifest().version || ''; } catch { return ''; }
  })();
  if (version) host.dataset.bisVersion = version;
  document.documentElement.appendChild(host);
  let payload = null, selected = '', collapsed = false, formOpen = false, lastAlert = '', alertsInitialized = false;
  // Bitunix futures pages are /contract-trade/<SYMBOL>. Picking a card opens
  // that pair's chart; the pick is remembered so the reloaded panel shows it.
  const onBitunix = () => /(^|\.)bitunix\.com$/.test(location.hostname);
  const pageSymbol = () => (location.pathname.match(/^\/contract-trade\/([A-Z0-9]+)\/?$/i) || [])[1]?.toUpperCase() || '';
  const chartUrl = symbol => `${location.origin}/contract-trade/${encodeURIComponent(symbol)}`;
  function openChart(symbol) {
    if (!onBitunix() || !/^[A-Z0-9]{2,30}$/.test(symbol) || pageSymbol() === symbol) return false;
    try { chrome.storage?.local?.set({ pendingSymbol: symbol }); } catch { /* selection resets after reload */ }
    location.assign(chartUrl(symbol));
    return true;
  }
  const spokenEnter = new Set();
  const spokenRisk = new Set();
  let pendingSpeech = '';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const money = value => Number.isFinite(value) ? '$' + value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—';
  // "What does it need" line: the price levels the backend says would move a
  // card to its next state (breakout / pullback / trigger close / entry zone).
  const actionText = items => (Array.isArray(items) ? items : [])
    .filter(a => a && typeof a.label === 'string' && Number.isFinite(a.price))
    .map(a => `${a.label} ${price(a.price)}${Number.isFinite(a.price2) ? ' – ' + price(a.price2) : ''}`)
    .join(' · ');
  // "Next 5 min" line: the backend's drift-plus-ATR envelope for the next few
  // minutes, recomputed every scan. A volatility band, not a forecast.
  const NEXT_TITLE = 'Last hour of drift extrapolated over the next few minutes, with a band of one trigger-candle ATR either side. Recomputed every scan from the live price. A volatility envelope, not a forecast of direction.';
  const nextText = p => p && Number.isFinite(p.price) && Number.isFinite(p.low) && Number.isFinite(p.high)
    ? `Next ${Number.isFinite(p.horizon_minutes) ? p.horizon_minutes : 5} min ≈ ${price(p.price)} · ${price(p.low)} – ${price(p.high)}${Number.isFinite(p.drift_pct) ? ' · ' + (p.drift_pct >= 0 ? '+' : '') + fixed(p.drift_pct) + '%' : ''}`
    : '';
  const price = value => Number.isFinite(value) ? value.toLocaleString('en-US', { maximumFractionDigits: value < 1 ? 8 : value < 100 ? 5 : 2 }) : '—';
  const fixed = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—';
  // Both profiles scan long and short. A WAIT card says which kind of move the
  // profile is waiting for so the state is never a mystery.
  const PROFILES = {
    trend: { holds: [1, 2], min: 20, max: 100, label: 'Trend · long / short', option: 'Trend · long and short · 1-2h · 20-100x', path: '1h bias → 15m structure → 5m entry', wait: 'Trend continuation, long or short. No aligned 1h/15m setup yet: ' },
    scalp: { holds: [1, 2], min: 1, max: 125, label: 'Scalp · long / short', option: 'Scalp · long and short · 1-2h · 1-125x', path: 'Extended → climax → crowded → failed high / low', wait: 'High-leverage scalp, long or short. No exhaustion move to fade yet: ' },
  };
  // Saved rows from older releases still say "swing" / "scalp_short".
  const ALIASES = { swing: 'trend', scalp_short: 'scalp' };
  const profileOf = value => PROFILES[ALIASES[value] || value] ? (ALIASES[value] || value) : 'trend';
  const isScalp = value => profileOf(value) === 'scalp';
  let profileName = 'trend';
  const label = value => String(value || 'WAIT').replaceAll('_', ' ');
  const waitHint = state => state === 'WAIT' ? PROFILES[profileName].wait : '';
  const tone = state => state === 'SET_STOP' ? 'stop' : state?.startsWith('EXIT') || state?.startsWith('CLOSE') || state === 'CONSIDER_CLOSE' ? 'exit' : state?.includes('LONG') ? 'long' : state?.includes('SHORT') ? 'short' : 'wait';
  const ago = timestamp => timestamp ? Math.max(0, Math.floor(Date.now() / 1000 - timestamp)) + 's ago' : 'Waiting for data';
  // When the worker last received a snapshot from the dashboard (ms). Lets the
  // panel say whether it is showing stale data even if the backend is fine.
  let polledAt = 0;
  const agoMs = ms => ms ? Math.max(0, Math.floor((Date.now() - ms) / 1000)) + 's ago' : 'not yet';
  const checkedAt = row => row?.evaluated_at || row?.as_of || 0;
  const gateSummary = row => {
    const checks = Array.isArray(row?.checks) ? row.checks : [];
    const passed = checks.filter(c => c.passed).length;
    const blocker = checks.find(c => !c.passed && !c.waiting) || checks.find(c => !c.passed);
    return { passed, total: checks.length, blocker: blocker?.label || '' };
  };
  const clock = timestamp => Number.isFinite(timestamp) && timestamp > 0
    ? new Date(timestamp * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '—';
  const remain = seconds => {
    if (!Number.isFinite(seconds) || seconds < 0) return '';
    const whole = Math.floor(seconds);
    const minutes = Math.floor(whole / 60);
    return minutes > 0 ? `${minutes}m ${whole % 60}s` : `${whole}s`;
  };
  function fresh(row) { return !payload?.error && row?.as_of > 0 && Date.now() / 1000 - row.as_of <= (payload?.data_max_age || 60); }
  function actionable(row) { return fresh(row) && row?.state?.startsWith('ENTER_') && row.plan?.expires_at > Date.now() / 1000; }
  function enterKey(row) {
    return row?.signal_id || `${row?.symbol}:${row?.state}:${row?.bar_time || row?.as_of || ''}`;
  }
  function speak(text) {
    pendingSpeech = text;
    try { window.__bisLastSpeak = text; } catch { /* tests */ }
    try {
      const synth = window.speechSynthesis;
      if (!synth) return;
      synth.cancel();
      let utterance = { text, rate: 1, volume: 1 };
      try {
        if (typeof SpeechSynthesisUtterance === 'function') {
          utterance = new SpeechSynthesisUtterance(text);
          utterance.rate = 1;
          utterance.volume = 1;
        }
      } catch { /* headless or mocked speech uses the plain object */ }
      synth.speak(utterance);
    } catch { /* laptop speech is best-effort */ }
  }
  function announceNewEntries() {
    if (!payload?.symbols || payload.error) return;
    const rows = Object.values(payload.symbols).filter(row => fresh(row) && row.state?.startsWith('ENTER_') && row.plan);
    if (!alertsInitialized) {
      rows.forEach(row => spokenEnter.add(enterKey(row)));
      return;
    }
    for (const row of rows) {
      const key = enterKey(row);
      if (spokenEnter.has(key)) continue;
      spokenEnter.add(key);
      const side = row.state.includes('SHORT') ? 'short' : 'long';
      const market = String(row.symbol || '').replace(/USDT$/i, ' U S D T');
      const settings = payload.settings || {};
      const sizing = Number.isFinite(settings.leverage) && Number.isFinite(settings.hold_hours)
        ? ` ${settings.leverage} x, ${settings.hold_hours * 60} minute hold.`
        : '';
      speak(`Trade entry waiting. ${market}. Enter ${side}.${sizing}`);
    }
  }
  function announceRisk() {
    if (!payload?.trades) return;
    for (const trade of payload.trades) {
      const suggestion = trade.suggestion || trade.state || '';
      const urgent = suggestion === 'SET_STOP' || suggestion.startsWith('CLOSE_') || String(trade.state || '').startsWith('EXIT_');
      if (!urgent) continue;
      const key = `${trade.id}:${suggestion}`;
      if (spokenRisk.has(key)) continue;
      spokenRisk.add(key);
      const market = String(trade.symbol || '').replace(/USDT$/i, ' U S D T');
      if (suggestion === 'SET_STOP') {
        speak(`Set the Bitunix stop now. ${market}. Stop at ${trade.current_stop}. Do not wait for a reversal.`);
      } else {
        speak(`Close the trade now. ${market}. Do not wait for a reversal.`);
      }
    }
  }
  const connectionError = 'Extension connection interrupted. Reload the extension, then this Bitunix tab.';
  async function send(type, body) {
    let timer;
    try {
      return await Promise.race([
        chrome.runtime.sendMessage({ type, body }),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(connectionError)), 12000); }),
      ]);
    } finally { clearTimeout(timer); }
  }
  host.innerHTML = `<header title="Drag the title or grip to move. Double-click or use Reset to restore the default position."><div class="bis-drag"><span class="bis-grip" aria-hidden="true"></span><div><span class="bis-eyebrow">BITUNIX · INTRADAY${version ? ' · v' + version : ''}</span><strong>Trade signals</strong></div></div><div class="bis-actions"><button data-action="reset-layout" title="Reset size and position" aria-label="Reset size and position">⤢</button><button data-action="settings" title="Connection settings" aria-label="Connection settings">⚙</button><button data-action="collapse" aria-label="Collapse panel">−</button></div></header><div id="bis-body"><div id="bis-status" role="status"></div><div id="bis-planning"></div><div id="bis-handoff" hidden></div><div id="bis-queue"></div><div id="bis-selection"></div><div id="bis-card"></div><div id="bis-trades"></div><details id="bis-scan-wrap" open><summary>Scan feed · when each coin was last checked</summary><div id="bis-scan"></div></details><details id="bis-history-wrap"><summary>Recent alerts</summary><div id="bis-history"></div></details><details><summary>Recorded closures</summary><div id="bis-closed"></div></details><footer>Alerts only except the Set Bitunix stop button, which places a position-level stop. It does not open or close trades.<br>Speakers say “Trade entry waiting” on ENTER and “Close the trade now” or “Set the Bitunix stop” on exits. Click the panel once if Chrome blocks speech.<br>Do not wait for a reversal without an exchange stop.</footer></div><div id="bis-form"></div><div class="bis-resize" role="separator" aria-orientation="horizontal" aria-label="Resize panel" title="Drag the corner to resize"></div>`;
  const MIN_W = 280, MIN_H = 200, EDGE = 8;
  let layout = null;
  function box() {
    const rect = host.getBoundingClientRect();
    return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
  }
  function clamp(next) {
    const maxW = Math.max(MIN_W, window.innerWidth - EDGE * 2);
    const maxH = Math.max(MIN_H, window.innerHeight - EDGE * 2);
    const width = Math.min(Math.max(next.width, MIN_W), maxW);
    const height = Math.min(Math.max(next.height, MIN_H), maxH);
    return {
      left: Math.min(Math.max(next.left, EDGE), window.innerWidth - width - EDGE),
      top: Math.min(Math.max(next.top, EDGE), window.innerHeight - height - EDGE),
      width,
      height,
    };
  }
  function persistLayout() {
    if (!layout) return;
    try { chrome.storage?.local?.set({ panelLayout: layout }); } catch { /* keep the in-memory layout */ }
  }
  function applyLayout(next, persist) {
    layout = clamp(next);
    host.classList.add('bis-placed');
    host.style.left = layout.left + 'px';
    host.style.top = layout.top + 'px';
    host.style.right = 'auto';
    host.style.width = collapsed ? '230px' : layout.width + 'px';
    host.style.height = collapsed ? 'auto' : layout.height + 'px';
    host.style.maxWidth = 'none';
    host.style.maxHeight = 'none';
    if (persist) persistLayout();
  }
  function resetLayout() {
    layout = null;
    host.classList.remove('bis-placed');
    host.style.left = host.style.top = host.style.right = '';
    host.style.width = host.style.height = host.style.maxWidth = host.style.maxHeight = '';
    try { chrome.storage?.local?.remove('panelLayout'); } catch { /* default CSS position */ }
  }
  const listeners = [];
  const timers = [];
  function bindDrag(target, onMove) {
    if (!target) return;
    const onDown = event => {
      if (event.button && event.button !== 0) return;
      const hit = event.target;
      if (!(hit instanceof Node) || !target.contains(hit)) return;
      if (hit.closest?.('button, a, input, select, textarea, summary, option')) return;
      if (formOpen) return;
      event.preventDefault();
      event.stopPropagation();
      const start = box();
      const pointer = { x: event.clientX, y: event.clientY };
      host.classList.add('bis-dragging');
      try { target.setPointerCapture(event.pointerId); } catch { /* keep window listeners */ }
      const move = ev => {
        ev.preventDefault();
        ev.stopPropagation();
        onMove(start, ev.clientX - pointer.x, ev.clientY - pointer.y);
      };
      const up = ev => {
        ev.stopPropagation();
        host.classList.remove('bis-dragging');
        window.removeEventListener('pointermove', move, true);
        window.removeEventListener('pointerup', up, true);
        window.removeEventListener('pointercancel', up, true);
        if (layout) persistLayout();
      };
      window.addEventListener('pointermove', move, true);
      window.addEventListener('pointerup', up, true);
      window.addEventListener('pointercancel', up, true);
    };
    // Window capture runs before Bitunix document listeners that stopPropagation.
    window.addEventListener('pointerdown', onDown, true);
    listeners.push(() => window.removeEventListener('pointerdown', onDown, true));
  }
  bindDrag(host.querySelector('header'), (start, dx, dy) => {
    applyLayout({ left: start.left + dx, top: start.top + dy, width: start.width, height: start.height });
  });
  bindDrag(host.querySelector('.bis-resize'), (start, dx, dy) => {
    applyLayout({ left: start.left, top: start.top, width: start.width + dx, height: start.height + dy });
  });
  host.querySelector('header').addEventListener('dblclick', event => {
    if (!event.target.closest('button')) resetLayout();
  });
  const onViewport = () => { if (layout) applyLayout(layout, false); };
  window.addEventListener('resize', onViewport);
  listeners.push(() => window.removeEventListener('resize', onViewport));
  try {
    const remembered = chrome.storage?.local?.get?.('pendingSymbol');
    if (remembered && typeof remembered.then === 'function') {
      remembered.then(stored => {
        const symbol = stored?.pendingSymbol;
        if (typeof symbol !== 'string') return;
        chrome.storage?.local?.remove?.('pendingSymbol');
        if (symbol && symbol === pageSymbol()) { selected = symbol; if (payload) render(); }
      }).catch(() => {});
    }
  } catch { /* start on the best setup */ }
  try {
    const pending = chrome.storage?.local?.get?.('panelLayout');
    if (pending && typeof pending.then === 'function') {
      pending.then(stored => {
        const saved = stored?.panelLayout;
        if (saved && [saved.left, saved.top, saved.width, saved.height].every(Number.isFinite)) applyLayout(saved, false);
      }).catch(() => {});
    }
  } catch { /* default CSS position */ }
  function render() {
    try { renderContent(); }
    catch {
      // Discard an incompatible response instead of leaving a partially drawn entry.
      payload = { error: 'Signal data is incomplete or incompatible. Update the backend, then retry.' };
      formOpen = false;
      host.querySelector('#bis-form').replaceChildren();
      renderContent();
    }
  }
  function renderContent() {
    if (formOpen) {
      const exit = payload?.trades?.find(t => t.state.startsWith('EXIT_'));
      host.querySelector('.bis-form-live').textContent = payload?.error || (exit ? `${exit.symbol}: ${label(exit.state)} — ${exit.reason}` : '');
      return;
    }
    const checksOpen = host.querySelector('#bis-card details')?.open || false;
    const openTradeChecks = new Set([...host.querySelectorAll('#bis-trades article[data-id] details')].filter(node => node.open).map(node => node.closest('article')?.dataset.id).filter(Boolean));
    const status = host.querySelector('#bis-status');
    if (!payload || (payload.error && !payload.symbols)) {
      status.className = 'bis-notice'; status.textContent = payload?.error || 'Connecting to your signal scanner…';
      for (const id of ['card', 'selection', 'planning', 'queue', 'trades', 'scan', 'history', 'closed']) host.querySelector('#bis-' + id).replaceChildren();
      const handoffEmpty = host.querySelector('#bis-handoff');
      if (handoffEmpty) { handoffEmpty.hidden = true; handoffEmpty.replaceChildren(); }
      host.querySelector('#bis-card').innerHTML = '<div class="bis-buttons"><button data-action="settings">Open Settings</button><button data-action="refresh">Retry connection</button></div><p class="bis-empty">Signals need a running intraday backend and its dashboard password. Use Save and test connection in Settings.</p>';
      return;
    }
    const settings = payload.settings;
    profileName = profileOf(settings?.profile);
    const liveCount = (payload.trades || []).filter(t => t.kind === 'exchange' || t.exchange_position_id).length;
    const liveBit = liveCount ? ` · ${liveCount} live position${liveCount === 1 ? '' : 's'}` : '';
    status.className = payload.error ? 'bis-notice' : 'bis-status';
    const scan = payload.scan;
    const scanBit = scan && Number.isFinite(scan.hot) && Number.isFinite(scan.universe)
      ? ` · ${scan.hot} hot / ${scan.universe} universe`
      : '';
    const ft = payload.forward_test;
    const ftBit = ft && Number.isFinite(ft.count) && ft.count > 0
      ? ` · forward test ${ft.resolved}/${ft.count}${Number.isFinite(ft.hit_rate) ? ' · hit ' + Math.round(ft.hit_rate * 100) + '%' : ''}${Number.isFinite(ft.average_r) ? ' · avg ' + ft.average_r.toFixed(2) + 'R' : ''}${ft.liquidation_touches ? ' · ' + ft.liquidation_touches + ' liq' : ''}`
      : '';
    const freshBit = scan && scan.last_scan ? ` · last scan ${ago(scan.last_scan)}` : '';
    const polledBit = polledAt ? ` · panel updated ${agoMs(polledAt)}` : '';
    status.textContent = payload.error || payload.status?.error || (payload.status?.ready ? '● Monitoring liquid USDT perpetuals' + scanBit + freshBit + polledBit + liveBit + ftBit : 'Waiting for complete market data');
    host.querySelector('#bis-planning').innerHTML = `<div><small>Planning equity</small><b>${money(settings.planning_equity)}</b></div><div><small>Risk / trade</small><b>${fixed(settings.risk_pct)}%</b></div><div><small>Leverage / hold</small><b>${settings.leverage}x · ≤${settings.hold_hours * 60} min</b></div><div><small>Profile</small><b>${PROFILES[profileOf(settings.profile)].label}</b></div><button data-action="planning">Edit</button>`;
    const rows = Object.values(payload.symbols || {});
    if (selected && !payload.symbols[selected]) selected = '';
    const symbol = selected && payload.symbols[selected] ? selected : payload.best_symbol;
    const row = payload.symbols[symbol];
    const nowSec = Date.now() / 1000;
    const handoff = payload.handoff;
    const handoffBox = host.querySelector('#bis-handoff');
    if (handoff && handoff.to_symbol && Number(handoff.expires_at) > nowSec) {
      const left = remain(handoff.expires_at - nowSec);
      handoffBox.hidden = false;
      handoffBox.className = 'bis-handoff';
      handoffBox.innerHTML = `<strong>Switching to ${esc(handoff.to_symbol)}</strong><span>${esc(left)} left</span><small>${esc(handoff.reason || 'Signal change warning')} · now ${esc(handoff.from_symbol || symbol || '')}</small>`;
    } else {
      handoffBox.hidden = true;
      handoffBox.replaceChildren();
    }
    const queue = Array.isArray(payload.queue) && payload.queue.length
      ? payload.queue
      : rows.slice(0, 5).map(item => ({
        symbol: item.symbol, state: item.state, side: item.side, setup: item.setup,
        as_of: item.as_of, state_since: item.state_since, expires_at: item.plan?.expires_at,
        reason: item.reasons?.[0] || '', price: item.price, actions: item.actions,
      }));
    host.querySelector('#bis-queue').innerHTML = `<h3>Top setups <span>${Math.min(queue.length, 5)}</span></h3>${queue.slice(0, 5).map((item, index) => {
      const live = fresh(payload.symbols?.[item.symbol]) ? item.state : 'WAIT';
      const left = item.expires_at && live.startsWith('ENTER_') ? remain(item.expires_at - nowSec) : '';
      return `<button type="button" class="bis-queue-row ${tone(live)}${item.symbol === symbol ? ' active' : ''}" data-action="pick" data-symbol="${esc(item.symbol)}" title="Show this card and open the ${esc(item.symbol)} chart on Bitunix"><b>${index + 1}</b><div><strong>${esc(item.symbol)}</strong><small>${esc(label(live))}${item.setup ? ' · ' + esc(item.setup) : ''}</small><small>Checked ${esc(ago(checkedAt(item)))} · ${esc(label(live).toLowerCase())} since ${esc(clock(item.state_since || item.as_of))}${left ? ' · ' + left + ' left' : ''}</small>${fresh(payload.symbols?.[item.symbol]) && actionText(item.actions) ? `<small class="bis-need">${esc(actionText(item.actions))}</small>` : ''}${fresh(payload.symbols?.[item.symbol]) && nextText(item.projection) ? `<small class="bis-next" title="${esc(NEXT_TITLE)}">${esc(nextText(item.projection))}</small>` : ''}</div><span>${price(item.price)}</span></button>`;
    }).join('') || '<p class="bis-empty">No ranked setups yet.</p>'}`;
    if (!host.querySelector('#bis-symbol:focus')) host.querySelector('#bis-selection').innerHTML = `<label>Market <select id="bis-symbol"><option value="">Best setup</option>${rows.map(r => `<option value="${esc(r.symbol)}" ${selected === r.symbol ? 'selected' : ''}>${esc(r.symbol)} · ${esc(fresh(r) ? label(r.state) : 'WAIT')}</option>`).join('')}</select></label><button data-action="refresh" title="Refresh signals">↻</button>`;
    if (row) {
      const state = fresh(row) && (!row.plan || row.plan.expires_at > Date.now() / 1000) ? row.state : 'WAIT';
      const plan = row.plan;
      const expiryLeft = plan && state.startsWith('ENTER_') ? remain(plan.expires_at - nowSec) : '';
      const checkGroups = [
        ['market', 'Market'], ['setup', 'Setup'], ['plan', 'Plan'], ['portfolio', 'Book'],
      ].map(([key, title]) => {
        const items = (row.checks || []).filter(c => (c.group || (key === 'market' ? 'market' : '')) === key);
        if (!items.length) return '';
        const pass = items.filter(c => c.passed).length;
        const waiting = items.filter(c => c.waiting && !c.passed).length;
        const scored = items.length - waiting;
        const heading = waiting === items.length
          ? `${title} waiting`
          : waiting
            ? `${title} ${pass}/${scored} scored · ${waiting} waiting`
            : `${title} ${pass}/${items.length}`;
        return `<li class="bis-check-group">${esc(heading)}</li>` + items.map(c => {
          const kind = c.passed ? 'pass' : c.waiting ? 'wait' : 'fail';
          const mark = c.passed ? '✓' : c.waiting ? '…' : '○';
          return `<li class="${kind}"><span>${mark}</span><div><b>${esc(c.label)}</b><small>${esc(c.detail)}</small></div></li>`;
        }).join('');
      }).join('');
      host.querySelector('#bis-card').innerHTML = `<article class="bis-signal ${tone(state)}"><div class="bis-row"><strong>${esc(row.symbol)}</strong><span>${price(row.price)}${onBitunix() && pageSymbol() !== row.symbol ? ` <button type="button" class="bis-chart" data-action="chart" data-symbol="${esc(row.symbol)}" title="Open the ${esc(row.symbol)} chart on Bitunix">Open chart ↗</button>` : ''}</span></div><div class="bis-state" aria-live="polite">${esc(label(state))}</div>${fresh(row) && actionText(row.actions) ? `<div class="bis-need">${esc(actionText(row.actions))}</div>` : ''}${fresh(row) && nextText(row.projection) ? `<div class="bis-next" title="${esc(NEXT_TITLE)}">${esc(nextText(row.projection))}</div>` : ''}<p>${esc(fresh(row) ? waitHint(state) + (row.reasons?.[0] || '') : 'Data is stale. Entry alerts are paused.')}</p><div class="bis-meta">${esc(row.setup || PROFILES[profileOf(settings.profile)].path)}<br>In this state since ${esc(clock(row.state_since || row.as_of))} · ${esc(ago(row.state_since || row.as_of))}<br>${(() => { const g = gateSummary(row); const at = checkedAt(row); const late = scan?.refresh_seconds && at && nowSec - at > 2 * scan.refresh_seconds; return `Checklist run ${esc(ago(at))}${g.total ? ` · ${g.passed}/${g.total} gates passed` : ''}${late ? ' · <b class="bis-late">not re-checked in the latest scan</b>' : ''}`; })()}<br>Market data ${esc(clock(row.as_of))} · ${esc(ago(row.as_of))}</div>${plan ? `<div class="bis-levels"><div><small>Entry zone</small><b>${price(plan.entry_low)} – ${price(plan.entry_high)}</b></div><div><small>Stop loss</small><b>${price(plan.stop)}</b></div><div><small>Profit target</small><b>${price(plan.target)}</b></div><div><small>Net reward / risk</small><b>${fixed(plan.net_reward_risk)}R</b></div><div><small>Planned loss incl. costs</small><b>${money(plan.risk_usdt)} · ${fixed(plan.risk_pct)}%</b></div><div><small>Notional / margin</small><b>${money(plan.notional)} / ${money(plan.margin)}</b></div><div><small>Leverage / hold</small><b>${plan.leverage}x · ≤${plan.hold_hours * 60} min</b></div></div><div class="bis-meta">Entry expires ${esc(clock(plan.expires_at))}${expiryLeft ? ' · ' + expiryLeft + ' left' : ''} · Planned ${plan.leverage}x · hold ≤ ${plan.hold_hours * 60} min · time-stop ${plan.hold_hours * 60} min after fill · Estimated leverage ceiling ${plan.max_leverage}x · stop ${fixed(plan.stop_pct)}% inside est. liquidation ${price(plan.liquidation_estimate)}</div><div class="bis-buttons"><button data-action="paper" ${actionable(row) ? '' : 'disabled'}>Track paper trade</button><button data-action="manual" ${actionable(row) ? '' : 'disabled'}>Record my fill</button></div>` : ''}<details><summary>Why this signal · ${row.checks.filter(c => c.passed).length}/${row.checks.length} checks</summary><p class="bis-empty">Same ${row.checks.length} gates on every card. Waiting gates are locked until the prior stage prints. Failed gates were scored and blocked the entry.</p><ul class="bis-checks">${checkGroups}</ul><div class="bis-meta">${isScalp(settings.profile) ? `1h ${fixed(row.metrics.gain_1h_pct)}% · 4h ${fixed(row.metrics.gain_4h_pct)}% · ${fixed(row.metrics.extension_atr, 1)} ATR above 1h EMA20 · climax ${fixed(row.metrics.climax_volume, 1)}× · OI ${row.metrics.oi_change_pct == null ? 'warming up' : fixed(row.metrics.oi_change_pct) + '%'}<br>4h bias: ${esc(row.metrics.trend_4h || '—')} · 1h structure: ${esc(row.metrics.trend_1h || '—')}<br>ATR: ${fixed(row.metrics.atr_pct)}% · 1h ATR: ${fixed(row.metrics.hourly_atr_pct)}% · Volume: ${fixed(row.metrics.relative_volume)}×<br>UTC session VWAP: ${price(row.metrics.vwap)}<br>BTC relative strength (6h): ${fixed(row.metrics.relative_strength_pct)}%<br>` : `1h bias: ${esc(row.metrics.trend_1h || '—')} · 15m structure: ${esc(row.metrics.trend_15m || '—')} · ext ${fixed(row.metrics.extension_atr, 1)} ATR · session vol ${fixed(row.metrics.session_volume_ratio, 2)}× · OI 1h ${row.metrics.oi_change_pct == null ? 'warming up' : fixed(row.metrics.oi_change_pct) + '%'}<br>ATR 5m: ${fixed(row.metrics.atr_pct)}% · 15m: ${fixed(row.metrics.atr_15m_pct)}% · 1h: ${fixed(row.metrics.hourly_atr_pct)}% · Volume: ${fixed(row.metrics.relative_volume)}×<br>UTC session VWAP: ${price(row.metrics.vwap)}<br>BTC relative strength (2h): ${fixed(row.metrics.relative_strength_pct)}%<br>`}Funding / interval: ${fixed(row.metrics.funding_rate_pct, 4)}%<br>Open interest: ${row.metrics.open_interest == null ? 'Unavailable' : price(row.metrics.open_interest)}</div>${plan ? `<p class="bis-meta">Estimated liquidation: ${price(plan.liquidation_estimate)}. Isolated margin, no extra collateral; verify on Bitunix. Estimated total costs ${fixed(plan.cost_pct)}%, including ${plan.funding_payments} projected funding payments. Future rates can change. Target 2 (context only): ${price(plan.target2)}. Stop loss = ${fixed(100 * plan.risk_usdt / plan.margin, 0)}% of posted margin at ${plan.leverage}x.</p>` : ''}</details></article>`;
    } else host.querySelector('#bis-card').innerHTML = '<p class="bis-empty">Scanner warming up. Missing data blocks entries.</p>';
    const kindLabel = kind => kind === 'paper' ? 'PAPER' : kind === 'exchange' ? 'LIVE' : 'USER RECORDED';
    const pnlText = value => Number.isFinite(value) ? `${value >= 0 ? '+' : '−'}${money(Math.abs(value))}` : '';
    const tradesEmpty = payload.positions?.error
      || (payload.positions?.connected === false
        ? 'Add Bitunix API keys on the backend to import live positions. You can still record a fill.'
        : 'No open Bitunix positions. Record a fill to track a paper or manual trade.');
    function tradeArticle(raw) {
      const t = payload.error && !raw.state.startsWith('EXIT_')
        ? { ...raw, state: 'REVIEW', suggestion: 'REVIEW', reason: 'Connection unavailable. Check Bitunix; this is the last recorded plan.', hold_confidence: null }
        : raw;
      const suggestion = t.suggestion || t.state;
      const holdChecks = Array.isArray(t.checks) ? t.checks : [];
      const holdPass = holdChecks.filter(item => item.passed).length;
      const conf = Number.isFinite(t.hold_confidence) ? t.hold_confidence : (holdChecks.length ? Math.round(100 * holdPass / holdChecks.length) : null);
      const mustClose = suggestion.startsWith('CLOSE_') || String(t.state || '').startsWith('EXIT_');
      const needsStop = !mustClose && suggestion === 'SET_STOP';
      const banner = mustClose
        ? `<div class="bis-get-out" role="alert"><strong>Close on Bitunix now</strong><span>Do not wait for a reversal. Overlay alerts cannot prevent liquidation.</span><b>Working stop ${price(t.current_stop)}</b></div>`
        : needsStop
        ? `<div class="bis-get-out" role="alert"><strong>Set the Bitunix stop</strong><span>One click sends this stop to Bitunix. It closes the whole position at market if hit.</span><b>${price(t.current_stop)}</b></div>`
        : '';
      const groups = [['risk', 'Risk'], ['structure', 'Structure'], ['tape', 'Tape'], ['cost', 'Cost']].map(([key, title]) => {
        const items = holdChecks.filter(item => (item.group || 'risk') === key);
        if (!items.length) return '';
        const pass = items.filter(item => item.passed).length;
        return `<li class="bis-check-group">${esc(title)} ${pass}/${items.length}</li>` + items.map(item => `<li class="${item.passed ? 'pass' : 'fail'}"><span>${item.passed ? '✓' : '○'}</span><div><b>${esc(item.label)}</b><small>${esc(item.detail)}</small></div></li>`).join('');
      }).join('');
      const confidence = mustClose || needsStop || conf == null
        ? ''
        : `<div class="bis-confidence"><small>Hold confidence</small><b>${conf}%</b><div class="bis-confidence-bar" role="meter" aria-label="Hold confidence" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${conf}"><i style="width:${conf}%"></i></div><small>Checklist score from the close-out gates. Not a measured win rate.</small></div>`;
      const stopButton = needsStop
        ? `<div class="bis-buttons"><button data-action="place-stop" data-id="${esc(t.id)}">Set Bitunix stop at ${price(t.current_stop)}</button></div><button data-action="confirm-stop" data-id="${esc(t.id)}">I already placed it</button>`
        : '';
      return `<article class="bis-trade ${tone(suggestion)}" data-id="${esc(t.id)}"><div class="bis-row"><b>${esc(t.symbol)} · ${esc(t.plan.side.toUpperCase())}</b><small>${kindLabel(t.kind)}</small></div><small class="bis-live">LIVE SUGGESTION${t.exchange_stop_confirmed ? ' · STOP ON BITUNIX' : ''}</small><strong class="bis-trade-state" aria-live="polite">${esc(label(suggestion))}</strong>${banner}${confidence}<p>${esc(t.reason)}</p><div class="bis-meta">Entry ${price(t.plan.entry)}${Number.isFinite(t.mark_price) ? ' · Mark ' + price(t.mark_price) : ''} · Stop ${price(t.current_stop)} · Target ${price(t.plan.target)}${Number.isFinite(t.unrealized_pnl) ? '<br>Unrealized ' + pnlText(t.unrealized_pnl) : ''}<br>${t.plan.hold_hours <= 2 ? `Held ${Math.round((Date.now() / 1000 - t.opened_at) / 60)} min / ${t.plan.hold_hours * 60} min max` : `Held ${fixed((Date.now() / 1000 - t.opened_at) / 3600, 1)}h / ${t.plan.hold_hours}h max`}${(() => { const at = t.evaluated_at || t.checked_at; if (!at) return ''; const late = scan?.refresh_seconds && nowSec - at > 2 * scan.refresh_seconds; return `<br>Checked ${esc(ago(at))}${late ? ' · <b class="bis-late">not re-checked in the latest scan</b>' : ''}`; })()}</div>${holdChecks.length ? `<details${openTradeChecks.has(t.id) ? ' open' : ''}><summary>Hold / close checks · ${holdPass}/${holdChecks.length}</summary><p class="bis-empty">Same 13 gates on every open trade. Failures are reasons to close or review, not exchange orders.</p><ul class="bis-checks">${groups}</ul></details>` : ''}${stopButton}<button data-action="close" data-id="${esc(t.id)}">Record closure</button></article>`;
    }
    host.querySelector('#bis-trades').innerHTML = `<h3>Tracked trades <span>${payload.trades.length}</span></h3>${payload.trades.length ? payload.trades.map(tradeArticle).join('') : `<p class="bis-empty">${esc(tradesEmpty)}</p>`}`;
    {
      const evaluated = new Set(Array.isArray(scan?.evaluated) ? scan.evaluated : []);
      const maxAge = Number.isFinite(payload.data_max_age) ? payload.data_max_age : 60;
      const feedRows = rows.slice().sort((a, b) => checkedAt(b) - checkedAt(a));
      const nextIn = scan?.next_scan ? Math.max(0, Math.ceil(scan.next_scan - nowSec)) : null;
      const head = `<p class="bis-scan-head">${scan?.last_scan ? `Last scan <b>${esc(ago(scan.last_scan))}</b>` : 'Scan time unknown'}${nextIn != null ? ` · next in ~${nextIn}s` : ''}${evaluated.size ? ` · ${evaluated.size} coins re-checked` : ''} · panel updated <b>${esc(agoMs(polledAt))}</b>${scan?.refresh_seconds ? `<br>Every ${scan.refresh_seconds}s the backend re-runs all gates for hot coins and a rotating batch of the rest. A coin shown here is only as current as its last check.` : ''}</p>`;
      host.querySelector('#bis-scan').innerHTML = head + (feedRows.map(item => {
        const g = gateSummary(item);
        const at = checkedAt(item);
        const age = at ? nowSec - at : Infinity;
        const cls = age > maxAge ? 'stale' : (scan?.refresh_seconds && age > 2 * scan.refresh_seconds) ? 'late' : 'ok';
        const live = fresh(item) ? item.state : 'WAIT';
        const verdict = live.startsWith('ENTER_') ? 'all gates passed' : g.blocker ? `blocked by ${g.blocker}` : '';
        return `<div class="bis-scan-row ${cls} ${tone(live)}${item.symbol === symbol ? ' active' : ''}"><time>${esc(at ? ago(at) : 'never')}</time><div><b>${esc(item.symbol)}</b> · ${esc(label(live))}${g.total ? ` · ${g.passed}/${g.total} gates` : ''}${evaluated.has(item.symbol) ? ' · <i>in latest scan</i>' : ''}<small>${esc(verdict)}${verdict && item.as_of ? ' · ' : ''}${item.as_of ? 'market data ' + esc(ago(item.as_of)) : ''}</small></div></div>`;
      }).join('') || '<p class="bis-empty">No coins evaluated yet.</p>');
    }
    host.querySelector('#bis-history').innerHTML = payload.history.slice(0, 30).map(event => `<div class="bis-history-row"><div><b>${esc(event.symbol)}</b> · ${esc(label(event.state))}${event.setup ? ' · ' + esc(event.setup) : ''}<small>${esc(event.reason)}</small></div><time datetime="${esc(event.time ? new Date(event.time * 1000).toISOString() : '')}">${esc(clock(event.time))}<small>${esc(ago(event.time))}</small></time></div>`).join('') || '<p class="bis-empty">WATCH and ENTER alerts appear here with timestamps.</p>';
    host.querySelector('#bis-closed').innerHTML = payload.closed_trades.slice(0, 10).map(t => `<div class="bis-history-row"><div><b>${esc(t.symbol)}</b> · ${esc(t.kind)}<small>Recorded exit ${price(t.exit_price)} · estimated net</small></div><b>${money(t.estimated_net_pnl)}</b></div>`).join('') || '<p class="bis-empty">No recorded closures. Signal alerts are not completed trades.</p>';
    if (host.querySelector('#bis-card details')) host.querySelector('#bis-card details').open = checksOpen;
    announceNewEntries();
    announceRisk();
    const alert = payload.history[0];
    if (alert && alert.id !== lastAlert) {
      if (alertsInitialized && Date.now() / 1000 - alert.time < 90) {
        host.classList.remove('bis-flash'); void host.offsetWidth; host.classList.add('bis-flash');
      }
      lastAlert = alert.id;
    }
    alertsInitialized = true;
  }
  function openForm(title, contents, submitLabel, onSubmit) {
    formOpen = true;
    const container = host.querySelector('#bis-form');
    container.innerHTML = `<div class="bis-modal" role="dialog" aria-modal="true" aria-label="${esc(title)}"><form><h3>${esc(title)}</h3><p class="bis-form-live" role="alert"></p>${contents}<p class="bis-form-error" role="alert"></p><div class="bis-buttons"><button type="button" data-action="cancel">Cancel</button><button type="submit">${esc(submitLabel)}</button></div></form></div>`;
    const form = container.querySelector('form');
    form.querySelector('input, select')?.focus();
    form.addEventListener('submit', async event => {
      event.preventDefault();
      const button = form.querySelector('[type="submit"]'); button.disabled = true;
      try {
        const response = await onSubmit(new FormData(form));
        if (!response?.ok) throw new Error(response?.error || 'Could not save.');
        closeForm(); await refresh();
      } catch (error) { form.querySelector('.bis-form-error').textContent = error.message; button.disabled = false; }
    });
  }
  function closeForm() { formOpen = false; host.querySelector('#bis-form').replaceChildren(); render(); }
  let speechUnlocked = false;
  host.addEventListener('pointerdown', () => {
    if (speechUnlocked) return;
    speechUnlocked = true;
    try {
      const synth = window.speechSynthesis;
      if (!synth) return;
      let warm = { text: pendingSpeech || ' ', volume: pendingSpeech ? 1 : 0 };
      try {
        if (typeof SpeechSynthesisUtterance === 'function') {
          warm = new SpeechSynthesisUtterance(pendingSpeech || ' ');
          warm.volume = pendingSpeech ? 1 : 0;
        }
      } catch { /* mocked speech */ }
      synth.speak(warm);
    } catch { /* gesture unlock for Chrome speech */ }
  }, true);
  host.addEventListener('change', event => { if (event.target.id === 'bis-symbol') { selected = event.target.value; render(); } });
  host.addEventListener('keydown', event => {
    if (event.key === 'Escape' && formOpen) closeForm();
    if (event.key === 'Tab' && formOpen) {
      const elements = [...host.querySelectorAll('.bis-modal input, .bis-modal select, .bis-modal button:not(:disabled)')];
      const first = elements[0], last = elements.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });
  host.addEventListener('click', event => {
    const button = event.target.closest('button[data-action]');
    if (!button || button.disabled) return;
    const action = button.dataset.action;
    if (action === 'settings') send('open-options').catch(() => { payload = { error: connectionError }; render(); });
    if (action === 'pick' && button.dataset.symbol) { selected = button.dataset.symbol; render(); openChart(selected); }
    if (action === 'chart' && button.dataset.symbol) openChart(button.dataset.symbol);
    if (action === 'refresh') refresh(true);
    if (action === 'cancel') closeForm();
    if (action === 'reset-layout') resetLayout();
    if (action === 'collapse') {
      collapsed = !collapsed; host.classList.toggle('bis-collapsed', collapsed); button.textContent = collapsed ? '+' : '−';
      button.setAttribute('aria-label', collapsed ? 'Expand panel' : 'Collapse panel');
      if (layout) applyLayout(layout, false);
    }
    if (action === 'planning') {
      const s = payload.settings;
      const current = profileOf(s.profile);
      const holdOptions = (profile, selected) => PROFILES[profile].holds
        .map(h => `<option value="${h}" ${Number(selected) === h ? 'selected' : ''}>${h} hour${h === 1 ? '' : 's'} (${h * 60} min)</option>`).join('');
      const p = PROFILES[current];
      openForm('Planning settings', `<p>These are planning values, not your exchange balance. Existing tracked plans remain unchanged. <b>Trend</b> scans 1h EMA bias, 15m structure and completed 5m continuation triggers for a 1 or 2 hour hold at 20-100x (50x default); the stop is the 5m structural low or high, capped at 0.60%, and must sit at least 0.25% inside the estimated isolated liquidation, so the card reports the real ceiling. <b>Scalp</b> fades parabolic exhaustion on 1m bars, long or short, for a 1 or 2 hour hold up to the pair's cap. Leverage never tightens a stop; it only changes margin and liquidation distance.</p><label>Profile<select name="profile" id="bis-profile">${Object.keys(PROFILES).map(k => `<option value="${k}" ${current === k ? 'selected' : ''}>${PROFILES[k].option}</option>`).join('')}</select></label><label>Planning equity (USDT)<input name="planning_equity" type="number" min="10" max="100000000" step="0.01" value="${s.planning_equity}" required></label><label>Risk per trade (%)<input name="risk_pct" type="number" min="0.01" max="2" step="0.01" value="${s.risk_pct}" required></label><label>Leverage (isolated)<input name="leverage" type="number" min="${p.min}" max="${p.max}" step="1" value="${s.leverage}" required></label><label>Maximum hold<select name="hold_hours">${holdOptions(current, s.hold_hours)}</select></label>`, 'Save settings', data => send('save-planning', Object.fromEntries([...data].map(([k, v]) => [k, k === 'profile' ? String(v) : Number(v)]))));
      const profileSelect = host.querySelector('#bis-profile');
      profileSelect?.addEventListener('change', () => {
        const next = PROFILES[profileOf(profileSelect.value)];
        const leverage = host.querySelector('.bis-modal [name="leverage"]');
        const hold = host.querySelector('.bis-modal [name="hold_hours"]');
        if (leverage) {
          leverage.min = next.min; leverage.max = next.max;
          const value = Number(leverage.value);
          if (Number.isFinite(value)) leverage.value = Math.min(Math.max(value, next.min), next.max);
        }
        if (hold) {
          const keep = next.holds.includes(Number(hold.value)) ? Number(hold.value) : next.holds[0];
          hold.innerHTML = holdOptions(profileOf(profileSelect.value), keep);
        }
      });
    }
    if (action === 'paper' || action === 'manual') {
      const row = payload.symbols[selected || payload.best_symbol];
      if (!actionable(row)) return;
      const stopCopy = `Stop ${price(row.plan.stop)} · Target ${price(row.plan.target)}.`;
      const intro = action === 'paper'
        ? `<p>Simulated tracking only. No order is submitted.</p>`
        : `<p>Enter the fill you already executed on Bitunix. This records it for alerts; it does not place an order or attach a stop.</p>`;
      const stopField = action === 'manual'
        ? `<p>${stopCopy} After tracking, click Set Bitunix stop. That places the exchange stop for you.</p><label class="bis-check"><input name="exchange_stop_confirmed" type="checkbox"> I already placed the Bitunix stop-loss at ${price(row.plan.stop)}</label>`
        : `<p>${stopCopy} Paper tracking does not place an exchange stop.</p>`;
      openForm(action === 'paper' ? 'Track a paper trade' : 'Record your Bitunix fill', `${intro}<label>Entry price<input name="entry" type="number" min="${row.plan.entry_low}" max="${row.plan.entry_high}" step="any" value="${row.plan.entry}" required></label><label>Quantity<input name="quantity" type="number" min="0.000000001" max="${row.plan.quantity}" step="any" value="${row.plan.quantity}" required></label>${stopField}`, 'Start tracking', data => send('track-entry', {
        signal_id: row.signal_id,
        kind: action,
        entry: Number(data.get('entry')),
        quantity: Number(data.get('quantity')),
        ...(action === 'manual' ? { exchange_stop_confirmed: data.get('exchange_stop_confirmed') === 'on' } : {}),
      }));
    }
    if (action === 'place-stop') {
      button.disabled = true;
      send('place-stop', { id: button.dataset.id }).then(response => {
        if (!response?.ok) throw new Error(response?.error || 'Could not place the Bitunix stop.');
        return refresh();
      }).catch(error => { payload = { ...(payload || {}), error: error.message }; render(); });
    }
    if (action === 'confirm-stop') {
      send('confirm-stop', { id: button.dataset.id }).then(response => {
        if (!response?.ok) throw new Error(response?.error || 'Could not confirm the stop.');
        return refresh();
      }).catch(error => { payload = { ...(payload || {}), error: error.message }; render(); });
    }
    if (action === 'close') {
      const trade = payload.trades.find(t => t.id === button.dataset.id);
      const current = payload.symbols[trade.symbol]?.price || trade.plan.entry;
      const liveNote = trade.kind === 'exchange' || trade.exchange_position_id
        ? ' If this Bitunix position is still open, the next scan will import it again.'
        : '';
      openForm('Record trade closure', '<p>This ends tracking only. Close any real position on Bitunix first.' + liveNote + '</p><label>Recorded exit price<input name="exit_price" type="number" min="0.000000001" step="any" value="' + current + '" required></label>', 'Record closure', data => send('close-track', { id: trade.id, exit_price: Number(data.get('exit_price')) }));
    }
  });
  let refreshing = false;
  async function refresh(force = false) {
    if (refreshing) return;
    refreshing = true;
    try {
      const response = await send(force ? 'force-refresh' : 'request-latest');
      if (!response?.payload) throw new Error(connectionError);
      payload = response.payload;
      if (Number.isFinite(response.fetchedAt)) polledAt = response.fetchedAt;
    } catch { payload = { ...(payload?.symbols ? payload : {}), error: connectionError }; }
    finally { refreshing = false; render(); }
  }
  render();
  try {
    chrome.runtime.onMessage.addListener(message => {
      if (message.type === 'signals-update') {
        payload = message.payload || { error: 'Signal data is incomplete. Retry the connection.' };
        if (Number.isFinite(message.fetchedAt)) polledAt = message.fetchedAt;
        render();
      }
    });
  } catch { payload = { error: connectionError }; render(); }
  timers.push(setInterval(() => refresh(), 5000));
  timers.push(setInterval(() => {
    if (!payload || formOpen || host.querySelector('select:focus, input:focus, textarea:focus')) return;
    render();
  }, 1000));
  window.__bisIntradayTeardown = () => {
    timers.forEach(clearInterval);
    listeners.forEach(unlisten => unlisten());
    host.remove();
    if (window.__bisIntradayTeardown) delete window.__bisIntradayTeardown;
  };
  refresh();
})();
