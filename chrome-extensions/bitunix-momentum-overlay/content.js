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
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const money = value => Number.isFinite(value) ? '$' + value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—';
  const price = value => Number.isFinite(value) ? value.toLocaleString('en-US', { maximumFractionDigits: value < 1 ? 8 : value < 100 ? 5 : 2 }) : '—';
  const fixed = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—';
  const label = value => String(value || 'WAIT').replaceAll('_', ' ');
  const tone = state => state?.startsWith('EXIT') ? 'exit' : state?.includes('LONG') ? 'long' : state?.includes('SHORT') ? 'short' : 'wait';
  const ago = timestamp => timestamp ? Math.max(0, Math.floor(Date.now() / 1000 - timestamp)) + 's ago' : 'Waiting for data';
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
  host.innerHTML = `<header title="Drag the title or grip to move. Double-click or use Reset to restore the default position."><div class="bis-drag"><span class="bis-grip" aria-hidden="true"></span><div><span class="bis-eyebrow">BITUNIX · INTRADAY${version ? ' · v' + version : ''}</span><strong>Trade signals</strong></div></div><div class="bis-actions"><button data-action="reset-layout" title="Reset size and position" aria-label="Reset size and position">⤢</button><button data-action="settings" title="Connection settings" aria-label="Connection settings">⚙</button><button data-action="collapse" aria-label="Collapse panel">−</button></div></header><div id="bis-body"><div id="bis-status" role="status"></div><div id="bis-planning"></div><div id="bis-handoff" hidden></div><div id="bis-queue"></div><div id="bis-selection"></div><div id="bis-card"></div><div id="bis-trades"></div><details id="bis-history-wrap"><summary>Recent alerts</summary><div id="bis-history"></div></details><details><summary>Recorded closures</summary><div id="bis-closed"></div></details><footer>Alerts only · Live positions import read-only. Orders and stops stay on Bitunix.<br>Candidate rules under evaluation; no measured win probability.</footer></div><div id="bis-form"></div><div class="bis-resize" role="separator" aria-orientation="horizontal" aria-label="Resize panel" title="Drag the corner to resize"></div>`;
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
      if (hit.closest?.('button, a, input, select, textarea, summary')) return;
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
    const status = host.querySelector('#bis-status');
    if (!payload || (payload.error && !payload.symbols)) {
      status.className = 'bis-notice'; status.textContent = payload?.error || 'Connecting to your signal scanner…';
      for (const id of ['card', 'selection', 'planning', 'queue', 'trades', 'history', 'closed']) host.querySelector('#bis-' + id).replaceChildren();
      const handoffEmpty = host.querySelector('#bis-handoff');
      if (handoffEmpty) { handoffEmpty.hidden = true; handoffEmpty.replaceChildren(); }
      host.querySelector('#bis-card').innerHTML = '<div class="bis-buttons"><button data-action="settings">Open Settings</button><button data-action="refresh">Retry connection</button></div><p class="bis-empty">Signals need a running intraday backend and its dashboard password. Use Save and test connection in Settings.</p>';
      return;
    }
    const settings = payload.settings;
    const liveCount = (payload.trades || []).filter(t => t.kind === 'exchange' || t.exchange_position_id).length;
    const liveBit = liveCount ? ` · ${liveCount} live position${liveCount === 1 ? '' : 's'}` : '';
    status.className = payload.error ? 'bis-notice' : 'bis-status';
    status.textContent = payload.error || payload.status?.error || (payload.status?.ready ? '● Monitoring liquid USDT perpetuals' + liveBit : 'Waiting for complete market data');
    host.querySelector('#bis-planning').innerHTML = `<div><small>Planning equity</small><b>${money(settings.planning_equity)}</b></div><div><small>Risk / trade</small><b>${fixed(settings.risk_pct)}%</b></div><div><small>Leverage / hold</small><b>${settings.leverage}x · ≤${settings.hold_hours}h</b></div><button data-action="planning">Edit</button>`;
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
        reason: item.reasons?.[0] || '', price: item.price,
      }));
    host.querySelector('#bis-queue').innerHTML = `<h3>Top setups <span>${Math.min(queue.length, 5)}</span></h3>${queue.slice(0, 5).map((item, index) => {
      const live = fresh(payload.symbols?.[item.symbol]) ? item.state : 'WAIT';
      const left = item.expires_at && live.startsWith('ENTER_') ? remain(item.expires_at - nowSec) : '';
      return `<button type="button" class="bis-queue-row ${tone(live)}${item.symbol === symbol ? ' active' : ''}" data-action="pick" data-symbol="${esc(item.symbol)}"><b>${index + 1}</b><div><strong>${esc(item.symbol)}</strong><small>${esc(label(live))}${item.setup ? ' · ' + esc(item.setup) : ''}</small><small>${esc(clock(item.state_since || item.as_of))}${left ? ' · ' + left + ' left' : ''}</small></div><span>${price(item.price)}</span></button>`;
    }).join('') || '<p class="bis-empty">No ranked setups yet.</p>'}`;
    if (document.activeElement?.id !== 'bis-symbol') host.querySelector('#bis-selection').innerHTML = `<label>Market <select id="bis-symbol"><option value="">Best setup</option>${rows.map(r => `<option value="${esc(r.symbol)}" ${selected === r.symbol ? 'selected' : ''}>${esc(r.symbol)} · ${esc(fresh(r) ? label(r.state) : 'WAIT')}</option>`).join('')}</select></label><button data-action="refresh" title="Refresh signals">↻</button>`;
    if (row) {
      const state = fresh(row) && (!row.plan || row.plan.expires_at > Date.now() / 1000) ? row.state : 'WAIT';
      const plan = row.plan;
      const expiryLeft = plan && state.startsWith('ENTER_') ? remain(plan.expires_at - nowSec) : '';
      host.querySelector('#bis-card').innerHTML = `<article class="bis-signal ${tone(state)}"><div class="bis-row"><strong>${esc(row.symbol)}</strong><span>${price(row.price)}</span></div><div class="bis-state" aria-live="polite">${esc(label(state))}</div><p>${esc(fresh(row) ? row.reasons?.[0] : 'Data is stale. Entry alerts are paused.')}</p><div class="bis-meta">${esc(row.setup || '4h bias → 1h structure → 15m entry')}<br>Shown ${esc(clock(row.state_since || row.as_of))} · ${esc(ago(row.state_since || row.as_of))}<br>Market snapshot ${esc(clock(row.as_of))} · ${esc(ago(row.as_of))}</div>${plan ? `<div class="bis-levels"><div><small>Entry zone</small><b>${price(plan.entry_low)} – ${price(plan.entry_high)}</b></div><div><small>Stop loss</small><b>${price(plan.stop)}</b></div><div><small>Profit target</small><b>${price(plan.target)}</b></div><div><small>Net reward / risk</small><b>${fixed(plan.net_reward_risk)}R</b></div><div><small>Planned loss incl. costs</small><b>${money(plan.risk_usdt)} · ${fixed(plan.risk_pct)}%</b></div><div><small>Notional / margin</small><b>${money(plan.notional)} / ${money(plan.margin)}</b></div></div><div class="bis-meta">Entry expires ${esc(clock(plan.expires_at))}${expiryLeft ? ' · ' + expiryLeft + ' left' : ''} · Estimated leverage ceiling ${plan.max_leverage}x · 25-40x band</div><div class="bis-buttons"><button data-action="paper" ${actionable(row) ? '' : 'disabled'}>Track paper trade</button><button data-action="manual" ${actionable(row) ? '' : 'disabled'}>Record my fill</button></div>` : ''}<details><summary>Why this signal · ${row.checks.filter(c => c.passed).length}/${row.checks.length} checks</summary><ul class="bis-checks">${row.checks.map(c => `<li class="${c.passed ? 'pass' : 'fail'}"><span>${c.passed ? '✓' : '○'}</span><div><b>${esc(c.label)}</b><small>${esc(c.detail)}</small></div></li>`).join('')}</ul><div class="bis-meta">4h bias: ${esc(row.metrics.trend_4h || '—')} · 1h structure: ${esc(row.metrics.trend_1h || '—')}<br>ATR: ${fixed(row.metrics.atr_pct)}% · 1h ATR: ${fixed(row.metrics.hourly_atr_pct)}% · Volume: ${fixed(row.metrics.relative_volume)}×<br>UTC session VWAP: ${price(row.metrics.vwap)}<br>BTC relative strength (6h): ${fixed(row.metrics.relative_strength_pct)}%<br>Funding / interval: ${fixed(row.metrics.funding_rate_pct, 4)}%<br>Open interest: ${row.metrics.open_interest == null ? 'Unavailable' : price(row.metrics.open_interest)}</div>${plan ? `<p class="bis-meta">Estimated liquidation: ${price(plan.liquidation_estimate)}. Isolated margin, no extra collateral; verify on Bitunix. Estimated total costs ${fixed(plan.cost_pct)}%, including ${plan.funding_payments} projected funding payments. Future rates can change. Target 2 (context only): ${price(plan.target2)}.</p>` : ''}</details></article>`;
    } else host.querySelector('#bis-card').innerHTML = '<p class="bis-empty">Scanner warming up. Missing data blocks entries.</p>';
    const kindLabel = kind => kind === 'paper' ? 'PAPER' : kind === 'exchange' ? 'LIVE' : 'USER RECORDED';
    const pnlText = value => Number.isFinite(value) ? `${value >= 0 ? '+' : '−'}${money(Math.abs(value))}` : '';
    const tradesEmpty = payload.positions?.error
      || (payload.positions?.connected === false
        ? 'Add Bitunix API keys on the backend to import live positions. You can still record a fill.'
        : 'No open Bitunix positions. Record a fill to track a paper or manual trade.');
    host.querySelector('#bis-trades').innerHTML = `<h3>Tracked trades <span>${payload.trades.length}</span></h3>${payload.trades.length ? payload.trades.map(t => payload.error && !t.state.startsWith('EXIT_') ? { ...t, state: 'REVIEW', reason: 'Connection unavailable. Check Bitunix; this is the last recorded plan.' } : t).map(t => `<article class="bis-trade ${tone(t.state)}"><div class="bis-row"><b>${esc(t.symbol)} · ${esc(t.plan.side.toUpperCase())}</b><small>${kindLabel(t.kind)}</small></div><strong class="bis-trade-state">${esc(label(t.state))}</strong><p>${esc(t.reason)}</p><div class="bis-meta">Entry ${price(t.plan.entry)}${Number.isFinite(t.mark_price) ? ' · Mark ' + price(t.mark_price) : ''} · Stop ${price(t.current_stop)} · Target ${price(t.plan.target)}${Number.isFinite(t.unrealized_pnl) ? '<br>Unrealized ' + pnlText(t.unrealized_pnl) : ''}<br>Held ${fixed((Date.now() / 1000 - t.opened_at) / 3600, 1)}h / ${t.plan.hold_hours}h max</div><button data-action="close" data-id="${esc(t.id)}">Record closure</button></article>`).join('') : `<p class="bis-empty">${esc(tradesEmpty)}</p>`}`;
    host.querySelector('#bis-history').innerHTML = payload.history.slice(0, 30).map(event => `<div class="bis-history-row"><div><b>${esc(event.symbol)}</b> · ${esc(label(event.state))}${event.setup ? ' · ' + esc(event.setup) : ''}<small>${esc(event.reason)}</small></div><time datetime="${esc(event.time ? new Date(event.time * 1000).toISOString() : '')}">${esc(clock(event.time))}<small>${esc(ago(event.time))}</small></time></div>`).join('') || '<p class="bis-empty">WATCH and ENTER alerts appear here with timestamps.</p>';
    host.querySelector('#bis-closed').innerHTML = payload.closed_trades.slice(0, 10).map(t => `<div class="bis-history-row"><div><b>${esc(t.symbol)}</b> · ${esc(t.kind)}<small>Recorded exit ${price(t.exit_price)} · estimated net</small></div><b>${money(t.estimated_net_pnl)}</b></div>`).join('') || '<p class="bis-empty">No recorded closures. Signal alerts are not completed trades.</p>';
    if (host.querySelector('#bis-card details')) host.querySelector('#bis-card details').open = checksOpen;
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
    if (action === 'pick' && button.dataset.symbol) { selected = button.dataset.symbol; render(); }
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
      openForm('Planning settings', `<p>These are planning values, not your exchange balance. Existing tracked plans remain unchanged. The scanner is built for isolated 25-40x and a 12h or 24h hold. Leverage never tightens the stop; if 40x would liquidate before the structural stop, the entry stays blocked.</p><label>Planning equity (USDT)<input name="planning_equity" type="number" min="10" max="100000000" step="0.01" value="${s.planning_equity}" required></label><label>Risk per trade (%)<input name="risk_pct" type="number" min="0.01" max="2" step="0.01" value="${s.risk_pct}" required></label><label>Leverage (isolated, 25-40x intended)<input name="leverage" type="number" min="1" max="40" step="1" value="${s.leverage}" required></label><label>Maximum hold<select name="hold_hours"><option value="12" ${s.hold_hours === 12 ? 'selected' : ''}>12 hours</option><option value="24" ${s.hold_hours === 24 ? 'selected' : ''}>24 hours</option></select></label>`, 'Save settings', data => send('save-planning', Object.fromEntries([...data].map(([k, v]) => [k, Number(v)]))));
    }
    if (action === 'paper' || action === 'manual') {
      const row = payload.symbols[selected || payload.best_symbol];
      if (!actionable(row)) return;
      openForm(action === 'paper' ? 'Track a paper trade' : 'Record your Bitunix fill', `<p>${action === 'paper' ? 'Simulated tracking only. No order is submitted.' : 'Enter the fill you already executed on Bitunix. This records it for alerts; it does not place an order or attach a stop.'}</p><label>Entry price<input name="entry" type="number" min="${row.plan.entry_low}" max="${row.plan.entry_high}" step="any" value="${row.plan.entry}" required></label><label>Quantity<input name="quantity" type="number" min="0.000000001" max="${row.plan.quantity}" step="any" value="${row.plan.quantity}" required></label><p>Stop ${price(row.plan.stop)} · Target ${price(row.plan.target)}. For a real trade, set your stop on Bitunix.</p>`, 'Start tracking', data => send('track-entry', { signal_id: row.signal_id, kind: action, entry: Number(data.get('entry')), quantity: Number(data.get('quantity')) }));
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
    } catch { payload = { ...(payload?.symbols ? payload : {}), error: connectionError }; }
    finally { refreshing = false; render(); }
  }
  render();
  try {
    chrome.runtime.onMessage.addListener(message => {
      if (message.type === 'signals-update') {
        payload = message.payload || { error: 'Signal data is incomplete. Retry the connection.' };
        render();
      }
    });
  } catch { payload = { error: connectionError }; render(); }
  timers.push(setInterval(() => refresh(), 5000));
  timers.push(setInterval(() => { if (payload && !formOpen) render(); }, 1000));
  window.__bisIntradayTeardown = () => {
    timers.forEach(clearInterval);
    listeners.forEach(unlisten => unlisten());
    host.remove();
    if (window.__bisIntradayTeardown) delete window.__bisIntradayTeardown;
  };
  refresh();
})();
