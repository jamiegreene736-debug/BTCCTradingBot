(() => {
  if (document.getElementById('bis-panel')) return;
  const host = document.createElement('aside');
  host.id = 'bis-panel';
  host.setAttribute('aria-label', 'Bitunix intraday signals');
  document.documentElement.appendChild(host);
  let payload = null, selected = '', collapsed = false, formOpen = false, lastAlert = '', alertsInitialized = false;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const money = value => Number.isFinite(value) ? '$' + value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—';
  const price = value => Number.isFinite(value) ? value.toLocaleString('en-US', { maximumFractionDigits: value < 1 ? 8 : value < 100 ? 5 : 2 }) : '—';
  const fixed = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—';
  const label = value => String(value || 'WAIT').replaceAll('_', ' ');
  const tone = state => state?.startsWith('EXIT') ? 'exit' : state?.includes('LONG') ? 'long' : state?.includes('SHORT') ? 'short' : 'wait';
  const ago = timestamp => timestamp ? Math.max(0, Math.floor(Date.now() / 1000 - timestamp)) + 's ago' : 'Waiting for data';
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
  host.innerHTML = `<header><div><span class="bis-eyebrow">BITUNIX · INTRADAY</span><strong>Trade signals</strong></div><div class="bis-actions"><button data-action="settings" title="Connection settings" aria-label="Connection settings">⚙</button><button data-action="collapse" aria-label="Collapse panel">−</button></div></header><div id="bis-body"><div id="bis-status" role="status"></div><div id="bis-planning"></div><div id="bis-selection"></div><div id="bis-card"></div><div id="bis-trades"></div><details><summary>Recent alerts</summary><div id="bis-history"></div></details><details><summary>Recorded closures</summary><div id="bis-closed"></div></details><footer>Alerts only · Orders and stops stay on Bitunix.<br>Candidate rules under evaluation; no measured win probability.</footer></div><div id="bis-form"></div>`;
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
      for (const id of ['card', 'selection', 'planning', 'trades', 'history', 'closed']) host.querySelector('#bis-' + id).replaceChildren();
      host.querySelector('#bis-card').innerHTML = '<div class="bis-buttons"><button data-action="settings">Open Settings</button><button data-action="refresh">Retry connection</button></div><p class="bis-empty">Signals need a running intraday backend and its dashboard password. Use Save and test connection in Settings.</p>';
      return;
    }
    const settings = payload.settings;
    status.className = payload.error ? 'bis-notice' : 'bis-status';
    status.textContent = payload.error || payload.status?.error || (payload.status?.ready ? '● Monitoring liquid USDT perpetuals' : 'Waiting for complete market data');
    host.querySelector('#bis-planning').innerHTML = `<div><small>Planning equity</small><b>${money(settings.planning_equity)}</b></div><div><small>Risk / trade</small><b>${fixed(settings.risk_pct)}%</b></div><div><small>Leverage / hold</small><b>${settings.leverage}x · ≤${settings.hold_hours}h</b></div><button data-action="planning">Edit</button>`;
    const rows = Object.values(payload.symbols || {});
    if (selected && !payload.symbols[selected]) selected = '';
    const symbol = selected && payload.symbols[selected] ? selected : payload.best_symbol;
    const row = payload.symbols[symbol];
    if (document.activeElement?.id !== 'bis-symbol') host.querySelector('#bis-selection').innerHTML = `<label>Market <select id="bis-symbol"><option value="">Best setup</option>${rows.map(r => `<option value="${esc(r.symbol)}" ${selected === r.symbol ? 'selected' : ''}>${esc(r.symbol)} · ${esc(fresh(r) ? label(r.state) : 'WAIT')}</option>`).join('')}</select></label><button data-action="refresh" title="Refresh signals">↻</button>`;
    if (row) {
      const state = fresh(row) && (!row.plan || row.plan.expires_at > Date.now() / 1000) ? row.state : 'WAIT';
      const plan = row.plan;
      host.querySelector('#bis-card').innerHTML = `<article class="bis-signal ${tone(state)}"><div class="bis-row"><strong>${esc(row.symbol)}</strong><span>${price(row.price)}</span></div><div class="bis-state" aria-live="polite">${esc(label(state))}</div><p>${esc(fresh(row) ? row.reasons?.[0] : 'Data is stale. Entry alerts are paused.')}</p><div class="bis-meta">${esc(row.setup || '4h bias → 1h structure → 15m entry')} · ${esc(ago(row.as_of))}</div>${plan ? `<div class="bis-levels"><div><small>Entry zone</small><b>${price(plan.entry_low)} – ${price(plan.entry_high)}</b></div><div><small>Stop loss</small><b>${price(plan.stop)}</b></div><div><small>Profit target</small><b>${price(plan.target)}</b></div><div><small>Net reward / risk</small><b>${fixed(plan.net_reward_risk)}R</b></div><div><small>Planned loss incl. costs</small><b>${money(plan.risk_usdt)} · ${fixed(plan.risk_pct)}%</b></div><div><small>Notional / margin</small><b>${money(plan.notional)} / ${money(plan.margin)}</b></div></div><div class="bis-meta">Entry expires ${new Date(plan.expires_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} · Estimated leverage ceiling ${plan.max_leverage}x · 25-40x band</div><div class="bis-buttons"><button data-action="paper" ${actionable(row) ? '' : 'disabled'}>Track paper trade</button><button data-action="manual" ${actionable(row) ? '' : 'disabled'}>Record my fill</button></div>` : ''}<details><summary>Why this signal · ${row.checks.filter(c => c.passed).length}/${row.checks.length} checks</summary><ul class="bis-checks">${row.checks.map(c => `<li class="${c.passed ? 'pass' : 'fail'}"><span>${c.passed ? '✓' : '○'}</span><div><b>${esc(c.label)}</b><small>${esc(c.detail)}</small></div></li>`).join('')}</ul><div class="bis-meta">4h bias: ${esc(row.metrics.trend_4h || '—')} · 1h structure: ${esc(row.metrics.trend_1h || '—')}<br>ATR: ${fixed(row.metrics.atr_pct)}% · 1h ATR: ${fixed(row.metrics.hourly_atr_pct)}% · Volume: ${fixed(row.metrics.relative_volume)}×<br>UTC session VWAP: ${price(row.metrics.vwap)}<br>BTC relative strength (6h): ${fixed(row.metrics.relative_strength_pct)}%<br>Funding / interval: ${fixed(row.metrics.funding_rate_pct, 4)}%<br>Open interest: ${row.metrics.open_interest == null ? 'Unavailable' : price(row.metrics.open_interest)}</div>${plan ? `<p class="bis-meta">Estimated liquidation: ${price(plan.liquidation_estimate)}. Isolated margin, no extra collateral; verify on Bitunix. Estimated total costs ${fixed(plan.cost_pct)}%, including ${plan.funding_payments} projected funding payments. Future rates can change. Target 2 (context only): ${price(plan.target2)}.</p>` : ''}</details></article>`;
    } else host.querySelector('#bis-card').innerHTML = '<p class="bis-empty">Scanner warming up. Missing data blocks entries.</p>';
    host.querySelector('#bis-trades').innerHTML = `<h3>Tracked trades <span>${payload.trades.length}</span></h3>${payload.trades.length ? payload.trades.map(t => payload.error && !t.state.startsWith('EXIT_') ? { ...t, state: 'REVIEW', reason: 'Connection unavailable. Check Bitunix; this is the last recorded plan.' } : t).map(t => `<article class="bis-trade ${tone(t.state)}"><div class="bis-row"><b>${esc(t.symbol)} · ${esc(t.plan.side.toUpperCase())}</b><small>${t.kind === 'paper' ? 'PAPER' : 'USER RECORDED'}</small></div><strong class="bis-trade-state">${esc(label(t.state))}</strong><p>${esc(t.reason)}</p><div class="bis-meta">Entry ${price(t.plan.entry)} · Stop ${price(t.current_stop)} · Target ${price(t.plan.target)}<br>Held ${fixed((Date.now() / 1000 - t.opened_at) / 3600, 1)}h / ${t.plan.hold_hours}h max</div><button data-action="close" data-id="${esc(t.id)}">Record closure</button></article>`).join('') : '<p class="bis-empty">Record an entry to receive hold and exit guidance. Tracking never submits an order.</p>'}`;
    host.querySelector('#bis-history').innerHTML = payload.history.slice(0, 10).map(event => `<div class="bis-history-row"><div><b>${esc(event.symbol)}</b> · ${esc(label(event.state))}<small>${esc(event.reason)}</small></div><time>${new Date(event.time * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time></div>`).join('') || '<p class="bis-empty">Confirmed entries and exit changes appear here.</p>';
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
    if (action === 'refresh') refresh(true);
    if (action === 'cancel') closeForm();
    if (action === 'collapse') {
      collapsed = !collapsed; host.classList.toggle('bis-collapsed', collapsed); button.textContent = collapsed ? '+' : '−';
      button.setAttribute('aria-label', collapsed ? 'Expand panel' : 'Collapse panel');
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
      openForm('Record trade closure', '<p>This ends tracking only. Close any real position on Bitunix first.</p><label>Recorded exit price<input name="exit_price" type="number" min="0.000000001" step="any" value="' + current + '" required></label>', 'Record closure', data => send('close-track', { id: trade.id, exit_price: Number(data.get('exit_price')) }));
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
  setInterval(() => refresh(), 5000);
  refresh();
})();
