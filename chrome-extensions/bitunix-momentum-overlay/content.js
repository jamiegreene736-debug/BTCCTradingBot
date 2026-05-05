// Bitunix pump-fade overlay content script.
// Renders the floating scanner panel and receives data from the MV3
// background worker. All DOM/CSS names are prefixed with bxm-.

(function () {
  if (window.__bxmOverlayInstalled) return;
  window.__bxmOverlayInstalled = true;

  let latest = null;
  let fetchedAt = 0;
  let activeSymbol = localStorage.getItem("bxm-active-symbol") || "";
  let collapsed = localStorage.getItem("bxm-collapsed") === "1";
  let closeAttempts = {};
  let panelEl = null;

  function pick(obj, ...keys) {
    for (const key of keys) {
      if (obj && obj[key] !== undefined && obj[key] !== null) return obj[key];
    }
    return null;
  }

  function num(value, fallback = 0) {
    const out = Number(value);
    return Number.isFinite(out) ? out : fallback;
  }

  function normSymbol(value) {
    return String(value || "").trim().toUpperCase();
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;",
    }[ch]));
  }

  function fmtPrice(value) {
    const p = Number(value);
    if (!Number.isFinite(p) || p <= 0) return "--";
    if (p >= 1000) return p.toLocaleString(undefined, { maximumFractionDigits: 1 });
    if (p >= 1) return p.toLocaleString(undefined, { maximumFractionDigits: 3 });
    return p.toLocaleString(undefined, { maximumFractionDigits: 6 });
  }

  function fmtMoney(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "--";
    const sign = n > 0 ? "+" : n < 0 ? "-" : "";
    return `${sign}$${Math.abs(n).toFixed(4)}`;
  }

  function fmtPct(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "--";
    return `${n.toFixed(2)}%`;
  }

  function fmtAge(secs) {
    const s = Math.max(0, Math.floor(Number(secs) || 0));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ago`;
  }

  function fmtCountdown(secs) {
    const s = Math.max(0, Math.floor(Number(secs) || 0));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  function positionKey(symbol, pos) {
    return `${normSymbol(symbol)}:${pick(pos, "position_id", "positionId") || pick(pos, "opened_at", "openedAt") || "open"}`;
  }

  function positionCountdown(pos) {
    const rawCloseAt = Number(pick(pos, "auto_close_at", "autoCloseAt"));
    if (Number.isFinite(rawCloseAt) && rawCloseAt > 0) {
      const closeAtMs = rawCloseAt < 1_000_000_000_000 ? rawCloseAt * 1000 : rawCloseAt;
      return Math.max(0, Math.ceil((closeAtMs - Date.now()) / 1000));
    }
    const rawRemaining = Number(pick(pos, "seconds_remaining", "secondsRemaining"));
    if (Number.isFinite(rawRemaining)) {
      const elapsed = fetchedAt ? (Date.now() - fetchedAt) / 1000 : 0;
      return Math.max(0, Math.ceil(rawRemaining - elapsed));
    }
    return null;
  }

  function activeOpenPosition(symData, symbol) {
    const sym = normSymbol(symbol);
    const rows = [];
    const add = (value) => {
      if (!value) return;
      if (Array.isArray(value)) rows.push(...value);
      else rows.push(value);
    };
    add(symData?.open_position);
    add(symData?.openPosition);
    add(symData?.open_positions);
    add(symData?.openPositions);
    add(latest?.open_positions);
    add(latest?.openPositions);

    for (const pos of rows) {
      const posSym = normSymbol(pick(pos, "symbol"));
      if (posSym && posSym !== sym) continue;
      const qty = Number(pick(pos, "qty", "size", "volume"));
      if (Number.isFinite(qty) && qty === 0) continue;
      return pos;
    }
    return null;
  }

  function triggerAutoClose(symbol, pos, remainingSecs) {
    if (remainingSecs === null || remainingSecs > 0) return;
    const key = positionKey(symbol, pos);
    const now = Date.now();
    const last = closeAttempts[key];
    if (last && now - last.at < 15000) return;

    closeAttempts[key] = { at: now, status: "closing" };
    chrome.runtime.sendMessage({
      type: "close-symbol",
      symbol: normSymbol(symbol),
      positionId: pick(pos, "position_id", "positionId") || null,
    }, (resp) => {
      closeAttempts[key] = {
        at: Date.now(),
        status: resp?.ok ? "closed" : "error",
        message: resp?.error || resp?.message || "",
      };
      chrome.runtime.sendMessage({ type: "force-refresh" }, (freshResp) => {
        if (freshResp) {
          latest = freshResp.payload;
          fetchedAt = freshResp.fetchedAt || Date.now();
          render();
        }
      });
    });
  }

  function decisionFor(symData) {
    return symData?.decision || symData?.recommendation || symData?.next_1h ||
      symData?.next1h || symData?.next_15m || symData?.next15m || {};
  }

  function stageFor(decision) {
    const action = String(decision?.action || "wait").toLowerCase();
    const setup = String(decision?.setup || "");
    const stage = String(decision?.setup_stage || decision?.setupStage || "").toLowerCase();
    if (action === "short" && setup === "parabolic_pump_fade") return "short";
    if (stage === "pump_building" || decision?.pre_pump_building || decision?.prePumpBuilding) return "building";
    if (stage === "pump_watch") return "watch";
    return "hunting";
  }

  function scoreFor(decision) {
    const stage = stageFor(decision);
    if (stage === "short") return num(pick(decision, "confidence_score", "confidenceScore"), 0);
    if (stage === "building") return num(pick(decision, "pre_pump_score", "prePumpScore", "checklist_score", "checklistScore"), 0);
    if (stage === "watch") return num(pick(decision, "checklist_score", "checklistScore"), 0);
    return num(pick(decision, "checklist_score", "checklistScore"), 0);
  }

  function stageCopy(stage, score) {
    if (stage === "short") return {
      title: "FADE SHORT",
      kicker: score >= 95 ? "AUTO-TRADE READY" : "SHORT READY",
      detail: "pump has stalled and bearish rejection is confirmed",
      icon: "v",
    };
    if (stage === "building") return {
      title: "PUMP BUILDING",
      kicker: "HEADS UP",
      detail: "buyers are pressing; wait for blow-off high and rejection",
      icon: "^",
    };
    if (stage === "watch") return {
      title: "PUMP WATCH",
      kicker: "WAITING",
      detail: "pump detected; waiting for near-high 1m fade entry",
      icon: "||",
    };
    return {
      title: "HUNTING",
      kicker: "NO TRADE",
      detail: "waiting for a small fast pump before looking for the short",
      icon: "||",
    };
  }

  function checkRows(decision) {
    const rows = decision?.pump_fade_checks || decision?.pumpFadeChecks || [];
    if (!rows.length) return "";
    return `<div class="bxm-checks">
      <div class="bxm-section-title">Pump fade checklist</div>
      ${rows.slice(0, 7).map((row) => `
        <div class="bxm-check ${row.passed ? "passed" : ""}">
          <div>
            <strong>${escapeHtml(row.label || row.key || "Check")}</strong>
            <span>${escapeHtml(row.detail || "")}</span>
          </div>
          <em>${row.passed ? "OK" : "WAIT"}</em>
        </div>
      `).join("")}
    </div>`;
  }

  function tradePlanHtml(decision, symData) {
    const plan = decision?.trade_plan || decision?.tradePlan || symData?.trade_plan || symData?.tradePlan;
    if (!plan || plan.status !== "ready") return "";
    const orderType = String(pick(plan, "order_type", "orderType") || "MARKET").replace(/_/g, " ");
    const entry = Number(pick(plan, "entry_price", "entryPrice"));
    const target = Number(pick(plan, "target_exit_price", "targetExitPrice", "max_exit_price", "maxExitPrice", "take_profit", "takeProfit"));
    const stop = Number(pick(plan, "stop_loss", "stopLoss"));
    const rewardPct = pick(plan, "reward_pct", "rewardPct");
    const riskPct = pick(plan, "risk_pct", "riskPct");
    return `<div class="bxm-plan">
      <div class="bxm-plan-head">
        <span>Pump fade entry</span>
        <strong>${escapeHtml(orderType)}</strong>
      </div>
      <div class="bxm-plan-price">${fmtPrice(entry)}</div>
      <div class="bxm-plan-grid">
        <div><span>Max exit</span><strong class="good">${fmtPrice(target)}</strong></div>
        <div><span>Stop</span><strong class="bad">${fmtPrice(stop)}</strong></div>
        <div><span>Reward</span><strong>${fmtPct(rewardPct)}</strong></div>
        <div><span>Risk</span><strong>${fmtPct(riskPct)}</strong></div>
      </div>
      ${plan.rationale ? `<div class="bxm-note">${escapeHtml(plan.rationale)}</div>` : ""}
    </div>`;
  }

  function countdownHtml(symData, symbol) {
    const pos = activeOpenPosition(symData, symbol);
    if (!pos) return "";
    const remaining = positionCountdown(pos);
    triggerAutoClose(symbol, pos, remaining);
    const key = positionKey(symbol, pos);
    const attempt = closeAttempts[key] || {};
    const expired = remaining !== null && remaining <= 0;
    const urgent = remaining !== null && remaining <= 30;
    const status = expired
      ? (attempt.status === "error" ? `Close retry pending${attempt.message ? ": " + attempt.message : ""}` : "Market close in progress")
      : "Full-position market close at 3:30";
    const side = pick(pos, "side") || "POSITION";
    const qty = pick(pos, "qty", "size", "volume");
    const entry = pick(pos, "avg_open_price", "avgOpenPrice", "entryPrice", "openPrice");
    return `<div class="bxm-countdown ${urgent ? "urgent" : ""} ${expired ? "expired" : ""}">
      <div><span>3:30 auto-close</span><strong>${remaining === null ? "--:--" : fmtCountdown(remaining)}</strong></div>
      <p>${escapeHtml(side)} ${qty ? escapeHtml(qty) : ""}${entry ? ` @ ${fmtPrice(entry)}` : ""} - ${escapeHtml(status)}</p>
    </div>`;
  }

  function closedTradesHtml(symData) {
    const stats = symData?.closed_trade_stats || symData?.closedTradeStats;
    const rows = symData?.closed_trades || symData?.closedTrades || symData?.trade_history || symData?.tradeHistory || [];
    if (!stats && !rows.length) return "";
    const total = stats?.count ?? rows.length;
    const winRate = stats?.win_rate ?? stats?.winRate;
    const net = stats?.net_pnl ?? stats?.netPnl;
    return `<div class="bxm-history">
      <div class="bxm-history-head">
        <div>
          <div class="bxm-section-title">Closed trades</div>
          <span>${total || 0} trades${winRate !== null && winRate !== undefined ? ` - ${Number(winRate).toFixed(1)}% win` : ""}</span>
        </div>
        <strong class="${num(net) >= 0 ? "good" : "bad"}">${fmtMoney(net)}</strong>
      </div>
      ${rows.slice(0, 4).map((r) => `
        <div class="bxm-trade">
          <strong class="${num(r.net_pnl ?? r.netPnl) >= 0 ? "good" : "bad"}">${fmtMoney(r.net_pnl ?? r.netPnl)}</strong>
          <span>${escapeHtml(r.side || "")} ${fmtPrice(r.entry_price ?? r.entryPrice)} -> ${fmtPrice(r.exit_price ?? r.exitPrice)} - ${fmtPct(r.price_pnl_pct ?? r.pricePnlPct)}</span>
        </div>
      `).join("")}
    </div>`;
  }

  function subRowsHtml(symData) {
    const h = symData?.horizons || {};
    const keys = ["h_15m", "h_30m", "h_1h"];
    return `<div class="bxm-subrows">
      ${keys.map((key) => {
        const row = h[key];
        if (!row) return "";
        const ls = Math.round(num(row.long_score) * 100);
        const ss = Math.round(num(row.short_score) * 100);
        const side = ss > ls ? "SHORT" : ls > ss ? "LONG" : "MIXED";
        const fill = Math.max(ls, ss);
        return `<div class="bxm-subrow">
          <span>${escapeHtml(row.label || key)}</span>
          <strong class="${side === "SHORT" ? "bad" : side === "LONG" ? "good" : ""}">${side}</strong>
          <em>L ${ls} / S ${ss}</em>
          <b><i style="width:${Math.min(100, fill)}%"></i></b>
        </div>`;
      }).join("")}
    </div>`;
  }

  function buildPanel() {
    const root = document.createElement("div");
    root.id = "bxm-overlay";
    if (collapsed) root.classList.add("bxm-collapsed");
    root.innerHTML = `
      <div class="bxm-header" id="bxm-drag">
        <span class="bxm-title">Pump Fade Radar</span>
        <span class="bxm-actions">
          <button id="bxm-refresh" title="Refresh now">R</button>
          <button id="bxm-settings" title="Settings">S</button>
          <button id="bxm-toggle" title="Collapse / expand">${collapsed ? "+" : "-"}</button>
        </span>
      </div>
      <div class="bxm-banner"></div>
      <div class="bxm-body">
        <div id="bxm-tabs" class="bxm-tabs"></div>
        <div id="bxm-symbol"></div>
      </div>
      <div class="bxm-footer"><span id="bxm-status">Connecting...</span><span id="bxm-fresh"></span></div>
    `;
    document.body.appendChild(root);

    let dragOff = null;
    const drag = root.querySelector("#bxm-drag");
    drag.addEventListener("mousedown", (e) => {
      const r = root.getBoundingClientRect();
      dragOff = { x: e.clientX - r.left, y: e.clientY - r.top };
      e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
      if (!dragOff) return;
      const x = Math.max(0, Math.min(window.innerWidth - 80, e.clientX - dragOff.x));
      const y = Math.max(0, Math.min(window.innerHeight - 40, e.clientY - dragOff.y));
      root.style.left = `${x}px`;
      root.style.top = `${y}px`;
      root.style.right = "auto";
    });
    document.addEventListener("mouseup", () => {
      if (!dragOff) return;
      localStorage.setItem("bxm-pos", JSON.stringify({ left: root.style.left, top: root.style.top, width: root.style.width, height: root.style.height }));
      dragOff = null;
    });

    try {
      const pos = JSON.parse(localStorage.getItem("bxm-pos") || "null");
      if (pos) {
        root.style.left = pos.left || "";
        root.style.top = pos.top || "";
        root.style.right = "auto";
        if (pos.width) root.style.width = pos.width;
        if (pos.height) root.style.height = pos.height;
      }
    } catch {}

    root.querySelector("#bxm-toggle").addEventListener("click", () => {
      collapsed = !collapsed;
      localStorage.setItem("bxm-collapsed", collapsed ? "1" : "0");
      root.classList.toggle("bxm-collapsed", collapsed);
      root.querySelector("#bxm-toggle").textContent = collapsed ? "+" : "-";
    });
    root.querySelector("#bxm-settings").addEventListener("click", () => {
      chrome.runtime.sendMessage({ type: "open-options" }).catch(() => {});
    });
    root.querySelector("#bxm-refresh").addEventListener("click", () => {
      chrome.runtime.sendMessage({ type: "force-refresh" }, (resp) => {
        if (resp) {
          latest = resp.payload;
          fetchedAt = resp.fetchedAt || Date.now();
          render();
        }
      });
    });
    return root;
  }

  function render() {
    if (!panelEl) panelEl = buildPanel();
    const tabs = panelEl.querySelector("#bxm-tabs");
    const block = panelEl.querySelector("#bxm-symbol");
    const status = panelEl.querySelector("#bxm-status");
    const fresh = panelEl.querySelector("#bxm-fresh");
    const banner = panelEl.querySelector(".bxm-banner");

    if (!latest) {
      status.textContent = "Waiting for first tick...";
      tabs.innerHTML = "";
      block.innerHTML = "";
      return;
    }

    if (latest.error) {
      tabs.innerHTML = "";
      block.innerHTML = `<div class="bxm-error">
        <strong>${escapeHtml(latest.error)}</strong>
        <p>${escapeHtml(latest.message || "")}</p>
        ${latest.error === "not_configured" ? `<button id="bxm-open-settings">Open settings</button>` : ""}
      </div>`;
      const btn = block.querySelector("#bxm-open-settings");
      if (btn) btn.addEventListener("click", () => chrome.runtime.sendMessage({ type: "open-options" }));
      status.textContent = "Error";
      fresh.textContent = "";
      return;
    }

    const symbols = Object.keys(latest.symbols || {});
    if (!symbols.length) {
      block.innerHTML = `<div class="bxm-error">No overlay data yet. The bot may still be warming up.</div>`;
      tabs.innerHTML = "";
      status.textContent = "No data";
      return;
    }

    if (!activeSymbol || !symbols.includes(activeSymbol)) {
      activeSymbol = symbols[0];
      localStorage.setItem("bxm-active-symbol", activeSymbol);
    }

    tabs.innerHTML = symbols.map((s) => {
      const d = decisionFor(latest.symbols[s]);
      const stage = stageFor(d);
      return `<button class="${s === activeSymbol ? "active" : ""} ${stage}" data-sym="${escapeHtml(s)}">
        <i></i>${escapeHtml(s.replace("USDT", ""))}
      </button>`;
    }).join("");
    tabs.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeSymbol = btn.dataset.sym;
        localStorage.setItem("bxm-active-symbol", activeSymbol);
        render();
      });
    });

    const symData = latest.symbols[activeSymbol] || {};
    const decision = decisionFor(symData);
    const stage = stageFor(decision);
    const score = scoreFor(decision);
    const copy = stageCopy(stage, score);
    const ageSecs = Math.max(0, Math.floor(Date.now() / 1000 - num(symData.as_of)));
    const warnings = decision?.warnings || [];
    const gate = symData?.symbol_trade_quality_gate || symData?.symbolTradeQualityGate;

    panelEl.classList.toggle("bxm-alert", stage === "short");
    panelEl.classList.toggle("bxm-building", stage === "building");
    banner.textContent = stage === "short"
      ? `AUTO SHORT READY - ${score}/100 - ${decision?.suggested_lev || decision?.suggestedLev || 100}x`
      : stage === "building"
        ? `PUMP BUILDING - ${score}/100`
        : "";

    block.innerHTML = `
      <div class="bxm-symbol-row">
        <span>${escapeHtml(activeSymbol)} @ ${fmtPrice(symData.price)}</span>
        <em>${fmtAge(ageSecs)}</em>
      </div>
      <div class="bxm-card stage-${stage}">
        <div class="bxm-card-main">
          <span>${escapeHtml(copy.icon)}</span>
          <strong>${escapeHtml(copy.title)}</strong>
          <b>${stage === "short" ? `${score}/100` : score ? `${score}/49` : ""}</b>
        </div>
        <p><strong>${escapeHtml(copy.kicker)}</strong> - ${escapeHtml(copy.detail)}</p>
      </div>
      ${warnings.length ? `<div class="bxm-warning">${escapeHtml(warnings[0])}</div>` : ""}
      ${gate?.reason ? `<div class="bxm-warning bad">${escapeHtml(gate.reason)}</div>` : ""}
      ${tradePlanHtml(decision, symData)}
      ${countdownHtml(symData, activeSymbol)}
      ${closedTradesHtml(symData)}
      ${subRowsHtml(symData)}
      ${checkRows(decision)}
    `;

    status.textContent = `${symbols.length} symbols - pump-fade shorts - poll ${latest.tick_seconds || 5}s`;
    fresh.textContent = fetchedAt ? `Fetched ${fmtAge(Math.floor((Date.now() - fetchedAt) / 1000))}` : "";
  }

  panelEl = buildPanel();
  render();

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.type !== "momentum-update") return;
    latest = msg.payload;
    fetchedAt = msg.fetchedAt || Date.now();
    render();
  });

  chrome.runtime.sendMessage({ type: "request-latest" }, (resp) => {
    if (!resp) return;
    latest = resp.payload;
    fetchedAt = resp.fetchedAt || 0;
    render();
  });

  setInterval(render, 1000);
})();
