// Bitunix pump-fade overlay content script.
// Renders the floating scanner panel and receives data from the MV3
// background worker. All DOM/CSS names are prefixed with bxm-.

(function () {
  if (window.__bxmOverlayInstalled) return;
  window.__bxmOverlayInstalled = true;

  let latest = null;
  let fetchedAt = 0;
  let activeSymbol = localStorage.getItem("bxm-active-symbol") || "";
  const AUTO_FOLLOW_KEY = "bxm-auto-follow-v2";
  let autoFollow = localStorage.getItem(AUTO_FOLLOW_KEY) !== "0";
  let collapsed = localStorage.getItem("bxm-collapsed") === "1";
  let panelEl = null;
  let lastAutoSwitchAt = 0;
  let manualHoldUntil = 0;
  let titleFlashTimer = null;
  let titleFlashKey = "";
  let titleFlashPrefix = "";
  let titleFlashOn = false;
  let cleanPageTitle = document.title;
  let pollMs = 0;
  const ALERT_HISTORY_KEY = "bxm-pump-alert-history-v1";
  let alertHistory = loadAlertHistory();
  let activeAlertStages = {};
  let lastAlertScanAt = 0;

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
    if (value === null || value === undefined || value === "") return "--";
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

  function fmtClock(ts) {
    const d = new Date(Number(ts) || Date.now());
    return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
  }

  function loadAlertHistory() {
    try {
      const rows = JSON.parse(localStorage.getItem(ALERT_HISTORY_KEY) || "[]");
      return Array.isArray(rows) ? rows.slice(0, 5) : [];
    } catch {
      return [];
    }
  }

  function saveAlertHistory() {
    localStorage.setItem(ALERT_HISTORY_KEY, JSON.stringify(alertHistory.slice(0, 5)));
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

  function decisionFor(symData) {
    return symData?.decision || symData?.recommendation || symData?.next_1h ||
      symData?.next1h || symData?.next_15m || symData?.next15m || {};
  }

  function tradePlanFor(decision, symData) {
    return decision?.trade_plan || decision?.tradePlan || symData?.trade_plan || symData?.tradePlan || null;
  }

  function simulationFromPlan(plan) {
    return plan?.max_leverage_simulation || plan?.maxLeverageSimulation ||
      plan?.estimated_pnl || plan?.estimatedPnl || plan?.simulation || null;
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

  function stagePriority(stage) {
    return { short: 4, watch: 3, building: 2, hunting: 1 }[stage] || 0;
  }

  function candidateFor(symbol, symData) {
    const decision = decisionFor(symData);
    const stage = stageFor(decision);
    const score = scoreFor(decision);
    const priority = stagePriority(stage);
    const blocked = Boolean(
      symData?.symbol_trade_quality_gate ||
      symData?.symbolTradeQualityGate ||
      pick(decision, "blocked_by", "blockedBy")
    );
    const rankScore = (priority * 100) + score - (blocked ? 1000 : 0);
    return {
      symbol,
      decision,
      stage,
      score,
      priority,
      rankScore,
      blocked,
    };
  }

  function rankedCandidates(symbols) {
    return symbols
      .map((s) => candidateFor(s, latest?.symbols?.[s] || {}))
      .sort((a, b) => {
        if (b.rankScore !== a.rankScore) return b.rankScore - a.rankScore;
        if (b.score !== a.score) return b.score - a.score;
        return a.symbol.localeCompare(b.symbol);
      });
  }

  function bestCandidate(candidates) {
    return candidates.find((c) => !c.blocked && c.stage !== "hunting") || candidates[0] || null;
  }

  function isPumpAlertStage(stage) {
    return stage === "building" || stage === "watch" || stage === "short";
  }

  function candidateScoreText(c) {
    if (!c) return "";
    if (c.stage === "short") return `${Math.round(c.score)}/100`;
    if (c.stage === "watch" || c.stage === "building") return `${Math.round(c.score)}/49`;
    return c.score ? `${Math.round(c.score)}/49` : "0/49";
  }

  function alertStageLabel(stage) {
    if (stage === "short") return "FADE SHORT";
    if (stage === "watch") return "PUMP WATCH";
    if (stage === "building") return "PUMP BUILDING";
    return "HUNTING";
  }

  function rememberPumpAlerts(candidates) {
    if (!latest || !fetchedAt || lastAlertScanAt === fetchedAt) return;
    lastAlertScanAt = fetchedAt;
    const now = Date.now();
    const currentAlerts = new Set();
    let changed = false;

    for (const c of candidates) {
      if (!c || !isPumpAlertStage(c.stage)) continue;
      currentAlerts.add(c.symbol);
      if (activeAlertStages[c.symbol] === c.stage) continue;

      const recentDuplicate = alertHistory.some((row) =>
        row.symbol === c.symbol && row.stage === c.stage && now - Number(row.ts || 0) < 120000
      );
      activeAlertStages[c.symbol] = c.stage;
      if (recentDuplicate) continue;

      const symData = latest.symbols?.[c.symbol] || {};
      const eta = c.decision?.fade_eta || c.decision?.fadeEta || {};
      const plan = tradePlanFor(c.decision, symData);
      const sim = simulationFromPlan(plan);
      alertHistory.unshift({
        ts: now,
        symbol: c.symbol,
        stage: c.stage,
        label: alertStageLabel(c.stage),
        scoreText: candidateScoreText(c),
        price: symData.price,
        eta: eta.label || "",
        simEntryPrice: pick(sim, "entry_price", "entryPrice", "limit_price", "limitPrice") || pick(plan, "entry_price", "entryPrice"),
        simTakeProfit: pick(sim, "take_profit", "takeProfit", "target_exit_price", "targetExitPrice") || pick(plan, "take_profit", "takeProfit", "target_exit_price", "targetExitPrice"),
        simStopLoss: pick(sim, "recommended_stop_loss", "recommendedStopLoss", "stop_loss", "stopLoss") || pick(plan, "stop_loss", "stopLoss"),
      });
      changed = true;
    }

    for (const symbol of Object.keys(activeAlertStages)) {
      if (!currentAlerts.has(symbol)) delete activeAlertStages[symbol];
    }

    if (changed) {
      alertHistory = alertHistory.slice(0, 5);
      saveAlertHistory();
    }
  }

  function maybeAutoSelectBest(candidates) {
    const best = bestCandidate(candidates);
    if (!autoFollow || !best) return best;
    const now = Date.now();
    if (best.symbol === activeSymbol) return best;

    // Pump-fade scalps are time-sensitive. In AUTO mode, any live
    // building/watch/ready candidate should take over the panel immediately;
    // the old cooldown/manual-hold behavior was too slow for the pump phase.
    if (isPumpAlertStage(best.stage) && !best.blocked) {
      activeSymbol = best.symbol;
      localStorage.setItem("bxm-active-symbol", activeSymbol);
      manualHoldUntil = 0;
      lastAutoSwitchAt = now;
      return best;
    }

    const current = candidateFor(activeSymbol, latest?.symbols?.[activeSymbol] || {});
    if (manualHoldUntil > now) return best;
    const actionable = best.stage !== "hunting";
    const cooledDown = now - lastAutoSwitchAt >= (actionable ? 5000 : 15000);
    const stageUpgrade = best.priority > current.priority;
    const scoreUpgrade = best.priority === current.priority && best.rankScore - current.rankScore >= 15;
    const huntingUpgrade = !actionable && current.stage === "hunting" && best.score >= current.score + 20;
    const currentIsDead = !current.symbol || !latest?.symbols?.[activeSymbol];
    if (cooledDown && (stageUpgrade || scoreUpgrade || huntingUpgrade || currentIsDead)) {
      activeSymbol = best.symbol;
      localStorage.setItem("bxm-active-symbol", activeSymbol);
      lastAutoSwitchAt = now;
    }
    return best;
  }

  function stopTitleFlash() {
    if (titleFlashTimer) clearInterval(titleFlashTimer);
    titleFlashTimer = null;
    titleFlashKey = "";
    titleFlashPrefix = "";
    titleFlashOn = false;
    if (
      document.title.startsWith("[PUMP BUILDING ")
      || document.title.startsWith("[PUMP WATCH ")
      || document.title.startsWith("[FADE SHORT ")
    ) {
      document.title = cleanPageTitle;
    }
  }

  function ensureTitleFlash(candidate) {
    if (!candidate || !isPumpAlertStage(candidate.stage)) {
      stopTitleFlash();
      return;
    }
    const key = `${candidate.stage}:${candidate.symbol}`;
    const label = candidate.stage === "short"
      ? "FADE SHORT"
      : candidate.stage === "watch"
        ? "PUMP WATCH"
        : "PUMP BUILDING";
    const symbol = candidate.symbol.replace("USDT", "");
    titleFlashPrefix = `[${label} ${symbol}]`;
    if (titleFlashKey !== key) {
      cleanPageTitle = document.title
        .replace(/^\[PUMP BUILDING [^\]]+\] /, "")
        .replace(/^\[PUMP WATCH [^\]]+\] /, "")
        .replace(/^\[FADE SHORT [^\]]+\] /, "");
      titleFlashKey = key;
      titleFlashOn = false;
    }
    if (titleFlashTimer) return;
    titleFlashTimer = setInterval(() => {
      titleFlashOn = !titleFlashOn;
      document.title = titleFlashOn ? `${titleFlashPrefix} ${cleanPageTitle}` : cleanPageTitle;
    }, 700);
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
      kicker: "WATCH NOW",
      detail: "buyers are pressing before the fade; no short until rejection",
      icon: "^",
    };
    if (stage === "watch") return {
      title: "PUMP WATCH",
      kicker: "GET READY",
      detail: "pump detected now; prepare for the fast fade-short entry",
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
    const plan = tradePlanFor(decision, symData);
    if (!plan || !["ready", "preview"].includes(String(plan.status || ""))) return "";
    const preview = plan.status === "preview" || plan.preview === true;
    const orderType = String(pick(plan, "order_type", "orderType") || "MARKET").replace(/_/g, " ");
    const entry = Number(pick(plan, "entry_price", "entryPrice"));
    const target = Number(pick(plan, "target_exit_price", "targetExitPrice", "max_exit_price", "maxExitPrice", "take_profit", "takeProfit"));
    const stop = Number(pick(plan, "stop_loss", "stopLoss"));
    const rewardPct = pick(plan, "reward_pct", "rewardPct");
    const riskPct = pick(plan, "risk_pct", "riskPct");
    const entryLabel = preview ? "Suggested short entry trigger" : "Suggested short entry";
    const targetLabel = preview ? "Preview take profit" : "Take profit / suggested exit";
    const stopLabel = preview ? "Preview stop loss" : "Stop loss";
    const rewardLabel = "Target move";
    const riskLabel = "Stop distance";
    return `<div class="bxm-plan ${preview ? "preview" : ""}">
      <div class="bxm-plan-head">
        <span>${entryLabel}</span>
        <strong>${escapeHtml(orderType)}</strong>
      </div>
      <div class="bxm-plan-price">${fmtPrice(entry)}</div>
      <div class="bxm-plan-grid">
        <div><span>${targetLabel}</span><strong class="good">${fmtPrice(target)}</strong></div>
        <div><span>${stopLabel}</span><strong class="bad">${fmtPrice(stop)}</strong></div>
        <div><span>${rewardLabel}</span><strong>${fmtPct(rewardPct)}</strong></div>
        <div><span>${riskLabel}</span><strong>${fmtPct(riskPct)}</strong></div>
      </div>
      ${plan.rationale ? `<div class="bxm-note">${escapeHtml(plan.rationale)}</div>` : ""}
    </div>`;
  }

  function fadeEtaHtml(decision) {
    const eta = decision?.fade_eta || decision?.fadeEta;
    if (!eta || !eta.label || eta.label === "--") return "";
    const status = String(eta.status || "watch");
    const reason = eta.reason || "rough timing estimate from pump speed and tape";
    return `<div class="bxm-eta status-${escapeHtml(status)}">
      <span>Fade ETA</span>
      <strong>${escapeHtml(eta.label)}</strong>
      <em>${escapeHtml(reason)}</em>
    </div>`;
  }

  function countdownHtml(symData, symbol) {
    const pos = activeOpenPosition(symData, symbol);
    if (!pos) return "";
    const remaining = positionCountdown(pos);
    const closeAfter = Number(pick(pos, "auto_close_after_seconds", "autoCloseAfterSeconds"));
    const timedCloseEnabled = Number.isFinite(closeAfter) && closeAfter > 0 && remaining !== null;
    const expired = timedCloseEnabled && remaining <= 0;
    const urgent = timedCloseEnabled && remaining <= 30;
    const status = timedCloseEnabled
      ? "Timed auto-close armed"
      : "Timed auto-close disabled";
    const side = pick(pos, "side") || "POSITION";
    const qty = pick(pos, "qty", "size", "volume");
    const entry = pick(pos, "avg_open_price", "avgOpenPrice", "entryPrice", "openPrice");
    return `<div class="bxm-countdown ${urgent ? "urgent" : ""} ${expired ? "expired" : ""}">
      <div><span>Open position</span><strong>${timedCloseEnabled ? fmtCountdown(remaining) : "manual"}</strong></div>
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
        <strong class="${num(net) >= 0 ? "good" : "bad"}">P&L ${fmtMoney(net)}</strong>
      </div>
      ${rows.slice(0, 4).map((r) => `
        <div class="bxm-trade">
          <strong class="${num(r.net_pnl ?? r.netPnl) >= 0 ? "good" : "bad"}">${fmtMoney(r.net_pnl ?? r.netPnl)}</strong>
          <span>${escapeHtml(r.side || "")} ${fmtPrice(r.entry_price ?? r.entryPrice)} -> ${fmtPrice(r.exit_price ?? r.exitPrice)} - ${fmtPct(r.price_pnl_pct ?? r.pricePnlPct)}</span>
        </div>
      `).join("")}
    </div>`;
  }

  function alertHistoryHtml() {
    return `<div class="bxm-alert-history">
      <div class="bxm-alert-history-head">
        <div class="bxm-section-title">Signal history - no P&L</div>
        <span>last 5</span>
      </div>
      <p class="bxm-alert-disclaimer">Alerts only. Estimated or actual P&L is shown only after a trade is closed.</p>
      ${alertHistory.length ? alertHistory.map((row) => `
        <div class="bxm-alert-event stage-${escapeHtml(row.stage || "")}">
          <time>${escapeHtml(fmtClock(row.ts))}</time>
          <div>
            <strong>${escapeHtml((row.symbol || "").replace("USDT", ""))} ${escapeHtml(row.label || alertStageLabel(row.stage))}</strong>
            <span>${escapeHtml(row.scoreText || "")}${row.eta ? ` - ETA ${escapeHtml(row.eta)}` : ""}${row.price ? ` - @ ${fmtPrice(row.price)}` : ""}</span>
            ${(row.simEntryPrice || row.simTakeProfit || row.simStopLoss) ? `<small>
              Levels: entry ${fmtPrice(row.simEntryPrice)} - TP ${fmtPrice(row.simTakeProfit)} - stop ${fmtPrice(row.simStopLoss)}
            </small>` : ""}
          </div>
        </div>
      `).join("") : `<div class="bxm-alert-empty">No pump warnings logged yet.</div>`}
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
          <button id="bxm-auto" title="Auto-select strongest pump-fade candidate">AUTO</button>
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
    const autoBtn = root.querySelector("#bxm-auto");
    autoBtn.classList.toggle("active", autoFollow);
    autoBtn.addEventListener("click", () => {
      autoFollow = !autoFollow;
      localStorage.setItem(AUTO_FOLLOW_KEY, autoFollow ? "1" : "0");
      manualHoldUntil = 0;
      autoBtn.classList.toggle("active", autoFollow);
      render();
    });
    root.querySelector("#bxm-refresh").addEventListener("click", () => {
      chrome.runtime.sendMessage({ type: "force-refresh" }, (resp) => {
        if (resp) {
          latest = resp.payload;
          fetchedAt = resp.fetchedAt || Date.now();
          pollMs = resp.pollMs || pollMs;
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

    const candidates = rankedCandidates(symbols);
    rememberPumpAlerts(candidates);
    const best = maybeAutoSelectBest(candidates);
    const orderedSymbols = candidates.map((c) => c.symbol);
    panelEl.querySelector("#bxm-auto")?.classList.toggle("active", autoFollow);
    const autoSelected = Boolean(autoFollow && best && best.symbol === activeSymbol);
    panelEl.classList.toggle("bxm-auto-selected", autoSelected);

    tabs.innerHTML = orderedSymbols.map((s) => {
      const c = candidateFor(s, latest.symbols[s]);
      return `<button class="${s === activeSymbol ? "active" : ""} ${best && s === best.symbol ? "best" : ""} ${c.stage}" data-sym="${escapeHtml(s)}" title="${escapeHtml(c.stage.toUpperCase())} ${escapeHtml(candidateScoreText(c))}">
        <i></i>${escapeHtml(s.replace("USDT", ""))}
      </button>`;
    }).join("");
    tabs.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeSymbol = btn.dataset.sym;
        localStorage.setItem("bxm-active-symbol", activeSymbol);
        manualHoldUntil = Date.now() + 60000;
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
    const bestStage = best?.stage || "";
    const bestText = best
      ? `${best.symbol.replace("USDT", "")} - ${best.stage.toUpperCase()} - ${candidateScoreText(best)}`
      : "";
    const bestLabel = autoSelected ? "AUTO SELECTED" : (autoFollow ? "AUTO BEST" : "BEST NOW");
    const alertCandidate = best && isPumpAlertStage(best.stage) ? best : null;

    panelEl.classList.toggle("bxm-alert", alertCandidate?.stage === "short");
    panelEl.classList.toggle("bxm-watch", alertCandidate?.stage === "watch");
    panelEl.classList.toggle("bxm-building", alertCandidate?.stage === "building" || (!alertCandidate && stage === "building"));
    banner.textContent = alertCandidate?.stage === "short"
      ? `FADE SHORT READY - ${alertCandidate.symbol.replace("USDT", "")} - ${candidateScoreText(alertCandidate)} - ${alertCandidate.decision?.suggested_lev || alertCandidate.decision?.suggestedLev || 100}x`
      : alertCandidate?.stage === "watch"
        ? `PUMP WATCH - ${alertCandidate.symbol.replace("USDT", "")} - ${candidateScoreText(alertCandidate)} - GET READY`
        : alertCandidate?.stage === "building"
          ? `PUMP BUILDING - ${alertCandidate.symbol.replace("USDT", "")} - ${candidateScoreText(alertCandidate)} - WATCH NOW`
        : stage === "building"
          ? `PUMP BUILDING - ${score}/49`
          : "";
    ensureTitleFlash(alertCandidate);

    block.innerHTML = `
      ${best ? `<button id="bxm-best-pick" class="bxm-best stage-${escapeHtml(bestStage)} ${best.symbol === activeSymbol ? "active" : ""}" title="Select the strongest current candidate">
        <span>${escapeHtml(bestLabel)}</span>
        <strong>${escapeHtml(bestText)}</strong>
      </button>` : ""}
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
      ${fadeEtaHtml(decision)}
      ${warnings.length ? `<div class="bxm-warning">${escapeHtml(warnings[0])}</div>` : ""}
      ${gate?.reason ? `<div class="bxm-warning bad">${escapeHtml(gate.reason)}</div>` : ""}
      ${tradePlanHtml(decision, symData)}
      ${alertHistoryHtml()}
      ${countdownHtml(symData, activeSymbol)}
      ${closedTradesHtml(symData)}
      ${subRowsHtml(symData)}
      ${checkRows(decision)}
    `;
    const bestBtn = block.querySelector("#bxm-best-pick");
    if (bestBtn && best) {
      bestBtn.addEventListener("click", () => {
        activeSymbol = best.symbol;
        localStorage.setItem("bxm-active-symbol", activeSymbol);
        manualHoldUntil = 0;
        render();
      });
    }

    const refreshText = pollMs ? `refresh ${(pollMs / 1000).toFixed(pollMs < 2000 ? 1 : 0)}s` : "refresh --";
    status.textContent = `${symbols.length} symbols - ${autoFollow ? "auto-best on" : "manual select"} - ${refreshText} - bot ${latest.tick_seconds || 5}s`;
    fresh.textContent = fetchedAt ? `Fetched ${fmtAge(Math.floor((Date.now() - fetchedAt) / 1000))}` : "";
  }

  panelEl = buildPanel();
  render();

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.type !== "momentum-update") return;
    latest = msg.payload;
    fetchedAt = msg.fetchedAt || Date.now();
    pollMs = msg.pollMs || pollMs;
    render();
  });

  chrome.runtime.sendMessage({ type: "request-latest" }, (resp) => {
    if (!resp) return;
    latest = resp.payload;
    fetchedAt = resp.fetchedAt || 0;
    pollMs = resp.pollMs || pollMs;
    render();
  });

  setInterval(render, 1000);
})();
