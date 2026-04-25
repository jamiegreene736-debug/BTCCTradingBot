"""Flask dashboard — runs alongside the trading worker on Railway's $PORT.

Routes:
  GET /            HTML dashboard with auto-refresh
  GET /api/state   JSON: balance, open positions, history, recent events
  GET /healthz     plain "ok" for uptime checks (no auth)

Auth: HTTP Basic. Username is hardcoded "admin"; password comes from the
DASHBOARD_PASSWORD env var. If the env var is unset/empty, the dashboard
returns 503 with a clear message — we never serve sensitive data unauthed.
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from base64 import b64decode
from typing import Any

from flask import Flask, Response, jsonify, request

from .client import BitunixClient, BitunixError
from .config import Config
from .state import get as get_state

log = logging.getLogger(__name__)


def create_app(cfg: Config, client: BitunixClient) -> Flask:
    app = Flask(__name__)
    state = get_state()
    password = os.environ.get("DASHBOARD_PASSWORD", "")

    # ------------------------------------------------------------------ auth

    def _check_auth() -> bool:
        if not password:
            return False
        h = request.headers.get("Authorization", "")
        if not h.startswith("Basic "):
            return False
        try:
            user, _, pw = b64decode(h[6:]).decode().partition(":")
        except Exception:
            return False
        return user == "admin" and secrets.compare_digest(pw, password)

    def _unauth() -> Response:
        if not password:
            return Response(
                "Dashboard disabled: set DASHBOARD_PASSWORD env var on Railway "
                "(Variables tab) and redeploy.",
                status=503,
                mimetype="text/plain",
            )
        return Response(
            "Auth required",
            status=401,
            headers={"WWW-Authenticate": 'Basic realm="bitunix-bot"'},
        )

    @app.before_request
    def _gate() -> Response | None:
        if request.path == "/healthz":
            return None
        if not _check_auth():
            return _unauth()
        return None

    # ------------------------------------------------------------------ routes

    @app.get("/healthz")
    def healthz() -> Response:
        return Response("ok", mimetype="text/plain")

    @app.get("/api/state")
    def api_state() -> Response:
        symbols = list(cfg.trading.symbols)
        out: dict[str, Any] = {
            "config": {
                "mode": cfg.mode,
                "symbols": symbols,
                "timeframe": cfg.trading.timeframe,
                "leverage": cfg.trading.leverage,
                "stop_loss_pct": cfg.risk.stop_loss_pct,
                "take_profit_r": cfg.risk.take_profit_r,
                "max_open_positions": cfg.trading.max_open_positions,
                "cooldown_seconds": cfg.trading.cooldown_seconds,
                "min_confluence": cfg.strategy.min_confluence,
                "pattern_weight": cfg.strategy.pattern_weight,
                "fire_threshold": cfg.strategy.fire_threshold,
            },
            "bot": state.snapshot(),
            "now": int(time.time()),
        }
        try:
            out["account"] = client.account()
        except BitunixError as e:
            out["account_error"] = f"{e.code}: {e.msg}"
        except Exception as e:
            out["account_error"] = str(e)

        # Pull positions / history across the whole universe (no symbol filter).
        try:
            positions = client.pending_positions()
            # Merge in TPSL trigger prices (Bitunix stores them as separate orders).
            try:
                tpsl_rows = client.pending_tpsl()
                tp_by_pos: dict[str, str] = {}
                sl_by_pos: dict[str, str] = {}
                for r in tpsl_rows:
                    pid = str(r.get("positionId") or "")
                    if r.get("tpPrice"):
                        tp_by_pos[pid] = r["tpPrice"]
                    if r.get("slPrice"):
                        sl_by_pos[pid] = r["slPrice"]
                for p in positions:
                    pid = str(p.get("positionId") or "")
                    p["slPrice"] = sl_by_pos.get(pid)
                    p["tpPrice"] = tp_by_pos.get(pid)
            except Exception as e:
                out["tpsl_error"] = str(e)
            out["open_positions"] = positions
        except Exception as e:
            out["open_positions_error"] = str(e)
            out["open_positions"] = []

        try:
            hist = client.history_positions(limit=50)
            closed = hist.get("positionList", [])
            out["history_positions"] = closed

            # Win rate: count closed positions where net PnL (realized + fee +
            # funding, all signed) is > 0. Bitunix's `realizedPNL` excludes
            # fees and funding per spec, so add them back signed.
            def _f(v):
                try:
                    return float(v) if v not in (None, "", "null") else 0.0
                except (ValueError, TypeError):
                    return 0.0
            wins = losses = 0
            for p in closed:
                net = _f(p.get("realizedPNL")) + _f(p.get("fee")) + _f(p.get("funding"))
                if net > 0:
                    wins += 1
                elif net < 0:
                    losses += 1
            total = wins + losses
            out["win_rate"] = {
                "wins": wins,
                "losses": losses,
                "total": total,
                "rate": round(wins / total * 100, 1) if total else None,
            }
        except Exception as e:
            out["history_positions_error"] = str(e)
            out["history_positions"] = []
            out["win_rate"] = {"wins": 0, "losses": 0, "total": 0, "rate": None}

        return jsonify(out)

    @app.get("/")
    def index() -> Response:
        return Response(_HTML, mimetype="text/html")

    return app


# Self-contained HTML/JS dashboard. Refreshes /api/state every 10s.
_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bitunix TraderBot</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { color-scheme: dark; }
  body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background:#0d1117; color:#e6edf3; margin:0; padding:24px; max-width:1100px; margin:auto; }
  h1 { margin:0 0 8px; font-size:20px; }
  h2 { margin:24px 0 8px; font-size:14px; text-transform:uppercase; color:#8b949e; letter-spacing:0.5px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; margin-right:6px; }
  .pill.live { background:#da3633; color:#fff; }
  .pill.paper { background:#1f6feb; color:#fff; }
  .pill.muted { background:#21262d; color:#8b949e; }
  .pill.pos   { background:#1f7a3a; color:#fff; }
  .pill.neg-pill { background:#7a2326; color:#fff; }
  table { width:100%; border-collapse:collapse; margin-top:6px; }
  th, td { text-align:left; padding:6px 10px; border-bottom:1px solid #21262d; font-variant-numeric:tabular-nums; }
  th { color:#8b949e; font-weight:500; font-size:12px; text-transform:uppercase; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }
  .card { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:12px; }
  .card .label { color:#8b949e; font-size:12px; text-transform:uppercase; }
  .card .val { font-size:20px; margin-top:4px; font-variant-numeric:tabular-nums; }
  .pos { color:#3fb950; }
  .neg { color:#f85149; }
  .ev-signal { color:#58a6ff; }
  .ev-order  { color:#3fb950; }
  .ev-skip   { color:#8b949e; }
  .ev-error  { color:#f85149; }
  .small { font-size:12px; color:#8b949e; }
  .err { background:#3a1a1a; padding:6px 10px; border-radius:6px; margin-top:6px; font-size:12px; color:#ffa198; }
  code { background:#21262d; padding:2px 6px; border-radius:4px; font-size:12px; }
</style>
</head>
<body>
<h1>Bitunix TraderBot</h1>
<div id="header" class="small"></div>

<h2>Account</h2>
<div id="account" class="grid"></div>
<div id="account-error"></div>

<h2>Open positions</h2>
<table id="open-positions"><thead></thead><tbody></tbody></table>

<h2>Recent bot activity</h2>
<table id="events"><thead><tr><th>Time</th><th>Kind</th><th>Detail</th></tr></thead><tbody></tbody></table>

<h2>Closed positions</h2>
<table id="hist-pos"><thead></thead><tbody></tbody></table>

<script>
const fmt = (n, d=2) => (n===null || n===undefined || n==='') ? '—' : Number(n).toLocaleString(undefined,{maximumFractionDigits:d});
const tsfmt = ms => { if (!ms) return '—'; const d = new Date(typeof ms==='string' ? Number(ms) : ms); return d.toLocaleString(); };
const sec = ts => ts ? new Date(ts*1000).toLocaleTimeString() : '—';
const pnl = v => { const n = Number(v||0); const cls = n>0?'pos':n<0?'neg':''; return `<span class="${cls}">${fmt(n,2)}</span>`; };

function renderHeader(s) {
  const cfg = s.config, b = s.bot, wr = s.win_rate || {};
  const modePill = cfg.mode === 'live' ? `<span class="pill live">LIVE</span>` : `<span class="pill paper">PAPER</span>`;
  const syms = (cfg.symbols || []).join(', ');
  const patPct = Math.round((cfg.pattern_weight || 0) * 100);
  let wrPill;
  if (!wr.total) {
    wrPill = `<span class="pill muted">win rate — (no closed trades yet)</span>`;
  } else {
    const cls = wr.rate >= 50 ? 'pos' : wr.rate >= 35 ? 'muted' : 'neg-pill';
    wrPill = `<span class="pill ${cls}">win rate ${wr.rate}% · ${wr.wins}W / ${wr.losses}L</span>`;
  }
  document.getElementById('header').innerHTML =
    modePill +
    `<span class="pill muted">${syms} · ${cfg.timeframe} · ${cfg.leverage}x</span>` +
    `<span class="pill muted">SL ${cfg.stop_loss_pct}% / TP ${cfg.take_profit_r}R</span>` +
    `<span class="pill muted">${patPct}% candle patterns + ${100-patPct}% indicators · fire ≥ ${cfg.fire_threshold}</span>` +
    wrPill +
    `<span class="pill muted">${b.signal_count} signals · ${b.order_count} orders · ${b.error_count} errors</span>` +
    `<span class="pill muted">last tick ${sec(b.last_tick_at)}</span>`;
}

function renderAccount(s) {
  const c = document.getElementById('account');
  document.getElementById('account-error').innerHTML =
    s.account_error ? `<div class="err">Account error: ${s.account_error}</div>` : '';
  if (!s.account) { c.innerHTML = ''; return; }
  const a = s.account;
  const num = v => Number(v||0);
  const upnl = num(a.crossUnrealizedPNL) + num(a.isolationUnrealizedPNL);
  const equity = num(a.available) + num(a.margin) + upnl;
  const coin = a.marginCoin || '';
  const cards = [
    ['Available', fmt(a.available, 2) + ' ' + coin],
    ['In margin', fmt(a.margin, 2) + ' ' + coin],
    ['Open uPnL', pnl(upnl)],
    ['Equity', fmt(equity, 2) + ' ' + coin],
  ];
  c.innerHTML = cards.map(([l,v]) => `<div class="card"><div class="label">${l}</div><div class="val">${v}</div></div>`).join('');
}

function renderOpen(s) {
  const t = document.getElementById('open-positions');
  const rows = s.open_positions || [];
  t.querySelector('thead').innerHTML = `<tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>SL</th><th>TP</th><th>uPnL</th><th>Liq</th></tr>`;
  if (s.open_positions_error) { t.querySelector('tbody').innerHTML = `<tr><td colspan="8" class="ev-error">API error: ${s.open_positions_error}</td></tr>`; return; }
  if (!rows.length) { t.querySelector('tbody').innerHTML = `<tr><td colspan="8" class="small">No open positions.</td></tr>`; return; }
  t.querySelector('tbody').innerHTML = rows.map(p => {
    const sideUp = (p.side==='BUY' || p.side==='LONG');
    const sideClass = sideUp ? 'pos' : 'neg';
    const sl = p.slPrice ? fmt(p.slPrice,4) : '<span class="small">—</span>';
    const tp = p.tpPrice ? fmt(p.tpPrice,4) : '<span class="small">—</span>';
    return `
    <tr>
      <td>${p.symbol||''}</td>
      <td class="${sideClass}">${p.side||''}</td>
      <td>${fmt(p.qty,4)}</td>
      <td>${fmt(p.avgOpenPrice,4)}</td>
      <td class="neg">${sl}</td>
      <td class="pos">${tp}</td>
      <td>${pnl(p.unrealizedPNL)}</td>
      <td>${fmt(p.liqPrice,4)}</td>
    </tr>`;
  }).join('');
}

function renderEvents(s) {
  const t = document.getElementById('events');
  const rows = (s.bot && s.bot.events) || [];
  if (!rows.length) { t.querySelector('tbody').innerHTML = `<tr><td colspan="3" class="small">No activity yet — first tick happens within ~15s.</td></tr>`; return; }
  t.querySelector('tbody').innerHTML = rows.map(e => `
    <tr>
      <td>${sec(e.ts)}</td>
      <td class="ev-${e.kind}">${e.kind}</td>
      <td>${e.text}</td>
    </tr>
  `).join('');
}

function renderHistPos(s) {
  const t = document.getElementById('hist-pos');
  const rows = s.history_positions || [];
  t.querySelector('thead').innerHTML = `<tr><th>Closed</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Close</th><th>Realized PnL</th><th>Fee</th><th>Lev</th></tr>`;
  if (s.history_positions_error) { t.querySelector('tbody').innerHTML = `<tr><td colspan="9" class="ev-error">API error: ${s.history_positions_error}</td></tr>`; return; }
  if (!rows.length) { t.querySelector('tbody').innerHTML = `<tr><td colspan="9" class="small">No closed positions yet (paper mode never reaches Bitunix).</td></tr>`; return; }
  t.querySelector('tbody').innerHTML = rows.map(p => `
    <tr>
      <td>${tsfmt(p.mtime)}</td>
      <td>${p.symbol||''}</td>
      <td class="${p.side==='LONG'?'pos':p.side==='SHORT'?'neg':''}">${p.side||''}</td>
      <td>${fmt(p.maxQty,4)}</td>
      <td>${fmt(p.entryPrice,2)}</td>
      <td>${fmt(p.closePrice,2)}</td>
      <td>${pnl(p.realizedPNL)}</td>
      <td>${fmt(p.fee,4)}</td>
      <td>${p.leverage||''}x</td>
    </tr>
  `).join('');
}


async function refresh() {
  try {
    const r = await fetch('/api/state', { cache: 'no-store' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const s = await r.json();
    renderHeader(s); renderAccount(s); renderOpen(s); renderEvents(s); renderHistPos(s);
  } catch (e) {
    document.getElementById('header').innerHTML = `<div class="err">Refresh failed: ${e}</div>`;
  }
}
refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""
