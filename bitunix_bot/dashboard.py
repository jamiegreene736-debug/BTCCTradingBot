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

import json
import logging
import os
import secrets
import time
from base64 import b64decode
from collections import deque
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file

from .client import BitunixClient, BitunixError
from .config import Config
from .state import get as get_state

log = logging.getLogger(__name__)


def create_app(cfg: Config, client: BitunixClient, bot: Any = None) -> Flask:
    """Build the Flask app.

    `bot` is the BitunixBot instance (optional — when provided, enables
    admin endpoints that mutate live in-memory state, e.g. resetting
    streak pauses without a full process restart). When omitted (e.g. in
    unit tests), admin endpoints return 503.
    """
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

    # ------------------------------------------------------------------ CORS
    #
    # The Chrome-extension overlay (running on bitunix.com pages) calls this
    # API across origins. Browsers send a CORS preflight (OPTIONS) before
    # any cross-origin GET that includes auth headers; we answer permissively
    # for a single-user dashboard, including the Authorization header.

    @app.after_request
    def _cors(resp: Response) -> Response:
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        resp.headers["Access-Control-Max-Age"] = "600"
        return resp

    @app.before_request
    def _gate() -> Response | None:
        # Preflights must succeed before the browser sends the auth header,
        # so let OPTIONS through without auth — the after_request hook adds
        # the CORS response headers.
        if request.method == "OPTIONS":
            return Response("", status=204)
        if request.path == "/healthz":
            return None
        if not _check_auth():
            return _unauth()
        return None

    # ------------------------------------------------------------------ routes

    @app.get("/healthz")
    def healthz() -> Response:
        # Fail loudly if the tick loop has stalled. Railway / external uptime
        # checks should treat 5xx as "redeploy this thing".
        snap = state.snapshot()
        last = snap.get("last_tick_at") or 0
        # If the bot has never ticked yet (just started), allow 60s grace.
        # After that, fail if no tick within 3× the configured tick interval.
        startup_grace = 60
        max_silence = max(startup_grace, cfg.loop.tick_seconds * 3)
        age = int(time.time()) - last
        if last and age > max_silence:
            return Response(f"stale: last tick {age}s ago", status=503,
                            mimetype="text/plain")
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

    @app.get("/api/journal")
    def journal_recent() -> Response:
        """Return recent trade-journal events as JSON.

        Query params:
          limit  — max events to return (default 50, capped at 1000)
          kind   — filter to "entry" or "exit" (default: both)
          since  — unix timestamp; only events after this time

        The journal is the structured per-trade log that captures: signal
        score, threshold used, conviction multiplier, indicator vote count,
        per-factor breakdown (T/M/F/C), all reason tags, ATR%, ADX, spread,
        depth, tape signals (CVD/aggression/activity), session weight,
        adaptive_adj, recent-trade R-sum, entry/SL/TP, leverage. Plus
        exit-side: hold time, max favorable R, net PnL, fees.
        """
        if bot is None or not hasattr(bot, "journal"):
            return Response("journal not available (bot not wired)",
                            status=503, mimetype="text/plain")
        path = Path(bot.journal.path)
        if not path.exists():
            return jsonify({"events": [], "count": 0,
                             "note": "no journal yet — bot hasn't traded since deploy"})

        try:
            limit = max(1, min(1000, int(request.args.get("limit", 50))))
        except ValueError:
            limit = 50
        kind_filter = request.args.get("kind")  # "entry" | "exit" | None
        try:
            since = float(request.args.get("since", 0))
        except ValueError:
            since = 0

        # Stream-parse and keep only the most recent N matching events.
        buf: deque[dict[str, Any]] = deque(maxlen=limit)
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if kind_filter and ev.get("kind") != kind_filter:
                        continue
                    if since and float(ev.get("ts") or 0) < since:
                        continue
                    buf.append(ev)
        except Exception as e:
            return Response(f"error reading journal: {e}",
                            status=500, mimetype="text/plain")
        return jsonify({"events": list(buf), "count": len(buf),
                         "path": str(path)})

    @app.get("/api/journal/download")
    def journal_download() -> Response:
        """Download the full trade-journal JSONL file. Useful for offline
        analysis (jq, pandas, Excel)."""
        if bot is None or not hasattr(bot, "journal"):
            return Response("journal not available", status=503,
                            mimetype="text/plain")
        path = Path(bot.journal.path)
        if not path.exists():
            return Response("no journal yet", status=404,
                            mimetype="text/plain")
        return send_file(str(path), mimetype="application/x-ndjson",
                         as_attachment=True, download_name="trades.jsonl")

    @app.get("/api/feeds/status")
    def feeds_status() -> Response:
        """Diagnostic snapshot of both WebSocket feeds (OB + trade tape).

        Returns connection lifecycle counters, per-symbol freshness, and
        captured samples of unrecognized payloads — invaluable when one
        of the feeds silently stops delivering data (e.g. Bitunix changes
        a payload schema, or Railway's networking interferes with WSS).

        Read this when you see /api/journal entries with tob_bid=null +
        order_type=MARKET — that's the OB feed being offline forcing the
        bot to fall through to taker market entries.
        """
        if bot is None:
            return Response("bot reference not wired (admin endpoint disabled)",
                            status=503, mimetype="text/plain")
        ob_status = bot.ob_feed.get_status() if getattr(bot, "ob_feed", None) else None
        tape_status = bot.tape_feed.get_status() if getattr(bot, "tape_feed", None) else None
        return jsonify({
            "now": time.time(),
            "ob_feed": ob_status,
            "tape_feed": tape_status,
        })

    @app.post("/api/admin/reset-streaks")
    def reset_streaks() -> Response:
        """Clear all streak / mini-cooldown / consecutive-loss state on the
        live bot. Useful after a chop session pauses every symbol — lets
        the bot resume trading without a full process restart.

        Does NOT clear: cascade detector (price-driven, would re-trip if
        chaos persists), daily DD breaker (equity-driven, recovers when
        equity does), recent_trade_r (the rolling tally that drives
        adaptive self-defense; that's the system reacting to your actual
        performance — overriding it would defeat the adaptive layer).

        Auth-required; returns 503 if the dashboard wasn't constructed
        with a bot reference (e.g. in unit tests).
        """
        if bot is None:
            return Response("admin endpoints not available", status=503,
                            mimetype="text/plain")

        cleared_streak = list(bot.streak_pause_until.keys())
        cleared_mini = list(bot.mini_cooldown_until.keys())
        cleared_consec = dict(bot.consec_losses)
        cleared_recent = {k: len(v) for k, v in bot.recent_losses.items()}

        bot.streak_pause_until.clear()
        bot.mini_cooldown_until.clear()
        bot.consec_losses.clear()
        bot.recent_losses.clear()

        log.warning("ADMIN: reset-streaks cleared streak=%s mini=%s "
                    "consec=%s recent_losses=%s",
                    cleared_streak, cleared_mini, cleared_consec, cleared_recent)
        state.record_order(
            "ADMIN reset-streaks: cleared "
            f"{len(cleared_streak)} streak pauses, "
            f"{len(cleared_mini)} mini-cooldowns"
        )
        return jsonify({
            "ok": True,
            "cleared": {
                "streak_pauses": cleared_streak,
                "mini_cooldowns": cleared_mini,
                "consec_losses": cleared_consec,
                "recent_losses_count": cleared_recent,
            },
            "preserved": {
                "cascade_active": bot._cascade_active,
                "daily_dd_breached": bot.daily_dd_breached,
                "recent_trade_r_len": len(bot.recent_trade_r),
            },
        })

    @app.get("/api/momentum")
    def api_momentum() -> Response:
        """Per-symbol reversal scores for the Chrome-extension overlay.

        Returns the most recent overlay snapshot the bot wrote on its tick
        loop. Unlike /api/state this endpoint does no Bitunix REST calls —
        all data is served from the in-process state cache, so polling at
        a few seconds per request is cheap.

        long_score:  0.0–1.0 — "how strongly indicators favor going LONG".
                     For someone holding a SHORT position, high values are
                     a reversal warning ("exit your short").
        short_score: mirror — "exit your long" warning.

        Note: these are confluence scores, not calibrated probabilities.
        """
        snap = state.overlay_snapshot()
        return jsonify({
            "now": int(time.time()),
            "tick_seconds": cfg.loop.tick_seconds,
            "timeframe": cfg.trading.timeframe,
            "fire_threshold": cfg.strategy.fire_threshold,
            "symbols": snap,
        })

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
const ageFmt = ms => {
  if (!ms || ms < 0) return '—';
  const s = Math.floor(ms / 1000);
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h + 'h' + (m ? m + 'm' : '');
};
const rFmt = r => {
  if (r === null || r === undefined || isNaN(r)) return '—';
  const cls = r >= 1 ? 'pos' : r > 0 ? '' : r < 0 ? 'neg' : '';
  const sign = r > 0 ? '+' : '';
  return `<span class="${cls}">${sign}${r.toFixed(2)}R</span>`;
};

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
  t.querySelector('thead').innerHTML =
    `<tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Mark</th>` +
    `<th>R</th><th>Age</th><th>SL</th><th>TP</th><th>uPnL</th><th>Liq</th></tr>`;
  if (s.open_positions_error) {
    t.querySelector('tbody').innerHTML =
      `<tr><td colspan="11" class="ev-error">API error: ${s.open_positions_error}</td></tr>`;
    return;
  }
  if (!rows.length) {
    t.querySelector('tbody').innerHTML =
      `<tr><td colspan="11" class="small">No open positions.</td></tr>`;
    return;
  }
  const now = Date.now();
  t.querySelector('tbody').innerHTML = rows.map(p => {
    const sideUp = (p.side === 'BUY' || p.side === 'LONG');
    const sideClass = sideUp ? 'pos' : 'neg';
    // Mark price: derived from entry + uPnL/qty (no extra API call needed).
    const entry = Number(p.avgOpenPrice || 0);
    const qty = Number(p.qty || 0);
    const upnl = Number(p.unrealizedPNL || 0);
    const priceDelta = qty > 0 ? (upnl / qty) : 0;
    const mark = sideUp ? entry + priceDelta : entry - priceDelta;
    // R-multiple measured against the ORIGINAL SL distance (entry × cfg pct),
    // not the current ratcheted SL distance. After a break-even SL ratchet,
    // the live SL is much closer to entry; using it would inflate R wildly
    // (e.g. +5R when the trade is actually at +1R favorable).
    const slPctCfg = Number((s.config && s.config.stop_loss_pct) || 0);
    const origSlDist = (entry > 0 && slPctCfg > 0) ? entry * slPctCfg / 100 : 0;
    const favor = sideUp ? (mark - entry) : (entry - mark);
    const r = origSlDist > 0 ? (favor / origSlDist) : null;
    // Age since opened.
    const age = p.ctime ? (now - Number(p.ctime)) : null;
    const sl = p.slPrice ? fmt(p.slPrice, 4) : '<span class="small">—</span>';
    const tp = p.tpPrice ? fmt(p.tpPrice, 4) : '<span class="small">—</span>';
    return `
      <tr>
        <td>${p.symbol||''}</td>
        <td class="${sideClass}">${p.side||''}</td>
        <td>${fmt(p.qty,4)}</td>
        <td>${fmt(entry,4)}</td>
        <td>${fmt(mark,4)}</td>
        <td>${rFmt(r)}</td>
        <td class="small">${ageFmt(age)}</td>
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
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
