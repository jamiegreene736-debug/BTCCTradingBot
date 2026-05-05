# Bitunix TraderBot

High-leverage, 100% technical futures bot for [Bitunix](https://www.bitunix.com).
Conservative stop loss, aggressive take profit. No news, no sentiment, no
fundamentals — only price action and indicators.

> The folder is named `BTCC TraderBot` for historical reasons (the bot was
> originally written against BTCC's API, then ported to Bitunix after the BTCC
> self-service API flow turned out to be read-only). The Python package is
> `bitunix_bot`.

## What it does

* Fetches klines from `https://fapi.bitunix.com/api/v1/futures/market/kline`
  at a configurable timeframe.
* Dynamically scans Bitunix USDT perpetuals from `trading_pairs`/`tickers` and
  keeps the highest-liquidity symbols that pass volume and leverage floors.
* Trades only the dedicated parabolic pump-fade SHORT setup when auto execution
  is enabled: 5m pump, near-high location, 1m rejection, bearish tape/flow, and
  trend-risk checks.
* Opens positions on `/api/v1/futures/trade/place_order` with **native**
  `tpPrice` / `slPrice` attached — Bitunix enforces both server-side, so your
  SL still fires even if the bot crashes.
* Stop loss is a tight % of entry price (default 0.25%). Take profit can be
  capped by `margin_profit_target_pct`; the current pump-fade profile targets
  roughly 15% gross margin profit before fees.
* **Multi-symbol, multi-position**: trades a list of symbols simultaneously
  with a global position cap, per-symbol cap, and per-symbol cooldown.
* **Position exit**: timed auto-close is disabled by default. Positions stay
  open until TP/SL, manual close, or a future explicit exit rule handles them.
* **Bar-dedupe**: within the same candle, a symbol is only evaluated once —
  no double-firing on the same bar.
* **Web dashboard** at `/` with live balance, open positions, closed-position
  history, order history, and a stream of recent bot decisions (signals,
  skips, orders, errors). Auto-refreshes every 10s. Protected by HTTP Basic
  auth — set `DASHBOARD_PASSWORD`.

## Why this architecture

Bitunix isn't supported by ccxt at the time of writing but has a well-documented
OpenAPI. The REST layer is hand-built, the signing is verified against the
[official spec](https://www.bitunix.com/api-docs/futures/common/sign.html) and
the unofficial SDK at [0xCherryBlueZu/bitunix](https://github.com/0xCherryBlueZu/bitunix).
Everything else — indicators, strategy, risk — is exchange-agnostic and would
port cleanly to any other USDT-perp venue.

## Setup

```bash
cd "BTCC TraderBot"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in your Bitunix API key + secret
```

Generate a key pair under **Profile → API Management → Futures OpenAPI** on
Bitunix. Enable **Read** and **Trade** permissions. Copy both values to `.env`
immediately — the secret is shown only once.

## Running

```bash
# PAPER mode (default): signals are logged but no orders are sent
python run.py

# LIVE: edit config.yaml and set `mode: live`
```

Logs stream to stdout and `logs/bot.log`.

## Config knobs (`config.yaml`)

| Group      | Key                       | Default                     | Purpose |
|------------|---------------------------|-----------------------------|---------|
| `trading`  | `symbols`                 | `[BTCUSDT,ETHUSDT,DOGEUSDT,XRPUSDT]` | Pinned symbols; dynamic scan can append more |
| `trading`  | `dynamic_symbols_enabled` | `true`                      | Scan liquid USDT perpetuals every refresh window |
| `trading`  | `dynamic_symbol_min_quote_volume_usdt` | `10000000`      | 24h quote-volume floor for dynamic symbols |
| `trading`  | `timeframe`               | `5m`                        | Legacy strategy timeframe; overlay uses 1m entry + 5m pump |
| `trading`  | `leverage`                | `100`                       | Generic cap; pump-fade auto leverage is separate |
| `trading`  | `margin_coin`             | `USDT`                      | |
| `trading`  | `margin_mode`             | `ISOLATION`                 | ISOLATION / CROSS |
| `trading`  | `risk_per_trade_pct`      | `1.0`                       | % of free margin risked if SL hits |
| `trading`  | `max_open_positions`      | `2`                         | Global cap across all symbols |
| `trading`  | `max_positions_per_symbol`| `1`                         | Never pyramid into the same trade |
| `trading`  | `cooldown_seconds`        | `60`                        | Min seconds between trades on same symbol |
| `trading`  | `max_position_age_seconds`| `0`                         | Timed auto-close disabled; set >0 to enable |
| `trading`  | `pump_fade_auto_min_confidence` | `96`                  | Ultra-confidence gate before auto market short |
| `trading`  | `pump_fade_auto_leverage` | `200`                       | BTC/ETH may use 200x; alts are capped lower |
| `risk`     | `stop_loss_pct`           | `0.25`                      | Tight SL as % of entry price |
| `risk`     | `take_profit_r`           | `1.0`                       | R fallback before margin-profit cap |
| `risk`     | `margin_profit_target_pct`| `15.0`                      | Cap TP to roughly this gross margin % |
| `risk`     | `use_atr`                 | `true`                      | Widen SL in volatility expansion |
| `strategy` | `min_confluence`          | `4`                         | Need 4 of 7 rules to agree |
| `strategy` | `adx_min`                 | `22.0`                      | Trend-strength filter floor |
| `strategy` | `supertrend_period`       | `10`                        | ATR period for supertrend |
| `strategy` | `supertrend_mult`         | `3.0`                       | ATR multiplier for supertrend |

## Realistic fee math at high leverage

Read this before going live. Per round-trip on Bitunix:

```
0.05% taker fee × 2  (entry + exit) = 0.10%
~0.05% slippage × 2                  = 0.10%
funding (15-min hold)                ≈ 0.01%
TOTAL per-trade cost                 ≈ 0.20% of notional
```

That **0.20% of notional** translates to:

| Leverage | Fee drag (% of margin) | SL hit (loss + fees) | TP win (gain - fees) | Breakeven win rate |
|----------|------------------------|----------------------|----------------------|--------------------|
| **100x** | 20% per round-trip     | -45%                 | +105%                | ~30% (3 losses = liq) |
| **50x**  | 10%                    | -22.5%               | +52.5%               | ~30% (more headroom)  |
| **25x**  | 5%                     | -11.25%              | +26.25%              | ~30% (much safer)     |

The breakeven win rate is the same (~30%), but the **drawdown tolerance is wildly
different**. At 100x with isolated margin, three consecutive SL hits liquidates
the position. At 25x, you can take six losses before liquidation. Real backtests
on the proven scalping strategies (freqtrade `Strategy002`, `SmoothScalp`,
hummingbot supertrend controllers) typically show 50–60% win rates on 1m, but
*streaks* of 4–6 losses do happen. **The actual sweet spot in production repos
is 25–50x, not 100x**. The bot defaults to 100x because you asked for it, but
consider dropping to 50x until you see real results.

On first **live** run the bot calls Bitunix's `change_position_mode`,
`change_margin_mode`, and `change_leverage` endpoints so your account matches
the config. These are best-effort — failures (e.g. "leverage already set") are
logged and ignored.

## Architecture

```
run.py                    # Spawns the trading worker thread + Flask app on $PORT
 └── bitunix_bot/
      ├── config.py       # YAML + .env loader
      ├── client.py       # REST client (place_order, account, klines, history, signing)
      ├── indicators.py   # EMA / RSI / MACD / Bollinger / ATR (pure numpy)
      ├── strategy.py     # 5-rule confluence signal
      ├── risk.py         # SL/TP, 15% margin target, leverage-aware sizing
      ├── symbol_meta.py  # Dynamic symbol filters + risk multipliers
      ├── state.py        # Thread-safe shared state for the dashboard
      ├── dashboard.py    # Flask app + HTML — basic auth on every route
      └── bot.py          # Main trading loop
```

## Dashboard

The bot serves a dashboard on Railway's auto-assigned `*.up.railway.app` URL.
Routes:

| Path                 | What it returns                                         |
|----------------------|----------------------------------------------------------|
| `/`                  | HTML dashboard (auth required)                          |
| `/api/state`         | JSON snapshot — account, positions, history, events     |
| `/api/momentum`      | Overlay status, pump-fade-only short decision, open positions, and closed-trade P&L history |
| `/api/feeds/status`  | WebSocket feed health (OB + tape connection state)      |
| `/api/journal`       | Trade journal events (entries + exits, JSONL-backed)    |
| `/healthz`           | Plain `ok`, no auth — for uptime checks                 |

Set `DASHBOARD_PASSWORD` in Railway → Variables. The username is `admin`.
If the env var is unset, every route except `/healthz` returns 503.

### Asking AI reviewers for feedback

`scripts/ask_reviewers.py` pulls live state from the dashboard, builds a
Markdown export, and POSTs it to Grok (xAI) and/or ChatGPT (OpenAI) for
review. The reviews are saved to `logs/exports/reviews/<provider>-<ts>.md`
and printed to stdout. **The script never edits config or code** — you
read the recommendations and choose what to apply.

```bash
# Set keys in your shell or .env
export BOT_DASHBOARD_PASSWORD='<dashboard password>'
export XAI_API_KEY='<key from console.x.ai>'
export OPENAI_API_KEY='<key from platform.openai.com>'

scripts/ask_reviewers.py                 # ask both
scripts/ask_reviewers.py --grok          # ask Grok only
scripts/ask_reviewers.py --openai        # ask OpenAI only
scripts/ask_reviewers.py --question "..." # custom prompt
scripts/ask_reviewers.py --dry-run       # build the export, skip API calls
```

### Backtesting the pump-fade filter

The lightweight replay script uses recent Bitunix 1m klines, resamples them
into 5m/15m context, runs the same pump-fade decision tree, and reports rough
margin-PnL hit rates. It does not include live order-book or trade-tape data,
so use it as a structural sanity check rather than a full execution simulator.

```bash
scripts/backtest_pump_fade.py --symbols BTCUSDT,ETHUSDT --limit 500
```

Set `TRADE_WEBHOOK_URL` to send each journaled entry/exit JSON payload to a
Discord/Slack/custom webhook.

## Safety notes

* **Always run in `paper` mode first.** The bot prints the exact order it
  would have sent. No credentials scope is needed for paper mode beyond Read.
* Extremely high leverage + tight stop loss means full-size SL hits happen
  often. Track your win rate over ~30 round-trips before deciding it's working.
* The bot never places an order without `slPrice` attached. If you ever see
  one go out without it, that's a bug — stop trading and file it.
* Start with `risk_per_trade_pct: 0.5` and `leverage: 25` until you've watched
  several full round trips.
* Bitunix's 60-second signature-window means the host clock must be accurate.
  If your machine drifts, `timestamp` rejections start happening — run `ntpd`.

## Reference

* [Bitunix OpenAPI index](https://www.bitunix.com/api-docs/)
* [Signing spec](https://www.bitunix.com/api-docs/futures/common/sign.html)
* [Place order spec](https://www.bitunix.com/api-docs/futures/trade/place_order.html)
* Legacy BTCC spec PDFs still in the repo as `btcc_tradeapi.pdf` / `btcc_quote_ws.pdf`
  — kept for reference if you ever want to port back.
