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
* Evaluates 7 technical rules: EMA stack, EMA-fast cross, RSI window, MACD
  momentum, Bollinger basis, **ADX trend strength** (whipsaw filter), and
  **Supertrend regime**. Requires a configurable N-of-7 confluence to fire.
* Opens positions on `/api/v1/futures/trade/place_order` with **native**
  `tpPrice` / `slPrice` attached — Bitunix enforces both server-side, so your
  SL still fires even if the bot crashes.
* Stop loss is a tight % of entry price (default 0.25%). Take profit is a
  multiple of the SL distance (default 5R = 1.25% price move). ATR-based
  alternative available.
* **Multi-symbol, multi-position**: trades a list of symbols simultaneously
  with a global position cap, per-symbol cap, and per-symbol cooldown.
* **Time-based exit**: force-closes positions older than
  `max_position_age_seconds` (default 15 min). Prevents stale trades from
  bleeding funding fees.
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
| `trading`  | `symbols`                 | `[BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT]` | Universe of perpetuals to trade |
| `trading`  | `timeframe`               | `1m`                        | 1m / 5m / 15m / 30m / 1h / 2h / 4h |
| `trading`  | `leverage`                | `100`                       | Bitunix max is 200x on BTCUSDT — see fee math below |
| `trading`  | `margin_coin`             | `USDT`                      | |
| `trading`  | `margin_mode`             | `ISOLATION`                 | ISOLATION / CROSS |
| `trading`  | `risk_per_trade_pct`      | `1.0`                       | % of free margin risked if SL hits |
| `trading`  | `max_open_positions`      | `4`                         | Global cap across all symbols |
| `trading`  | `max_positions_per_symbol`| `1`                         | Never pyramid into the same trade |
| `trading`  | `cooldown_seconds`        | `60`                        | Min seconds between trades on same symbol |
| `trading`  | `max_position_age_seconds`| `900`                       | Force-close after 15min (0 disables) |
| `risk`     | `stop_loss_pct`           | `0.25`                      | Tight SL as % of entry price |
| `risk`     | `take_profit_r`           | `5.0`                       | TP distance = R × SL distance |
| `risk`     | `use_atr`                 | `false`                     | Flip to true for ATR-based SL/TP |
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
      ├── risk.py         # Conservative SL, aggressive TP, leverage-aware sizing
      ├── state.py        # Thread-safe shared state for the dashboard
      ├── dashboard.py    # Flask app + HTML — basic auth on every route
      └── bot.py          # Main trading loop
```

## Dashboard

The bot serves a dashboard on Railway's auto-assigned `*.up.railway.app` URL.
Routes:

| Path          | What it returns                                         |
|---------------|----------------------------------------------------------|
| `/`           | HTML dashboard (auth required)                          |
| `/api/state`  | JSON snapshot — account, positions, history, events     |
| `/healthz`    | Plain `ok`, no auth — for uptime checks                 |

Set `DASHBOARD_PASSWORD` in Railway → Variables. The username is `admin`.
If the env var is unset, every route except `/healthz` returns 503.

## Safety notes

* **Always run in `paper` mode first.** The bot prints the exact order it
  would have sent. No credentials scope is needed for paper mode beyond Read.
* Extremely high leverage + tight stop loss means full-size SL hits happen
  often. The 5R target relies on asymmetric payoff, not hit rate. Track your
  win rate over ~30 round-trips before deciding it's working.
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
