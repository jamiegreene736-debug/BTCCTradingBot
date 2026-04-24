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
* Evaluates 5 technical rules (EMA stack, EMA-fast cross, RSI window, MACD
  momentum, Bollinger basis). Requires a configurable N-of-5 confluence to fire.
* Opens positions on `/api/v1/futures/trade/place_order` with **native**
  `tpPrice` / `slPrice` attached — Bitunix enforces both server-side, so your
  SL still fires even if the bot crashes.
* Stop loss is set as a tight % of entry price (default 0.25%). Take profit
  is a multiple of the SL distance (default 5R = 1.25% price move). Both are
  configurable, and an ATR-based alternative is available.
* One position at a time per symbol. Waits for a TP/SL exit before evaluating
  again.

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

| Group      | Key                  | Default    | Purpose |
|------------|----------------------|------------|---------|
| `trading`  | `symbol`             | BTCUSDT    | Bitunix perpetual symbol |
| `trading`  | `timeframe`          | 5m         | 1m / 5m / 15m / 30m / 1h / 2h / 4h |
| `trading`  | `leverage`           | 100        | Bitunix advertises up to 125x on BTCUSDT |
| `trading`  | `margin_coin`        | USDT       | |
| `trading`  | `margin_mode`        | ISOLATION  | ISOLATION / CROSS |
| `trading`  | `risk_per_trade_pct` | 1.0        | % of free margin risked if SL hits |
| `risk`     | `stop_loss_pct`      | 0.25       | Tight SL as % of entry price |
| `risk`     | `take_profit_r`      | 5.0        | TP distance = R × SL distance |
| `risk`     | `use_atr`            | false      | Flip to true for ATR-based SL/TP |
| `strategy` | `min_confluence`     | 4          | Need 4 of 5 rules to agree |

On first **live** run the bot calls Bitunix's `change_position_mode`,
`change_margin_mode`, and `change_leverage` endpoints so your account matches
the config. These are best-effort — failures (e.g. "leverage already set") are
logged and ignored.

## Architecture

```
run.py
 └── bitunix_bot/
      ├── config.py      # YAML + .env loader
      ├── client.py      # REST client (place_order, account, klines, signing)
      ├── indicators.py  # EMA / RSI / MACD / Bollinger / ATR (pure numpy)
      ├── strategy.py    # 5-rule confluence signal
      ├── risk.py        # Conservative SL, aggressive TP, leverage-aware sizing
      └── bot.py         # Main loop + signal handlers
```

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
