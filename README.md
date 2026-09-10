# Bitunix Intraday Signals

An alerts-only scanner and Chrome overlay for long and short trades with a
maximum holding period of 12 or 24 hours. The new strategy uses completed
4h / 1h / 15m candles. It never places orders, changes leverage, modifies stops,
or closes exchange positions.

## Start the backend

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export DASHBOARD_PASSWORD='choose-a-private-dashboard-password'
.venv/bin/python run.py
```

`signals.enabled: true` is the shipped default. It overrides legacy `mode: live`
and pump-fade execution flags. A read-only exchange-client guard also rejects
every exchange POST. The old automatic entry/exit loop is bypassed, and old
extension admin POSTs receive HTTP 403. Public market scanning needs no API key.
Legacy account/dashboard reads may still need read credentials.

Load `chrome-extensions/bitunix-momentum-overlay` unpacked through
`chrome://extensions`. Connect it to your HTTPS Railway dashboard and password.
After updating, reload the extension and then the Bitunix tab. The new overlay
requires the matching backend release; it does not interpret old pump-fade scores.

## Signals

1. 4h supplies directional bias only (20/50 EMA stack and slope). Confirmed 4h
   swings take 16 hours to print and arrive too late for a 12/24h hold. 1h must
   still show matching confirmed swing structure, EMA stack, and slope. Mixed
   1h structure or an opposite 4h bias means WAIT.
2. A 15m trend pullback must touch the hourly or 15m EMA, hourly support, or
   UTC-session VWAP, then reclaim the prior close in the trend direction with
   relative volume. Entries do not wait for a break of the prior high.
3. Alternatively, a volume-backed breakout must precede a separate retest that
   holds the old range boundary, or a 15m impulse of at least 1.1 ATR must
   pull back without breaking its origin and then reclaim. Long and short
   rules are symmetric.
4. The stop sits beyond the setup's structural low/high plus an ATR allowance.
   It must accommodate at least 0.75 ATR. The nearest confirmed structural
   target that still provides 2R after estimated costs, and that sits inside a
   ≤24h travel budget (8× 1h ATR or 3× 4h ATR), is used. Near swings that fail
   2R are skipped instead of blocking the trade. A second target is contextual
   only; the default exit is the first target.
5. Altcoins additionally require aligned BTC direction and matching relative
   strength over six hours. Spread, 24h USDT volume, 1h ATR (not dead, not
   blow-off), planned order size, depth, projected funding drag, and an
   estimated isolated-margin liquidation buffer must pass. The intended
   isolated-margin band is 25-40x; leverage never narrows the stop.

The scanner ranks the twelve most liquid eligible USDT perpetuals, plus BTC and
any actively tracked symbols. All gates are visible in the signal checklist.
There are no fabricated confidence percentages or calibrated win probabilities.
Funding and open interest are context, not independent buy/sell triggers; missing
open interest is labeled unavailable. Missing funding, depth, or maintenance
tiers blocks a new entry. This release does not include a news/event feed.

## Planning and tracking

The overlay's Edit button sets planning equity, risk per trade, leverage (1–40x),
and maximum hold (12h/24h). Initial planning defaults are explicitly hypothetical:
1,000 USDT equity, 0.5% risk, 25x, 24 hours. These are not an exchange balance.

Position size accounts for the structural stop plus estimated costs. The scanner
counts funding settlements over the maximum hold using the provider's actual
interval and next-settlement timestamp, then projects the current funding rate.
Expected funding receipts do not subsidize the trade. Fees and slippage remain
configurable estimates in `config.yaml`, and future funding rates can change.

Leverage never narrows a stop. The scanner estimates a leverage ceiling using the
position's maintenance tier, mark/last basis, costs, and a buffer of at least
0.5% of entry or half an ATR. This is an estimate for isolated margin without
extra collateral, not the exchange's exact liquidation price or a guarantee.

- **Track paper trade** records a simulated entry without placing an order.
- **Record my fill** records a fill you already executed, within the confirmed
  entry zone and planned size/risk. It does not detect/import exchange positions.
- Original stop, target, risk and holding limit are frozen when recorded.
- Exit alerts cover stop/target touches, opposite completed 1h structure, failure
  to make 0.25R progress within four hours, and maximum age. After 1R the stop
  can advance to cover estimated costs; a trailing stop can advance after 1.5R
  and never widen. Stop changes are suggestions to apply manually.
- Exit alerts remain latched until you record closure. They do not reverse into
  an opposite entry or assume that an exchange order filled.
- **Record closure** ends tracking only. Estimated net results use the recorded
  exit price and planned costs; they are not exchange-settled P&L.

Tracked exposure is capped at 1.5% of planning equity and two trades in the same
direction. This includes both paper and user-recorded trades and cannot account
for unrecorded exchange positions. Put actual protective stops on Bitunix;
browser alerts are not a replacement for exchange-side protection.

## Data integrity and persistence

Market reads are paced below eight requests per second, cached by timeframe,
invalidated at candle boundaries, and backed off after provider failures. A
provider circuit breaker prevents retry storms. Invalid, missing, incomplete,
or stale data cannot produce a confirmed entry. Tracking shows REVIEW on an
outage; the maximum-hold exit still applies. Candle wicks before entry never
trigger a tracked exit, and trailing stops are not applied retroactively.

Bitunix candle payloads use `quoteVol` for coin volume and `baseVol` for turnover;
tickers use `quoteVol` for USDT turnover. Some opens carry the prior close outside
the traded high/low; normalization includes this opening gap in the range.

Planning settings, trades and deduplicated alert history are stored in SQLite at
`logs/intraday-signals.sqlite3`. Set `SIGNAL_STATE_PATH` to a path on a persistent
Railway volume to retain them across deployments. Without a persistent volume,
container replacement can lose local state. Existing tracks survive ordinary
process restarts. Passwords are stored locally in Chrome, migrated out of sync
storage, and are never sent to the page or an arbitrary dashboard origin.

## Validation

```sh
python3 -m pytest tests/test_e2e.py tests/test_intraday.py -q
node --test tests/extension_worker.test.cjs
# Requires Playwright; CHROME_PATH can override the local Chrome executable.
node tests/extension_ui.cjs
python3 scripts/backtest_intraday.py --days 7 --symbols BTCUSDT,ETHUSDT \
  --leverage 25 --output /tmp/intraday-replay.json
```

The replay is chronological and uses only candles completed at each decision.
It declares its spread, funding, margin and depth assumptions. Missing-data
windows are counted and excluded; an open trade crossing a gap is unresolved,
not assigned an invented result. Ambiguous candles test stops before targets.
Historical book depth, changing funding/tier rules, mark basis, liquidation
paths and actual fills are not available from candles. This is a structural
check, not evidence of profitability. Forward paper observation remains necessary.

## Components and API

- `intraday.py`: pure signal/plan calculations.
- `signal_scanner.py`: public data, freshness, ranking and planning actions.
- `signal_store.py`: persistent tracking and exit state.
- `signal_routes.py`: authenticated `/api/signals` and tracking endpoints.
- `chrome-extensions/bitunix-momentum-overlay`: display and manual records.

`GET /api/signals` returns the latest snapshot without making blocking market
requests. `POST /api/signals/settings`, `/track`, and `/close` update local
planning/tracking state only. `/api/momentum` returns the new snapshot in signal
mode for compatibility detection. All require existing dashboard Basic auth.
The old strategy modules remain for explicit legacy regression/replay use.

Provider references: [candles](https://www.bitunix.com/api-docs/futures/market/get_kline.html),
[funding](https://www.bitunix.com/api-docs/futures/market/get_funding_rate.html),
[depth](https://www.bitunix.com/api-docs/futures/market/get_depth.html),
[position tiers](https://www.bitunix.com/api-docs/futures/position/get_position_tiers.html).
