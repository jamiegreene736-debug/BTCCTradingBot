# Bitunix Pump Fade Overlay

Chrome extension for the Bitunix pump-fade scanner. It renders a floating
overlay on `bitunix.com` and reads `/api/momentum` from the private Railway
dashboard.

## Load Locally

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click `Load unpacked`.
4. Select this folder:
   `chrome-extensions/bitunix-momentum-overlay`.
5. Click the extension icon and open Settings.
6. Add the Railway dashboard URL and dashboard password.

## What It Shows

- `PUMP BUILDING`: earliest pump-ignition alert. The overlay auto-selects it,
  flashes yellow, and flashes the browser tab title before the selloff confirms.
- `PUMP WATCH`: pump is active now; the overlay auto-selects the coin, flashes
  bright yellow, and flashes the browser tab title so you can get ready before
  the fade entry confirms.
- `FADE SHORT`: the backend has a confirmed pump-fade short setup.
- `Fade ETA`: rough timing window (`NOW`, `10-30s`, `30-90s`, etc.) estimated
  from pump speed, range position, 1m rejection, CVD/tape, and whether buyers
  are still in control. This is a timing aid, not a guaranteed countdown.
- `Suggested short entry trigger`: preview price to watch during `PUMP BUILDING`
  and `PUMP WATCH`. It includes preview take-profit and stop-loss prices, but it
  is not an entry until the rejection confirms.
- `Suggested short entry`: confirmed short entry price. The card also shows
  `Take profit / suggested exit`, `Stop loss`, target profit, and max loss.
- Per-symbol closed trade history and net P&L.
- Open-position display. Timed auto-close is currently disabled.
- `AUTO` mode ranks all scanned coins, keeps the strongest pump-fade candidate
  selected, and highlights the chosen coin in bright yellow. Urgent `PUMP
  BUILDING`, `PUMP WATCH`, and `FADE SHORT` alerts can override a manual click
  hold so the best coin does not stay buried.
- The extension polls every 5s normally and speeds up to about 2s while a
  building, watch, or short-ready candidate exists. The backend 1m overlay
  cache refreshes every bot tick so active pumps do not sit on stale data.

The extension does not compute signals locally. The backend is the source of
truth; this folder only renders the overlay and relays admin actions.
