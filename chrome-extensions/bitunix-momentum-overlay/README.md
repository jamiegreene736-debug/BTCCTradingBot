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

- `PUMP BUILDING`: early buy pressure before the short setup is ready.
- `PUMP WATCH`: pump is active, waiting for near-high rejection.
- `FADE SHORT`: the backend has a confirmed pump-fade short setup.
- Per-symbol closed trade history and net P&L.
- Open-position display. Timed auto-close is currently disabled.
- `AUTO` mode ranks all scanned coins, keeps the strongest pump-fade candidate
  selected, and highlights the chosen coin in bright yellow. It pauses for
  60 seconds when you manually click a coin.

The extension does not compute signals locally. The backend is the source of
truth; this folder only renders the overlay and relays admin actions.
