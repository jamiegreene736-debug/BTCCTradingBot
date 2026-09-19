#!/usr/bin/env bash
# Deploy the trading bot and refresh the Mac unpacked overlay in the same step.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export BTCC_REPO="${BTCC_REPO:-$HOME/BTCCTradingBot}"

"$HERE/update_overlay.sh" "$@"

cd "$BTCC_REPO"
if command -v railway >/dev/null 2>&1 && { [[ -f railway.toml ]] || [[ -d .railway ]] || [[ -f railway.json ]]; }; then
  railway up
else
  echo "Overlay folder updated. Push/merge to deploy the backend; Railway cannot copy files onto this Mac."
fi

echo "Reload Chrome from: $BTCC_REPO/chrome-extensions/bitunix-momentum-overlay"
