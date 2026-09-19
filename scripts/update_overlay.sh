#!/usr/bin/env bash
# Update the unpacked Chrome overlay on this Mac. Railway cannot do this.
set -euo pipefail

ROOT="${BTCC_REPO:-$HOME/BTCCTradingBot}"
OVERLAY="$ROOT/chrome-extensions/bitunix-momentum-overlay"
TARGET="${1:-}"

if [[ ! -d "$ROOT/.git" || ! -f "$OVERLAY/manifest.json" ]]; then
  echo "No BTCCTradingBot clone at $ROOT" >&2
  echo "This is not the rental-tracker repo. Set BTCC_REPO if Chrome loads another copy." >&2
  exit 1
fi

cd "$ROOT"

if ! git remote get-url origin | grep -Eq 'BTCCTradingBot'; then
  echo "origin is not BTCCTradingBot: $(git remote get-url origin)" >&2
  echo "You are in the wrong folder." >&2
  exit 1
fi

git fetch origin

if [[ -z "$TARGET" ]]; then
  if git show-ref --verify --quiet refs/remotes/origin/cursor/persist-signals-and-shorts-a6df; then
    TARGET="cursor/persist-signals-and-shorts-a6df"
  else
    TARGET="$(git symbolic-ref -q --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || true)"
    TARGET="${TARGET:-main}"
  fi
fi

git checkout "$TARGET"
git pull --ff-only origin "$TARGET"

if [[ -d "$ROOT/scripts/githooks" ]]; then
  git config core.hooksPath scripts/githooks
fi

VERSION="$(python3 -c "import json; print(json.load(open('$OVERLAY/manifest.json'))['version'])")"
echo "Updated overlay folder:"
echo "  $OVERLAY"
echo "Version: $VERSION"
echo "Now: chrome://extensions → Reload Bitunix Intraday Signals → reload the Bitunix tab."
