#!/usr/bin/env bash
# Keep the unpacked Chrome overlay on this Mac current. Installs a LaunchAgent
# so later deploys pull themselves. Do not run this in Rental-Community-Tracker.
set -euo pipefail

ROOT="${BTCC_REPO:-$HOME/BTCCTradingBot}"
OVERLAY="$ROOT/chrome-extensions/bitunix-momentum-overlay"
LABEL="com.bitunix.intraday-overlay-sync"
INSTALL_ONLY=0
QUIET=0
TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-only) INSTALL_ONLY=1; shift ;;
    --quiet) QUIET=1; shift ;;
    *) TARGET="$1"; shift ;;
  esac
done

say() { [[ "$QUIET" -eq 1 ]] || echo "$@"; }

if [[ ! -d "$ROOT/.git" || ! -f "$OVERLAY/manifest.json" ]]; then
  echo "No BTCCTradingBot clone at $ROOT" >&2
  exit 1
fi

cd "$ROOT"

if ! git remote get-url origin | grep -Eq 'BTCCTradingBot'; then
  echo "origin is not BTCCTradingBot: $(git remote get-url origin)" >&2
  exit 1
fi

install_agent() {
  [[ "$(uname -s)" == "Darwin" ]] || return 0
  local plist_dir="$HOME/Library/LaunchAgents"
  local plist="$plist_dir/$LABEL.plist"
  mkdir -p "$plist_dir" "$HOME/Library/Logs"
  cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$ROOT/scripts/update_overlay.sh</string>
    <string>--quiet</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$ROOT</string>
  <key>StartInterval</key>
  <integer>120</integer>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$HOME/Library/Logs/bitunix-overlay-sync.log</string>
  <key>StandardErrorPath</key>
  <string>$HOME/Library/Logs/bitunix-overlay-sync.log</string>
</dict>
</plist>
EOF
  local uid
  uid="$(id -u)"
  launchctl bootout "gui/$uid/$LABEL" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$uid" "$plist" >/dev/null 2>&1 || launchctl load -w "$plist" >/dev/null 2>&1 || true
  launchctl enable "gui/$uid/$LABEL" >/dev/null 2>&1 || true
  launchctl kickstart -k "gui/$uid/$LABEL" >/dev/null 2>&1 || true
}

install_agent
if [[ -d "$ROOT/scripts/githooks" ]]; then
  git config core.hooksPath scripts/githooks
fi
if [[ "$INSTALL_ONLY" -eq 1 ]]; then
  say "Overlay auto-sync is installed. Chrome reloads itself after a pull."
  exit 0
fi

git fetch origin --quiet

if [[ -z "$TARGET" ]]; then
  if git show-ref --verify --quiet refs/remotes/origin/cursor/persist-signals-and-shorts-a6df; then
    TARGET="cursor/persist-signals-and-shorts-a6df"
  else
    TARGET="$(git symbolic-ref -q --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || true)"
    TARGET="${TARGET:-main}"
  fi
fi

if [[ -n "$(git status --porcelain)" ]]; then
  say "BTCCTradingBot has local changes; auto-sync skipped a checkout."
  exit 0
fi

git checkout --quiet "$TARGET"
git pull --ff-only --quiet origin "$TARGET"

VERSION="$(python3 -c "import json; print(json.load(open('$OVERLAY/manifest.json'))['version'])")"
say "Overlay $OVERLAY is v$VERSION. Chrome reloads itself when the backend version changes."
