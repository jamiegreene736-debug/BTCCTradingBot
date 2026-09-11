#!/usr/bin/env bash
# Idempotent Cloud Agent setup for the Bitunix Intraday Signals backend.
# Installs the venv toolchain, Python dependencies, and the pytest runner.
set -euo pipefail

cd "$(dirname "$0")/.."

# The default image ships Python 3.12 but not the venv/pip modules.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12-venv python3-pip
fi

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
# pytest is the test runner used by tests/ but is not a runtime dependency.
.venv/bin/pip install pytest
