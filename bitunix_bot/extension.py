"""Single source of truth for the unpacked Chrome overlay version."""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "chrome-extensions"
    / "bitunix-momentum-overlay"
    / "manifest.json"
)
MAC_OVERLAY = (
    "/Users/jamiegreene/BTCCTradingBot/chrome-extensions/bitunix-momentum-overlay"
)


def required_extension_version() -> str:
    return str(json.loads(MANIFEST.read_text())["version"])
