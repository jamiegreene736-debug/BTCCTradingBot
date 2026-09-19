"""Install the Mac LaunchAgent that pulls the unpacked overlay by itself."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)
MAC_ROOT = Path.home() / "BTCCTradingBot"


def install_mac_overlay_sync() -> None:
    script = MAC_ROOT / "scripts" / "update_overlay.sh"
    if os.uname().sysname != "Darwin" or not script.is_file():
        return
    try:
        subprocess.Popen(
            ["/bin/bash", str(script), "--install-only"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        log.debug("Overlay auto-sync not installed: %s", exc)
