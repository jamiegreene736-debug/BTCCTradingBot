#!/usr/bin/env python3
"""Entry point. Runs the Flask dashboard on $PORT and the trading worker
in a background thread inside the same process."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

from bitunix_bot.bot import BitunixBot, configure_logging
from bitunix_bot.config import load
from bitunix_bot.dashboard import create_app

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bitunix high-leverage technical trading bot")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-e", "--env", default=".env")
    args = ap.parse_args()

    cfg = load(args.config, args.env)
    configure_logging(cfg)

    bot = BitunixBot(cfg)
    bot._resolve_symbol_meta()
    if cfg.is_live:
        bot._configure_account()

    # Trading loop runs as a daemon thread so the process exits cleanly when
    # Flask shuts down. We don't install signal handlers in this thread because
    # only the main thread can do that.
    worker = threading.Thread(target=bot.run_forever, name="bitunix-worker", daemon=True)
    worker.start()

    # Forward SIGINT/SIGTERM to the bot's stop flag so we drain cleanly.
    def _shutdown(*_):
        log.info("Shutdown signal received")
        bot.stop_flag = True
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    app = create_app(cfg, bot.client, bot=bot)
    port = int(os.environ.get("PORT", "8080"))
    log.info("Dashboard listening on 0.0.0.0:%d", port)
    # Use Werkzeug's built-in server. For Railway's traffic levels this is fine;
    # we're a single-user dashboard, not a public web app.
    app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
