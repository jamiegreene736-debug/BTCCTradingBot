#!/usr/bin/env python3
"""Entry point."""
from __future__ import annotations

import argparse
import sys

from bitunix_bot.bot import BitunixBot, configure_logging
from bitunix_bot.config import load


def main() -> int:
    ap = argparse.ArgumentParser(description="Bitunix high-leverage technical trading bot")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-e", "--env", default=".env")
    args = ap.parse_args()

    cfg = load(args.config, args.env)
    configure_logging(cfg)

    BitunixBot(cfg).start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
