#!/usr/bin/env python3
"""Ask Grok and/or ChatGPT to review the bot's current state and recommend changes.

The script fetches live state from the Railway dashboard (/api/state,
/api/feeds/status, /api/journal), builds the same Markdown export you'd
otherwise paste into a chat manually, then POSTs it to one or both
review providers via their OpenAI-compatible chat-completions endpoints.

Both responses are saved to logs/exports/reviews/<provider>-<ts>.md and
printed side-by-side. The user decides what to apply — this script
NEVER edits config or code on its own.

Usage:
  scripts/ask_reviewers.py                       ask both
  scripts/ask_reviewers.py --grok                ask only Grok
  scripts/ask_reviewers.py --openai              ask only OpenAI
  scripts/ask_reviewers.py --question "..."      override the default prompt
  scripts/ask_reviewers.py --export PATH         skip fetch, use existing markdown
  scripts/ask_reviewers.py --dry-run             build export, don't call APIs

Env vars (with defaults):
  XAI_API_KEY              xAI / Grok API key                 (required for --grok)
  OPENAI_API_KEY           OpenAI API key                     (required for --openai)
  XAI_MODEL                model name for xAI                 (grok-4-latest)
  OPENAI_MODEL             model name for OpenAI              (gpt-5)
  XAI_BASE_URL             override xAI endpoint              (https://api.x.ai/v1)
  OPENAI_BASE_URL          override OpenAI endpoint           (https://api.openai.com/v1)
  BOT_BASE_URL             Railway dashboard URL              (https://btcc-trading-bot-production.up.railway.app)
  BOT_DASHBOARD_USER       Basic auth username                (admin)
  BOT_DASHBOARD_PASSWORD   Basic auth password                (REQUIRED — no default)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from base64 import b64encode
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------- defaults

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = REPO_ROOT / "logs" / "exports"
REVIEWS_DIR = EXPORTS_DIR / "reviews"

DEFAULT_BASE_URL = "https://btcc-trading-bot-production.up.railway.app"
DEFAULT_USER = "admin"

DEFAULT_XAI_BASE = "https://api.x.ai/v1"
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_XAI_MODEL = "grok-4-latest"
DEFAULT_OPENAI_MODEL = "gpt-5"

DEFAULT_QUESTION = (
    "Review the export above. The bot is a Bitunix futures scalping bot on a "
    "$35 USDT account, 1m timeframe, 25× leverage, fees ~0.117% round-trip on "
    "market orders. Lifetime journal: 0/130 trades positive realized PnL even "
    "though 33% had positive net_pnl before fees. Latest changes: Grok v8 "
    "(ticker confirmation + market entries), then Grok v9 (relaxed stale exit "
    "6m/0.5R → 12m/0.2R, BE move 1R → 0.5R, trailing 1.5R → 1R).\n\n"
    "Please answer:\n"
    "1. Top 3 issues you see — most leverage first.\n"
    "2. Concrete config changes (parameter name → new value, with brief rationale).\n"
    "3. Any code/logic changes if config alone isn't enough.\n"
    "4. Anything from the recent commits you would REVERT, and why.\n"
    "5. Honest assessment: is this strategy salvageable on a $35 account, or "
    "is the fee floor too high to overcome regardless of tuning?\n\n"
    "Be specific and concrete. The user will decide what to apply."
)

SYSTEM_PROMPT = (
    "You are reviewing a live cryptocurrency trading bot's state export. "
    "Give concrete, actionable feedback — exact parameter names and values, "
    "specific code changes if needed. Avoid generic advice. Acknowledge "
    "uncertainty where the data is ambiguous. The user is running this on a "
    "small account ($35 USDT) so any changes that increase risk meaningfully "
    "should be flagged."
)


# ----------------------------------------------------------------- HTTP helpers

def _basic_auth_header(user: str, password: str) -> str:
    token = b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def _http_get_json(url: str, *, headers: dict[str, str] | None = None,
                   timeout: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_post_json(url: str, payload: dict[str, Any], *,
                    headers: dict[str, str] | None = None,
                    timeout: int = 180) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_bot_data(base_url: str, user: str,
                   password: str) -> tuple[dict, dict, dict]:
    """Pull /api/state, /api/feeds/status, /api/journal from the dashboard."""
    auth = {"Authorization": _basic_auth_header(user, password)}
    state = _http_get_json(f"{base_url}/api/state", headers=auth)
    feeds = _http_get_json(f"{base_url}/api/feeds/status", headers=auth)
    journal = _http_get_json(f"{base_url}/api/journal?limit=300", headers=auth)
    return state, feeds, journal


# ----------------------------------------------------------------- export builder

def _git_log(n: int = 25) -> str:
    try:
        proc = subprocess.run(
            ["git", "log", "--oneline", f"-{n}"],
            capture_output=True, text=True, cwd=REPO_ROOT, check=True
        )
        return proc.stdout
    except Exception as e:
        return f"(git log unavailable: {e})\n"


def _read_strategy_stack() -> str:
    """Best-effort: extract recent strategy-relevant fields from config.yaml."""
    cfg_path = REPO_ROOT / "config.yaml"
    if not cfg_path.exists():
        return "(config.yaml not found)\n"
    out_lines: list[str] = []
    try:
        for raw in cfg_path.read_text().splitlines():
            line = raw.strip()
            for key in (
                "fire_threshold", "min_adx_for_trade", "pattern_weight",
                "stop_loss_pct", "atr_multiplier_sl", "take_profit_r",
                "breakeven_at_r", "trailing_activate_r", "trailing_distance_r",
                "stale_exit_min", "stale_exit_max_favor_r",
                "partial_tp_at_r", "partial_tp_close_pct",
                "use_post_only_entries", "confirm_with_ticker",
                "leverage", "cooldown_seconds", "max_open_positions",
            ):
                if line.startswith(f"{key}:"):
                    out_lines.append(line)
                    break
    except Exception as e:
        return f"(config.yaml read failed: {e})\n"
    return "\n".join(out_lines)


def build_export(state: dict, feeds: dict, journal: dict) -> str:
    """Render the same Markdown export as the inline curl pipeline."""
    now = time.time()
    bot = state.get("bot", {}) or {}
    acct = state.get("account", {}) or {}
    cfg = state.get("config", {}) or {}
    hist = state.get("history_positions", []) or []
    events = bot.get("events") or []
    journal_events = journal.get("events") or []

    entries = [e for e in journal_events if e.get("kind") == "entry"]
    exits = [e for e in journal_events if e.get("kind") == "exit"]
    wins = [e for e in exits if (e.get("realized_pnl") or 0) > 0]
    losses = [e for e in exits if (e.get("realized_pnl") or 0) <= 0]
    net_wins = [e for e in exits if (e.get("net_pnl") or 0) > 0]
    total_realized = sum(e.get("realized_pnl", 0) or 0 for e in exits)
    total_fees = sum(e.get("fee", 0) or 0 for e in exits)
    mech_buckets = collections.Counter(
        e.get("entry_mech", "unknown") for e in entries
    )
    exit_reasons = collections.Counter(
        e.get("exit_reason", "unknown") for e in exits
    )

    skip_buckets: collections.Counter[str] = collections.Counter()
    for e in events:
        if e.get("kind") == "skip":
            text = e.get("text", "")
            if ": " in text:
                bucket = " ".join(text.split(": ", 1)[1].split()[:5])
                skip_buckets[bucket] += 1

    ob = feeds.get("ob_feed", {}) or {}
    tape = feeds.get("tape_feed", {}) or {}

    out: list[str] = []
    w = out.append

    w("# Bitunix Bot Status Export")
    w("")
    w(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now))}")
    w("")

    # Run state
    w("## Bot run state")
    w("")
    uptime_s = now - (bot.get("started_at") or now)
    w(f"- Uptime: {uptime_s / 60:.1f} min")
    w(f"- Mode: {cfg.get('mode')}")
    w(f"- Tick count: {bot.get('tick_count')}")
    w(f"- Signal count: {bot.get('signal_count')}")
    w(f"- Order count: {bot.get('order_count')}")
    w(f"- Error count: {bot.get('error_count')}")
    last_sig = bot.get("last_signal_text") or "(none)"
    w(f"- Last signal text:\n  > {last_sig}")
    w("")

    # Account
    w("## Account")
    w("")
    w(f"- Available: {acct.get('available')} USDT")
    w(f"- Margin in use: {acct.get('margin')}")
    w("")

    # Live config snapshot
    w("## Active config (from /api/state)")
    w("")
    w("```yaml")
    for k, v in cfg.items():
        w(f"{k}: {v}")
    w("```")
    w("")

    # Strategy stack from yaml
    w("## Strategy gate stack (from config.yaml)")
    w("")
    w("```yaml")
    w(_read_strategy_stack())
    w("```")
    w("")

    # Recent state events
    w("## Recent in-memory events (state)")
    w("")
    w(f"Last {len(events)} events visible in /api/state.")
    w("")
    w("| Age (min) | Kind | Text |")
    w("|---|---|---|")
    for e in events[:30]:
        age = (now - (e.get("ts") or 0)) / 60
        text = (e.get("text") or "")[:170].replace("|", "\\|")
        w(f"| {age:.1f} | {e.get('kind', '?')} | {text} |")
    w("")

    if skip_buckets:
        w("**Skip-reason buckets (from visible state events):**")
        w("")
        for r, n in skip_buckets.most_common():
            w(f"- {n}× `{r}`")
        w("")

    # Journal aggregates
    w("## Trade journal aggregates (lifetime, persisted on Railway volume)")
    w("")
    w(f"- Total journal events: {len(journal_events)}")
    w(f"- ENTRY events: {len(entries)}")
    w(f"- EXIT events: {len(exits)}")
    w(f"- Wins (realized_pnl > 0): {len(wins)}")
    w(f"- Losses (realized_pnl ≤ 0): {len(losses)}")
    if exits:
        w(f"- Win rate by realized: {len(wins) / len(exits) * 100:.1f}%")
        w(f"- Win rate by net_pnl (price-only, before fees): "
          f"{len(net_wins) / len(exits) * 100:.1f}%")
    w(f"- Total realized PnL: {total_realized:.4f} USDT")
    w(f"- Total fees paid: {total_fees:.4f} USDT")
    w(f"- Net (realized - fees): {total_realized - total_fees:.4f} USDT")
    if exits:
        w(f"- Average loss per trade: {total_realized / len(exits):.4f} USDT")
        w(f"- Average fee per trade: {total_fees / len(exits):.4f} USDT")
    w("")

    w("**Exit reason distribution:**")
    w("")
    for r, n in exit_reasons.most_common():
        w(f"- {n}× `{r}`")
    w("")

    w("**Entry mechanism distribution (post-Grok v8 should be MARKET):**")
    w("")
    if mech_buckets:
        for m, n in mech_buckets.most_common():
            w(f"- {n}× `{m}`")
    else:
        w("- (no entry events recorded)")
    w("")

    if entries:
        w("## Most recent 15 ENTRY events")
        w("")
        w("| Age | Symbol | Side | Score | Entry mech | Entry price |")
        w("|---|---|---|---|---|---|")
        for e in entries[-15:][::-1]:
            age = (now - (e.get("ts") or 0)) / 60
            w(f"| {age:.1f}m | {e.get('symbol', '?')} | "
              f"{e.get('side', '?')} | {e.get('score', '?')} | "
              f"{e.get('entry_mech', '?')} | {e.get('entry_price', '?')} |")
        w("")

    w("## Most recent 25 EXITS")
    w("")
    w("| Age | Symbol | Side | Reason | Hold | Realized | Fee | Net | "
      "Max favor R | Entry mech |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for e in exits[-25:][::-1]:
        age = (now - (e.get("ts") or 0)) / 60
        w(f"| {age:.1f}m | {e.get('symbol', '?')} | {e.get('side', '?')} | "
          f"{e.get('exit_reason', '?')} | {e.get('hold_time_sec', '?')}s | "
          f"{e.get('realized_pnl', '?')} | {e.get('fee', '?')} | "
          f"{e.get('net_pnl', '?')} | {e.get('max_favor_r', '?')} | "
          f"{e.get('entry_mech', '?')} |")
    w("")

    favors = [e.get("max_favor_r") for e in exits
              if e.get("max_favor_r") is not None]
    if favors:
        favors_sorted = sorted(favors)
        n = len(favors_sorted)
        w(f"**Max favor R distribution ({n} trades with tracked favor):**")
        w("")
        w(f"- min: {favors_sorted[0]:.4f}")
        w(f"- p25: {favors_sorted[n // 4]:.4f}")
        w(f"- median: {favors_sorted[n // 2]:.4f}")
        w(f"- p75: {favors_sorted[3 * n // 4]:.4f}")
        w(f"- max: {favors_sorted[-1]:.4f}")
        w(f"- count ≥ 0.5R: {sum(1 for f in favors if f >= 0.5)}")
        w(f"- count ≥ 1.0R: {sum(1 for f in favors if f >= 1.0)}")
        w("")

    # WS health
    w("## WebSocket feed health")
    w("")
    w("### OB feed (`depth_books`)")
    w(f"- connected: {ob.get('connected')}")
    w(f"- last msg age: {ob.get('last_message_age_secs', 0):.2f}s")
    msgs = ob.get("messages") or {}
    w(f"- data msgs: {msgs.get('data_msg_count')}")
    w(f"- unrecognized: {msgs.get('unrecognized_count')}")
    w(f"- unparseable: {msgs.get('unparseable_count')}")
    lc = ob.get("lifecycle") or {}
    w(f"- connects/disconnects/errors: {lc.get('connect_count')}/"
      f"{lc.get('disconnect_count')}/{lc.get('error_count')}")
    for sym, b in (ob.get("books") or {}).items():
        w(f"  - {sym}: top_bid={b.get('top_bid')} "
          f"top_ask={b.get('top_ask')} age={b.get('age_secs', 0):.1f}s")
    w("")
    w("### Tape feed (`trade`)")
    w(f"- connected: {tape.get('connected')}")
    w(f"- last msg age: {tape.get('last_message_age_secs', 0):.2f}s")
    tmsgs = tape.get("messages") or {}
    w(f"- data msgs: {tmsgs.get('data_msg_count')}")
    tlc = tape.get("lifecycle") or {}
    w(f"- connects/disconnects/errors: {tlc.get('connect_count')}/"
      f"{tlc.get('disconnect_count')}/{tlc.get('error_count')}")
    w("")

    # History
    if hist:
        w("## Last 8 historical positions (Bitunix history endpoint)")
        w("")
        w("| Symbol | Side | Entry | Exit | Qty | Notional | Fee | "
          "Realized | Closed UTC |")
        w("|---|---|---|---|---|---|---|---|---|")
        for p in hist[:8]:
            ctime_ms = int(p.get("mtime", 0) or 0)
            when = (time.strftime("%H:%M:%S", time.gmtime(ctime_ms / 1000))
                    if ctime_ms else "?")
            try:
                notional = float(p.get("entryPrice", 0)) * float(p.get("qty", 0))
            except (TypeError, ValueError):
                notional = 0.0
            w(f"| {p.get('symbol')} | {p.get('side')} | "
              f"{p.get('entryPrice')} | {p.get('closePrice')} | "
              f"{p.get('qty')} | ${notional:.2f} | {p.get('fee')} | "
              f"{p.get('realizedPNL')} | {when} |")
        w("")

    # Commits
    w("## Recent commits (strategy evolution)")
    w("")
    w("```")
    w(_git_log(25).rstrip())
    w("```")
    w("")

    return "\n".join(out)


# ----------------------------------------------------------------- API callers

def call_chat_completions(base_url: str, model: str, api_key: str,
                          system_prompt: str, user_content: str) -> str:
    """OpenAI-compatible chat completions. Works for both xAI and OpenAI."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = _http_post_json(url, payload, headers=headers, timeout=300)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {body}"
        ) from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e.reason}") from None

    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in response from {url}: {resp}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"Empty content in response: {resp}")
    return content


# ----------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--grok", action="store_true",
                   help="Ask only Grok (xAI). Default: ask both if no flag set.")
    p.add_argument("--openai", action="store_true",
                   help="Ask only OpenAI. Default: ask both if no flag set.")
    p.add_argument("--question", default=None,
                   help="Override the default review prompt.")
    p.add_argument("--export", default=None,
                   help="Use existing markdown export instead of fetching live state.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build the export, save it, but skip API calls.")
    p.add_argument("--show-export", action="store_true",
                   help="Print the full export to stdout before/instead of asking.")
    return p.parse_args()


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or default


def main() -> int:
    args = parse_args()

    # If neither flag set, ask both. If one set, only that one.
    ask_grok = args.grok or not (args.grok or args.openai)
    ask_openai = args.openai or not (args.grok or args.openai)

    # Build or load export
    if args.export:
        export_path = Path(args.export).resolve()
        if not export_path.exists():
            print(f"Export file not found: {export_path}", file=sys.stderr)
            return 1
        export_md = export_path.read_text()
        print(f"Loaded export from {export_path} ({len(export_md)} chars)")
    else:
        base_url = _env("BOT_BASE_URL", DEFAULT_BASE_URL)
        user = _env("BOT_DASHBOARD_USER", DEFAULT_USER)
        password = _env("BOT_DASHBOARD_PASSWORD")
        if not password:
            print("ERROR: BOT_DASHBOARD_PASSWORD env var not set.\n"
                  "  export BOT_DASHBOARD_PASSWORD='...'  (Railway dashboard password)",
                  file=sys.stderr)
            return 1

        print(f"Fetching live state from {base_url} ...")
        try:
            state, feeds, journal = fetch_bot_data(base_url, user, password)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} fetching dashboard: {e.read().decode(errors='replace')[:300]}",
                  file=sys.stderr)
            return 1
        except urllib.error.URLError as e:
            print(f"Network error fetching dashboard: {e.reason}", file=sys.stderr)
            return 1

        export_md = build_export(state, feeds, journal)
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        export_path = EXPORTS_DIR / f"bot-export-{ts}.md"
        export_path.write_text(export_md)
        # Update LATEST symlink-equivalent (just rewrite stable copy)
        (EXPORTS_DIR / "bot-export-LATEST.md").write_text(export_md)
        print(f"Wrote export → {export_path} ({len(export_md)} chars)")

    if args.show_export:
        print()
        print("=" * 70)
        print("EXPORT MARKDOWN")
        print("=" * 70)
        print(export_md)
        print("=" * 70)

    if args.dry_run:
        print("[--dry-run] skipping API calls.")
        return 0

    question = args.question or DEFAULT_QUESTION
    user_content = f"{export_md}\n\n---\n\n{question}"

    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    results: list[tuple[str, str | Exception, Path | None]] = []

    if ask_grok:
        key = _env("XAI_API_KEY")
        if not key:
            print("WARNING: XAI_API_KEY not set; skipping Grok.", file=sys.stderr)
        else:
            model = _env("XAI_MODEL", DEFAULT_XAI_MODEL)
            base = _env("XAI_BASE_URL", DEFAULT_XAI_BASE)
            print(f"\n→ Asking Grok ({model}) at {base} ...")
            try:
                content = call_chat_completions(
                    base, model, key, SYSTEM_PROMPT, user_content
                )
            except Exception as e:
                results.append(("Grok", e, None))
            else:
                path = REVIEWS_DIR / f"grok-{ts}.md"
                path.write_text(
                    f"# Grok review ({model})\n\n"
                    f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_\n\n"
                    f"{content}\n"
                )
                results.append(("Grok", content, path))

    if ask_openai:
        key = _env("OPENAI_API_KEY")
        if not key:
            print("WARNING: OPENAI_API_KEY not set; skipping OpenAI.", file=sys.stderr)
        else:
            model = _env("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
            base = _env("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE)
            print(f"\n→ Asking OpenAI ({model}) at {base} ...")
            try:
                content = call_chat_completions(
                    base, model, key, SYSTEM_PROMPT, user_content
                )
            except Exception as e:
                results.append(("OpenAI", e, None))
            else:
                path = REVIEWS_DIR / f"openai-{ts}.md"
                path.write_text(
                    f"# OpenAI review ({model})\n\n"
                    f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_\n\n"
                    f"{content}\n"
                )
                results.append(("OpenAI", content, path))

    # Display
    any_ok = False
    for provider, content, path in results:
        print()
        print("=" * 70)
        print(f"{provider} review")
        if path:
            print(f"saved to: {path}")
        print("=" * 70)
        if isinstance(content, Exception):
            print(f"ERROR: {content}")
        else:
            any_ok = True
            print(content)

    if not results:
        print("\nNo reviewers were called. Set --grok / --openai flags or env keys.")
        return 1

    return 0 if any_ok else 2


if __name__ == "__main__":
    sys.exit(main())
