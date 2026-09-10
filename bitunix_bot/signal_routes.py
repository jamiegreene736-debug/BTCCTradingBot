"""Authenticated routes registered beneath the dashboard's existing auth gate."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import asdict

from flask import Flask, Response, jsonify, request

from .signal_scanner import SignalScanner

log = logging.getLogger(__name__)


def register_signal_routes(app: Flask, scanner: SignalScanner | None) -> None:
    @app.get("/api/signals")
    def signals() -> Response | tuple[Response, int]:
        if scanner is None:
            return jsonify(
                {"error": "Enable signals.enabled and restart the backend"}
            ), 503
        return jsonify(scanner.snapshot())

    @app.post("/api/signals/<action>")
    def update_signals(action: str) -> Response | tuple[Response, int]:
        if scanner is None:
            return jsonify({"error": "Intraday scanner unavailable"}), 503
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "A JSON object is required"}), 400
        try:
            if action == "settings":
                scanner.update_settings(body)
                return jsonify({"ok": True})
            if action == "track":
                return jsonify({"ok": True, "trade": asdict(scanner.track(body))})
            if action == "close":
                return jsonify({"ok": True, "trade": asdict(scanner.close_track(body))})
            return jsonify({"error": "Unknown tracking action"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except sqlite3.Error:
            log.exception("Could not persist intraday tracking change")
            return jsonify(
                {"error": "Could not save tracking state; check backend storage"}
            ), 503
