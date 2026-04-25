"""Bitunix Futures REST client.

Signing (verified against the unofficial SDK at github.com/0xCherryBlueZu/bitunix
and the Bitunix signing spec):

    message = nonce + timestamp + api_key + query_string + body
    digest  = sha256_hex(message)
    sign    = sha256_hex(digest + secret_key)

  * query_string = urlencode(sorted(params.items()))   — empty for no params
  * body         = json.dumps(body, separators=(',',':'), sort_keys=True) — empty for GET
  * timestamp    = str(int(time.time() * 1000))        — ms since epoch
  * nonce        = base64(32 random bytes)             — any 32-ish char random

Headers: api-key, timestamp, nonce, sign, Content-Type (on POST).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import time
from typing import Any
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://fapi.bitunix.com"

SIDE_BUY = "BUY"
SIDE_SELL = "SELL"
TRADE_OPEN = "OPEN"
TRADE_CLOSE = "CLOSE"


class BitunixError(RuntimeError):
    def __init__(self, code: int, msg: str, payload: dict[str, Any]):
        super().__init__(f"Bitunix error {code}: {msg}")
        self.code = code
        self.msg = msg
        self.payload = payload


class BitunixClient:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = BASE_URL,
        margin_coin: str = "USDT",
        timeout: float = 10.0,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.margin_coin = margin_coin
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------ signing

    @staticmethod
    def _nonce() -> str:
        return base64.b64encode(secrets.token_bytes(32)).decode()

    def _sign(self, nonce: str, timestamp: str, query_string: str, body: str) -> str:
        message = f"{nonce}{timestamp}{self.api_key}{query_string}{body}"
        digest = hashlib.sha256(message.encode()).hexdigest()
        return hashlib.sha256((digest + self.secret_key).encode()).hexdigest()

    def _headers(self, query_string: str, body: str, is_post: bool) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        n = self._nonce()
        h = {
            "api-key": self.api_key,
            "timestamp": ts,
            "nonce": n,
            "sign": self._sign(n, ts, query_string, body),
        }
        if is_post:
            h["Content-Type"] = "application/json"
        return h

    # ------------------------------------------------------------------ request

    @staticmethod
    def _query_signing_string(params: dict[str, Any]) -> str:
        # Bitunix sign spec: sort by key ASCII, then concatenate key+value pairs
        # with NO separators. Example: {"id":1,"uid":200} -> "id1uid200".
        return "".join(f"{k}{v}" for k, v in sorted(params.items()))

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        qs = self._query_signing_string(params) if params else ""
        headers = self._headers(qs, "", is_post=False)
        url = f"{self.base_url}{path}"
        log.debug("GET %s params=%s", path, params)
        r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        return self._parse(r)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        body_str = json.dumps(body, separators=(",", ":"), sort_keys=True)
        headers = self._headers("", body_str, is_post=True)
        url = f"{self.base_url}{path}"
        log.debug("POST %s body=%s", path, body_str)
        r = self.session.post(url, data=body_str, headers=headers, timeout=self.timeout)
        return self._parse(r)

    @staticmethod
    def _parse(r: requests.Response) -> dict[str, Any]:
        r.raise_for_status()
        data = r.json()
        code = data.get("code", 0)
        if code != 0:
            raise BitunixError(code, data.get("msg", ""), data)
        return data

    # ------------------------------------------------------------------ account

    def account(self) -> dict[str, Any]:
        data = self._get("/api/v1/futures/account", {"marginCoin": self.margin_coin})
        return data.get("data") or {}

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return self._post(
            "/api/v1/futures/account/change_leverage",
            {"marginCoin": self.margin_coin, "symbol": symbol, "leverage": leverage},
        )

    def set_margin_mode(self, symbol: str, mode: str = "ISOLATION") -> dict[str, Any]:
        """mode: ISOLATION or CROSS."""
        return self._post(
            "/api/v1/futures/account/change_margin_mode",
            {"marginCoin": self.margin_coin, "symbol": symbol, "marginMode": mode},
        )

    def set_position_mode(self, mode: str = "ONE_WAY") -> dict[str, Any]:
        """mode: ONE_WAY or HEDGE."""
        return self._post(
            "/api/v1/futures/account/change_position_mode",
            {"positionMode": mode},
        )

    # ------------------------------------------------------------------ market

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        price_type: str = "LAST_PRICE",
    ) -> list[dict[str, Any]]:
        data = self._get(
            "/api/v1/futures/market/kline",
            {"symbol": symbol, "interval": interval, "limit": limit, "type": price_type},
        )
        return data.get("data") or []

    def ticker(self, symbol: str) -> dict[str, Any]:
        data = self._get("/api/v1/futures/market/tickers", {"symbols": symbol})
        rows = data.get("data") or []
        return rows[0] if rows else {}

    def trading_pairs(self) -> list[dict[str, Any]]:
        data = self._get("/api/v1/futures/market/trading_pairs")
        return data.get("data") or []

    def funding_rate(self, symbol: str) -> dict[str, Any]:
        """Returns dict with: symbol, markPrice, lastPrice, fundingRate,
        fundingInterval (hours), nextFundingTime (ms).

        fundingRate is per interval (typically 8h on Bitunix). Positive =
        longs pay shorts (crowded longs); negative = shorts pay longs.

        Note: the spec docs show `data` as a list but the live API returns
        it as a dict. We handle both shapes for safety.
        """
        data = self._get("/api/v1/futures/market/funding_rate", {"symbol": symbol})
        d = data.get("data")
        if isinstance(d, list):
            return d[0] if d else {}
        return d or {}

    # ------------------------------------------------------------------ positions

    def pending_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        data = self._get("/api/v1/futures/position/get_pending_positions", {"symbol": symbol})
        return data.get("data") or []

    def history_positions(
        self, symbol: str | None = None, limit: int = 50, skip: int = 0
    ) -> dict[str, Any]:
        data = self._get(
            "/api/v1/futures/position/get_history_positions",
            {"symbol": symbol, "limit": limit, "skip": skip},
        )
        return data.get("data") or {"positionList": [], "total": 0}

    def history_orders(
        self, symbol: str | None = None, limit: int = 50, skip: int = 0
    ) -> dict[str, Any]:
        data = self._get(
            "/api/v1/futures/trade/get_history_orders",
            {"symbol": symbol, "limit": limit, "skip": skip},
        )
        return data.get("data") or {"orderList": [], "total": 0}

    def modify_tpsl_order(
        self,
        order_id: str,
        tp_price: str | None = None,
        sl_price: str | None = None,
        tp_qty: str | None = None,
        sl_qty: str | None = None,
        tp_stop_type: str = "LAST_PRICE",
        sl_stop_type: str = "LAST_PRICE",
        tp_order_type: str = "MARKET",
        sl_order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """Modify a SPECIFIC TPSL trigger order by its orderId.

        IMPORTANT: Bitunix stores TPSL as separate trigger orders. When you
        place an order with both tpPrice and slPrice, two separate triggers
        are created: one TP-only row and one SL-only row, each with its own
        `id` returned by /tpsl/get_pending_orders.

        The /tpsl/position/modify_order endpoint (deprecated by us — it
        doesn't actually persist changes to the underlying triggers) is
        useless for ratcheting. Use this method with the trigger's own
        orderId from pending_tpsl() instead.
        """
        body: dict[str, Any] = {"orderId": str(order_id)}
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
            body["tpStopType"] = tp_stop_type
            body["tpOrderType"] = tp_order_type
            if tp_qty:
                body["tpQty"] = str(tp_qty)
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
            body["slStopType"] = sl_stop_type
            body["slOrderType"] = sl_order_type
            if sl_qty:
                body["slQty"] = str(sl_qty)
        return self._post("/api/v1/futures/tpsl/modify_order", body)

    def cancel_tpsl_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        return self._post(
            "/api/v1/futures/tpsl/cancel_order",
            {"symbol": symbol, "orderId": str(order_id)},
        )

    def pending_tpsl(self, symbol: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch pending TP/SL trigger orders. Bitunix stores TPSL as separate
        trigger orders, one per side (one TP-only row + one SL-only row per
        position), so the position object alone does not carry SL/TP prices."""
        data = self._get(
            "/api/v1/futures/tpsl/get_pending_orders",
            {"symbol": symbol, "limit": limit},
        )
        d = data.get("data") or {}
        if isinstance(d, dict):
            return d.get("orderList") or d.get("list") or []
        return d if isinstance(d, list) else []

    # ------------------------------------------------------------------ trading

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: str,
        order_type: str = "MARKET",
        price: str | None = None,
        trade_side: str = TRADE_OPEN,
        tp_price: str | None = None,
        sl_price: str | None = None,
        tp_order_type: str = "MARKET",
        sl_order_type: str = "MARKET",
        tp_stop_type: str = "LAST_PRICE",
        sl_stop_type: str = "LAST_PRICE",
        reduce_only: bool = False,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "qty": str(qty),
            "orderType": order_type,
            "tradeSide": trade_side,
        }
        if price is not None:
            body["price"] = str(price)
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
            body["tpOrderType"] = tp_order_type
            body["tpStopType"] = tp_stop_type
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
            body["slOrderType"] = sl_order_type
            body["slStopType"] = sl_stop_type
        if reduce_only:
            body["reduceOnly"] = True
        if client_id:
            body["clientId"] = client_id
        data = self._post("/api/v1/futures/trade/place_order", body)
        return data.get("data") or {}

    def flash_close_position(self, position_id: str) -> dict[str, Any]:
        return self._post(
            "/api/v1/futures/trade/flash_close_position",
            {"positionId": position_id},
        )

    def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        """Cancel a single futures order by orderId.

        POST /api/v1/futures/trade/cancel_orders. Bitunix takes a list
        shape even for single cancellation; we pass a one-element orderList.
        """
        return self._post(
            "/api/v1/futures/trade/cancel_orders",
            {"symbol": symbol, "orderList": [{"orderId": str(order_id)}]},
        )
