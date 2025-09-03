"""
WS manager facade (currently wraps BybitWS).
API:
  ws_subscribe(symbol, kind="kline", interval="1m")
  ws_get(symbol, kind="kline", interval="1m")
"""
from __future__ import annotations
from typing import Optional
from ingest.ws_bybit import get_client, ensure_orderbook, ensure_kline

def ws_subscribe(symbol: str, kind: str = "kline", interval: str = "1m"):
    c = get_client()
    if kind == "orderbook":
        c.subscribe_orderbook(symbol)
    elif kind == "kline":
        c.subscribe_kline(symbol, interval)
    elif kind == "trades":
        c.subscribe_trades(symbol)

def ws_ensure(symbol: str, kind: str = "kline", interval: str = "1m"):
    if kind == "orderbook":
        ensure_orderbook(symbol)
    elif kind == "kline":
        ensure_kline(symbol, interval)

def ws_get(symbol: str, kind: str = "kline", interval: str = "1m") -> Optional[dict]:
    c = get_client()
    if kind == "orderbook":
        return c.get_orderbook_snapshot(symbol)
    elif kind == "kline":
        return c.get_last_kline(symbol, interval)
    elif kind == "trades":
        return c.get_last_trade(symbol)
    return None
