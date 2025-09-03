"""
REAL orderbook via ccxt (no fallback).
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import ccxt

def get_exchange(name: str):
    name = name.lower()
    if not hasattr(ccxt, name):
        raise ValueError(f"Exchange '{name}' not supported by ccxt.")
    ex = getattr(ccxt, name)({"enableRateLimit": True})
    return ex

def fetch_orderbook(exchange_name: str, symbol: str, limit: int = 10) -> Dict[str, Any]:
    ex = get_exchange(exchange_name)
    ob = ex.fetch_order_book(symbol, limit=limit)
    if not ob or "bids" not in ob or "asks" not in ob:
        raise RuntimeError(f"Failed to fetch orderbook for {symbol} on {exchange_name}.")
    return {"bids": ob.get("bids", []), "asks": ob.get("asks", []), "ts": ob.get("timestamp")}
