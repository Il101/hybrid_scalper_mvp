"""
REAL OHLCV via ccxt (no synthetic fallback).
Requires: ccxt installed + reachable exchange API.
Default exchange: binance, market_type: futures.
Enhanced with retry logic and error handling for robust operation.
"""
from __future__ import annotations
import time
import logging
import random
from typing import Optional, List, Dict, Any
import pandas as pd

import ccxt

# Setup logging
logger = logging.getLogger(__name__)

# Caching for scalping
_ohlcv_cache = {}

_TIMEFRAME_ALIAS = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
    "1d": "1d"
}

def _exponential_backoff_with_jitter(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """Calculate exponential backoff delay with jitter"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    # Add jitter (±20%)
    jitter = delay * 0.2 * (2 * random.random() - 1)
    return max(0.1, delay + jitter)

def _retry_on_network_error(func, max_retries: int = 3, *args, **kwargs):
    """Retry decorator for network operations"""
    last_exception = Exception("Max retries exceeded")
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            last_exception = e
            if attempt < max_retries:
                delay = _exponential_backoff_with_jitter(attempt)
                logger.warning(f"Network error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
            else:
                logger.error(f"Max retries ({max_retries}) exceeded for network operation")
        except (ccxt.ExchangeError, ccxt.BadSymbol) as e:
            logger.error(f"Exchange error (no retry): {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error (no retry): {e}")
            raise e
    
    raise last_exception

_TIMEFRAME_ALIAS = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
    "1d": "1d"
}

def _to_unified_symbol(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"
    if symbol.endswith("USD"):
        return symbol[:-3] + "/USD"
    return symbol

def _build_exchange(name: str, market_type: str = "futures"):
    name = (name or "binance").lower()
    if not hasattr(ccxt, name):
        raise ValueError(f"Exchange '{name}' not supported by ccxt.")
    klass = getattr(ccxt, name)
    kwargs: Dict[str, Any] = {"enableRateLimit": True}
    if name in ("binance",):
        kwargs["options"] = {"defaultType": "future" if market_type == "futures" else "spot"}
    elif name in ("bybit",):
        kwargs["options"] = {"defaultType": "swap" if market_type == "futures" else "spot"}
    return klass(kwargs)

def _paginate_ohlcv(ex, symbol: str, timeframe: str, limit: int) -> list:
    """Paginate OHLCV with retry logic"""
    out = []
    per_call = min(limit, 1000)
    ms_per_bar = ex.parse_timeframe(timeframe) * 1000
    now_ms = ex.milliseconds()
    since = now_ms - ms_per_bar * limit
    
    while len(out) < limit:
        # Use retry wrapper for fetch_ohlcv
        chunk = _retry_on_network_error(
            ex.fetch_ohlcv, 
            max_retries=3,
            symbol=symbol, 
            timeframe=timeframe, 
            since=since, 
            limit=per_call
        )
        
        if not chunk:
            break
        out += chunk
        since = chunk[-1][0] + ms_per_bar
        
        # Rate limiting with jitter
        sleep_time = (ex.rateLimit / 1000.0 if getattr(ex, "rateLimit", 0) else 0.2)
        sleep_time += random.uniform(0, 0.1)  # Add jitter
        time.sleep(sleep_time)
        
        if len(chunk) < per_call:
            break
    return out[-limit:]

def get_ohlcv(symbol: str, tf: str, exchange: str = "binance", market_type: str = "futures",
              limit: int = 1000) -> pd.DataFrame:
    """Get OHLCV with robust error handling and retry logic"""
    logger.debug(f"Fetching OHLCV: {symbol}, {tf}, {exchange}, limit={limit}")
    
    try:
        ex = _build_exchange(exchange, market_type)
        ex.load_markets()
        
        sym = _to_unified_symbol(symbol)
        tf_ex = _TIMEFRAME_ALIAS.get(tf, "5m")
        
        raw = _paginate_ohlcv(ex, sym, tf_ex, limit=limit)
        
        if not raw:
            raise RuntimeError(f"No OHLCV data returned for {sym} on {exchange} ({market_type})")
        
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
        logger.info(f"Successfully fetched {len(df)} OHLCV bars for {sym}")
        return df.set_index("timestamp")
        
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
        raise

def get_ohlcv_cached(symbol: str, tf: str, exchange: str = "binance", 
                    market_type: str = "futures", limit: int = 1000, 
                    ttl_seconds: int = 30) -> pd.DataFrame:
    """Cached OHLCV for high-frequency scalping"""
    key = f"{symbol}_{tf}_{exchange}_{market_type}_{limit}"
    now = time.time()
    
    if key in _ohlcv_cache:
        data, timestamp = _ohlcv_cache[key]
        if now - timestamp < ttl_seconds:
            return data.copy()
    
    data = get_ohlcv(symbol, tf, exchange, market_type, limit)
    _ohlcv_cache[key] = (data, now)
    return data
