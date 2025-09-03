"""
Multi-symbol candidate scanner with priority-based selection.
Implements watchlist building, WS subscriptions, priority scoring, and symbol switching logic.
Enhanced with per-symbol calibrations for optimal performance.
"""
from __future__ import annotations
import time
import yaml
import ccxt
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ingest.ws_bybit import get_client
from features.symbol_specific import get_calibration_manager
import logging

logger = logging.getLogger(__name__)

@dataclass
class CandidateScore:
    symbol: str
    priority: float
    vol_score: float
    flow_score: float
    info_score: float
    cost_score: float
    spread_bps: float
    depth_usd: float
    atr_pct: float
    timestamp: float

@dataclass
class ScannerState:
    active_symbol: Optional[str] = None
    lock_until: float = 0
    cooldown_until: float = 0
    last_switch_ts: float = 0
    switch_count: int = 0

def load_scanner_config() -> dict:
    """Load scanner configuration from config.yaml"""
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("scanner", {})

def build_watchlist(exchange="bybit", top_n=20) -> List[str]:
    """
    Build watchlist of top-N futures symbols by 24h volume.
    Returns list of symbols like ['BTCUSDT', 'ETHUSDT', ...]
    """
    try:
        ex = ccxt.bybit({"enableRateLimit": True})
        markets = ex.fetch_markets()
        
        # Filter futures markets
        futures = [m for m in markets if m and m.get('type') == 'swap' and m.get('settle') == 'USDT']
        
        # Get tickers for volume data
        tickers = ex.fetch_tickers()
        if not tickers:
            tickers = {}
        
        # Combine and sort by volume
        candidates = []
        for market in futures:
            if not market:
                continue
            symbol = market.get('symbol')
            if symbol and symbol in tickers:
                ticker = tickers[symbol]
                volume = ticker.get('quoteVolume', 0) if ticker else 0
                volume = volume or 0
                candidates.append((symbol, volume))
        
        # Sort by volume descending and take top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        watchlist = [symbol.replace('/', '') for symbol, _ in candidates[:top_n]]
        
        logger.info(f"Built watchlist: {watchlist[:5]}... ({len(watchlist)} total)")
        return watchlist
        
    except Exception as e:
        logger.error(f"Failed to build watchlist: {e}")
        # Fallback to hardcoded list
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT']

def subscribe_watchlist(symbols: List[str]) -> None:
    """
    Subscribe to WS feeds for all symbols in watchlist:
    - orderbook.50.SYMBOL
    - kline.1.SYMBOL (1m)
    - publicTrade.SYMBOL
    """
    try:
        client = get_client()
        
        for symbol in symbols:
            client.subscribe_orderbook(symbol, depth=50)
            client.subscribe_kline(symbol, interval="1m")
            client.subscribe_trades(symbol)
            
        logger.info(f"Subscribed to WS feeds for {len(symbols)} symbols")
        
    except Exception as e:
        logger.error(f"Failed to subscribe watchlist: {e}")

def compute_priority(symbol: str, config: dict) -> Optional[CandidateScore]:
    """
    Compute priority score P = Vol + Flow + Info - Cost for a symbol.
    Uses symbol-specific calibrations for optimal performance.
    Returns CandidateScore or None if data unavailable.
    """
    try:
        client = get_client()
        cal_manager = get_calibration_manager()
        now = time.time()
        
        # Get symbol-specific thresholds
        thresholds = cal_manager.get_symbol_thresholds(symbol)
        
        # Try WebSocket first, fallback to REST
        ob = client.get_orderbook_snapshot(symbol)
        
        # REST API fallback if WS data not available
        if not ob or not ob.get('bids') or not ob.get('asks'):
            logger.debug(f"📡 {symbol}: WS data not available, falling back to REST API")
            import ccxt
            ccxt_client = ccxt.bybit({
                'sandbox': False,
                'options': {'defaultType': 'linear'}
            })
            
            # Convert symbol format: BTCUSDT:USDT -> BTC/USDT
            ccxt_symbol = symbol.replace(':USDT', '').replace('USDT', '/USDT')
            ob = ccxt_client.fetch_order_book(ccxt_symbol)
        
        if not ob or not ob.get('bids') or not ob.get('asks'):
            logger.warning(f"❌ {symbol}: No orderbook data")
            return None
            
        # Compute spread and depth
        try:
            bid_price = ob['bids'][0][0] if ob['bids'] and ob['bids'][0] and ob['bids'][0][0] is not None else "0"
            ask_price = ob['asks'][0][0] if ob['asks'] and ob['asks'][0] and ob['asks'][0][0] is not None else "0"
            best_bid = float(str(bid_price))
            best_ask = float(str(ask_price))
        except (ValueError, IndexError, TypeError):
            logger.warning(f"❌ {symbol}: Invalid bid/ask data")
            return None
            
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        
        if mid_price == 0:
            logger.warning(f"❌ {symbol}: Invalid mid price")
            return None
            
        spread_bps = ((best_ask - best_bid) / mid_price) * 10000
        
        # Top 10 levels depth in USD
        depth_usd = 0.0
        try:
            for price, size in (ob['bids'][:10] + ob['asks'][:10]):
                if price is not None and size is not None:
                    depth_usd += float(str(price)) * float(str(size))
        except (ValueError, TypeError):
            logger.warning(f"❌ {symbol}: Invalid depth data")
            depth_usd = 0.0
            
        logger.debug(f"🔍 {symbol}: spread={spread_bps:.2f}bps, depth=${depth_usd:,.0f}")
        
        # Base score components (0-100 scale)
        base_vol_score = min(100, depth_usd / 50000)     # Volume proxy from depth
        base_flow_score = max(0, 100 - spread_bps * 2)  # Tighter spread = better flow  
        base_info_score = min(100, depth_usd / 100000)  # Liquidity information
        base_cost_score = min(100, spread_bps * 3)      # Transaction cost penalty
        
        # Apply symbol-specific calibration weights
        priority = cal_manager.compute_symbol_priority(
            symbol, base_vol_score, base_flow_score, base_info_score, base_cost_score
        )
        
        logger.info(f"🧮 {symbol}: P={priority:.1f} (V={base_vol_score:.1f}, "
                   f"F={base_flow_score:.1f}, I={base_info_score:.1f}, C={base_cost_score:.1f})")
        
        # Apply symbol-specific filters
        if spread_bps > thresholds['max_spread_bps']:
            logger.warning(f"❌ {symbol}: spread {spread_bps:.2f} > {thresholds['max_spread_bps']}")
            return None
        if depth_usd < thresholds['min_depth_usd']:
            logger.warning(f"❌ {symbol}: depth ${depth_usd:,.0f} < ${thresholds['min_depth_usd']:,.0f}")
            return None
            
        return CandidateScore(
            symbol=symbol,
            priority=float(priority),
            vol_score=float(base_vol_score),
            flow_score=float(base_flow_score),
            info_score=float(base_info_score),
            cost_score=float(base_cost_score),
            spread_bps=float(spread_bps),
            depth_usd=float(depth_usd),
            atr_pct=0,  # Simplified for demo
            timestamp=now
        )
        
    except Exception as e:
        logger.error(f"Failed to compute priority for {symbol}: {e}")
        return None

def pick_active(prev_symbol: Optional[str], scores: List[CandidateScore], 
                state: ScannerState, config: dict) -> Tuple[str, ScannerState]:
    """
    Pick active symbol using hysteresis logic:
    - lead_margin: new symbol must be X% better to switch
    - lock_min_sec: minimum time before allowing switch
    - cooldown_after_close_sec: cooldown after closing position
    - trade_one_at_a_time: only one active symbol
    """
    now = time.time()
    
    # Apply cooldown and lock constraints
    if now < state.lock_until or now < state.cooldown_until:
        return prev_symbol or scores[0].symbol if scores else 'BTCUSDT', state
    
    # Filter valid candidates (priority > 0)
    valid = [s for s in scores if s.priority > 0]
    if not valid:
        return prev_symbol or 'BTCUSDT', state
        
    # Sort by priority descending
    valid.sort(key=lambda x: x.priority, reverse=True)
    best_candidate = valid[0]
    
    # If no previous symbol, pick the best
    if not prev_symbol:
        new_state = ScannerState(
            active_symbol=best_candidate.symbol,
            lock_until=now + config.get('lock_min_sec', 120),
            last_switch_ts=now,
            switch_count=state.switch_count + 1
        )
        logger.info(f"Initial symbol selection: {best_candidate.symbol} (P={best_candidate.priority:.1f})")
        return best_candidate.symbol, new_state
    
    # Find current symbol score
    current_score = None
    for s in valid:
        if s.symbol == prev_symbol:
            current_score = s
            break
            
    # Apply hysteresis: new symbol must be lead_margin% better
    lead_margin = config.get('lead_margin', 0.15)
    if current_score and best_candidate.priority > current_score.priority * (1 + lead_margin):
        new_state = ScannerState(
            active_symbol=best_candidate.symbol,
            lock_until=now + config.get('lock_min_sec', 120),
            last_switch_ts=now,
            switch_count=state.switch_count + 1
        )
        logger.info(f"Symbol switch: {prev_symbol} -> {best_candidate.symbol} "
                   f"(P: {current_score.priority:.1f} -> {best_candidate.priority:.1f})")
        return best_candidate.symbol, new_state
    
    # No switch - keep current
    return prev_symbol, state

def log_scanner_data(scores: List[CandidateScore], active_symbol: str) -> None:
    """Log scanner data to logs/scanner.csv"""
    import os
    
    log_path = "logs/scanner.csv"
    os.makedirs("logs", exist_ok=True)
    
    # Create header if file doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("ts,symbol,is_active,priority,vol_score,flow_score,info_score,cost_score,spread_bps,depth_usd,atr_pct\n")
    
    # Append data
    with open(log_path, 'a') as f:
        for score in scores:
            is_active = 1 if score.symbol == active_symbol else 0
            f.write(f"{score.timestamp},{score.symbol},{is_active},{score.priority:.2f},"
                   f"{score.vol_score:.2f},{score.flow_score:.2f},{score.info_score:.2f},"
                   f"{score.cost_score:.2f},{score.spread_bps:.4f},{score.depth_usd:.0f},"
                   f"{score.atr_pct:.6f}\n")
