from __future__ import annotations
from typing import Optional, Dict, Any
import yaml
import os

# Load configuration for orderflow parameters
def _load_orderflow_config() -> Dict[str, Any]:
    """Load orderflow configuration from config.yaml"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('orderflow', {})
    except Exception:
        return {}

_CONFIG = _load_orderflow_config()

def compute_spread_bps(orderbook: Optional[Dict[str, Any]]) -> float:
    """Compute bid-ask spread in basis points"""
    default_spread = _CONFIG.get('default_spread_bps', 10.0)
    
    if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
        return default_spread
        
    best_bid = float(orderbook['bids'][0][0])
    best_ask = float(orderbook['asks'][0][0])
    mid = (best_bid + best_ask) / 2.0
    
    if mid <= 0: 
        return default_spread
        
    return float((best_ask - best_bid) / mid * 10000.0)

def compute_obi(orderbook: Optional[Dict[str, Any]], depth_levels: Optional[int] = None) -> float:
    """Compute Order Book Imbalance with configurable depth"""
    if depth_levels is None:
        depth_levels = _CONFIG.get('obi_depth_levels', 5)
        
    if not orderbook: 
        return 0.0
        
    bids = orderbook.get('bids', [])[:depth_levels]
    asks = orderbook.get('asks', [])[:depth_levels]
    
    bid_vol = sum(float(x[1]) for x in bids)
    ask_vol = sum(float(x[1]) for x in asks)
    denom = bid_vol + ask_vol
    
    if denom <= 0: 
        return 0.0
        
    return float((bid_vol - ask_vol) / denom)

def obi_to_score(obi: float) -> float:
    """Convert OBI (-1 to +1) to score (0 to 100)"""
    return float(max(0.0, min(100.0, (obi + 1.0) * 50.0)))
