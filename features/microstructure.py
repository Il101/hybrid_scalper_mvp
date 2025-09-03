"""
Microstructure features for advanced scalping
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd

def order_flow_imbalance(trades_data: List[Dict[str, Any]]) -> float:
    """
    Calculate aggressive buy/sell ratio from trade ticks
    Returns value between -1 (all sells) and +1 (all buys)
    """
    if not trades_data:
        return 0.0
    
    buy_vol = sum(float(t.get('qty', 0)) for t in trades_data if t.get('side') == 'buy')
    sell_vol = sum(float(t.get('qty', 0)) for t in trades_data if t.get('side') == 'sell')
    
    total_vol = buy_vol + sell_vol
    if total_vol == 0:
        return 0.0
    
    return float((buy_vol - sell_vol) / total_vol)

def price_impact_decay(orderbook_snapshots: List[Dict[str, Any]]) -> float:
    """
    Measure how fast orderbook recovers after large orders
    Returns recovery speed factor (higher = faster recovery)
    """
    if len(orderbook_snapshots) < 2:
        return 1.0
    
    spreads = []
    for snapshot in orderbook_snapshots:
        asks = snapshot.get('asks', [])
        bids = snapshot.get('bids', [])
        
        if asks and bids:
            best_ask = float(asks[0][0])
            best_bid = float(bids[0][0])
            spread = (best_ask - best_bid) / best_bid * 10000  # bps
            spreads.append(spread)
    
    if len(spreads) < 2:
        return 1.0
    
    # Calculate how spread normalizes over time
    max_spread = max(spreads)
    min_spread = min(spreads)
    
    if max_spread == min_spread:
        return 1.0
    
    # Recovery factor: how much spread contracted from max to min
    recovery_factor = 1.0 - (spreads[-1] - min_spread) / (max_spread - min_spread)
    return float(max(0.0, min(2.0, recovery_factor)))

def volume_profile_score(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Analyze volume distribution across price levels
    Returns score 0-100 based on volume concentration
    """
    if len(df) < lookback:
        return 50.0
    
    recent = df.tail(lookback)
    
    # Create price buckets
    price_range = recent['high'].max() - recent['low'].min()
    if price_range == 0:
        return 50.0
    
    bucket_size = price_range / 10  # 10 price buckets
    
    # Assign volume to price buckets
    volume_profile = np.zeros(10)
    
    for _, row in recent.iterrows():
        # Approximate volume distribution within each bar
        mid_price = (row['high'] + row['low']) / 2
        bucket_idx = min(9, int((mid_price - recent['low'].min()) / bucket_size))
        volume_profile[bucket_idx] += row['volume']
    
    # Calculate concentration - higher concentration = more directional
    total_volume = volume_profile.sum()
    if total_volume == 0:
        return 50.0
    
    volume_profile_norm = volume_profile / total_volume
    
    # Gini coefficient for concentration
    sorted_profile = np.sort(volume_profile_norm)
    n = len(sorted_profile)
    cumsum = np.cumsum(sorted_profile)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    # Convert to score (higher gini = more concentration = higher score)
    concentration_score = gini * 100
    
    return float(max(0.0, min(100.0, concentration_score)))

def tick_rule_momentum(trades_data: List[Dict[str, Any]]) -> float:
    """
    Calculate momentum based on tick rule (upticks vs downticks)
    Returns value between -1 and +1
    """
    if len(trades_data) < 2:
        return 0.0
    
    upticks = 0
    downticks = 0
    
    for i in range(1, len(trades_data)):
        current_price = float(trades_data[i].get('price', 0))
        prev_price = float(trades_data[i-1].get('price', 0))
        
        if current_price > prev_price:
            upticks += 1
        elif current_price < prev_price:
            downticks += 1
    
    total_ticks = upticks + downticks
    if total_ticks == 0:
        return 0.0
    
    return float((upticks - downticks) / total_ticks)

def bid_ask_pressure(orderbook: Dict[str, Any], depth_levels: int = 5) -> float:
    """
    Calculate pressure from bid/ask size imbalance
    Returns value between -1 (ask heavy) and +1 (bid heavy)
    """
    asks = orderbook.get('asks', [])
    bids = orderbook.get('bids', [])
    
    if not asks or not bids:
        return 0.0
    
    ask_volume = sum(float(ask[1]) for ask in asks[:depth_levels])
    bid_volume = sum(float(bid[1]) for bid in bids[:depth_levels])
    
    total_volume = ask_volume + bid_volume
    if total_volume == 0:
        return 0.0
    
    return float((bid_volume - ask_volume) / total_volume)

def compute_spread_bps(orderbook: Dict[str, Any]) -> float:
    """
    Calculate bid-ask spread in basis points
    Returns spread in bps, or 0.0 if invalid orderbook
    """
    asks = orderbook.get('asks', [])
    bids = orderbook.get('bids', [])
    
    if not asks or not bids:
        return 0.0
    
    try:
        best_ask = float(asks[0][0])
        best_bid = float(bids[0][0])
        
        if best_bid <= 0:
            return 0.0
            
        spread_bps = (best_ask - best_bid) / best_bid * 10000
        return float(max(0.0, spread_bps))
        
    except (IndexError, ValueError, TypeError):
        return 0.0

def compute_obi(orderbook: Dict[str, Any], depth_levels: int = 5) -> float:
    """
    Calculate Order Book Imbalance (OBI)
    Returns value between -1 and +1
    """
    asks = orderbook.get('asks', [])
    bids = orderbook.get('bids', [])
    
    if not asks or not bids:
        return 0.0
    
    try:
        # Calculate volumes at different levels
        ask_volume = sum(float(ask[1]) for ask in asks[:depth_levels] if len(ask) >= 2)
        bid_volume = sum(float(bid[1]) for bid in bids[:depth_levels] if len(bid) >= 2)
        
        total_volume = ask_volume + bid_volume
        if total_volume == 0:
            return 0.0
        
        # OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        obi = (bid_volume - ask_volume) / total_volume
        return float(max(-1.0, min(1.0, obi)))
        
    except (IndexError, ValueError, TypeError):
        return 0.0

def compute_volume_profile(df: pd.DataFrame, bins: int = 10) -> dict:
    """
    Compute volume profile for price levels
    Returns dict with price levels and corresponding volumes
    """
    if len(df) == 0:
        return {}
    
    try:
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        if price_max == price_min:
            return {float(price_min): df['volume'].sum()}
        
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        volume_profile = {}
        
        for i in range(len(bin_edges) - 1):
            bin_low = bin_edges[i]
            bin_high = bin_edges[i + 1]
            bin_center = (bin_low + bin_high) / 2
            
            # Find bars that overlap with this price bin
            overlapping = df[
                (df['low'] <= bin_high) & (df['high'] >= bin_low)
            ]
            
            total_volume = overlapping['volume'].sum()
            volume_profile[float(bin_center)] = float(total_volume)
        
        return volume_profile
        
    except Exception:
        return {}

def compute_market_depth(orderbook: dict, levels: int = 5) -> dict:
    """
    Compute market depth metrics from orderbook
    Returns dict with depth metrics
    """
    asks = orderbook.get('asks', [])
    bids = orderbook.get('bids', [])
    
    if not asks or not bids:
        return {'bid_depth': 0.0, 'ask_depth': 0.0, 'total_depth': 0.0}
    
    try:
        # Calculate depth for specified levels
        bid_depth = sum(float(bid[1]) for bid in bids[:levels] if len(bid) >= 2)
        ask_depth = sum(float(ask[1]) for ask in asks[:levels] if len(ask) >= 2)
        total_depth = bid_depth + ask_depth
        
        return {
            'bid_depth': float(bid_depth),
            'ask_depth': float(ask_depth), 
            'total_depth': float(total_depth)
        }
        
    except (IndexError, ValueError, TypeError):
        return {'bid_depth': 0.0, 'ask_depth': 0.0, 'total_depth': 0.0}

def estimate_price_impact(orderbook: dict, trade_size_usd: float, side: str = 'buy') -> float:
    """
    Estimate price impact for a given trade size
    Returns impact as percentage
    """
    if side not in ['buy', 'sell']:
        return 0.0
        
    book_side = 'asks' if side == 'buy' else 'bids'
    orders = orderbook.get(book_side, [])
    
    if not orders:
        return 0.0
        
    try:
        remaining_size = trade_size_usd
        total_cost = 0.0
        total_shares = 0.0
        
        for price_str, size_str in orders:
            price = float(price_str)
            size = float(size_str)
            
            order_value = price * size
            
            if remaining_size <= order_value:
                # Частичное исполнение этого ордера
                shares_needed = remaining_size / price
                total_cost += shares_needed * price
                total_shares += shares_needed
                break
            else:
                # Полное исполнение этого ордера
                total_cost += order_value
                total_shares += size
                remaining_size -= order_value
        
        if total_cost == 0 or total_shares == 0:
            return 0.0
            
        # Средняя цена исполнения (исправленная формула)
        avg_execution_price = total_cost / total_shares
        
        # Цена лучшего предложения
        best_price = float(orders[0][0])
        
        # Процент воздействия на цену
        impact = abs(avg_execution_price - best_price) / best_price * 100
        
        return float(min(impact, 10.0))  # Ограничиваем максимальным воздействием 10%
        
    except (IndexError, ValueError, TypeError, ZeroDivisionError):
        return 0.0
