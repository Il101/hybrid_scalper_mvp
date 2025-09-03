"""
Advanced execution timing logic for scalping
"""
import time
from typing import Literal, Optional
import numpy as np

OrderType = Literal["MARKET", "LIMIT", "POST_ONLY"]

def optimal_entry_timing(signal_strength: float, spread_bps: float, 
                        volatility: float, urgency_factor: float = 1.0) -> OrderType:
    """
    Determine optimal order type based on market conditions
    
    Args:
        signal_strength: 0-100, higher = stronger signal
        spread_bps: Current bid-ask spread in basis points
        volatility: Recent volatility measure
        urgency_factor: 1.0 = patient, 2.0 = urgent
    
    Returns:
        Order type recommendation
    """
    # Strong signal + tight spread = go market
    if signal_strength > 80 and spread_bps < 3:
        return "MARKET"
    
    # Very wide spread = be patient
    if spread_bps > 8:
        return "POST_ONLY"
    
    # High volatility + strong signal = market
    if volatility > 0.005 and signal_strength > 70:
        return "MARKET"
    
    # High urgency = market
    if urgency_factor > 1.5:
        return "MARKET"
    
    # Default: try to get better fill
    return "LIMIT"

def should_wait_for_better_fill(current_spread_bps: float, 
                               avg_spread_5min_bps: float,
                               patience_factor: float = 1.5) -> bool:
    """
    Decide if current spread is too wide compared to recent average
    """
    if avg_spread_5min_bps <= 0:
        return False
    
    return current_spread_bps > avg_spread_5min_bps * patience_factor

def calculate_limit_price_offset(signal_strength: float, spread_bps: float, 
                               volatility: float, side: str) -> float:
    """
    Calculate optimal offset from best price for limit orders
    
    Args:
        signal_strength: 0-100
        spread_bps: Current spread
        volatility: Recent volatility
        side: 'long' or 'short'
    
    Returns:
        Offset in basis points from best price
    """
    # Base offset: try to get filled within spread
    base_offset_bps = spread_bps * 0.3  # 30% into the spread
    
    # Adjust based on signal strength
    strength_factor = (100 - signal_strength) / 100  # Weaker signal = more patient
    patience_offset = base_offset_bps * strength_factor
    
    # Volatility adjustment - more volatile = less patient
    vol_adjustment = min(spread_bps * 0.2, volatility * 1000)  # Cap at 20% of spread
    
    final_offset = patience_offset + vol_adjustment
    
    return float(max(0.1, min(spread_bps * 0.8, final_offset)))  # Cap between 0.1 and 80% of spread

def execution_urgency_score(time_since_signal_sec: float, 
                           signal_strength: float,
                           market_impact_bps: float) -> float:
    """
    Calculate urgency score that increases over time
    
    Returns:
        Urgency factor (1.0 = normal, 2.0 = very urgent)
    """
    # Time decay - signals get stale
    time_decay = np.exp(-time_since_signal_sec / 60.0)  # Half-life of 1 minute
    
    # Strong signals are more urgent
    signal_urgency = signal_strength / 100.0
    
    # High market impact reduces urgency (want better fills)
    impact_penalty = max(0.5, 1.0 - market_impact_bps / 20.0)
    
    urgency = 1.0 + (signal_urgency * time_decay * impact_penalty)
    
    return float(max(1.0, min(3.0, urgency)))

def should_cancel_and_rechase(order_age_sec: float, 
                             fill_ratio: float,
                             market_moved_bps: float,
                             max_age_sec: float = 30) -> bool:
    """
    Decide if unfilled order should be cancelled and replaced
    
    Args:
        order_age_sec: How long order has been open
        fill_ratio: Portion filled (0.0 to 1.0)
        market_moved_bps: How much price moved since order placed
        max_age_sec: Maximum age before forced cancellation
    
    Returns:
        True if should cancel and replace
    """
    # Force cancel old orders
    if order_age_sec > max_age_sec:
        return True
    
    # If partially filled and market moving away, chase
    if fill_ratio > 0 and fill_ratio < 0.8 and abs(market_moved_bps) > 2:
        return True
    
    # If no fills and market moved significantly, chase
    if fill_ratio == 0 and abs(market_moved_bps) > 5:
        return True
    
    return False

def adaptive_order_size(base_size_usd: float, 
                       volatility: float,
                       spread_bps: float,
                       recent_fill_rate: float) -> float:
    """
    Adjust order size based on current market conditions
    
    Args:
        base_size_usd: Desired position size
        volatility: Recent volatility measure
        spread_bps: Current spread
        recent_fill_rate: Recent fill success rate (0.0 to 1.0)
    
    Returns:
        Adjusted size in USD
    """
    # Reduce size in high volatility to reduce impact
    vol_adjustment = max(0.5, 1.0 - volatility * 100)
    
    # Reduce size if spreads are wide
    spread_adjustment = max(0.7, 1.0 - spread_bps / 20.0)
    
    # Increase size if getting good fills recently
    fill_adjustment = 0.8 + (recent_fill_rate * 0.4)  # 0.8 to 1.2 range
    
    adjustment_factor = vol_adjustment * spread_adjustment * fill_adjustment
    
    return float(base_size_usd * max(0.3, min(1.5, adjustment_factor)))

class ExecutionTimer:
    """Track timing statistics for execution optimization"""
    
    def __init__(self):
        self.fill_times = []
        self.cancel_times = []
        self.last_signal_time = None
    
    def record_signal(self):
        self.last_signal_time = time.time()
    
    def record_fill(self):
        if self.last_signal_time:
            fill_time = time.time() - self.last_signal_time
            self.fill_times.append(fill_time)
            # Keep only recent history
            self.fill_times = self.fill_times[-50:]
    
    def record_cancel(self):
        if self.last_signal_time:
            cancel_time = time.time() - self.last_signal_time
            self.cancel_times.append(cancel_time)
            self.cancel_times = self.cancel_times[-50:]
    
    def avg_fill_time_sec(self) -> float:
        return float(np.mean(self.fill_times)) if self.fill_times else 15.0
    
    def fill_rate(self) -> float:
        total_attempts = len(self.fill_times) + len(self.cancel_times)
        return len(self.fill_times) / max(1, total_attempts)
