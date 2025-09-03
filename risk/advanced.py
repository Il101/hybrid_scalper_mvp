"""
Advanced risk management for scalping
"""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    size_usd: float
    entry_price: float
    side: str  # 'long' or 'short'
    timestamp: float
    unrealized_pnl: float = 0.0

def kelly_position_sizing(win_rate: float, avg_win: float, avg_loss: float, 
                         max_kelly_frac: float = 0.25) -> float:
    """
    Calculate optimal position size using Kelly criterion
    
    Args:
        win_rate: Probability of winning trade (0.0 to 1.0)
        avg_win: Average winning trade amount (positive)
        avg_loss: Average losing trade amount (positive)
        max_kelly_frac: Maximum Kelly fraction to use (cap for safety)
    
    Returns:
        Optimal fraction of capital to risk (0.0 to max_kelly_frac)
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0
    
    b = avg_win / avg_loss  # Win/loss ratio
    p = win_rate  # Win probability
    
    kelly_f = (b * p - (1 - p)) / b
    
    # Apply safety cap
    optimal_f = max(0.0, min(max_kelly_frac, kelly_f))
    
    return float(optimal_f)

def correlation_position_limit(open_positions: List[Position], 
                              new_symbol: str,
                              correlation_matrix: Dict[str, Dict[str, float]],
                              max_correlated_exposure: float = 0.3) -> float:
    """
    Calculate position size limit based on correlation with existing positions
    
    Args:
        open_positions: List of current positions
        new_symbol: Symbol for new position
        correlation_matrix: Dict of symbol correlations
        max_correlated_exposure: Max total exposure to correlated assets
    
    Returns:
        Maximum allowed position size as fraction of capital
    """
    if not open_positions:
        return 1.0
    
    # Calculate current correlated exposure
    correlated_exposure = 0.0
    
    for position in open_positions:
        correlation = correlation_matrix.get(new_symbol, {}).get(position.symbol, 0.0)
        
        # Weight by correlation and position size
        if abs(correlation) > 0.3:  # Only consider significant correlations
            correlated_exposure += abs(position.size_usd * correlation)
    
    # Reduce available capital by correlated exposure
    remaining_capacity = max(0.0, max_correlated_exposure - correlated_exposure)
    
    return float(remaining_capacity)

def dynamic_stop_loss(entry_price: float, 
                     current_price: float,
                     atr: float,
                     side: str,
                     profit_multiple: float = 2.0,
                     min_stop_atr: float = 1.5,
                     max_stop_atr: float = 3.0) -> float:
    """
    Calculate dynamic trailing stop loss based on ATR and profit
    
    Args:
        entry_price: Original entry price
        current_price: Current market price
        atr: Average True Range
        side: 'long' or 'short'
        profit_multiple: ATR multiple to trail behind high/low
        min_stop_atr: Minimum stop distance in ATR units
        max_stop_atr: Maximum stop distance in ATR units
    
    Returns:
        Stop loss price
    """
    # Calculate current profit in ATR terms
    if side == 'long':
        profit_atr = (current_price - entry_price) / atr
    else:
        profit_atr = (entry_price - current_price) / atr
    
    # Determine stop distance based on profit
    if profit_atr < 0:
        # In loss - use maximum stop distance
        stop_distance_atr = max_stop_atr
    elif profit_atr < 1:
        # Small profit - use minimum stop distance
        stop_distance_atr = min_stop_atr
    else:
        # Larger profit - trail behind by profit multiple
        stop_distance_atr = min(max_stop_atr, max(min_stop_atr, profit_multiple))
    
    # Calculate stop price
    if side == 'long':
        stop_price = current_price - (stop_distance_atr * atr)
    else:
        stop_price = current_price + (stop_distance_atr * atr)
    
    return float(stop_price)

def portfolio_heat(positions: List[Position], 
                  total_equity: float,
                  current_prices: Dict[str, float]) -> float:
    """
    Calculate total portfolio heat (risk exposure)
    
    Args:
        positions: List of open positions
        total_equity: Total account equity
        current_prices: Current market prices
    
    Returns:
        Portfolio heat as fraction of equity
    """
    if not positions or total_equity <= 0:
        return 0.0
    
    total_risk = 0.0
    
    for position in positions:
        current_price = current_prices.get(position.symbol, position.entry_price)
        
        # Calculate current position value
        if position.side == 'long':
            position_value = (current_price / position.entry_price) * position.size_usd
            # Risk is from current price to potential loss
            risk = position_value  # Could lose entire position
        else:
            position_value = (position.entry_price / current_price) * position.size_usd
            risk = position_value  # Could lose entire position
        
        total_risk += risk
    
    return float(total_risk / total_equity)

def position_sizing_with_heat_limit(base_size_usd: float,
                                   current_heat: float,
                                   max_heat: float = 0.2,
                                   target_heat: float = 0.15) -> float:
    """
    Adjust position size to maintain portfolio heat within limits
    
    Args:
        base_size_usd: Desired position size
        current_heat: Current portfolio heat (0.0 to 1.0)
        max_heat: Maximum allowed heat
        target_heat: Target heat level
    
    Returns:
        Adjusted position size
    """
    if current_heat >= max_heat:
        return 0.0  # No new positions if at max heat
    
    # Scale down size if approaching heat limit
    heat_capacity = max_heat - current_heat
    heat_utilization = 1.0 - (current_heat / target_heat) if target_heat > 0 else 1.0
    
    size_multiplier = min(1.0, heat_capacity / max(0.01, target_heat * 0.5))
    
    return float(base_size_usd * max(0.1, size_multiplier))

def drawdown_based_sizing(current_equity: float,
                         peak_equity: float,
                         base_risk_pct: float,
                         max_dd_reduction: float = 0.5) -> float:
    """
    Reduce position sizing during drawdown periods
    
    Args:
        current_equity: Current account equity
        peak_equity: Historical peak equity
        base_risk_pct: Base risk percentage per trade
        max_dd_reduction: Maximum reduction in risk (0.0 to 1.0)
    
    Returns:
        Adjusted risk percentage
    """
    if peak_equity <= 0:
        return base_risk_pct
    
    current_dd = (peak_equity - current_equity) / peak_equity
    
    # Reduce sizing linearly with drawdown
    if current_dd <= 0:
        return base_risk_pct  # At new highs
    
    # Scale reduction with drawdown severity
    dd_factor = min(max_dd_reduction, current_dd * 2)  # 2x multiplier for sensitivity
    reduction = 1.0 - dd_factor
    
    return float(base_risk_pct * max(0.2, reduction))  # Never reduce below 20% of base

def volatility_adjusted_sizing(base_risk_pct: float,
                             current_volatility: float,
                             long_term_volatility: float,
                             vol_lookback_ratio: float = 1.5) -> float:
    """
    Adjust position size based on current volatility vs long-term average
    
    Args:
        base_risk_pct: Base risk percentage
        current_volatility: Recent volatility measure
        long_term_volatility: Long-term volatility average
        vol_lookback_ratio: Ratio threshold for adjustment
    
    Returns:
        Volatility-adjusted risk percentage
    """
    if long_term_volatility <= 0:
        return base_risk_pct
    
    vol_ratio = current_volatility / long_term_volatility
    
    # Reduce size when volatility is high
    if vol_ratio > vol_lookback_ratio:
        vol_adjustment = vol_lookback_ratio / vol_ratio
    else:
        # Slight increase when volatility is low (but capped)
        vol_adjustment = min(1.2, vol_ratio / vol_lookback_ratio + 0.8)
    
    return float(base_risk_pct * max(0.3, min(1.3, vol_adjustment)))

def intraday_time_based_sizing(current_hour_utc: int,
                              base_risk_pct: float,
                              high_activity_hours: Optional[List[int]] = None) -> float:
    """
    Adjust position sizing based on time of day
    
    Args:
        current_hour_utc: Current hour in UTC (0-23)
        base_risk_pct: Base risk percentage
        high_activity_hours: Hours with higher market activity
    
    Returns:
        Time-adjusted risk percentage
    """
    if high_activity_hours is None:
        # Default high activity hours (London + NY open)
        high_activity_hours = [8, 9, 10, 13, 14, 15, 16, 17]
    
    if current_hour_utc in high_activity_hours:
        return base_risk_pct  # Full size during active hours
    else:
        return base_risk_pct * 0.7  # Reduced size during quiet hours
