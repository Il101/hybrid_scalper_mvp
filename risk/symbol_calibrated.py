"""
Symbol-calibrated risk management system.
Provides advanced risk models with per-symbol optimization and portfolio-level coordination.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
from features.symbol_specific import get_calibration_manager, SymbolCalibration

logger = logging.getLogger(__name__)

@dataclass
class PositionRisk:
    """Risk parameters for a single position"""
    symbol: str
    size_usd: float
    max_loss_usd: float
    stop_loss_price: Optional[float]
    confidence: float
    kelly_fraction: float
    correlation_risk: float  # Risk from correlations with other positions
    market_impact_bps: float

@dataclass 
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_exposure_usd: float
    net_exposure_usd: float
    max_portfolio_risk_usd: float
    correlation_adjusted_var: float  # Value at Risk
    concentration_risk: float
    active_positions: int
    available_buying_power: float

class SymbolRiskCalculator:
    """Advanced risk calculations for individual symbols"""
    
    @staticmethod
    def calculate_position_size(symbol: str, account_balance: float, 
                               signal_confidence: float, 
                               current_price: float,
                               stop_loss_price: Optional[float] = None) -> float:
        """Calculate optimal position size using symbol-specific parameters"""
        cal_manager = get_calibration_manager()
        calib = cal_manager.get_calibration(symbol)
        
        # Base position size from calibration
        base_size = calib.base_position_size
        
        # Adjust for account size
        account_factor = min(2.0, max(0.5, account_balance / 10000))  # Scale for account size
        adjusted_size = base_size * account_factor
        
        # Apply Kelly criterion if we have enough trade history
        if calib.total_trades > 20:
            win_rate = calib.win_rate
            avg_win = calib.avg_win_pct  
            avg_loss = calib.avg_loss_pct
            
            if avg_loss > 0:
                # Kelly formula: f = (bp - q) / b where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
                b = avg_win / avg_loss
                kelly_f = (b * win_rate - (1 - win_rate)) / b
                kelly_f = max(0.05, min(calib.kelly_fraction, kelly_f))  # Cap at max kelly
                
                adjusted_size *= kelly_f
                logger.debug(f"{symbol}: Kelly fraction {kelly_f:.3f}, size adjusted to ${adjusted_size:.0f}")
        
        # Confidence adjustment
        confidence_factor = 0.5 + 0.5 * signal_confidence  # Scale 0.5-1.0
        adjusted_size *= confidence_factor
        
        # Risk multiplier from calibration
        adjusted_size *= calib.risk_multiplier
        
        # Position size limits
        final_size = min(calib.max_position_size, max(calib.base_position_size * 0.5, adjusted_size))
        
        # Convert to shares if stop loss provided
        if stop_loss_price and current_price > 0:
            max_loss_pct = abs(current_price - stop_loss_price) / current_price
            if max_loss_pct > 0:
                # Risk-based sizing: don't risk more than X% of account on any trade
                max_account_risk = 0.02  # 2% max account risk per trade
                max_loss_usd = account_balance * max_account_risk
                max_shares = max_loss_usd / (max_loss_pct * current_price)
                max_size_from_risk = max_shares * current_price
                
                final_size = min(final_size, max_size_from_risk)
        
        return final_size
    
    @staticmethod
    def calculate_stop_loss(symbol: str, entry_price: float, 
                           position_side: str, atr_value: Optional[float] = None) -> float:
        """Calculate symbol-specific stop loss using calibrated parameters"""
        cal_manager = get_calibration_manager()
        calib = cal_manager.get_calibration(symbol)
        
        # Use ATR-based stops if available, otherwise use percentage
        if atr_value and atr_value > 0:
            # ATR multiplier based on symbol volatility and performance
            atr_multiplier = 1.5  # Default
            
            # Adjust based on win rate - lower win rate = tighter stops
            if calib.win_rate < 0.4:
                atr_multiplier = 1.2  # Tighter stops for poor performers
            elif calib.win_rate > 0.6:
                atr_multiplier = 2.0  # Wider stops for good performers
                
            stop_distance = atr_value * atr_multiplier
        else:
            # Percentage-based stops
            base_stop_pct = 0.015  # 1.5% default
            
            # Adjust based on symbol's historical performance
            if calib.avg_loss_pct > 0:
                # Use slightly tighter than average historical loss
                stop_pct = min(0.03, calib.avg_loss_pct * 0.8)  
            else:
                stop_pct = base_stop_pct
                
            stop_distance = entry_price * stop_pct
        
        # Calculate stop price
        if position_side.upper() == 'LONG':
            stop_price = entry_price - stop_distance
        else:  # SHORT
            stop_price = entry_price + stop_distance
            
        return stop_price
    
    @staticmethod
    def calculate_correlation_risk(symbol: str, active_positions: Dict[str, float]) -> float:
        """Estimate correlation risk with existing positions"""
        # Simplified correlation matrix (in production, use historical data)
        correlation_groups = {
            'BTC_GROUP': ['BTCUSDT:USDT', 'ETHUSDT:USDT'],
            'ALTCOIN_GROUP': ['SOLUSDT:USDT', 'ADAUSDT:USDT', 'DOGEUSDT:USDT'],
            'DEFI_GROUP': ['LINKUSDT:USDT', 'AVAXUSDT:USDT'],
        }
        
        correlation_risk = 0.0
        
        for group_name, group_symbols in correlation_groups.items():
            if symbol in group_symbols:
                # Calculate exposure to correlated assets
                correlated_exposure = sum(
                    abs(size) for sym, size in active_positions.items() 
                    if sym in group_symbols and sym != symbol
                )
                
                # Higher correlation = higher risk multiplier
                correlation_factor = 0.7 if group_name == 'BTC_GROUP' else 0.5
                correlation_risk = correlated_exposure * correlation_factor
                break
                
        return correlation_risk

class PortfolioRiskManager:
    """Portfolio-level risk management and coordination"""
    
    def __init__(self, max_portfolio_risk_pct: float = 0.15):
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.active_positions: Dict[str, PositionRisk] = {}
        
    def can_open_position(self, symbol: str, proposed_size_usd: float,
                         account_balance: float, signal_confidence: float) -> Tuple[bool, str]:
        """Check if new position can be opened within risk limits"""
        
        # Calculate current portfolio risk
        current_exposure = sum(pos.size_usd for pos in self.active_positions.values())
        max_portfolio_value = account_balance * self.max_portfolio_risk_pct
        
        # Check portfolio concentration
        if current_exposure + proposed_size_usd > max_portfolio_value:
            return False, f"Portfolio risk limit: ${current_exposure + proposed_size_usd:,.0f} > ${max_portfolio_value:,.0f}"
        
        # Check individual position limits
        cal_manager = get_calibration_manager()
        calib = cal_manager.get_calibration(symbol)
        
        if proposed_size_usd > calib.max_position_size:
            return False, f"Position size limit: ${proposed_size_usd:,.0f} > ${calib.max_position_size:,.0f}"
        
        # Check correlation limits
        correlation_risk = SymbolRiskCalculator.calculate_correlation_risk(
            symbol, {pos.symbol: pos.size_usd for pos in self.active_positions.values()}
        )
        
        max_correlated_exposure = account_balance * 0.25  # 25% max correlated exposure
        if correlation_risk > max_correlated_exposure:
            return False, f"Correlation risk limit: ${correlation_risk:,.0f} > ${max_correlated_exposure:,.0f}"
        
        # Check confidence threshold
        min_confidence = 0.6  # Minimum signal confidence
        if signal_confidence < min_confidence:
            return False, f"Signal confidence too low: {signal_confidence:.3f} < {min_confidence:.3f}"
            
        return True, "Position approved"
    
    def add_position(self, symbol: str, size_usd: float, max_loss_usd: float,
                    stop_loss_price: Optional[float] = None, confidence: float = 0.5) -> None:
        """Add position to portfolio tracking"""
        
        correlation_risk = SymbolRiskCalculator.calculate_correlation_risk(
            symbol, {pos.symbol: pos.size_usd for pos in self.active_positions.values()}
        )
        
        # Estimate market impact (simplified)
        market_impact_bps = min(10.0, size_usd / 10000)  # Rough approximation
        
        position_risk = PositionRisk(
            symbol=symbol,
            size_usd=size_usd,
            max_loss_usd=max_loss_usd,
            stop_loss_price=stop_loss_price,
            confidence=confidence,
            kelly_fraction=get_calibration_manager().get_calibration(symbol).kelly_fraction,
            correlation_risk=correlation_risk,
            market_impact_bps=market_impact_bps
        )
        
        self.active_positions[symbol] = position_risk
        logger.info(f"Added position: {symbol} ${size_usd:,.0f} (risk: ${max_loss_usd:,.0f})")
    
    def remove_position(self, symbol: str, realized_pnl_pct: float, was_winner: bool) -> None:
        """Remove position and update calibrations"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            
            # Update symbol calibration with trade result
            cal_manager = get_calibration_manager()
            confidence = self.active_positions.get(symbol, PositionRisk(symbol, 0, 0, None, 0.5, 0.25, 0, 0)).confidence
            cal_manager.update_performance(symbol, realized_pnl_pct, was_winner, confidence)
            
            logger.info(f"Removed position: {symbol} (PnL: {realized_pnl_pct:+.3f}%, Winner: {was_winner})")
    
    def get_portfolio_risk(self, account_balance: float) -> PortfolioRisk:
        """Calculate current portfolio risk metrics"""
        
        total_exposure = sum(pos.size_usd for pos in self.active_positions.values())
        total_risk = sum(pos.max_loss_usd for pos in self.active_positions.values())
        
        # Simple correlation-adjusted VaR (in production use proper covariance matrix)
        correlation_adjustment = 0.8  # Assume 80% correlation on average
        adjusted_var = total_risk * correlation_adjustment
        
        # Concentration risk (max single position / total portfolio)
        max_position = max([pos.size_usd for pos in self.active_positions.values()], default=0)
        concentration_risk = max_position / max(total_exposure, 1)
        
        return PortfolioRisk(
            total_exposure_usd=total_exposure,
            net_exposure_usd=total_exposure,  # Simplified - no short positions
            max_portfolio_risk_usd=total_risk,
            correlation_adjusted_var=adjusted_var,
            concentration_risk=concentration_risk,
            active_positions=len(self.active_positions),
            available_buying_power=max(0, account_balance * self.max_portfolio_risk_pct - total_exposure)
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get human-readable risk summary"""
        portfolio_risk = self.get_portfolio_risk(10000)  # Assume 10k account for summary
        
        return {
            'active_positions': portfolio_risk.active_positions,
            'total_exposure': f"${portfolio_risk.total_exposure_usd:,.0f}",
            'max_risk': f"${portfolio_risk.max_portfolio_risk_usd:,.0f}",
            'concentration': f"{portfolio_risk.concentration_risk:.1%}",
            'available_bp': f"${portfolio_risk.available_buying_power:,.0f}",
            'positions': {symbol: f"${pos.size_usd:,.0f}" for symbol, pos in self.active_positions.items()}
        }

# Global portfolio manager instance
_portfolio_manager: Optional[PortfolioRiskManager] = None

def get_portfolio_manager() -> PortfolioRiskManager:
    """Get global portfolio risk manager instance"""
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioRiskManager()
    return _portfolio_manager
