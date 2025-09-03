"""
Symbol-specific feature engineering and calibration system.
Provides individual parameter optimization for each trading symbol.
"""
from __future__ import annotations
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class SymbolCalibration:
    """Per-symbol calibration parameters"""
    symbol: str
    
    # Priority scoring weights (customized per symbol)
    vol_weight: float = 1.0
    flow_weight: float = 1.0  
    info_weight: float = 1.0
    cost_weight: float = 1.0
    
    # Dynamic thresholds
    max_spread_bps: float = 15.0
    min_depth_usd: float = 25000.0
    min_atr_pct: float = 0.001
    
    # Risk parameters
    base_position_size: float = 1000.0
    max_position_size: float = 5000.0
    kelly_fraction: float = 0.25
    risk_multiplier: float = 1.0
    
    # Technical analysis periods (adaptive)
    atr_period: int = 14
    ema_short: int = 50
    ema_long: int = 200
    rsi_period: int = 14
    
    # Performance tracking
    win_rate: float = 0.5
    avg_win_pct: float = 0.15
    avg_loss_pct: float = 0.10
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    
    # Calibration metadata
    last_updated: float = 0.0
    confidence_score: float = 0.5  # 0-1 scale
    calibration_sample_size: int = 0

class SymbolCalibrationManager:
    """Manages symbol-specific calibrations and adaptations"""
    
    def __init__(self, config_path: str = "config/symbol_calibrations.yaml"):
        self.config_path = config_path
        self.calibrations: Dict[str, SymbolCalibration] = {}
        self.load_calibrations()
        
    def load_calibrations(self) -> None:
        """Load symbol calibrations from YAML file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    data = yaml.safe_load(f) or {}
                    
                for symbol, params in data.items():
                    self.calibrations[symbol] = SymbolCalibration(symbol=symbol, **params)
                    
                logger.info(f"Loaded calibrations for {len(self.calibrations)} symbols")
            else:
                logger.info("No existing calibrations found, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load calibrations: {e}")
            
    def save_calibrations(self) -> None:
        """Save calibrations to YAML file"""
        try:
            # Ensure config directory exists
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for symbol, calib in self.calibrations.items():
                data[symbol] = {
                    'vol_weight': calib.vol_weight,
                    'flow_weight': calib.flow_weight,
                    'info_weight': calib.info_weight,
                    'cost_weight': calib.cost_weight,
                    'max_spread_bps': calib.max_spread_bps,
                    'min_depth_usd': calib.min_depth_usd,
                    'min_atr_pct': calib.min_atr_pct,
                    'base_position_size': calib.base_position_size,
                    'max_position_size': calib.max_position_size,
                    'kelly_fraction': calib.kelly_fraction,
                    'risk_multiplier': calib.risk_multiplier,
                    'atr_period': calib.atr_period,
                    'ema_short': calib.ema_short,
                    'ema_long': calib.ema_long,
                    'rsi_period': calib.rsi_period,
                    'win_rate': calib.win_rate,
                    'avg_win_pct': calib.avg_win_pct,
                    'avg_loss_pct': calib.avg_loss_pct,
                    'sharpe_ratio': calib.sharpe_ratio,
                    'max_drawdown': calib.max_drawdown,
                    'total_trades': calib.total_trades,
                    'last_updated': calib.last_updated,
                    'confidence_score': calib.confidence_score,
                    'calibration_sample_size': calib.calibration_sample_size,
                }
                
            with open(self.config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
                
            logger.info(f"Saved calibrations for {len(self.calibrations)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to save calibrations: {e}")
    
    def get_calibration(self, symbol: str) -> SymbolCalibration:
        """Get calibration for symbol, create default if not exists"""
        if symbol not in self.calibrations:
            self.calibrations[symbol] = SymbolCalibration(symbol=symbol)
            logger.info(f"Created default calibration for {symbol}")
            
        return self.calibrations[symbol]
    
    def update_performance(self, symbol: str, trade_pnl_pct: float, 
                          was_win: bool, confidence: Optional[float] = None) -> None:
        """Update symbol performance metrics from trade results"""
        calib = self.get_calibration(symbol)
        
        # Update trade statistics
        calib.total_trades += 1
        
        # Exponential moving averages for win rate and P&L
        alpha = 0.1  # Smoothing factor
        if was_win:
            calib.win_rate = (1 - alpha) * calib.win_rate + alpha * 1.0
            calib.avg_win_pct = (1 - alpha) * calib.avg_win_pct + alpha * abs(trade_pnl_pct)
        else:
            calib.win_rate = (1 - alpha) * calib.win_rate + alpha * 0.0
            calib.avg_loss_pct = (1 - alpha) * calib.avg_loss_pct + alpha * abs(trade_pnl_pct)
        
        # Update confidence if provided
        if confidence is not None:
            calib.confidence_score = (1 - alpha) * calib.confidence_score + alpha * confidence
            
        calib.last_updated = time.time()
        calib.calibration_sample_size += 1
        
        # Auto-adjust parameters based on performance
        self._adapt_parameters(symbol)
        
        logger.debug(f"Updated {symbol} performance: WR={calib.win_rate:.3f}, "
                    f"AvgWin={calib.avg_win_pct:.3f}, Conf={calib.confidence_score:.3f}")
    
    def _adapt_parameters(self, symbol: str) -> None:
        """Automatically adapt parameters based on performance"""
        calib = self.get_calibration(symbol)
        
        # Only adapt after sufficient sample size
        if calib.total_trades < 10:
            return
            
        # Adjust risk multiplier based on win rate
        if calib.win_rate > 0.65:
            # High win rate -> increase risk slightly
            calib.risk_multiplier = min(1.5, calib.risk_multiplier * 1.02)
        elif calib.win_rate < 0.45:
            # Low win rate -> decrease risk
            calib.risk_multiplier = max(0.5, calib.risk_multiplier * 0.98)
            
        # Adjust spread tolerance based on recent performance
        if calib.confidence_score > 0.7:
            # High confidence -> can be more selective (tighter spreads)
            calib.max_spread_bps = max(5.0, calib.max_spread_bps * 0.99)
        elif calib.confidence_score < 0.3:
            # Low confidence -> be less selective (wider spreads)
            calib.max_spread_bps = min(30.0, calib.max_spread_bps * 1.01)
            
        # Adjust position sizing based on risk-reward
        expected_value = calib.win_rate * calib.avg_win_pct - (1 - calib.win_rate) * calib.avg_loss_pct
        if expected_value > 0.05:  # Good EV
            calib.base_position_size = min(calib.max_position_size, 
                                         calib.base_position_size * 1.01)
        elif expected_value < 0.01:  # Poor EV
            calib.base_position_size = max(500.0, calib.base_position_size * 0.99)
    
    def compute_symbol_priority(self, symbol: str, base_vol_score: float,
                               base_flow_score: float, base_info_score: float,
                               base_cost_score: float) -> float:
        """Compute symbol priority using calibrated weights"""
        calib = self.get_calibration(symbol)
        
        priority = (base_vol_score * calib.vol_weight +
                   base_flow_score * calib.flow_weight +
                   base_info_score * calib.info_weight -
                   base_cost_score * calib.cost_weight)
        
        # Apply confidence-based adjustment
        confidence_multiplier = 0.5 + 0.5 * calib.confidence_score
        priority *= confidence_multiplier
        
        return priority
    
    def get_symbol_thresholds(self, symbol: str) -> Dict[str, float]:
        """Get symbol-specific filtering thresholds"""
        calib = self.get_calibration(symbol)
        
        return {
            'max_spread_bps': calib.max_spread_bps,
            'min_depth_usd': calib.min_depth_usd,
            'min_atr_pct': calib.min_atr_pct,
        }
    
    def get_position_size(self, symbol: str, base_size: float = 1000.0) -> float:
        """Get symbol-specific position size"""
        calib = self.get_calibration(symbol)
        
        # Scale by symbol-specific parameters
        size = base_size * calib.risk_multiplier
        
        # Apply Kelly fraction if we have enough data
        if calib.total_trades > 20:
            win_rate = calib.win_rate
            avg_win = calib.avg_win_pct
            avg_loss = calib.avg_loss_pct
            
            if avg_loss > 0:
                kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
                kelly_f = max(0.1, min(calib.kelly_fraction, kelly_f))  # Cap Kelly
                size *= kelly_f
        
        return min(calib.max_position_size, max(calib.base_position_size, size))
    
    def get_calibration_summary(self) -> Dict[str, Dict]:
        """Get summary of all calibrations for monitoring"""
        summary = {}
        
        for symbol, calib in self.calibrations.items():
            summary[symbol] = {
                'trades': calib.total_trades,
                'win_rate': f"{calib.win_rate:.3f}",
                'avg_win': f"{calib.avg_win_pct:.3f}",
                'avg_loss': f"{calib.avg_loss_pct:.3f}",
                'confidence': f"{calib.confidence_score:.3f}",
                'risk_mult': f"{calib.risk_multiplier:.3f}",
                'max_spread': f"{calib.max_spread_bps:.1f}",
                'position_size': f"{calib.base_position_size:.0f}",
                'last_updated': time.strftime("%Y-%m-%d %H:%M:%S", 
                                            time.localtime(calib.last_updated)) if calib.last_updated > 0 else "Never"
            }
            
        return summary

# Global manager instance
_calibration_manager: Optional[SymbolCalibrationManager] = None

def get_calibration_manager() -> SymbolCalibrationManager:
    """Get global calibration manager instance"""
    global _calibration_manager
    if _calibration_manager is None:
        _calibration_manager = SymbolCalibrationManager()
    return _calibration_manager
