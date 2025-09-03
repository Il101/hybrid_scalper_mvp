from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os, csv

@dataclass
class Position:
    side: Optional[str] = None            # "long"|"short"|None
    size_usd: float = 0.0
    avg_price: float = 0.0
    stop_price: float = 0.0
    bars_in_pos: int = 0
    timestamp: Optional[str] = None       # For test compatibility
    symbol: Optional[str] = None          # For test compatibility
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized PnL (requires current price)"""
        # This is a placeholder - in real usage you'd pass current price
        return 0.0

@dataclass
class PaperBroker:
    fees_bps_round: float = 6.0           # round-trip bps (пример: 6 bps)
    cash_usd: float = 10000.0
    position: Position = field(default_factory=Position)
    log_path: str = "logs/trades.csv"

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "ts","symbol","side","action","price","size_usd","fees_usd","pnl_usd","equity","reason"
                ])

    def _fee_usd_per_side(self, size_usd: float) -> float:
        # половина от round-trip на одну сторону
        return float(size_usd * (self.fees_bps_round / 10000.0) / 2.0)

    def _log(self, row):
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def equity_mark(self, price: float) -> float:
        if self.position.side is None or self.position.size_usd == 0 or self.position.avg_price <= 0:
            return self.cash_usd
        mult = 1.0 if self.position.side == "long" else -1.0
        pnl = (price - self.position.avg_price) / self.position.avg_price * self.position.size_usd * mult
        return self.cash_usd + pnl

    def flat(self) -> bool:
        return self.position.side is None or self.position.size_usd <= 0

    def open(self, ts: str, symbol: str, side: str, price: float, size_usd: float, stop_price: float, reason: str = "OPEN"):
        fee = self._fee_usd_per_side(size_usd)
        self.cash_usd -= fee
        self.position = Position(side=side, size_usd=size_usd, avg_price=price, stop_price=stop_price, bars_in_pos=0)
        self._log([ts, symbol, side, "OPEN", price, size_usd, fee, 0.0, self.cash_usd, reason])

    def close(self, ts: str, symbol: str, price: float, reason: str = "CLOSE"):
        if self.flat(): return 0.0
        mult = 1.0 if self.position.side == "long" else -1.0
        pnl = (price - self.position.avg_price) / self.position.avg_price * self.position.size_usd * mult
        fee = self._fee_usd_per_side(self.position.size_usd)
        self.cash_usd += pnl - fee
        self._log([ts, symbol, self.position.side, "CLOSE", price, self.position.size_usd, fee, pnl, self.cash_usd, reason])
        self.position = Position()
        return pnl - fee

    def partial_close(self, ts: str, symbol: str, price: float, frac: float, reason: str = "PARTIAL"):
        if self.flat() or frac <= 0.0: return 0.0
        close_usd = self.position.size_usd * min(1.0, frac)
        mult = 1.0 if self.position.side == "long" else -1.0
        pnl = (price - self.position.avg_price) / self.position.avg_price * close_usd * mult
        fee = self._fee_usd_per_side(close_usd)
        self.cash_usd += pnl - fee
        self.position.size_usd -= close_usd
        self._log([ts, symbol, self.position.side, "PARTIAL", price, close_usd, fee, pnl, self.cash_usd, reason])
        if self.position.size_usd <= 1e-6:
            self.position = Position()
        return pnl - fee

    def flip(self, ts: str, symbol: str, new_side: str, price: float, size_usd: float, stop_price: float, reason: str = "FLIP"):
        # close existing (market)
        self.close(ts, symbol, price, reason="FLIP_CLOSE")
        # open new
        self.open(ts, symbol, new_side, price, size_usd, stop_price, reason="FLIP_OPEN")

    def _simulate_latency_ms(self) -> int:
        """Simulate execution latency"""
        import random
        return random.randint(5, 50)  # 5-50ms latency

    def _dynamic_slippage(self, size_usd: float, recent_volatility: float) -> float:
        """Calculate dynamic slippage based on size and volatility"""
        base_slip_bps = size_usd * 0.0003  # 3bps base
        vol_penalty_bps = recent_volatility * 1000  # volatility spike penalty
        return min(base_slip_bps + vol_penalty_bps, size_usd * 0.002)  # Cap at 20bps

    def enhanced_open(self, ts: str, symbol: str, side: str, price: float, 
                     size_usd: float, stop_price: float, volatility: float = 0.001,
                     decision_price: Optional[float] = None, reason: str = "OPEN"):
        """Enhanced open with latency simulation and dynamic slippage"""
        import time
        
        # Simulate latency
        latency_ms = self._simulate_latency_ms()
        time.sleep(latency_ms / 1000.0)  # Simulate in real-time for testing
        
        # Apply dynamic slippage
        slippage_bps = self._dynamic_slippage(size_usd, volatility)
        slippage_factor = slippage_bps / 10000.0
        
        if side == "long":
            adjusted_price = price * (1 + slippage_factor)
        else:
            adjusted_price = price * (1 - slippage_factor)
        
        # Calculate fees with slippage
        fee = self._fee_usd_per_side(size_usd)
        self.cash_usd -= fee
        
        self.position = Position(
            side=side, 
            size_usd=size_usd, 
            avg_price=adjusted_price, 
            stop_price=stop_price, 
            bars_in_pos=0
        )
        
        # Log with additional fields
        log_row = [ts, symbol, side, "OPEN", adjusted_price, size_usd, fee, 0.0, 
                  self.cash_usd, reason]
        
        # Add decision price if provided (for implementation shortfall tracking)
        if decision_price is not None:
            log_row.extend([decision_price, adjusted_price])
        
        self._log(log_row)


class Broker(PaperBroker):
    """Base broker class for trading operations - inherits from PaperBroker"""
    def __init__(self, initial_cash_usd: float = 10000.0, fees_bps_round: float = 6.0):
        super().__init__(fees_bps_round=fees_bps_round, cash_usd=initial_cash_usd)
        self.initial_capital = initial_cash_usd
    
    def get_equity(self, current_price: float) -> float:
        """Get current equity including unrealized PnL"""
        return self.equity_mark(current_price)
    
    def is_flat(self) -> bool:
        """Check if position is flat"""
        return self.flat()
    
    def open_position(self, side: str, price: float, size_usd: float, stop_price: float = 0.0) -> bool:
        """Open a new position - compatibility wrapper"""
        if not self.flat():
            return False  # Already have a position
            
        # Use PaperBroker's open method
        from datetime import datetime
        ts = datetime.now().isoformat()
        try:
            self.open(ts, "BTCUSDT", side, price, size_usd, stop_price, "API_OPEN")
            return True
        except Exception:
            return False
    
    def close_position(self, price: float) -> float:
        """Close current position and return PnL - compatibility wrapper"""
        if self.flat():
            return 0.0
            
        # Calculate expected PnL before closing
        mult = 1.0 if self.position.side == "long" else -1.0
        pnl = (price - self.position.avg_price) / self.position.avg_price * self.position.size_usd * mult
        fee = self._fee_usd_per_side(self.position.size_usd)
        expected_net_pnl = pnl - fee
        
        # Use PaperBroker's close method
        from datetime import datetime
        ts = datetime.now().isoformat()
        try:
            self.close(ts, "BTCUSDT", price, "API_CLOSE")
            return expected_net_pnl
        except Exception:
            return 0.0
