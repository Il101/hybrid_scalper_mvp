#!/usr/bin/env python3
"""
P2A Machine Learning Infrastructure - Feature Engineering Demo
Demonstrates advanced feature extraction for ML-enhanced trading system.
"""
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    """Single orderbook snapshot"""
    timestamp: float
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float

@dataclass
class MLFeatures:
    """Machine learning features for a single timestamp"""
    timestamp: float
    symbol: str
    
    # Microstructure features
    order_flow_imbalance: float      # Bid volume - Ask volume imbalance
    price_impact: float              # Expected price impact of market order
    tick_direction: int              # -1, 0, 1 for down, flat, up tick
    spread_volatility: float         # Recent spread volatility
    
    # Technical features
    rsi_1m: float                   # 1-minute RSI
    macd_signal: float              # MACD signal strength
    volume_profile: float           # VWAP deviation
    momentum_score: float           # Multi-timeframe momentum
    
    # Regime features  
    volatility_regime: str          # "low", "medium", "high"
    trend_strength: float           # ADX-based trend strength
    market_phase: str               # "trending", "ranging", "volatile"
    
    # Target variables (for training)
    price_change_5s: Optional[float] = None   # 5-second forward return
    price_change_30s: Optional[float] = None  # 30-second forward return
    direction_5s: Optional[int] = None        # 1 for up, -1 for down, 0 for flat

class AdvancedFeatureEngine:
    """Advanced feature engineering for ML models"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.orderbook_history: Dict[str, deque] = {}
        self.feature_history: Dict[str, deque] = {}
        
        # Feature computation windows
        self.rsi_window = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.volatility_window = 20
        
        logger.info(f"🔧 Initialized AdvancedFeatureEngine with {lookback_periods} lookback periods")
    
    def add_orderbook_snapshot(self, snapshot: OrderBookSnapshot) -> Optional[MLFeatures]:
        """Add new orderbook snapshot and compute features"""
        symbol = snapshot.symbol
        
        # Initialize history for new symbol
        if symbol not in self.orderbook_history:
            self.orderbook_history[symbol] = deque(maxlen=self.lookback_periods)
            self.feature_history[symbol] = deque(maxlen=self.lookback_periods)
        
        # Store snapshot
        self.orderbook_history[symbol].append(snapshot)
        
        # Need sufficient history for feature computation
        if len(self.orderbook_history[symbol]) < 20:
            return None
        
        # Compute features
        features = self._compute_features(symbol, snapshot)
        
        # Store features
        self.feature_history[symbol].append(features)
        
        return features
    
    def _compute_features(self, symbol: str, snapshot: OrderBookSnapshot) -> MLFeatures:
        """Compute all ML features for current snapshot"""
        history = list(self.orderbook_history[symbol])
        
        # Microstructure features
        order_flow_imbalance = self._compute_order_flow_imbalance(snapshot)
        price_impact = self._compute_price_impact(snapshot)
        tick_direction = self._compute_tick_direction(history[-10:])  # Last 10 ticks
        spread_volatility = self._compute_spread_volatility(history[-20:])  # Last 20 snapshots
        
        # Technical features
        prices = [h.mid_price for h in history]
        rsi_1m = self._compute_rsi(prices, self.rsi_window)
        macd_signal = self._compute_macd_signal(prices)
        volume_profile = self._compute_volume_profile(history[-30:])
        momentum_score = self._compute_momentum_score(prices)
        
        # Regime features
        volatility_regime = self._classify_volatility_regime(prices[-self.volatility_window:])
        trend_strength = self._compute_trend_strength(prices)
        market_phase = self._classify_market_phase(prices, volatility_regime, trend_strength)
        
        return MLFeatures(
            timestamp=snapshot.timestamp,
            symbol=symbol,
            order_flow_imbalance=order_flow_imbalance,
            price_impact=price_impact,
            tick_direction=tick_direction,
            spread_volatility=spread_volatility,
            rsi_1m=rsi_1m,
            macd_signal=macd_signal,
            volume_profile=volume_profile,
            momentum_score=momentum_score,
            volatility_regime=volatility_regime,
            trend_strength=trend_strength,
            market_phase=market_phase
        )
    
    def _compute_order_flow_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        """Compute order flow imbalance from top-of-book"""
        if not snapshot.bids or not snapshot.asks:
            return 0.0
        
        # Top 5 levels bid/ask volume
        bid_volume = sum(size for _, size in snapshot.bids[:5])
        ask_volume = sum(size for _, size in snapshot.asks[:5])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        # Normalize to [-1, 1] where 1 = all bids, -1 = all asks
        imbalance = (bid_volume - ask_volume) / total_volume
        return imbalance
    
    def _compute_price_impact(self, snapshot: OrderBookSnapshot) -> float:
        """Estimate price impact of a market order"""
        if not snapshot.bids or not snapshot.asks:
            return 0.0
        
        # Assume $10,000 market order, compute price impact
        order_size_usd = 10000
        
        # Buy impact (hitting asks)
        cumulative_value = 0
        weighted_price = 0
        
        for price, size in snapshot.asks:
            level_value = price * size
            if cumulative_value + level_value >= order_size_usd:
                # Partial fill on this level
                remaining = order_size_usd - cumulative_value
                remaining_size = remaining / price
                weighted_price += price * remaining_size
                break
            else:
                cumulative_value += level_value
                weighted_price += price * size
        
        if cumulative_value == 0:
            return 0.0
        
        # Price impact as basis points
        avg_fill_price = weighted_price / (order_size_usd / np.mean([p for p, s in snapshot.asks[:5]]))
        impact_bps = ((avg_fill_price - snapshot.best_ask) / snapshot.best_ask) * 10000
        
        return float(impact_bps)
    
    def _compute_tick_direction(self, recent_snapshots: List[OrderBookSnapshot]) -> int:
        """Compute tick direction based on recent price changes"""
        if len(recent_snapshots) < 2:
            return 0
        
        recent_prices = [s.mid_price for s in recent_snapshots[-3:]]
        
        if len(recent_prices) < 2:
            return 0
        
        price_change = recent_prices[-1] - recent_prices[-2]
        
        if price_change > 0:
            return 1
        elif price_change < 0:
            return -1
        else:
            return 0
    
    def _compute_spread_volatility(self, recent_snapshots: List[OrderBookSnapshot]) -> float:
        """Compute volatility of bid-ask spread"""
        if len(recent_snapshots) < 5:
            return 0.0
        
        spreads = [s.spread_bps for s in recent_snapshots]
        return float(np.std(spreads))
    
    def _compute_rsi(self, prices: List[float], window: int) -> float:
        """Compute RSI indicator"""
        if len(prices) < window + 1:
            return 50.0  # Neutral RSI
        
        price_changes = np.diff(prices[-window-1:])
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _compute_macd_signal(self, prices: List[float]) -> float:
        """Compute MACD signal strength"""
        if len(prices) < self.macd_slow:
            return 0.0
        
        # Exponential moving averages
        ema_fast = self._compute_ema(prices, self.macd_fast)
        ema_slow = self._compute_ema(prices, self.macd_slow)
        
        macd_line = ema_fast - ema_slow
        
        # Normalize by price level
        normalized_macd = (macd_line / prices[-1]) * 10000  # in basis points
        
        return float(normalized_macd)
    
    def _compute_ema(self, prices: List[float], window: int) -> float:
        """Compute exponential moving average"""
        if len(prices) < window:
            return float(np.mean(prices))
        
        alpha = 2.0 / (window + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _compute_volume_profile(self, recent_snapshots: List[OrderBookSnapshot]) -> float:
        """Compute VWAP deviation as volume profile indicator"""
        if len(recent_snapshots) < 5:
            return 0.0
        
        # Approximate VWAP using mid prices and spread as volume proxy
        total_volume = 0
        weighted_price = 0
        
        for snapshot in recent_snapshots:
            # Use inverse spread as volume proxy (tighter spread = more volume)
            volume_proxy = 1.0 / max(snapshot.spread_bps, 0.1)
            total_volume += volume_proxy
            weighted_price += snapshot.mid_price * volume_proxy
        
        if total_volume == 0:
            return 0.0
        
        vwap = weighted_price / total_volume
        current_price = recent_snapshots[-1].mid_price
        
        # VWAP deviation in basis points
        deviation_bps = ((current_price - vwap) / vwap) * 10000
        
        return float(deviation_bps)
    
    def _compute_momentum_score(self, prices: List[float]) -> float:
        """Compute multi-timeframe momentum score"""
        if len(prices) < 20:
            return 0.0
        
        # Different timeframe returns
        ret_5 = (prices[-1] / prices[-6] - 1) if len(prices) >= 6 else 0  # 5-period return
        ret_10 = (prices[-1] / prices[-11] - 1) if len(prices) >= 11 else 0  # 10-period return
        ret_20 = (prices[-1] / prices[-21] - 1) if len(prices) >= 21 else 0  # 20-period return
        
        # Weighted momentum score (recent periods weighted more)
        momentum = (3 * ret_5 + 2 * ret_10 + 1 * ret_20) / 6
        
        # Convert to basis points
        momentum_bps = momentum * 10000
        
        return float(momentum_bps)
    
    def _classify_volatility_regime(self, prices: List[float]) -> str:
        """Classify current volatility regime"""
        if len(prices) < 10:
            return "medium"
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(len(returns))  # Annualized volatility proxy
        
        # Classify based on thresholds (these would be calibrated)
        if volatility < 0.01:
            return "low"
        elif volatility > 0.03:
            return "high"
        else:
            return "medium"
    
    def _compute_trend_strength(self, prices: List[float]) -> float:
        """Compute ADX-like trend strength indicator"""
        if len(prices) < 14:
            return 0.0
        
        # Simplified ADX calculation
        price_changes = np.diff(prices[-14:])
        
        # Directional movement
        plus_dm = np.where(price_changes > 0, price_changes, 0)
        minus_dm = np.where(price_changes < 0, -price_changes, 0)
        
        # Average directional movement
        avg_plus_dm = np.mean(plus_dm)
        avg_minus_dm = np.mean(minus_dm)
        
        # Directional indicator
        total_dm = avg_plus_dm + avg_minus_dm
        if total_dm == 0:
            return 0.0
        
        dx = abs(avg_plus_dm - avg_minus_dm) / total_dm * 100
        
        return float(dx)
    
    def _classify_market_phase(self, prices: List[float], volatility_regime: str, trend_strength: float) -> str:
        """Classify current market phase"""
        if trend_strength > 25:  # Strong trend
            return "trending"
        elif volatility_regime == "high":
            return "volatile"
        else:
            return "ranging"
    
    def get_feature_summary(self, symbol: str) -> Dict:
        """Get summary statistics of computed features"""
        if symbol not in self.feature_history or not self.feature_history[symbol]:
            return {}
        
        features_list = list(self.feature_history[symbol])
        
        return {
            'total_features': len(features_list),
            'avg_order_flow_imbalance': np.mean([f.order_flow_imbalance for f in features_list]),
            'avg_price_impact': np.mean([f.price_impact for f in features_list]),
            'avg_rsi': np.mean([f.rsi_1m for f in features_list]),
            'avg_trend_strength': np.mean([f.trend_strength for f in features_list]),
            'volatility_regimes': {
                'low': len([f for f in features_list if f.volatility_regime == 'low']),
                'medium': len([f for f in features_list if f.volatility_regime == 'medium']),
                'high': len([f for f in features_list if f.volatility_regime == 'high'])
            },
            'market_phases': {
                'trending': len([f for f in features_list if f.market_phase == 'trending']),
                'ranging': len([f for f in features_list if f.market_phase == 'ranging']),
                'volatile': len([f for f in features_list if f.market_phase == 'volatile'])
            }
        }

def generate_demo_orderbook_data(symbol: str, num_snapshots: int = 50) -> List[OrderBookSnapshot]:
    """Generate realistic demo orderbook data"""
    snapshots = []
    base_price = 50000 if 'BTC' in symbol else 2500 if 'ETH' in symbol else 150
    
    current_price = base_price
    timestamp = time.time()
    
    for i in range(num_snapshots):
        # Random walk price
        price_change = np.random.normal(0, base_price * 0.0001)  # 0.01% volatility
        current_price += price_change
        
        # Generate orderbook levels
        spread_bps = np.random.uniform(0.5, 3.0)  # 0.5-3 bps spread
        spread = current_price * spread_bps / 10000
        
        best_bid = current_price - spread / 2
        best_ask = current_price + spread / 2
        
        # Generate 10 levels each side
        bids = []
        asks = []
        
        for level in range(10):
            bid_price = best_bid - level * spread * 0.1
            ask_price = best_ask + level * spread * 0.1
            
            # Random sizes
            bid_size = np.random.uniform(0.1, 2.0)
            ask_size = np.random.uniform(0.1, 2.0)
            
            bids.append((bid_price, bid_size))
            asks.append((ask_price, ask_size))
        
        snapshot = OrderBookSnapshot(
            timestamp=timestamp + i,
            symbol=symbol,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=current_price,
            spread_bps=spread_bps
        )
        
        snapshots.append(snapshot)
    
    return snapshots

def demo_feature_engineering():
    """Demonstrate advanced feature engineering"""
    print("🔬 P2A Machine Learning Feature Engineering Demo")
    print("=" * 60)
    
    # Initialize feature engine
    feature_engine = AdvancedFeatureEngine(lookback_periods=100)
    
    # Generate demo data for multiple symbols
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    print(f"\n📊 Generating demo orderbook data for {len(symbols)} symbols...")
    
    all_features = []
    
    for symbol in symbols:
        print(f"\n🔍 Processing {symbol}...")
        
        # Generate orderbook snapshots
        snapshots = generate_demo_orderbook_data(symbol, num_snapshots=60)
        
        symbol_features = []
        for i, snapshot in enumerate(snapshots):
            features = feature_engine.add_orderbook_snapshot(snapshot)
            
            if features:
                symbol_features.append(features)
                all_features.append(features)
                
                # Show progress for first few features
                if len(symbol_features) <= 3:
                    print(f"   📈 Feature {len(symbol_features)}: "
                          f"RSI={features.rsi_1m:.1f}, "
                          f"Imbalance={features.order_flow_imbalance:.3f}, "
                          f"Regime={features.volatility_regime}, "
                          f"Phase={features.market_phase}")
        
        # Show summary for this symbol
        summary = feature_engine.get_feature_summary(symbol)
        print(f"   ✅ {symbol}: {summary['total_features']} features computed")
        print(f"      📊 Avg RSI: {summary['avg_rsi']:.1f}")
        print(f"      ⚖️ Avg Order Flow Imbalance: {summary['avg_order_flow_imbalance']:.3f}")
        print(f"      💥 Avg Price Impact: {summary['avg_price_impact']:.2f} bps")
        print(f"      📈 Avg Trend Strength: {summary['avg_trend_strength']:.1f}")
        print(f"      🌊 Volatility Regimes: {summary['volatility_regimes']}")
        print(f"      📊 Market Phases: {summary['market_phases']}")
    
    print(f"\n🎯 Feature Engineering Summary:")
    print(f"   📊 Total features generated: {len(all_features)}")
    print(f"   🔬 Feature dimensions per sample: 12")
    print(f"   ⏱️ Time span: {len(all_features)} snapshots")
    
    # Feature statistics across all symbols
    if all_features:
        print(f"\n📈 Cross-Symbol Feature Analysis:")
        
        rsi_values = [f.rsi_1m for f in all_features]
        print(f"   📊 RSI distribution: min={min(rsi_values):.1f}, max={max(rsi_values):.1f}, avg={np.mean(rsi_values):.1f}")
        
        imbalance_values = [f.order_flow_imbalance for f in all_features]
        print(f"   ⚖️ Order Flow Imbalance: min={min(imbalance_values):.3f}, max={max(imbalance_values):.3f}, avg={np.mean(imbalance_values):.3f}")
        
        impact_values = [f.price_impact for f in all_features]
        print(f"   💥 Price Impact: min={min(impact_values):.2f}, max={max(impact_values):.2f}, avg={np.mean(impact_values):.2f} bps")
        
        # Regime distribution
        regime_counts = {}
        phase_counts = {}
        
        for f in all_features:
            regime_counts[f.volatility_regime] = regime_counts.get(f.volatility_regime, 0) + 1
            phase_counts[f.market_phase] = phase_counts.get(f.market_phase, 0) + 1
        
        print(f"   🌊 Volatility Regimes: {regime_counts}")
        print(f"   📊 Market Phases: {phase_counts}")
    
    print(f"\n🚀 P2A Feature Engineering Complete!")
    print(f"✅ Ready for ML model training with:")
    print(f"   • Microstructure features (order flow, price impact)")
    print(f"   • Technical indicators (RSI, MACD, momentum)")
    print(f"   • Market regime classification")
    print(f"   • Multi-timeframe analysis")
    print(f"\n🎯 Next: P2B - ML Model Training & Prediction")

if __name__ == "__main__":
    demo_feature_engineering()
