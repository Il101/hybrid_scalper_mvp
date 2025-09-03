"""
Enhanced scalping strategy example using all new components
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional

from ingest.prices import get_ohlcv_cached
from features.ta_indicators import ta_score, atr_pct, microstructure_score
from features.microstructure import order_flow_imbalance, bid_ask_pressure
from features.news_metrics import news_score
from features.sm_metrics import sm_score
from ml.features_advanced import market_microstructure_score
from signals.ensemble import ComponentScores, combine_with_volatility_filter
from exec.timing import optimal_entry_timing, should_wait_for_better_fill, ExecutionTimer
from exec.fast_execution import FastExecutor, execute_immediately
from risk.advanced import (
    kelly_position_sizing, correlation_position_limit, dynamic_stop_loss,
    portfolio_heat, position_sizing_with_heat_limit, drawdown_based_sizing,
    volatility_adjusted_sizing
)
from exec.simulator import PaperBroker
from utils.kpis import sharpe_ratio, avg_trade_duration_minutes, implementation_shortfall_bps

class EnhancedScalpingBot:
    """
    Advanced scalping bot with all enhancements integrated
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.broker = PaperBroker(cash_usd=initial_capital, log_path="logs/enhanced_trades.csv")
        self.executor = FastExecutor()
        self.timer = ExecutionTimer()
        self.positions: List = []
        self.correlation_matrix = self._build_correlation_matrix()
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'avg_trade_duration_min': 0.0
        }
    
    def _build_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build correlation matrix for position sizing"""
        # Simplified correlation matrix - in real implementation, 
        # this would be calculated from historical data
        return {
            'BTCUSDT': {'ETHUSDT': 0.8, 'ADAUSDT': 0.6, 'SOLUSDT': 0.7},
            'ETHUSDT': {'BTCUSDT': 0.8, 'ADAUSDT': 0.5, 'SOLUSDT': 0.6},
            'ADAUSDT': {'BTCUSDT': 0.6, 'ETHUSDT': 0.5, 'SOLUSDT': 0.4},
            'SOLUSDT': {'BTCUSDT': 0.7, 'ETHUSDT': 0.6, 'ADAUSDT': 0.4}
        }
    
    async def analyze_opportunity(self, symbol: str, timeframe: str = '1m') -> Dict:
        """
        Comprehensive opportunity analysis
        """
        # Get market data with caching for speed
        df = get_ohlcv_cached(symbol, timeframe, ttl_seconds=10)
        
        if df is None or len(df) < 200:
            return {'valid': False, 'reason': 'insufficient_data'}
        
        # Traditional signals
        s_ta = ta_score(df)
        s_news = news_score(symbol)
        s_sm = sm_score(symbol)
        
        # Advanced microstructure analysis
        microstructure = market_microstructure_score(df)
        current_atr = atr_pct(df)
        
        # Enhanced ensemble with volatility filter
        gates = {
            'liquidity': True,
            'regime': True,
            'news_blackout': False,
            'microstructure': microstructure['scalping_favorability'] != 'unfavorable'
        }
        
        comp = ComponentScores(
            s_news=s_news,
            s_sm=s_sm,
            s_ta=(s_ta + microstructure['microstructure_score']) / 2,
            gates=gates
        )
        
        signal = combine_with_volatility_filter(
            comp, current_atr, vol_threshold=0.0008  # 0.08% minimum vol
        )
        
        # Execution timing analysis
        execution_type = optimal_entry_timing(
            signal_strength=signal.score,
            spread_bps=4.0,  # Estimated spread
            volatility=current_atr
        )
        
        return {
            'valid': signal.direction is not None,
            'signal': signal,
            'microstructure': microstructure,
            'atr_pct': current_atr,
            'execution_type': execution_type,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                              current_atr: float) -> float:
        """
        Advanced position sizing with multiple risk controls
        """
        # Base parameters from historical performance
        # These should be calculated from actual trading history
        historical_stats = {
            'win_rate': 0.58,
            'avg_win': 22.0,  # bps
            'avg_loss': 15.0,  # bps
            'long_term_vol': 0.015  # 1.5% daily vol
        }
        
        # Kelly criterion base sizing
        kelly_fraction = kelly_position_sizing(
            historical_stats['win_rate'],
            historical_stats['avg_win'],
            historical_stats['avg_loss']
        )
        
        base_risk_pct = kelly_fraction * 100  # Convert to percentage
        
        # Volatility adjustment
        vol_adjusted_risk = volatility_adjusted_sizing(
            base_risk_pct, current_atr * 252**0.5,  # Annualized vol
            historical_stats['long_term_vol'] * 252**0.5
        )
        
        # Drawdown adjustment
        current_equity = self.broker.cash_usd
        peak_equity = max(10000.0, current_equity)  # Track peak
        dd_adjusted_risk = drawdown_based_sizing(
            current_equity, peak_equity, vol_adjusted_risk
        )
        
        # Portfolio heat limit
        current_heat = portfolio_heat(self.positions, current_equity, {symbol: 50000})
        heat_adjusted_size_usd = position_sizing_with_heat_limit(
            current_equity * dd_adjusted_risk / 100,
            current_heat,
            max_heat=0.15  # 15% max heat
        )
        
        # Correlation limit
        correlation_limit = correlation_position_limit(
            self.positions, symbol, self.correlation_matrix
        )
        
        final_size = min(heat_adjusted_size_usd, current_equity * correlation_limit)
        
        return max(50.0, final_size)  # Minimum $50 position
    
    async def execute_trade(self, symbol: str, side: str, size_usd: float, 
                           price: float, execution_type: str = 'LIMIT') -> bool:
        """
        Execute trade with advanced timing and slippage handling
        """
        self.timer.record_signal()
        
        try:
            if execution_type == 'MARKET':
                # Ultra-fast market execution
                result = await execute_immediately(symbol, side, size_usd)
                if result.success:
                    self.timer.record_fill()
                    return True
                else:
                    self.timer.record_cancel()
                    return False
            
            else:
                # Limit order with intelligent timing
                # This would integrate with actual exchange API
                # For now, simulate with paper broker
                
                stop_price = self._calculate_stop_price(price, side, symbol)
                
                self.broker.open(
                    ts=datetime.now().isoformat(),
                    symbol=symbol,
                    side=side,
                    price=price,
                    size_usd=size_usd,
                    stop_price=stop_price,
                    reason=f"SCALP_{execution_type}"
                )
                
                self.timer.record_fill()
                return True
                
        except Exception as e:
            print(f"Execution failed: {e}")
            self.timer.record_cancel()
            return False
    
    def _calculate_stop_price(self, entry_price: float, side: str, symbol: str) -> float:
        """Calculate dynamic stop loss"""
        # Get recent data for ATR calculation
        df = get_ohlcv_cached(symbol, '1m', ttl_seconds=30)
        if df is None or len(df) < 20:
            # Fallback to simple percentage stop
            stop_distance = 0.008  # 0.8%
        else:
            atr = atr_pct(df)
            stop_distance = max(0.005, min(0.015, atr * 2.0))  # 2x ATR, capped
        
        if side == 'long':
            return entry_price * (1 - stop_distance)
        else:
            return entry_price * (1 + stop_distance)
    
    async def run_scalping_loop(self, symbols: List[str], max_positions: int = 3):
        """
        Main scalping loop
        """
        print("🚀 Starting Enhanced Scalping Bot")
        print(f"📊 Monitoring {len(symbols)} symbols")
        print(f"💰 Initial capital: ${self.broker.cash_usd:,.2f}")
        
        while True:
            try:
                # Check each symbol for opportunities
                for symbol in symbols:
                    # Skip if we already have max positions
                    if len(self.positions) >= max_positions:
                        continue
                    
                    # Skip if we already have position in this symbol
                    if any(pos.get('symbol') == symbol for pos in self.positions):
                        continue
                    
                    # Analyze opportunity
                    analysis = await self.analyze_opportunity(symbol)
                    
                    if not analysis['valid']:
                        continue
                    
                    signal = analysis['signal']
                    
                    # Strong signal threshold for scalping
                    if signal.score < 75:
                        continue
                    
                    print(f"\n📈 Signal: {symbol} {signal.direction} "
                          f"(score: {signal.score:.1f})")
                    
                    # Calculate position size
                    size_usd = self.calculate_position_size(
                        symbol, signal.score, analysis['atr_pct']
                    )
                    
                    # Get current price (mock)
                    current_price = 50000.0  # This should come from real market data
                    
                    # Execute trade
                    success = await self.execute_trade(
                        symbol, 
                        'buy' if signal.direction == 'long' else 'sell',
                        size_usd,
                        current_price,
                        analysis['execution_type']
                    )
                    
                    if success:
                        print(f"✅ Executed {signal.direction} {symbol} "
                              f"${size_usd:.0f} @ ${current_price:.2f}")
                        
                        # Track position
                        self.positions.append({
                            'symbol': symbol,
                            'side': signal.direction,
                            'size_usd': size_usd,
                            'entry_price': current_price,
                            'timestamp': time.time()
                        })
                    
                    # Brief pause between symbol checks
                    await asyncio.sleep(0.1)
                
                # Check for exit conditions on existing positions
                await self._check_exits()
                
                # Performance update
                if len(self.positions) > 0 or self.performance_stats['total_trades'] > 0:
                    self._update_performance_stats()
                
                # Main loop interval for scalping
                await asyncio.sleep(2.0)  # 2 second intervals
                
            except KeyboardInterrupt:
                print("\n⏹️  Shutting down bot...")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_exits(self):
        """Check exit conditions for open positions"""
        current_time = time.time()
        
        for i, position in enumerate(self.positions[:]):  # Copy list for safe iteration
            # Time-based exit (scalping should be quick)
            position_age_minutes = (current_time - position['timestamp']) / 60
            
            if position_age_minutes > 15:  # Max 15 minutes per position
                print(f"⏰ Time exit: {position['symbol']} after {position_age_minutes:.1f}min")
                self.positions.remove(position)
                continue
            
            # Price-based exits would go here
            # This would require real-time price monitoring
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        if self.broker.log_path:
            try:
                self.performance_stats['sharpe_ratio'] = sharpe_ratio(self.broker.log_path)
                self.performance_stats['avg_trade_duration_min'] = avg_trade_duration_minutes(self.broker.log_path)
            except Exception as e:
                print(f"Warning: Could not update performance stats: {e}")

async def main():
    """Run the enhanced scalping bot"""
    bot = EnhancedScalpingBot(initial_capital=10000.0)
    
    # Define symbols to trade
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    # Run the bot
    await bot.run_scalping_loop(symbols, max_positions=2)

if __name__ == "__main__":
    asyncio.run(main())
