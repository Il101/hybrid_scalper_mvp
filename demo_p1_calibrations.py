#!/usr/bin/env python3
"""
P1.1 Demo - Symbol-Specific Calibrations and Risk Management
Demonstrates per-symbol parameter optimization and advanced risk models.
"""
import sys
import time
import logging
from features.symbol_specific import get_calibration_manager
from risk.symbol_calibrated import get_portfolio_manager, SymbolRiskCalculator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_symbol_calibrations():
    """Demo symbol-specific calibration system"""
    print("🎯 P1.1.1 - Symbol Calibration Demo")
    print("=" * 50)
    
    cal_manager = get_calibration_manager()
    
    # Test symbols
    symbols = ['BTCUSDT:USDT', 'ETHUSDT:USDT', 'SOLUSDT:USDT']
    
    print("📊 Initial Calibrations:")
    for symbol in symbols:
        calib = cal_manager.get_calibration(symbol)
        print(f"  {symbol}:")
        print(f"    Risk Multiplier: {calib.risk_multiplier:.2f}")
        print(f"    Max Spread: {calib.max_spread_bps:.1f} bps")
        print(f"    Position Size: ${calib.base_position_size:,.0f}")
        print(f"    Kelly Fraction: {calib.kelly_fraction:.3f}")
        print()
    
    # Simulate some trade results for calibration
    print("🔄 Simulating trade results for calibration...")
    trade_results = [
        ('BTCUSDT:USDT', 0.025, True, 0.8),   # 2.5% win, high confidence
        ('BTCUSDT:USDT', -0.015, False, 0.7), # 1.5% loss, good confidence
        ('BTCUSDT:USDT', 0.031, True, 0.9),   # 3.1% win, very high confidence
        ('ETHUSDT:USDT', -0.022, False, 0.5), # 2.2% loss, medium confidence  
        ('ETHUSDT:USDT', 0.018, True, 0.6),   # 1.8% win, medium confidence
        ('SOLUSDT:USDT', 0.045, True, 0.7),   # 4.5% win, good confidence
        ('SOLUSDT:USDT', -0.031, False, 0.4), # 3.1% loss, low confidence
    ]
    
    for symbol, pnl_pct, was_win, confidence in trade_results:
        cal_manager.update_performance(symbol, pnl_pct, was_win, confidence)
        print(f"  📈 {symbol}: {'WIN' if was_win else 'LOSS'} {pnl_pct:+.1%} (conf: {confidence:.1f})")
    
    print("\n📊 Updated Calibrations After Trading:")
    for symbol in symbols:
        calib = cal_manager.get_calibration(symbol)
        print(f"  {symbol}:")
        print(f"    Win Rate: {calib.win_rate:.1%}")
        print(f"    Avg Win: {calib.avg_win_pct:.1%}")  
        print(f"    Avg Loss: {calib.avg_loss_pct:.1%}")
        print(f"    Confidence: {calib.confidence_score:.3f}")
        print(f"    Risk Mult: {calib.risk_multiplier:.3f}")
        print(f"    Max Spread: {calib.max_spread_bps:.1f} bps")
        print()
    
    # Show calibration summary
    print("📋 Calibration Summary:")
    summary = cal_manager.get_calibration_summary()
    for symbol, stats in summary.items():
        print(f"  {symbol}: {stats['trades']} trades, WR={stats['win_rate']}, Conf={stats['confidence']}")
    
    return cal_manager

def demo_risk_management(cal_manager):
    """Demo advanced risk management system"""
    print("\n🛡️ P1.1.2 - Advanced Risk Management Demo")  
    print("=" * 50)
    
    portfolio_manager = get_portfolio_manager()
    account_balance = 10000.0
    
    # Test position sizing
    print("💰 Position Sizing Tests:")
    test_signals = [
        ('BTCUSDT:USDT', 0.8, 58000.0, 56500.0),  # High confidence BTC long
        ('ETHUSDT:USDT', 0.6, 2800.0, 2720.0),    # Medium confidence ETH long  
        ('SOLUSDT:USDT', 0.7, 140.0, 136.0),      # Good confidence SOL long
    ]
    
    for symbol, confidence, price, stop_price in test_signals:
        # Calculate optimal position size
        position_size = SymbolRiskCalculator.calculate_position_size(
            symbol, account_balance, confidence, price, stop_price
        )
        
        # Calculate stop loss
        calculated_stop = SymbolRiskCalculator.calculate_stop_loss(
            symbol, price, 'LONG', atr_value=price * 0.02  # 2% ATR estimate
        )
        
        print(f"  {symbol}:")
        print(f"    Signal Confidence: {confidence:.1%}")
        print(f"    Current Price: ${price:,.0f}")
        print(f"    Suggested Stop: ${calculated_stop:,.0f}")
        print(f"    Position Size: ${position_size:,.0f}")
        print(f"    Risk Amount: ${position_size * abs(price - calculated_stop) / price:,.0f}")
        print()
    
    # Test portfolio risk limits
    print("🎛️ Portfolio Risk Management Tests:")
    
    proposed_positions = [
        ('BTCUSDT:USDT', 3000.0, 0.8),
        ('ETHUSDT:USDT', 2000.0, 0.6), 
        ('SOLUSDT:USDT', 1500.0, 0.7),
        ('ADAUSDT:USDT', 1200.0, 0.5),  # This should trigger limits
    ]
    
    for symbol, size, confidence in proposed_positions:
        can_open, reason = portfolio_manager.can_open_position(
            symbol, size, account_balance, confidence
        )
        
        print(f"  {symbol} (${size:,.0f}):")
        print(f"    Status: {'✅ APPROVED' if can_open else '❌ REJECTED'}")
        print(f"    Reason: {reason}")
        
        if can_open:
            # Add position to portfolio
            max_loss = size * 0.02  # 2% max loss estimate
            portfolio_manager.add_position(symbol, size, max_loss, confidence=confidence)
        print()
    
    # Show portfolio risk summary
    print("📊 Portfolio Risk Summary:")
    risk_summary = portfolio_manager.get_risk_summary()
    for key, value in risk_summary.items():
        if key == 'positions':
            print(f"  {key}:")
            for pos_symbol, pos_size in value.items():
                print(f"    {pos_symbol}: {pos_size}")
        else:
            print(f"  {key}: {value}")
    print()
    
    return portfolio_manager

def demo_correlation_analysis():
    """Demo correlation risk analysis"""
    print("🔗 P1.1.3 - Correlation Risk Analysis")
    print("=" * 40)
    
    # Simulate active positions
    active_positions = {
        'BTCUSDT:USDT': 3000.0,
        'ETHUSDT:USDT': 2000.0,
    }
    
    # Test correlation risk for new positions
    test_symbols = ['SOLUSDT:USDT', 'ADAUSDT:USDT', 'LINKUSDT:USDT']
    
    for symbol in test_symbols:
        corr_risk = SymbolRiskCalculator.calculate_correlation_risk(symbol, active_positions)
        print(f"  {symbol}:")
        print(f"    Correlation Risk: ${corr_risk:,.0f}")
        print(f"    Risk Level: {'HIGH' if corr_risk > 2000 else 'MEDIUM' if corr_risk > 1000 else 'LOW'}")
        print()

def main():
    """Run P1.1 demonstration"""
    print("🚀 P1 Phase Demo - Symbol Calibration & Risk Management")
    print("=" * 60)
    print("Demonstrating advanced quality improvements for trading system")
    print()
    
    try:
        # Demo 1: Symbol calibrations
        cal_manager = demo_symbol_calibrations()
        
        # Demo 2: Risk management
        portfolio_manager = demo_risk_management(cal_manager)
        
        # Demo 3: Correlation analysis
        demo_correlation_analysis()
        
        # Save calibrations
        print("💾 Saving calibrations to config/symbol_calibrations.yaml...")
        cal_manager.save_calibrations()
        
        print("\n✅ P1.1 Demo Complete!")
        print("\nKey Improvements:")
        print("  📈 Individual symbol optimization")
        print("  🎯 Adaptive parameter tuning")
        print("  🛡️ Advanced risk management")
        print("  🔗 Correlation-aware position sizing")
        print("  💾 Persistent calibration storage")
        
        print(f"\n📝 Check config/symbol_calibrations.yaml for saved calibrations")
        
    except Exception as e:
        print(f"❌ Error in P1.1 demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
