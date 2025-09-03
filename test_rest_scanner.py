#!/usr/bin/env python3
"""
Simple scanner demo with REST API fallback.
"""
import time
import logging
import ccxt
from typing import Optional, Dict
from scanner.candidates import CandidateScore, load_scanner_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ccxt_client():
    """Get CCXT client for REST API calls"""
    return ccxt.bybit({
        'sandbox': False,
        'options': {'defaultType': 'linear'}  # USDT perpetual
    })

def compute_priority_rest(symbol: str, config: Dict) -> Optional[CandidateScore]:
    """Compute priority using REST API (fallback for demo)"""
    try:
        client = get_ccxt_client()
        
        # Convert symbol format: BTCUSDT:USDT -> BTC/USDT
        ccxt_symbol = symbol.replace(':USDT', '').replace('USDT', '/USDT')
        
        # Get orderbook
        ob = client.fetch_order_book(ccxt_symbol)
        if not ob or not ob.get('bids') or not ob.get('asks'):
            logger.warning(f"❌ {symbol}: No orderbook data")
            return None
            
        try:
            best_bid = float(str(ob['bids'][0][0])) if ob['bids'] and ob['bids'][0] else 0.0
            best_ask = float(str(ob['asks'][0][0])) if ob['asks'] and ob['asks'][0] else 0.0
        except (ValueError, IndexError, TypeError):
            logger.warning(f"❌ {symbol}: Invalid bid/ask data")
            return None
            
        mid_price = (best_bid + best_ask) / 2
        
        if mid_price == 0:
            return None
            
        spread_bps = ((best_ask - best_bid) / mid_price) * 10000
        
        # Top 10 levels depth in USD
        depth_usd = 0.0
        try:
            for price, size in (ob['bids'][:10] + ob['asks'][:10]):
                if price is not None and size is not None:
                    depth_usd += float(str(price)) * float(str(size))
        except (ValueError, TypeError):
            depth_usd = 0.0
            
        logger.info(f"🔍 {symbol}: spread={spread_bps:.2f}bps, depth=${depth_usd:,.0f}")
        
        # Score components (0-100 scale)
        vol_score = min(100, depth_usd / 50000)     # Volume proxy from depth
        flow_score = max(0, 100 - spread_bps * 2)  # Tighter spread = better flow  
        info_score = min(100, depth_usd / 100000)  # Liquidity information
        cost_score = min(100, spread_bps * 3)      # Transaction cost penalty
        
        # Priority = Benefits - Costs
        priority = vol_score + flow_score + info_score - cost_score
        
        # Apply filters
        max_spread = config.get('max_spread_bps', 25)
        min_depth = config.get('min_depth_usd_top10', 25000)
        
        if spread_bps > max_spread:
            logger.warning(f"❌ {symbol}: spread {spread_bps:.2f} > {max_spread}")
            return None
        if depth_usd < min_depth:
            logger.warning(f"❌ {symbol}: depth ${depth_usd:,.0f} < ${min_depth:,.0f}")
            return None
            
        logger.info(f"🧮 {symbol}: P={priority:.1f} (V={vol_score:.1f}, F={flow_score:.1f}, I={info_score:.1f}, C={cost_score:.1f})")
        
        return CandidateScore(
            symbol=symbol,
            priority=float(priority),
            vol_score=float(vol_score),
            flow_score=float(flow_score),
            info_score=float(info_score),
            cost_score=float(cost_score),
            spread_bps=float(spread_bps),
            depth_usd=float(depth_usd),
            atr_pct=0.0,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Failed to compute priority for {symbol}: {e}")
        return None

def main():
    print("🚀 Simple Scanner Demo (REST API)")
    print("==================================")
    
    # Test symbols
    test_symbols = ['ETHUSDT:USDT', 'BTCUSDT:USDT', 'SOLUSDT:USDT']
    config = load_scanner_config()
    
    results = []
    
    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")
        result = compute_priority_rest(symbol, config)
        if result:
            results.append(result)
            print(f"✅ Priority: {result.priority:.2f}")
        else:
            print(f"❌ Filtered out or no data")
    
    if results:
        results.sort(key=lambda x: x.priority, reverse=True)
        print(f"\n🏆 Best candidate: {results[0].symbol} with priority {results[0].priority:.2f}")
    else:
        print("\n❌ No valid candidates found")

if __name__ == "__main__":
    main()
