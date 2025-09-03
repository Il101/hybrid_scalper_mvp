#!/usr/bin/env python3
"""
Quick test script for scanner debugging.
"""
import time
import logging
from scanner.candidates import build_watchlist, subscribe_watchlist, compute_priority, load_scanner_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("🔍 Scanner Debug Test")
    print("====================")
    
    try:
        # Load config
        config = load_scanner_config()
        print(f"📋 Config loaded: max_spread={config.get('max_spread_bps')}, min_depth={config.get('min_depth_usd_top10')}")
        
        # Build watchlist
        print("\n🏗️ Building watchlist...")
        watchlist = build_watchlist(top_n=5)  # Just top 5 for quick test
        print(f"📋 Watchlist: {watchlist}")
        
        # Subscribe to WS feeds
        print("\n📡 Subscribing to WebSocket feeds...")
        subscribe_watchlist(watchlist)
        
        # Wait for data to arrive
        print("⏳ Waiting 10 seconds for WebSocket data...")
        time.sleep(10)
        
        # Test priority computation
        print(f"\n🧮 Computing priorities...")
        results = []
        
        for symbol in watchlist:
            print(f"\n--- Testing {symbol} ---")
            result = compute_priority(symbol, config)
            if result:
                results.append(result)
                print(f"✅ {symbol}: Priority={result.priority:.2f}")
            else:
                print(f"❌ {symbol}: No data or filtered out")
        
        # Summary
        print(f"\n📊 Summary")
        print(f"Valid candidates: {len(results)}/{len(watchlist)}")
        
        if results:
            results.sort(key=lambda x: x.priority, reverse=True)
            print("🏆 Top candidates:")
            for i, score in enumerate(results[:3]):
                print(f"{i+1}. {score.symbol}: P={score.priority:.1f} "
                      f"(V={score.vol_score:.1f}, F={score.flow_score:.1f}, "
                      f"I={score.info_score:.1f}, C={score.cost_score:.1f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
