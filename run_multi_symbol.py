#!/usr/bin/env python3
"""
Demo script for multi-symbol scanner system (P0 implementation).
Tests the scanner with relaxed filters for better trade generation.
"""

import sys
import time
import logging
from backtest.runner import MultiSymbolRunner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("🚀 Starting Multi-Symbol Scanner Demo")
    print("=====================================")
    
    # Configuration
    max_runtime = 300      # 5 minutes for demo
    
    print(f"⏳ Max runtime: {max_runtime}s")
    print(f"📊 Using relaxed filters for better trade generation")
    print()
    
    # Initialize runner
    runner = None
    try:
        runner = MultiSymbolRunner()
        
        print("🔍 Initializing scanner...")
        runner.initialize()
        
        start_time = time.time()
        refresh_sec = runner.config.get('refresh_sec', 30)
        last_scan = 0
        
        print("🔍 Starting scan loop...")
        print("Press Ctrl+C to stop gracefully")
        print("-" * 50)
        
        while True:
            current_time = time.time()
            
            # Check runtime limit
            if current_time - start_time > max_runtime:
                print(f"\n⏰ Max runtime ({max_runtime}s) reached, stopping...")
                break
            
            # Periodic scanning
            if current_time - last_scan >= refresh_sec:
                print(f"\n🔄 [T+{int(current_time - start_time)}s] Running scanner...")
                
                # Get current active symbol
                active_symbol = runner.current_symbol
                print(f"📍 Current symbol: {active_symbol or 'None'}")
                
                # Run scan and potentially switch
                new_symbol = runner.scan_and_pick()
                if new_symbol:
                    if new_symbol != active_symbol:
                        runner.on_symbol_switch(active_symbol, new_symbol)
                        runner.current_symbol = new_symbol
                        print(f"� Switched to: {new_symbol}")
                    else:
                        print(f"✅ Staying with: {active_symbol}")
                else:
                    print("❌ No suitable symbol found")
                
                last_scan = current_time
            
            # Brief pause before next check
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Keyboard interrupt received")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Demo complete")
        print("✅ Multi-symbol scanner demo finished")
        print(f"📝 Check logs/scanner.csv for detailed scan data")

if __name__ == "__main__":
    main()
