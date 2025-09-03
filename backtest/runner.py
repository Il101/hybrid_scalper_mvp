"""
Multi-symbol trading runner with scanner integration.
Implements scan -> pick -> run active loop with soft symbol switching.
"""
from __future__ import annotations
import time
import signal
import sys
import logging
from typing import Optional
from scanner.candidates import (
    build_watchlist, subscribe_watchlist, compute_priority, 
    pick_active, log_scanner_data, load_scanner_config, ScannerState
)
from backtest.sim_loop import run_simulation

logger = logging.getLogger(__name__)

# Global state for graceful shutdown
_shutdown_requested = False

def signal_handler(signum, frame):
    global _shutdown_requested
    print(f"\n🛑 Получен сигнал {signum}, начинаем graceful shutdown...")
    _shutdown_requested = True

class MultiSymbolRunner:
    def __init__(self):
        self.config = load_scanner_config()
        self.watchlist = []
        self.scanner_state = ScannerState()
        self.current_symbol = None
        self.simulation_process = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def initialize(self):
        """Initialize watchlist and WS subscriptions"""
        logger.info("🔍 Initializing multi-symbol scanner...")
        
        # Build watchlist
        top_n = self.config.get('watchlist_top_n', 20)
        self.watchlist = build_watchlist(top_n=top_n)
        
        # Subscribe to WS feeds
        subscribe_watchlist(self.watchlist)
        
        # Wait for WS data to populate
        logger.info("⏳ Waiting for WS data to populate...")
        time.sleep(10)
        
        logger.info(f"✅ Scanner initialized with {len(self.watchlist)} symbols")
        
    def scan_and_pick(self) -> Optional[str]:
        """Scan all symbols and pick the best candidate"""
        scores = []
        
        logger.info("📊 Scanning symbols for priority scores...")
        
        for symbol in self.watchlist:
            score = compute_priority(symbol, self.config)
            if score:
                scores.append(score)
                
        if not scores:
            logger.warning("❌ No valid candidates found")
            return None
            
        # Pick active symbol with hysteresis
        new_symbol, self.scanner_state = pick_active(
            self.current_symbol, scores, self.scanner_state, self.config
        )
        
        # Log scanner data
        log_scanner_data(scores, new_symbol)
        
        # Print top 5 candidates
        scores.sort(key=lambda x: x.priority, reverse=True)
        logger.info("🏆 Top 5 candidates:")
        for i, score in enumerate(scores[:5]):
            active_marker = "👑" if score.symbol == new_symbol else "  "
            logger.info(f"{active_marker} {i+1}. {score.symbol}: P={score.priority:.1f} "
                       f"(V={score.vol_score:.0f}, F={score.flow_score:.0f}, "
                       f"I={score.info_score:.0f}, C={score.cost_score:.0f})")
        
        return new_symbol
    
    def on_symbol_switch(self, old_symbol: Optional[str], new_symbol: str):
        """Handle symbol switch: close positions, reset state"""
        if old_symbol and old_symbol != new_symbol:
            logger.info(f"🔄 Symbol switch detected: {old_symbol} -> {new_symbol}")
            
            # TODO: Close any open positions in old symbol
            # TODO: Reset counters/buffers
            # TODO: Set cooldown period
            
            cooldown_sec = self.config.get('cooldown_after_close_sec', 60)
            self.scanner_state.cooldown_until = time.time() + cooldown_sec
            
    def run(self):
        """Main runner loop: scan -> pick -> trade active symbol"""
        logger.info("🚀 Starting multi-symbol trading runner...")
        
        self.initialize()
        
        refresh_sec = self.config.get('refresh_sec', 30)
        last_scan = 0
        
        while not _shutdown_requested:
            current_time = time.time()
            
            # Periodic scanning
            if current_time - last_scan >= refresh_sec:
                new_symbol = self.scan_and_pick()
                
                if new_symbol and new_symbol != self.current_symbol:
                    self.on_symbol_switch(self.current_symbol, new_symbol)
                    self.current_symbol = new_symbol
                    
                last_scan = current_time
            
            # Trade active symbol (simplified - normally would integrate with sim_loop)
            if self.current_symbol:
                logger.info(f"📈 Trading active symbol: {self.current_symbol}")
                
                # In real implementation, this would be integrated with sim_loop
                # For now, just log and sleep
                time.sleep(min(refresh_sec, 10))
            else:
                logger.warning("⚠️ No active symbol selected")
                time.sleep(5)
                
        logger.info("✅ Multi-symbol runner shutdown complete")

def main():
    """Entry point for multi-symbol runner"""
    runner = MultiSymbolRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
    except Exception as e:
        logger.error(f"💥 Runner failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
