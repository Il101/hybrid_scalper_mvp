from __future__ import annotations
from dotenv import load_dotenv; load_dotenv()
import os
import yaml
import signal
import sys
import time
import csv
from features.ta_indicators import ta_score, atr_pct
from features.news_metrics import news_score
from features.sm_metrics import sm_score
from features.orderflow import compute_spread_bps, compute_obi, obi_to_score
from ingest.prices import get_ohlcv
from ingest.orderbook_ccxt import fetch_orderbook
from ingest.ws_bybit import get_client, ensure_orderbook
from signals.ensemble import ComponentScores, combine_with_meta
from exec.simulator import PaperBroker
from exec.slippage import estimate_slippage_bps, compute_size_usd, compute_size_with_slip, estimate_slippage_bps_ob

# Global flag for graceful shutdown
_shutdown_requested = False

def signal_handler(signum, frame):
    global _shutdown_requested
    print(f"\n🛑 Получен сигнал {signum}, начинаем graceful shutdown...")
    _shutdown_requested = True

# Cache last non-empty orderbook to avoid blocking waits and allow brief gaps in WS stream
_last_orderbook_cache: dict[str, dict] = {}

def log_scanner_data(symbol: str, priority: float, vol_score: float, flow_score: float, 
                    info_score: float, cost_score: float, selected: bool, reason: str = ""):
    """Log scanner data to logs/scanner.csv"""
    scanner_log_path = "logs/scanner.csv"
    
    # Create header if file doesn't exist
    if not os.path.exists(scanner_log_path):
        with open(scanner_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'symbol', 'priority', 'vol_score', 'flow_score', 
                           'info_score', 'cost_score', 'selected', 'reason'])
    
    # Append data
    with open(scanner_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            int(time.time() * 1000),
            symbol,
            round(priority, 4),
            round(vol_score, 4),
            round(flow_score, 4), 
            round(info_score, 4),
            round(cost_score, 4),
            selected,
            reason
        ])

def _get_ws_orderbook_fresh(symbol: str, max_age_ms: int = 2000) -> dict | None:
    """Return a fresh WS orderbook snapshot if available, else last non-empty cached one.
    Fresh = bids/asks non-empty and ts within max_age_ms.
    """
    try:
        c = get_client()
        ob = c.get_orderbook_snapshot(symbol)
        now_ms = int(time.time() * 1000)
        if ob and ob.get('bids') and ob.get('asks'):
            ts = int(ob.get('ts', 0))
            if ts and (now_ms - ts) <= max_age_ms:
                ob['_source'] = 'ws'
                _last_orderbook_cache[symbol] = ob
                return ob
        # Fallback to last cached non-empty orderbook (may be slightly stale)
        cached = _last_orderbook_cache.get(symbol)
        if cached:
            # mark cached snapshot so caller can decide
            cached['_source'] = 'cached_ws'
        return cached
    except Exception:
        cached = _last_orderbook_cache.get(symbol)
        if cached:
            cached['_source'] = 'cached_ws'
        return cached

def on_symbol_switch(old_symbol: str, new_symbol: str, broker: 'PaperBroker') -> None:
    """
    Handle symbol switching: close positions, reset state/buffers.
    Called when multi-symbol runner switches active symbol.
    """
    if old_symbol and old_symbol != new_symbol:
        print(f"🔄 Symbol switch: {old_symbol} -> {new_symbol}")
        
        # Close any open positions
        if not broker.flat():
            import time
            ts = str(int(time.time() * 1000))
            current_price = 0.0  # Would need to fetch current price
            broker.close(ts, old_symbol, current_price, reason="SYMBOL_SWITCH")
            print(f"📕 Closed position in {old_symbol} due to symbol switch")
        
        # Clear symbol-specific caches
        _last_orderbook_cache.pop(old_symbol, None)
        
        print(f"✅ Symbol switch complete: now trading {new_symbol}")

def run_simulation(symbol="BTCUSDT", timeframe="5m", steps=0, use_ws=False, use_ws_prices=False, fast_mode=True):
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    tf = timeframe
    exchange_for_ob = "bybit"
    print(f"Starting simulation: {symbol}, {tf}, steps={'∞' if steps == 0 else steps}, use_ws={use_ws}, use_ws_prices={use_ws_prices}, fast_mode={fast_mode}")
    
    with open("config.yaml","r",encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
    broker = PaperBroker(cash_usd=10000.0)
    
    print("Config loaded, broker initialized")

    # If live WS is enabled, subscribe once (avoid per-step blocking)
    if use_ws and exchange_for_ob.lower()=="bybit":
        from ingest.ws_manager import ws_subscribe, ws_ensure
        ws_subscribe(symbol, kind="kline", interval="1m" if tf=="1m" else "5m")
        ws_subscribe(symbol, kind="orderbook")  # Subscribe to orderbook too!
        # Ensure only once at start; later we'll use non-blocking cache getter
        try:
            ws_ensure(symbol, kind="kline", interval="1m" if tf=="1m" else "5m")
            ws_ensure(symbol, kind="orderbook")  # Ensure orderbook subscription
        except Exception as e:
            print(f"[warn] WS ensure failed: {e}")

    print("Getting OHLCV data...")
    try:
        df = get_ohlcv(symbol, tf, exchange=exchange_for_ob, market_type="futures", limit=1200)
        print(f"Got {len(df)} OHLCV bars")
    except Exception as e:
        print(f"[ERROR] Failed to get OHLCV data: {e}")
        # Re-raise exception for negative steps (used in testing)
        if steps < 0:
            raise
        return None
    
    # Get news score once at the beginning to avoid repeated API calls
    print("Getting news score once...")
    cached_news_score = news_score(symbol)
    print(f"Cached news score: {cached_news_score}")

    cached_news_score = news_score(symbol)
    print(f"Cached news score: {cached_news_score}")

    # Determine loop bounds
    last_processed_i = 200  # Track last processed index for shutdown
    if steps == 0:
        # Infinite mode - keep running and periodically refresh OHLCV data
        print("🔄 Запуск в бесконечном режиме. Нажмите Ctrl+C для остановки.")
        i = 200
        last_refresh = time.time()
        while not _shutdown_requested:
            # Refresh OHLCV data every 5 minutes in infinite mode
            if time.time() - last_refresh > 300:  # 5 minutes
                print("🔄 Обновление OHLCV данных...")
                old_len = len(df)
                df = get_ohlcv(symbol, tf, exchange=exchange_for_ob, market_type="futures", limit=1200)
                last_refresh = time.time()
                print(f"Обновлено {len(df)} OHLCV баров (было: {old_len})")
                # Reset i to work with refreshed data from a reasonable lookback
                i = max(200, len(df) - 100)  # Start from near the end of fresh data
            
            # Use current time index for infinite mode
            if len(df) > i - 200:
                process_step(df, i, symbol, tf, CFG, broker, cached_news_score, use_ws, use_ws_prices, fast_mode, exchange_for_ob)
                last_processed_i = i
                i += 1
            else:
                # Reached end of available data, force refresh and reset
                print("🔄 Достигнут конец данных, принудительное обновление...")
                old_len = len(df)
                df = get_ohlcv(symbol, tf, exchange=exchange_for_ob, market_type="futures", limit=1200)
                last_refresh = time.time()
                print(f"Принудительно обновлено {len(df)} OHLCV баров (было: {old_len})")
                i = max(200, len(df) - 100)  # Restart from near the end
                time.sleep(10)  # Short pause before continuing
                
            time.sleep(1)  # Faster iteration cadence in infinite mode
    else:
        # Finite mode - process specified number of steps
        end_step = min(len(df), 200 + steps)
        for i in range(200, end_step):
            if _shutdown_requested:
                break
            process_step(df, i, symbol, tf, CFG, broker, cached_news_score, use_ws, use_ws_prices, fast_mode, exchange_for_ob)
            last_processed_i = i

    # Graceful shutdown
    if not broker.flat():
        try:
            last_price = float(df["close"].iloc[-1])
            # Use the last processed step index for timestamp
            if fast_mode:
                ts = f"2024-01-01T00:{last_processed_i:02d}:00"
            else:
                ts = df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else str(df.index[-1])
            broker.close(ts, symbol, last_price, reason="SHUTDOWN")
            print("✅ Позиции закрыты при завершении работы")
        except Exception as e:
            print(f"⚠️ Ошибка при закрытии позиций: {e}")
    
    # Close WebSocket if used
    if use_ws:
        try:
            get_client().stop()
            print("✅ WebSocket соединение закрыто")
        except Exception as e:
            print(f"⚠️ Ошибка при закрытии WebSocket: {e}")
    
    return {"cash_final": broker.cash_usd, "log_path": "logs/trades.csv"}

def process_step(df, i, symbol, tf, CFG, broker, cached_news_score, use_ws, use_ws_prices, fast_mode, exchange_for_ob):
    """Process a single simulation step"""
    if i % 10 == 0:  # Print progress every 10 steps
        print(f"Processing step {i}")
    elif i == 200:  # Add detailed debug for first step
        print(f"First step {i}: Starting detailed debug...")
        
    window = df.iloc[:i].copy()
    # Обеспечиваем что индекс корректен
    if window.index.dtype == 'int64' or fast_mode:
        # Создаем mock timestamp для integer index
        ts = f"2024-01-01T00:{i:02d}:00"
    else:
        ts = window.index[-1].isoformat() if hasattr(window.index[-1], 'isoformat') else str(window.index[-1])
    
    # Price source: WS-first with REST fallback
    price_source = 'rest'
    price_age_ms = None
    price = float(window["close"].iloc[-1])  # Default REST price
    
    if use_ws_prices and use_ws and exchange_for_ob.lower() == "bybit":
        from ingest.ws_manager import ws_get
        try:
            ws_kline = ws_get(symbol, kind="kline", interval="1m" if tf=="1m" else "5m")
            if ws_kline and ws_kline.get('close'):
                kline_ts = int(ws_kline.get('ts', 0))
                current_ms = int(time.time() * 1000)
                age_ms = current_ms - kline_ts
                # Use WS price if fresh (within 10 seconds)
                if age_ms <= 10000:
                    price = float(ws_kline['close'])
                    price_source = 'ws'
                    price_age_ms = age_ms
                    if i == 200:
                        print(f"First step {i}: Using WS price {price} (age: {age_ms}ms)")
                elif i == 200:
                    print(f"First step {i}: WS price too stale ({age_ms}ms), using REST fallback")
        except Exception as e:
            if i == 200:
                print(f"First step {i}: WS price failed ({e}), using REST fallback")
    
    if i == 200 and price_source == 'rest':
        print(f"First step {i}: Using REST price {price}")

    if i == 200:
        print(f"First step {i}: Window created, calculating TA...")
    s_ta = ta_score(window)
    
    if i == 200:
        print(f"First step {i}: TA calculated, using cached news...")
    s_news = cached_news_score  # Use cached news score instead of calling API each time
    
    if i == 200:
        print(f"First step {i}: Getting SM score...")
    s_sm = sm_score(symbol)

    if i == 200:
        print(f"First step {i}: Getting orderbook...")
    if fast_mode:
        # Использовать мок-данные для orderbook вместо медленных API вызовов
        ob = {
            'bids': [[58000.0, 1.5], [57999.0, 2.0], [57998.0, 1.0]],
            'asks': [[58001.0, 1.2], [58002.0, 1.8], [58003.0, 0.8]]
        }
        ob_source = 'mock'
        ob_age_ms = 0
        if i == 200:
            print(f"First step {i}: Using mock orderbook data (fast_mode=True)")
    else:
        ob = None
        ob_source = None
        ob_age_ms = None
        if exchange_for_ob.lower() == "bybit" and use_ws:
            # Non-blocking: try fresh WS snapshot, fallback to last cached
            ob = _get_ws_orderbook_fresh(symbol, max_age_ms=2000)

        if ob is None:
            # Fallback REST API if WS missing
            try:
                ob = fetch_orderbook(exchange_for_ob, symbol.replace("USDT", "/USDT"))
                ob_source = 'rest'
                if i == 200:
                    print(f"First step {i}: Orderbook fetched successfully (REST fallback)")
            except Exception as e:
                if i == 200:
                    print(f"First step {i}: Orderbook fetch failed: {e}")
                ob = None
        else:
            ob_source = ob.get('_source', 'ws')
            if i == 200:
                print(f"First step {i}: Using WS orderbook (source: {ob_source})")
            # compute approximate age if timestamp present
            try:
                ts = int(ob.get('ts', 0))
                ob_age_ms = int(time.time() * 1000) - ts
            except Exception:
                ob_age_ms = None
            
    if i == 200:
        print(f"First step {i}: Computing spread and OBI...")
    spread_bps = compute_spread_bps(ob)
    obi = compute_obi(ob)
    s_obi = obi_to_score(obi)  # пока не используем в ensemble, но можно добавить

    gates = {
        "liquidity": spread_bps < CFG["gates"]["max_spread_bps"],
        "regime": True,
        "news_blackout": not CFG["gates"].get("news_blackout", False),
        "risk": True
    }

    comp = ComponentScores(s_news=s_news, s_sm=s_sm, s_ta=s_ta, gates=gates)
    final = combine_with_meta(window, comp, CFG)

    # === Risk filters ===
    if spread_bps > CFG["execution"]["spread_cap_bps"]:
        if not broker.flat():
            broker.close(ts, symbol, price, reason="SPREAD_CAP")
        return

    # === ATR-based stop sizing ===
    atrp = atr_pct(window) or 0.005
    stop_pct = CFG["execution"]["k_stop_atr"] * atrp

    # Confidence scaling from signal score 0..100
    conf_scale = max(0.3, min(1.0, final.score/100.0))

    # Update bars_in_pos if in trade
    if not broker.flat():
        broker.position.bars_in_pos += 1
        # manage trailing/time stops below...

    # Desired direction from signal
    dir_sig = final.direction

    # Obtain side-specific depth-aware size/slip only if we have a direction and orderbook
    size_usd = 0.0
    slip_bps = 0.0
    if dir_sig is not None and ob:
        size_usd, slip_bps = compute_size_with_slip(
            equity_usd=broker.cash_usd,
            risk_pct=CFG["risk"]["per_trade_pct"] * conf_scale,
            stop_pct=stop_pct,
            fee_bps=broker.fees_bps_round,
            spread_bps=spread_bps,
            orderbook=ob,
            side=dir_sig,
            max_frac=CFG["risk"]["max_position_frac"],
            levels=10,
            k_impact=30.0
        )

    # === Manage existing position (trailing, partials, time stop) ===
    one_r_pct = stop_pct
    if not broker.flat():
        atr_abs = atrp * price
        if broker.position.side == "long":
            # TP1 and BE
            if price >= broker.position.avg_price * (1.0 + CFG["execution"]["rr_tp1"] * one_r_pct) and broker.position.size_usd > 0:
                broker.partial_close(ts, symbol, price, CFG["execution"]["tp1_frac"], reason="TP1")
                broker.position.stop_price = broker.position.avg_price
            # Trailing
            trail = price - CFG["execution"]["trail_atr_k"] * atr_abs
            broker.position.stop_price = max(broker.position.stop_price, trail)
            # Time stop
            if broker.position.bars_in_pos >= CFG["execution"]["time_stop_bars"]:
                if price < broker.position.avg_price * (1.0 + 0.5 * one_r_pct):
                    broker.close(ts, symbol, price, reason="TIME_STOP")
        else:  # short
            if price <= broker.position.avg_price * (1.0 - CFG["execution"]["rr_tp1"] * one_r_pct) and broker.position.size_usd > 0:
                broker.partial_close(ts, symbol, price, CFG["execution"]["tp1_frac"], reason="TP1")
                broker.position.stop_price = broker.position.avg_price
            trail = price + CFG["execution"]["trail_atr_k"] * atr_abs
            broker.position.stop_price = min(broker.position.stop_price, trail) if broker.position.stop_price>0 else trail
            if broker.position.bars_in_pos >= CFG["execution"]["time_stop_bars"]:
                if price > broker.position.avg_price * (1.0 - 0.5 * one_r_pct):
                    broker.close(ts, symbol, price, reason="TIME_STOP")

        # Hard stop checks
        if broker.position.side == "long" and broker.position.stop_price>0 and price <= broker.position.stop_price:
            broker.close(ts, symbol, price, reason="STOP")
        if broker.position.side == "short" and broker.position.stop_price>0 and price >= broker.position.stop_price:
            broker.close(ts, symbol, price, reason="STOP")

    # === Entry / Flip execution with depth-aware slippage ===
    if dir_sig is None:
        pass
    else:
        if size_usd > 0:
            exec_price = price * (1 + (slip_bps/10000.0) if dir_sig=="long" else 1 - (slip_bps/10000.0))
            if broker.flat():
                stop_price = exec_price * (1.0 - one_r_pct) if dir_sig=="long" else exec_price * (1.0 + one_r_pct)
                broker.open(ts, symbol, dir_sig, exec_price, size_usd, stop_price, reason="SIGNAL")
            elif broker.position.side != dir_sig:
                stop_price = exec_price * (1.0 - one_r_pct) if dir_sig=="long" else exec_price * (1.0 + one_r_pct)
                broker.flip(ts, symbol, dir_sig, exec_price, size_usd, stop_price, reason="REVERSAL")

    # === Circuit breakers ===
    # Daily max loss and consecutive losses (approx: use equity mark-to-market at current price)
    # For simplicity, track since sim start
    # (Optional: could reset per calendar day using ts date)

    # --- signal logging ---
    os.makedirs("logs", exist_ok=True)
    sig_path = "logs/signals.csv"
    if not os.path.exists(sig_path):
        import csv
        with open(sig_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts","symbol","tf","score","direction","reason","s_news","s_sm","s_ta","spread_bps","obi","ob_source","ob_age_ms","price_source","price_age_ms"])
    import csv
    with open(sig_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, symbol, tf, final.score, final.direction or "", final.reason, s_news, s_sm, s_ta, spread_bps, obi, ob_source or '', ob_age_ms if ob_age_ms is not None else '', price_source, price_age_ms if price_age_ms is not None else ''])

if __name__ == "__main__":
    print(run_simulation(use_ws=False, use_ws_prices=False, steps=50))  # 50 шагов для тестирования
