"""
Bybit WebSocket v5 (public, linear) client for:
- orderbook (orderbook.50.SYMBOL)
- klines (kline.INTERVAL.SYMBOL)  e.g., kline.1.BTCUSDT (1 = 1m), kline.5.BTCUSDT (5 = 5m)
- trades  (publicTrade.SYMBOL)

Stores latest snapshots in memory for fast access.
"""
from __future__ import annotations
import json, threading, time
from typing import Dict, Optional, List
from websocket import WebSocketApp

WS_URL = "wss://stream.bybit.com/v5/public/linear"

_INTERVAL_MAP = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60"
}

class BybitWS:
    def __init__(self):
        self._ws: Optional[WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._subs: Dict[str, bool] = {}
        self._books: Dict[str, dict] = {}
        self._klines: Dict[str, dict] = {}   # key: f"{symbol}:{interval}"
        self._trades: Dict[str, dict] = {}   # last trade per symbol
        self._running = False

    def _on_open(self, ws):
        self._connection_attempts = 0  # Reset on successful connection
        self._last_activity = time.time()
        
        with self._lock:
            topics = [t for t, on in self._subs.items() if on]
        print(f"[BybitWS] Connected! Subscribing to topics: {topics}")
        if topics:
            try:
                # Small delay to allow underlying socket to fully initialize
                time.sleep(0.2)
                ws.send(json.dumps({"op": "subscribe", "args": topics}))
            except Exception as e:
                print(f"[BybitWS] Failed to send subscribe on open: {e}")

    def _on_message(self, ws, message: str):
        self._last_activity = time.time()  # Update activity timestamp
        
        try:
            msg = json.loads(message)
        except Exception:
            return

        topic = msg.get("topic", "")
        data = msg.get("data")
        if not topic or data is None:
            return

        # ORDERBOOK
        if topic.startswith("orderbook."):
            if isinstance(data, dict):
                bids = data.get("b") or []
                asks = data.get("a") or []
                ts = data.get("ts") or int(time.time()*1000)
                symbol = data.get("s") or topic.split(".")[-1]
                if bids or asks:
                    with self._lock:
                        self._books[symbol] = {"bids": [[float(p), float(q)] for p, q in bids],
                                               "asks": [[float(p), float(q)] for p, q in asks],
                                               "ts": ts}
        # KLINE: data is a list of dicts; take the last item
        elif topic.startswith("kline."):
            # topic: kline.{interval}.{symbol}
            parts = topic.split(".")
            if len(parts) >= 3:
                interval, symbol = parts[1], parts[2]
            else:
                interval = None; symbol = None
            if symbol and interval and isinstance(data, list) and data:
                item = data[-1]
                # item fields doc: https://bybit-exchange.github.io/docs/v5/websocket/public/kline
                # {'start':123,'end':..., 'interval':'1', 'open':'','close':'','high':'','low':'','volume':'','turnover':'','confirm':True, ...}
                try:
                    k = {
                        "start": int(item.get("start", 0)),
                        "end": int(item.get("end", 0)),
                        "open": float(item.get("open", 0.0)),
                        "high": float(item.get("high", 0.0)),
                        "low": float(item.get("low", 0.0)),
                        "close": float(item.get("close", 0.0)),
                        "volume": float(item.get("volume", 0.0)),
                        "confirm": bool(item.get("confirm", False)),
                        "ts": int(item.get("end", 0))
                    }
                    key = f"{symbol}:{interval}"
                    with self._lock:
                        self._klines[key] = k
                except Exception:
                    pass
        # TRADES: publicTrade.SYMBOL
        elif topic.startswith("publicTrade."):
            symbol = topic.split(".")[-1]
            # data is list of trades; take last
            try:
                item = data[-1] if isinstance(data, list) and data else None
                if item:
                    t = {
                        "price": float(item.get("p", 0.0)),
                        "qty": float(item.get("q", 0.0)),
                        "side": item.get("S", ""),
                        "ts": int(item.get("T", 0))
                    }
                    with self._lock:
                        self._trades[symbol] = t
            except Exception:
                pass

    def _on_error(self, ws, error):
        # Surface errors to stderr for easier debugging
        try:
            print(f"[BybitWS] WebSocket error: {error}")
        except Exception:
            pass

    def _on_ping(self, ws, message):
        try:
            print(f"[BybitWS] Received ping: {message}")
        except Exception:
            pass

    def _on_pong(self, ws, message):
        try:
            # Reduced verbosity - only log occasionally for monitoring
            current_time = time.time()
            if not hasattr(self, '_last_pong_log') or current_time - self._last_pong_log > 300:  # Log every 5 minutes
                print(f"[BybitWS] Heartbeat OK (last pong: {len(message)} bytes)")
                self._last_pong_log = current_time
        except Exception:
            pass

    def _on_close(self, ws, status_code, msg):
        try:
            print(f"[BybitWS] WebSocket closed: code={status_code} msg={msg}")
        except Exception:
            pass

    def start(self):
        if self._running:
            return
        print("[BybitWS] Starting WebSocket client...")
        self._running = True
        self._connection_attempts = 0
        self._last_activity = time.time()
        
        self._ws = WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_ping=self._on_ping,
            on_pong=self._on_pong
        )
        
        def _run():
            while self._running:
                try:
                    if self._ws is not None:
                        # Exponential backoff for reconnection
                        backoff_delay = min(2.0 * (1.5 ** self._connection_attempts), 30.0)
                        if self._connection_attempts > 0:
                            # Add jitter to prevent thundering herd
                            jitter = backoff_delay * 0.1 * (2 * time.time() % 1 - 1)  # ±10% jitter
                            backoff_delay += jitter
                            print(f"[BybitWS] Reconnecting in {backoff_delay:.2f}s (attempt {self._connection_attempts + 1})")
                            time.sleep(backoff_delay)
                        
                        print(f"[BybitWS] Connecting to {WS_URL}")
                        self._ws.run_forever(ping_interval=15, ping_timeout=5)
                        self._connection_attempts += 1
                        
                except Exception as e:
                    print(f"[BybitWS] Connection failed: {e}")
                    self._connection_attempts += 1
                    
                # Check for stale connection (no activity for too long)
                if time.time() - self._last_activity > 60:  # 1 minute timeout
                    print("[BybitWS] Stale connection detected, forcing reconnect")
                    self._last_activity = time.time()
                    
                time.sleep(0.1)
                
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        # Allow connection establishment
        time.sleep(4.0)

    def stop(self):
        self._running = False
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _subscribe(self, topic: str):
        with self._lock:
            already = self._subs.get(topic, False)
            self._subs[topic] = True
        # Only try sending immediately if the underlying socket is connected.
        sock = getattr(self._ws, 'sock', None)
        connected = False
        try:
            connected = bool(getattr(sock, 'connected', False))
        except Exception:
            connected = False

        if not already and self._ws and connected:
            try:
                self._ws.send(json.dumps({"op": "subscribe", "args": [topic]}))
            except Exception as e:
                # If send fails unexpectedly, leave subscription for on_open
                print(f"[BybitWS] subscribe send failed for {topic}: {e}")
        else:
            # Will be sent from _on_open once the connection establishes
            if not connected:
                print(f"[BybitWS] subscribe queued for {topic} (socket not connected yet)")

    # Public APIs
    def subscribe_orderbook(self, symbol: str, depth: int = 50):
        if not self._running: self.start()
        topic = f"orderbook.{depth}.{symbol}"
        self._subscribe(topic)

    def subscribe_kline(self, symbol: str, interval: str = "1m"):
        if not self._running: self.start()
        iv = _INTERVAL_MAP.get(interval, "1")
        topic = f"kline.{iv}.{symbol}"
        self._subscribe(topic)

    def subscribe_trades(self, symbol: str):
        if not self._running: self.start()
        topic = f"publicTrade.{symbol}"
        self._subscribe(topic)

    def get_orderbook_snapshot(self, symbol: str) -> Optional[dict]:
        with self._lock:
            return self._books.get(symbol)

    def get_last_kline(self, symbol: str, interval: str = "1m") -> Optional[dict]:
        key = f"{symbol}:{_INTERVAL_MAP.get(interval,'1')}"
        with self._lock:
            return self._klines.get(key)

    def get_last_trade(self, symbol: str) -> Optional[dict]:
        with self._lock:
            return self._trades.get(symbol)

# Singleton factory
_client: Optional[BybitWS] = None
def get_client() -> BybitWS:
    global _client
    if _client is None:
        _client = BybitWS()
        _client.start()
    return _client

def ensure_orderbook(symbol: str, depth: int = 50):
    c = get_client()
    c.subscribe_orderbook(symbol, depth=depth)
    # Увеличиваем таймаут и добавляем переподключение при необходимости
    for attempt in range(40):  # 40 * 0.5s = 20 секунд
        snap = c.get_orderbook_snapshot(symbol)
        if snap: 
            print(f"[ensure_orderbook] Got orderbook data for {symbol}")
            return
            
        # Если нет соединения, перезапускаем клиент
        if not c._running or not c._ws:
            print(f"[ensure_orderbook] Restarting WS client (attempt {attempt+1})")
            c.stop()
            time.sleep(1.0)
            c.start()
            c.subscribe_orderbook(symbol, depth=depth)
            
        time.sleep(0.5)
    raise RuntimeError(f"No Bybit WS orderbook snapshot for {symbol} after 20s.")

def ensure_kline(symbol: str, interval: str = "1m"):
    c = get_client()
    c.subscribe_kline(symbol, interval)
    # Увеличиваем таймаут и добавляем переподключение при необходимости
    for attempt in range(60):  # 60 * 0.5s = 30 секунд
        k = c.get_last_kline(symbol, interval)
        if k: 
            print(f"[ensure_kline] Got kline data for {symbol}:{interval}")
            return
        
        # Если нет соединения, перезапускаем клиент
        if not c._running or not c._ws:
            print(f"[ensure_kline] Restarting WS client (attempt {attempt+1})")
            c.stop()
            time.sleep(1.0)
            c.start()
            c.subscribe_kline(symbol, interval)
            
        time.sleep(0.5)
    raise RuntimeError(f"No Bybit WS kline for {symbol} interval {interval} after 30s.")
