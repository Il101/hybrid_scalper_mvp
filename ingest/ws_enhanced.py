"""
Enhanced WebSocket manager with reliability monitoring and intelligent failover.
Integrates health monitoring, quality checks, and automatic recovery mechanisms.
"""
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass
import aiohttp
from .ws_monitor import WSHealthMonitor, get_health_monitor

if TYPE_CHECKING:
    from aiohttp import ClientWebSocket

logger = logging.getLogger(__name__)

@dataclass 
class WSConnectionConfig:
    """WebSocket connection configuration"""
    url: str
    symbols: List[str]
    max_reconnect_attempts: int = 10
    reconnect_delay_seconds: float = 5.0
    heartbeat_interval: float = 30.0
    connection_timeout: float = 10.0
    priority: int = 1  # Lower number = higher priority

class EnhancedWSManager:
    """Enhanced WebSocket manager with reliability monitoring"""
    
    def __init__(self, configs: List[WSConnectionConfig], 
                 data_callback: Optional[Callable[[str, Dict], None]] = None):
        self.configs = configs
        self.data_callback = data_callback
        self.connections: Dict[str, Dict] = {}  # connection_id -> connection info
        self.active_connections: List[str] = []
        self.health_monitor = get_health_monitor()
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Failover settings
        self.primary_connection: Optional[str] = None
        self.failover_threshold_seconds = 15.0
        self.min_healthy_connections = 1
        
        # Symbol routing - which connection handles which symbol
        self.symbol_routing: Dict[str, str] = {}
        self._setup_symbol_routing()
    
    def _setup_symbol_routing(self) -> None:
        """Setup initial symbol routing to primary connections"""
        # Sort configs by priority (lower number = higher priority)
        sorted_configs = sorted(self.configs, key=lambda c: c.priority)
        
        for config in sorted_configs:
            conn_id = self._generate_connection_id(config)
            
            # Register with health monitor
            self.health_monitor.register_connection(conn_id, config.url)
            
            # Assign symbols to this connection
            for symbol in config.symbols:
                if symbol not in self.symbol_routing:
                    self.symbol_routing[symbol] = conn_id
            
            # Set primary connection to highest priority
            if self.primary_connection is None:
                self.primary_connection = conn_id
    
    def _generate_connection_id(self, config: WSConnectionConfig) -> str:
        """Generate unique connection ID"""
        # Extract exchange name from URL
        if 'bybit' in config.url.lower():
            exchange = 'bybit'
        elif 'binance' in config.url.lower():
            exchange = 'binance'
        else:
            exchange = 'unknown'
        
        return f"{exchange}_{config.priority}"
    
    async def start(self) -> None:
        """Start all WebSocket connections"""
        if self._running:
            return
        
        self._running = True
        logger.info(f"🚀 Starting enhanced WebSocket manager with {len(self.configs)} connections")
        
        # Start connection tasks
        for config in self.configs:
            task = asyncio.create_task(self._connection_loop(config))
            self._tasks.append(task)
        
        # Start failover monitoring task
        task = asyncio.create_task(self._failover_monitor())
        self._tasks.append(task)
        
        logger.info("✅ Enhanced WebSocket manager started")
    
    async def stop(self) -> None:
        """Stop all connections"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        self.connections.clear()
        logger.info("🛑 Enhanced WebSocket manager stopped")
    
    async def _connection_loop(self, config: WSConnectionConfig) -> None:
        """Main connection loop with reconnection logic"""
        conn_id = self._generate_connection_id(config)
        reconnect_attempts = 0
        
        while self._running:
            try:
                # Connect and handle messages
                await self._handle_connection(config, conn_id)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error(f"❌ Connection {conn_id} error: {e}")
                
                # Record disconnection
                self.health_monitor.record_connection_event(conn_id, "disconnected")
                
                # Remove from active connections
                if conn_id in self.active_connections:
                    self.active_connections.remove(conn_id)
                
                reconnect_attempts += 1
                
                if reconnect_attempts >= config.max_reconnect_attempts:
                    logger.error(f"💀 Connection {conn_id} exceeded max reconnection attempts")
                    break
                
                # Exponential backoff
                delay = config.reconnect_delay_seconds * (2 ** min(reconnect_attempts - 1, 5))
                logger.info(f"⏳ Reconnecting {conn_id} in {delay}s (attempt {reconnect_attempts})")
                
                # Record reconnect attempt
                self.health_monitor.record_connection_event(conn_id, "reconnect_attempt")
                
                await asyncio.sleep(delay)
        
        logger.info(f"🔌 Connection loop ended for {conn_id}")
    
    async def _handle_connection(self, config: WSConnectionConfig, conn_id: str) -> None:
        """Handle individual WebSocket connection"""
        timeout = aiohttp.ClientTimeout(total=config.connection_timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(config.url) as ws:
                logger.info(f"🔗 Connected {conn_id} to {config.url}")
                
                # Record successful connection
                self.health_monitor.record_connection_event(conn_id, "connected")
                self.active_connections.append(conn_id)
                
                # Store connection info
                self.connections[conn_id] = {
                    'config': config,
                    'websocket': ws,
                    'last_heartbeat': time.time()
                }
                
                # Send subscription message
                await self._subscribe_to_symbols(ws, config.symbols)
                
                # Start heartbeat task
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(ws, conn_id, config.heartbeat_interval)
                )
                
                try:
                    # Message handling loop
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_message(conn_id, msg.data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error on {conn_id}: {ws.exception()}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            logger.info(f"WebSocket closed for {conn_id}")
                            break
                
                finally:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
    
    async def _subscribe_to_symbols(self, ws: Any, symbols: List[str]) -> None:
        """Send subscription message for symbols"""
        # Bybit format
        subscribe_msg = {
            "op": "subscribe",
            "args": [f"orderbook.1.{symbol}" for symbol in symbols]
        }
        
        await ws.send_str(json.dumps(subscribe_msg))
        logger.info(f"📡 Subscribed to {len(symbols)} symbols: {symbols}")
    
    async def _heartbeat_loop(self, ws: Any, conn_id: str, interval: float) -> None:
        """Send periodic heartbeat/ping messages"""
        while not ws.closed:
            try:
                start_time = time.time()
                
                # Send ping
                await ws.send_str(json.dumps({"op": "ping"}))
                
                # Wait for pong (simplified - in real implementation, track actual pong response)
                await asyncio.sleep(0.1)  
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Record heartbeat
                self.health_monitor.record_connection_event(
                    conn_id, "heartbeat", {"latency_ms": latency_ms}
                )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Heartbeat failed for {conn_id}: {e}")
                break
    
    async def _handle_message(self, conn_id: str, message: str) -> None:
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Record message received
            self.health_monitor.record_connection_event(conn_id, "message_received")
            
            # Skip non-data messages
            if 'topic' not in data or 'data' not in data:
                return
            
            # Extract symbol and orderbook data
            topic = data['topic']
            if not topic.startswith('orderbook'):
                return
            
            symbol = topic.split('.')[-1]  # Extract symbol from topic
            orderbook = data['data']
            
            # Quality check
            quality_issues = self.health_monitor.check_data_quality(symbol, orderbook)
            if quality_issues:
                # Log critical issues
                critical_issues = [i for i in quality_issues if i.severity == "CRITICAL"]
                if critical_issues:
                    logger.warning(f"⚠️ Critical data quality issues for {symbol}: "
                                 f"{[i.description for i in critical_issues]}")
            
            # Forward to callback if this connection handles this symbol
            if (self.data_callback and 
                self.symbol_routing.get(symbol) == conn_id):
                
                self.data_callback(symbol, orderbook)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message from {conn_id}: {e}")
        except Exception as e:
            logger.error(f"Error handling message from {conn_id}: {e}")
    
    async def _failover_monitor(self) -> None:
        """Monitor connections and trigger failover when needed"""
        while self._running:
            try:
                await self._check_failover_conditions()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Failover monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _check_failover_conditions(self) -> None:
        """Check if failover is needed and execute if necessary"""
        if len(self.active_connections) < self.min_healthy_connections:
            logger.warning(f"🚨 Only {len(self.active_connections)} healthy connections, minimum is {self.min_healthy_connections}")
            return
        
        # Check if primary connection is unhealthy
        if self.primary_connection:
            health_summary = self.health_monitor.get_health_summary()
            primary_health = health_summary['connections'].get(self.primary_connection, {})
            
            if not primary_health.get('healthy', False):
                await self._execute_failover()
    
    async def _execute_failover(self) -> None:
        """Execute failover to backup connection"""
        logger.warning(f"🔄 Executing failover from primary connection {self.primary_connection}")
        
        # Find healthy backup connection
        health_summary = self.health_monitor.get_health_summary()
        healthy_connections = [
            conn_id for conn_id, info in health_summary['connections'].items()
            if info.get('healthy', False) and conn_id != self.primary_connection
        ]
        
        if not healthy_connections:
            logger.error("❌ No healthy backup connections available for failover")
            return
        
        # Choose backup with best health metrics
        best_backup = min(healthy_connections, 
                         key=lambda c: float(health_summary['connections'][c].get('latency_ms', '999')))
        
        logger.info(f"✅ Failing over to backup connection: {best_backup}")
        
        # Reroute all symbols to backup connection
        for symbol in self.symbol_routing:
            if self.symbol_routing[symbol] == self.primary_connection:
                self.symbol_routing[symbol] = best_backup
        
        # Update primary connection
        old_primary = self.primary_connection
        self.primary_connection = best_backup
        
        logger.info(f"🔄 Failover complete: {old_primary} -> {best_backup}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get comprehensive connection status"""
        health_summary = self.health_monitor.get_health_summary()
        
        return {
            'timestamp': time.time(),
            'primary_connection': self.primary_connection,
            'active_connections': len(self.active_connections),
            'symbol_routing': dict(self.symbol_routing),
            'health_summary': health_summary,
            'quality_issues_last_hour': len(self.health_monitor.get_quality_issues(hours=1)),
        }
