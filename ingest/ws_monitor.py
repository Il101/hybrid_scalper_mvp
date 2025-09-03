"""
WebSocket reliability monitoring and enhanced failover system.
Provides comprehensive connection health tracking, quality metrics, and intelligent failover.
"""
from __future__ import annotations
import time
import json
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import logging
from statistics import mean
import queue

logger = logging.getLogger(__name__)

@dataclass
class ConnectionMetrics:
    """WebSocket connection health metrics"""
    connection_id: str
    url: str
    connected_since: Optional[float] = None
    last_heartbeat: Optional[float] = None
    
    # Latency tracking
    ping_times: deque = field(default_factory=lambda: deque(maxlen=100))
    avg_latency_ms: float = 0.0
    
    # Message tracking
    messages_received: int = 0
    messages_per_second: float = 0.0
    last_message_time: Optional[float] = None
    
    # Quality metrics
    connection_drops: int = 0
    reconnection_attempts: int = 0
    data_quality_score: float = 1.0  # 0-1 scale
    uptime_pct: float = 100.0
    
    # Alert flags
    is_healthy: bool = True
    alert_level: str = "GREEN"  # GREEN, YELLOW, RED
    last_alert_time: Optional[float] = None

@dataclass
class DataQualityIssue:
    """Data quality problem detected"""
    timestamp: float
    symbol: str
    issue_type: str  # "stale_data", "crossed_book", "unrealistic_spread", "missing_levels"
    severity: str    # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    description: str
    data_snapshot: Optional[Dict] = None

class WSHealthMonitor:
    """Real-time WebSocket connection health monitoring"""
    
    def __init__(self, alert_callback: Optional[Callable[[str, str], None]] = None):
        self.connections: Dict[str, ConnectionMetrics] = {}
        self.quality_issues: deque = deque(maxlen=1000)  # Store recent issues
        self.alert_callback = alert_callback
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Thresholds for alerts
        self.thresholds = {
            'max_latency_ms': 200,
            'min_message_rate': 1.0,  # Messages per second
            'max_stale_seconds': 30,
            'min_uptime_pct': 95.0,
            'critical_latency_ms': 500,
        }
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("🔍 WebSocket health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("🛑 WebSocket health monitoring stopped")
    
    def register_connection(self, connection_id: str, url: str) -> None:
        """Register a new connection for monitoring"""
        with self._lock:
            self.connections[connection_id] = ConnectionMetrics(connection_id, url)
            logger.info(f"📡 Registered connection: {connection_id} -> {url}")
    
    def record_connection_event(self, connection_id: str, event_type: str, 
                               data: Optional[Dict] = None) -> None:
        """Record connection event (connect, disconnect, message, ping, etc.)"""
        with self._lock:
            if connection_id not in self.connections:
                return
                
            conn = self.connections[connection_id]
            now = time.time()
            
            if event_type == "connected":
                conn.connected_since = now
                conn.is_healthy = True
                conn.alert_level = "GREEN"
                
            elif event_type == "disconnected":
                conn.connection_drops += 1
                conn.connected_since = None
                conn.is_healthy = False
                conn.alert_level = "RED"
                self._trigger_alert(connection_id, "CONNECTION_LOST", 
                                   f"Connection {connection_id} disconnected")
                
            elif event_type == "message_received":
                conn.messages_received += 1
                conn.last_message_time = now
                
            elif event_type == "heartbeat":
                conn.last_heartbeat = now
                if data and 'latency_ms' in data:
                    conn.ping_times.append(data['latency_ms'])
                    if conn.ping_times:
                        conn.avg_latency_ms = mean(conn.ping_times)
                        
            elif event_type == "reconnect_attempt":
                conn.reconnection_attempts += 1
    
    def check_data_quality(self, symbol: str, orderbook: Dict) -> List[DataQualityIssue]:
        """Check orderbook data for quality issues"""
        issues = []
        now = time.time()
        
        if not orderbook:
            issues.append(DataQualityIssue(
                timestamp=now,
                symbol=symbol,
                issue_type="missing_data",
                severity="HIGH", 
                description="Empty orderbook received"
            ))
            return issues
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        # Check for crossed book
        if bids and asks:
            best_bid = float(bids[0][0]) if bids[0] else 0
            best_ask = float(asks[0][0]) if asks[0] else 0
            
            if best_bid >= best_ask:
                issues.append(DataQualityIssue(
                    timestamp=now,
                    symbol=symbol,
                    issue_type="crossed_book",
                    severity="CRITICAL",
                    description=f"Crossed book: bid {best_bid} >= ask {best_ask}",
                    data_snapshot={'best_bid': best_bid, 'best_ask': best_ask}
                ))
            
            # Check for unrealistic spreads
            if best_ask > 0:
                spread_pct = (best_ask - best_bid) / best_ask * 100
                if spread_pct > 5.0:  # >5% spread is suspicious
                    issues.append(DataQualityIssue(
                        timestamp=now,
                        symbol=symbol,
                        issue_type="unrealistic_spread", 
                        severity="MEDIUM",
                        description=f"Very wide spread: {spread_pct:.2f}%",
                        data_snapshot={'spread_pct': spread_pct}
                    ))
        
        # Check for insufficient depth
        if len(bids) < 5 or len(asks) < 5:
            issues.append(DataQualityIssue(
                timestamp=now,
                symbol=symbol,
                issue_type="insufficient_depth",
                severity="MEDIUM",
                description=f"Low depth: {len(bids)} bids, {len(asks)} asks"
            ))
        
        # Store issues
        with self._lock:
            self.quality_issues.extend(issues)
        
        return issues
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring:
            try:
                self._check_connection_health()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _check_connection_health(self) -> None:
        """Check health of all connections"""
        now = time.time()
        
        with self._lock:
            for conn_id, conn in self.connections.items():
                if not conn.connected_since:
                    continue  # Not connected
                
                # Update uptime
                uptime_seconds = now - conn.connected_since
                total_time = uptime_seconds + (conn.connection_drops * 30)  # Assume 30s downtime per drop
                conn.uptime_pct = (uptime_seconds / total_time) * 100 if total_time > 0 else 100
                
                # Update message rate
                if conn.last_message_time:
                    time_window = min(60, now - (conn.connected_since or now))  # Last 60 seconds
                    if time_window > 0:
                        conn.messages_per_second = conn.messages_received / time_window
                
                # Check for issues
                self._evaluate_connection_health(conn)
    
    def _evaluate_connection_health(self, conn: ConnectionMetrics) -> None:
        """Evaluate and update connection health status"""
        now = time.time()
        issues = []
        
        # Check latency
        if conn.avg_latency_ms > self.thresholds['critical_latency_ms']:
            issues.append(("CRITICAL", f"Very high latency: {conn.avg_latency_ms:.1f}ms"))
        elif conn.avg_latency_ms > self.thresholds['max_latency_ms']:
            issues.append(("HIGH", f"High latency: {conn.avg_latency_ms:.1f}ms"))
        
        # Check message rate
        if conn.messages_per_second < self.thresholds['min_message_rate']:
            issues.append(("MEDIUM", f"Low message rate: {conn.messages_per_second:.2f}/sec"))
        
        # Check for stale data
        if conn.last_message_time and (now - conn.last_message_time) > self.thresholds['max_stale_seconds']:
            issues.append(("HIGH", f"Stale data: {now - conn.last_message_time:.0f}s since last message"))
        
        # Check uptime
        if conn.uptime_pct < self.thresholds['min_uptime_pct']:
            issues.append(("MEDIUM", f"Low uptime: {conn.uptime_pct:.1f}%"))
        
        # Update alert level
        if any(severity == "CRITICAL" for severity, _ in issues):
            new_alert_level = "RED"
        elif any(severity == "HIGH" for severity, _ in issues):
            new_alert_level = "RED" 
        elif any(severity == "MEDIUM" for severity, _ in issues):
            new_alert_level = "YELLOW"
        else:
            new_alert_level = "GREEN"
        
        # Trigger alerts on status change
        if new_alert_level != conn.alert_level and new_alert_level != "GREEN":
            conn.alert_level = new_alert_level
            conn.is_healthy = new_alert_level == "GREEN"
            
            issue_summary = "; ".join([msg for _, msg in issues])
            self._trigger_alert(conn.connection_id, f"HEALTH_{new_alert_level}", issue_summary)
    
    def _trigger_alert(self, connection_id: str, alert_type: str, message: str) -> None:
        """Trigger alert via callback"""
        logger.warning(f"🚨 {alert_type}: {connection_id} - {message}")
        
        if self.alert_callback:
            try:
                self.alert_callback(f"{alert_type}:{connection_id}", message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Update alert timestamp
        if connection_id in self.connections:
            self.connections[connection_id].last_alert_time = time.time()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        with self._lock:
            summary = {
                'timestamp': time.time(),
                'total_connections': len(self.connections),
                'healthy_connections': sum(1 for c in self.connections.values() if c.is_healthy),
                'connections': {},
                'recent_quality_issues': len([q for q in self.quality_issues 
                                            if time.time() - q.timestamp < 300]),  # Last 5 minutes
                'avg_latency_ms': 0.0,
                'total_uptime_pct': 0.0,
            }
            
            if self.connections:
                latencies = [c.avg_latency_ms for c in self.connections.values() if c.avg_latency_ms > 0]
                uptimes = [c.uptime_pct for c in self.connections.values()]
                
                summary['avg_latency_ms'] = mean(latencies) if latencies else 0.0
                summary['total_uptime_pct'] = mean(uptimes) if uptimes else 0.0
            
            for conn_id, conn in self.connections.items():
                summary['connections'][conn_id] = {
                    'url': conn.url,
                    'healthy': conn.is_healthy,
                    'alert_level': conn.alert_level,
                    'latency_ms': f"{conn.avg_latency_ms:.1f}",
                    'msg_rate': f"{conn.messages_per_second:.2f}/sec",
                    'uptime': f"{conn.uptime_pct:.1f}%",
                    'drops': conn.connection_drops,
                    'reconnects': conn.reconnection_attempts,
                }
        
        return summary
    
    def get_quality_issues(self, hours: int = 1) -> List[DataQualityIssue]:
        """Get data quality issues from last N hours"""
        cutoff = time.time() - (hours * 3600)
        
        with self._lock:
            return [issue for issue in self.quality_issues if issue.timestamp > cutoff]

# Global monitor instance
_health_monitor: Optional[WSHealthMonitor] = None

def get_health_monitor() -> WSHealthMonitor:
    """Get global WebSocket health monitor"""
    global _health_monitor
    if _health_monitor is None:
        def default_alert_callback(alert_type: str, message: str):
            logger.warning(f"🚨 WS Alert [{alert_type}]: {message}")
        
        _health_monitor = WSHealthMonitor(alert_callback=default_alert_callback)
        _health_monitor.start_monitoring()
    return _health_monitor
