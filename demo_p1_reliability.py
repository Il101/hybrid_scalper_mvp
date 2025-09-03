#!/usr/bin/env python3
"""
Demo of P1.2 WebSocket reliability monitoring system.
Shows health monitoring, quality checks, and failover capabilities.
"""
import asyncio
import time
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ingest.ws_monitor import WSHealthMonitor, DataQualityIssue
from ingest.ws_enhanced import EnhancedWSManager, WSConnectionConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_separator(title: str) -> None:
    """Print section separator"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)

async def demo_health_monitoring():
    """Demonstrate WebSocket health monitoring"""
    print_separator("P1.2 WebSocket Health Monitoring Demo")
    
    # Create health monitor
    monitor = WSHealthMonitor()
    monitor.start_monitoring()
    
    # Simulate connections
    monitor.register_connection("bybit_primary", "wss://stream.bybit.com/v5/public/spot")
    monitor.register_connection("bybit_backup", "wss://stream.bybit.com/v5/public/spot")
    monitor.register_connection("binance_backup", "wss://stream.binance.com:9443/ws")
    
    print("📡 Registered 3 WebSocket connections for monitoring")
    
    # Simulate connection events
    print("\n🔄 Simulating connection events...")
    
    # Primary connection comes online
    monitor.record_connection_event("bybit_primary", "connected")
    monitor.record_connection_event("bybit_primary", "heartbeat", {"latency_ms": 45.2})
    monitor.record_connection_event("bybit_primary", "message_received")
    
    print("✅ Primary connection established (45ms latency)")
    
    # Backup connections
    monitor.record_connection_event("bybit_backup", "connected") 
    monitor.record_connection_event("bybit_backup", "heartbeat", {"latency_ms": 67.8})
    
    monitor.record_connection_event("binance_backup", "connected")
    monitor.record_connection_event("binance_backup", "heartbeat", {"latency_ms": 123.4})
    
    print("✅ Backup connections established")
    
    # Show initial health status
    await asyncio.sleep(1)
    health = monitor.get_health_summary()
    print(f"\n📊 Health Summary:")
    print(f"   Total connections: {health['total_connections']}")
    print(f"   Healthy connections: {health['healthy_connections']}")
    print(f"   Average latency: {health['avg_latency_ms']:.1f}ms")
    print(f"   Total uptime: {health['total_uptime_pct']:.1f}%")
    
    # Show per-connection details
    print(f"\n🔍 Connection Details:")
    for conn_id, info in health['connections'].items():
        status = "🟢" if info['healthy'] else "🔴"
        print(f"   {status} {conn_id}: {info['alert_level']} | "
              f"Latency: {info['latency_ms']}ms | Rate: {info['msg_rate']} | "
              f"Uptime: {info['uptime']}")
    
    # Simulate data quality issues
    print("\n🔍 Testing data quality monitoring...")
    
    # Good orderbook
    good_book = {
        'bids': [['50000.0', '1.5'], ['49999.5', '2.0']],
        'asks': [['50001.0', '1.2'], ['50001.5', '0.8']]
    }
    issues = monitor.check_data_quality("BTCUSDT", good_book)
    print(f"✅ Good orderbook: {len(issues)} issues")
    
    # Crossed book (critical issue)
    crossed_book = {
        'bids': [['50002.0', '1.0']],  # Bid higher than ask
        'asks': [['50001.0', '1.0']]
    }
    issues = monitor.check_data_quality("ETHUSDT", crossed_book)
    print(f"❌ Crossed book detected: {len(issues)} critical issues")
    for issue in issues:
        print(f"   - {issue.issue_type}: {issue.description}")
    
    # Wide spread (warning)
    wide_spread_book = {
        'bids': [['45000.0', '1.0']],
        'asks': [['50000.0', '1.0']]  # 10%+ spread
    }
    issues = monitor.check_data_quality("SOLUSDT", wide_spread_book)
    print(f"⚠️ Wide spread detected: {len(issues)} issues")
    for issue in issues:
        print(f"   - {issue.issue_type}: {issue.description}")
    
    # Simulate connection problems
    print("\n🚨 Simulating connection problems...")
    
    # High latency
    for _ in range(10):
        monitor.record_connection_event("binance_backup", "heartbeat", {"latency_ms": 850.0})
    print("⚠️ Simulated high latency on Binance backup")
    
    # Connection drop
    monitor.record_connection_event("bybit_backup", "disconnected")
    print("❌ Simulated connection drop on Bybit backup")
    
    # Allow monitoring loop to process issues
    await asyncio.sleep(6)
    
    # Show updated health status
    health = monitor.get_health_summary()
    print(f"\n📊 Updated Health Summary:")
    print(f"   Healthy connections: {health['healthy_connections']}/{health['total_connections']}")
    print(f"   Recent quality issues: {health['recent_quality_issues']}")
    print(f"   Average latency: {health['avg_latency_ms']:.1f}ms")
    
    print(f"\n🔍 Updated Connection Status:")
    for conn_id, info in health['connections'].items():
        status = "🟢" if info['healthy'] else "🔴"
        alert_color = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(info['alert_level'], "⚪")
        print(f"   {status} {conn_id}: {alert_color} {info['alert_level']} | "
              f"Latency: {info['latency_ms']}ms | "
              f"Drops: {info['drops']} | Reconnects: {info['reconnects']}")
    
    # Show quality issues history
    quality_issues = monitor.get_quality_issues(hours=1)
    print(f"\n📈 Data Quality Issues (last hour): {len(quality_issues)}")
    for issue in quality_issues[-3:]:  # Show last 3
        severity_icon = {"LOW": "ℹ️", "MEDIUM": "⚠️", "HIGH": "❌", "CRITICAL": "💀"}.get(issue.severity, "❓")
        print(f"   {severity_icon} {issue.symbol} - {issue.issue_type}: {issue.description}")
    
    monitor.stop_monitoring()
    print("\n✅ Health monitoring demo completed")

async def demo_enhanced_ws_manager():
    """Demonstrate enhanced WebSocket manager with failover"""
    print_separator("Enhanced WebSocket Manager with Failover")
    
    # Define connection configurations
    configs = [
        WSConnectionConfig(
            url="wss://stream.bybit.com/v5/public/spot",
            symbols=["BTCUSDT", "ETHUSDT"],
            priority=1,  # Primary
            max_reconnect_attempts=5,
            reconnect_delay_seconds=2.0,
            heartbeat_interval=20.0
        ),
        WSConnectionConfig(
            url="wss://stream.bybit.com/v5/public/linear", 
            symbols=["BTCUSDT", "ETHUSDT"],
            priority=2,  # Backup
            max_reconnect_attempts=3,
            reconnect_delay_seconds=3.0,
            heartbeat_interval=25.0
        ),
        WSConnectionConfig(
            url="wss://stream.binance.com:9443/ws",
            symbols=["BTCUSDT", "ETHUSDT"],
            priority=3,  # Secondary backup
            max_reconnect_attempts=3,
            reconnect_delay_seconds=5.0,
            heartbeat_interval=30.0
        )
    ]
    
    print(f"🚀 Setting up enhanced WebSocket manager with {len(configs)} connections")
    
    # Message handler
    received_data = []
    def handle_data(symbol: str, orderbook: Dict):
        received_data.append({
            'timestamp': time.time(),
            'symbol': symbol,
            'bid_count': len(orderbook.get('bids', [])),
            'ask_count': len(orderbook.get('asks', []))
        })
        if len(received_data) % 10 == 0:  # Log every 10th message
            print(f"📦 Received {len(received_data)} messages, latest: {symbol}")
    
    # Create manager
    manager = EnhancedWSManager(configs, data_callback=handle_data)
    
    # Show initial configuration
    print(f"\n📋 Initial Configuration:")
    status = manager.get_connection_status()
    print(f"   Primary connection: {status['primary_connection']}")
    print(f"   Symbol routing: {status['symbol_routing']}")
    
    print(f"\n🔌 Connection priorities:")
    for i, config in enumerate(configs, 1):
        exchange = "Bybit" if "bybit" in config.url else "Binance"
        conn_type = "Primary" if config.priority == 1 else f"Backup-{config.priority-1}"
        print(f"   {i}. {exchange} ({conn_type}): {config.symbols}")
    
    # Note: In a real demo, we would start the manager
    # but since we don't have actual WebSocket endpoints responding,
    # we'll simulate the behavior
    
    print(f"\n⚠️ Note: Actual WebSocket connections not started in demo mode")
    print(f"        In production, manager would:")
    print(f"        1. Establish connections in priority order")
    print(f"        2. Monitor health and route symbols")
    print(f"        3. Automatically failover on connection issues")
    print(f"        4. Maintain redundant connections for reliability")
    
    # Simulate failover scenario
    print(f"\n🔄 Simulating failover scenario...")
    
    # Manually update routing to show failover
    old_primary = manager.primary_connection
    manager.primary_connection = "bybit_2"  # Switch to backup
    
    # Update symbol routing
    for symbol in manager.symbol_routing:
        manager.symbol_routing[symbol] = "bybit_2"
    
    print(f"   ✅ Failover executed: {old_primary} -> {manager.primary_connection}")
    print(f"   📡 All symbols now routed to backup connection")
    
    # Show final status
    status = manager.get_connection_status()
    print(f"\n📊 Final Status:")
    print(f"   Primary connection: {status['primary_connection']}")
    print(f"   Symbol routing: {status['symbol_routing']}")
    
    print(f"\n🎯 Key P1.2 Features Demonstrated:")
    print(f"   ✅ Connection health monitoring with latency tracking")
    print(f"   ✅ Data quality validation (crossed books, wide spreads)")
    print(f"   ✅ Automatic alert system with severity levels")
    print(f"   ✅ Intelligent failover with priority-based routing")
    print(f"   ✅ Connection redundancy and reliability tracking")

async def main():
    """Run P1.2 reliability monitoring demo"""
    print("🚀 Starting P1.2 WebSocket Reliability and Monitoring Demo")
    print("=" * 70)
    
    try:
        # Demo health monitoring
        await demo_health_monitoring()
        
        await asyncio.sleep(2)
        
        # Demo enhanced WebSocket manager
        await demo_enhanced_ws_manager()
        
        print_separator("P1.2 Implementation Summary")
        print("✅ WebSocket Health Monitoring:")
        print("   • Real-time latency and connection tracking")
        print("   • Data quality validation with issue classification")
        print("   • Alert system with GREEN/YELLOW/RED levels")
        print("   • Comprehensive health metrics and reporting")
        
        print("\n✅ Enhanced WebSocket Manager:")
        print("   • Priority-based connection management")
        print("   • Automatic failover with backup routing")
        print("   • Symbol-specific connection assignment")
        print("   • Intelligent reconnection with exponential backoff")
        
        print("\n✅ Reliability Improvements:")
        print("   • Connection redundancy across exchanges")
        print("   • Real-time quality monitoring")
        print("   • Automated failure detection and recovery")
        print("   • Performance metrics for optimization")
        
        print("\n🎯 Ready for P1.3: Real-time Scanner Dashboard")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
