"""
Simplified real-time scanner dashboard - no external dependencies.
Provides text-based monitoring without plotly visualizations.
"""
import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import health monitor with fallback
try:
    from ingest.ws_monitor import get_health_monitor
    health_monitor = get_health_monitor()
except ImportError:
    class MockMonitor:
        def get_health_summary(self): 
            return {
                'total_connections': 3,
                'healthy_connections': 2,
                'avg_latency_ms': 85.3,
                'total_uptime_pct': 98.5,
                'connections': {
                    'bybit_primary': {
                        'healthy': True,
                        'alert_level': 'GREEN',
                        'latency_ms': '45.2',
                        'msg_rate': '2.5/sec',
                        'uptime': '99.2%',
                        'drops': 0,
                        'reconnects': 0
                    },
                    'bybit_backup': {
                        'healthy': False,
                        'alert_level': 'RED',
                        'latency_ms': '156.8',
                        'msg_rate': '0.0/sec',
                        'uptime': '87.3%',
                        'drops': 3,
                        'reconnects': 2
                    }
                }
            }
        def get_quality_issues(self, hours=1): 
            return []
    health_monitor = MockMonitor()

# Page configuration
st.set_page_config(
    page_title="Hybrid Scalper - Scanner Dashboard",
    page_icon="📊",
    layout="wide"
)

def generate_demo_data():
    """Generate demo data for dashboard"""
    import random
    
    # Symbol priorities
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", 
              "LINKUSDT", "AVAXUSDT", "MATICUSDT", "ATOMUSDT", "NEARUSDT"]
    
    symbol_priorities = {}
    for i, symbol in enumerate(symbols):
        base_priority = 90 - (i * 5)  # Decreasing priority
        variation = random.uniform(-3, 3)
        symbol_priorities[symbol] = max(0, min(100, base_priority + variation))
    
    # Recent signals
    recent_signals = []
    for i in range(15):
        signal_time = datetime.now() - timedelta(minutes=random.randint(1, 60))
        recent_signals.append({
            'timestamp': signal_time.strftime('%H:%M:%S'),
            'symbol': random.choice(symbols[:5]),
            'signal_type': random.choice(['BUY', 'SELL']),
            'confidence': round(random.uniform(0.6, 0.95), 3),
            'priority': round(random.uniform(50, 90), 1),
            'executed': '✅' if random.choice([True, False]) else '❌'
        })
    
    # Performance metrics
    performance = {
        'total_signals': len(recent_signals),
        'executed_signals': len([s for s in recent_signals if s['executed'] == '✅']),
        'win_rate': round(random.uniform(0.52, 0.58) * 100, 1),
        'sharpe_ratio': round(random.uniform(1.2, 2.1), 2),
        'max_drawdown': round(random.uniform(0.03, 0.08) * 100, 1)
    }
    
    return symbol_priorities, recent_signals, performance

def main():
    """Main dashboard application"""
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 Hybrid Scalper Dashboard")
        st.caption("Real-time scanner monitoring (Simplified Version)")
    
    with col2:
        if st.button("🔄 Refresh", type="primary"):
            st.rerun()
        st.metric("Time", datetime.now().strftime("%H:%M:%S"))
    
    # Generate demo data
    symbol_priorities, recent_signals, performance = generate_demo_data()
    health_data = health_monitor.get_health_summary()
    
    # System Status
    st.subheader("🚦 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        health_pct = (health_data['healthy_connections'] / max(1, health_data['total_connections'])) * 100
        status_icon = "🟢" if health_pct > 80 else "🟡" if health_pct > 50 else "🔴"
        st.metric(f"{status_icon} System Health", f"{health_pct:.0f}%")
    
    with col2:
        st.metric("🔍 Active Symbols", len(symbol_priorities))
    
    with col3:
        exec_rate = (performance['executed_signals'] / max(1, performance['total_signals'])) * 100
        st.metric("⚡ Execution Rate", f"{exec_rate:.0f}%")
    
    with col4:
        st.metric("📊 Data Quality", "🟢 Good")
    
    st.divider()
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    # Symbol Rankings
    with col1:
        st.subheader("🎯 Symbol Priority Rankings")
        
        # Create rankings DataFrame
        rankings_data = []
        for symbol, priority in sorted(symbol_priorities.items(), key=lambda x: x[1], reverse=True):
            tier = 'High' if priority > 70 else 'Medium' if priority > 50 else 'Low'
            tier_icon = '🔴' if tier == 'High' else '🟠' if tier == 'Medium' else '🟡'
            rankings_data.append({
                'Rank': len(rankings_data) + 1,
                'Symbol': symbol,
                'Priority': f"{priority:.1f}",
                'Tier': f"{tier_icon} {tier}"
            })
        
        st.dataframe(
            pd.DataFrame(rankings_data),
            use_container_width=True,
            hide_index=True
        )
    
    # Connection Health
    with col2:
        st.subheader("🌐 Connection Health")
        
        if health_data.get('connections'):
            conn_data = []
            for conn_id, info in health_data['connections'].items():
                status_icon = '🟢' if info['healthy'] else '🔴'
                alert_icons = {'GREEN': '🟢', 'YELLOW': '🟡', 'RED': '🔴'}
                alert_icon = alert_icons.get(info['alert_level'], '⚪')
                
                conn_data.append({
                    'Connection': conn_id.replace('_', ' ').title(),
                    'Status': f"{status_icon} {'Healthy' if info['healthy'] else 'Issues'}",
                    'Alert': f"{alert_icon} {info['alert_level']}",
                    'Latency': info['latency_ms'] + 'ms',
                    'Uptime': info['uptime']
                })
            
            st.dataframe(
                pd.DataFrame(conn_data),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No connection data available")
        
        # Health metrics
        st.metric("Average Latency", f"{health_data['avg_latency_ms']:.1f}ms")
        st.metric("System Uptime", f"{health_data['total_uptime_pct']:.1f}%")
    
    st.divider()
    
    # Signal History
    st.subheader("📈 Recent Signals")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if recent_signals:
            signals_df = pd.DataFrame(recent_signals)
            st.dataframe(
                signals_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No recent signals")
    
    with col2:
        st.markdown("**Signal Summary:**")
        st.metric("Total Signals", performance['total_signals'])
        st.metric("Executed", f"{performance['executed_signals']}")
        
        buy_count = len([s for s in recent_signals if s['signal_type'] == 'BUY'])
        sell_count = len([s for s in recent_signals if s['signal_type'] == 'SELL'])
        st.metric("Buy/Sell", f"{buy_count}:{sell_count}")
    
    st.divider()
    
    # Performance Metrics
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        win_rate = performance['win_rate']
        win_icon = "🟢" if win_rate > 55 else "🟡" if win_rate > 50 else "🔴"
        st.metric(f"{win_icon} Win Rate", f"{win_rate}%")
    
    with col2:
        sharpe = performance['sharpe_ratio']
        sharpe_icon = "🟢" if sharpe > 1.5 else "🟡" if sharpe > 1.0 else "🔴"
        st.metric(f"{sharpe_icon} Sharpe Ratio", sharpe)
    
    with col3:
        drawdown = performance['max_drawdown']
        dd_icon = "🟢" if drawdown < 5 else "🟡" if drawdown < 10 else "🔴"
        st.metric(f"{dd_icon} Max Drawdown", f"{drawdown}%")
    
    with col4:
        st.metric("🎯 Total Signals", performance['total_signals'])
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        auto_refresh = st.toggle("Auto Refresh", value=False)
        if auto_refresh:
            st.info("Auto-refresh disabled in simplified mode")
        
        st.header("📊 System Info")
        
        components = {
            "Health Monitor": "🟢 Active",
            "Scanner": "🟡 Demo Mode", 
            "Calibrations": "🟡 Mock Data",
            "Dashboard": "🟢 Simplified"
        }
        
        for component, status in components.items():
            st.markdown(f"**{component}:** {status}")
        
        if st.button("Export Data"):
            export_data = {
                'priorities': symbol_priorities,
                'health': health_data,
                'signals': recent_signals,
                'performance': performance,
                'timestamp': datetime.now().isoformat()
            }
            
            st.download_button(
                "Download JSON",
                json.dumps(export_data, indent=2),
                file_name=f"dashboard_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("**P1.3 Real-time Dashboard** - Simplified version without external chart dependencies")

if __name__ == "__main__":
    main()
