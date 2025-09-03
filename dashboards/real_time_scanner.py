"""
Real-time scanner dashboard with performance analytics and monitoring.
Provides comprehensive visualization of scanner state, priorities, and system health.
"""
import streamlit as st
import pandas as pd
import time
import json
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any

# Try to import plotly, provide fallback if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Mock plotly objects for type checking
    go = px = None

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import project components, provide fallbacks

try:
    from ingest.ws_monitor import get_health_monitor
except ImportError:
    from typing import Any as _Any
    def get_health_monitor() -> _Any:  # Fallback with broad typing to satisfy type-checkers
        class MockMonitor:
            def get_health_summary(self):
                return {}
            def get_quality_issues(self, hours: int = 1):
                return []
        return MockMonitor()

try:
    from features.symbol_specific import SymbolCalibrationManager
except ImportError:
    SymbolCalibrationManager = None

try:
    from risk.symbol_calibrated import PortfolioRiskManager
except ImportError:
    PortfolioRiskManager = None

# (Optional KPI utilities are available in utils.kpis as functions; no manager class required here)

# Page configuration
st.set_page_config(
    page_title="Hybrid Scalper - Real-Time Scanner Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .status-healthy { color: #28a745; }
    .status-warning { color: #ffc107; } 
    .status-critical { color: #dc3545; }
    .priority-high { background-color: #ffebee; }
    .priority-medium { background-color: #fff3e0; }
    .priority-low { background-color: #f3e5f5; }
</style>
""", unsafe_allow_html=True)

class DashboardData:
    """Container for all dashboard data"""
    
    def __init__(self):
        # Initialize components with proper None handling
        self.scanner = None
        self.calibration_manager = None
        self.risk_manager = None
        self.kpi_manager = None
        self.health_monitor = get_health_monitor()
        
        # Data storage
        self.symbol_priorities: Dict[str, float] = {}
        self.connection_health: Dict = {}
        self.quality_issues: List = []
        self.recent_signals: List = []
        self.performance_metrics: Dict = {}
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize scanner components"""
        try:
            # Initialize with demo symbols if actual scanner not available
            self.demo_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"]
            
            if SymbolCalibrationManager is not None:
                self.calibration_manager = SymbolCalibrationManager("config/symbol_calibrations.yaml")
                # Load existing calibrations
                self.calibration_manager.load_calibrations()
            
            # Generate demo data
            self._generate_demo_data()
            
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
    
    def _generate_demo_data(self):
        """Generate realistic demo data for dashboard"""
        import random
        import numpy as np
        
        # Symbol priorities with realistic distributions
        base_priorities = {
            "BTCUSDT": 85.2, "ETHUSDT": 78.5, "SOLUSDT": 71.8,
            "ADAUSDT": 65.3, "DOTUSDT": 59.7, "LINKUSDT": 54.2,
            "AVAXUSDT": 48.9, "MATICUSDT": 43.6, "ATOMUSDT": 38.4,
            "NEARUSDT": 33.1
        }
        
        # Add some time-based variation
        for symbol, base_priority in base_priorities.items():
            variation = random.uniform(-5, 5)
            self.symbol_priorities[symbol] = max(0, min(100, base_priority + variation))
        
        # Recent signals
        for i in range(20):
            signal_time = datetime.now() - timedelta(minutes=random.randint(1, 60))
            self.recent_signals.append({
                'timestamp': signal_time,
                'symbol': random.choice(self.demo_symbols),
                'signal_type': random.choice(['BUY', 'SELL']),
                'confidence': random.uniform(0.6, 0.95),
                'priority': random.uniform(30, 90),
                'executed': random.choice([True, False])
            })
        
        # Performance metrics
        self.performance_metrics = {
            'total_signals': len(self.recent_signals),
            'executed_signals': len([s for s in self.recent_signals if s['executed']]),
            'avg_confidence': np.mean([s['confidence'] for s in self.recent_signals]),
            'avg_priority': np.mean([s['priority'] for s in self.recent_signals]),
            'win_rate': random.uniform(0.52, 0.58),
            'sharpe_ratio': random.uniform(1.2, 2.1),
            'max_drawdown': random.uniform(0.03, 0.08)
        }
        
        # Connection health (from health monitor)
        self.connection_health = self.health_monitor.get_health_summary()
        
        # Quality issues
        self.quality_issues = self.health_monitor.get_quality_issues(hours=4)

@st.cache_data(ttl=5)  # Cache for 5 seconds
def get_dashboard_data():
    """Get cached dashboard data"""
    return DashboardData()

def render_header():
    """Render dashboard header"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("📊 Hybrid Scalper Dashboard")
        st.caption("Real-time scanner monitoring and performance analytics")
    
    with col2:
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.metric("Current Time", current_time)

def render_system_status(data: DashboardData):
    """Render system status overview"""
    st.subheader("🚦 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Overall health
    with col1:
        health_score = data.connection_health.get('healthy_connections', 0) / max(1, data.connection_health.get('total_connections', 1)) * 100
        status_color = "🟢" if health_score > 80 else "🟡" if health_score > 50 else "🔴"
        st.metric(
            f"{status_color} System Health",
            f"{health_score:.0f}%",
            delta=f"{data.connection_health.get('total_uptime_pct', 0):.1f}% uptime"
        )
    
    # Scanner status
    with col2:
        active_symbols = len(data.symbol_priorities)
        st.metric(
            "🔍 Active Symbols",
            active_symbols,
            delta=f"{len([p for p in data.symbol_priorities.values() if p > 50])} high priority"
        )
    
    # Signal performance
    with col3:
        execution_rate = data.performance_metrics['executed_signals'] / max(1, data.performance_metrics['total_signals']) * 100
        st.metric(
            "⚡ Execution Rate", 
            f"{execution_rate:.0f}%",
            delta=f"{data.performance_metrics['avg_confidence']:.2f} avg confidence"
        )
    
    # Data quality
    with col4:
        recent_issues = len([i for i in data.quality_issues if (datetime.now() - datetime.fromtimestamp(i.timestamp)).total_seconds() < 300])
        status = "🟢 Good" if recent_issues == 0 else "🟡 Issues" if recent_issues < 3 else "🔴 Critical"
        st.metric(
            "📊 Data Quality",
            status,
            delta=f"{recent_issues} issues (5min)"
        )

def render_priority_rankings(data: DashboardData):
    """Render symbol priority rankings"""
    st.subheader("🎯 Symbol Priority Rankings")
    
    # Create priority DataFrame
    priority_df = pd.DataFrame([
        {
            'Symbol': symbol,
            'Priority': priority,
            'Tier': 'High' if priority > 70 else 'Medium' if priority > 50 else 'Low'
        }
        for symbol, priority in sorted(data.symbol_priorities.items(), key=lambda x: x[1], reverse=True)
    ])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not PLOTLY_AVAILABLE:
            st.warning("Plotly not available - install with: pip install plotly")
            st.dataframe(priority_df.head(10), use_container_width=True)
        else:
            if PLOTLY_AVAILABLE and px is not None:
                fig = px.bar(
                    priority_df.head(10),
                    x='Priority',
                    y='Symbol',
                    color='Tier',
                    orientation='h',
                    title="Top 10 Symbol Priorities",
                    color_discrete_map={
                        'High': '#ff6b6b',
                        'Medium': '#ffa500', 
                        'Low': '#95a5a6'
                    }
                )
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Priority distribution pie
        if PLOTLY_AVAILABLE and px is not None:
            tier_counts = priority_df['Tier'].value_counts()
            fig = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                title="Priority Distribution",
                color_discrete_map={
                    'High': '#ff6b6b',
                    'Medium': '#ffa500',
                    'Low': '#95a5a6'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            tier_counts = priority_df['Tier'].value_counts()
            st.write("**Priority Distribution:**")
            for tier, count in tier_counts.items():
                st.write(f"- {tier}: {count} symbols")

def render_connection_health(data: DashboardData):
    """Render WebSocket connection health"""
    st.subheader("🌐 Connection Health")
    
    if not data.connection_health.get('connections'):
        st.warning("No active connections to display")
        return
    
    # Connection status table
    connections_data = []
    alert_icons = {'GREEN': '🟢', 'YELLOW': '🟡', 'RED': '🔴'}
    for conn_id, info in data.connection_health['connections'].items():
        alert_level = info.get('alert_level', 'GREEN')
        icon = alert_icons.get(alert_level, '⚪')
        connections_data.append({
            'Connection ID': conn_id,
            'Status': '🟢 Healthy' if info['healthy'] else '🔴 Unhealthy',
            'Alert Level': f"{icon} {alert_level}",
            'Latency (ms)': info['latency_ms'],
            'Message Rate': info['msg_rate'],
            'Uptime': info['uptime'],
            'Drops': info['drops'],
            'Reconnects': info['reconnects']
        })
    
    st.dataframe(
        pd.DataFrame(connections_data),
        use_container_width=True,
        hide_index=True
    )
    
    # Connection metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_latency = data.connection_health.get('avg_latency_ms', 0)
        latency_status = "🟢" if avg_latency < 100 else "🟡" if avg_latency < 200 else "🔴"
        st.metric(f"{latency_status} Average Latency", f"{avg_latency:.1f}ms")
    
    with col2:
        healthy_ratio = data.connection_health.get('healthy_connections', 0) / max(1, data.connection_health.get('total_connections', 1))
        st.metric("🔗 Healthy Connections", f"{healthy_ratio:.0%}")
    
    with col3:
        recent_issues = len([i for i in data.quality_issues if (datetime.now() - datetime.fromtimestamp(i.timestamp)).total_seconds() < 900])
        st.metric("⚠️ Recent Issues", f"{recent_issues} (15min)")

def render_signal_history(data: DashboardData):
    """Render recent signal history"""
    st.subheader("📈 Recent Signals")
    
    if not data.recent_signals:
        st.info("No recent signals to display")
        return
    
    # Convert to DataFrame
    signals_df = pd.DataFrame(data.recent_signals)
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    signals_df = signals_df.sort_values('timestamp', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Signals over time
        if PLOTLY_AVAILABLE and px is not None:
            fig = px.scatter(
                signals_df,
                x='timestamp',
                y='priority',
                color='signal_type',
                size='confidence',
                hover_data=['symbol', 'executed'],
                title="Signals by Priority and Time",
                color_discrete_map={'BUY': '#00ff00', 'SELL': '#ff0000'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("**Recent Signals Timeline:**")
            st.dataframe(signals_df[['timestamp', 'symbol', 'signal_type', 'priority', 'confidence', 'executed']], use_container_width=True)
    
    with col2:
        # Signal summary
        st.markdown("**Signal Summary:**")
        
        total_signals = len(signals_df)
        executed_signals = len(signals_df[signals_df['executed']])
        buy_signals = len(signals_df[signals_df['signal_type'] == 'BUY'])
        
        st.metric("Total Signals", total_signals)
        st.metric("Executed", f"{executed_signals} ({executed_signals/max(1,total_signals)*100:.0f}%)")
        st.metric("Buy/Sell Ratio", f"{buy_signals}:{total_signals-buy_signals}")
    
    # Recent signals table
    display_df = signals_df.head(10).copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
    display_df['confidence'] = display_df['confidence'].round(3)
    display_df['priority'] = display_df['priority'].round(1)
    display_df['executed'] = display_df['executed'].map({True: '✅', False: '❌'})
    
    st.dataframe(
        display_df[['timestamp', 'symbol', 'signal_type', 'confidence', 'priority', 'executed']],
        use_container_width=True,
        hide_index=True
    )

def render_performance_metrics(data: DashboardData):
    """Render performance analytics"""
    st.subheader("📊 Performance Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Key metrics
    with col1:
        win_rate = data.performance_metrics.get('win_rate', 0) * 100
        win_color = "🟢" if win_rate > 55 else "🟡" if win_rate > 50 else "🔴"
        st.metric(f"{win_color} Win Rate", f"{win_rate:.1f}%")
    
    with col2:
        sharpe = data.performance_metrics.get('sharpe_ratio', 0)
        sharpe_color = "🟢" if sharpe > 1.5 else "🟡" if sharpe > 1.0 else "🔴"
        st.metric(f"{sharpe_color} Sharpe Ratio", f"{sharpe:.2f}")
    
    with col3:
        max_dd = data.performance_metrics.get('max_drawdown', 0) * 100
        dd_color = "🟢" if max_dd < 5 else "🟡" if max_dd < 10 else "🔴"
        st.metric(f"{dd_color} Max Drawdown", f"{max_dd:.1f}%")
    
    with col4:
        total_signals = data.performance_metrics.get('total_signals', 0)
        st.metric("🎯 Total Signals", total_signals)
    
    # Performance over time chart
    if data.recent_signals:
        signals_df = pd.DataFrame(data.recent_signals)
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        
        # Hourly signal count
        hourly_signals = signals_df.set_index('timestamp').resample('1H').size().reset_index()
        hourly_signals.columns = ['hour', 'signal_count']
        
        if PLOTLY_AVAILABLE and px is not None:
            fig = px.line(
                hourly_signals,
                x='hour',
                y='signal_count',
                title="Signal Generation Rate (Hourly)",
                markers=True
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("**Signal Generation Rate (Hourly):**")
            st.dataframe(hourly_signals, use_container_width=True)

def render_quality_monitoring(data: DashboardData):
    """Render data quality monitoring"""
    st.subheader("🔍 Data Quality Monitoring")
    
    if not data.quality_issues:
        st.success("🟢 No data quality issues detected")
        return
    
    # Group issues by type and severity
    issues_df = pd.DataFrame([
        {
            'timestamp': datetime.fromtimestamp(issue.timestamp),
            'symbol': issue.symbol,
            'issue_type': issue.issue_type,
            'severity': issue.severity,
            'description': issue.description
        }
        for issue in data.quality_issues[-50:]  # Last 50 issues
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Issues by type
        issue_counts = issues_df['issue_type'].value_counts()
        if PLOTLY_AVAILABLE and px is not None:
            fig = px.bar(
                x=issue_counts.index,
                y=issue_counts.values,
                title="Issues by Type",
                labels={'x': 'Issue Type', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("**Issues by Type:**")
            for issue_type, count in issue_counts.items():
                st.write(f"- {issue_type}: {count}")
    
    with col2:
        # Issues by severity
        severity_counts = issues_df['severity'].value_counts()
        if PLOTLY_AVAILABLE and px is not None:
            colors = {'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#fd7e14', 'CRITICAL': '#dc3545'}
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Issues by Severity",
                color=severity_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("**Issues by Severity:**")
            for severity, count in severity_counts.items():
                st.write(f"- {severity}: {count}")
    
    # Recent issues table
    st.markdown("**Recent Quality Issues:**")
    display_issues = issues_df.tail(10).copy()
    display_issues['timestamp'] = display_issues['timestamp'].dt.strftime('%H:%M:%S')
    
    st.dataframe(
        display_issues[['timestamp', 'symbol', 'issue_type', 'severity', 'description']],
        use_container_width=True,
        hide_index=True
    )

def render_sidebar_controls(data: DashboardData):
    """Render sidebar controls"""
    st.sidebar.header("🎛️ Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 15)
        st.sidebar.info(f"Auto-refreshing every {refresh_interval}s")
    
    # System controls
    st.sidebar.header("📊 System Info")
    
    # Component status
    components = {
        "Scanner": "🟢 Active" if data.symbol_priorities else "🔴 Inactive",
        "Calibration Manager": "🟢 Active" if data.calibration_manager else "🔴 Inactive", 
        "Health Monitor": "🟢 Active",
        "Risk Manager": "🟡 Demo Mode"
    }
    
    for component, status in components.items():
        st.sidebar.markdown(f"**{component}:** {status}")
    
    # Statistics
    st.sidebar.header("📈 Quick Stats")
    if data.connection_health:
        st.sidebar.metric("Active Connections", data.connection_health.get('total_connections', 0))
        st.sidebar.metric("Avg Latency", f"{data.connection_health.get('avg_latency_ms', 0):.0f}ms")
    
    if data.recent_signals:
        st.sidebar.metric("Recent Signals", len(data.recent_signals))
        st.sidebar.metric("Execution Rate", f"{len([s for s in data.recent_signals if s['executed']])/len(data.recent_signals)*100:.0f}%")
    
    # Advanced controls
    with st.sidebar.expander("🔧 Advanced"):
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
        
        if st.button("Export Data"):
            st.download_button(
                "Download JSON",
                json.dumps({
                    'priorities': data.symbol_priorities,
                    'health': data.connection_health,
                    'signals': [s for s in data.recent_signals],
                    'timestamp': datetime.now().isoformat()
                }, indent=2, default=str),
                file_name=f"dashboard_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def main():
    """Main dashboard application"""
    render_header()
    
    # Load data
    try:
        data = get_dashboard_data()
    except Exception as e:
        st.error(f"Failed to load dashboard data: {e}")
        st.stop()
    
    # Render sidebar
    render_sidebar_controls(data)
    
    # Main content
    with st.container():
        render_system_status(data)
        
        st.divider()
        
        # Two-column layout for priority and connections
        col1, col2 = st.columns([3, 2])
        
        with col1:
            render_priority_rankings(data)
        
        with col2:
            render_connection_health(data)
        
        st.divider()
        
        # Signal history and performance
        render_signal_history(data)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            render_performance_metrics(data)
        
        with col2:
            render_quality_monitoring(data)

if __name__ == "__main__":
    main()
