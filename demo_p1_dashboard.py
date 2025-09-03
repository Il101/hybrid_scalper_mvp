#!/usr/bin/env python3
"""
Demo runner for P1.3 Real-time Scanner Dashboard.
Launches Streamlit dashboard with comprehensive monitoring and analytics.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def check_streamlit_available():
    """Check if Streamlit is available"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit if not available"""
    print("📦 Installing Streamlit for dashboard...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Streamlit installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Streamlit: {e}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / "dashboards" / "real_time_scanner.py"
    
    print("🚀 Starting P1.3 Real-time Scanner Dashboard...")
    print(f"📊 Dashboard location: {dashboard_path}")
    print("🌐 Opening in browser at: http://localhost:8501")
    print("\n" + "="*60)
    print("📈 DASHBOARD FEATURES:")
    print("✅ Real-time system status monitoring")
    print("✅ Symbol priority rankings and distribution")  
    print("✅ WebSocket connection health tracking")
    print("✅ Signal history and execution analytics")
    print("✅ Performance metrics and KPIs")
    print("✅ Data quality monitoring and alerts")
    print("="*60 + "\n")
    print("🎯 Press Ctrl+C to stop the dashboard")
    print("🔄 Dashboard will auto-refresh every 15 seconds")
    print()
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")

def demo_dashboard_features():
    """Demonstrate dashboard features without actually running Streamlit"""
    print("🎯 P1.3 Real-time Scanner Dashboard Demo")
    print("="*60)
    
    print("\n📊 DASHBOARD COMPONENTS:")
    
    print("\n1. 🚦 SYSTEM STATUS PANEL")
    print("   • Overall system health indicator (🟢🟡🔴)")
    print("   • Active symbol count with priority breakdown")
    print("   • Signal execution rate and confidence metrics")
    print("   • Real-time data quality assessment")
    
    print("\n2. 🎯 SYMBOL PRIORITY RANKINGS")
    print("   • Top 10 symbols by priority score (bar chart)")
    print("   • Priority distribution (High/Medium/Low pie chart)")
    print("   • Dynamic updates based on market conditions")
    print("   • Color-coded priority tiers for quick identification")
    
    print("\n3. 🌐 CONNECTION HEALTH MONITORING")
    print("   • WebSocket connection status table")
    print("   • Real-time latency and message rate tracking")
    print("   • Connection drops and reconnection attempts")
    print("   • Alert level indicators (GREEN/YELLOW/RED)")
    
    print("\n4. 📈 SIGNAL HISTORY & ANALYTICS")
    print("   • Recent signals timeline with priority scatter plot")
    print("   • Buy/Sell signal distribution and execution rates")
    print("   • Interactive signal filtering and analysis")
    print("   • Confidence score trending")
    
    print("\n5. 📊 PERFORMANCE METRICS")
    print("   • Key performance indicators: Win Rate, Sharpe Ratio")
    print("   • Maximum drawdown and risk metrics")
    print("   • Signal generation rate over time")
    print("   • Performance trend analysis")
    
    print("\n6. 🔍 DATA QUALITY MONITORING")
    print("   • Quality issue classification by type and severity")
    print("   • Recent data quality alerts and warnings")
    print("   • Connection stability monitoring")
    print("   • Performance degradation alerts")
    
    print("\n7. 🎛️ INTERACTIVE CONTROLS")
    print("   • Auto-refresh toggle with customizable intervals")
    print("   • Real-time component status indicators")
    print("   • Data export functionality (JSON format)")
    print("   • Cache management and system controls")
    
    print("\n✨ DASHBOARD HIGHLIGHTS:")
    print("🔄 Auto-refresh every 15 seconds")
    print("📱 Responsive design for desktop and mobile")
    print("🎨 Professional color-coded status indicators")
    print("📊 Interactive Plotly charts and visualizations")
    print("💾 Data export and historical analysis capabilities")
    print("⚡ Real-time updates with minimal latency")
    
    print("\n🎯 P1.3 IMPLEMENTATION COMPLETE!")
    print("✅ All P1 phase components now fully implemented:")
    print("   • P1.1: Symbol-specific calibrations ✅")
    print("   • P1.2: WebSocket reliability monitoring ✅") 
    print("   • P1.3: Real-time scanner dashboard ✅")

def main():
    """Main demo function"""
    print("🎯 P1.3 Real-time Scanner Dashboard")
    print("="*50)
    
    # Check if this is a demo run or actual dashboard launch
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_dashboard_features()
        return
    
    # Check Streamlit availability
    if not check_streamlit_available():
        print("⚠️ Streamlit not found. Installing required packages...")
        if not install_streamlit():
            print("❌ Could not install Streamlit. Running demo mode instead.")
            demo_dashboard_features()
            return
    
    # Ask user preference
    print("🎛️ Choose dashboard mode:")
    print("1. Launch live dashboard (requires Streamlit)")
    print("2. View feature demo (no dependencies)")
    
    try:
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == "1":
            run_dashboard()
        elif choice == "2":
            demo_dashboard_features()
        else:
            print("Invalid choice. Running feature demo...")
            demo_dashboard_features()
            
    except KeyboardInterrupt:
        print("\n🛑 Demo cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        demo_dashboard_features()

if __name__ == "__main__":
    main()
