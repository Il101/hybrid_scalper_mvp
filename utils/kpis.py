from __future__ import annotations
import pandas as pd
import numpy as np

def equity_curve_from_trades(trades_csv: str) -> pd.DataFrame:
    df = pd.read_csv(trades_csv)
    if df.empty:
        return pd.DataFrame(columns=["ts","equity"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    # Take rows where equity is updated (OPEN logs also have equity; we cumulative forward-fill)
    df = df.sort_values("ts")
    eq = df[["ts","equity"]].copy()
    eq["equity"] = eq["equity"].ffill()
    return eq

def kpis_from_trades(trades_csv: str) -> dict:
    df = pd.read_csv(trades_csv)
    if df.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "pnl_total": 0.0, "max_dd": 0.0}
    # Consider only CLOSE rows for PnL
    closed = df[df["action"]=="CLOSE"].copy()
    pnl = closed["pnl_usd"].sum()
    wins = closed[closed["pnl_usd"] > 0]["pnl_usd"].sum()
    losses = -closed[closed["pnl_usd"] < 0]["pnl_usd"].sum()
    win_rate = (len(closed[closed["pnl_usd"] > 0]) / max(1, len(closed))) * 100.0
    profit_factor = (wins / max(1e-9, losses)) if losses > 0 else float("inf")
    # Max drawdown from equity series
    eq = equity_curve_from_trades(trades_csv)
    if len(eq) == 0:
        max_dd = 0.0
    else:
        e = eq["equity"].values.astype(float)
        peak = np.maximum.accumulate(e)
        dd = (e - peak) / peak
        max_dd = float(dd.min() * 100.0)
    return {
        "trades": int(len(closed)),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor if profit_factor!=float("inf") else 999.0),
        "pnl_total": float(pnl),
        "max_dd": float(max_dd)
    }

def sharpe_ratio(trades_csv: str, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio from equity curve"""
    eq = equity_curve_from_trades(trades_csv)
    if len(eq) < 2: 
        return 0.0
    returns = eq["equity"].pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess = returns.mean() - risk_free_rate/252
    return float(excess / max(returns.std(), 1e-9) * np.sqrt(252))

def avg_trade_duration_minutes(trades_csv: str) -> float:
    """Calculate average trade duration in minutes"""
    df = pd.read_csv(trades_csv)
    if df.empty:
        return 0.0
    
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    opens = df[df["action"]=="OPEN"].copy()
    closes = df[df["action"]=="CLOSE"].copy()
    
    durations = []
    for _, close_row in closes.iterrows():
        # Find matching open by position_id if available, or by closest timestamp
        if "position_id" in df.columns:
            matching_open = opens[opens["position_id"] == close_row.get("position_id")]
        else:
            # Fallback: match by symbol and closest prior timestamp
            symbol_opens = opens[opens["symbol"] == close_row["symbol"]]
            matching_open = symbol_opens[symbol_opens["ts"] <= close_row["ts"]]
        
        if not matching_open.empty:
            open_time = matching_open.iloc[-1]["ts"]  # Take latest matching open
            duration_min = (close_row["ts"] - open_time).total_seconds() / 60.0
            durations.append(duration_min)
    
    return float(np.mean(durations)) if durations else 0.0

def implementation_shortfall_bps(trades_csv: str) -> float:
    """Measure slippage from decision price to execution price"""
    df = pd.read_csv(trades_csv)
    if df.empty or "decision_price" not in df.columns:
        return 0.0
    
    shortfalls = []
    for _, row in df.iterrows():
        if pd.notna(row.get("decision_price")) and pd.notna(row.get("fill_price")):
            decision_price = float(row["decision_price"])
            fill_price = float(row["fill_price"])
            shortfall_bps = abs(fill_price - decision_price) / decision_price * 10000
            shortfalls.append(shortfall_bps)
    
    return float(np.mean(shortfalls)) if shortfalls else 0.0
