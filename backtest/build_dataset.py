from __future__ import annotations
from dotenv import load_dotenv; load_dotenv()
"""
Build an offline feature dataset with auto-labeling for training the meta-model.
- Uses existing ingest + features modules.
- Computes labels: did LONG TP hit before SL within horizon? and same for SHORT.
- Saves Parquet at data/features.parquet
"""
import os, math, argparse, yaml
import pandas as pd
import numpy as np
from ingest.prices import get_ohlcv
from ingest.orderbook_ccxt import fetch_orderbook
from features.ta_indicators import ta_score, atr_pct
from features.news_metrics import news_score
from features.sm_metrics import sm_score
from features.orderflow import compute_spread_bps, compute_obi
from features.regime import detect_regime, regime_id

def label_future(df: pd.DataFrame, take_pct: float, stop_pct: float, horizon: int):
    """
    For each bar t, check in t+1..t+horizon:
      - LONG wins if future_high >= close_t*(1+take_pct) BEFORE future_low <= close_t*(1-stop_pct)
      - SHORT wins if future_low <= close_t*(1-stop_pct) BEFORE future_high >= close_t*(1+take_pct)
    Returns two boolean arrays (as int 0/1): long_win, short_win.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    long_win = np.zeros(n, dtype=int)
    short_win = np.zeros(n, dtype=int)

    for t in range(n - horizon - 1):
        entry = close[t]
        tp = entry * (1.0 + take_pct)
        sl = entry * (1.0 - stop_pct)
        # scan forward
        win_long = 0; win_short = 0
        for k in range(1, horizon+1):
            i = t + k
            if i >= n: break
            # For long: if high hits TP first -> win, if low hits SL first -> loss
            if win_long == 0 and (high[i] >= tp or low[i] <= sl):
                win_long = 1 if high[i] >= tp and not (low[i] <= sl and low[i] < tp) else 0
            # For short: inverse
            tp_s = entry * (1.0 - take_pct)
            sl_s = entry * (1.0 + stop_pct)
            if win_short == 0 and (low[i] <= tp_s or high[i] >= sl_s):
                win_short = 1 if low[i] <= tp_s and not (high[i] >= sl_s and high[i] > tp_s) else 0
            if win_long != 0 or (high[i] <= sl and low[i] >= tp): # continue scanning if ambiguous
                pass
            # Early stop when both resolved
            if (high[i] >= tp or low[i] <= sl) and (low[i] <= tp_s or high[i] >= sl_s):
                break
        long_win[t] = win_long
        short_win[t] = win_short

    return long_win, short_win

def build(symbol="BTCUSDT", tf="5m", horizon=20, take_bps=25, stop_bps=18,
          exchange_for_ob="binance", out_path="data/features.parquet", limit_rows=2000):
    """
    horizon: number of bars ahead to evaluate
    take/stop in basis points (bps)
    """
    df = get_ohlcv(symbol, tf, exchange=exchange_for_ob, market_type="futures", limit=2000)
    if limit_rows and len(df) > limit_rows:
        df = df.iloc[-limit_rows:].copy()

    # Expensive external features: compute once per dataset build (constant over window)
    # This avoids O(n) HTTP calls that can hit rate limits and make retraining unbearably slow.
    try:
        s_news_global = news_score(symbol)
    except Exception:
        s_news_global = 50.0
    try:
        s_sm_global = sm_score(symbol)
    except Exception:
        s_sm_global = 50.0

    # Orderbook snapshot: fetch sparsely to avoid rate limits; reuse last snapshot between samples
    FETCH_OB_EVERY = 100  # fetch once every N samples
    last_ob = None

    rows = []
    for i in range(200, len(df)-horizon-1):
        # CRITICAL FIX: Use only historical data up to (but not including) bar i
        # Previously included bar i which creates look-ahead bias
        window = df.iloc[max(0, i-200):i].copy()  # Use last 200 bars up to current point
        ts = df.index[i]  # timestamp of the bar we're predicting for

        s_ta = ta_score(window)
        s_news = s_news_global
        s_sm = s_sm_global
        atrp = atr_pct(window)

        # optional orderbook
        ob = None
        try:
            if (i - 200) % FETCH_OB_EVERY == 0:
                last_ob = fetch_orderbook(exchange_for_ob, symbol.replace("USDT", "/USDT"))
            ob = last_ob
        except Exception:
            ob = last_ob  # keep previous if fetch fails
        spread_bps = compute_spread_bps(ob)
        obi = compute_obi(ob)

        regime = detect_regime(window)
        rid = regime_id(regime)

        rows.append({
            "timestamp": ts,
            "symbol": symbol,
            "tf": tf,
            "S_news": s_news,
            "S_sm": s_sm,
            "S_ta": s_ta,
            "spread_bps": spread_bps,
            "atr_pct": atrp,
            "obi": obi,
            "regime_id": rid,
            # Store index for later label alignment
            "df_index": i,
        })

    feat = pd.DataFrame(rows).reset_index(drop=True)

    # Compute labels using only future data (no look-ahead bias)
    long_win, short_win = label_future(df, take_pct=take_bps/10000.0, stop_pct=stop_bps/10000.0, horizon=horizon)
    
    # Align labels to features using stored df_index
    feat["label_long_win"] = [long_win[idx] for idx in feat["df_index"]]
    feat["label_short_win"] = [short_win[idx] for idx in feat["df_index"]]
    # default label: long win (можешь потом выбрать нужное при тренинге)
    feat["label_win"] = feat["label_long_win"]

    # Remove the index column used for alignment
    feat = feat.drop(columns=["df_index"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feat.to_parquet(out_path, index=False)
    return {"rows": len(feat), "out": out_path}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--take_bps", type=int, default=25)
    ap.add_argument("--stop_bps", type=int, default=18)
    ap.add_argument("--exchange_for_ob", default="binance")
    ap.add_argument("--out", default="data/features.parquet")
    ap.add_argument("--limit_rows", type=int, default=2000)
    args = ap.parse_args()
    res = build(args.symbol, args.tf, args.horizon, args.take_bps, args.stop_bps, args.exchange_for_ob, args.out, args.limit_rows)
    print(res)
