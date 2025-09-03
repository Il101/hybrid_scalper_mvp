import pandas as pd

def detect_regime(df: pd.DataFrame) -> str:
    if len(df) < 200:
        return "range"
    close = df["close"]
    ema50 = close.ewm(span=50).mean().iloc[-1]
    ema200 = close.ewm(span=200).mean().iloc[-1]
    slope50 = close.ewm(span=50).mean().diff().iloc[-5:].mean()
    tr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    vol_norm = close.iloc[-1] * 0.01

    if tr > 3 * vol_norm:
        return "storm"
    if abs((ema50 - ema200) / max(1e-9, ema200)) < 0.002:
        return "range"
    if ema50 > ema200 and slope50 > 0:
        return "trend"
    return "range"

def regime_id(name: str) -> int:
    return {"range":0,"trend":1,"storm":2}.get(name, 0)
