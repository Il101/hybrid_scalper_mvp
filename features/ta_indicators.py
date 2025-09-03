import pandas as pd
import yaml
import os

def _load_ta_config():
    """Load TA indicator configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('ta_indicators', {})
    except Exception:
        # Return defaults if config loading fails
        return {
            'ema_short_period': 50,
            'ema_long_period': 200,
            'rsi_period': 14,
            'atr_period': 14,
            'vwap_window': 20,
            'momentum_window': 10,
            'vol_surge_window': 20,
            'trend_bullish_score': 70,
            'trend_bearish_score': 30,
            'momentum_neutral_range': [45, 65],
            'momentum_extended_range': [35, 75],
            'momentum_good_score': 70,
            'momentum_neutral_score': 55,
            'momentum_poor_score': 40,
            'vol_good_score': 60,
            'vol_poor_score': 40
        }

def ta_score(df: pd.DataFrame) -> float:
    config = _load_ta_config()
    
    min_length = max(config['ema_long_period'], config['rsi_period'] * 2)
    if len(df) < min_length:
        return 0.0
        
    close = df['close']
    
    # Trend analysis with configurable periods
    ema_short = close.ewm(span=config['ema_short_period']).mean().iloc[-1]
    ema_long = close.ewm(span=config['ema_long_period']).mean().iloc[-1]
    trend = config['trend_bullish_score'] if ema_short > ema_long else config['trend_bearish_score']
    
    # RSI momentum with configurable period
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/config['rsi_period']).mean().iloc[-1]
    down = (-delta.clip(upper=0)).ewm(alpha=1/config['rsi_period']).mean().iloc[-1] + 1e-9
    rs = up/down
    rsi = 100 - (100/(1+rs))
    
    # Configurable momentum scoring
    neutral_min, neutral_max = config['momentum_neutral_range']
    extended_min, extended_max = config['momentum_extended_range']
    
    if neutral_min < rsi < neutral_max:
        momentum = config['momentum_good_score']
    elif extended_min < rsi <= extended_max:
        momentum = config['momentum_neutral_score']
    else:
        momentum = config['momentum_poor_score']
    
    # Volatility analysis with configurable ATR period
    tr = (df['high']-df['low']).rolling(config['atr_period']).mean().iloc[-1]
    vol_norm = close.iloc[-1]*0.01
    vol_score = config['vol_good_score'] if tr < 2*vol_norm else config['vol_poor_score']
    
    return float(min(100, max(0, 0.5*trend+0.3*momentum+0.2*vol_score)))

def atr_pct(df: pd.DataFrame) -> float:
    config = _load_ta_config()
    atr_period = config.get('atr_period', 14)
    
    if len(df) < atr_period + 1:
        return 0.0
    tr = (df["high"] - df["low"]).rolling(atr_period).mean().iloc[-1]
    c = df["close"].iloc[-1]
    return float(tr / max(c, 1e-9))

def microstructure_score(df: pd.DataFrame) -> float:
    """Volume-price divergence and intrabar momentum for scalping"""
    config = _load_ta_config()
    vwap_window = config.get('vwap_window', 20)
    momentum_window = config.get('momentum_window', 10)
    vol_surge_window = config.get('vol_surge_window', 20)
    
    if len(df) < max(vwap_window, momentum_window, vol_surge_window) + 5: 
        return 50.0
    
    # Volume-weighted average price with configurable window
    vwap = (df['volume'] * df['close']).rolling(vwap_window).sum() / df['volume'].rolling(vwap_window).sum()
    price_vs_vwap = (df['close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
    
    # Intrabar momentum with configurable window
    body_pct = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
    momentum_score = body_pct.rolling(momentum_window).mean().iloc[-1] * 100
    
    # Volume surge detection with configurable window
    avg_vol = df['volume'].rolling(vol_surge_window).mean().iloc[-1]
    current_vol = df['volume'].iloc[-1]
    vol_surge = current_vol / (avg_vol + 1e-9)
    
    final_score = 50 + price_vs_vwap * 1000 + momentum_score * 0.5 + min(20, vol_surge * 5)
    return float(max(0, min(100, final_score)))

def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)"""
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = (-delta.clip(upper=0))
    
    # Используем EWM для расчета RSI
    gain = up.ewm(alpha=1/window).mean()
    loss = down.ewm(alpha=1/window).mean()
    
    rs = gain / (loss + 1e-10)  # Избегаем деления на ноль
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def compute_macd(data, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Compute MACD (Moving Average Convergence Divergence)"""
    # Handle both DataFrame and Series input
    if isinstance(data, pd.DataFrame):
        price_series = data['close']
    else:
        price_series = data
        
    exp1 = price_series.ewm(span=fast).mean()
    exp2 = price_series.ewm(span=slow).mean()
    
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def compute_bollinger_bands(data, window: int = 20, num_std: float = 2) -> tuple:
    """Compute Bollinger Bands"""
    # Handle both DataFrame and Series input
    if isinstance(data, pd.DataFrame):
        price_series = data['close']
    else:
        price_series = data
        
    sma = price_series.rolling(window=window).mean()
    std = price_series.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return upper_band, sma, lower_band

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)"""
    high = df['high']
    low = df['low']  
    close = df['close']
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr
