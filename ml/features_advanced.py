"""
Advanced machine learning features for scalping
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def volatility_regime_features(df: pd.DataFrame, lookback_short: int = 20, 
                              lookback_long: int = 100) -> Dict[str, Any]:
    """
    GARCH-style volatility clustering and regime detection
    
    Args:
        df: OHLCV DataFrame
        lookback_short: Short-term volatility window
        lookback_long: Long-term volatility window
    
    Returns:
        Dictionary with volatility regime features
    """
    if len(df) < lookback_long:
        return {
            "vol_regime": "unknown",
            "vol_percentile": 0.5,
            "vol_clustering": 0.0,
            "vol_mean_reversion": 0.0
        }
    
    # Calculate returns
    returns = df['close'].pct_change().dropna()
    
    # Short and long-term volatility
    vol_short = returns.rolling(lookback_short).std()
    vol_long = returns.rolling(lookback_long).std()
    
    current_vol_short = vol_short.iloc[-1]
    current_vol_long = vol_long.iloc[-1]
    
    # Volatility regime classification
    vol_ratio = current_vol_short / (current_vol_long + 1e-9)
    
    if vol_ratio > 1.5:
        vol_regime = "high"
    elif vol_ratio < 0.7:
        vol_regime = "low"
    else:
        vol_regime = "normal"
    
    # Volatility percentile (current vs historical)
    vol_long_values = vol_long.dropna().values
    if len(vol_long_values) > 0:
        vol_percentile = np.sum(vol_long_values <= current_vol_long) / len(vol_long_values)
    else:
        vol_percentile = 0.5
    
    # Volatility clustering (autocorrelation of volatility)
    vol_squared = returns.rolling(5).std() ** 2
    vol_clustering = vol_squared.autocorr(lag=1) if len(vol_squared) > 10 else 0.0
    
    # Mean reversion tendency
    vol_deviations = (vol_short - vol_long) / (vol_long + 1e-9)
    vol_mean_reversion = -vol_deviations.autocorr(lag=1) if len(vol_deviations) > 10 else 0.0
    
    return {
        "vol_regime": vol_regime,
        "vol_percentile": float(vol_percentile),
        "vol_clustering": float(vol_clustering) if not np.isnan(vol_clustering) else 0.0,
        "vol_mean_reversion": float(vol_mean_reversion) if not np.isnan(vol_mean_reversion) else 0.0,
        "vol_ratio": float(vol_ratio)
    }

def momentum_decay_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how momentum fades over different timeframes
    
    Returns:
        Dictionary with momentum decay characteristics
    """
    if len(df) < 30:
        return {
            "momentum_decay": 1.0,
            "momentum_persistence": 0.0,
            "momentum_strength": 0.0
        }
    
    # Calculate momentum over different horizons
    mom_1 = df['close'].pct_change(1).iloc[-1]
    mom_3 = df['close'].pct_change(3).iloc[-1] if len(df) >= 3 else mom_1
    mom_5 = df['close'].pct_change(5).iloc[-1] if len(df) >= 5 else mom_1
    mom_10 = df['close'].pct_change(10).iloc[-1] if len(df) >= 10 else mom_1
    
    # Momentum decay factor
    if abs(mom_10) > 1e-9:
        momentum_decay = abs(mom_1) / (abs(mom_10) + 1e-9)
    else:
        momentum_decay = 1.0
    
    # Momentum persistence (correlation across timeframes)
    momentums = [mom_1, mom_3, mom_5, mom_10]
    momentum_signs = [1 if m > 0 else -1 if m < 0 else 0 for m in momentums]
    momentum_persistence = abs(np.mean(momentum_signs))
    
    # Overall momentum strength
    momentum_strength = np.mean([abs(m) for m in momentums])
    
    return {
        "momentum_decay": float(momentum_decay),
        "momentum_persistence": float(momentum_persistence),
        "momentum_strength": float(momentum_strength),
        "momentum_1min": float(mom_1),
        "momentum_consistency": float(np.std(momentums) / (abs(np.mean(momentums)) + 1e-9))
    }

def price_action_patterns(df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    """
    Detect price action patterns relevant for scalping
    """
    if len(df) < window:
        return {
            "pattern_strength": 0.0,
            "breakout_potential": 0.0,
            "support_resistance": 0.0
        }
    
    recent = df.tail(window)
    
    # Higher highs, higher lows pattern
    highs = recent['high'].to_numpy()
    lows = recent['low'].to_numpy()
    
    # Check for trending patterns
    high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
    low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
    
    # Pattern strength based on trend consistency
    if high_trend > 0 and low_trend > 0:
        pattern_strength = min(high_trend, low_trend) / recent['close'].iloc[-1]
        pattern_type = "uptrend"
    elif high_trend < 0 and low_trend < 0:
        pattern_strength = abs(max(high_trend, low_trend)) / recent['close'].iloc[-1]
        pattern_type = "downtrend"
    else:
        pattern_strength = 0.0
        pattern_type = "sideways"
    
    # Breakout potential (price near bounds)
    recent_high = recent['high'].max()
    recent_low = recent['low'].min()
    current_price = recent['close'].iloc[-1]
    
    price_range = recent_high - recent_low
    if price_range > 0:
        high_proximity = (recent_high - current_price) / price_range
        low_proximity = (current_price - recent_low) / price_range
        breakout_potential = 1.0 - min(high_proximity, low_proximity)
    else:
        breakout_potential = 0.0
    
    # Support/resistance strength
    support_tests = sum(1 for low in lows if abs(low - recent_low) / recent_low < 0.005)
    resistance_tests = sum(1 for high in highs if abs(high - recent_high) / recent_high < 0.005)
    support_resistance = (support_tests + resistance_tests) / len(recent)
    
    return {
        "pattern_strength": float(pattern_strength * 1000),  # Scale for readability
        "pattern_type": pattern_type,
        "breakout_potential": float(breakout_potential),
        "support_resistance": float(support_resistance),
        "price_position": float((current_price - recent_low) / (price_range + 1e-9))
    }

def liquidity_features(df: pd.DataFrame, volume_window: int = 20) -> Dict[str, Any]:
    """
    Extract liquidity-related features for scalping
    """
    if len(df) < volume_window:
        return {
            "liquidity_score": 50.0,
            "volume_trend": 0.0,
            "volume_volatility": 0.0
        }
    
    recent = df.tail(volume_window)
    
    # Volume trend
    volumes = recent['volume'].to_numpy()
    volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
    volume_trend_normalized = volume_trend / (np.mean(volumes) + 1e-9)
    
    # Volume volatility (consistency of volume)
    volume_cv = np.std(volumes) / (np.mean(volumes) + 1e-9)
    
    # Current vs average volume
    current_volume = recent['volume'].iloc[-1]
    avg_volume = np.mean(volumes)
    volume_surge = current_volume / (avg_volume + 1e-9)
    
    # Liquidity score (higher = better for scalping)
    liquidity_score = 50.0  # Base score
    liquidity_score += min(20.0, float(volume_surge * 10))  # Volume surge bonus
    liquidity_score -= min(20.0, float(volume_cv * 50))  # Penalty for erratic volume
    liquidity_score = max(0.0, min(100.0, liquidity_score))
    
    return {
        "liquidity_score": float(liquidity_score),
        "volume_trend": float(volume_trend_normalized),
        "volume_volatility": float(volume_cv),
        "volume_surge": float(volume_surge),
        "volume_consistency": float(1.0 / (1.0 + volume_cv))
    }

def market_microstructure_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Combined microstructure score for scalping opportunities
    """
    if len(df) < 50:
        return {
            "microstructure_score": 50.0,
            "scalping_favorability": "unknown",
            "optimal_holding_period": 5.0
        }
    
    # Get individual feature sets
    vol_features = volatility_regime_features(df)
    momentum_features = momentum_decay_features(df)
    pattern_features = price_action_patterns(df)
    liquidity_features_dict = liquidity_features(df)
    
    # Combine into overall score
    score = 50.0  # Base score
    
    # Volatility component (moderate vol is good for scalping)
    if vol_features['vol_regime'] == 'normal':
        score += 10
    elif vol_features['vol_regime'] == 'high':
        score += 5  # Some vol is good, too much is bad
    else:
        score -= 5  # Low vol is less favorable
    
    # Momentum component
    if momentum_features['momentum_persistence'] > 0.7:
        score += 15  # Strong consistent momentum
    elif momentum_features['momentum_persistence'] > 0.3:
        score += 5   # Some momentum
    else:
        score -= 10  # No clear direction
    
    # Pattern component
    score += pattern_features['breakout_potential'] * 10
    
    # Liquidity component
    score += (liquidity_features_dict['liquidity_score'] - 50) * 0.3
    
    # Final score normalization
    microstructure_score = max(0.0, min(100.0, score))
    
    # Determine scalping favorability
    if microstructure_score > 70:
        scalping_favorability = "favorable"
        optimal_holding_period = 2.0  # 2 minutes
    elif microstructure_score > 50:
        scalping_favorability = "neutral"
        optimal_holding_period = 5.0  # 5 minutes
    else:
        scalping_favorability = "unfavorable"
        optimal_holding_period = 10.0  # 10 minutes
    
    return {
        "microstructure_score": float(microstructure_score),
        "scalping_favorability": scalping_favorability,
        "optimal_holding_period": float(optimal_holding_period),
        "component_scores": {
            "volatility": vol_features,
            "momentum": momentum_features,
            "patterns": pattern_features,
            "liquidity": liquidity_features_dict
        }
    }

def feature_vector_for_ml(df: pd.DataFrame) -> np.ndarray:
    """
    Create feature vector for machine learning models
    
    Returns:
        numpy array with normalized features
    """
    try:
        # Get all feature components
        vol_features = volatility_regime_features(df)
        momentum_features = momentum_decay_features(df)
        pattern_features = price_action_patterns(df)
        liquidity_features_dict = liquidity_features(df)
        
        # Combine into feature vector
        features = [
            vol_features['vol_percentile'],
            vol_features['vol_clustering'],
            vol_features['vol_mean_reversion'],
            vol_features['vol_ratio'],
            
            momentum_features['momentum_decay'],
            momentum_features['momentum_persistence'],
            momentum_features['momentum_strength'],
            momentum_features['momentum_consistency'],
            
            pattern_features['pattern_strength'],
            pattern_features['breakout_potential'],
            pattern_features['support_resistance'],
            pattern_features['price_position'],
            
            liquidity_features_dict['liquidity_score'] / 100.0,  # Normalize
            liquidity_features_dict['volume_trend'],
            liquidity_features_dict['volume_volatility'],
            liquidity_features_dict['volume_surge'],
        ]
        
        # Handle any NaN values
        features = [f if not np.isnan(f) else 0.0 for f in features]
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        # Return default feature vector if calculation fails
        return np.zeros(16, dtype=np.float32)
