from __future__ import annotations
import os, time, subprocess, shlex, sys
from typing import Tuple

ART_MODEL = "model/artifacts/meta_model.pkl"
ART_CALIB = "model/artifacts/calibrator.pkl"

def artifacts_exist() -> bool:
    return os.path.exists(ART_MODEL) and os.path.exists(ART_CALIB)

def artifacts_age_seconds() -> float:
    if not artifacts_exist():
        return 1e18
    t1 = os.path.getmtime(ART_MODEL)
    t2 = os.path.getmtime(ART_CALIB)
    return time.time() - min(t1, t2)

def validate_artifacts() -> bool:
    """Validate that artifacts are loadable and have required methods"""
    try:
        import joblib
        model = joblib.load(ART_MODEL)
        calib = joblib.load(ART_CALIB)
        return hasattr(model, 'predict') and hasattr(calib, 'predict_proba')
    except Exception:
        return False

def run_retrain(symbol="BTCUSDT", tf="5m", exchange="bybit", limit=2000, horizon=20, take_bps=25, stop_bps=18) -> int:
    """
    Runs the automated retrain pipeline (dataset -> train -> enable) as a subprocess.
    Returns exit code.
    """
    py = shlex.quote(sys.executable or "python3")
    cmd = (
        f"{py} -m model.retrain --symbol {shlex.quote(symbol)} --tf {shlex.quote(tf)} "
        f"--exchange {shlex.quote(exchange)} --limit {int(limit)} "
        f"--horizon {int(horizon)} --take_bps {int(take_bps)} --stop_bps {int(stop_bps)}"
    )
    return subprocess.call(cmd, shell=True)

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file
    Returns config dictionary or default config if file doesn't exist
    """
    import yaml
    
    default_config = {
        'trading': {
            'take_profit_bps': 25,
            'stop_loss_bps': 18,
            'max_position_size': 1000,
            'risk_per_trade': 0.02
        },
        'data': {
            'symbol': 'BTCUSDT',
            'timeframe': '5m',
            'lookback_bars': 100
        },
        'model': {
            'retrain_interval_hours': 24,
            'min_accuracy': 0.55
        }
    }
    
    if not os.path.exists(config_path):
        return default_config
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if config else default_config
    except Exception:
        return default_config
