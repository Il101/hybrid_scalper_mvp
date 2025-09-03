from dotenv import load_dotenv; load_dotenv()
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd, yaml
from typing import Optional, Dict
from features.ta_indicators import ta_score, atr_pct, microstructure_score
from features.news_metrics import news_score
from features.sm_metrics import sm_score
from features.microstructure import order_flow_imbalance, bid_ask_pressure
from signals.ensemble import ComponentScores, combine_with_meta, combine_with_volatility_filter
from ml.features_advanced import market_microstructure_score
from exec.timing import optimal_entry_timing, execution_urgency_score
from risk.advanced import kelly_position_sizing, portfolio_heat
from ingest.prices import get_ohlcv, get_ohlcv_cached

with open('config.yaml','r',encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

app = FastAPI(title='Hybrid Scalper MVP')

class SignalResponse(BaseModel):
    symbol: str
    timeframe: str
    score: float
    direction: Optional[str]
    reason: str
    components: Dict[str, float]
    microstructure: Optional[Dict[str, float]] = None
    execution_timing: Optional[str] = None
    position_sizing: Optional[Dict[str, float]] = None

@app.get('/health')
def health(): return {'ok': True}

@app.get('/signal/{symbol}', response_model=SignalResponse)
def signal(symbol: str, tf: str = Query(default=CFG['runtime']['default_timeframe'])):
    # Use cached data for better performance
    df = get_ohlcv_cached(symbol, tf, ttl_seconds=15)  # 15 second cache for scalping
    if df is None or len(df) < 200:
        return SignalResponse(
            symbol=symbol, timeframe=tf, score=0.0, direction=None, 
            reason='not_enough_data', 
            components={'news':0.0,'smart_money':0.0,'ta':0.0}
        )
    
    # Traditional scores
    s_ta = ta_score(df)
    s_news = news_score(symbol)
    s_sm = sm_score(symbol)
    
    # Advanced microstructure analysis
    microstructure_data = market_microstructure_score(df)
    s_micro = microstructure_data['microstructure_score']
    
    # Enhanced TA with microstructure
    s_ta_enhanced = (s_ta + s_micro) / 2.0
    
    # Calculate volatility for filtering
    current_atr = atr_pct(df)
    
    # Enhanced gates with microstructure
    gates = {
        'liquidity': True,
        'regime': True,
        'news_blackout': not CFG['gates'].get('news_blackout', False),
        'microstructure': microstructure_data['scalping_favorability'] != 'unfavorable'
    }
    
    comp = ComponentScores(
        s_news=s_news, 
        s_sm=s_sm, 
        s_ta=s_ta_enhanced, 
        gates=gates
    )
    
    # Use volatility-filtered ensemble
    final = combine_with_volatility_filter(
        comp, 
        current_atr, 
        vol_threshold=0.001,  # 0.1% minimum volatility
        **CFG.get('weights', {}),
        **CFG.get('thresholds', {})
    )
    
    # Execution timing recommendation
    execution_timing = optimal_entry_timing(
        signal_strength=final.score,
        spread_bps=5.0,  # Placeholder - should come from real orderbook
        volatility=current_atr
    )
    
    # Position sizing recommendation
    kelly_size = kelly_position_sizing(
        win_rate=0.55,  # Should come from historical performance
        avg_win=25.0,   # Should be calculated from trades
        avg_loss=18.0   # Should be calculated from trades
    )
    
    return SignalResponse(
        symbol=symbol, 
        timeframe=tf, 
        score=final.score, 
        direction=final.direction, 
        reason=final.reason, 
        components={
            'news': s_news, 
            'smart_money': s_sm, 
            'ta': s_ta,
            'microstructure': s_micro
        },
        microstructure={
            'score': microstructure_data['microstructure_score'],
            'favorability': microstructure_data['scalping_favorability'],
            'optimal_holding_minutes': microstructure_data['optimal_holding_period'],
            'atr_pct': current_atr * 100
        },
        execution_timing=execution_timing,
        position_sizing={
            'kelly_fraction': kelly_size,
            'recommended_risk_pct': min(2.0, kelly_size * 100)  # Cap at 2%
        }
    )

# === Auto retrain (daily + on-startup stale check) ===
import os, threading, time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
from utils.model_utils import artifacts_exist, artifacts_age_seconds, run_retrain

AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "0") == "1"
RETRAIN_DAILY_HOUR = int(os.environ.get("RETRAIN_DAILY_HOUR", "3"))  # 03:00 local tz
RETRAIN_MAX_AGE_H = int(os.environ.get("RETRAIN_MAX_AGE_HOURS", "24"))
LOCAL_TZ = os.environ.get("LOCAL_TZ", "Europe/Vienna")

scheduler: BackgroundScheduler | None = None

def maybe_start_scheduler():
    global scheduler
    if not AUTO_RETRAIN:
        return
    if scheduler is None:
        scheduler = BackgroundScheduler(timezone=LOCAL_TZ)
        # daily at RETRAIN_DAILY_HOUR
        scheduler.add_job(lambda: run_retrain(), 'cron', hour=RETRAIN_DAILY_HOUR, minute=0, id='daily_retrain', replace_existing=True)
        scheduler.start()

@app.on_event("startup")
def startup_hooks():
    # load dotenv already executed earlier
    maybe_start_scheduler()
    # stale-on-startup check (non-blocking)
    if not artifacts_exist() or (artifacts_age_seconds() > RETRAIN_MAX_AGE_H * 3600):
        threading.Thread(target=lambda: run_retrain(), daemon=True).start()
