from dataclasses import dataclass
from typing import Optional

@dataclass
class ComponentScores:
    s_news: float
    s_sm: float
    s_ta: float
    gates: dict

@dataclass
class FinalSignal:
    score: float
    direction: Optional[str]
    reason: str

def combine(scores: ComponentScores,
            w_news=0.35, w_sm=0.40, w_ta=0.25,
            long_thresh=65, short_thresh=35) -> FinalSignal:
    if not all(scores.gates.values()):
        return FinalSignal(0.0, None, f"Gate blocked: {scores.gates}")
    total = (w_news*scores.s_news + w_sm*scores.s_sm + w_ta*scores.s_ta)
    if total >= long_thresh:
        return FinalSignal(float(total), "long", f"news={scores.s_news:.0f}, sm={scores.s_sm:.0f}, ta={scores.s_ta:.0f}")
    if total <= short_thresh:
        return FinalSignal(float(total), "short", f"news={scores.s_news:.0f}, sm={scores.s_sm:.0f}, ta={scores.s_ta:.0f}")
    return FinalSignal(float(total), None, f"flat: news={scores.s_news:.0f}, sm={scores.s_sm:.0f}, ta={scores.s_ta:.0f}")

def combine_with_meta(df, scores: ComponentScores, config: dict) -> FinalSignal:
    # Simplified: call combine() by default (meta disabled in this minimal rebuild)
    return combine(scores,
                   w_news=config['weights']['news'],
                   w_sm=config['weights']['smart_money'],
                   w_ta=config['weights']['ta'],
                   long_thresh=config['thresholds']['long'],
                   short_thresh=config['thresholds']['short'])

def combine_with_volatility_filter(scores: ComponentScores, atr_pct: float, 
                                 vol_threshold: float = 0.002,
                                 w_news=0.35, w_sm=0.40, w_ta=0.25,
                                 long_thresh=65, short_thresh=35) -> FinalSignal:
    """Enhanced ensemble with volatility filter for scalping"""
    if atr_pct < vol_threshold:
        return FinalSignal(0.0, None, f"Low volatility: {atr_pct:.4f}")
    
    if not all(scores.gates.values()):
        return FinalSignal(0.0, None, f"Gate blocked: {scores.gates}")
    
    # Volatility boost - higher vol = higher confidence
    vol_multiplier = min(2.0, atr_pct / vol_threshold)
    
    total = (w_news*scores.s_news + w_sm*scores.s_sm + w_ta*scores.s_ta) * vol_multiplier
    
    if total >= long_thresh:
        return FinalSignal(float(total), "long", f"vol_adj: news={scores.s_news:.0f}, sm={scores.s_sm:.0f}, ta={scores.s_ta:.0f}, atr={atr_pct:.4f}")
    if total <= short_thresh:
        return FinalSignal(float(total), "short", f"vol_adj: news={scores.s_news:.0f}, sm={scores.s_sm:.0f}, ta={scores.s_ta:.0f}, atr={atr_pct:.4f}")
    return FinalSignal(float(total), None, f"flat: vol_adj total={total:.1f}, atr={atr_pct:.4f}")
