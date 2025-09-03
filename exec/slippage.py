from typing import Tuple

def estimate_slippage_bps(size_usd: float, spread_bps: float, depth_usd: float) -> float:
    baseline = spread_bps / 2.0
    k = 20.0
    impact = 0.0 if depth_usd <= 0 else (size_usd / (depth_usd + 1.0)) * k
    return float(baseline + impact)


def compute_size_usd(equity_usd: float, risk_pct: float, stop_pct: float, fee_bps: float, slip_bps: float, max_frac: float = 0.5) -> float:
    """
    ATR-based risk sizing:
    - risk_usd = equity * risk_pct
    - stop_eff = stop_pct + (fees+slip)/10000
    - size_usd = risk_usd / stop_eff, capped by max_frac of equity
    """
    risk_usd = max(0.0, equity_usd * max(0.0, risk_pct))
    stop_eff = max(1e-6, stop_pct + (fee_bps + slip_bps)/10000.0)
    size = risk_usd / stop_eff
    return float(max(0.0, min(size, equity_usd * max_frac)))

def depth_usd_from_orderbook(orderbook: dict, side: str = "asks", levels: int = 10) -> float:
    """
    Sum price*qty over top-N levels on the specified side ('asks' for long entry, 'bids' for short).
    Returns USD depth approximation.
    """
    if not orderbook or side not in orderbook:
        return 0.0
    lvls = orderbook[side][:max(1, levels)]
    total = 0.0
    for p, q in lvls:
        try:
            total += float(p) * float(q)
        except Exception:
            continue
    return float(total)

def estimate_slippage_bps_ob(size_usd: float, spread_bps: float, orderbook: dict, side: str, levels: int = 10, k_impact: float = 30.0) -> float:
    """
    Depth-aware slippage:
      slip_bps = half-spread + k_impact * (size_usd / depth_usd_levels)
    where k_impact is a tunable coefficient (empirically 20-50 works reasonably).
    """
    baseline = spread_bps / 2.0
    depth_usd = depth_usd_from_orderbook(orderbook, "asks" if side=="long" else "bids", levels)
    if depth_usd <= 0:
        return baseline + k_impact  # very conservative if no depth
    impact = (size_usd / depth_usd) * k_impact
    return float(baseline + max(0.0, impact))

def compute_size_with_slip(equity_usd: float, risk_pct: float, stop_pct: float,
                           fee_bps: float, spread_bps: float, orderbook: dict, side: str,
                           max_frac: float = 0.5, levels: int = 10, k_impact: float = 30.0) -> Tuple[float, float]:
    """
    Solve for size with depth-aware slippage by 1–2 fixed-point iterations.
    """
    # initial guess: half-spread as slip
    slip_guess = spread_bps / 2.0
    size = compute_size_usd(equity_usd, risk_pct, stop_pct, fee_bps, slip_guess, max_frac)
    # refine once
    slip = estimate_slippage_bps_ob(size, spread_bps, orderbook, side, levels, k_impact)
    size = compute_size_usd(equity_usd, risk_pct, stop_pct, fee_bps, slip, max_frac)
    # one more refinement (optional)
    slip = estimate_slippage_bps_ob(size, spread_bps, orderbook, side, levels, k_impact)
    size = compute_size_usd(equity_usd, risk_pct, stop_pct, fee_bps, slip, max_frac)
    return float(size), float(slip)

def estimate_slippage_with_urgency(size_usd: float, spread_bps: float, 
                                 orderbook: dict, side: str, urgency_factor: float = 1.0,
                                 levels: int = 10, k_impact: float = 30.0) -> float:
    """
    Enhanced slippage with urgency factor for market vs limit orders
    urgency_factor: 1.0 = patient limit, 2.0 = urgent market order
    """
    base_slip = estimate_slippage_bps_ob(size_usd, spread_bps, orderbook, side, levels, k_impact)
    
    # Urgency penalty - market orders pay more slippage
    urgency_penalty = (urgency_factor - 1.0) * spread_bps * 0.5
    
    return float(base_slip + urgency_penalty)
