"""
tp_sl_retracement_phase2fixed_1_25.py (2026-03-04: Phase 2-FIXED variant at 1.25:1 RR)

TP/SL calculation for Phase 2-FIXED A/B test variant.
- ATR-based 1.25:1 Risk:Reward ratio (aggressive, ultra-tight targets)
- TP = Entry ± (1.25 × ATR)
- SL = Entry ± (1.0 × ATR)
- Always returns achieved_rr = 1.25

PHASE 2-FIXED STRATEGY:
- Deploy in parallel with 1.5:1 RR variant (main.py)
- Run 7-day A/B test (Mar 4-10 2026)
- Monitor WR improvement: target ≥ 51% with 1.25:1
- If successful, upgrade main daemon to 1.25:1

HYPOTHESIS:
- Simulation shows: +19.35pp WR, +$40,118 P&L improvement at 1.25:1
- Expected outcome: 51%+ WR, $87k+ total P&L
- Risk: Aggressive targets may miss some quality signals
- Mitigation: Run parallel, don't abandon 1.5:1 if 1.25:1 underperforms

NOTE: Refactored to use consolidated calculations from calculations.py
to avoid duplication of ATR and TP/SL logic.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Import consolidated calculation functions
from calculations import calculate_atr_for_tp_sl, calculate_tp_sl_from_atr

DEFAULT_ATR_LOOKBACK = int(os.getenv("TP_SL_LOOKBACK", "14"))
DEFAULT_ATR_MULT_TP = float(os.getenv("TP_SL_ATR_MULT_TP_PHASE2FIXED", "1.25"))
DEFAULT_ATR_MULT_SL = float(os.getenv("TP_SL_ATR_MULT_SL", "1.0"))


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert value to float."""
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        try:
            return float(str(v))
        except Exception:
            return default


def calculate_tp_sl(df: pd.DataFrame, entry_price: float, direction: str,
                    atr_multiplier: Optional[float] = None, lookback: Optional[int] = None,
                    regime: Optional[str] = None) -> Dict[str, Any]:
    """
    ATR-Based 1.25:1 RR TP/SL Calculation (PHASE 2-FIXED VARIANT - 2026-03-04).
    TP = Entry ± (1.25 × ATR)
    SL = Entry ± (1.0 × ATR)
    
    OPTION C: For RANGE regime, increase TP multiplier by 1.5x:
    TP = Entry ± (1.875 × ATR)  [1.25 × 1.5 = 1.875]
    SL = Entry ± (1.0 × ATR)  [unchanged - keep same risk]
    RR becomes 1.875:1 instead of 1.25:1
    
    PHASE 2-FIXED STRATEGY:
    - Ultra-tight targets for maximum WR
    - Simulation: 51.4% WR, $87k P&L
    - Run parallel to 1.5:1 main variant
    - Validate 7 days, decide Mar 10
    
    REFACTORED: Uses consolidated functions from calculations.py
    """
    if lookback is None:
        try:
            lookback = int(os.getenv("TP_SL_LOOKBACK", str(DEFAULT_ATR_LOOKBACK)))
        except Exception:
            lookback = DEFAULT_ATR_LOOKBACK

    atr_mult_tp = DEFAULT_ATR_MULT_TP
    atr_mult_sl = DEFAULT_ATR_MULT_SL

    entry_f = _safe_float(entry_price, None)
    if entry_f is None:
        raise ValueError("entry_price must be numeric")

    if df is None or len(df) == 0:
        raise ValueError("df must be a non-empty DataFrame")
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            raise ValueError("Unable to coerce df to DataFrame")

    cols = set(df.columns)
    if not {'high', 'low', 'close'}.issubset(cols):
        if df.shape[1] >= 4:
            df = df.copy()
            col_names = list(df.columns[:4])
            df = df.rename(columns={col_names[0]: 'open', col_names[1]: 'high', col_names[2]: 'low', col_names[3]: 'close'})
        else:
            raise ValueError("DataFrame must contain 'high','low','close' columns")

    # Convert to numeric
    high = pd.to_numeric(df['high'], errors='coerce').astype(float)
    low = pd.to_numeric(df['low'], errors='coerce').astype(float)
    close = pd.to_numeric(df['close'], errors='coerce').astype(float)
    df_clean = df.copy()
    df_clean['high'] = high
    df_clean['low'] = low
    df_clean['close'] = close

    # Use consolidated ATR calculation
    atr = calculate_atr_for_tp_sl(df_clean, entry_f, lookback)

    # Use consolidated TP/SL calculation from ATR
    # Pass regime for Option C: RANGE trades get 3:1 RR instead of 2:1
    result = calculate_tp_sl_from_atr(entry_f, atr, direction, atr_mult_tp, atr_mult_sl, regime=regime)

    try:
        print(
            f"[tp_sl_retracement_phase2fixed_1_25] ATR_1_25_1_RR | direction={direction} entry={entry_f:.8f} "
            f"atr={atr:.8f} tp={result['tp']:.8f} sl={result['sl']:.8f} rr={result['achieved_rr']}",
            flush=True
        )
    except Exception:
        pass

    return result
