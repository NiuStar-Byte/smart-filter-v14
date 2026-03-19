"""
tp_sl_retracement.py (2026-03-19 PHASE 3 FIX: 1.25:1 Fallback RR + 2.5:1 Market-Driven Cap)

TP/SL calculation for Smart Filter.
- Market-driven S&R PREFERRED: Uses actual support/resistance levels
- Market-driven RR CAP: Maximum 2.5:1 (capped if higher)
- ATR-based FALLBACK: 1.25:1 Risk:Reward ratio (when market structure unavailable)
- TP = Entry ± (1.25 × ATR)
- SL = Entry ± (1.0 × ATR)

PHASE 3 Fixes:
- Changed fallback from 1.5:1 to 1.25:1 (user specification)
- Added 2.5:1 cap to market-driven RR
- PEC Executor now uses historical close at timeout (not current price)

NOTE: Refactored to use consolidated calculations from calculations.py
to avoid duplication of ATR and TP/SL logic.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Import consolidated calculation functions
from calculations import calculate_atr_for_tp_sl, calculate_tp_sl_from_atr, calculate_tp_sl_from_df

DEFAULT_ATR_LOOKBACK = int(os.getenv("TP_SL_LOOKBACK", "14"))
DEFAULT_ATR_MULT_TP = float(os.getenv("TP_SL_ATR_MULT_TP", "1.25"))  # 1.25:1 RR (user-specified fallback, PHASE 3 FIX 2026-03-19)
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
    PHASE 3 FIX (2026-03-19): Market-Driven TP/SL with 1.25:1 Fallback
    
    STRATEGY (in order of preference):
    1. MARKET-DRIVEN: Uses actual support/resistance levels
       - If RR > 2.5:1, cap at 2.5:1 max
    2. FALLBACK (no market structure): ATR-based 1.25:1 RR
       - TP = Entry ± (1.25 × ATR)
       - SL = Entry ± (1.0 × ATR)
    
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

    # TRY MARKET-DRIVEN TP/SL FIRST (2026-03-18 NEW)
    md_result = calculate_tp_sl_from_df(df_clean, entry_f, direction, regime=regime)
    
    if md_result is not None:
        # Market-driven succeeded
        try:
            print(
                f"[tp_sl_retracement] MARKET_DRIVEN | direction={direction} entry={entry_f:.8f} "
                f"tp={md_result['tp']:.8f} sl={md_result['sl']:.8f} rr={md_result['achieved_rr']} source={md_result['source']}",
                flush=True
            )
        except Exception:
            pass
        return md_result
    else:
        # Market-driven rejected or no structure, fall back to ATR
        try:
            print(
                f"[tp_sl_retracement] MARKET_DRIVEN returned None, falling back to ATR (regime={regime})",
                flush=True
            )
        except Exception:
            pass
        
        atr = calculate_atr_for_tp_sl(df_clean, entry_f, lookback)
        result = calculate_tp_sl_from_atr(entry_f, atr, direction, atr_mult_tp, atr_mult_sl, regime=regime)

        try:
            print(
                f"[tp_sl_retracement] ATR_FALLBACK_REGIME_AWARE | direction={direction} entry={entry_f:.8f} "
                f"atr={atr:.8f} tp={result['tp']:.8f} sl={result['sl']:.8f} rr={result['achieved_rr']} regime={regime}",
                flush=True
            )
        except Exception:
            pass

        return result
