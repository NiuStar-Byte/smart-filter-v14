"""
tp_sl_retracement.py (2026-02-23: ATR-Based 2:1 RR)

TP/SL calculation for Smart Filter.
- ATR-based 2:1 Risk:Reward ratio
- TP = Entry ± (2.0 × ATR)
- SL = Entry ± (1.0 × ATR)
- Always returns achieved_rr = 2.0
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_ATR_LOOKBACK = int(os.getenv("TP_SL_LOOKBACK", "14"))
DEFAULT_ATR_MULT_TP = float(os.getenv("TP_SL_ATR_MULT_TP", "2.0"))
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
                    atr_multiplier: Optional[float] = None, lookback: Optional[int] = None) -> Dict[str, Any]:
    """
    ATR-Based 2:1 RR TP/SL Calculation.
    TP = Entry ± (2.0 × ATR)
    SL = Entry ± (1.0 × ATR)
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

    high = pd.to_numeric(df['high'], errors='coerce').astype(float)
    low = pd.to_numeric(df['low'], errors='coerce').astype(float)
    close = pd.to_numeric(df['close'], errors='coerce').astype(float)

    try:
        tr_values = np.maximum.reduce([
            (high - low).values,
            (high - close.shift(1)).abs().values,
            (low - close.shift(1)).abs().values
        ])
        tr = pd.Series(tr_values, index=df.index)
    except Exception:
        tr = pd.concat([
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        tr = tr.fillna((high - low)).astype(float)

    atr_period = int(min(max(1, lookback), len(df)))
    atr = tr.rolling(atr_period, min_periods=1).mean().iat[-1] if len(tr) > 0 else 0.0
    atr = float(0.0 if np.isnan(atr) else atr)

    if atr <= 0:
        atr = entry_f * 0.01

    dir_up = str(direction).strip().upper() == "LONG"
    dir_down = str(direction).strip().upper() == "SHORT"

    tp = None
    sl = None
    source = None

    try:
        if dir_up:
            tp = entry_f + (atr_mult_tp * atr)
            sl = entry_f - (atr_mult_sl * atr)
            source = "atr_2_to_1_long"
        elif dir_down:
            tp = entry_f - (atr_mult_tp * atr)
            sl = entry_f + (atr_mult_sl * atr)
            source = "atr_2_to_1_short"
        else:
            tp = entry_f + (atr_mult_tp * atr)
            sl = entry_f - (atr_mult_sl * atr)
            source = "atr_2_to_1_default_long"
    except Exception as e:
        print(f"[tp_sl_retracement] Exception: {e}", flush=True)
        tp = entry_f + (atr_mult_tp * atr)
        sl = entry_f - (atr_mult_sl * atr)
        source = "atr_2_to_1_exception"

    tp_f = _safe_float(tp, entry_f + atr_mult_tp * atr)
    sl_f = _safe_float(sl, entry_f - atr_mult_sl * atr)

    tp_rounded = round(float(tp_f), 8)
    sl_rounded = round(float(sl_f), 8)
    achieved_rr = 2.0

    try:
        print(
            f"[tp_sl_retracement] ATR_2_1_RR | direction={direction} entry={entry_f:.8f} "
            f"atr={atr:.8f} tp={tp_rounded:.8f} sl={sl_rounded:.8f} rr={achieved_rr}",
            flush=True
        )
    except Exception:
        pass

    return {
        'tp': tp_rounded,
        'sl': sl_rounded,
        'source': source,
        'achieved_rr': achieved_rr,
        'atr_value': float(atr),
        'fib_levels': None,
        'chosen_ratio': None,
        'sl_capped': False
    }
