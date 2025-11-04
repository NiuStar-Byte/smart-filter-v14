import os
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

# Aggressive defaults (override with env vars)
DEFAULT_LOOKBACK = int(os.getenv("TP_SL_LOOKBACK", "100"))      # bars to search for swing high/low
DEFAULT_ATR_MULT = float(os.getenv("TP_SL_ATR_MULT", "1.0"))   # tightened SL buffer (default tightened)
DEFAULT_FALLBACK_TP_PCT = float(os.getenv("FALLBACK_TP_PCT", "0.05"))  # 5% TP fallback (more aggressive TP)
DEFAULT_FALLBACK_SL_PCT = float(os.getenv("FALLBACK_SL_PCT", "0.02"))  # 2% SL fallback (tighter SL)
DEFAULT_MAX_SL_PCT = float(os.getenv("TP_SL_MAX_SL_PCT", "0.03"))      # absolute cap on SL distance (3% of entry)

def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
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
    Aggressive TP / Tight SL calculation:
      - Prefers 'deeper' fib targets (furthest fib on profitable side) to increase TP distance.
      - Uses a tighter ATR multiplier by default (TP_SL_ATR_MULT, default 1.0).
      - Caps SL distance to TP_SL_MAX_SL_PCT of entry to avoid very large SLs.
      - Fallback TP/SL use larger TP pct and tighter SL pct by default.
    Returns dict: {'tp', 'sl', 'fib_levels', 'chosen_ratio', 'source', 'achieved_rr', 'sl_capped'}
    """
    # Resolve env defaults
    if lookback is None:
        try:
            lookback = int(os.getenv("TP_SL_LOOKBACK", str(DEFAULT_LOOKBACK)))
        except Exception:
            lookback = DEFAULT_LOOKBACK
    if atr_multiplier is None:
        try:
            atr_multiplier = float(os.getenv("TP_SL_ATR_MULT", str(DEFAULT_ATR_MULT)))
        except Exception:
            atr_multiplier = DEFAULT_ATR_MULT

    try:
        fallback_tp_pct = float(os.getenv("FALLBACK_TP_PCT", str(DEFAULT_FALLBACK_TP_PCT)))
    except Exception:
        fallback_tp_pct = DEFAULT_FALLBACK_TP_PCT
    try:
        fallback_sl_pct = float(os.getenv("FALLBACK_SL_PCT", str(DEFAULT_FALLBACK_SL_PCT)))
    except Exception:
        fallback_sl_pct = DEFAULT_FALLBACK_SL_PCT
    try:
        max_sl_pct = float(os.getenv("TP_SL_MAX_SL_PCT", str(DEFAULT_MAX_SL_PCT)))
    except Exception:
        max_sl_pct = DEFAULT_MAX_SL_PCT

    # Normalize entry
    entry_f = _safe_float(entry_price, None)
    if entry_f is None:
        raise ValueError("entry_price must be numeric")

    # Coerce df
    if df is None or len(df) == 0:
        raise ValueError("df must be a non-empty DataFrame")
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            raise ValueError("Unable to coerce df to DataFrame")

    # Map columns if necessary
    cols = set(df.columns)
    if not {'high', 'low', 'close'}.issubset(cols):
        if df.shape[1] >= 4:
            df = df.copy()
            col_names = list(df.columns[:4])
            df = df.rename(columns={col_names[0]: 'open', col_names[1]: 'high', col_names[2]: 'low', col_names[3]: 'close'})
        else:
            raise ValueError("DataFrame must contain 'high','low','close' columns or provide OHLC positions")

    high = pd.to_numeric(df['high'], errors='coerce').astype(float)
    low = pd.to_numeric(df['low'], errors='coerce').astype(float)
    close = pd.to_numeric(df['close'], errors='coerce').astype(float)

    # Compute True Range as pandas Series
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

    if len(tr) > 0:
        try:
            if pd.isna(tr.iloc[0]) or tr.iloc[0] == 0:
                tr.iloc[0] = float((high - low).iat[0] if len(high) > 0 else 0.0)
        except Exception:
            try:
                tr.iloc[0] = float((high - low).values[0])
            except Exception:
                pass

    atr = tr.rolling(14, min_periods=1).mean().iat[-1] if len(tr) > 0 else 0.0
    atr = float(0.0 if np.isnan(atr) else atr)

    # Determine lookback window
    lb = int(min(max(3, lookback), len(df)))
    use_df = df.iloc[-lb:].copy()

    recent_high = _safe_float(pd.to_numeric(use_df['high'], errors='coerce').max(), None)
    recent_low = _safe_float(pd.to_numeric(use_df['low'], errors='coerce').min(), None)

    if recent_high is None or recent_low is None or np.isnan(recent_high) or np.isnan(recent_low):
        # Percent fallbacks
        tp_fb = round(entry_f * (1 + fallback_tp_pct), 8)
        sl_fb = round(entry_f * (1 - fallback_sl_pct), 8)
        return {'tp': tp_fb, 'sl': sl_fb, 'fib_levels': None, 'chosen_ratio': None, 'source': 'percent_fallback_missing_swings', 'achieved_rr': None, 'sl_capped': False}

    recent_high = float(recent_high)
    recent_low = float(recent_low)
    price_range = recent_high - recent_low

    # Build fib levels (all ratios)
    fib_ratios_all = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    # Aggressive preference: prefer deeper fibs first (increase TP magnitude)
    fib_ratios_preferred_deep = [0.786, 0.618, 0.5, 0.382, 0.236]

    fib_levels = {f"{r:.3f}": (recent_low + r * price_range) for r in fib_ratios_all} if price_range > 0 else None

    dir_up = str(direction).strip().upper() == "LONG"
    dir_down = str(direction).strip().upper() == "SHORT"

    tp = None
    sl = None
    chosen_ratio = None
    source = None
    sl_capped = False
    achieved_rr = None

    try:
        if price_range > 0 and fib_levels is not None:
            if dir_up:
                # LONG: prefer the highest fib > entry (deepest TP)
                candidates = [(r, fib_levels[f"{r:.3f}"]) for r in fib_ratios_preferred_deep if fib_levels.get(f"{r:.3f}", float('nan')) > entry_f]
                if not candidates:
                    candidates = [(r, fib_levels[f"{r:.3f}"]) for r in fib_ratios_all if fib_levels.get(f"{r:.3f}", float('nan')) > entry_f]
                if candidates:
                    # pick the deepest (max level -> biggest profit)
                    chosen_ratio, chosen_val = max(candidates, key=lambda x: x[1])
                    tp = float(chosen_val)
                    source = "fib_deep"
                else:
                    tp = recent_high
                    source = "fib_fallback"
                # SL candidate (tight): entry - atr*mult
                sl_candidate = entry_f - atr_multiplier * atr if atr > 0 else recent_low
                # SL cannot be below recent_low; pick the higher (more conservative) of the two
                sl = float(max(recent_low, sl_candidate))
                # Cap SL distance to max_sl_pct
                max_sl_price = entry_f * (1 - max_sl_pct)  # for LONG, SL is below entry; we ensure sl >= max_sl_price
                if sl < max_sl_price:
                    sl = float(max_sl_price)
                    sl_capped = True
            elif dir_down:
                # SHORT: prefer the lowest fib < entry (deepest TP downwards)
                candidates = [(r, fib_levels[f"{r:.3f}"]) for r in fib_ratios_preferred_deep if fib_levels.get(f"{r:.3f}", float('nan')) < entry_f]
                if not candidates:
                    candidates = [(r, fib_levels[f"{r:.3f}"]) for r in fib_ratios_all if fib_levels.get(f"{r:.3f}", float('nan')) < entry_f]
                if candidates:
                    # pick the deepest (min level -> biggest profit distance)
                    chosen_ratio, chosen_val = min(candidates, key=lambda x: x[1])
                    tp = float(chosen_val)
                    source = "fib_deep"
                else:
                    tp = recent_low
                    source = "fib_fallback"
                # SL candidate (tight): entry + atr*mult
                sl_candidate = entry_f + atr_multiplier * atr if atr > 0 else recent_high
                # SL must be at least recent_high (conservative)
                sl = float(max(recent_high, sl_candidate))
                # Cap SL distance to max_sl_pct above entry
                max_sl_price = entry_f * (1 + max_sl_pct)
                if sl > max_sl_price:
                    sl = float(max_sl_price)
                    sl_capped = True
            else:
                # unknown direction: percent fallback
                tp = entry_f * (1 + fallback_tp_pct)
                sl = entry_f * (1 - fallback_sl_pct)
                source = "percent_fallback_unknown_direction"
        else:
            tp = entry_f * (1 + fallback_tp_pct)
            sl = entry_f * (1 - fallback_sl_pct)
            source = "percent_fallback_degenerate_range"
    except Exception as e:
        tp = entry_f * (1 + fallback_tp_pct)
        sl = entry_f * (1 - fallback_sl_pct)
        source = "percent_fallback_exception"
        print(f"[tp_sl_retracement] Exception during TP/SL compute: {e}", flush=True)

    # Final sanitization
    tp_f = _safe_float(tp, None)
    sl_f = _safe_float(sl, None)
    if tp_f is None or sl_f is None:
        tp_f = entry_f * (1 + fallback_tp_pct)
        sl_f = entry_f * (1 - fallback_sl_pct)
        source = "percent_fallback_final_sanitize"
        chosen_ratio = None
        fib_levels = fib_levels if fib_levels is not None else None
        sl_capped = False

    # compute achieved_rr consistently (direction-aware)
    try:
        tp_val = float(tp_f)
        sl_val = float(sl_f)
        if dir_down:
            tp_dist = entry_f - tp_val
            sl_dist = sl_val - entry_f
        elif dir_up:
            tp_dist = tp_val - entry_f
            sl_dist = entry_f - sl_val
        else:
            tp_dist = abs(entry_f - tp_val)
            sl_dist = abs(sl_val - entry_f)
        achieved_rr = float(tp_dist / sl_dist) if sl_dist and sl_dist > 0 else None
    except Exception:
        achieved_rr = None

    tp_rounded = round(float(tp_f), 8)
    sl_rounded = round(float(sl_f), 8)

    try:
        print(
            f"[tp_sl_retracement] direction={direction} entry={entry_f:.8f} recent_high={recent_high:.8f} recent_low={recent_low:.8f} "
            f"atr={atr:.8f} lookback={lb} chosen_ratio={chosen_ratio} source={source} tp={tp_rounded:.8f} sl={sl_rounded:.8f} achieved_rr={achieved_rr} sl_capped={sl_capped}",
            flush=True
        )
    except Exception:
        print(f"[tp_sl_retracement] computed TP/SL (source={source})", flush=True)

    return {
        'tp': tp_rounded,
        'sl': sl_rounded,
        'fib_levels': fib_levels,
        'chosen_ratio': chosen_ratio,
        'source': source,
        'achieved_rr': achieved_rr,
        'sl_capped': sl_capped
    }
