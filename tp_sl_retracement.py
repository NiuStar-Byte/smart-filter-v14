import pandas as pd
import numpy as np

def calculate_tp_sl(df, entry_price, direction, atr_multiplier=1.5, lookback=50):
    """
    Robust TP/SL (fib-based) calculation for LONG and SHORT.

    Behavior:
    - Computes ATR from True Range (14) safely (uses available history if <14 bars).
    - Finds recent swing high / swing low over `lookback` bars (or available history).
    - Builds symmetric Fibonacci levels between recent_low and recent_high:
        0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0
      (these values are the same numbers used for LONG and SHORT).
    - For LONG:
        - TP = the nearest fib level strictly above entry (shallowest possible).
        - SL = min(recent_low, entry - atr_multiplier * atr) (conservative)
      For SHORT:
        - TP = the nearest fib level strictly below entry (shallowest possible)
        - SL = max(recent_high, entry + atr_multiplier * atr)
    - If no fib level is on the profitable side of entry, fall back to using recent_high/recent_low
      or an ATR-based level so we always return numeric TP/SL.
    - Returns dict: {'tp': float, 'sl': float, 'fib_levels': dict_of_levels}
    """
    # Defensive conversions
    try:
        entry_price = float(entry_price)
    except Exception:
        raise ValueError("entry_price must be numeric")

    # Defensive: ensure DataFrame has required columns
    if df is None or df.empty or not set(['high', 'low', 'close']).issubset(df.columns):
        raise ValueError("DataFrame must contain 'high','low','close' and have rows")

    # Compute True Range (vectorized), do not modify original df in place
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    tr = np.maximum.reduce([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ])
    # fill missing first TR with high-low
    tr.iloc[0] = (high - low).iat[0] if len(tr) > 0 else 0.0

    atr = tr.rolling(14, min_periods=1).mean().iat[-1] if len(tr) > 0 else 0.0
    if np.isnan(atr) or atr is None:
        atr = float(tr.mean()) if len(tr) else 0.0
    atr = float(atr)

    # Determine lookback window (bounded by available history)
    lb = int(min(max(3, lookback), len(df)))

    # Use previous completed bars (exclude the current in-progress bar if present)
    use_df = df.iloc[-lb:].copy()

    recent_high = float(use_df['high'].max())
    recent_low = float(use_df['low'].min())

    # Ensure a non-zero price range
    price_range = recent_high - recent_low
    if price_range <= 0:
        # Degenerate case: fallback to ATR percent levels around entry
        tp = entry_price + atr_multiplier * atr
        sl = entry_price - atr_multiplier * atr
        return {'tp': round(float(tp), 8), 'sl': round(float(sl), 8), 'fib_levels': None}

    # Build symmetric fib levels (from low -> high)
    fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_levels = {}
    for r in fib_ratios:
        fib_levels[f"{r:.3f}"] = recent_low + r * price_range

    dir_up = str(direction).strip().upper() == "LONG"
    dir_down = str(direction).strip().upper() == "SHORT"

    tp = None
    sl = None

    if dir_up:
        # LONG: pick the smallest fib level that is strictly above the entry (closest profitable target)
        above = [(float(k), v) for k, v in fib_levels.items() if v > entry_price]
        if above:
            # choose lowest level > entry (shallowest profitable fib)
            chosen = min(above, key=lambda x: x[1])
            tp = float(chosen[1])
        else:
            # nothing above entry (rare) -> use recent_high as TP
            tp = recent_high

        # SL: conservative (lower of recent_low and entry - atr*mult)
        sl_candidate = entry_price - atr_multiplier * atr if atr and atr > 0 else recent_low
        sl = float(min(recent_low, sl_candidate))

    elif dir_down:
        # SHORT: pick the largest fib level that is strictly below the entry (closest profitable target)
        below = [(float(k), v) for k, v in fib_levels.items() if v < entry_price]
        if below:
            # choose highest level < entry (shallowest fib below entry)
            chosen = max(below, key=lambda x: x[1])
            tp = float(chosen[1])
        else:
            # nothing below entry (rare) -> use recent_low as TP
            tp = recent_low

        # SL: conservative (higher of recent_high and entry + atr*mult)
        sl_candidate = entry_price + atr_multiplier * atr if atr and atr > 0 else recent_high
        sl = float(max(recent_high, sl_candidate))

    else:
        # Unknown direction, fallback to percent/ATR around entry
        tp = entry_price + atr_multiplier * atr
        sl = entry_price - atr_multiplier * atr

    # Final sanity: ensure tp and sl are numeric and not equal to entry
    try:
        tp = float(tp)
    except Exception:
        tp = entry_price + atr_multiplier * atr
    try:
        sl = float(sl)
    except Exception:
        sl = entry_price - atr_multiplier * atr

    # Round to a suitable precision (keep 6-8 decimals for small-price coins)
    tp_rounded = round(tp, 8)
    sl_rounded = round(sl, 8)

    return {'tp': tp_rounded, 'sl': sl_rounded, 'fib_levels': fib_levels}
