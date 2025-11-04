import pandas as pd
import numpy as np

def calculate_tp_sl(df, entry_price, direction, atr_multiplier=1.5, lookback=50):
    """
    Robust TP/SL (fib-based) calculation for LONG and SHORT with debug logging.
    Returns: {'tp': tp, 'sl': sl, 'fib_levels': fib_levels, 'chosen_ratio': ratio_or_None, 'source': 'fib'|'fib_fallback'|'atr_fallback'}
    """
    # Defensive conversions
    try:
        entry_price = float(entry_price)
    except Exception:
        raise ValueError("entry_price must be numeric")

    if df is None or len(df) == 0:
        raise ValueError("DataFrame must contain rows")

    # If df is not a DataFrame, try to coerce to one (best-effort)
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            raise ValueError("Unable to coerce df to DataFrame")

    # Ensure expected columns exist or attempt to map common positions
    cols = set(df.columns)
    if not {'high', 'low', 'close'}.issubset(cols):
        # Try common OHLCV order if columns are positional
        # If df has at least 4 columns, assume [open, high, low, close, ...]
        if df.shape[1] >= 4:
            df = df.copy()
            df.columns = list(df.columns[:4])  # preserve names, we'll remap below
            # Map by position
            df = df.rename(columns={df.columns[0]: 'open', df.columns[1]: 'high', df.columns[2]: 'low', df.columns[3]: 'close'})
        else:
            raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")

    # Convert to float series
    high = pd.to_numeric(df['high'], errors='coerce').astype(float)
    low = pd.to_numeric(df['low'], errors='coerce').astype(float)
    close = pd.to_numeric(df['close'], errors='coerce').astype(float)

    # Compute True Range as a pandas Series (avoid ndarray to preserve .rolling / .iloc usage)
    try:
        tr_values = np.maximum.reduce([
            (high - low).values,
            (high - close.shift(1)).abs().values,
            (low - close.shift(1)).abs().values
        ])
        # Create pandas Series with same index as df
        tr = pd.Series(tr_values, index=df.index)
    except Exception:
        # Fallback safe calculation using vectorized pandas ops
        tr = pd.concat([
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        tr = tr.fillna((high - low)).astype(float)

    # Ensure first element exists
    if len(tr) > 0 and (pd.isna(tr.iloc[0]) or tr.iloc[0] == 0):
        tr.iloc[0] = float((high - low).iat[0] if len(high) > 0 else 0.0)

    # ATR (rolling mean)
    atr = tr.rolling(14, min_periods=1).mean().iat[-1] if len(tr) > 0 else 0.0
    if pd.isna(atr):
        atr = float(tr.mean()) if len(tr) else 0.0
    atr = float(0.0 if np.isnan(atr) else atr)

    # Determine lookback window (bounded by available history)
    lb = int(min(max(3, lookback), len(df)))

    # Use previous completed bars (exclude the current in-progress bar if present)
    use_df = df.iloc[-lb:].copy()

    recent_high = float(pd.to_numeric(use_df['high'], errors='coerce').max())
    recent_low = float(pd.to_numeric(use_df['low'], errors='coerce').min())

    price_range = recent_high - recent_low

    # Build symmetric fib levels (from low -> high)
    fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_levels = {f"{r:.3f}": (recent_low + r * price_range) for r in fib_ratios} if price_range > 0 else None

    dir_up = str(direction).strip().upper() == "LONG"
    dir_down = str(direction).strip().upper() == "SHORT"

    tp = None
    sl = None
    chosen_ratio = None
    source = None

    try:
        if price_range > 0 and fib_levels is not None:
            if dir_up:
                # LONG: pick the shallowest fib above entry
                above = [(r, v) for r, v in [(float(k), v) for k, v in fib_levels.items()] if v > entry_price]
                if above:
                    chosen_ratio, chosen_val = min(above, key=lambda x: x[1])
                    tp = float(chosen_val)
                    source = "fib"
                else:
                    tp = recent_high
                    source = "fib_fallback"
                sl_candidate = entry_price - atr_multiplier * atr if atr > 0 else recent_low
                sl = float(min(recent_low, sl_candidate))
            elif dir_down:
                # SHORT: pick the shallowest fib below entry
                below = [(r, v) for r, v in [(float(k), v) for k, v in fib_levels.items()] if v < entry_price]
                if below:
                    chosen_ratio, chosen_val = max(below, key=lambda x: x[1])
                    tp = float(chosen_val)
                    source = "fib"
                else:
                    tp = recent_low
                    source = "fib_fallback"
                sl_candidate = entry_price + atr_multiplier * atr if atr > 0 else recent_high
                sl = float(max(recent_high, sl_candidate))
            else:
                tp = entry_price + atr_multiplier * atr
                sl = entry_price - atr_multiplier * atr
                source = "atr_fallback"
        else:
            tp = entry_price + atr_multiplier * atr
            sl = entry_price - atr_multiplier * atr
            source = "atr_fallback"
    except Exception as e:
        tp = entry_price + atr_multiplier * atr
        sl = entry_price - atr_multiplier * atr
        source = "atr_fallback"
        print(f"[tp_sl_retracement] Exception during TP/SL compute: {e}", flush=True)

    # Final sanity
    try:
        tp = float(tp)
        sl = float(sl)
    except Exception:
        tp = entry_price + atr_multiplier * atr
        sl = entry_price - atr_multiplier * atr
        source = "atr_fallback"

    tp_rounded = round(tp, 8)
    sl_rounded = round(sl, 8)

    # Debug log: always print what we computed
    try:
        print(
            f"[tp_sl_retracement] direction={direction} entry={entry_price:.8f} recent_high={recent_high:.8f} recent_low={recent_low:.8f} "
            f"atr={atr:.8f} chosen_ratio={chosen_ratio} source={source} tp={tp_rounded:.8f} sl={sl_rounded:.8f}",
            flush=True
        )
    except Exception:
        print(f"[tp_sl_retracement] computed TP/SL (source={source})", flush=True)

    return {'tp': tp_rounded, 'sl': sl_rounded, 'fib_levels': fib_levels, 'chosen_ratio': chosen_ratio, 'source': source}
