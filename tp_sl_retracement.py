import pandas as pd
import numpy as np

def calculate_tp_sl(df, entry_price, direction, atr_multiplier=1.5, lookback=50):
    """
    Robust TP/SL (fib-based) calculation for LONG and SHORT with debug logging.
    Returns: {'tp': tp, 'sl': sl, 'fib_levels': fib_levels, 'chosen_ratio': ratio_or_None, 'source': 'calc'|'fib'|'atr_fallback'}
    """
    # Defensive conversions
    try:
        entry_price = float(entry_price)
    except Exception:
        raise ValueError("entry_price must be numeric")

    if df is None or df.empty or not set(['high', 'low', 'close']).issubset(df.columns):
        raise ValueError("DataFrame must contain 'high','low','close' and have rows")

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    tr = np.maximum.reduce([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ])
    tr.iloc[0] = (high - low).iat[0] if len(tr) > 0 else 0.0
    atr = tr.rolling(14, min_periods=1).mean().iat[-1] if len(tr) > 0 else 0.0
    atr = float(0.0 if np.isnan(atr) else atr)

    lb = int(min(max(3, lookback), len(df)))
    use_df = df.iloc[-lb:].copy()
    recent_high = float(use_df['high'].max())
    recent_low = float(use_df['low'].min())
    price_range = recent_high - recent_low

    # Build fib levels
    fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_levels = {f"{r:.3f}": recent_low + r * price_range for r in fib_ratios} if price_range > 0 else None

    chosen_ratio = None
    source = None
    tp = None
    sl = None

    try:
        # Prefer canonical fib-based selection
        if price_range > 0 and fib_levels is not None:
            if str(direction).strip().upper() == "LONG":
                above = [(r, v) for r, v in [(float(k), v) for k, v in fib_levels.items()] if v > entry_price]
                if above:
                    # pick shallowest profitable fib (closest above entry)
                    chosen_ratio, chosen_val = min(above, key=lambda x: x[1])
                    tp = float(chosen_val)
                    source = "fib"
                else:
                    tp = recent_high
                    source = "fib_fallback"
                sl_candidate = entry_price - atr_multiplier * atr if atr > 0 else recent_low
                sl = float(min(recent_low, sl_candidate))
            elif str(direction).strip().upper() == "SHORT":
                below = [(r, v) for r, v in [(float(k), v) for k, v in fib_levels.items()] if v < entry_price]
                if below:
                    # pick shallowest fib below entry (closest profitable)
                    chosen_ratio, chosen_val = max(below, key=lambda x: x[1])
                    tp = float(chosen_val)
                    source = "fib"
                else:
                    tp = recent_low
                    source = "fib_fallback"
                sl_candidate = entry_price + atr_multiplier * atr if atr > 0 else recent_high
                sl = float(max(recent_high, sl_candidate))
            else:
                # Unknown direction -> ATR fallback
                tp = entry_price + atr_multiplier * atr
                sl = entry_price - atr_multiplier * atr
                source = "atr_fallback"
        else:
            # Degenerate price_range -> ATR fallback
            tp = entry_price + atr_multiplier * atr
            sl = entry_price - atr_multiplier * atr
            source = "atr_fallback"
    except Exception as e:
        # Safe fallback on any error
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
        # Best-effort print
        print(f"[tp_sl_retracement] computed TP/SL (source={source})", flush=True)

    return {'tp': tp_rounded, 'sl': sl_rounded, 'fib_levels': fib_levels, 'chosen_ratio': chosen_ratio, 'source': source}
