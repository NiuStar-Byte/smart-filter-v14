import os
import pandas as pd
import numpy as np

# Environment-driven defaults (can be overridden by env vars)
DEFAULT_LOOKBACK = int(os.getenv("TP_SL_LOOKBACK", "100"))         # bars to search for swing high/low
DEFAULT_ATR_MULT = float(os.getenv("TP_SL_ATR_MULT", "1.8"))      # ATR multiplier for SL buffer
DEFAULT_FALLBACK_TP_PCT = float(os.getenv("FALLBACK_TP_PCT", "0.03"))   # 3% TP fallback
DEFAULT_FALLBACK_SL_PCT = float(os.getenv("FALLBACK_SL_PCT", "0.015"))  # 1.5% SL fallback
DEFAULT_DESIRED_RR = float(os.getenv("TP_SL_DESIRED_RR", "1.0"))   # preferred RR (TP/SL)
DEFAULT_MIN_RR = float(os.getenv("TP_SL_MIN_RR", "0.5"))          # minimum acceptable RR


def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        try:
            return float(str(v))
        except Exception:
            return default


def calculate_tp_sl(df, entry_price, direction, atr_multiplier=None, lookback=None):
    """
    Robust TP/SL (fib-based) calculation for LONG and SHORT with R:R-aware TP selection.

    Behavior:
      - Uses lookback bars (default TP_SL_LOOKBACK) to find recent_high and recent_low.
      - Builds fib levels and prefers intermediate ratios in order:
          [0.382, 0.5, 0.618, 0.236, 0.786]
      - SL buffer is entry +/- (atr_multiplier * ATR). Default ATR multiplier comes from TP_SL_ATR_MULT.
      - Chooses TP to satisfy desired R:R (TP distance / SL distance) when possible.
      - Falls back to percent-based TP/SL (FALLBACK_TP_PCT / FALLBACK_SL_PCT) if swings or candidates are invalid.
    Returns:
      {'tp': float, 'sl': float, 'fib_levels': dict_or_None, 'chosen_ratio': float_or_None, 'source': str, 'achieved_rr': float_or_None}
    """
    # Resolve env-driven defaults
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
        desired_rr = float(os.getenv("TP_SL_DESIRED_RR", str(DEFAULT_DESIRED_RR)))
    except Exception:
        desired_rr = DEFAULT_DESIRED_RR
    try:
        min_rr = float(os.getenv("TP_SL_MIN_RR", str(DEFAULT_MIN_RR)))
    except Exception:
        min_rr = DEFAULT_MIN_RR

    try:
        fallback_tp_pct = float(os.getenv("FALLBACK_TP_PCT", str(DEFAULT_FALLBACK_TP_PCT)))
    except Exception:
        fallback_tp_pct = DEFAULT_FALLBACK_TP_PCT
    try:
        fallback_sl_pct = float(os.getenv("FALLBACK_SL_PCT", str(DEFAULT_FALLBACK_SL_PCT)))
    except Exception:
        fallback_sl_pct = DEFAULT_FALLBACK_SL_PCT

    # Validate entry
    entry_f = _safe_float(entry_price, None)
    if entry_f is None:
        raise ValueError("entry_price must be numeric")

    # Coerce df into DataFrame if needed
    if df is None:
        raise ValueError("df is None")
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            raise ValueError("Unable to coerce df to DataFrame")

    # Ensure or map columns
    cols = set(df.columns)
    if not {'high', 'low', 'close'}.issubset(cols):
        # Try to map positional columns if present (open, high, low, close)
        if df.shape[1] >= 4:
            df = df.copy()
            col_names = list(df.columns[:4])
            df = df.rename(columns={col_names[0]: 'open', col_names[1]: 'high', col_names[2]: 'low', col_names[3]: 'close'})
        else:
            raise ValueError("DataFrame must contain 'high','low','close' columns or provide OHLC positions")

    # Convert series to numeric floats (coerce non-numeric to NaN)
    high = pd.to_numeric(df['high'], errors='coerce').astype(float)
    low = pd.to_numeric(df['low'], errors='coerce').astype(float)
    close = pd.to_numeric(df['close'], errors='coerce').astype(float)

    # Compute True Range as pandas Series (avoid ndarray .iloc issues)
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

    # Ensure first TR is present
    if len(tr) > 0:
        try:
            if pd.isna(tr.iloc[0]) or tr.iloc[0] == 0:
                tr.iloc[0] = float((high - low).iat[0] if len(high) > 0 else 0.0)
        except Exception:
            try:
                tr.iloc[0] = float((high - low).values[0])
            except Exception:
                pass

    # ATR (rolling mean)
    atr = tr.rolling(14, min_periods=1).mean().iat[-1] if len(tr) > 0 else 0.0
    if pd.isna(atr):
        atr = float(tr.mean()) if len(tr) else 0.0
    atr = float(0.0 if np.isnan(atr) else atr)

    # Determine lookback bounded by available history
    lb = int(min(max(3, lookback), len(df)))
    use_df = df.iloc[-lb:].copy()

    recent_high = _safe_float(pd.to_numeric(use_df['high'], errors='coerce').max(), None)
    recent_low = _safe_float(pd.to_numeric(use_df['low'], errors='coerce').min(), None)

    if recent_high is None or recent_low is None or np.isnan(recent_high) or np.isnan(recent_low):
        # If no valid swings, fall back to percent-based targets around entry
        tp = round(entry_f * (1 + fallback_tp_pct), 8)
        sl = round(entry_f * (1 - fallback_sl_pct), 8)
        source = "percent_fallback_missing_swings"
        fib_levels = None
        chosen_ratio = None
        print(f"[tp_sl_retracement] missing swings -> fallback tp={tp} sl={sl} source={source}", flush=True)
        return {'tp': tp, 'sl': sl, 'fib_levels': fib_levels, 'chosen_ratio': chosen_ratio, 'source': source, 'achieved_rr': None}

    recent_high = float(recent_high)
    recent_low = float(recent_low)
    price_range = recent_high - recent_low

    # Build fib ratios and levels
    fib_ratios_preferred = [0.382, 0.5, 0.618, 0.236, 0.786]
    fib_ratios_all = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_levels = {f"{r:.3f}": (recent_low + r * price_range) for r in fib_ratios_all} if price_range > 0 else None

    dir_up = str(direction).strip().upper() == "LONG"
    dir_down = str(direction).strip().upper() == "SHORT"

    tp = None
    sl = None
    chosen_ratio = None
    source = None
    achieved_rr = None

    # Helper: select candidate by R:R preferences
    def _select_tp_by_rr(entry_f, sl_val, candidates, desired_rr_local=desired_rr, min_rr_local=min_rr):
        """
        candidates: list of tuples (ratio, level)
        Returns (chosen_ratio, chosen_tp, achieved_rr) or (None, None, 0.0)
        Preference:
          1) first candidate with rr >= desired_rr (closest to entry)
          2) first candidate with rr >= min_rr (closest to entry)
          3) candidate with max rr
        RR:
          SHORT: rr = (entry - tp) / (sl - entry)
          LONG:  rr = (tp - entry) / (entry - sl)
        """
        if not candidates:
            return None, None, 0.0
        rr_list = []
        sl_dist = abs(sl_val - entry_f)
        if sl_dist == 0:
            sl_dist = 1e-9
        for r, lvl in candidates:
            tp_dist = abs(entry_f - lvl)
            rr = tp_dist / sl_dist if sl_dist else 0.0
            rr_list.append((r, float(lvl), rr, tp_dist))
        # 1) candidate with rr >= desired_rr (choose smallest tp_dist among them)
        ge_desired = [c for c in rr_list if c[2] >= desired_rr_local]
        if ge_desired:
            chosen = min(ge_desired, key=lambda x: x[3])
            return chosen[0], chosen[1], chosen[2]
        # 2) candidate with rr >= min_rr
        ge_min = [c for c in rr_list if c[2] >= min_rr_local]
        if ge_min:
            chosen = min(ge_min, key=lambda x: x[3])
            return chosen[0], chosen[1], chosen[2]
        # 3) choose candidate with max rr
        best = max(rr_list, key=lambda x: x[2])
        return best[0], best[1], best[2]

    try:
        if price_range > 0 and fib_levels is not None:
            if dir_up:
                # SL candidate before selection
                sl_candidate = entry_f - atr_multiplier * atr if atr > 0 else recent_low
                # preferred candidates above entry
                preferred_above = [(r, fib_levels[f"{r:.3f}"]) for r in fib_ratios_preferred if fib_levels.get(f"{r:.3f}", float('nan')) > entry_f]
                any_above = [(r, fib_levels[f"{r:.3f}"]) for r in fib_ratios_all if fib_levels.get(f"{r:.3f}", float('nan')) > entry_f]
                # Try preferred with RR logic
                chosen_ratio, chosen_val, achieved_rr = _select_tp_by_rr(entry_f, sl_candidate, preferred_above)
                if chosen_val is None and any_above:
                    chosen_ratio, chosen_val, achieved_rr = _select_tp_by_rr(entry_f, sl_candidate, any_above)
                if chosen_val is not None:
                    tp = float(chosen_val)
                    source = "fib"
                else:
                    if any_above:
                        chosen_ratio, chosen_val = min(any_above, key=lambda x: x[1])
                        tp = float(chosen_val)
                        source = "fib"
                    else:
                        tp = entry_f * (1 + fallback_tp_pct)
                        source = "percent_fallback_no_profitable_fib"
                sl = float(min(recent_low, sl_candidate))
            elif dir_down:
                sl_candidate = entry_f + atr_multiplier * atr if atr > 0 else recent_high
                preferred_below = [(r, fib_levels[f"{r:.3f}"]) for r in fib_ratios_preferred if fib_levels.get(f"{r:.3f}", float('nan')) < entry_f]
                any_below = [(r, fib_levels[f"{r:.3f}"]) for r in fib_ratios_all if fib_levels.get(f"{r:.3f}", float('nan')) < entry_f]
                chosen_ratio, chosen_val, achieved_rr = _select_tp_by_rr(entry_f, sl_candidate, preferred_below)
                if chosen_val is None and any_below:
                    chosen_ratio, chosen_val, achieved_rr = _select_tp_by_rr(entry_f, sl_candidate, any_below)
                if chosen_val is not None:
                    tp = float(chosen_val)
                    source = "fib"
                else:
                    if any_below:
                        chosen_ratio, chosen_val = max(any_below, key=lambda x: x[1])
                        tp = float(chosen_val)
                        source = "fib"
                    else:
                        tp = entry_f * (1 - fallback_tp_pct)
                        source = "percent_fallback_no_profitable_fib"
                sl = float(max(recent_high, sl_candidate))
            else:
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

    # Final numeric sanitization
    tp_f = _safe_float(tp, None)
    sl_f = _safe_float(sl, None)
    if tp_f is None or sl_f is None:
        tp_f = entry_f * (1 + fallback_tp_pct)
        sl_f = entry_f * (1 - fallback_sl_pct)
        source = "percent_fallback_final_sanitize"
        chosen_ratio = None
        fib_levels = fib_levels if fib_levels is not None else None
        achieved_rr = None

    # compute achieved_rr if not set
    if achieved_rr is None:
        try:
            sl_dist = abs(sl_f - entry_f) if sl_f is not None else 0.0
            if sl_dist == 0:
                achieved_rr = None
            else:
                achieved_rr = abs(entry_f - tp_f) / sl_dist
        except Exception:
            achieved_rr = None

    # Round for readability
    tp_rounded = round(float(tp_f), 8)
    sl_rounded = round(float(sl_f), 8)

    # Debug line
    try:
        print(
            f"[tp_sl_retracement] direction={direction} entry={entry_f:.8f} recent_high={recent_high:.8f} recent_low={recent_low:.8f} "
            f"atr={atr:.8f} lookback={lb} chosen_ratio={chosen_ratio} source={source} tp={tp_rounded:.8f} sl={sl_rounded:.8f} achieved_rr={achieved_rr}",
            flush=True
        )
    except Exception:
        print(f"[tp_sl_retracement] computed TP/SL (source={source})", flush=True)

    return {'tp': tp_rounded, 'sl': sl_rounded, 'fib_levels': fib_levels, 'chosen_ratio': chosen_ratio, 'source': source, 'achieved_rr': achieved_rr}
