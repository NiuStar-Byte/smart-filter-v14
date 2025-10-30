"""
Test harness for SmartFilter changes:
- Verifies support/resistance behavior (LONG/SHORT)
- Verifies get_signal_direction fallback allows SHORT when weighted scores favor SHORT
- Runs a few scenarios and prints results with debug info

Usage:
    python scripts/test_harness.py

This script assumes `smart_filter.py` containing SmartFilter is importable from the repository root.
"""
import math
import pandas as pd
import numpy as np
from types import SimpleNamespace

# Try to import SmartFilter from smart_filter.py in repo
try:
    from smart_filter import SmartFilter
except Exception as e:
    raise ImportError("Couldn't import SmartFilter from smart_filter.py. Run this from repo root. Error: " + str(e))


def make_price_df(length=30, start_price=100.0, step=0.1, vol_base=1000):
    """
    Create a simple DataFrame with price series and required columns.
    We'll include columns needed by support/resistance and vwap checks.
    """
    idx = list(range(length))
    close = [start_price + i * step for i in idx]
    # small variations for open/high/low
    open_ = [c - (0.02 * ((-1) ** i)) for i, c in enumerate(close)]
    high = [max(o, c) + 0.05 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.05 for o, c in zip(open_, close)]
    volume = [vol_base + (50 * (i % 3)) for i in idx]
    # simple vwap: for this harness we set vwap slightly above/below close as needed later
    vwap = list(np.array(close) + 0.0)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": vwap,
        # placeholders for other filters that might access them
        "bid": [np.nan] * length,
        "ask": [np.nan] * length,
        "higher_tf_volume": [np.nan] * length,
        "atr": [0.5] * length,
        "atr_ma": [0.5] * length,
        "bb_upper": [c + 1.0 for c in close],
        "bb_lower": [c - 1.0 for c in close],
        "kc_upper": [c + 1.2 for c in close],
        "kc_lower": [c - 1.2 for c in close],
        "ema6": [c - 0.05 for c in close],
        "ema9": [c - 0.04 for c in close],
        "ema10": [c - 0.03 for c in close],
        "ema13": [c - 0.02 for c in close],
        "ema20": [c - 0.01 for c in close],
        "ema21": [c - 0.015 for c in close],
        "ema50": [c - 0.5 for c in close],
        "ema200": [c - 1.0 for c in close],
        "macd": [0.1] * length,
        "RSI": [55.0] * length,
        "chop_zone": [20.0] * length,
        "spread": [0.1] * length,
        "vwap_div": [0.0] * length,
    })
    return df


def scenario_support_long():
    # Build df where close is very near rolling low -> expect LONG
    df = make_price_df()
    # Make rolling low small so support is e.g. ~100.0
    # Ensure last close is within buffer (0.5%) of support
    df["low"] = 100.0  # constant low for rolling min
    df["high"] = df["low"] + 1.0
    df["close"] = 100.2  # within 0.5% buffer of support=100
    df["close"].iat[-2] = 100.1
    df["volume"].iat[-1] = 1200
    df["volume"].iat[-2] = 1000

    sf = SmartFilter.__new__(SmartFilter)
    sf.df = df
    sf.symbol = "TEST_SUPPORT_LONG"

    res = SmartFilter._check_support_resistance(sf, window=20, buffer_pct=0.005, min_cond=2, debug=True)
    print("Support LONG scenario -> result:", res)
    return res


def scenario_support_short():
    # Build df where close is very near rolling high -> expect SHORT
    df = make_price_df()
    df["high"] = 200.0  # constant high for rolling max
    df["low"] = df["high"] - 1.0
    df["close"] = 199.4  # within 0.5% below resistance (200 * (1 - 0.005) = 199.0 => 199.4 is inside)
    df["close"].iat[-2] = 199.6
    df["volume"].iat[-1] = 900
    df["volume"].iat[-2] = 1000

    sf = SmartFilter.__new__(SmartFilter)
    sf.df = df
    sf.symbol = "TEST_SUPPORT_SHORT"

    res = SmartFilter._check_support_resistance(sf, window=20, buffer_pct=0.005, min_cond=2, debug=True)
    print("Support SHORT scenario -> result:", res)
    return res


def test_get_signal_direction_fallback_short():
    dummy = SimpleNamespace()
    dummy.symbol = "DUMMY"
    dummy.gatekeepers = ["Candle Confirmation", "Support/Resistance"]
    dummy.soft_gatekeepers = ["Support/Resistance"]
    dummy.filter_names = ["A", "B", "C", "D", "E"] + dummy.gatekeepers
    dummy.filter_weights_long = {name: 1.0 for name in dummy.filter_names}
    dummy.filter_weights_short = {name: 1.0 for name in dummy.filter_names}
    dummy.filter_weights_short["A"] = 6.0
    dummy.filter_weights_short["B"] = 5.0
    dummy.filter_weights_long["A"] = 1.0
    dummy.filter_weights_long["B"] = 1.0
    dummy.min_score = 2
    dummy.fallback_margin = 2.0

    results_long = {f: False for f in dummy.filter_names}
    results_short = {f: False for f in dummy.filter_names}
    results_long["Candle Confirmation"] = False
    results_short["Candle Confirmation"] = False

    results_short["A"] = True
    results_short["B"] = True
    results_long["D"] = True

    direction = SmartFilter.get_signal_direction(dummy, results_long, results_short)
    print("get_signal_direction fallback scenario -> result:", direction)
    if hasattr(dummy, "_debug_sums"):
        print("debug sums:", dummy._debug_sums)
    return direction


def test_get_signal_direction_both_gks_short():
    dummy = SimpleNamespace()
    dummy.symbol = "DUMMY2"
    dummy.gatekeepers = ["Candle Confirmation", "Support/Resistance"]
    dummy.soft_gatekeepers = ["Support/Resistance"]
    dummy.filter_names = ["A", "B", "C", "D"] + dummy.gatekeepers

    dummy.filter_weights_long = {name: 1.0 for name in dummy.filter_names}
    dummy.filter_weights_short = {name: 1.0 for name in dummy.filter_names}
    dummy.filter_weights_short["A"] = 6.0
    dummy.filter_weights_short["B"] = 5.0

    dummy.min_score = 1
    dummy.fallback_margin = 2.0

    results_long = {f: False for f in dummy.filter_names}
    results_short = {f: False for f in dummy.filter_names}
    results_long["Candle Confirmation"] = True
    results_short["Candle Confirmation"] = True

    results_short["A"] = True
    results_short["B"] = True
    results_long["C"] = True

    direction = SmartFilter.get_signal_direction(dummy, results_long, results_short)
    print("get_signal_direction both GKs passed scenario -> result:", direction)
    if hasattr(dummy, "_debug_sums"):
        print("debug sums:", dummy._debug_sums)
    return direction


if __name__ == "__main__":
    print("=== Running Support/Resistance LONG test ===")
    support_long = scenario_support_long()
    print()

    print("=== Running Support/Resistance SHORT test ===")
    support_short = scenario_support_short()
    print()

    print("=== Running get_signal_direction fallback -> expect SHORT ===")
    fallback_short = test_get_signal_direction_fallback_short()
    print()

    print("=== Running get_signal_direction both GK pass -> expect SHORT ===")
    both_gk_short = test_get_signal_direction_both_gks_short()
    print()

    print("Summary of results:")
    print("support_long:", support_long)
    print("support_short:", support_short)
    print("fallback_short:", fallback_short)
    print("both_gk_short:", both_gk_short)
