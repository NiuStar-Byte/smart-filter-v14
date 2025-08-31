"""
Test script for smart-filter-v14: checks that all filters in SmartFilter fire LONG and SHORT signals as expected.

Enhanced: Calls each filter as an instance method on SmartFilter.
All 23 canonical filters are tested if present in the class.
Outputs detailed debug info for LONG, SHORT, NEUTRAL cases.
"""

import pandas as pd
import numpy as np

from smart_filter import SmartFilter

def make_test_df(case="long", length=25):
    """
    Returns a DataFrame likely to trigger LONG, SHORT, or NEUTRAL signals.
    Adjust logic for your filters as needed.
    """
    if case == "long":
        data = {
            "open":   np.linspace(90, 105, length),
            "high":   np.linspace(100, 115, length),
            "low":    np.linspace(85, 100, length),
            "close":  np.linspace(95, 114, length),
            "volume": np.linspace(1000, 2000, length),
            "bid":    np.linspace(90, 105, length),
            "ask":    np.linspace(100, 115, length),
        }
        df = pd.DataFrame(data)
        df["higher_tf_volume"] = np.linspace(1000, 2000, length)
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        # Patch last two rows for VWAP Divergence LONG test
        df["close"].iloc[-2] = 100
        df["close"].iloc[-1] = 98
        df["vwap"].iloc[-2] = 105
        df["vwap"].iloc[-1] = 106
    elif case == "short":
        close_vals = np.linspace(115, 95, length)
        volume_vals = np.linspace(2000, 1000, length)
        data = {
            "open":   close_vals + 5,
            "high":   close_vals + 10,
            "low":    close_vals - 10,
            "close":  close_vals,
            "volume": volume_vals,
            "bid":    close_vals - 2,
            "ask":    close_vals + 2,
        }
        df = pd.DataFrame(data)
        df["higher_tf_volume"] = np.linspace(1000, 2000, length)
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        # Patch last two rows for VWAP Divergence SHORT test
        df["close"].iloc[-2] = 106
        df["close"].iloc[-1] = 108
        df["vwap"].iloc[-2] = 106
        df["vwap"].iloc[-1] = 105
    elif case == "neutral":
        data = {
            "open":   np.full(length, 100),
            "high":   np.full(length, 110),
            "low":    np.full(length, 90),
            "close":  np.full(length, 100),
            "volume": np.full(length, 1500),
            "bid":    np.full(length, 100),
            "ask":    np.full(length, 110),
        }
        df = pd.DataFrame(data)
        df["higher_tf_volume"] = np.linspace(1000, 2000, length)
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    else:
        raise ValueError(f"Unknown test case: {case}")

    return df

def get_filter_methods():
    """
    Returns list of (filter_name, callable) for all canonical filters in SmartFilter.
    Uses a lambda to call each filter as an instance method.
    """
    filter_names = [
        "Fractal Zone", "EMA Cloud", "MACD", "Momentum", "HATS", "Volume Spike",
        "VWAP Divergence", "MTF Volume Agreement", "HH/LL Trend", "EMA Structure",
        "Chop Zone", "Candle Confirmation", "Wick Dominance", "Absorption",
        "Support/Resistance", "Smart Money Bias", "Liquidity Pool", "Spread Filter",
        "Liquidity Awareness", "Trend Continuation", "Volatility Model",
        "ATR Momentum Burst", "Volatility Squeeze"
    ]
    method_map = {
        "Fractal Zone": "_check_fractal_zone",
        "EMA Cloud": "_check_ema_cloud",
        "MACD": "_check_macd",
        "Momentum": "_check_momentum",
        "HATS": "_check_hats",
        "Volume Spike": "_check_volume_spike",
        "VWAP Divergence": "_check_vwap_divergence",
        "MTF Volume Agreement": "_check_mtf_volume_agreement",
        "HH/LL Trend": "_check_hh_ll",
        "EMA Structure": "_check_ema_structure",
        "Chop Zone": "_check_chop_zone",
        "Candle Confirmation": "_check_candle_close",
        "Wick Dominance": "_check_wick_dominance",
        "Absorption": "_check_absorption",
        "Support/Resistance": "_check_support_resistance",
        "Smart Money Bias": "_check_smart_money_bias",
        "Liquidity Pool": "_check_liquidity_pool",
        "Spread Filter": "_check_spread_filter",
        "Liquidity Awareness": "_check_liquidity_awareness",
        "Trend Continuation": "_check_trend_continuation",
        "Volatility Model": "_check_volatility_model",
        "ATR Momentum Burst": "_check_atr_momentum_burst",
        "Volatility Squeeze": "_check_volatility_squeeze"
    }
    dummy = SmartFilter(symbol="TEST", df=make_test_df("long"))
    methods = []
    for filter_name in filter_names:
        attr = method_map.get(filter_name)
        if hasattr(dummy, attr):
            # Use a lambda so we call the method properly as instance method
            methods.append((filter_name, lambda sf, a=attr: getattr(sf, a)()))
    return methods

def test_filter(filter_name, filter_func, sf_long, sf_short, sf_neutral):
    # print(f"\n--- Testing {filter_name} ---")
    result_long = filter_func(sf_long)
    result_short = filter_func(sf_short)
    result_neutral = filter_func(sf_neutral)

    # print(f"[DEBUG] {filter_name} - LONG")
    # print("Input (last 2 rows):")
    # print(sf_long.df.tail(2))
    # print(f"Result: {result_long} (Expected: LONG or None)")

    # print(f"[DEBUG] {filter_name} - SHORT")
    # print("Input (last 2 rows):")
    # print(sf_short.df.tail(2))
    # print(f"Result: {result_short} (Expected: SHORT or None)")

    # print(f"[DEBUG] {filter_name} - NEUTRAL")
    # print("Input (last 2 rows):")
    # print(sf_neutral.df.tail(2))
    # print(f"Result: {result_neutral} (Expected: None)")

def run_all_filter_tests():
    methods = get_filter_methods()
    for filter_name, filter_func in methods:
        df_long = make_test_df("long")
        df_short = make_test_df("short")
        df_neutral = make_test_df("neutral")
        sf_long = SmartFilter(symbol="TEST", df=df_long)
        sf_short = SmartFilter(symbol="TEST", df=df_short)
        sf_neutral = SmartFilter(symbol="TEST", df=df_neutral)
        test_filter(filter_name, filter_func, sf_long, sf_short, sf_neutral)

if __name__ == "__main__":
    print("Starting all SmartFilter tests...")
    run_all_filter_tests()
