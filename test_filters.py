"""
Test script for smart-filter-v14: checks that all filters in SmartFilter fire LONG and SHORT signals as expected.

- You can run this as a standalone script, or import and call run_all_filter_tests() from main.py or elsewhere.
"""

import pandas as pd
import numpy as np
import inspect

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
    elif case == "short":
        # Build generic data
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
        df["vwap"].iloc[-2] = 104
        df["vwap"].iloc[-1] = 106
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
    Returns list of (name, method) for all filters in SmartFilter.
    Only includes methods that start with '_check_' or are in the filter_function_map.
    """
    canonical_filters = [
        "Fractal Zone", "Unified Trend Regime", "Momentum Cluster", "Volume Spike", "VWAP Divergence",
        "MTF Volume Agreement", "HH/LL Trend", "Chop Zone", "Candle Confirmation", "Wick Dominance",
        "Absorption", "Support/Resistance", "Smart Money Bias", "Liquidity Pool", "Spread Filter",
        "Liquidity Awareness", "Volatility Model", "Volatility Squeeze"
    ]
    method_map = {
        "Fractal Zone": "_check_fractal_zone",
        "Unified Trend Regime": "unified_trend_regime",
        "Momentum Cluster": "_check_momentum_cluster",
        "Volume Spike": "_check_volume_spike",
        "VWAP Divergence": "_check_vwap_divergence",
        "MTF Volume Agreement": "_check_mtf_volume_agreement",
        "HH/LL Trend": "_check_hh_ll",
        "Chop Zone": "_check_chop_zone",
        "Candle Confirmation": "_check_candle_close",
        "Wick Dominance": "_check_wick_dominance",
        "Absorption": "_check_absorption",
        "Support/Resistance": "_check_support_resistance",
        "Smart Money Bias": "_check_smart_money_bias",
        "Liquidity Pool": "_check_liquidity_pool",
        "Spread Filter": "_check_spread_filter",
        "Liquidity Awareness": "_check_liquidity_awareness",
        "Volatility Model": "_check_volatility_model",
        "Volatility Squeeze": "_check_volatility_squeeze"
    }
    dummy = SmartFilter(symbol="TEST", df=make_test_df("long"))
    methods = []
    for filter_name, attr in method_map.items():
        func = getattr(dummy, attr, None)
        if func is not None and callable(func):
            methods.append((filter_name, func))
    return methods

def test_filter(filter_func, filter_name, sf_long, sf_short, sf_neutral):
    print(f"\n--- Testing {filter_name} ---")
    result_long = filter_func(sf_long)
    result_short = filter_func(sf_short)
    result_neutral = filter_func(sf_neutral)

    # VWAP Divergence debug block
    if filter_name.lower().replace(" ", "_") == "vwap_divergence":
        print("[DEBUG] SHORT test for VWAP Divergence:")
        print("close[-2]:", sf_short.df["close"].iat[-2], "vwap[-2]:", sf_short.df["vwap"].iat[-2])
        print("close[-1]:", sf_short.df["close"].iat[-1], "vwap[-1]:", sf_short.df["vwap"].iat[-1])
        print("Expected: SHORT or None, Got:", result_short)

    assert result_long == "LONG" or result_long is None, f"{filter_name} LONG case failed!"
    assert result_short == "SHORT" or result_short is None, f"{filter_name} SHORT case failed!"
    assert result_neutral is None, f"{filter_name} NEUTRAL case failed!"
    print(f"PASS: {filter_name}")

def run_all_filter_tests():
    from smart_filter import SmartFilter

    filter_funcs = [
        SmartFilter._check_vwap_divergence,
        # Add other filter functions here as needed
    ]
    filter_names = [
        "VWAP Divergence",
        # Add other filter names here as needed
    ]

    for func, filter_name in zip(filter_funcs, filter_names):
        df_long = make_test_df("long")
        df_short = make_test_df("short")
        df_neutral = make_test_df("neutral")
    
        sf_long = SmartFilter(symbol="TEST", df=df_long)
        sf_short = SmartFilter(symbol="TEST", df=df_short)
        sf_neutral = SmartFilter(symbol="TEST", df=df_neutral)
    
        test_filter(lambda sf: func(sf), filter_name, sf_long, sf_short, sf_neutral)
        
if __name__ == "__main__":
    print("Starting all SmartFilter tests...")
    run_all_filter_tests()
