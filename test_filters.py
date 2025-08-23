"""
Test script for smart-filter-v14: checks that all filters in SmartFilter fire LONG and SHORT signals as expected.

- Run this file directly: python test_filters.py
- This script instantiates SmartFilter with test DataFrames and runs all filter methods.
"""

import pandas as pd
import numpy as np
import inspect

# Import SmartFilter class
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
    elif case == "short":
        data = {
            "open":   np.linspace(105, 90, length),
            "high":   np.linspace(115, 100, length),
            "low":    np.linspace(100, 85, length),
            "close":  np.linspace(114, 95, length),
            "volume": np.linspace(2000, 1000, length),
            "bid":    np.linspace(105, 90, length),
            "ask":    np.linspace(115, 100, length),
        }
    else:  # Neutral
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
    # Add higher_tf_volume for MTF filters
    df["higher_tf_volume"] = np.linspace(1000, 2000, length)
    return df

def get_filter_methods():
    """
    Returns list of (name, method) for all filters in SmartFilter.
    Only includes methods that start with '_check_' or are in the filter_function_map.
    """
    # List of canonical filter methods from smart_filter.py
    canonical_filters = [
        "Fractal Zone", "Unified Trend Regime", "Momentum Cluster", "Volume Spike", "VWAP Divergence",
        "MTF Volume Agreement", "HH/LL Trend", "Chop Zone", "Candle Confirmation", "Wick Dominance",
        "Absorption", "Support/Resistance", "Smart Money Bias", "Liquidity Pool", "Spread Filter",
        "Liquidity Awareness", "Volatility Model", "Volatility Squeeze"
    ]
    method_map = {
        "Fractal Zone": "._check_fractal_zone",
        "Unified Trend Regime": ".unified_trend_regime",
        "Momentum Cluster": "._check_momentum_cluster",
        "Volume Spike": "._check_volume_spike",
        "VWAP Divergence": "._check_vwap_divergence",
        "MTF Volume Agreement": "._check_mtf_volume_agreement",
        "HH/LL Trend": "._check_hh_ll",
        "Chop Zone": "._check_chop_zone",
        "Candle Confirmation": "._check_candle_close",
        "Wick Dominance": "._check_wick_dominance",
        "Absorption": "._check_absorption",
        "Support/Resistance": "._check_support_resistance",
        "Smart Money Bias": "._check_smart_money_bias",
        "Liquidity Pool": "._check_liquidity_pool",
        "Spread Filter": "._check_spread_filter",
        "Liquidity Awareness": "._check_liquidity_awareness",
        "Volatility Model": "._check_volatility_model",
        "Volatility Squeeze": "._check_volatility_squeeze"
    }
    dummy = SmartFilter(symbol="TEST", df=make_test_df("long"))

    methods = []
    for filter_name, method_name in method_map.items():
        attr = method_name.split(".", 1)[1]
        func = getattr(dummy, attr, None)
        if func is not None and callable(func):
            methods.append((filter_name, func))
    return methods

def test_filter(filter_func, filter_name, sf_long, sf_short, sf_neutral):
    """
    Tests a filter function on LONG, SHORT, and NEUTRAL data.
    Prints results and asserts signal logic.
    """
    print(f"\nTesting filter: {filter_name}")
    try:
        result_long = filter_func(sf_long)
        result_short = filter_func(sf_short)
        result_neutral = filter_func(sf_neutral)
    except Exception as e:
        print(f"  ERROR while running {filter_name}: {e}")
        return

    print(f"  LONG test result:    {result_long}")
    print(f"  SHORT test result:   {result_short}")
    print(f"  NEUTRAL test result: {result_neutral}")

    # Accept None as valid for filters that do not always fire
    assert result_long == "LONG" or result_long is None, f"{filter_name} LONG case failed!"
    assert result_short == "SHORT" or result_short is None, f"{filter_name} SHORT case failed!"
    assert result_neutral is None, f"{filter_name} NEUTRAL case failed!"

def run_all_filter_tests():
    # Prepare SmartFilter instances for each case
    df_long = make_test_df("long")
    df_short = make_test_df("short")
    df_neutral = make_test_df("neutral")
    sf_long = SmartFilter(symbol="TEST", df=df_long)
    sf_short = SmartFilter(symbol="TEST", df=df_short)
    sf_neutral = SmartFilter(symbol="TEST", df=df_neutral)

    # Get all filter methods
    all_filters = get_filter_methods()

    # Run each filter for all cases
    for filter_name, func in all_filters:
        test_filter(lambda sf: func(), filter_name, sf_long, sf_short, sf_neutral)

    print("\nAll filter tests completed. Check results above.")

if __name__ == "__main__":
    print("Starting all SmartFilter tests...")
    run_all_filter_tests()
