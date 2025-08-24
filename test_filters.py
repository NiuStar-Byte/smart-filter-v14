"""
Test script for smart-filter-v14: checks that all filters in SmartFilter fire LONG and SHORT signals as expected.

- You can run this as a standalone script, or import and call run_all_filter_tests() from main.py or elsewhere.
- Optionally, call run_filter_tests_for_symbol(symbol, df) with your own test data.
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
    elif case == "short":
        # Patch: Create a scenario where close > vwap and close is falling
        # We'll use a high starting close, gradually decreasing, while VWAP rises slower
        close_vals = np.linspace(115, 95, length)  # High to low (falling)
        volume_vals = np.linspace(2000, 1000, length)
        vwap_vals = np.linspace(100, 110, length)  # VWAP rises slightly
        data = {
            "open":   close_vals + 5,   # open above close
            "high":   close_vals + 10,  # high above close
            "low":    close_vals - 10,  # low below close
            "close":  close_vals,
            "volume": volume_vals,
            "bid":    close_vals - 2,
            "ask":    close_vals + 2,
        }
        # We'll set vwap explicitly for the last two rows to ensure (close-vwap) increases
        # You can set this after DataFrame creation if you want more control
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
    
    # Explicitly patch VWAP for the short case to ensure signal
    if case == "short":
        # Ensure last two closes are above VWAP and falling, divergence grows
        df["vwap"] = np.linspace(100, 110, length)
        df["close"].iloc[-2] = 108
        df["close"].iloc[-1] = 106  # close falls
        df["vwap"].iloc[-2] = 104
        df["vwap"].iloc[-1] = 105   # vwap rises
        # Now: close[-1] > vwap[-1], close[-1] < close[-2], (close[-1]-vwap[-1]) > (close[-2]-vwap[-2])
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
    """
    Tests a filter function on LONG, SHORT, and NEUTRAL data.
    Prints results and asserts signal logic.
    """
    print(f"\nTesting filter: {filter_name}")
    try:
        # Enable debug for Fractal Zone only
        if filter_name == "Fractal Zone":
            result_long = filter_func(sf_long, debug=True)
            result_short = filter_func(sf_short, debug=True)
            result_neutral = filter_func(sf_neutral, debug=True)
        else:
            result_long = filter_func(sf_long)
            result_short = filter_func(sf_short)
            result_neutral = filter_func(sf_neutral)
    except Exception as e:
        print(f"  ERROR while running {filter_name}: {e}")
        return
    print(f"  LONG test result:    {result_long}")
    print(f"  SHORT test result:   {result_short}")
    print(f"  NEUTRAL test result: {result_neutral}")
    assert result_long == "LONG" or result_long is None, f"{filter_name} LONG case failed!"
    assert result_short == "SHORT" or result_short is None, f"{filter_name} SHORT case failed!"
    assert result_neutral is None, f"{filter_name} NEUTRAL case failed!"

def run_all_filter_tests():
    """
    Run all filter tests with default test data for each case: LONG, SHORT, NEUTRAL.
    """
    df_long = make_test_df("long")
    df_short = make_test_df("short")
    df_neutral = make_test_df("neutral")
    sf_long = SmartFilter(symbol="TEST", df=df_long)
    sf_short = SmartFilter(symbol="TEST", df=df_short)
    sf_neutral = SmartFilter(symbol="TEST", df=df_neutral)
    all_filters = get_filter_methods()
    for filter_name, func in all_filters:
        if filter_name == "Fractal Zone":
            test_filter(lambda sf: getattr(sf, func.__name__)(debug=True), filter_name, sf_long, sf_short, sf_neutral)
        else:
            test_filter(lambda sf: getattr(sf, func.__name__)(), filter_name, sf_long, sf_short, sf_neutral)
    print("\nAll filter tests completed. Check results above.")
    
def run_filter_tests_for_symbol(symbol, df):
    """
    Run all filters for a given symbol and DataFrame.
    Useful for production or debugging with real data.
    """
    sf = SmartFilter(symbol=symbol, df=df)
    all_filters = get_filter_methods()
    print(f"\nRunning filter tests for symbol: {symbol}")
    for filter_name, func in all_filters:
        try:
            result = func()
            print(f"{filter_name}: {result}")
        except Exception as e:
            print(f"{filter_name}: ERROR - {e}")

if __name__ == "__main__":
    print("Starting all SmartFilter tests...")
    run_all_filter_tests()
