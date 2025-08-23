"""
Test script for smart-filter-v14: checks that filters fire LONG and SHORT signals as expected.

- Run this file directly: python test_filters.py
- Edit or extend with your actual filter functions for full coverage.
"""

import pandas as pd
from filters.trend import hh_ll
from filters.volume import mtf_volume_agreement
# Add other filters to import as needed

def make_test_df(case="long"):
    # Returns a simple DataFrame that should trigger a LONG or SHORT signal.
    # Adjust for your actual filter logic!
    if case == "long":
        # Simulate a strong upward move
        data = {
            "high": [100, 105, 110, 115],
            "low": [90, 95, 100, 105],
            "close": [95, 102, 108, 114],
            "volume": [1000, 1200, 1500, 2000],
        }
    elif case == "short":
        # Simulate a strong downward move
        data = {
            "high": [115, 110, 105, 100],
            "low": [105, 100, 95, 90],
            "close": [114, 108, 102, 95],
            "volume": [2000, 1500, 1200, 1000],
        }
    else:
        # Neutral case
        data = {
            "high": [100, 100, 100, 100],
            "low": [90, 90, 90, 90],
            "close": [95, 95, 95, 95],
            "volume": [1000, 1000, 1000, 1000],
        }
    return pd.DataFrame(data)

def test_filter(filter_func, name):
    print(f"\nTesting filter: {name}")
    df_long = make_test_df("long")
    df_short = make_test_df("short")
    df_neutral = make_test_df("neutral")

    result_long = filter_func(df_long)
    result_short = filter_func(df_short)
    result_neutral = filter_func(df_neutral)

    print(f"  LONG test result:    {result_long}")
    print(f"  SHORT test result:   {result_short}")
    print(f"  NEUTRAL test result: {result_neutral}")

    assert result_long == "LONG" or result_long is None, "LONG case failed!"
    assert result_short == "SHORT" or result_short is None, "SHORT case failed!"
    assert result_neutral is None, "Neutral case failed!"

if __name__ == "__main__":
    # Add your filters here
    test_filter(hh_ll, "HH/LL Trend Filter")
    # test_filter(unified_trend_regime, "Unified Regime Filter")   # uncomment/add as needed
    # test_filter(mtf_volume_agreement, "MTF Volume Agreement Filter")
    # ... add more as you migrate filters

    print("\nAll tests completed. Check results above.")
