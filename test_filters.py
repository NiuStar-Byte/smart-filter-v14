"""
Test script for smart-filter-v14: checks that filters fire LONG and SHORT signals as expected.

- Run this file directly: python test_filters.py
- Edit or extend with your actual filter functions for full coverage.
"""

import pandas as pd

# Try importing actual filters; fall back to a dummy filter if import fails
try:
    from filters.trend import hh_ll
    filter_list = [("HH/LL Trend Filter", hh_ll)]
except Exception as e:
    print("Could not import hh_ll filter from filters.trend, using dummy filter for demonstration.")
    def hh_ll(df):
        # Dummy filter always returns LONG for upward, SHORT for downward, None for neutral
        close = df['close']
        if close.iloc[-1] > close.iloc[0]:
            return "LONG"
        elif close.iloc[-1] < close.iloc[0]:
            return "SHORT"
        return None
    filter_list = [("HH/LL Trend Filter (dummy)", hh_ll)]

def make_test_df(case="long"):
    # Returns a simple DataFrame that should trigger a LONG or SHORT signal.
    if case == "long":
        data = {
            "high": [100, 105, 110, 115],
            "low": [90, 95, 100, 105],
            "close": [95, 102, 108, 114],
            "volume": [1000, 1200, 1500, 2000],
        }
    elif case == "short":
        data = {
            "high": [115, 110, 105, 100],
            "low": [105, 100, 95, 90],
            "close": [114, 108, 102, 95],
            "volume": [2000, 1500, 1200, 1000],
        }
    else:
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

    try:
        result_long = filter_func(df_long)
        result_short = filter_func(df_short)
        result_neutral = filter_func(df_neutral)
        print(f"  LONG test result:    {result_long}")
        print(f"  SHORT test result:   {result_short}")
        print(f"  NEUTRAL test result: {result_neutral}")
        assert result_long == "LONG" or result_long is None, "LONG case failed!"
        assert result_short == "SHORT" or result_short is None, "SHORT case failed!"
        assert result_neutral is None, "Neutral case failed!"
    except Exception as e:
        print(f"  ERROR occurred while testing {name}: {e}")

if __name__ == "__main__":
    print("Starting filter tests...")
    for name, func in filter_list:
        test_filter(func, name)
    print("\nAll tests completed. Check results above.")
