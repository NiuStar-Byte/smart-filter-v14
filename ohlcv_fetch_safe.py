#!/usr/bin/env python3
"""
ohlcv_fetch_safe.py

Safe OHLCV fetching with independent TF handling.
Replaces the "skip entire symbol if any TF missing" logic with independent TF processing.

OLD LOGIC (BREAKS ALL 3 TFs IF ONE FAILS):
  if df15 is None or df30 is None or df1h is None:
      skip_symbol

NEW LOGIC (EACH TF INDEPENDENT):
  df15 = safe_fetch_15m()  # May return None
  df30 = safe_fetch_30m()  # May return None
  df1h = safe_fetch_1h()   # May return None
  # Each TF block checks independently
"""

import pandas as pd
from typing import Optional, Dict, Tuple
import traceback

def safe_fetch_ohlcv_by_tf(symbol: str, get_ohlcv_func) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Safely fetch OHLCV data for all 3 timeframes independently.
    
    Args:
        symbol: Trading pair (e.g., "BTC-USDT")
        get_ohlcv_func: Function that takes (symbol, interval, limit) and returns DataFrame
    
    Returns:
        {
            "15min": DataFrame or None,
            "30min": DataFrame or None,
            "1h": DataFrame or None,
            "fetch_errors": {"15min": error_msg or None, ...}
        }
    
    Key: Each TF is fetched independently. If 30m fails, 15m & 1h still work.
    """
    OHLCV_LIMIT = 250
    
    result = {
        "15min": None,
        "30min": None,
        "1h": None,
        "fetch_errors": {
            "15min": None,
            "30min": None,
            "1h": None
        }
    }
    
    # Fetch 15min
    try:
        df15 = get_ohlcv_func(symbol, interval="15min", limit=OHLCV_LIMIT)
        if df15 is not None and not df15.empty:
            result["15min"] = df15
        else:
            result["fetch_errors"]["15min"] = "Empty or None"
    except Exception as e:
        result["fetch_errors"]["15min"] = f"{type(e).__name__}: {str(e)[:50]}"
        print(f"[OHLCV] 15min fetch failed for {symbol}: {result['fetch_errors']['15min']}", flush=True)
    
    # Fetch 30min
    try:
        df30 = get_ohlcv_func(symbol, interval="30min", limit=OHLCV_LIMIT)
        if df30 is not None and not df30.empty:
            result["30min"] = df30
        else:
            result["fetch_errors"]["30min"] = "Empty or None"
    except Exception as e:
        result["fetch_errors"]["30min"] = f"{type(e).__name__}: {str(e)[:50]}"
        print(f"[OHLCV] 30min fetch failed for {symbol}: {result['fetch_errors']['30min']}", flush=True)
    
    # Fetch 1h
    try:
        df1h = get_ohlcv_func(symbol, interval="1h", limit=OHLCV_LIMIT)
        if df1h is not None and not df1h.empty:
            result["1h"] = df1h
        else:
            result["fetch_errors"]["1h"] = "Empty or None"
    except Exception as e:
        result["fetch_errors"]["1h"] = f"{type(e).__name__}: {str(e)[:50]}"
        print(f"[OHLCV] 1h fetch failed for {symbol}: {result['fetch_errors']['1h']}", flush=True)
    
    return result


def check_tf_data_available(ohlcv_result: Dict, timeframe: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    """
    Check if a specific timeframe has data available.
    
    Args:
        ohlcv_result: Result from safe_fetch_ohlcv_by_tf()
        timeframe: "15min", "30min", or "1h"
    
    Returns:
        (data_available: bool, dataframe: DataFrame or None, error: error_msg or None)
    """
    df = ohlcv_result.get(timeframe)
    error = ohlcv_result.get("fetch_errors", {}).get(timeframe)
    
    if df is not None and not df.empty:
        return (True, df, None)
    else:
        return (False, None, error)


def should_skip_symbol(ohlcv_result: Dict) -> Tuple[bool, str]:
    """
    Determine if symbol should be skipped entirely.
    OLD: Skip if ANY TF is missing
    NEW: Skip only if ALL TFs are missing
    
    Args:
        ohlcv_result: Result from safe_fetch_ohlcv_by_tf()
    
    Returns:
        (should_skip: bool, reason: str)
    """
    has_15m = ohlcv_result.get("15min") is not None
    has_30m = ohlcv_result.get("30min") is not None
    has_1h = ohlcv_result.get("1h") is not None
    
    if not (has_15m or has_30m or has_1h):
        return (True, "ALL timeframes missing data")
    
    return (False, "")
