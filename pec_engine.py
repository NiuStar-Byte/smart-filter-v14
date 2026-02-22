# pec_engine.py

import pandas as pd
from datetime import datetime
from exit_condition_debug import log_exit_conditions
import logging
from exit_logs_tele import log_exit_conditions
import pytz  # To handle time zone conversion

# Set timezone to WIB (Western Indonesian Time)
WIB = pytz.timezone('Asia/Jakarta')

def find_closest_ohlcv_bar(fired_time_utc, ohlcv_df, tf):
    """
    Find the closest OHLCV bar to the fired time using timestamp matching.
    
    Args:
        fired_time_utc: pd.Timestamp or datetime in UTC
        ohlcv_df: DataFrame with datetime index in UTC
        tf: timeframe string ('3min', '5min', etc.)
    
    Returns:
        tuple: (bar_index, bar_time, time_diff_minutes) or (None, None, None) if no match
    """
    try:
        if not isinstance(fired_time_utc, pd.Timestamp):
            fired_time_utc = pd.Timestamp(fired_time_utc)
        
        # Ensure fired_time is in UTC
        if fired_time_utc.tz is None:
            fired_time_utc = fired_time_utc.tz_localize('UTC')
        elif fired_time_utc.tz != pd.Timestamp.now().tz_localize('UTC').tz:
            fired_time_utc = fired_time_utc.tz_convert('UTC')
        
        # Ensure OHLCV index is in UTC
        if ohlcv_df.index.tz is None:
            ohlcv_times = pd.to_datetime(ohlcv_df.index).tz_localize('UTC')
        else:
            ohlcv_times = pd.to_datetime(ohlcv_df.index).tz_convert('UTC')
        
        # Find the closest bar by absolute time difference
        time_diffs = (ohlcv_times - fired_time_utc).abs()
        closest_idx = time_diffs.idxmin()
        closest_bar_idx = ohlcv_df.index.get_loc(closest_idx)
        closest_bar_time = ohlcv_times[closest_idx]
        time_diff_minutes = abs((closest_bar_time - fired_time_utc).total_seconds() / 60)
        
        return closest_bar_idx, closest_bar_time, time_diff_minutes
        
    except Exception as e:
        print(f"[TIMESTAMP_MATCH_ERROR] Failed to match timestamp: {e}")
        return None, None, None

def find_realistic_exit(df, entry_idx, entry_price, signal_type, max_bars=20, tp_price=None, sl_price=None, tp_pct=None, sl_pct=None, maker_fee=0.001):
    """
    Find realistic exit price with Take Profit, Stop Loss, and fee modeling.
    
    Args:
        df: DataFrame with OHLCV data
        entry_idx: Index where signal was fired
        entry_price: Entry price (close at signal time)
        signal_type: "LONG" or "SHORT"
        max_bars: Maximum bars to hold (configurable, NOT hardcoded to 20)
        tp_price: Actual TP price from signal (takes precedence over tp_pct)
        sl_price: Actual SL price from signal (takes precedence over sl_pct)
        tp_pct: Take Profit percentage (fallback, default 1.5%)
        sl_pct: Stop Loss percentage (fallback, default 1.0%)
        maker_fee: Maker fee as decimal (KuCoin default 0.001 = 0.1%)
    
    Returns:
        dict: {
            'exit_price': float,
            'exit_reason': str ('TP' | 'SL' | 'TIMEOUT'),
            'exit_idx': int,
            'mfe': float (Max Favorable Excursion %),
            'mae': float (Max Adverse Excursion %)
        }
    """
    if entry_idx + max_bars >= len(df):
        max_bars = len(df) - entry_idx - 1
    
    df_future = df.iloc[entry_idx+1:entry_idx+max_bars+1].copy()
    
    if len(df_future) == 0:
        return None
    
    # Use actual prices from signal if provided, otherwise fall back to percentages
    if tp_price is None:
        tp_pct = tp_pct if tp_pct is not None else 1.5
        if signal_type == "LONG":
            tp_level = entry_price * (1 + tp_pct / 100)
        else:
            tp_level = entry_price * (1 - tp_pct / 100)
    else:
        tp_level = tp_price
    
    if sl_price is None:
        sl_pct = sl_pct if sl_pct is not None else 1.0
        if signal_type == "LONG":
            sl_level = entry_price * (1 - sl_pct / 100)
        else:
            sl_level = entry_price * (1 + sl_pct / 100)
    else:
        sl_level = sl_price
    
    # Calculate TP and SL levels
    if signal_type == "LONG":
        # Track MFE and MAE
        mfe = ((df_future['high'].max() - entry_price) / entry_price) * 100
        mae = ((df_future['low'].min() - entry_price) / entry_price) * 100
        
        # Find first bar that hits TP or SL
        for idx, (i, row) in enumerate(df_future.iterrows()):
            if row['high'] >= tp_level:
                return {
                    'exit_price': tp_level,
                    'exit_reason': 'TP',
                    'exit_idx': entry_idx + idx + 1,
                    'mfe': mfe,
                    'mae': mae
                }
            elif row['low'] <= sl_level:
                return {
                    'exit_price': sl_level,
                    'exit_reason': 'SL',
                    'exit_idx': entry_idx + idx + 1,
                    'mfe': mfe,
                    'mae': mae
                }
    else:  # SHORT
        # Track MFE and MAE
        mfe = ((entry_price - df_future['low'].min()) / entry_price) * 100
        mae = ((entry_price - df_future['high'].max()) / entry_price) * 100
        
        # Find first bar that hits TP or SL
        for idx, (i, row) in enumerate(df_future.iterrows()):
            if row['low'] <= tp_level:
                return {
                    'exit_price': tp_level,
                    'exit_reason': 'TP',
                    'exit_idx': entry_idx + idx + 1,
                    'mfe': mfe,
                    'mae': mae
                }
            elif row['high'] >= sl_level:
                return {
                    'exit_price': sl_level,
                    'exit_reason': 'SL',
                    'exit_idx': entry_idx + idx + 1,
                    'mfe': mfe,
                    'mae': mae
                }
    
    # No TP/SL hit: exit at last bar close (TIMEOUT)
    # Use the close price of the last bar in the window
    timeout_idx = min(entry_idx + len(df_future), len(df) - 1)
    timeout_price = float(df.iloc[timeout_idx]['close'])
    
    mfe = ((df_future['high'].max() - entry_price) / entry_price) * 100 if signal_type == "LONG" else ((entry_price - df_future['low'].min()) / entry_price) * 100
    mae = ((df_future['low'].min() - entry_price) / entry_price) * 100 if signal_type == "LONG" else ((entry_price - df_future['high'].max()) / entry_price) * 100
    
    return {
        'exit_price': timeout_price,  # ← EXPLICIT: this is the timeout exit price
        'timeout_price': timeout_price,  # ← NEW: added for clarity
        'exit_reason': 'TIMEOUT',
        'exit_idx': timeout_idx,
        'mfe': mfe,
        'mae': mae
    }

def process_signal_in_pec(signal, df, maker_fee=0.001, max_bars=None, tp_price=None, sl_price=None):
    """
    Process a single fired signal for realistic backtesting with TP/SL.

    Args:
        signal (dict): A dictionary containing the signal details.
        df (DataFrame): The OHLCV data frame for the specific symbol and timeframe.
        maker_fee (float): Maker fee as decimal (default 0.001 = 0.1% for KuCoin)
        max_bars (int): Maximum bars to hold (overrides default 20)
        tp_price (float): Actual TP price from signal (from JSONL storage)
        sl_price (float): Actual SL price from signal (from JSONL storage)

    Returns:
        dict: The result of the processed signal, including realistic PnL calculations.
    """
    symbol = signal["symbol"]
    tf = signal["tf"]
    signal_type = signal["signal_type"]
    fired_time = signal["fired_time"]
    entry_idx = signal["entry_idx"]
    entry_price = df.iloc[entry_idx]["close"]
    
    # Use provided max_bars or fall back to default 20
    if max_bars is None:
        max_bars = 20
    
    # Use signal's stored TP/SL if provided, otherwise let find_realistic_exit calculate from percentages
    exit_result = find_realistic_exit(
        df, entry_idx, entry_price, signal_type, 
        max_bars=max_bars,
        tp_price=tp_price or signal.get("tp_target"),
        sl_price=sl_price or signal.get("sl_target")
    )
    
    if exit_result is None:
        return None
    
    exit_price = exit_result['exit_price']
    exit_reason = exit_result['exit_reason']
    exit_idx = exit_result['exit_idx']
    
    # Calculate PnL with fee modeling
    if signal_type == "LONG":
        pnl = (exit_price - entry_price) - (entry_price * maker_fee) - (exit_price * maker_fee)
    else:  # SHORT
        pnl = (entry_price - exit_price) - (entry_price * maker_fee) - (exit_price * maker_fee)
    
    pnl_percent = (pnl / entry_price) * 100
    
    return {
        "symbol": symbol,
        "tf": tf,
        "signal_type": signal_type,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "pnl_percent": pnl_percent,
        "mfe": exit_result['mfe'],
        "mae": exit_result['mae'],
        "fired_time": fired_time,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx
    }

def run_pec_check(
    symbol,
    fired_time_utc,     # NEW: Use timestamp instead of entry_idx
    tf,
    signal_type,
    entry_price,
    ohlcv_df,
    pec_bars=5,
    filter_result=None,          # NEW: expects a dict of filter pass/fail per signal
    score=None,                  # NEW: total score (int or float)
    confidence=None,             # NEW: confidence (float or pct)
    passed_gk_count=None,        # NEW: count of passed GK filters
):
    """
    Perform post-entry quality control (PEC) simulation for a fired signal.
    Args:
        symbol: str, e.g. "SPK-USDT"
        fired_time_utc: pd.Timestamp in UTC, timestamp when signal was fired
        tf: str, e.g. "3min"
        signal_type: "LONG" or "SHORT"
        entry_price: float, actual entry price
        ohlcv_df: pd.DataFrame with columns: ["open", "high", "low", "close", ...]
        pec_bars: int, how many bars ahead to check (default 5)
        filter_result: dict, key=filter name, value=True/False (pass/fail)
    Returns:
        result: dict with key stats & verdicts, PLUS diagnostics
    """
    try:
        print(f"[PEC_ENGINE] Starting PEC check for {symbol} using timestamp matching.")

        # Find the entry bar using timestamp matching
        entry_idx, matched_bar_time, time_diff_minutes = find_closest_ohlcv_bar(fired_time_utc, ohlcv_df, tf)
        
        if entry_idx is None:
            return {"status": "timestamp matching failed", "error": "Could not match fired_time to OHLCV bar"}

        print(f"[PEC_ENGINE] Matched signal to bar index {entry_idx} (time diff: {time_diff_minutes:.2f} min)")

        # Capture Signal Time in UTC
        signal_time_utc = fired_time_utc if isinstance(fired_time_utc, pd.Timestamp) else pd.Timestamp(fired_time_utc)
        signal_time = signal_time_utc.astimezone(WIB)  # Convert to WIB time zone

        # Get the relevant data for backtest
        pec_data = ohlcv_df.iloc[entry_idx: entry_idx + pec_bars + 1].copy()
        if pec_data.shape[0] < pec_bars + 1:
            print(f"[PEC_ENGINE] Not enough data for PEC: {symbol} at index {entry_idx}.")
            return {"status": "not enough data for PEC"}

        # ENTRY TIME from matched OHLCV bar
        entry_time = str(matched_bar_time.astimezone(WIB))  # Convert entry time to WIB

        # Calculate MFE and MAE for the trade
        if signal_type == "LONG":
            max_up = (pec_data["high"].max() - entry_price) / entry_price * 100
            max_dn = (pec_data["low"].min() - entry_price) / entry_price * 100
            final_ret = (pec_data["close"].iloc[-1] - entry_price) / entry_price * 100
        else:
            max_up = (entry_price - pec_data["low"].min()) / entry_price * 100
            max_dn = (entry_price - pec_data["high"].max()) / entry_price * 100
            final_ret = (entry_price - pec_data["close"].iloc[-1]) / entry_price * 100

        print(f"[PEC_ENGINE] MFE: {max_up:.2f}%, MAE: {max_dn:.2f}%, Final Return: {final_ret:.2f}%")

        # Count favorable bars
        up_bars = 0
        for i in range(1, len(pec_data)):
            if signal_type == "LONG":
                if pec_data["close"].iloc[i] > pec_data["close"].iloc[0]:
                    up_bars += 1
            else:
                if pec_data["close"].iloc[i] < pec_data["close"].iloc[0]:
                    up_bars += 1

        # Entry Follow-Through: Did it move at least +0.5% in your favor?
        follow_through = False  # Disabled follow-through condition

        # Trailing Stop Survival (0.5% from entry)
        stop_width = 0.5 / 100 * entry_price
        survived = False  # Disabled survival condition

        # Volume Confirmation (at least 3/5 bars have above-average volume)
        avg_vol = ohlcv_df["volume"].iloc[max(0, entry_idx-30):entry_idx].mean()
        vol_pass = False  # Disabled volume condition

        # Logic for logging the exit conditions (EXIT TIME and EXIT PRICE)
        exit_time = None
        exit_price = None
        condition_met = False  # Disabled condition logic

        # Log the exit conditions to both the file and Telegram
        log_exit_conditions(exit_time, exit_price, follow_through, survived, vol_pass, condition_met)
        # Format diagnostics for filter-level pass/fail
        filter_diag_str = ""
        if filter_result is not None and isinstance(filter_result, dict):
            filter_diag_str = "\n- Filter Diagnostics:"
            for fname, fval in filter_result.items():
                filter_diag_str += f"\n    {fname}: {'✅' if fval else '❌'}"

        summary = f"""# PEC for {symbol} {tf} {signal_type} @ {entry_price:.5f} (Fired: {matched_bar_time})
- Follow-Through: {'✅' if follow_through else '❌'} (MaxFavorable={max_up:.2f}%)
- Max Adverse Excursion: {max_dn:.2f}%
- Final Return: {final_ret:.2f}%
- Trailing Stop (0.5%): {'Survived' if survived else 'Stopped'}
- Volume Confirmation: {'PASS' if vol_pass else 'FAIL'}
- Signal Persistence: {up_bars}/{pec_bars} bars favorable
- SmartFilter Score: {score if score is not None else '-'}
- Confidence: {confidence if confidence is not None else '-'}
- GK Passed: {passed_gk_count if passed_gk_count is not None else '-'}{filter_diag_str}
"""

        # Calculate # BAR Exit (last bar number based on pec_bars)
        exit_bar = entry_idx + pec_bars  # Exit after specified number of bars

        result = {
            "symbol": symbol,
            "tf": tf,
            "entry_time": entry_time,  # This will now be correctly populated
            "signal_type": signal_type,
            "entry_price": entry_price,
            "max_favorable": max_up,
            "max_adverse": max_dn,
            "final_return": final_ret,
            "follow_through": follow_through,
            "trailing_stop_survived": survived,
            "volume_confirmation": vol_pass,
            "signal_persistence": up_bars,
            "smartfilter_score": score,
            "confidence": confidence,
            "passed_gk_count": passed_gk_count,
            "filter_level_results": filter_result,
            "exit_time": exit_time,   # NEW: exit time
            "exit_bar": exit_bar,     # NEW: exit bar index
            "signal_time": str(signal_time),  # Capture signal time when the signal is fired
            "summary": summary
        }
        return result

    except Exception as e:
        print(f"[PEC_ENGINE] Error in run_pec_check: {e}")
        return {
            "status": "error",
            "error": str(e),
            "summary": f"# PEC ERROR: {symbol} {tf}: {str(e)}"
        }

def export_pec_log(result, filename="pec_debug_temp.txt", custom_header=None):
    """
    Export a single PEC result (dict) to the given txt file.
    If custom_header is supplied, prepend it as the first line (for numbered, apple-to-apple log).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = result.get("summary", str(result))
    with open(filename, "a") as f:
        if custom_header:
            f.write(f"\n{custom_header}\n")
        else:
            f.write(f"\n# PEC Result Export (created: {now})\n")
        f.write(summary)
        f.write("\n" + "="*32 + "\n")
