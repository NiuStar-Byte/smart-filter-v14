import pandas as pd
from datetime import datetime
from exit_condition_debug import log_exit_conditions
import logging
from exit_logs_tele import log_exit_conditions
import pytz  # To handle time zone conversion

# Set timezone to WIB (Western Indonesian Time)
WIB = pytz.timezone('Asia/Jakarta')

def process_signal_in_pec(signal, df):
    """
    Process a single fired signal for backtesting.

    Args:
        signal (dict): A dictionary containing the signal details.
        df (DataFrame): The OHLCV data frame for the specific symbol and timeframe.

    Returns:
        dict: The result of the processed signal, including PnL calculations and additional data.
    """
    symbol = signal["symbol"]
    tf = signal["tf"]
    signal_type = signal["signal_type"]
    fired_time = signal["fired_time"]
    entry_idx = signal["entry_idx"]

    # Retrieve the entry price and calculate the PnL based on the next 5 bars
    entry_price = df.iloc[entry_idx]["close"]
    exit_idx = entry_idx + PEC_BARS  # Look ahead 5 bars
    
    # Retrieve the entry price and calculate the PnL based on the next 5 bars
    entry_price = df.iloc[entry_idx]["close"]
    exit_idx = entry_idx + PEC_BARS  # Look ahead 5 bars

    # Force Exit Rules (TP and SL)
    tp = entry_price * 1.20  # 20% Take Profit
    sl = entry_price * 0.90  # 10% Stop Loss

    for i in range(entry_idx, entry_idx + PEC_BARS):
        current_price = df.iloc[i]["close"]
        
        # Check if the current price hits the TP or SL
        if current_price >= tp:
            exit_price = tp
            pnl = (exit_price - entry_price)
            pnl_percent = (pnl / entry_price) * 100  # Percentage profit
            exit_time = df.iloc[i]["time"]
            return {"symbol": signal["symbol"], "entry_price": entry_price, "exit_price": exit_price, 
                    "pnl": pnl, "pnl_percent": pnl_percent, "exit_time": exit_time, "exit_reason": "Take Profit"}

        elif current_price <= sl:
            exit_price = sl
            pnl = (entry_price - exit_price)
            pnl_percent = (pnl / entry_price) * 100  # Percentage loss
            exit_time = df.iloc[i]["time"]
            return {"symbol": signal["symbol"], "entry_price": entry_price, "exit_price": exit_price, 
                    "pnl": pnl, "pnl_percent": pnl_percent, "exit_time": exit_time, "exit_reason": "Stop Loss"}

    
    if exit_idx < len(df):  # Ensure we don't go out of bounds
        exit_price = df.iloc[entry_idx + PEC_BARS]["close"]

        # Calculate PnL
        pnl = (exit_price - entry_price) if signal_type == "LONG" else (entry_price - exit_price)
        pnl_percent = (pnl / entry_price) * 100  # Calculate PnL percentage

        # Return the results for this signal
        return {
            "symbol": symbol,
            "tf": tf,
            "signal_type": signal_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "fired_time": fired_time,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "exit_time": df.iloc[entry_idx + PEC_BARS]["time"]
        }
    else:
        # If not enough data for 5 bars, return None or an empty result
        return None

def run_pec_check(
    symbol,
    entry_idx,
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
        entry_idx: int, index of entry bar in ohlcv_df
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
        print(f"[PEC_ENGINE] Starting PEC check for {symbol} at index {entry_idx}.")

        # Capture Signal Time in UTC
        signal_time_utc = datetime.now(pytz.utc)  # Capture time in UTC first
        signal_time = signal_time_utc.astimezone(WIB)  # Convert to WIB time zone

        # Get the relevant data for backtest
        pec_data = ohlcv_df.iloc[entry_idx: entry_idx + pec_bars + 1].copy()
        if pec_data.shape[0] < pec_bars + 1:
            print(f"[PEC_ENGINE] Not enough data for PEC: {symbol} at index {entry_idx}.")
            return {"status": "not enough data for PEC"}

        # ENTRY TIME from OHLCV
        entry_time = str(ohlcv_df.index[entry_idx].tz_localize('UTC').astimezone(WIB))  # Convert entry time to WIB

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

        # Capture the signal time
        signal_time = datetime.datetime.now()  # Current datetime when the signal is fired

        # Get the entry time (datetime index from OHLCV data)
        entry_time = str(ohlcv_df.index[entry_idx].tz_localize('UTC').astimezone(WIB))  # Convert entry time to WIB

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
        log_exit_conditions(exit_time, exit_price, follow_through, stop_survival, vol_pass, condition_met)
        
        try:
            # Disabled exit checks here
            # if follow_through:  # Assuming exit is based on the follow-through condition
            #     exit_price = pec_data["close"].max()  # Exit price (take profit condition)
            #     exit_time = ohlcv_df.index[entry_idx + pec_bars]  # Exit time is the timestamp of the exit bar

            # if stop_survival:  # Check if trailing stop condition met
            #     if pec_data["close"].iloc[-1] < pec_data["close"].iloc[-2] * 0.995:
            #         exit_price = pec_data["close"].iloc[-1]
            #         exit_time = ohlcv_df.index[entry_idx + pec_bars]  # Exit time when the condition is met

            # if volume_condition:  # Check if volume condition met
            #     if pec_data["volume"].iloc[-1] > pec_data["volume"].mean():
            #         exit_price = pec_data["close"].iloc[-1]
            #         exit_time = ohlcv_df.index[entry_idx + pec_bars]  # Exit time when volume condition met

            # Add this log statement just before calling log_exit_conditions
            logging.debug("Preparing to log exit conditions.")
            
            # Log exit conditions
            log_exit_conditions(exit_time, exit_price, follow_through, stop_survival, vol_pass, condition_met)
        
        except Exception as e:
            logging.error(f"Error in logging exit conditions: {e}")
        
        # Format diagnostics for filter-level pass/fail
        filter_diag_str = ""
        if filter_result is not None and isinstance(filter_result, dict):
            filter_diag_str = "\n- Filter Diagnostics:"
            for fname, fval in filter_result.items():
                filter_diag_str += f"\n    {fname}: {'✅' if fval else '❌'}"

        summary = f"""# PEC for {symbol} {tf} {signal_type} @ {entry_price:.5f} (Fired: {ohlcv_df.index[entry_idx]})
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
