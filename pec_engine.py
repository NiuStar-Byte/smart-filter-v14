# pec_engine.py

import pandas as pd
from datetime import datetime
from exit_condition_debug import log_exit_conditions

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
        score: int/float, total SmartFilter score
        confidence: float, SmartFilter confidence
        passed_gk_count: int, number of passed GK filters
    Returns:
        result: dict with key stats & verdicts, PLUS diagnostics
    """
    try:
        print(f"[PEC_ENGINE] Starting PEC check for {symbol} at index {entry_idx}.")
        
        # Get the relevant data for backtest
        pec_data = ohlcv_df.iloc[entry_idx: entry_idx + pec_bars + 1].copy()
        if pec_data.shape[0] < pec_bars + 1:
            print(f"[PEC_ENGINE] Not enough data for PEC: {symbol} at index {entry_idx}.")
            return {"status": "not enough data for PEC"}
        
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
    
        # Entry Follow-Through: Did it move at least +0.5% in your favor?
        follow_through = max_up >= 0.5

        # Trailing Stop Survival (0.5% from entry)
        stop_width = 0.5 / 100 * entry_price
        survived = True
        for bar in pec_data.itertuples():
            if signal_type == "LONG" and bar.low < entry_price - stop_width:
                survived = False
                break
            if signal_type == "SHORT" and bar.high > entry_price + stop_width:
                survived = False
                break

        # Signal Persistence (How many closes in same direction?)
        up_bars = ((pec_data["close"] > entry_price) if signal_type == "LONG" else (pec_data["close"] < entry_price)).sum()

        # Volume Confirmation (at least 3/5 bars have above-average volume)
        avg_vol = ohlcv_df["volume"].iloc[max(0, entry_idx-30):entry_idx].mean()
        vol_pass = (pec_data["volume"].iloc[1:] > avg_vol).sum() >= 3

def check_exit_conditions():
    # Your logic for calculating exit conditions
    
    # Assuming you've already calculated these variables:
    exit_time = None
    exit_price = None
    follow_through = False
    stop_survival = False
    volume_condition = False
    condition_met = False

    # Logic to determine if exit conditions are met
    if some_exit_condition:
        exit_price = calculated_exit_price
        exit_time = calculated_exit_time
        condition_met = True

    # Log exit conditions
    log_exit_conditions(exit_time, exit_price, follow_through, stop_survival, volume_condition, condition_met)
        
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

        # Find the time of exit (Exit Time) when the condition is met
        exit_time = None
        exit_price = None
        if follow_through:  # Assuming exit is based on the follow-through condition
            exit_price = pec_data["close"].max()  # Exit price (take profit condition)
            exit_time = ohlcv_df.index[exit_bar]  # Exit time is the timestamp of the exit bar

def track_exit_conditions_example(pec_data, follow_through, stop_survival, volume_condition):
    # Track when the exit price and time are populated based on conditions in pec_data.
    exit_time = None
    exit_price = None
    condition_met = False

    # If follow-through condition is met
    if follow_through:
        exit_price = pec_data["close"].max()
        exit_time = pec_data.index[-1]  # Assuming pec_data has datetime index

    # If trailing stop condition is met
    if stop_survival:
        # Example of stop survival condition
        if pec_data["close"].iloc[-1] < pec_data["close"].iloc[-2] * 0.995:  # Stop condition at 0.5% decrease
            exit_price = pec_data["close"].iloc[-1]
            exit_time = pec_data.index[-1] 

    # If volume condition is met
    if volume_condition:
        # Example of volume condition logic
        if pec_data["volume"].iloc[-1] > pec_data["volume"].mean():
            exit_price = pec_data["close"].iloc[-1]
            exit_time = pec_data.index[-1]

    # Call the logging function to record the exit conditions
    log_exit_conditions(exit_time, exit_price, follow_through, stop_survival, volume_condition, condition_met)
        
        # If exit condition isn't met, leave exit time and exit price as N/A
        if exit_time is None or exit_price is None:
            exit_time = "N/A"
            exit_price = "N/A"


        
        result = {
            "symbol": symbol,
            "tf": tf,
            "signal_type": signal_type,
            "entry_price": entry_price,
            "entry_time": str(ohlcv_df.index[entry_idx]),
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
