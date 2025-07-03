import pandas as pd
import csv
from smart_filter import SmartFilter
from pec_engine import run_pec_check
from telegram_alert import send_telegram_file
import os
import datetime
import pytz

def save_to_csv(results, filename="pec_results.csv"):
    # Define the column headers
    headers = ["Signal Type", "Symbol", "TF", "Entry Time", "Entry Price", "Exit Price", 
               "PnL ($)", "PnL (%)", "Score", "Max Score", "Confidence", "Weighted Confidence", 
               "Gatekeepers Passed", "Filter Results", "GK Flags", "Result", "Exit Time", "# BAR Exit", "Signal Time"]
     
    # Open the CSV file for writing (mode 'a' appends to the file)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write headers only if the file is empty (using file.tell())
        if file.tell() == 0:
            writer.writerow(headers)

        # Write each result row
        for result in results:
            try:
                entry_time = result["Entry Time"]
                exit_time = result["Exit Time"]
                signal_time = result["Signal Time"]

                # Convert all times to UTC and strip microseconds
                if isinstance(entry_time, pd.Timestamp):
                    entry_time = entry_time.to_pydatetime()
                if isinstance(exit_time, pd.Timestamp):
                    exit_time = exit_time.to_pydatetime()
                if isinstance(signal_time, pd.Timestamp):
                    signal_time = signal_time.to_pydatetime()

                if entry_time.tzinfo is not None:
                    entry_time = entry_time.astimezone(pytz.UTC)
                if exit_time.tzinfo is not None:
                    exit_time = exit_time.astimezone(pytz.UTC)
                if signal_time.tzinfo is not None:
                    signal_time = signal_time.astimezone(pytz.UTC)

                entry_time_str = entry_time.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
                exit_time_str = exit_time.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
                signal_time_str = signal_time.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                entry_time_str = str(result["Entry Time"])
                exit_time_str = str(result["Exit Time"])
                signal_time_str = str(result["Signal Time"])

        # Write the PEC result data to the CSV
        for result in results:
            writer.writerow([
                result["Signal Type"],
                result["Symbol"],
                result["TF"],
                result["Entry Time"].replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S") if hasattr(result["Entry Time"], "strftime") else result["Entry Time"],
                result["Entry Price"],
                result["Exit Price"],
                result["PnL ($)"],
                result["PnL (%)"],
                result["Score"],
                result["Max Score"],
                result["Confidence"],
                result["Weighted Confidence"],
                result["Gatekeepers Passed"],
                result["Filter Results"],
                result["GK Flags"],
                result["Result"],
                result["Exit Time"].replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S") if hasattr(result["Exit Time"], "strftime") else result["Exit Time"],
                result["# BAR Exit"],
                result["Signal Time"].replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S") if hasattr(result["Signal Time"], "strftime") else result["Signal Time"]
            ])
  
    print(f"[{datetime.datetime.now()}] [SCHEDULER] PEC results saved to {filename}")


def run_pec_backtest(
    TOKENS,
    get_ohlcv,
    get_local_wib,
    PEC_WINDOW_MINUTES,
    PEC_BARS,
    OHLCV_LIMIT,
):
    """
    Runs PEC backtest on ALL tokens from TOKENS.
    Uses $100 per trade, 5-bar fixed exit. Exports clean block with all detail,
    including filter pass/fail, GK pass, total score, confidence, etc.
    Output filenames are timestamped for version tracking.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    long_file = f"pec_long_results_{timestamp}.csv"
    short_file = f"pec_short_results_{timestamp}.csv"

    print(f"[BACKTEST PEC] Running PEC simulation for last {PEC_WINDOW_MINUTES} minutes on ALL tokens: {', '.join(TOKENS)}...")

    pec_counter = 1
    long_blocks = []
    short_blocks = []

    for symbol in TOKENS:
        for tf, tf_minutes in [("3min", 3), ("5min", 5)]:
            print(f"[BACKTEST PEC] {symbol} {tf} ...")
            df = get_ohlcv(symbol, interval=tf, limit=OHLCV_LIMIT)
            if df is None or df.empty or len(df) < PEC_BARS + 2:
                print(f"[BACKTEST PEC] No data for {symbol} {tf}. Skipping.")
                continue
            times = pd.to_datetime(df.index)
            window_start = times[-1] - pd.Timedelta(minutes=PEC_WINDOW_MINUTES)
            candidate_indices = [i for i in range(len(df) - PEC_BARS) if times[i] >= window_start]
            for i in candidate_indices:
                df_slice = df.iloc[:i+1]
                sf = SmartFilter(symbol, df_slice, df3m=df_slice, df5m=df_slice, tf=tf)
                res = sf.analyze()
                if isinstance(res, dict) and res.get("valid_signal") is True:
                    entry_idx = i
                    if entry_idx + PEC_BARS >= len(df):
                        continue
                    entry_price = float(df["close"].iloc[entry_idx])
                    exit_idx = entry_idx + PEC_BARS
                    exit_price = float(df["close"].iloc[exit_idx])
                    fired_dt = times[entry_idx]
                    local_time_str = get_local_wib(fired_dt)
                    signal_type = res.get("bias", "LONG").upper()
                    score = res.get("score")
                    score_max = res.get("score_max")
                    passes = res.get("passes")
                    gk_total = res.get("gatekeepers_total")
                    confidence = res.get("confidence")
                    weighted = res.get("passed_weight")
                    total_weight = res.get("total_weight")

                    # Add Exit Time and Bar Exit here if necessary
                    exit_time = res.get("exit_time", "N/A")  # Fallback to "N/A" if not available
                    bar_exit = res.get("exit_bar", "N/A")    # Fallback to "N/A" if not available

                    # Filter-level pass/fail export
                    filter_results = res.get("filter_results", {})
                    filter_passes = {k: ("✅" if v else "❌") for k, v in filter_results.items()}
                    filter_pass_str = ", ".join(f"{k}:{v}" for k, v in filter_passes.items())

                    # GK-level pass/fail export
                    gk_flags = getattr(sf, "gatekeepers", [])
                    gk_pass_str = ", ".join(str(gk) for gk in gk_flags)

                    # $100 notional logic
                    if signal_type == "LONG":
                        pnl_abs = 100 * (exit_price - entry_price) / entry_price
                    else:  # SHORT
                        pnl_abs = 100 * (entry_price - exit_price) / entry_price
                    pnl_pct = 100 * pnl_abs / 100
                    win_loss = "WIN" if pnl_abs > 0 else "LOSS"

                    # Get Exit Time and # BAR Exit (populated with mock data)
                    exit_time = times[entry_idx + PEC_BARS]  # Set Exit Time as the timestamp of the exit bar
                    bar_exit = PEC_BARS  # Set Exit Bar as the number of bars after the entry

                    # Capture signal time & ensure signal_time is always a valid datetime object
                    signal_time = datetime.datetime.now().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S") # Remove microseconds

                    # Compose data for CSV export
                    pec_result = {
                        'signal_type': signal_type,
                        'symbol': symbol,
                        'tf': tf,
                        'entry_time': local_time_str,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_abs': pnl_abs,
                        'pnl_pct': pnl_pct,
                        'score': score,
                        'score_max': score_max,
                        'confidence': confidence,
                        'weighted_confidence': weighted,
                        'gatekeepers_passed': passes,
                        'filter_results': filter_pass_str,
                        'gk_flags': gk_pass_str,
                        'win_loss': win_loss,
                        'exit_time': exit_time,  # Exit Time
                        'exit_bar': bar_exit,    # # BAR Exit
                        'signal_time': signal_time  # Signal Time
                    }

                    # Append result to respective block
                    if signal_type == "LONG":
                        long_blocks.append(pec_result)
                    else:
                        short_blocks.append(pec_result)
                    pec_counter += 1
            print(f"[BACKTEST PEC] Done for {symbol} {tf}.")

    # Save the results to CSV
    save_to_csv(long_blocks, long_file)
    save_to_csv(short_blocks, short_file)

    print(f"[DEBUG] {long_file} written, {len(long_blocks)} signals.")
    print(f"[DEBUG] {short_file} written, {len(short_blocks)} signals.")

    # Send the CSV files to Telegram
    print("[DEBUG] Sending PEC long file to Telegram...")
    send_telegram_file(long_file, caption=f"All PEC LONG results for ALL tokens [{timestamp}]")
    print("[DEBUG] Sending PEC short file to Telegram...")
    send_telegram_file(short_file, caption=f"All PEC SHORT results for ALL tokens [{timestamp}]")

    print("[BACKTEST PEC] All done. PEC logs grouped in", long_file, "and", short_file)
