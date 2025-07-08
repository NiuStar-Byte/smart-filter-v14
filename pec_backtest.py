# pec_backtest.py

import pandas as pd
import csv
from smart_filter import SmartFilter
from pec_engine import run_pec_check
from telegram_alert import send_telegram_file
import os
import datetime
import uuid

def log_fired_signal(symbol, tf, signal_type, fired_time, entry_idx, csv_path="fired_signals_temp.csv"):
    """
    Appends a fired signal with a UUID to the fired_signals_temp.csv file.
    """
    try:
        fired_uuid = str(uuid.uuid4())
        file_exists = os.path.isfile(csv_path)
        abs_path = os.path.abspath(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write header if file is empty
            if not file_exists or os.stat(csv_path).st_size == 0:
                writer.writerow(['uuid','symbol','tf','signal_type','fired_time','entry_idx'])
            writer.writerow([fired_uuid, symbol, tf, signal_type, fired_time, entry_idx])
            file.flush()
            os.fsync(file.fileno())
        print(f"[FIRED] Logged: {fired_uuid}, {symbol}, {tf}, {signal_type}, {fired_time}, {entry_idx}")
        print(f"[DEBUG] Written to: {abs_path}")
        return fired_uuid
    except Exception as e:
        print(f"[ERROR] Failed to log fired signal: {e}")
        return None

def load_fired_signals():
    """Load successfully fired signals from the CSV for backtesting"""
    signals = []
    try:
        with open("fired_signals_temp.csv", "r") as file:
            next(file)  # Skip header
            for line in file:
                columns = line.strip().split(",")
                uuid_, symbol, tf, signal_type, fired_time, entry_idx = columns
                signals.append({
                    "uuid": uuid_,
                    "symbol": symbol,
                    "tf": tf,
                    "signal_type": signal_type,
                    "fired_time": fired_time,
                    "entry_idx": int(entry_idx)
                })
    except Exception as e:
        print(f"[ERROR] Failed to load fired signals: {e}")
    return signals
    
def save_to_csv(results, filename="pec_results.csv"):
    # Define the column headers
    headers = ["Signal Type", "Symbol", "TF", "Entry Time", "Entry Price", "Exit Price", 
               "PnL ($)", "PnL (%)", "Score", "Max Score", "Confidence", "Weighted Confidence", 
               "Gatekeepers Passed", "Filter Results", "GK Flags", "Result", "Exit Time", "# BAR Exit", "Signal Time"]
     
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(headers)
        for result in results:
            # Format signal time (remove microseconds)
            signal_time = result.get("signal_time")
            if isinstance(signal_time, (datetime.datetime, pd.Timestamp)):
                signal_time = signal_time.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(signal_time, str) and "." in signal_time:
                signal_time = signal_time.split(".")[0]
            # Write row
            writer.writerow([
                result.get('signal_type', ''),
                result.get('symbol', ''),
                result.get('tf', ''),
                result.get('entry_time', ''),
                result.get('entry_price', ''),
                result.get('exit_price', ''),
                result.get('pnl_abs', ''),
                result.get('pnl_pct', ''),
                result.get('score', ''),
                result.get('score_max', ''),
                result.get('confidence', ''),
                result.get('weighted_confidence', ''),
                result.get('gatekeepers_passed', ''),
                result.get('filter_results', ''),
                result.get('gk_flags', ''),
                result.get('win_loss', ''),
                result.get('exit_time', ''),
                result.get('exit_bar', ''),
                signal_time or ''
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    long_file = f"pec_long_results_{timestamp}.csv"
    short_file = f"pec_short_results_{timestamp}.csv"

    print(f"[BACKTEST PEC] Running PEC simulation for last {PEC_WINDOW_MINUTES} minutes on ALL tokens: {', '.join(TOKENS)}...")

    pec_counter = 1
    long_blocks = []
    short_blocks = []

    """Run backtest on the fired signals"""
    signals = load_fired_signals()
    
    for signal in signals:
        symbol = signal["symbol"]
        tf = signal["tf"]
 
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
                    # Exit time and # BAR Exit
                    exit_time = times[entry_idx + PEC_BARS]
                    bar_exit = PEC_BARS
                    # Signal time (fired bar time, not "now")
                    signal_time = times[entry_idx].replace(microsecond=0)
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
                        'exit_time': exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'exit_bar': bar_exit,
                        'signal_time': signal_time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    # Append result to respective block
                    if signal_type == "LONG":
                        long_blocks.append(pec_result)
                    else:
                        short_blocks.append(pec_result)
                    pec_counter += 1
            print(f"[BACKTEST PEC] Done for {symbol} {tf}.")

    save_to_csv(long_blocks, long_file)
    save_to_csv(short_blocks, short_file)

    print(f"[DEBUG] {long_file} written, {len(long_blocks)} signals.")
    print(f"[DEBUG] {short_file} written, {len(short_blocks)} signals.")

    print("[DEBUG] Sending PEC long file to Telegram...")
    send_telegram_file(long_file, caption=f"All PEC LONG results for ALL tokens [{timestamp}]")
    print("[DEBUG] Sending PEC short file to Telegram...")
    send_telegram_file(short_file, caption=f"All PEC SHORT results for ALL tokens [{timestamp}]")
    print("[BACKTEST PEC] All done. PEC logs grouped in", long_file, "and", short_file)
