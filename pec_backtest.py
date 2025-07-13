# pec_backtest.py

import csv
import pandas as pd
import re
import os
import datetime
from datetime import timezone
from collections import defaultdict
from smart_filter import SmartFilter
from pec_engine import run_pec_check
from telegram_alert import send_telegram_file

# Set this to limit simulation to only recent fired signals (e.g., 720 minutes = 12 hours)
MINUTES_LIMIT = 720


# Configuration: Log file path for reading fired signals
# Can be changed to point to different log files if needed
FIRED_SIGNALS_LOG_PATH = "logs.txt"

def load_fired_signals_from_log(log_file_path=None, minutes_limit=None):
    """
    Parse fired signals from LIVE MODE LOGS, extracting only those from the last 'minutes_limit' minutes.
    Expected log format: [FIRED] Logged: {uuid}, {symbol}, {tf}, {signal_type}, {local_timestamp}, {other_field}
    
    Args:
        log_file_path (str): Path to log file. If None, uses FIRED_SIGNALS_LOG_PATH.
        minutes_limit (int): Only include signals from last N minutes using local timestamp. If None, includes all.
    
    Returns:
        tuple: (signals_list, unique_symbols_set, unique_timeframes_set)
    """
    signals = []
    unique_symbols = set()
    unique_timeframes = set()
    
    if log_file_path is None:
        log_file_path = FIRED_SIGNALS_LOG_PATH
    
    try:
        if not os.path.exists(log_file_path):
            print(f"[LOG_PARSER] Log file not found: {log_file_path}")
            return signals, unique_symbols, unique_timeframes
            
        with open(log_file_path, 'r') as file:
            log_content = file.read()
        
        # Regex pattern to match FIRED log entries from LIVE MODE
        # Pattern: [FIRED] Logged: uuid, symbol, tf, signal_type, local_timestamp, other_field
        pattern = r'\[FIRED\] Logged: ([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*(\d+)'
        
        lines = log_content.split('\n')
        now_utc = datetime.datetime.now(timezone.utc)
        
        if minutes_limit is not None:
            cutoff_time = now_utc - datetime.timedelta(minutes=minutes_limit)
        else:
            cutoff_time = None
        
        for line in lines:
            match = re.search(pattern, line.strip())
            if match:
                uuid_val, symbol, tf, signal_type, fired_time_str, entry_idx = match.groups()
                
                try:
                    # Parse the local timestamp - handle ISO format with microseconds
                    if 'T' in fired_time_str:
                        # ISO format: 2025-07-13T09:10:28.802346 or with timezone
                        if '+' in fired_time_str or 'Z' in fired_time_str:
                            fired_dt = datetime.datetime.fromisoformat(fired_time_str.replace('Z', '+00:00'))
                        else:
                            # Treat as UTC if no timezone specified
                            fired_dt = datetime.datetime.fromisoformat(fired_time_str).replace(tzinfo=timezone.utc)
                    else:
                        # Try parsing as simple datetime string
                        fired_dt = datetime.datetime.strptime(fired_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    
                    # Always collect symbols and timeframes for automated processing
                    symbol_clean = symbol.strip()
                    tf_clean = tf.strip()
                    unique_symbols.add(symbol_clean)
                    unique_timeframes.add(tf_clean)
                    
                    # Filter by time limit if specified (using local timestamp from logs)
                    if cutoff_time is None or fired_dt >= cutoff_time:
                        signals.append({
                            "uuid": uuid_val.strip(),
                            "symbol": symbol_clean,
                            "tf": tf_clean,
                            "signal_type": signal_type.strip(),
                            "fired_time": fired_time_str.strip(),
                            "entry_idx": int(entry_idx.strip())
                        })
                        
                except Exception as e:
                    print(f"[LOG_PARSER] Failed to parse signal line: {line.strip()}")
                    print(f"[LOG_PARSER] Error: {e}")
                    continue
        
        print(f"[LOG_PARSER] Found {len(signals)} fired signals from LIVE MODE LOGS from last {minutes_limit or 'all'} minutes")
        print(f"[LOG_PARSER] Discovered {len(unique_symbols)} unique symbols: {sorted(unique_symbols)}")
        print(f"[LOG_PARSER] Discovered {len(unique_timeframes)} unique timeframes: {sorted(unique_timeframes)}")
        
    except Exception as e:
        print(f"[LOG_PARSER] Error reading log file {log_file_path}: {e}")
    
    return signals, unique_symbols, unique_timeframes

# OBSOLETE: CSV-based signal loading (kept for backward compatibility)
# This function is now replaced by load_fired_signals_from_log()

def log_fired_signal(symbol, tf, signal_type, fired_time, entry_idx, csv_path="fired_signals_temp.csv"):
    """
    DEPRECATED: This function is kept for backward compatibility but should not be used.
    The new workflow uses log-based parsing instead of CSV files.
    """
    try:
        fired_uuid = str(uuid.uuid4())
        file_exists = os.path.isfile(csv_path)
        abs_path = os.path.abspath(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
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

# OBSOLETE: CSV-based signal loading (kept for backward compatibility)
# This function is now replaced by load_fired_signals_from_log()
def load_fired_signals(minutes_limit=None):
    """
    DEPRECATED: Use load_fired_signals_from_log() instead.
    This function is kept for backward compatibility and now uses LIVE MODE log parsing.
    """
    print("[WARNING] load_fired_signals() is deprecated. Use load_fired_signals_from_log() for LIVE MODE log-based parsing.")
    signals, _, _ = load_fired_signals_from_log(minutes_limit=minutes_limit or MINUTES_LIMIT)
    return signals

def save_to_csv(results, filename="pec_results.csv"):
    headers = ["Signal Type", "Symbol", "TF", "Entry Time", "Entry Price", "Exit Price",
               "PnL ($)", "PnL (%)", "Score", "Max Score", "Confidence", "Weighted Confidence",
               "Gatekeepers Passed", "Filter Results", "GK Flags", "Result", "Exit Time", "# BAR Exit", "Signal Time"]

    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists or os.stat(filename).st_size == 0:
            writer.writerow(headers)  # Write header only if file doesn't exist or is empty
        for result in results:
            signal_time = result.get("signal_time")
            if isinstance(signal_time, (datetime.datetime, pd.Timestamp)):
                signal_time = signal_time.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(signal_time, str) and "." in signal_time:
                signal_time = signal_time.split(".")[0]
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
    print(f"[{datetime.datetime.now()}] [PEC_BACKTEST] PEC results appended to {filename} ({len(results)} signals)")
    
def run_pec_backtest(
    TOKENS,
    get_ohlcv,
    get_local_wib,
    PEC_WINDOW_MINUTES,  # Renamed from PEC_BARS to match function signature expectations
    PEC_BARS,
    OHLCV_LIMIT,
):
    """
    Run PEC backtest using LIVE MODE log-based fired signal parsing.
    Automatically detects all symbols and timeframes from the logs - no manual configuration needed.
    
    Args:
        TOKENS: List of trading symbols (DEPRECATED - now auto-detected from logs)
        get_ohlcv: Function to fetch OHLCV data
        get_local_wib: Function to convert timezone to WIB
        PEC_WINDOW_MINUTES: Time window in minutes to look back for fired signals (default 720 = 12 hours)
        PEC_BARS: Number of bars to simulate after signal entry
        OHLCV_LIMIT: Limit for OHLCV data fetching
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    pec_file = f"pec_results_{timestamp}.csv"  # Updated naming format

    print(f"[BACKTEST PEC] Running PEC simulation for FIRED signals from LIVE MODE LOGS (last {PEC_WINDOW_MINUTES} minutes)...")

    # 1. Load fired signals from LIVE MODE logs and auto-detect symbols/timeframes
    signals, discovered_symbols, discovered_timeframes = load_fired_signals_from_log(minutes_limit=PEC_WINDOW_MINUTES)
    print(f"[BACKTEST PEC] Loaded {len(signals)} fired signals from LIVE MODE LOGS")
    
    if not signals:
        print("[BACKTEST PEC] No fired signals found in LIVE MODE LOGS. Nothing to process.")
        return
    
    signals_by_symbol_tf = defaultdict(list)
    for sig in signals:
        signals_by_symbol_tf[(sig["symbol"], sig["tf"])].append(sig)

    pec_blocks = []

    # 2. Process ALL discovered symbols and timeframes automatically (no hardcoding)
    print(f"[BACKTEST PEC] Processing {len(discovered_symbols)} symbols and {len(discovered_timeframes)} timeframes...")
    
    for symbol in sorted(discovered_symbols):
        for tf in sorted(discovered_timeframes):
            relevant_signals = signals_by_symbol_tf.get((symbol, tf), [])
            if not relevant_signals:
                continue  # Skip if no signals for this symbol/tf combination
                
            print(f"[BACKTEST PEC] Processing {symbol} {tf} ({len(relevant_signals)} signals)...")
            
            for signal in relevant_signals:
                entry_idx = signal["entry_idx"]
                fired_time = pd.to_datetime(signal["fired_time"])

                df = get_ohlcv(symbol, interval=tf, limit=OHLCV_LIMIT)
                if df is None or df.empty or entry_idx >= len(df):
                    print(f"[PEC] No data or entry_idx out of range for {symbol} {tf}. Skipping.")
                    continue
                    
                print(f"[DEBUG] Fired_time from log: {fired_time}")
                print(f"[DEBUG] Fired_time floored: {fired_time.floor('5T')}")
                print(f"[DEBUG] Last 5 DataFrame bars: {df.index[-5:]}")
                print(f"[DEBUG] DataFrame bar at entry_idx ({entry_idx}): {df.index[entry_idx]}")
    
                times = pd.to_datetime(df.index)
                if not pd.Timestamp(times[entry_idx]).floor('s') == fired_time.floor('s'):
                    print(f"[PEC] entry_idx time mismatch for {symbol} {tf} @ idx {entry_idx}. Skipping.")
                    continue

                df_slice = df.iloc[:entry_idx+1]
                sf = SmartFilter(symbol, df_slice, df3m=df_slice, df5m=df_slice, tf=tf)
                res = sf.analyze()
                if not (isinstance(res, dict) and res.get("valid_signal") is True):
                    print(f"[PEC] Signal not valid at idx {entry_idx} for {symbol} {tf}. Skipping.")
                    continue

                if entry_idx + PEC_BARS >= len(df):
                    print(f"[PEC] Not enough bars for exit simulation for {symbol} {tf} @ idx {entry_idx}. Skipping.")
                    continue

                entry_price = float(df["close"].iloc[entry_idx])
                exit_idx = entry_idx + PEC_BARS
                exit_price = float(df["close"].iloc[exit_idx])
                # Use signal_type from logs instead of analyzing again
                signal_type = signal.get("signal_type", "LONG").upper()
                score = res.get("score")
                score_max = res.get("score_max")
                passes = res.get("passes")
                gk_total = res.get("gatekeepers_total")
                confidence = res.get("confidence")
                weighted = res.get("passed_weight")
                total_weight = res.get("total_weight")
                filter_results = res.get("filter_results", {})
                filter_passes = {k: ("✅" if v else "❌") for k, v in filter_results.items()}
                filter_pass_str = ", ".join(f"{k}:{v}" for k, v in filter_passes.items())
                gk_flags = getattr(sf, "gatekeepers", [])
                gk_pass_str = ", ".join(str(gk) for gk in gk_flags)
                
                # Calculate PnL based on signal direction from logs
                if signal_type == "LONG":
                    pnl_abs = 100 * (exit_price - entry_price) / entry_price
                else:
                    pnl_abs = 100 * (entry_price - exit_price) / entry_price
                    
                pnl_pct = 100 * pnl_abs / 100
                win_loss = "WIN" if pnl_abs > 0 else "LOSS"
                exit_time = times[exit_idx]
                bar_exit = PEC_BARS
                signal_time = times[entry_idx].replace(microsecond=0)

                pec_result = {
                    'signal_type': signal_type,
                    'symbol': symbol,
                    'tf': tf,
                    'entry_time': signal_time.strftime("%Y-%m-%d %H:%M:%S"),
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
                pec_blocks.append(pec_result)
            print(f"[BACKTEST PEC] Done for {symbol} {tf}.")

    # 3. Combine all LONG and SHORT signals into single CSV output
    save_to_csv(pec_blocks, pec_file)
    print(f"[BACKTEST PEC] Completed PEC backtest. Output: {pec_file} with {len(pec_blocks)} signals processed.")
    
    # Summary of processed signals by type
    long_signals = [r for r in pec_blocks if r['signal_type'] == 'LONG']
    short_signals = [r for r in pec_blocks if r['signal_type'] == 'SHORT']
    print(f"[BACKTEST PEC] Combined results: {len(long_signals)} LONG signals, {len(short_signals)} SHORT signals")

    print("[DEBUG] Sending PEC results file to Telegram...")
    send_telegram_file(pec_file, caption=f"PEC results from LIVE MODE logs (last {PEC_WINDOW_MINUTES} minutes) [{timestamp}]")
    print("[BACKTEST PEC] All done. PEC results saved to", pec_file)
