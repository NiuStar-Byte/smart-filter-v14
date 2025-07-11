# pec_backtest.py

import pandas as pd
import csv
from smart_filter import SmartFilter
from pec_engine import run_pec_check
from telegram_alert import send_telegram_file
import os
import datetime
import uuid
from collections import defaultdict
import re
import subprocess
import sys

# Set this to limit simulation to only recent fired signals (e.g., 720 minutes = 12 hours)
MINUTES_LIMIT = 720

def parse_fired_signals_from_logs(minutes_limit=720, log_sources=None):
    """
    Parse fired signals directly from logs within the specified time limit.
    
    Args:
        minutes_limit (int): Only consider signals fired within this many minutes
        log_sources (list): List of log sources to search. If None, searches common locations.
    
    Returns:
        list: List of signal dictionaries with parsed information
    """
    signals = []
    now = pd.Timestamp.utcnow()
    cutoff_time = now - pd.Timedelta(minutes=minutes_limit)
    
    # Pattern to match [FIRED] log entries
    fired_pattern = r'\[FIRED\] Logged: ([^,]+), ([^,]+), ([^,]+), ([^,]+), ([^,]+), (.+)'
    
    if log_sources is None:
        # Default log sources to search
        log_sources = [
            'logs.txt',
            '/tmp/smart_filter.log',
            '/var/log/smart_filter.log'
        ]
        
        # Also try to get recent console output if available
        try:
            # Try to get journalctl logs for current session
            result = subprocess.run(['journalctl', '--since', f'{minutes_limit} minutes ago', '--no-pager'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                log_sources.append('journalctl_output')
                # Write journalctl output to a temporary location for parsing
                with open('/tmp/journalctl_output.log', 'w') as f:
                    f.write(result.stdout)
                log_sources.append('/tmp/journalctl_output.log')
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
    
    # Search through all available log sources
    for log_source in log_sources:
        if not os.path.exists(log_source):
            continue
            
        try:
            with open(log_source, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    match = re.search(fired_pattern, line)
                    if match:
                        try:
                            fired_uuid, symbol, tf, signal_type, fired_time_str, entry_idx_str = match.groups()
                            
                            # Parse timestamp
                            fired_time = pd.to_datetime(fired_time_str.strip())
                            
                            # Check if within time limit
                            if fired_time < cutoff_time:
                                continue
                                
                            signals.append({
                                "uuid": fired_uuid.strip(),
                                "symbol": symbol.strip(),
                                "tf": tf.strip(),
                                "signal_type": signal_type.strip(),
                                "fired_time": fired_time_str.strip(),
                                "entry_idx": int(entry_idx_str.strip())
                            })
                            
                        except (ValueError, IndexError) as e:
                            print(f"[WARNING] Failed to parse FIRED log line {line_num} in {log_source}: {e}")
                            continue
                            
        except Exception as e:
            print(f"[WARNING] Failed to read log source {log_source}: {e}")
            continue
    
    print(f"[LOG_PARSER] Found {len(signals)} fired signals from logs within last {minutes_limit} minutes")
    return signals
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

def load_fired_signals(minutes_limit=None):
    """
    DEPRECATED: Use parse_fired_signals_from_logs() instead.
    This function is kept for backward compatibility.
    """
    print("[WARNING] load_fired_signals() is deprecated. Use parse_fired_signals_from_logs() for log-based parsing.")
    return parse_fired_signals_from_logs(minutes_limit or MINUTES_LIMIT)

def save_to_csv(results, filename="pec_results.csv"):
    headers = ["Signal Type", "Symbol", "TF", "Entry Time", "Entry Price", "Exit Price",
               "PnL ($)", "PnL (%)", "Score", "Max Score", "Confidence", "Weighted Confidence",
               "Gatekeepers Passed", "Filter Results", "GK Flags", "Result", "Exit Time", "# BAR Exit", "Signal Time"]

    with open(filename, mode='w', newline='') as file:  # Changed from 'a' to 'w' to overwrite
        writer = csv.writer(file)
        writer.writerow(headers)  # Always write header for new file
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
    print(f"[{datetime.datetime.now()}] [PEC_BACKTEST] PEC results saved to {filename} ({len(results)} signals)")

def run_pec_backtest(
    TOKENS,
    get_ohlcv,
    get_local_wib,
    PEC_WINDOW_MINUTES,  # Renamed from PEC_BARS to match function signature expectations
    PEC_BARS,
    OHLCV_LIMIT,
):
    """
    Run PEC backtest using log-based fired signal parsing.
    
    Args:
        TOKENS: List of trading symbols
        get_ohlcv: Function to fetch OHLCV data
        get_local_wib: Function to convert timezone to WIB
        PEC_WINDOW_MINUTES: Time window in minutes to look back for fired signals
        PEC_BARS: Number of bars to simulate after signal entry
        OHLCV_LIMIT: Limit for OHLCV data fetching
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    pec_file = f"pec_results_{timestamp}.csv"  # Updated naming format

    print(f"[BACKTEST PEC] Running PEC simulation for FIRED signals from last {PEC_WINDOW_MINUTES} minutes...")

    # 1. Load fired signals from logs instead of CSV
    signals = parse_fired_signals_from_logs(minutes_limit=PEC_WINDOW_MINUTES)
    print(f"[BACKTEST PEC] Loaded {len(signals)} fired signals from logs")
    
    signals_by_symbol_tf = defaultdict(list)
    for sig in signals:
        signals_by_symbol_tf[(sig["symbol"], sig["tf"])].append(sig)

    pec_blocks = []

    # 2. Loop over all tokens and timeframes
    timeframes = ["3m", "5m"]  # <-- adjust this list if you use other timeframes
    for symbol in TOKENS:
        for tf in timeframes:
            relevant_signals = signals_by_symbol_tf.get((symbol, tf), [])
            if not relevant_signals:
                print(f"[BACKTEST PEC] Skipping {symbol} {tf} (no fired signals)")
                continue
            print(f"[BACKTEST PEC] {symbol} {tf} ...")
            for signal in relevant_signals:
                entry_idx = signal["entry_idx"]
                fired_time = pd.to_datetime(signal["fired_time"])

                df = get_ohlcv(symbol, interval=tf, limit=OHLCV_LIMIT)
                if df is None or df.empty or entry_idx >= len(df):
                    print(f"[PEC] No data or entry_idx out of range for {symbol} {tf}. Skipping.")
                    continue

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
                signal_type = res.get("bias", "LONG").upper()
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

    save_to_csv(pec_blocks, pec_file)
    print(f"[BACKTEST PEC] Completed PEC backtest. Output: {pec_file} with {len(pec_blocks)} signals processed.")

    print("[DEBUG] Sending PEC results file to Telegram...")
    send_telegram_file(pec_file, caption=f"PEC results from last {PEC_WINDOW_MINUTES} minutes [{timestamp}]")
    print("[BACKTEST PEC] All done. PEC results saved to", pec_file)
