# pec_backtest.py

import pandas as pd
import csv
from smart_filter import SmartFilter
from pec_engine import run_pec_check
from telegram_alert import send_telegram_file
import os
import datetime
import uuid
import pytz

# Add these timezone helpers at the top of your script
wib = pytz.timezone('Asia/Jakarta')
utc = pytz.UTC
# Set this to limit simulation to only recent fired signals (e.g., 720 minutes = 12 hours)
MINUTES_LIMIT = 720

def to_wib(dt):
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=utc)
    return dt.astimezone(wib)

def to_utc(dt):
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=utc)
    return dt.astimezone(utc)

def log_fired_signal(symbol, tf, signal_type, fired_time, entry_idx, csv_path="fired_signals_temp.csv"):
    try:
        fired_uuid = str(uuid.uuid4())
        file_exists = os.path.isfile(csv_path)
        abs_path = os.path.abspath(csv_path)

        # Ensure fired_time is always local (WIB) and formatted
        import pytz
        from datetime import datetime
        wib = pytz.timezone('Asia/Jakarta')
        if isinstance(fired_time, datetime):
            if fired_time.tzinfo is None:
                fired_time = wib.localize(fired_time)
            fired_time = fired_time.astimezone(wib).strftime('%Y-%m-%d %H:%M:%S')

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
    signals = []
    try:
        with open("fired_signals_temp.csv", "r") as file:
            next(file)  # Skip header
            for line in file:
                columns = line.strip().split(",")
                uuid_, symbol, tf, signal_type, fired_time, entry_idx = columns
                fired_dt = pd.to_datetime(fired_time)
                if minutes_limit is not None:
                    now = pd.Timestamp.utcnow()
                    if (now - fired_dt).total_seconds() > minutes_limit * 60:
                        continue
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
    headers = ["Signal Type", "Symbol", "TF", "Entry Time", "Entry Price", "Exit Price",
               "PnL ($)", "PnL (%)", "Score", "Max Score", "Confidence", "Weighted Confidence",
               "Gatekeepers Passed", "Filter Results", "GK Flags", "Result", "Exit Time", "# BAR Exit", "Signal Time"]

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(headers)
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
    print(f"[{datetime.datetime.now()}] [SCHEDULER] PEC results saved to {filename}")

def run_pec_backtest(
    TOKENS,
    get_ohlcv,
    get_local_wib,
    PEC_BARS,
    OHLCV_LIMIT,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    pec_file = f"pec_fired_results_{timestamp}.csv"

    print(f"[BACKTEST PEC] Running PEC simulation for FIRED signals from last {MINUTES_LIMIT} minutes...")

    signals = load_fired_signals(minutes_limit=MINUTES_LIMIT)
    pec_blocks = []

    for signal in signals:
        symbol = signal["symbol"]
        tf = signal["tf"]
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

        entry_time_utc = times[entry_idx]
        exit_time_utc = times[exit_idx]
        signal_time_utc = times[entry_idx]  # Use the correct signal_time if available

        entry_time_str = to_wib(entry_time_utc).strftime("%Y-%m-%d %H:%M:%S")      # ENTRY TIME in WIB
        exit_time_str = to_utc(exit_time_utc).strftime("%Y-%m-%d %H:%M:%S")        # EXIT TIME in UTC
        signal_time_str = to_utc(signal_time_utc).strftime("%Y-%m-%d %H:%M:%S")    # SIGNAL TIME in UTC
        
        pec_result = {
            'signal_type': signal_type,
            'symbol': symbol,
            'tf': tf,
            'entry_time': entry_time_str,      # local time (WIB)
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
            'exit_time': exit_time_str,        # UTC
            'exit_bar': bar_exit,
            'signal_time': signal_time_str,    # UTC
        }
        pec_blocks.append(pec_result)

    save_to_csv(pec_blocks, pec_file)
    print(f"[DEBUG] {pec_file} written, {len(pec_blocks)} signals.")

    print("[DEBUG] Sending PEC fired-signal file to Telegram...")
    send_telegram_file(pec_file, caption=f"PEC results for FIRED signals from last {MINUTES_LIMIT} minutes [{timestamp}]")
    print("[BACKTEST PEC] All done. PEC logs grouped in", pec_file)
