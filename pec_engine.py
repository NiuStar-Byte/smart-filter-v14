import pandas as pd
from datetime import datetime
from exit_condition_debug import log_exit_conditions
import logging
from exit_logs_tele import log_exit_conditions as log_exit_conditions_tele
import pytz  # To handle time zone conversion
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter

# === CONFIG ===
PEC_BARS = 5
WIB = pytz.timezone('Asia/Jakarta')
TRACKER_FILE = "fired_signals_temp.csv"
TP_RATIO = 1.20  # 20% TP
SL_RATIO = 0.90  # 10% SL

def load_signal_tracker(filename=TRACKER_FILE):
    """
    Load fired signals (tracker) as DataFrame.
    Must have columns: symbol, tf, signal_type, fired_time, entry_idx
    """
    df = pd.read_csv(filename)
    if "symbol" not in df or "tf" not in df or "signal_type" not in df or "entry_idx" not in df:
        raise Exception(f"{filename} missing required columns.")
    return df

def process_signal_in_pec(signal, ohlcv_df):
    """
    Process a single fired signal for backtesting, with TP/SL exit logic.
    Args:
        signal (Series or dict): must have symbol, tf, signal_type, entry_idx, fired_time
        ohlcv_df (pd.DataFrame): OHLCV data
    Returns:
        dict result with all diagnostics
    """
    try:
        symbol = signal["symbol"]
        tf = signal["tf"]
        signal_type = signal["signal_type"]
        fired_time = signal.get("fired_time", "")
        entry_idx = int(signal["entry_idx"])
        entry_price = ohlcv_df.iloc[entry_idx]["close"]
        exit_idx = entry_idx + PEC_BARS
        tp = entry_price * TP_RATIO
        sl = entry_price * SL_RATIO
        exit_reason = "Time Exit"
        exit_price = None
        exit_time = None

        # Iterate through next 5 bars for TP/SL
        for i in range(entry_idx, min(exit_idx, len(ohlcv_df))):
            current_price = ohlcv_df.iloc[i]["close"]
            if signal_type == "LONG":
                if current_price >= tp:
                    exit_price = tp
                    exit_time = ohlcv_df.index[i]
                    exit_reason = "Take Profit"
                    break
                if current_price <= sl:
                    exit_price = sl
                    exit_time = ohlcv_df.index[i]
                    exit_reason = "Stop Loss"
                    break
            elif signal_type == "SHORT":
                # For SHORT: TP is lower, SL is higher
                if current_price <= entry_price * (2 - TP_RATIO):  # 0.80 for 20% move down
                    exit_price = entry_price * (2 - TP_RATIO)
                    exit_time = ohlcv_df.index[i]
                    exit_reason = "Take Profit"
                    break
                if current_price >= entry_price * (2 - SL_RATIO):  # 1.10 for 10% up move
                    exit_price = entry_price * (2 - SL_RATIO)
                    exit_time = ohlcv_df.index[i]
                    exit_reason = "Stop Loss"
                    break

        # If no TP/SL hit, exit at time bar
        if exit_price is None:
            if exit_idx < len(ohlcv_df):
                exit_price = ohlcv_df.iloc[exit_idx]["close"]
                exit_time = ohlcv_df.index[exit_idx]
            else:
                exit_price = ohlcv_df.iloc[-1]["close"]
                exit_time = ohlcv_df.index[-1]
            exit_reason = "Time Exit"

        pnl = (exit_price - entry_price) if signal_type == "LONG" else (entry_price - exit_price)
        pnl_percent = (pnl / entry_price) * 100

        result = {
            "symbol": symbol,
            "tf": tf,
            "signal_type": signal_type,
            "entry_time": str(ohlcv_df.index[entry_idx]),
            "entry_price": entry_price,
            "exit_time": str(exit_time),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "fired_time": fired_time,
        }
        return result
    except Exception as e:
        print(f"[PEC_ENGINE ERROR] {e}")
        return {"error": str(e), "symbol": signal.get("symbol", "NA")}

def run_pec_batch():
    # Load only successfully fired signals
    tracker_df = load_signal_tracker()
    print(f"[PEC_ENGINE] Loaded {len(tracker_df)} fired signals from tracker.")
    results = []
    for idx, signal in tracker_df.iterrows():
        symbol = signal["symbol"]
        tf = signal["tf"]
        entry_idx = int(signal["entry_idx"])
        # Load OHLCV (from your live or backtest source)
        ohlcv = get_ohlcv(symbol, interval=tf, limit=1000)  # Adjust limit if needed
        if ohlcv is None or entry_idx >= len(ohlcv):
            print(f"[PEC_ENGINE] OHLCV data missing for {symbol} {tf} or bad idx.")
            continue
        result = process_signal_in_pec(signal, ohlcv)
        results.append(result)
    # Save results as CSV
    pd.DataFrame(results).to_csv("pec_output_results.csv", index=False)
    print(f"[PEC_ENGINE] Done! {len(results)} signals processed (see pec_output_results.csv)")

if __name__ == "__main__":
    run_pec_batch()
