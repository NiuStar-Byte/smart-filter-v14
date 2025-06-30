# pec_scheduler.py

import os
from datetime import datetime
from kucoin_data import get_ohlcv
from telegram_alert import send_telegram_file
from pec_backtest import run_pec_backtest
from main import get_local_wib  # Added to pass as argument

# === CONFIGURATION ===
TOKENS = [
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]
PEC_WINDOW_MINUTES = 500  # The window of data retrieval in minutes
PEC_BARS = 5
OHLCV_LIMIT = 1000  # Number of bars to retrieve for each signal
INTERVAL_SECONDS = 60  # Frequency to run the process (1 minute)

def fetch_ohlcv_data(symbol, timeframe, pec_window_minutes=PEC_WINDOW_MINUTES):
    """
    Fetch OHLCV data for a given symbol and timeframe, considering PEC_WINDOW_MINUTES.
    """
    if timeframe == '5min':
        bars_needed = pec_window_minutes // 5  # 500 minutes ÷ 5 minutes per bar
    elif timeframe == '3min':
        bars_needed = pec_window_minutes // 3  # 500 minutes ÷ 3 minutes per bar
    
    # Fetch OHLCV data
    ohlcv_data = get_ohlcv(symbol, interval=timeframe, limit=bars_needed)
    return ohlcv_data

def run_backtest():
    print(f"[{datetime.now()}] [SCHEDULER] Running PEC backtest for all tokens...")
    for symbol in TOKENS:
        # Example: Running PEC for each token
        try:
            ohlcv_data = fetch_ohlcv_data(symbol, "5min")  # Adjust timeframe as needed
            # Run PEC backtest with get_local_wib for timestamp conversion
            run_pec_backtest([symbol], "5min", PEC_WINDOW_MINUTES, PEC_BARS, ohlcv_data, get_local_wib)
            print(f"[{datetime.now()}] [SCHEDULER] PEC backtest for {symbol} completed successfully.")
        except Exception as e:
            print(f"[{datetime.now()}] [SCHEDULER] Error in backtest for {symbol}: {e}")

def main():
    print(f"[{datetime.now()}] [SCHEDULER] Starting PEC Process Scheduler (1-minute interval).")
    
    # Run backtest immediately
    run_backtest()

    # Sleep for the defined interval and repeat if necessary
    print(f"[{datetime.now()}] [SCHEDULER] Sleeping for {INTERVAL_SECONDS/60:.1f} minutes...")
    time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
