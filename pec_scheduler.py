import time
from datetime import datetime
from kucoin_data import get_ohlcv  # Assuming get_ohlcv is imported from kucoin_data.py
from pec_engine import run_pec_check  # Assuming run_pec_check is imported from pec_engine.py
from pec_backtest import run_pec_backtest  # Import run_pec_backtest here to avoid circular dependency
from telegram_alert import send_telegram_file

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

def is_backtest_mode():
    """
    Check if the system is in backtest mode by reading the PEC_BACKTEST_ONLY environment variable.
    """
    return os.getenv("PEC_BACKTEST_ONLY", "false").lower() == "true"

def fetch_ohlcv_data(symbol, timeframe, pec_window_minutes=PEC_WINDOW_MINUTES):
    """
    Fetch OHLCV data for a given symbol and timeframe, considering PEC_WINDOW_MINUTES.
    Args:
        symbol (str): The symbol to fetch data for (e.g., 'BTC-USDT').
        timeframe (str): Timeframe for OHLCV (e.g., '5min').
        pec_window_minutes (int): The time window in minutes to retrieve data for.
    Returns:
        DataFrame: A pandas DataFrame with the OHLCV data.
    """
    if timeframe == '5min':
        bars_needed = pec_window_minutes // 5  # 500 minutes ÷ 5 minutes per bar
    elif timeframe == '3min':
        bars_needed = pec_window_minutes // 3  # 500 minutes ÷ 3 minutes per bar
    
    # Fetch OHLCV data
    ohlcv_data = get_ohlcv(symbol, interval=timeframe, limit=bars_needed)
    return ohlcv_data

def run_backtest():
    """
    Run the backtest for all tokens and write the results to CSV.
    """
    print(f"[{datetime.now()}] [SCHEDULER] Running PEC backtest for all tokens...")
    
    for symbol in TOKENS:
        # Fetch OHLCV data for each token
        try:
            ohlcv_data = fetch_ohlcv_data(symbol, "5min")  # Adjust timeframe as needed
            # Run PEC backtest for each symbol
            run_pec_backtest(TOKENS, get_ohlcv, get_local_wib, PEC_WINDOW_MINUTES, PEC_BARS, OHLCV_LIMIT)
            print(f"[{datetime.now()}] [SCHEDULER] PEC backtest for {symbol} completed successfully.")
        except Exception as e:
            print(f"[{datetime.now()}] [SCHEDULER] Error in backtest for {symbol}: {e}")

def main():
    """
    Main function to run the PEC Process Scheduler.
    """
    print(f"[{datetime.now()}] [SCHEDULER] Starting PEC Process Scheduler (1-minute interval).")
    
    # Run backtest immediately
    run_backtest()

    # Sleep for the defined interval and repeat if necessary
    print(f"[{datetime.now()}] [SCHEDULER] Sleeping for {INTERVAL_SECONDS/60:.1f} minutes...")
    time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
