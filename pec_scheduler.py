import time
import os
from datetime import datetime
from pec_backtest import run_pec_backtest
from kucoin_data import get_ohlcv
from telegram_alert import send_telegram_file
from main import get_local_wib

# === CONFIGURATION ===
TOKENS = [
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]
PEC_WINDOW_MINUTES = 720     # Adjust as needed
PEC_BARS = 5
OHLCV_LIMIT = 1000           # Adjust as needed
INTERVAL_SECONDS = 60        # 1 minute

def is_backtest_mode():
    # Reads the same variable as main.py (case-insensitive).
    return os.getenv("PEC_BACKTEST_ONLY", "false").lower() == "true"
    
def run_pec_scheduler():
    """Schedule and execute the PEC engine with the latest fired signals"""
    while True:
        # Example of running backtest logic
        print("[INFO] Running PEC backtest...")
        run_pec_backtest(
            TOKENS, get_ohlcv, get_local_wib,
            PEC_WINDOW_MINUTES, PEC_BARS, OHLCV_LIMIT
        )
        time.sleep(60)  # Wait before running the next cycle

def main():
    print(f"[{datetime.now()}] [SCHEDULER] Starting PEC Backtest Scheduler (1-minute interval).")
    first_run = True
    while True:
        # Log scheduler status
        print(f"[{datetime.now()}] [SCHEDULER] Checking if backtest mode is on...")

        if is_backtest_mode():
            print(f"[{datetime.now()}] [SCHEDULER] Backtest mode is active.")
            
            if first_run:
                print(f"[{datetime.now()}] [SCHEDULER] First run after switching to backtest mode!")
            else:
                print(f"[{datetime.now()}] [SCHEDULER] Scheduled run (1 minute interval).")

            # Execute PEC backtest
            try:
                print(f"[{datetime.now()}] [SCHEDULER] Running PEC backtest for all tokens...")
                
                # Run PEC backtest with updated logic for Exit Time and # BAR Exit
                results = run_pec_backtest(TOKENS, get_ohlcv, get_local_wib, PEC_WINDOW_MINUTES, PEC_BARS, OHLCV_LIMIT)
                
                # Here you can send the results or save them, for now we just print the completion message
                print(f"[{datetime.now()}] [SCHEDULER] PEC backtest completed successfully.")
                
                # Optionally, you can process and send alerts for each result
                send_telegram_file(results)  # If you want to send the file via Telegram (optional)
                
            except Exception as e:
                print(f"[{datetime.now()}] [SCHEDULER] Error in backtest: {e}")

            first_run = False
        else:
            print(f"[{datetime.now()}] [SCHEDULER] Not in backtest mode. Waiting for mode switch...")
            time.sleep(5)  # Check every 5 seconds if backtest mode is on

        # Sleep for 1 minute and log
        print(f"[{datetime.now()}] [SCHEDULER] Sleeping for {INTERVAL_SECONDS/60:.1f} minutes...")
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
