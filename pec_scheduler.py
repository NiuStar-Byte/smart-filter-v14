import time
import os
from pec_backtest import run_pec_backtest
from kucoin_data import get_ohlcv
# If you have a utils.py or similar, adjust import accordingly.
# Otherwise, place your get_local_wib definition here.
from main import get_local_wib  # If not in utils, import from main
from datetime import datetime

# === CONFIGURATION ===
TOKENS = [
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]
PEC_WINDOW_MINUTES = 500     # Adjust as needed
PEC_BARS = 5
OHLCV_LIMIT = 1000           # Adjust as needed
INTERVAL_SECONDS = 60        # 1 minute

def is_backtest_mode():
    # Reads the same variable as main.py (case-insensitive).
    return os.getenv("PEC_BACKTEST_ONLY", "false").lower() == "true"

def main():
    print(f"[{datetime.now()}] [SCHEDULER] Starting PEC Backtest Scheduler (1-minute interval).")
    first_run = True
    while True:
        print(f"[{datetime.now()}] [SCHEDULER] Checking if backtest mode is on...")

        if is_backtest_mode():
            print(f"[{datetime.now()}] [SCHEDULER] Backtest mode is active.")

            if first_run:
                print(f"[{datetime.now()}] [SCHEDULER] First run after switching to backtest mode!")
            else:
                print(f"[{datetime.now()}] [SCHEDULER] Scheduled run (1 minute interval).")

            # Run PEC backtest output
            try:
                run_pec_backtest(TOKENS, get_ohlcv, get_local_wib, PEC_WINDOW_MINUTES, PEC_BARS, OHLCV_LIMIT)
                print(f"[{datetime.now()}] [SCHEDULER] PEC backtest completed successfully.")
            except Exception as e:
                print(f"[{datetime.now()}] [SCHEDULER] Error in backtest: {e}")

            first_run = False
        else:
            print(f"[{datetime.now()}] [SCHEDULER] Not in backtest mode. Waiting for mode switch...")
            time.sleep(5)  # Check every 5 seconds if backtest mode is on

        # Sleep for 1 minute (for testing purposes)
        print(f"[{datetime.now()}] [SCHEDULER] Sleeping for {INTERVAL_SECONDS/60:.1f} minutes...")
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
