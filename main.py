import sys
log_file = open("logs.txt", "a")
sys.stdout = sys.stderr = log_file

import os
import time
import pandas as pd
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert

# Only 15 available KuCoin Futures tokens (validated via API)
tokens = [
    "SPARK-USDT", "BID-USDT", "SKATE-USDT", "LA-USDT", "SPK-USDT",
    "ZKJ-USDT", "IP-USDT", "AERO-USDT", "BMT-USDT", "LQTY-USDT",
    "FUN-USDT", "SNT-USDT", "X-USDT", "BANK-USDT", "RAY-USDT"
]

cooldown_tracker = {}
cooldown_minutes = {
    "2m": 5,
    "3m": 12,
    "5m": 15
}

def run():
    timeframes = ["3m", "5m"]
    for tf in timeframes:
        print(f"\n[INFO] Checking timeframe: {tf}")
        for token in tokens:
            key = f"{token}-{tf}"
            cooldown = cooldown_minutes[tf]

            # Cooldown enforcement
            last_triggered = cooldown_tracker.get(key, 0)
            if time.time() - last_triggered < cooldown * 60:
                print(f"[COOLDOWN] Skipping {token} ({tf}) — cooldown active.")
                continue

            print(f"\n[CHECK] Running Smart Filter for {token} ({tf})")
            try:
                df = get_ohlcv(token, tf)
                if df is None or len(df) < 50:
                    print(f"[ERROR] No data for {token} on {tf}")
                    continue

                sf = SmartFilter(df, token, tf)
                result = sf.analyze()

                if result["signal"]:
                    send_telegram_alert(result)
                    cooldown_tracker[key] = time.time()
                else:
                    print(f"[{token}] ❌ No signal.")

            except Exception as e:
                print(f"[ERROR] Exception for {token} ({tf}): {e}")

if __name__ == "__main__":
    run()

