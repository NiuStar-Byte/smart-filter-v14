import sys
log_file = open("logs.txt", "a")
sys.stdout = sys.stderr = log_file

import os
import time
import pandas as pd
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert

# ——— CONFIGURATION ———————————————————————————————————
TOKENS = [
    "SKATE-USDT", "LA-USDT",  "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT",  "BMT-USDT", "LQTY-USDT", "X-USDT",   "RAY-USDT",
    "EPT-USDT",   "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]

# Cooldown settings in seconds (3min = 300s, 5min = 600s)
COOLDOWN = {"3min": 300, "5min": 600}
last_sent = {}  # Track last signal time per token per timeframe


# ——— MAIN LOOP ————————————————————————————————————————
def run():
    counter = 1
    while True:
        now = time.time()

        for symbol in TOKENS:
            # --- Fetch data from KuCoin ---
            df3 = get_ohlcv(symbol, interval="3min", limit=100)
            df5 = get_ohlcv(symbol, interval="5min", limit=100)

            if df3 is None or df3.empty:
                print(f"[{symbol}] ❌ Data fetch failed or empty.")
                continue

            # --- Analyze 3-minute signal ---
            tf = "3min"
            key = f"{symbol}_{tf}"
            cooldown = COOLDOWN.get(tf, 0)
            last_time = last_sent.get(key, 0)

            if cooldown > 0 and now - last_time < cooldown:
                print(f"[{symbol}] ⏳ Cooldown active. Skipping...")
                continue

            sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=df5, tf=tf)
            result = sf3.analyze()

            if result:
                text, sym, bias, price, tf_out, score_str, passed_str = result
                numbered_msg = f"{counter}. {sym} ({tf_out}) [V19 Confirmed] → {text}"

                # Send Telegram Alert
                print(f"[LOG] Sending alert for {sym}: {numbered_msg}")
                send_telegram_alert(
                    numbered_msg, sym, bias, price, tf_out, score_str, passed_str
                )

                last_sent[key] = now
                counter += 1

        print("✅ Cycle complete. Sleeping 60 seconds...\n")
        time.sleep(60)


# ——— ENTRY POINT ————————————————————————————————————————
if __name__ == "__main__":
    run()
