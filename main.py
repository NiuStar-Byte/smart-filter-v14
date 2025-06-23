import os
import time
import pandas as pd
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert

# List of 15 tokens to scan (KuCoin spot symbols)
TOKENS = [
    "SKATE-USDT", "LA-USDT",  "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT",  "BMT-USDT", "LQTY-USDT", "X-USDT",   "RAY-USDT",
    "EPT-USDT",   "ELDEUSDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]

# Set cooldown to 0 to disable
COOLDOWN = {"3min": 0, "5min": 0}
last_sent = {}  # tracks last alert timestamp per symbol/timeframe


def run():
    counter = 1
    while True:
        now = time.time()
        for symbol in TOKENS:
            # --- Fetch data ---
            df3 = get_ohlcv(symbol, interval="3min", limit=100)
            df5 = get_ohlcv(symbol, interval="5min", limit=100)
            if df3 is None or df3.empty:
                continue

            # --- Analyze 3-min signals ---
            key3 = f"{symbol}_3min"
            sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=df5, tf="3min")
            res3 = sf3.analyze()
            if res3:
                text, sym, bias, price, tf_out, score_str, passed_str = res3
                msg = f"{counter}. {sym} ({tf_out}) [V19 Confirmed] → {text}"
                # Log and send Telegram alert
                print(f"[LOG] Sending alert for {sym}: {msg}")
                send_telegram_alert(
                    msg, sym, bias, price, tf_out, score_str, passed_str
                )
                last_sent[key3] = now
                counter += 1

        # Wait before next cycle
        print("✅ Cycle complete. Sleeping 60 seconds...")
        time.sleep(60)


if __name__ == "__main__":
    run()
