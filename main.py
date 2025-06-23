import sys
import os
import time
import pandas as pd
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert

# --- Log output to logs.txt with real-time flush ---
class Logger:
    def __init__(self, file):
        self.terminal = sys.__stdout__
        self.log = open(file, "a", buffering=1)  # Line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = sys.stderr = Logger("logs.txt")

# --- Token list ---
TOKENS = [
    "SKATE-USDT", "LA-USDT",  "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT",  "BMT-USDT", "LQTY-USDT", "X-USDT",   "RAY-USDT",
    "EPT-USDT",   "ELDEUSDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]

# --- Cooldown per timeframe (seconds) ---
COOLDOWN = {"3min": 0, "5min": 0}
last_sent = {}  # Tracks last alert timestamp per symbol/timeframe

def run():
    counter = 1
    while True:
        now = time.time()
        for symbol in TOKENS:
            df3 = get_ohlcv(symbol, interval="3min", limit=100)
            df5 = get_ohlcv(symbol, interval="5min", limit=100)
            if df3 is None or df3.empty:
                print(f"[SKIP] No data for {symbol}")
                continue

            key3 = f"{symbol}_3min"
            sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=df5, tf="3min")
            res3 = sf3.analyze()
            if res3:
                text, sym, bias, price, tf_out, score_str, passed_str = res3
                msg = f"{counter}. {sym} ({tf_out}) [V19 Confirmed] → {text}"
                print(f"[LOG] Sending alert for {sym}: {msg}")
                send_telegram_alert(
                    msg, sym, bias, price, tf_out, score_str, passed_str
                )
                last_sent[key3] = now
                counter += 1

        print("✅ Cycle complete. Sleeping 60 seconds...\n")
        time.sleep(60)

if __name__ == "__main__":
    run()
