import sys

# Unbuffered log writer to logs.txt
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def flush(self):
        self.stream.flush()

log_file = open("logs.txt", "a")
sys.stdout = sys.stderr = Unbuffered(log_file)

# Core Imports
import os
import time
import pandas as pd
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert

# List of tokens
TOKENS = [
    "SKATE-USDT", "LA-USDT",  "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT",  "BMT-USDT", "LQTY-USDT", "X-USDT",   "RAY-USDT",
    "EPT-USDT",   "ELDEUSDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]

# Cooldown config
COOLDOWN = {"3min": 300, "5min": 600}  # seconds (example: 5 and 10 minutes)
last_sent = {}

def run():
    counter = 1
    while True:
        now = time.time()
        for symbol in TOKENS:
            # Fetch data
            df3 = get_ohlcv(symbol, interval="3min", limit=100)
            df5 = get_ohlcv(symbol, interval="5min", limit=100)
            if df3 is None or df3.empty:
                print(f"[WARN] No 3min data for {symbol}")
                continue

            # Analyze 3-min signals
            key3 = f"{symbol}_3min"
            cooldown_expired = key3 not in last_sent or now - last_sent[key3] > COOLDOWN["3min"]

            if cooldown_expired:
                sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=df5, tf="3min")
                res3 = sf3.analyze()
                if res3:
                    text, sym, bias, price, tf_out, score_str, passed_str = res3
                    msg = f"{counter}. {sym} ({tf_out}) [V19 Confirmed] → {text}"
                    print(f"[LOG] Sending alert for {sym}: {msg}")
                    send_telegram_alert(msg, sym, bias, price, tf_out, score_str, passed_str)
                    last_sent[key3] = now
                    counter += 1
            else:
                print(f"[INFO] Cooldown active for {symbol}")

        print("✅ Cycle complete. Sleeping 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    run()
