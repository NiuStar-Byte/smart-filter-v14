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

# Cooldown periods (in seconds) to avoid spamming same symbol/timeframe
COOLDOWN = {"3min": 720, "5min": 900}
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
                # enforce cooldown
                last = last_sent.get(key3, 0)
                if now - last >= COOLDOWN["3min"]:
                    text, sym, bias, price, tf_out, score_str, passed_str = res3
                    msg = f"{counter}. {sym} ({tf_out}) [V19 Confirmed] → {text}"
                    # send Telegram alert (skip if DRY_RUN=true)
                    if os.getenv("DRY_RUN", "false").lower() != "true":
                        # Extract numeric parts for templating
                        score_num = int(score_str.split('/')[0])
                        passed_num = int(passed_str.split('/')[0])
                        send_telegram_alert(
                            msg, sym, bias, price, tf_out, score_num, passed_num
                        )
                    last_sent[key3] = now
                    counter += 1

        # Wait a minute before next cycle
        print("✅ Cycle complete. Sleeping 60 seconds...")
        time.sleep(60)


if __name__ == "__main__":
    run()
