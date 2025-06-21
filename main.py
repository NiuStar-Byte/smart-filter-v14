import os
import time
from kucoin_data import fetch_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert

# Only KuCoin-available tickers
TOKENS = [
    "SKATE-USDT", "LA-USDT",  "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT",  "BMT-USDT", "LQTY-USDT","X-USDT",   "RAY-USDT",
    "EPT-USDT",   "ELDE-USDT","MAGIC-USDT","ACTSOL-USDT"
]

TIMEFRAMES = ["3min", "5min"]
COOLDOWN = {"3min": 720, "5min": 900}
last_sent = {}

def run():
    while True:
        now = time.time()
        counter = 1

        for symbol in TOKENS:
            # --- PHASE 1: 3-minute check ---
            key3 = f"{symbol}_3min"
            three_passed = False

            if not (key3 in last_sent and now - last_sent[key3] < COOLDOWN["3min"]):
                df3 = fetch_ohlcv(symbol, "3min")
                if df3 is not None:
                    sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=None, tf="3min")
                    res3 = sf3.analyze()
                    if res3:
                        text, sym, bias, price, tf_out, score, passed = res3
                        msg = f"{counter}. {sym} (3min) → {text}"
                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                        last_sent[key3] = now
                        counter += 1
                        three_passed = True

            # --- PHASE 2: 5-minute check (only if 3-min passed) ---
            if three_passed:
                key5 = f"{symbol}_5min"
                if not (key5 in last_sent and now - last_sent[key5] < COOLDOWN["5min"]):
                    df5 = fetch_ohlcv(symbol, "5min")
                    if df5 is not None:
                        sf5 = SmartFilter(symbol, df5, df3m=df3, df5m=df5, tf="5min")
                        res5 = sf5.analyze()
                        if res5:
                            text, sym, bias, price, tf_out, score, passed = res5
                            msg = f"{counter}. {sym} (5min) → {text}"
                            if os.getenv("DRY_RUN", "false").lower() != "true":
                                send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                            last_sent[key5] = now
                            counter += 1

        print("✅ Cycle complete. Sleeping 60 seconds...\n")
        time.sleep(60)

if __name__ == "__main__":
    run()
