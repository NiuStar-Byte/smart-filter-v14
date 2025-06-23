import os
import time
import pandas as pd
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert

# List of tokens to scan (KuCoin Futures symbols)
TOKENS = [
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]

# Cooldown periods per timeframe (in seconds)
COOLDOWN = {"3min": 720, "5min": 900}
last_sent = {}  # Track last alert per symbol-timeframe

def run():
    counter = 1
    while True:
        now = time.time()
        for symbol in TOKENS:
            df3 = get_ohlcv(symbol, interval="3min", limit=100)
            df5 = get_ohlcv(symbol, interval="5min", limit=100)
            if df3 is None or df3.empty or df5 is None or df5.empty:
                continue

            # --- Analyze 3-minute ---
            key3 = f"{symbol}_3min"
            sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=df5, tf="3min")
            res3 = sf3.analyze()
            if res3:
                last3 = last_sent.get(key3, 0)
                if now - last3 >= COOLDOWN["3min"]:
                    text, sym, bias, price, tf_out, score, passed, confidence, weighted = res3
                    conf_icon = "🟢" if confidence >= 75 else "🟡" if confidence >= 60 else "🔴"
                    msg = (
                        f"{counter}. {sym} ({tf_out}) [V19 Confirmed]\n"
                        f"📈 {bias} Signal\n"
                        f"💰 {price:.6f}\n"
                        f"✅ Score: {score}\n"
                        f"📌 Passed: {passed}\n"
                        f"{conf_icon} Confidence: {confidence:.1f}% (Weighted: {weighted:.1f}/48.8)"
                    )
                    print(f"[LOG] Sending 3min alert for {sym}: {msg}")
                    if os.getenv("DRY_RUN", "false").lower() != "true":
                        send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                    last_sent[key3] = now
                    counter += 1

            # --- Analyze 5-minute ---
            key5 = f"{symbol}_5min"
            sf5 = SmartFilter(symbol, df5, df3m=df3, df5m=df5, tf="5min")
            res5 = sf5.analyze()
            if res5:
                last5 = last_sent.get(key5, 0)
                if now - last5 >= COOLDOWN["5min"]:
                    text, sym, bias, price, tf_out, score, passed, confidence, weighted = res5
                    conf_icon = "🟢" if confidence >= 75 else "🟡" if confidence >= 60 else "🔴"
                    msg = (
                        f"{counter}. {sym} ({tf_out})\n"
                        f"📈 {bias} Signal\n"
                        f"💰 {price:.6f}\n"
                        f"✅ Score: {score}\n"
                        f"📌 Passed: {passed}\n"
                        f"{conf_icon} Confidence: {confidence:.1f}% (Weighted: {weighted:.1f}/48.8)"
                    )
                    print(f"[LOG] Sending 5min alert for {sym}: {msg}")
                    if os.getenv("DRY_RUN", "false").lower() != "true":
                        send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                    last_sent[key5] = now
                    counter += 1

        print("✅ Cycle complete. Sleeping 60 seconds...\n")
        time.sleep(60)

if __name__ == "__main__":
    run()
