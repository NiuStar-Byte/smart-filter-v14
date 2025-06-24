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
last_sent = {}

def run():
    print("🚀 Starting Smart Filter engine...\n")
    while True:
        now = time.time()
        for idx, symbol in enumerate(TOKENS, start=1):
            print(f"[INFO] Checking {symbol}...\n")

            try:
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
                        text, sym, bias, price, tf_out, score, passed, confidence, weighted, _ = res3
                        numbered_signal = f"{idx}.A"
                        print(f"[LOG] Sending 3min alert for {sym}")
                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            send_telegram_alert(
                                numbered_signal=numbered_signal,
                                symbol=sym,
                                signal_type=bias,
                                price=price,
                                tf=tf_out,
                                score=score,
                                passed=passed,
                                confidence=confidence,
                                weighted=weighted
                            )
                        last_sent[key3] = now

                # --- Analyze 5-minute ---
                key5 = f"{symbol}_5min"
                sf5 = SmartFilter(symbol, df5, df3m=df3, df5m=df5, tf="5min")
                res5 = sf5.analyze()
                if res5:
                    last5 = last_sent.get(key5, 0)
                    if now - last5 >= COOLDOWN["5min"]:
                        text, sym, bias, price, tf_out, score, passed, confidence, weighted, _ = res5
                        numbered_signal = f"{idx}.B"
                        print(f"[LOG] Sending 5min alert for {sym}")
                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            send_telegram_alert(
                                numbered_signal=numbered_signal,
                                symbol=sym,
                                signal_type=bias,
                                price=price,
                                tf=tf_out,
                                score=score,
                                passed=passed,
                                confidence=confidence,
                                weighted=weighted
                            )
                        last_sent[key5] = now

            except Exception as e:
                print(f"[ERROR] Exception in processing {symbol}: {e}\n")

        print("✅ Cycle complete. Sleeping 60 seconds...\n")
        time.sleep(60)

if __name__ == "__main__":
    run()
