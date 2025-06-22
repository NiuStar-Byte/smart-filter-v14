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

COOLDOWN = {"3min": 720, "5min": 900}
last_sent = {}
pending_3min = set()

def run():
    while True:
        now = time.time()
        counter = 1

        for symbol in TOKENS:
            key3 = f"{symbol}_3min"
            # Phase 1: handle pending confirmation or initial 3-min pass
            df3 = fetch_ohlcv(symbol, "3min")
            if df3 is None:
                continue

            # If previously flagged, confirm now
            if symbol in pending_3min:
                # check cooldown
                if not (key3 in last_sent and now - last_sent[key3] < COOLDOWN["3min"]):
                    sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=None, tf="3min")
                    res3 = sf3.analyze()
                    if res3:
                        text, sym, bias, price, tf_out, score, passed = res3
                        msg = f"{counter}. {sym} (3min) [Confirmed] → {text}"
                        if os.getenv("DRY_RUN","false").lower() != "true":
                            send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                        last_sent[key3] = now
                        counter += 1
                pending_3min.remove(symbol)
            else:
                # initial 3-min detection
                if not (key3 in last_sent and now - last_sent[key3] < COOLDOWN["3min"]):
                    sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=None, tf="3min")
                    res3 = sf3.analyze()
                    if res3:
                        pending_3min.add(symbol)

            # Phase 2: 5-min confirmation after confirmed 3-min
            key5 = f"{symbol}_5min"
            # only if we just confirmed in this cycle
            if key3 in last_sent:
                if not (key5 in last_sent and now - last_sent[key5] < COOLDOWN["5min"]):
                    df5 = fetch_ohlcv(symbol, "5min")
                    if df5 is None:
                        continue
                    sf5 = SmartFilter(symbol, df5, df3m=df3, df5m=df5, tf="5min")
                    res5 = sf5.analyze()
                    if res5:
                        text, sym, bias, price, tf_out, score, passed = res5
                        msg = f"{counter}. {sym} (5min) → {text}"
                        if os.getenv("DRY_RUN","false").lower() != "true":
                            send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                        last_sent[key5] = now
                        counter += 1

        print("✅ Cycle complete. Sleeping 60 seconds...
")
        time.sleep(60)

if __name__ == "__main__":
    run()
