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

# Cooldown periods per timeframe (seconds)
COOLDOWN = {"3min": 720, "5min": 900}
# Track timestamps for pending and confirmed per timeframe
last_sent = {}
# Tracks symbols that had an initial 3-min pass
pending_3min = set()


def run():
    while True:
        now = time.time()
        counter = 1

        for symbol in TOKENS:
            # --- PHASE 1: 3-minute double confirmation ---
            tf = "3min"
            pending_key = f"{symbol}_{tf}_pending"
            confirmed_key = f"{symbol}_{tf}_confirmed"

            # Fetch 3-min data
            df3 = fetch_ohlcv(symbol, tf)
            if df3 is None:
                continue

            # Step 1a: initial pending pass
            if not (pending_key in last_sent and now - last_sent[pending_key] < COOLDOWN[tf]):
                sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=None, tf=tf)
                res3 = sf3.analyze()
                if res3:
                    last_sent[pending_key] = now

            # Step 1b: send confirmed alert on second pass
            if pending_key in last_sent and not (confirmed_key in last_sent and now - last_sent[confirmed_key] < COOLDOWN[tf]):
                # Re-run or reuse df3
                sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=None, tf=tf)
                res3 = sf3.analyze()
                if res3:
                    text, sym, bias, price, tf_out, score, passed = res3
                    msg = f"{counter}. {sym} ({tf}) [Confirmed] → {text}"
                    if os.getenv("DRY_RUN", "false").lower() != "true":
                        send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                    last_sent[confirmed_key] = now
                    counter += 1

            # --- PHASE 2: 5-minute confirmation after 3-min confirmed ---
            if confirmed_key in last_sent:
                tf5 = "5min"
                pend5_key = f"{symbol}_{tf5}_pending"
                conf5_key = f"{symbol}_{tf5}_confirmed"

                # initial 5-min pending
                if not (pend5_key in last_sent and now - last_sent[pend5_key] < COOLDOWN[tf5]):
                    df5 = fetch_ohlcv(symbol, tf5)
                    if df5 is not None:
                        sf5 = SmartFilter(symbol, df5, df3m=df3, df5m=df5, tf=tf5)
                        res5 = sf5.analyze()
                        if res5:
                            last_sent[pend5_key] = now

                # send 5-min confirmed alert
                if pend5_key in last_sent and not (conf5_key in last_sent and now - last_sent[conf5_key] < COOLDOWN[tf5]):
                    # reuse df5 if available
                    if 'df5' not in locals() or df5 is None:
                        df5 = fetch_ohlcv(symbol, tf5)
                    if df5 is not None:
                        sf5 = SmartFilter(symbol, df5, df3m=df3, df5m=df5, tf=tf5)
                        res5 = sf5.analyze()
                        if res5:
                            text, sym, bias, price, tf_out, score, passed = res5
                            msg = f"{counter}. {sym} ({tf5}) [Confirmed] → {text}"
                            if os.getenv("DRY_RUN", "false").lower() != "true":
                                send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                            last_sent[conf5_key] = now
                            counter += 1

        print("✅ Cycle complete. Sleeping 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    run()
