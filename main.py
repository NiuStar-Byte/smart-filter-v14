import os
import time
from kucoin_data import fetch_ohlcv
from smart_filter import SmartFilterV18, SmartFilterV19
from telegram_alert import send_telegram_alert

TOKENS = [
    "SKATE-USDT", "LA-USDT",  "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT",  "BMT-USDT", "LQTY-USDT","X-USDT",   "RAY-USDT",
    "EPT-USDT",   "ELDE-USDT","MAGIC-USDT","ACTSOL-USDT"
]

COOLDOWN = {"3min": 720, "5min": 900}
last_sent = {}

def run():
    while True:
        now = time.time()
        counter = 1

        for symbol in TOKENS:
            tf = "3min"
            pending_key = f"{symbol}_{tf}_pending"
            confirmed_key = f"{symbol}_{tf}_confirmed"

            df3 = fetch_ohlcv(symbol, tf)
            if df3 is None:
                continue

            if not (pending_key in last_sent and now - last_sent[pending_key] < COOLDOWN[tf]):
                sf18 = SmartFilterV18(symbol, df3, df3, None, tf)
                sf19 = SmartFilterV19(symbol, df3, df3, None, tf)
                res18 = sf18.analyze()
                res19 = sf19.analyze()
                if res18 or res19:
                    last_sent[pending_key] = now

            if pending_key in last_sent and not (confirmed_key in last_sent and now - last_sent[confirmed_key] < COOLDOWN[tf]):
                sf18 = SmartFilterV18(symbol, df3, df3, None, tf)
                sf19 = SmartFilterV19(symbol, df3, df3, None, tf)
                res18 = sf18.analyze()
                res19 = sf19.analyze()

                if res19:
                    text, sym, bias, price, tf_out, score, passed = res19
                    msg = f"{counter}. {sym} ({tf}) [V19 Confirmed] → {text}"
                    if os.getenv("DRY_RUN", "false").lower() != "true":
                        send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                    last_sent[confirmed_key] = now
                    counter += 1

            if confirmed_key in last_sent:
                tf5 = "5min"
                pend5_key = f"{symbol}_{tf5}_pending"
                conf5_key = f"{symbol}_{tf5}_confirmed"

                if not (pend5_key in last_sent and now - last_sent[pend5_key] < COOLDOWN[tf5]):
                    df5 = fetch_ohlcv(symbol, tf5)
                    if df5 is not None:
                        sf18 = SmartFilterV18(symbol, df5, df3, df5, tf5)
                        sf19 = SmartFilterV19(symbol, df5, df3, df5, tf5)
                        res18 = sf18.analyze()
                        res19 = sf19.analyze()
                        if res18 or res19:
                            last_sent[pend5_key] = now

                if pend5_key in last_sent and not (conf5_key in last_sent and now - last_sent[conf5_key] < COOLDOWN[tf5]):
                    if 'df5' not in locals() or df5 is None:
                        df5 = fetch_ohlcv(symbol, tf5)
                    if df5 is not None:
                        sf18 = SmartFilterV18(symbol, df5, df3, df5, tf5)
                        sf19 = SmartFilterV19(symbol, df5, df3, df5, tf5)
                        res18 = sf18.analyze()
                        res19 = sf19.analyze()

                        if res19:
                            text, sym, bias, price, tf_out, score, passed = res19
                            msg = f"{counter}. {sym} ({tf5}) [V19 Confirmed] → {text}"
                            if os.getenv("DRY_RUN", "false").lower() != "true":
                                send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                            last_sent[conf5_key] = now
                            counter += 1

        print("✅ Cycle complete. Sleeping 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    run()
