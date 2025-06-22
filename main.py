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

# We run both 3m & 5m checks each cycle, but gate them separately
TIMEFRAMES = ["3min", "5min"]
COOLDOWN   = {"3min": 720, "5min": 900}  # in seconds
last_sent  = {}  # maps "<symbol>_<tf>_pending" or "_confirmed" → timestamp

def run():
    while True:
        now     = time.time()
        counter = 1

        for symbol in TOKENS:

            # ── PHASE 1: attempt the 3m filter ──────────────────────────────────
            tf = "3min"
            pend_key = f"{symbol}_{tf}_pending"
            conf_key = f"{symbol}_{tf}_confirmed"

            df3 = None
            # only run the pending check if we haven't done it within cooldown
            if not (pend_key in last_sent and now - last_sent[pend_key] < COOLDOWN[tf]):
                df3 = fetch_ohlcv(symbol, tf)
                if df3 is not None:
                    sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=None, tf=tf)
                    res3 = sf3.analyze()
                    if res3:
                        # mark that we've pending-passed this bar
                        last_sent[pend_key] = now

            # if pending passed, now try to confirm and alert (once per cooldown)
            if pend_key in last_sent:
                # only send the confirmed alert if we haven't done so yet
                if not (conf_key in last_sent and now - last_sent[conf_key] < COOLDOWN[tf]):
                    # ensure we still have fresh df3
                    df3 = df3 or fetch_ohlcv(symbol, tf)
                    if df3 is not None:
                        sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=None, tf=tf)
                        res3 = sf3.analyze()
                        if res3:
                            text, sym, bias, price, tf_out, score, passed = res3
                            msg = f"{counter}. {sym} ({tf}) [Confirmed] → {text}"
                            if os.getenv("DRY_RUN", "false").lower() != "true":
                                send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                            last_sent[conf_key] = now
                            counter += 1

            # ── PHASE 2: attempt the 5m filter ──────────────────────────────────
            # (exactly the same pattern, but gating off 5min keys,
            #  and only if the 3m already confirmed)
            if conf_key in last_sent:
                tf = "5min"
                pend_key = f"{symbol}_{tf}_pending"
                conf_key = f"{symbol}_{tf}_confirmed"

                df5 = None
                if not (pend_key in last_sent and now - last_sent[pend_key] < COOLDOWN[tf]):
                    df5 = fetch_ohlcv(symbol, tf)
                    if df5 is not None:
                        # pass both 3m & 5m into the filter so MTF volume runs
                        sf5 = SmartFilter(symbol, df5, df3m=df3, df5m=df5, tf=tf)
                        res5 = sf5.analyze()
                        if res5:
                            last_sent[pend_key] = now

                if pend_key in last_sent:
                    if not (conf_key in last_sent and now - last_sent[conf_key] < COOLDOWN[tf]):
                        df5 = df5 or fetch_ohlcv(symbol, tf)
                        if df5 is not None:
                            sf5 = SmartFilter(symbol, df5, df3m=df3, df5m=df5, tf=tf)
                            res5 = sf5.analyze()
                            if res5:
                                text, sym, bias, price, tf_out, score, passed = res5
                                msg = f"{counter}. {sym} ({tf}) [Confirmed] → {text}"
                                if os.getenv("DRY_RUN", "false").lower() != "true":
                                    send_telegram_alert(msg, sym, bias, price, tf_out, score, passed)
                                last_sent[conf_key] = now
                                counter += 1

        print("✅ Cycle complete. Sleeping 60 seconds…")
        time.sleep(60)

if __name__ == "__main__":
    run()
