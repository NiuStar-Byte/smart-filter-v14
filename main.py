import os
import time
import pandas as pd
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert, send_telegram_file
from signal_debug_log import dump_signal_debug_txt

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
    print("[INFO] Starting Smart Filter engine...\n")
    while True:
        now = time.time()
        for idx, symbol in enumerate(TOKENS, start=1):
            print(f"[INFO] Checking {symbol}...\n")

            df3 = get_ohlcv(symbol, interval="3min", limit=100)
            df5 = get_ohlcv(symbol, interval="5min", limit=100)
            if df3 is None or df3.empty or df5 is None or df5.empty:
                continue

            # --- Analyze 3-minute ---
            try:
                key3 = f"{symbol}_3min"
                sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=df5, tf="3min")
                res3 = sf3.analyze()
                if isinstance(res3, dict) and res3.get("valid_signal") is True:
                    last3 = last_sent.get(key3, 0)
                    if now - last3 >= COOLDOWN["3min"]:
                        numbered_signal = f"{idx}.A"
                        print(f"[LOG] Sending 3min alert for {res3['symbol']}")
                        # --- Dump debug log for this signal (temp file) ---
                        dump_signal_debug_txt(
                            symbol=res3["symbol"],
                            tf=res3["tf"],
                            bias=res3["bias"],
                            filter_weights=sf3.filter_weights,
                            gatekeepers=sf3.gatekeepers,
                            results=res3["filter_results"]
                        )
                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            send_telegram_alert(
                                numbered_signal=numbered_signal,
                                symbol=res3.get("symbol"),
                                signal_type=res3.get("bias"),
                                price=res3.get("price"),
                                tf=res3.get("tf"),
                                score=res3.get("score"),
                                score_max=res3.get("score_max"),
                                passed=res3.get("passes"),
                                gatekeepers_total=res3.get("gatekeepers_total"),
                                confidence=res3.get("confidence"),
                                weighted=res3.get("passed_weight"),
                                total_weight=res3.get("total_weight")
                            )
                            # Immediately send the temp debug file to Telegram group
                            send_telegram_file(
                                "signal_debug_temp.txt",
                                caption=f"Signal debug log for {res3.get('symbol')} {res3.get('tf')}"
                            )
                        last_sent[key3] = now
                else:
                    print(f"[INFO] No valid 3min signal for {symbol}.")
            except Exception as e:
                print(f"[ERROR] Exception in processing 3min for {symbol}: {e}")

            # --- Analyze 5-minute ---
            try:
                key5 = f"{symbol}_5min"
                sf5 = SmartFilter(symbol, df5, df3m=df3, df5m=df5, tf="5min")
                res5 = sf5.analyze()
                if isinstance(res5, dict) and res5.get("valid_signal") is True:
                    last5 = last_sent.get(key5, 0)
                    if now - last5 >= COOLDOWN["5min"]:
                        numbered_signal = f"{idx}.B"
                        print(f"[LOG] Sending 5min alert for {res5['symbol']}")
                        # --- Dump debug log for this signal (temp file) ---
                        dump_signal_debug_txt(
                            symbol=res5["symbol"],
                            tf=res5["tf"],
                            bias=res5["bias"],
                            filter_weights=sf5.filter_weights,
                            gatekeepers=sf5.gatekeepers,
                            results=res5["filter_results"]
                        )
                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            send_telegram_alert(
                                numbered_signal=numbered_signal,
                                symbol=res5.get("symbol"),
                                signal_type=res5.get("bias"),
                                price=res5.get("price"),
                                tf=res5.get("tf"),
                                score=res5.get("score"),
                                score_max=res5.get("score_max"),
                                passed=res5.get("passes"),
                                gatekeepers_total=res5.get("gatekeepers_total"),
                                confidence=res5.get("confidence"),
                                weighted=res5.get("passed_weight"),
                                total_weight=res5.get("total_weight")
                            )
                            # Immediately send the temp debug file to Telegram group
                            send_telegram_file(
                                "signal_debug_temp.txt",
                                caption=f"Signal debug log for {res5.get('symbol')} {res5.get('tf')}"
                            )
                        last_sent[key5] = now
                else:
                    print(f"[INFO] No valid 5min signal for {symbol}.")
            except Exception as e:
                print(f"[ERROR] Exception in processing 5min for {symbol}: {e}")

        print("[INFO] ✅ Cycle complete. Sleeping 60 seconds...\n")
        time.sleep(60)

if __name__ == "__main__":
    run()
