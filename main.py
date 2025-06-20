import os
import time
from kucoin_data import fetch_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert

TOKENS = [
    "SPARK-USDT", "BID-USDT", "SKATE-USDT", "LA-USDT", "SPK-USDT",
    "ZKJ-USDT", "IP-USDT", "AERO-USDT", "BMT-USDT", "LQTY-USDT",
    "FUN-USDT", "SNT-USDT", "X-USDT", "BANK-USDT", "RAY-USDT",
    "REX-USDT", "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACT-USDT"
]

TIMEFRAMES = ["2min", "3min", "5min"]
COOLDOWN = {
    "2min": 300,   # 5 minutes
    "3min": 720,   # 12 minutes
    "5min": 900    # 15 minutes
}
last_sent = {}

def run():
    while True:
        for symbol in TOKENS:
            for tf in TIMEFRAMES:
                key = f"{symbol}_{tf}"
                now = time.time()
                if key in last_sent and (now - last_sent[key]) < COOLDOWN[tf]:
                    print(f"[{key}] Skipped due to cooldown.")
                    continue

                try:
                    df = fetch_ohlcv(symbol, tf)
                    if df is not None:
                        filter = SmartFilter(symbol, df, tf=tf, min_score=9, required_passed=7)
                        result = filter.analyze()
                        if result and isinstance(result, tuple) and len(result) == 7:
                            signal_text, symbol, signal_type, price, tf, score, passed = result
                            if os.getenv("DRY_RUN", "false").lower() != "true":
                                send_telegram_alert(symbol, signal_type, price, tf, score, passed)
                            last_sent[key] = now
                    else:
                        print(f"[{symbol} {tf}] No data fetched.")
                except Exception as e:
                    print(f"[{symbol} {tf}] Unexpected error: {e}")

        print("✅ Cycle complete. Sleeping 3 minutes...\n")
        time.sleep(180)  # 3 minutes

if __name__ == "__main__":
    run()
