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

TIMEFRAMES = ["3min", "5min"]
COOLDOWN = {
    "3min": 720,  # 12 minutes
    "5min": 900   # 15 minutes
}
last_sent = {}

def run():
    while True:
        now = time.time()
        counter = 1  # Counter to number the signals
        for symbol in TOKENS:
            for tf in TIMEFRAMES:
                key = f"{symbol}_{tf}"
                if key in last_sent and (now - last_sent[key]) < COOLDOWN[tf]:
                    continue

                df = fetch_ohlcv(symbol, tf)
                if df is None:
                    continue

                filter = SmartFilter(symbol, df, tf=tf, min_score=9, required_passed=7)
                result = filter.analyze()

                if result and isinstance(result, tuple) and len(result) == 7:
                    signal_text, symbol, signal_type, price, tf, score, passed = result

                    # Add numbering to the signal for each token
                    numbered_signal = f"{counter}. {symbol} ({tf}) - {signal_text}"

                    if os.getenv("DRY_RUN", "false").lower() != "true":
                        send_telegram_alert(numbered_signal, symbol, signal_type, price, tf, score, passed)
                    last_sent[key] = now
                    counter += 1  # Increment the counter for the next signal

        print("✅ Cycle complete. Sleeping 60 seconds...\n")
        time.sleep(60)

if __name__ == "__main__":
    run()
