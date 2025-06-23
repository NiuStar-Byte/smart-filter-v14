import os
import time
import logging
import pandas as pd

from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert

# === SETUP LOGGING ===
log = logging.getLogger()
log.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

file_handler = logging.FileHandler("logs.txt", mode="a")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

# === CONFIGURATION ===
SYMBOLS = [
    "SPARK-USDT", "BIDUSDT", "SKATEUSDT", "LAUSDT", "SPKUSDT",
    "ZKJUSDT", "IPUSDT", "AEROUSDT", "BMTUSDT", "LQTYUSDT",
    "FUNUSDT", "SNTUSDT", "XUSDT", "BANKUSDT", "RAYUSDT"
]  # Only available tokens

TIMEFRAMES = ["3m", "5m"]

COOLDOWNS = {
    "3m": 12 * 60,
    "5m": 15 * 60
}

last_alert_times = {}

# === MAIN LOGIC ===
def run():
    log.info("Smart Filter Bot starting...")
    while True:
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                pair_key = f"{symbol}-{tf}"
                last_time = last_alert_times.get(pair_key, 0)
                if time.time() - last_time < COOLDOWNS[tf]:
                    continue

                try:
                    df = get_ohlcv(symbol, tf)
                    if df is None or df.empty:
                        log.warning(f"No data for {symbol} ({tf})")
                        continue

                    sf = SmartFilter(df, symbol=symbol, tf=tf)
                    result = sf.analyze()
                    if result:
                        log.info(f"[LOG] Sending alert for {symbol}: {result['id']}. {result['message']}")
                        send_telegram_alert(result)
                        last_alert_times[pair_key] = time.time()

                except Exception as e:
                    log.error(f"Error while processing {symbol} ({tf}): {e}", exc_info=True)

        time.sleep(10)

if __name__ == "__main__":
    run()
