import os
import pandas as pd
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter

TOKENS = [
    "SPARK-USDT", "BID-USDT", "SKATE-USDT", "LA-USDT", "SPK-USDT",
    "ZKJ-USDT", "IP-USDT", "AERO-USDT", "BMT-USDT", "LQTY-USDT",
    "FUN-USDT", "SNT-USDT", "X-USDT", "BANK-USDT", "RAY-USDT",
    "REX-USDT", "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT"
]

def run():
    for symbol in TOKENS:
        print(f"\nChecking {symbol}...")

        try:
            # Use 3min as primary TF
            df3 = get_ohlcv(symbol, interval="3min", limit=100)
            df5 = get_ohlcv(symbol, interval="5min", limit=100)

            if df3 is None or df3.empty:
                print(f"[{symbol}] No data.")
                continue

            # Apply Smart Filter using 3min TF
            sf = SmartFilter(
                symbol=symbol,
                df=df3,
                df3m=df3,
                df5m=df5,
                tf="3min",
                min_score=9,
                required_passed=7,
                volume_multiplier=2.0
            )

            result = sf.analyze()

            if result:
                signal_msg, sym, direction, price, tf, score, passed = result
                print(f"✔️ Signal: {signal_msg}")
            else:
                print(f"❌ No valid signal for {symbol}")

        except Exception as e:
            print(f"[{symbol}] ERROR: {e}")

if __name__ == "__main__":
    run()
