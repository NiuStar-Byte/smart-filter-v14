import os
import pandas as pd
from kucoin_data import fetch_ohlcv
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
            # Get multi-timeframe data
            df1 = fetch_ohlcv(symbol, tf="1min", limit=100)
            df3 = fetch_ohlcv(symbol, tf="3min", limit=100)
            df5 = fetch_ohlcv(symbol, tf="5min", limit=100)

            if df1 is None or df1.empty:
                print(f"[{symbol}] No data.")
                continue

            # Apply Smart Filter
            sf = SmartFilter(
                symbol=symbol,
                df=df1,
                df3m=df3,
                df5m=df5,
                tf="1min",
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
