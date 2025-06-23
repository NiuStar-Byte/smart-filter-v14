import os
import pandas as pd
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter

# Only tokens that actually exist on KuCoin spot 3m/5m
TOKENS = [
    "SKATE-USDT","LA-USDT","SPK-USDT","ZKJ-USDT","IP-USDT",
    "AERO-USDT","BMT-USDT","LQTY-USDT","EPT-USDT","ELDE-USDT",
    "MAGIC-USDT","ACTSOL-USDT","RAY-USDT","BANK-USDT","REX-USDT"
]

def run():
    for symbol in TOKENS:
        print(f"\nChecking {symbol}…")

        # 3-minute data (must exist)
        df3 = get_ohlcv(symbol, tf="3min", limit=100)
        if df3 is None or df3.empty:
            print(f"[{symbol}] No 3min data. Skipping.")
            continue

        # 5-minute data (optional for multi-TF checks)
        df5 = get_ohlcv(symbol, tf="5min", limit=100)

        sf = SmartFilter(
            symbol=symbol,
            df=df3,
            df3m=df3,
            df5m=df5,
            tf="3min",
            min_score=9,
            required_passed=8,
            volume_multiplier=2.0
        )

        result = sf.analyze()
        if result:
            signal_msg, sym, direction, price, tf_out, score, passed = result
            print(f"✔️ Signal: {signal_msg}")
        else:
            print(f"❌ No valid signal for {symbol}")

if __name__ == "__main__":
    run()
