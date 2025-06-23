import os
from kucoin_data import fetch_ohlcv
from smart_filter import SmartFilter

TOKENS = [
    "SPARK-USDT", "BID-USDT", "SKATE-USDT", "LA-USDT", "SPK-USDT",
    "ZKJ-USDT", "IP-USDT", "AERO-USDT", "BMT-USDT", "LQTY-USDT",
    "FUN-USDT", "SNT-USDT", "X-USDT", "BANK-USDT", "RAY-USDT",
    "REX-USDT", "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT"
]

# Main execution loop
# Only uses 3min and 5min timeframes via fetch_ohlcv

def run():
    for symbol in TOKENS:
        print(f"\nChecking {symbol}...")
        try:
            # Fetch 3min and 5min OHLCV
            df3 = fetch_ohlcv(symbol, "3min", limit=100)
            df5 = fetch_ohlcv(symbol, "5min", limit=100)

            if df3 is None or df3.empty:
                print(f"[{symbol}] No data.")
                continue

            # Apply SmartFilter on 3min data
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
                signal_msg, sym, direction, price, tf_out, score, passed = result
                print(f"✔️ Signal: {signal_msg}")
            else:
                print(f"❌ No valid signal for {symbol}")

        except Exception as e:
            print(f"[{symbol}] ERROR: {e}")

if __name__ == "__main__":
    run()
