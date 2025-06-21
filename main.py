import os
import time
import ccxt
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

# -- Symbol availability check at startup --
import ccxt

def check_symbol_availability():
    spot = ccxt.kucoin({ 'enableRateLimit': True, 'options': { 'defaultType': 'spot' } })
    swap = ccxt.kucoin({ 'enableRateLimit': True, 'options': { 'defaultType': 'swap' } })
    bn   = ccxt.binance()

    spot_markets = set(spot.load_markets().keys())
    swap_markets = set(swap.load_markets().keys())
    bn_markets   = set(bn.load_markets().keys())

    print("--- Symbol Availability Check ---")
    for tok in TOKENS:
        slash = tok.replace('-', '/')
        nobar = tok.replace('-', '')
        ok_spot = (slash in spot_markets) or (tok in spot_markets)
        ok_swap = (slash in swap_markets) or (tok in swap_markets) or (nobar in swap_markets)
        ok_bin  = nobar in bn_markets
        print(f"{tok:12} → Spot? {ok_spot}  Swap? {ok_swap}  Binance? {ok_bin}")
    print("---------------------------------")

# Run availability check once
check_symbol_availability():
    ku = ccxt.kucoin()
    bn = ccxt.binance()
    ku_markets = set(ku.load_markets().keys())
    bn_markets = set(bn.load_markets().keys())
    print("--- Symbol Availability Check ---")
    for tok in TOKENS:
        slash = tok.replace('-', '/')
        nobar = tok.replace('-', '')
        ok_ku = (slash in ku_markets) or (tok in ku_markets)
        ok_bn = nobar in bn_markets
        print(f"{tok:12} → KuCoin? {ok_ku}  Binance? {ok_bn}")
    print("---------------------------------")

# Run availability check once
check_symbol_availability()

def run():
    while True:
        now = time.time()
        counter = 1  # Counter to number the signals

        for symbol in TOKENS:
            # Pre-fetch both 3m and 5m frames for hybrid gating
            df3 = fetch_ohlcv(symbol, "3min")
            df5 = fetch_ohlcv(symbol, "5min")

            # Skip symbol entirely if data is missing
            if df3 is None or df5 is None:
                print(f"[{symbol}] Skipping: missing 3min or 5min data.")
                continue

            # 3-minute hybrid alert
            key3 = f"{symbol}_3min"
            if (key3 not in last_sent or (now - last_sent[key3]) >= COOLDOWN["3min"]):
                sf3 = SmartFilter(symbol, df3, tf="3min", min_score=9, required_passed=7)
                result3 = sf3.analyze()
                # Gate: require 5m volume spike (>1.5×10-bar avg)
                avg5 = df5["volume"].rolling(10).mean().iloc[-1]
                gate5 = df5["volume"].iloc[-1] > 1.5 * avg5
                if result3 and gate5:
                    numbered = f"{counter}. {symbol} (3min) - {result3[0]}"
                    if os.getenv("DRY_RUN", "false").lower() != "true":
                        send_telegram_alert(numbered, symbol, result3[2], result3[3], "3min", result3[5], result3[6])
                    last_sent[key3] = now
                    counter += 1

            # 5-minute standard alert
            key5 = f"{symbol}_5min"
            if (key5 not in last_sent or (now - last_sent[key5]) >= COOLDOWN["5min"]):
                sf5 = SmartFilter(symbol, df5, tf="5min", min_score=9, required_passed=7)
                result5 = sf5.analyze()
                if result5:
                    numbered = f"{counter}. {symbol} (5min) - {result5[0]}"
                    if os.getenv("DRY_RUN", "false").lower() != "true":
                        send_telegram_alert(numbered, symbol, result5[2], result5[3], "5min", result5[5], result5[6])
                    last_sent[key5] = now
                    counter += 1

        print("✅ Cycle complete. Sleeping 60 seconds...
")
        time.sleep(60)

if __name__ == "__main__":
    run()
 "__main__":
    run()
