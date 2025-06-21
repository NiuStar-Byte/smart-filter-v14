import os
import time
import ccxt
from kucoin_data import fetch_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert

TOKENS = [
    "SKATE-USDT",
    "LA-USDT",
    "SPK-USDT",
    "ZKJ-USDT",
    "IP-USDT",
    "AERO-USDT",
    "BMT-USDT",
    "LQTY-USDT",
    "X-USDT",
    "RAY-USDT",
    "EPT-USDT",
    "ELDE-USDT",
    "MAGIC-USDT",
    "FUN-USDT",       # newly added perpetual
    "ACTSOL-USDT",    # corrected symbol for ActSol perpetual
]

TIMEFRAMES = ["3min", "5min"]
COOLDOWN = {
    "3min": 720,  # 12 minutes
    "5min": 900   # 15 minutes
}
last_sent = {}

# -- Symbol availability check at startup --
def check_symbol_availability():
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
            for tf in TIMEFRAMES:
                key = f"{symbol}_{tf}"
                if key in last_sent and (now - last_sent[key]) < COOLDOWN[tf]:
                    continue

                # Fetch OHLCV; kucoin_data.py will retry formats if needed
                df = fetch_ohlcv(symbol, tf)
                if df is None:
                    print(f"[{symbol}] No OHLCV data fetched.")
                    continue

                sf = SmartFilter(symbol, df, df3m=None, df5m=None, tf=tf, min_score=9, required_passed=7)
                result = sf.analyze()

                if result and isinstance(result, tuple) and len(result) == 7:
                    signal_text, _, signal_type, price, tf, score, passed = result

                    numbered_signal = f"{counter}. {symbol} ({tf}) - {signal_text}"
                    if os.getenv("DRY_RUN", "false").lower() != "true":
                        send_telegram_alert(numbered_signal, symbol, signal_type, price, tf, score, passed)

                    last_sent[key] = now
                    counter += 1

        print("✅ Cycle complete. Sleeping 60 seconds...\n")
        time.sleep(60)

if __name__ == "__main__":
    run()
