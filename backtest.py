import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------- 1) FETCH OHLCV FROM KUCOIN ---------
TF_MAP = {"2min":"2min", "3min":"3min", "5min":"5min"}
def fetch_ohlcv(symbol: str, tf: str) -> pd.DataFrame | None:
    if tf not in TF_MAP:
        print(f"Unsupported timeframe: {tf}")
        return None
    for s in (symbol, symbol.replace('-', '/')):
        try:
            url = f"https://api.kucoin.com/api/v1/market/candles?type={TF_MAP[tf]}&symbol={s}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json().get('data') or []
            if not data:
                continue
            df = pd.DataFrame(data, columns=["timestamp","open","close","high","low","volume","turnover"]).iloc[::-1].reset_index(drop=True)
            df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            return df[["open","high","low","close","volume"]]
        except Exception:
            continue
    print(f"No OHLCV data for {symbol} @ {tf}")
    return None

# --------- 2) SMART FILTER CLASS (imported from your live code) ---------
from smart_filter import SmartFilter

# --------- 3) BACKTEST FUNCTION ---------
def backtest_symbol(symbol: str, tf_main: str = "5min", tf_mtf: str = "3min", lookback: int = 14, hold_bars: int = 1):
    df_main = fetch_ohlcv(symbol, tf_main)
    df_mtf  = fetch_ohlcv(symbol, tf_mtf)
    if df_main is None or df_mtf is None:
        print(f"Skipping {symbol}, missing data.")
        return []

    trades = []
    for i in range(lookback, len(df_main) - hold_bars):
        slice_main = df_main.iloc[:i+1]
        idx3 = min(int((i+1) * 5 / 3), len(df_mtf) - 1)
        slice_3m = df_mtf.iloc[:idx3+1]

        sf = SmartFilter(symbol, slice_main, df3m=slice_3m, df5m=slice_main, tf=tf_main)
        result = sf.analyze()
        if result:
            _, _, bias, entry_price, _, _, _ = result
            exit_price = df_main['close'].iat[i + hold_bars]
            pnl = (exit_price / entry_price - 1) * (1 if bias == 'LONG' else -1)
            trades.append(pnl)
    return trades

# --------- 4) RUN BACKTEST AND PLOT ---------
if __name__ == '__main__':
    symbols = ["SKATE-USDT", "LA-USDT", "SPK-USDT"]  # add more as needed
    all_results = {}
    for sym in symbols:
        pnl_list = backtest_symbol(sym)
        if not pnl_list:
            continue
        arr = np.array(pnl_list)
        print(f"{sym}: signals={len(arr)}, win_rate={np.mean(arr>0):.1%}, avg_ret={np.mean(arr):.2%}, max_dd={(np.maximum.accumulate(arr.cumsum()) - arr.cumsum()).max():.2%}")
        all_results[sym] = arr

    # Combined histogram
    plt.figure(figsize=(8,4))
    plt.hist(np.concatenate(list(all_results.values())), bins=30)
    plt.title('Backtest PnL Distribution')
    plt.xlabel('PnL %')
    plt.ylabel('Frequency')
    plt.show()
