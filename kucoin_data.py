import requests
import pandas as pd

# Supported spot timeframes mapping to KuCoin spot API 'type'
TF_MAP = {
    "2min": "2min",
    "3min": "3min",
    "5min": "5min"
}

# Futures timeframes mapping to granularity (in minutes)
FUTURES_GRAN = {
    "2min": 2,
    "3min": 3,
    "5min": 5
}

# Binance interval mapping
BINANCE_INTERVAL = {
    "3min": "3m",
    "5min": "5m"
}


def fetch_ohlcv(symbol: str, tf: str, limit: int = 50) -> pd.DataFrame | None:
    """
    Fetches OHLCV data for a given symbol and timeframe.
    1. Tries KuCoin Spot API.
    2. Falls back to KuCoin Futures API.
    3. Falls back to Binance Spot API.

    :param symbol: Token symbol, e.g. "SPARK-USDT"
    :param tf: Timeframe key, one of TF_MAP/FUTURES_GRAN, e.g. "3min"
    :param limit: number of bars to fetch (used by Binance)
    :return: DataFrame with columns [open, high, low, close, volume] indexed by timestamp, or None if no data
    """
    # Validate timeframe
    if tf not in TF_MAP:
        print(f"[{symbol}] Unsupported timeframe: {tf}")
        return None

    # 1) Try KuCoin Spot API
    for sym in (symbol, symbol.replace('-', '/')):
        try:
            url = f"https://api.kucoin.com/api/v1/market/candles?type={TF_MAP[tf]}&symbol={sym}"
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            bars = res.json().get('data') or []
            if not bars:
                continue
            df = pd.DataFrame(bars, columns=["timestamp","open","close","high","low","volume","turnover"])
            df = df.iloc[::-1].reset_index(drop=True)
            df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            return df[["open","high","low","close","volume"]]
        except Exception:
            continue

    # 2) Try KuCoin Futures API
    for sym in (symbol, symbol.replace('-', '/')):
        try:
            gran = FUTURES_GRAN.get(tf)
            if gran is None:
                continue
            url = f"https://api-futures.kucoin.com/api/v1/kline/query?symbol={sym}&granularity={gran}" 
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            bars = res.json().get('data') or []
            if not bars:
                continue
            df = pd.DataFrame(bars, columns=["timestamp","open","close","high","low","volume"])
            df = df.iloc[::-1].reset_index(drop=True)
            df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df[["open","high","low","close","volume"]]
        except Exception:
            continue

    # 3) Try Binance Spot API
    b_interval = BINANCE_INTERVAL.get(tf)
    if b_interval:
        sym_b = symbol.replace('-', '')
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={sym_b}&interval={b_interval}&limit={limit}"
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            bars = res.json() or []
            if bars:
                # Binance: [open_time, open, high, low, close, volume, ...]
                df = pd.DataFrame(bars, columns=[
                    "timestamp","open","high","low","close","volume",
                    "close_time","asset_volume","trades","taker_base_vol","taker_quote_vol","ignore"
                ])
                df = df.iloc[:, :6]  # keep timestamp->volume
                df.columns = ["timestamp","open","high","low","close","volume"]
                df = df.reset_index(drop=True)
                df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df[["open","high","low","close","volume"]]
        except Exception:
            pass

    print(f"[{symbol}] No OHLCV data fetched for any endpoint.")
    return None
