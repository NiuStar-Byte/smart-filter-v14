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


def fetch_ohlcv(symbol: str, tf: str, limit: int = None) -> pd.DataFrame | None:
    """
    Fetches OHLCV data for a given symbol and timeframe from KuCoin.
    Tries spot REST API first, then falls back to futures REST API if spot returns no data.

    :param symbol: Token symbol, e.g. "SPARK-USDT"
    :param tf: Timeframe key, one of TF_MAP/FUTURES_GRAN, e.g. "3min"
    :param limit: unused
    :return: DataFrame with columns [open, high, low, close, volume] indexed by timestamp, or None if no data
    """
    # Validate timeframe
    if tf not in TF_MAP:
        print(f"[{symbol}] Unsupported timeframe: {tf}")
        return None

    # 1) Try Spot API
    for sym in (symbol, symbol.replace('-', '/')):
        try:
            url = f"https://api.kucoin.com/api/v1/market/candles?type={TF_MAP[tf]}&symbol={sym}"
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            payload = res.json()
            bars = payload.get('data') or []
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

    # 2) Try Futures API
    for sym in (symbol, symbol.replace('-', '/')):
        try:
            gran = FUTURES_GRAN.get(tf)
            if gran is None:
                continue
            url = (
                f"https://api-futures.kucoin.com/api/v1/kline/query?symbol={sym}&granularity={gran}"
            )
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            payload = res.json()
            bars = payload.get('data') or []
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

    # No data found in either endpoint
    print(f"[{symbol}] No OHLCV data fetched for formats '{symbol}' or '{symbol.replace('-', '/')}'.")
    return None
