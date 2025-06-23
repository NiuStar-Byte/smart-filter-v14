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

# Binance interval mapping (fallback)
BINANCE_INTERVAL = {
    "3min": "3m",
    "5min": "5m"
}


def fetch_ohlcv(symbol: str, tf: str, limit: int = 50) -> pd.DataFrame | None:
    """
    Internal: Fetch OHLCV data for a given symbol & timeframe key (tf).
    Tries (in order):
      1) KuCoin Spot
      2) KuCoin Futures
      3) Binance Spot
    Returns a DataFrame [open, high, low, close, volume] or None.
    """
    # 1) Spot
    if tf in TF_MAP:
        for sym in (symbol, symbol.replace('-', '/')):
            try:
                url = f"https://api.kucoin.com/api/v1/market/candles?type={TF_MAP[tf]}&symbol={sym}"
                r = requests.get(url, timeout=10); r.raise_for_status()
                bars = r.json().get("data") or []
                if not bars: continue
                df = pd.DataFrame(bars, columns=[
                    "timestamp","open","close","high","low","volume","turnover"
                ]).iloc[::-1].reset_index(drop=True)
                df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df.set_index("timestamp", inplace=True)
                return df[["open","high","low","close","volume"]]
            except Exception:
                continue

    # 2) Futures
    if tf in FUTURES_GRAN:
        for sym in (symbol, symbol.replace('-', '/')):
            try:
                gran = FUTURES_GRAN[tf]
                url = f"https://api-futures.kucoin.com/api/v1/kline/query?symbol={sym}&granularity={gran}"
                r = requests.get(url, timeout=10); r.raise_for_status()
                bars = r.json().get("data") or []
                if not bars: continue
                df = pd.DataFrame(bars, columns=[
                    "timestamp","open","close","high","low","volume"
                ]).iloc[::-1].reset_index(drop=True)
                df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df[["open","high","low","close","volume"]]
            except Exception:
                continue

    # 3) Binance fallback
    b_interval = BINANCE_INTERVAL.get(tf)
    if b_interval:
        sym_b = symbol.replace('-', '')
        try:
            url = (
                f"https://api.binance.com/api/v3/klines"
                f"?symbol={sym_b}&interval={b_interval}&limit={limit}"
            )
            r = requests.get(url, timeout=10); r.raise_for_status()
            bars = r.json() or []
            if bars:
                df = pd.DataFrame(bars, columns=[
                    "timestamp","open","high","low","close","volume",
                    *([None] * (len(bars[0]) - 6))
                ]).iloc[:, :6]
                df.columns = ["timestamp","open","high","low","close","volume"]
                df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df
        except Exception:
            pass

    print(f"[{symbol}] No OHLCV data fetched for any endpoint.")
    return None


def get_ohlcv(symbol: str, interval: str, limit: int = 50) -> pd.DataFrame | None:
    """
    Public alias matching main.py signature.
      interval: one of "2min","3min","5min"
      limit: number of bars for Binance fallback
    """
    return fetch_ohlcv(symbol, tf=interval, limit=limit)
