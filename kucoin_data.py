import requests
import pandas as pd

TF_MAP = {
    "2min": "2min",
    "3min": "3min",
    "5min": "5min"
}
FUTURES_GRAN = {
    "2min": 2,
    "3min": 3,
    "5min": 5
}
BINANCE_INTERVAL = {
    "3min": "3m",
    "5min": "5m"
}

def fetch_ohlcv(symbol: str, tf: str, limit: int = 100) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a given symbol & timeframe key (tf).
    Tries (in order):
      1) KuCoin Spot
      2) KuCoin Futures
      3) Binance Spot
    Returns a DataFrame [open, high, low, close, volume] or None.
    NOTE: The limit should be >= max_entry_idx + PEC_BARS + 1 (e.g., 120 if you will process signals up to the last bar and look ahead 5 bars).
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
                df = df[["open","high","low","close","volume"]]
                if len(df) < limit:
                    print(f"[WARNING] {symbol} {tf}: Fetched only {len(df)} bars, less than requested {limit}.")
                if not df.index.is_monotonic_increasing:
                    print(f"[WARNING] {symbol} {tf}: Timestamps not sorted!")
                if df.index.duplicated().any():
                    print(f"[WARNING] {symbol} {tf}: Duplicate timestamps found!")
                return df
            except Exception as e:
                print(f"[ERROR] KuCoin Spot fetch failed for {symbol} {tf}: {e}")
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
                df = df[["open","high","low","close","volume"]]
                if len(df) < limit:
                    print(f"[WARNING] {symbol} {tf} Futures: Fetched only {len(df)} bars, less than requested {limit}.")
                if not df.index.is_monotonic_increasing:
                    print(f"[WARNING] {symbol} {tf}: Timestamps not sorted!")
                if df.index.duplicated().any():
                    print(f"[WARNING] {symbol} {tf}: Duplicate timestamps found!")
                return df
            except Exception as e:
                print(f"[ERROR] KuCoin Futures fetch failed for {symbol} {tf}: {e}")
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
                if len(df) < limit:
                    print(f"[WARNING] {symbol} {tf} Binance: Fetched only {len(df)} bars, less than requested {limit}.")
                if not df.index.is_monotonic_increasing:
                    print(f"[WARNING] {symbol} {tf}: Timestamps not sorted!")
                if df.index.duplicated().any():
                    print(f"[WARNING] {symbol} {tf}: Duplicate timestamps found!")
                return df
        except Exception as e:
            print(f"[ERROR] Binance fetch failed for {symbol} {tf}: {e}")
    print(f"[{symbol}] No OHLCV data fetched for any endpoint.")
    return pd.DataFrame(columns=["open","high","low","close","volume"])

def get_ohlcv(symbol: str, interval: str, limit: int = 120) -> pd.DataFrame | None:
    """
    Public alias matching main.py signature.
      interval: one of "2min","3min","5min"
      limit: number of bars for Binance fallback
      Default is 120 (enough for PEC lookahead from last signal bar)
    """
    return fetch_ohlcv(symbol, tf=interval, limit=limit)

# (Orderbook and tick functions unchanged)

# ====== NEW ORDERBOOK & TICK DATA FUNCTIONS ======

def fetch_orderbook_l2(symbol: str, depth: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch the L2 order book (top N levels) from KuCoin. Returns (bids_df, asks_df).
    """
    for sym in (symbol, symbol.replace('-', '/')):
        try:
            url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_{depth}?symbol={sym}"
            resp = requests.get(url, timeout=10); resp.raise_for_status()
            data = resp.json().get('data', {})
            bids = pd.DataFrame(data.get('bids', []), columns=['price', 'size']).astype(float)
            asks = pd.DataFrame(data.get('asks', []), columns=['price', 'size']).astype(float)
            return bids, asks
        except Exception:
            continue
    return pd.DataFrame(), pd.DataFrame()

def fetch_trade_ticks(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch recent trade ticks from KuCoin. Returns DataFrame with [price, size, side, time].
    """
    for sym in (symbol, symbol.replace('-', '/')):
        try:
            url = f"https://api.kucoin.com/api/v1/market/histories?symbol={sym}"
            resp = requests.get(url, timeout=10); resp.raise_for_status()
            data = resp.json().get('data', [])
            if not data: continue
            df = pd.DataFrame(data)
            df = df.rename(columns={'tradeId': 'id', 'tradeType': 'side'})
            df['price'] = df['price'].astype(float)
            df['size'] = df['size'].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            return df[['price', 'size', 'side', 'time']].sort_values('time').reset_index(drop=True)
        except Exception:
            continue
    return pd.DataFrame()
