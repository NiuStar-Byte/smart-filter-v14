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

def fetch_ohlcv(symbol: str, tf: str, limit: int = 100) -> pd.DataFrame | None:
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

def get_ohlcv(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame | None:
    """
    Public alias matching main.py signature.
      interval: one of "2min","3min","5min"
      limit: number of bars for Binance fallback
    """
    return fetch_ohlcv(symbol, tf=interval, limit=limit)

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

# =========================
# ENHANCEMENT: Entry Price
# =========================

# Define default slippage globally (recommended 0.1% for most liquid pairs)
DEFAULT_SLIPPAGE = 0.001  # 0.1%

def get_live_entry_price(
    symbol: str,
    signal_type: str,
    tf: str = "5min",  # Accepts any timeframe, "5min" is just default
    slippage: float = DEFAULT_SLIPPAGE,    # Use the global default
    long_adjust: float = 0.9900,
    short_adjust: float = 1.0100,
    debug: bool = False
) -> float | None:
    """
    Fetch the best bid/ask price from KuCoin's orderbook for the symbol.
    Use as entry price for market orders. Fallback to OHLCV close price if unavailable.
    :param symbol: Trading symbol.
    :param signal_type: "LONG" or "SHORT".
    :param tf: Timeframe (used for OHLCV fallback).
                Accepts any timeframe string, e.g., "3min", "5min", "15min", "1h", etc.
    :param slippage: Fractional slippage (e.g., 0.001 for 0.1%). Default: 0.1%.
    :param long_adjust: Multiplicative adjustment for LONG entries (default 0.995).
    :param short_adjust: Multiplicative adjustment for SHORT entries (default 1.005).
    :param debug: Print detailed log output.
    :return: Adjusted entry price (float) or None.
    """
    bids, asks = fetch_orderbook_l2(symbol, depth=20)
    entry_price = None
    source = None
    try:
        if signal_type.upper() == "LONG" and asks is not None and len(asks) > 0:
            entry_price = asks['price'].iloc[0] * (1 + slippage)
            source = "orderbook_ask"
        elif signal_type.upper() == "SHORT" and bids is not None and len(bids) > 0:
            entry_price = bids['price'].iloc[0] * (1 - slippage)
            source = "orderbook_bid"
        elif bids is not None and len(bids) > 0 and asks is not None and len(asks) > 0:
            entry_price = (bids['price'].iloc[0] + asks['price'].iloc[0]) / 2
            source = "orderbook_mid"
        else:
            entry_price = None
            source = "orderbook_none"
    except Exception as e:
        entry_price = None
        source = f"error: {e}"

    if entry_price is not None and debug:
        print(f"[EntryPrice] {symbol} {signal_type} ({tf}): {source} price before adjustment: {entry_price:.6f}, slippage={slippage}")

    # Final fallback: use OHLCV close price for the requested timeframe
    if entry_price is None:
        df = fetch_ohlcv(symbol, tf)
        if df is not None and not df.empty:
            entry_price = float(df["close"].iloc[-1])
            source = f"ohlcv_close_{tf}"
            if debug:
                print(f"[EntryPrice] {symbol} {signal_type} ({tf}): Fallback to OHLCV close price: {entry_price:.6f}")
        else:
            if debug:
                print(f"[EntryPrice] {symbol} {signal_type} ({tf}): No data available for entry price.")
            return None

    # Adjustment: configurable for LONG/SHORT
    if signal_type.upper() == "LONG":
        entry_price *= long_adjust
        if debug:
            print(f"[EntryPrice] {symbol} LONG ({tf}): Final adjusted entry price: {entry_price:.6f} (adjust={long_adjust})")
    elif signal_type.upper() == "SHORT":
        entry_price *= short_adjust
        if debug:
            print(f"[EntryPrice] {symbol} SHORT ({tf}): Final adjusted entry price: {entry_price:.6f} (adjust={short_adjust})")

    return float(entry_price)

