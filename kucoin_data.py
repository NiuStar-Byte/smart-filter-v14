import requests
import pandas as pd
import time
from typing import Optional, Tuple

# Supported spot timeframes mapping to KuCoin spot API 'type'
TF_MAP = {
    "1min": "1min",
    "3min": "3min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w"
}

# Futures timeframes mapping to granularity (in minutes)
FUTURES_GRAN = {
    "1min": 1,
    "3min": 3,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "12h": 720,
    "1d": 1440,
    "1w": 10080
}

# Binance interval mapping (fallback)
BINANCE_INTERVAL = {
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w"
}

def fetch_ohlcv(symbol: str, tf: str, limit: int = 250, retries: int = 2) -> Optional[pd.DataFrame]:
    """
    Internal: Fetch OHLCV data with retry logic.
    Tries (in order):
      1) KuCoin Spot (retries if timeout/error)
      2) KuCoin Futures (retries if timeout/error)
      3) Binance Spot (fallback, no retry)
    """
    # 1) KuCoin Spot (with retries)
    if tf in TF_MAP:
        for sym in (symbol, symbol.replace('-', '/')):
            for attempt in range(retries):
                try:
                    url = f"https://api.kucoin.com/api/v1/market/candles?type={TF_MAP[tf]}&symbol={sym}&limit={min(limit, 1500)}"
                    r = requests.get(url, timeout=15)
                    r.raise_for_status()
                    bars = r.json().get("data") or []
                    if not bars:
                        continue
                    
                    df = pd.DataFrame(bars, columns=[
                        "timestamp","open","close","high","low","volume","turnover"
                    ]).iloc[::-1].reset_index(drop=True)
                    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df.set_index("timestamp", inplace=True)
                    return df[["open","high","low","close","volume"]]
                except requests.Timeout:
                    if attempt < retries - 1:
                        time.sleep(1)  # Wait before retry
                    continue
                except Exception:
                    continue

    # 2) KuCoin Futures (with retries)
    if tf in FUTURES_GRAN:
        for sym in (symbol, symbol.replace('-', '/')):
            for attempt in range(retries):
                try:
                    gran = FUTURES_GRAN[tf]
                    url = f"https://api-futures.kucoin.com/api/v1/kline/query?symbol={sym}&granularity={gran}&limit={min(limit, 1500)}"
                    r = requests.get(url, timeout=15)
                    r.raise_for_status()
                    bars = r.json().get("data") or []
                    if not bars:
                        continue
                    
                    df = pd.DataFrame(bars, columns=[
                        "timestamp","open","close","high","low","volume"
                    ]).iloc[::-1].reset_index(drop=True)
                    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    return df[["open","high","low","close","volume"]]
                except requests.Timeout:
                    if attempt < retries - 1:
                        time.sleep(1)  # Wait before retry
                    continue
                except Exception:
                    continue

    # 3) Binance fallback (no retry)
    b_interval = BINANCE_INTERVAL.get(tf)
    if b_interval:
        sym_b = symbol.replace('-', '')
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={sym_b}&interval={b_interval}&limit={min(limit, 1000)}"
            r = requests.get(url, timeout=15)
            r.raise_for_status()
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

    return None

def get_ohlcv(symbol: str, interval: str, limit: int = 250) -> Optional[pd.DataFrame]:
    """Public alias matching main.py signature."""
    return fetch_ohlcv(symbol, tf=interval, limit=limit)

# ====== ORDERBOOK & TICK DATA ======

def fetch_orderbook_l2(symbol: str, depth: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the L2 order book (top N levels) from KuCoin. Returns (bids_df, asks_df)."""
    for sym in (symbol, symbol.replace('-', '/')):
        try:
            url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_{depth}?symbol={sym}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json().get('data', {})
            bids = pd.DataFrame(data.get('bids', []), columns=['price', 'size']).astype(float)
            asks = pd.DataFrame(data.get('asks', []), columns=['price', 'size']).astype(float)
            return bids, asks
        except Exception:
            continue
    return pd.DataFrame(), pd.DataFrame()

def fetch_trade_ticks(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch recent trade ticks from KuCoin. Returns DataFrame with [price, size, side, time]."""
    for sym in (symbol, symbol.replace('-', '/')):
        try:
            url = f"https://api.kucoin.com/api/v1/market/histories?symbol={sym}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json().get('data', [])
            if not data:
                continue
            df = pd.DataFrame(data)
            df = df.rename(columns={'tradeId': 'id', 'tradeType': 'side'})
            df['price'] = df['price'].astype(float)
            df['size'] = df['size'].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            return df[['price', 'size', 'side', 'time']].sort_values('time').reset_index(drop=True)
        except Exception:
            continue
    return pd.DataFrame()

# ========== ENTRY PRICE ==========

DEFAULT_SLIPPAGE = 0.001  # 0.1% for KuCoin

def get_live_entry_price(
    symbol: str,
    signal_type: str,
    tf: str = "15min",
    slippage: float = DEFAULT_SLIPPAGE,
    long_adjust: float = 0.9900,
    short_adjust: float = 1.0100,
    debug: bool = False
) -> Optional[float]:
    """
    Fetch best bid/ask price from KuCoin orderbook for entry.
    Falls back to OHLCV close if orderbook unavailable.
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
    except Exception as e:
        source = f"error: {e}"

    # Fallback to OHLCV close
    if entry_price is None:
        df = fetch_ohlcv(symbol, tf)
        if df is not None and not df.empty:
            entry_price = float(df["close"].iloc[-1])
            source = f"ohlcv_close_{tf}"
        else:
            return None

    # Apply adjustment
    if signal_type.upper() == "LONG":
        entry_price *= long_adjust
    elif signal_type.upper() == "SHORT":
        entry_price *= short_adjust

    return float(entry_price)
