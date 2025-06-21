import requests
import pandas as pd

# Supported timeframes mapping
TF_MAP = {
    "2min": "2min",
    "3min": "3min",
    "5min": "5min"
}


def fetch_ohlcv(symbol: str, tf: str, limit: int = None) -> pd.DataFrame | None:
    """
    Fetches OHLCV data for a given symbol and timeframe from KuCoin REST API.
    Tries both hyphenated and slash formats for symbol if needed.

    :param symbol: Token symbol, e.g. "SPARK-USDT"
    :param tf: Timeframe key, one of TF_MAP, e.g. "3min"
    :param limit: (unused) for future extension
    :return: DataFrame with columns [open, high, low, close, volume] indexed by timestamp, or None if no data
    """
    if tf not in TF_MAP:
        print(f"[{symbol}] Unsupported timeframe: {tf}")
        return None

    for sym in (symbol, symbol.replace('-', '/')):
        try:
            url = f"https://api.kucoin.com/api/v1/market/candles?type={TF_MAP[tf]}&symbol={sym}"
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            payload = res.json()

            bars = payload.get('data') or []
            if not bars:
                # no data in this format, try next
                continue

            # Data format: [time, open, close, high, low, volume, turnover]
            df = pd.DataFrame(
                bars,
                columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"]
            )
            # Reverse to chronological
            df = df.iloc[::-1].reset_index(drop=True)
            # Cast numeric types
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            # Convert timestamp to pandas datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            # Return DataFrame with only required columns
            return df[["open", "high", "low", "close", "volume"]]

        except Exception as e:
            # Try next symbol format
            continue

    # If both formats failed
    print(f"[{symbol}] No OHLCV data fetched for formats '{symbol}' or '{symbol.replace('-', '/')}'.")
    return None
