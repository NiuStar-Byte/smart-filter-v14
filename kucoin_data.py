import requests
import pandas as pd

def fetch_ohlcv(symbol, tf):
    try:
        tf_map = {
            "2min": "2min",
            "3min": "3min",
            "5min": "5min"
        }
        if tf not in tf_map:
            print(f"[{symbol}] Unsupported timeframe: {tf}")
            return None
        url = f"https://api.kucoin.com/api/v1/market/candles?type={tf_map[tf]}&symbol={symbol}"
        res = requests.get(url)
        data = res.json()
        if "data" not in data or not data["data"]:
            print(f"[{symbol}] No OHLCV data fetched.")
            return None
        df = pd.DataFrame(data["data"], columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
        df = df.iloc[::-1]
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        print(f"[{symbol}] Error fetching OHLCV: {e}")
        return None
