import requests
import pandas as pd

def fetch_orderbook(symbol: str, depth: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches KuCoin L2 order book for the given symbol and depth.
    Returns: (bids_df, asks_df), both as pandas DataFrames [price, size]
    """
    for sym in (symbol, symbol.replace('-', '/')):
        url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_{depth}?symbol={sym}"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json().get('data', {})
            bids = pd.DataFrame(data.get('bids', []), columns=['price', 'size']).astype(float)
            asks = pd.DataFrame(data.get('asks', []), columns=['price', 'size']).astype(float)
            return bids, asks
        except Exception:
            continue
    return None, None

def get_order_wall_delta(
    symbol: str,
    wall_levels: int = 10,
    min_wall_size: float = 0,
    depth: int = 100,
    band_pct: float = 0.005
) -> dict:
    """
    Computes order book wall delta for a given symbol.
    Returns:
        {
            'buy_wall': float (sum of top N bid sizes),
            'sell_wall': float (sum of top N ask sizes),
            'wall_delta': float (buy_wall - sell_wall),
            'midprice': float,
            'buy_top_levels': list of (price, size),
            'sell_top_levels': list of (price, size)
        }
    """
    bids, asks = fetch_orderbook(symbol, depth=depth)
    if bids is None or asks is None or len(bids) == 0 or len(asks) == 0:
        return {
            'buy_wall': 0.0,
            'sell_wall': 0.0,
            'wall_delta': 0.0,
            'midprice': None,
            'buy_top_levels': [],
            'sell_top_levels': []
        }
    bids = bids.sort_values("price", ascending=False).reset_index(drop=True)
    asks = asks.sort_values("price", ascending=True).reset_index(drop=True)
    best_bid = bids['price'].iloc[0]
    best_ask = asks['price'].iloc[0]
    midprice = (best_bid + best_ask) / 2
    low, high = midprice * (1 - band_pct), midprice * (1 + band_pct)
    bids_in_band = bids[bids['price'] >= low]
    asks_in_band = asks[asks['price'] <= high]
    buy_top = bids_in_band.head(wall_levels)
    sell_top = asks_in_band.head(wall_levels)
    if min_wall_size > 0:
        buy_top = buy_top[buy_top['size'] >= min_wall_size]
        sell_top = sell_top[sell_top['size'] >= min_wall_size]
    buy_wall = buy_top['size'].sum()
    sell_wall = sell_top['size'].sum()
    wall_delta = buy_wall - sell_wall
    return {
        'buy_wall': float(buy_wall),
        'sell_wall': float(sell_wall),
        'wall_delta': float(wall_delta),
        'midprice': float(midprice),
        'buy_top_levels': buy_top.values.tolist(),
        'sell_top_levels': sell_top.values.tolist()
    }

# --------- Print OrderBookDeltaLog for all tokens every run ---------
TOKEN_LIST = [
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]

for token in TOKEN_LIST:
    try:
        result = get_order_wall_delta(token)
        print(f"[OrderBookDeltaLog] {token} | buy_wall={result['buy_wall']} | sell_wall={result['sell_wall']} | wall_delta={result['wall_delta']} | midprice={result['midprice']}")
    except Exception as e:
        print(f"[OrderBookDeltaLog] {token} | ERROR: {e}")

# -- End of file --
