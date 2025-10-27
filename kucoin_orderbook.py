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
    wall_levels: int = 5,             # PATCHED: Reduced from 10 to 5
    min_wall_size: float = 0.5,       # PATCHED: Increased from 0 to 0.5
    depth: int = 100,
    band_pct: float = 0.01            # PATCHED: Increased from 0.005 to 0.01
) -> dict:
    """
    Computes order book wall delta for a given symbol.
    - wall_levels: number of top levels to consider as "walls"
    - min_wall_size: minimum wall size to consider (0 disables filter)
    - band_pct: +/- % band around midprice to look for walls (optional)
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
    # Sort by price descending (bids) and ascending (asks)
    bids = bids.sort_values("price", ascending=False).reset_index(drop=True)
    asks = asks.sort_values("price", ascending=True).reset_index(drop=True)

    # Midprice (for banding)
    best_bid = bids['price'].iloc[0]
    best_ask = asks['price'].iloc[0]
    midprice = (best_bid + best_ask) / 2

    # Optional: Only consider walls within a price band
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

def get_resting_order_density(
    symbol: str,
    top_n: int = 5,                   # PATCHED: Increased from 3 to 5
    depth: int = 100
) -> dict:
    """
    Calculate resting order density as % of total book at top N levels.
    Returns:
        {
            'bid_density': %,
            'ask_density': %,
            'top_n': int,
            'depth': int,
            'bid_total': float,
            'ask_total': float,
            'bid_top': float,
            'ask_top': float,
        }
    """
    bids, asks = fetch_orderbook(symbol, depth=depth)
    if bids is None or asks is None or len(bids) < top_n or len(asks) < top_n:
        return {
            'bid_density': 0.0, 'ask_density': 0.0,
            'top_n': top_n, 'depth': depth,
            'bid_total': 0.0, 'ask_total': 0.0,
            'bid_top': 0.0, 'ask_top': 0.0
        }
    bid_top = bids.sort_values("price", ascending=False)['size'].head(top_n).sum()
    ask_top = asks.sort_values("price", ascending=True)['size'].head(top_n).sum()
    bid_total = bids['size'].sum()
    ask_total = asks['size'].sum()
    bid_density = 100 * bid_top / bid_total if bid_total > 0 else 0.0
    ask_density = 100 * ask_top / ask_total if ask_total > 0 else 0.0
    return {
        'bid_density': round(bid_density, 2),
        'ask_density': round(ask_density, 2),
        'top_n': top_n,
        'depth': depth,
        'bid_total': float(bid_total),
        'ask_total': float(ask_total),
        'bid_top': float(bid_top),
        'ask_top': float(ask_top),
    }

# --- Force print for ALL 15 tokens on EVERY import/run ---
TOKENS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT",
    "AVAX-USDT", "XLM-USDT", "LINK-USDT", "POL-USDT", "BNB-USDT",
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT",
    "CROSS-USDT", "KNC-USDT", "AIN-USDT", "ARK-USDT", "PORTAL-USDT",
    "ICNT-USDT", "OMNI-USDT", "PARTI-USDT", "VINE-USDT", "ZORA-USDT",
    "DUCK-USDT", "AUCTION-USDT", "ROAM-USDT", "FUEL-USDT", "TUT-USDT",
    "VOXEL-USDT", "ALU-USDT", "TURBO-USDT", "PROMPT-USDT", "HIPPO-USDT", 
    "DOGE-USDT", "ALGO-USDT", "DOT-USDT", "NEWT-USDT", "SAHARA-USDT",
    "PEPE-USDT", "ERA-USDT", "PENGU-USDT", "CFX-USDT", "ENA-USDT",
    "SUI-USDT", "EIGEN-USDT", "UNI-USDT", "HYPE-USDT", "TON-USDT",
    "KAS-USDT", "HBAR-USDT", "ONDO-USDT", "VIRTUAL-USDT", "AAVE-USDT",
    "GALA-USDT", "PUMP-USDT", "PEPE-USDT", "WIF-USDT", "BERA-USDT", "DYDX-USDT",
    "KAITO-USDT", "ARKM-USDT", "ATH-USDT", "NMR-USDT", "ARB-USDT",
    "WLFI-USDT", "BIO-USDT", "ASTER-USDT", "XPL-USDT", "AVNT-USDT",
    "ORDER-USDT", "XAUT-USDT"
 ]


for symbol in TOKENS:
    result = get_order_wall_delta(symbol)
    print(
        f"[OrderBookDeltaLog] {symbol} | "
        f"buy_wall={result['buy_wall']} | "
        f"sell_wall={result['sell_wall']} | "
        f"wall_delta={result['wall_delta']} | "
        f"midprice={result['midprice']}"
    )

for symbol in TOKENS:
    res = get_resting_order_density(symbol)
    print(
        f"[RestingOrderDensityLog] {symbol} | "
        f"bid_density={res['bid_density']}% | "
        f"ask_density={res['ask_density']}% | "
        f"bid_top={res['bid_top']} | ask_top={res['ask_top']} | "
        f"bid_total={res['bid_total']} | ask_total={res['ask_total']}"
    )
