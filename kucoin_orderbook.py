import requests
import pandas as pd

def fetch_orderbook(symbol: str, depth: int = 100) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetches KuCoin L2 order book for the given symbol and depth.
    Returns: (bids_df, asks_df) both as pandas DataFrames with columns ['price','size'].
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
    wall_levels: int = 5,
    min_wall_size: float = 0.5,
    depth: int = 100,
    band_pct: float = 0.01
) -> dict:
    """
    Computes top-of-book wall sizes and delta for the given symbol.
    Returns a dict with buy_wall, sell_wall, wall_delta, midprice and top-level lists.
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

    # Ensure proper ordering
    bids = bids.sort_values("price", ascending=False).reset_index(drop=True)
    asks = asks.sort_values("price", ascending=True).reset_index(drop=True)

    best_bid = bids['price'].iat[0]
    best_ask = asks['price'].iat[0]
    midprice = (best_bid + best_ask) / 2

    # Restrict to a small band around midprice (helps ignore distant levels)
    low, high = midprice * (1 - band_pct), midprice * (1 + band_pct)
    bids_in_band = bids[bids['price'] >= low]
    asks_in_band = asks[asks['price'] <= high]

    buy_top = bids_in_band.head(wall_levels)
    sell_top = asks_in_band.head(wall_levels)

    if min_wall_size > 0:
        buy_top = buy_top[buy_top['size'] >= min_wall_size]
        sell_top = sell_top[sell_top['size'] >= min_wall_size]

    buy_wall = float(buy_top['size'].sum())
    sell_wall = float(sell_top['size'].sum())
    wall_delta = float(buy_wall - sell_wall)

    return {
        'buy_wall': buy_wall,
        'sell_wall': sell_wall,
        'wall_delta': wall_delta,
        'midprice': float(midprice),
        'buy_top_levels': buy_top.values.tolist(),
        'sell_top_levels': sell_top.values.tolist()
    }

def get_resting_order_density(
    symbol: str,
    top_n: int = 5,
    depth: int = 100
) -> dict:
    """
    Compute resting density near the top N levels.
    Returns a dict with bid_density (%) and ask_density (%) and raw totals.
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
    bid_total = float(bids['size'].sum())
    ask_total = float(asks['size'].sum())

    bid_density = round(100 * bid_top / bid_total, 2) if bid_total > 0 else 0.0
    ask_density = round(100 * ask_top / ask_total, 2) if ask_total > 0 else 0.0

    return {
        'bid_density': bid_density,
        'ask_density': ask_density,
        'top_n': top_n,
        'depth': depth,
        'bid_total': bid_total,
        'ask_total': ask_total,
        'bid_top': float(bid_top),
        'ask_top': float(ask_top),
    }

# No import-time side-effects (no automatic fetching or printing).
# Consumers must call fetch_orderbook(), get_order_wall_delta() or get_resting_order_density() explicitly.

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
