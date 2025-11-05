"""
kucoin_orderbook.py

Safe, import-friendly orderbook helpers for the Smart Filter project.

Changes made:
- Removed import-time scanning/printing side-effects.
- Added defensive parsing and retries for remote requests.
- Added an executable __main__ block so a manual quick-scan can be run
  without producing logs when imported by other modules.
- Exposes:
    - fetch_orderbook(symbol, depth=100, timeout=5, retries=1)
    - get_order_wall_delta(symbol, wall_levels=5, min_wall_size=0.5, depth=100, band_pct=0.01, timeout=5, retries=1)
    - get_resting_order_density(symbol, top_n=5, depth=100, timeout=5, retries=1)
  which are backward compatible with previous signatures (additional optional params).
"""

from typing import Tuple, Optional, List, Dict, Any
import requests
import pandas as pd
import time
import os

# Default token list (kept for convenience, not executed at import)
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


def _to_float_safe(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x))
        except Exception:
            return default


def fetch_orderbook(symbol: str, depth: int = 100, timeout: int = 5, retries: int = 1) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetches KuCoin L2 order book for the given symbol and depth.

    Returns: (bids_df, asks_df) both as pandas DataFrames with columns ['price','size'],
    or (None, None) on failure.

    Optional:
      - timeout: HTTP request timeout in seconds
      - retries: number of attempts (simple retry with short backoff)
    """
    endpoints = [symbol, symbol.replace('-', '/')]
    backoff = 0.25
    for attempt in range(max(1, retries)):
        for sym in endpoints:
            url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_{depth}?symbol={sym}"
            try:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                data = resp.json().get('data', {})
                bids_raw = data.get('bids', []) or []
                asks_raw = data.get('asks', []) or []
                # Normalize to DataFrame with correct dtypes; handle empty gracefully
                bids = pd.DataFrame(bids_raw, columns=['price', 'size']) if len(bids_raw) > 0 else pd.DataFrame(columns=['price', 'size'])
                asks = pd.DataFrame(asks_raw, columns=['price', 'size']) if len(asks_raw) > 0 else pd.DataFrame(columns=['price', 'size'])
                # Coerce numeric types; if conversion fails keep as empty frames
                if not bids.empty:
                    bids = bids.astype(float, errors='ignore')
                if not asks.empty:
                    asks = asks.astype(float, errors='ignore')
                return bids, asks
            except Exception:
                # Try next format / endpoint
                continue
        # simple backoff before next attempt
        time.sleep(backoff)
        backoff = min(backoff * 2, 2.0)
    return None, None


def get_order_wall_delta(
    symbol: str,
    wall_levels: int = 5,
    min_wall_size: float = 0.5,
    depth: int = 100,
    band_pct: float = 0.01,
    timeout: int = 5,
    retries: int = 1
) -> dict:
    """
    Computes top-of-book wall sizes and delta for the given symbol.

    Returns a dict:
    {
        'buy_wall': float,
        'sell_wall': float,
        'wall_delta': float,
        'midprice': float|None,
        'buy_top_levels': [[price, size], ...],
        'sell_top_levels': [[price, size], ...]
    }

    Parameters:
      - wall_levels: number of top levels to consider
      - min_wall_size: minimum size to count toward a wall
      - band_pct: fraction around midprice to consider (e.g., 0.01 -> +/-1%)
      - timeout, retries: passed to fetch_orderbook
    """
    bids, asks = fetch_orderbook(symbol, depth=depth, timeout=timeout, retries=retries)
    if bids is None or asks is None or bids.empty or asks.empty:
        return {
            'buy_wall': 0.0,
            'sell_wall': 0.0,
            'wall_delta': 0.0,
            'midprice': None,
            'buy_top_levels': [],
            'sell_top_levels': []
        }

    # Ensure numeric types and ordering
    try:
        bids['price'] = pd.to_numeric(bids['price'], errors='coerce')
        bids['size'] = pd.to_numeric(bids['size'], errors='coerce')
        asks['price'] = pd.to_numeric(asks['price'], errors='coerce')
        asks['size'] = pd.to_numeric(asks['size'], errors='coerce')
    except Exception:
        # best-effort; if coercion fails, fall back to default empty results
        return {
            'buy_wall': 0.0,
            'sell_wall': 0.0,
            'wall_delta': 0.0,
            'midprice': None,
            'buy_top_levels': [],
            'sell_top_levels': []
        }

    bids = bids.dropna(subset=['price', 'size']).sort_values("price", ascending=False).reset_index(drop=True)
    asks = asks.dropna(subset=['price', 'size']).sort_values("price", ascending=True).reset_index(drop=True)

    if bids.empty or asks.empty:
        return {
            'buy_wall': 0.0,
            'sell_wall': 0.0,
            'wall_delta': 0.0,
            'midprice': None,
            'buy_top_levels': [],
            'sell_top_levels': []
        }

    best_bid = float(bids['price'].iat[0])
    best_ask = float(asks['price'].iat[0])
    midprice = (best_bid + best_ask) / 2.0

    # Restrict to band around midprice
    low, high = midprice * (1 - band_pct), midprice * (1 + band_pct)
    bids_in_band = bids[bids['price'] >= low]
    asks_in_band = asks[asks['price'] <= high]

    buy_top = bids_in_band.head(wall_levels)
    sell_top = asks_in_band.head(wall_levels)

    if min_wall_size > 0:
        buy_top = buy_top[buy_top['size'] >= min_wall_size]
        sell_top = sell_top[sell_top['size'] >= min_wall_size]

    # Sum sizes; if empty, result is 0.0
    buy_wall = float(buy_top['size'].sum()) if not buy_top.empty else 0.0
    sell_wall = float(sell_top['size'].sum()) if not sell_top.empty else 0.0
    wall_delta = float(buy_wall - sell_wall)

    return {
        'buy_wall': buy_wall,
        'sell_wall': sell_wall,
        'wall_delta': wall_delta,
        'midprice': float(midprice),
        'buy_top_levels': buy_top[['price', 'size']].values.tolist() if not buy_top.empty else [],
        'sell_top_levels': sell_top[['price', 'size']].values.tolist() if not sell_top.empty else []
    }


def get_resting_order_density(
    symbol: str,
    top_n: int = 5,
    depth: int = 100,
    timeout: int = 5,
    retries: int = 1
) -> dict:
    """
    Compute resting density near the top N levels.

    Returns:
    {
      'bid_density': float (percentage 0..100),
      'ask_density': float,
      'top_n': int,
      'depth': int,
      'bid_total': float,
      'ask_total': float,
      'bid_top': float,
      'ask_top': float
    }
    """
    bids, asks = fetch_orderbook(symbol, depth=depth, timeout=timeout, retries=retries)
    if bids is None or asks is None or bids.empty or asks.empty:
        return {
            'bid_density': 0.0, 'ask_density': 0.0,
            'top_n': top_n, 'depth': depth,
            'bid_total': 0.0, 'ask_total': 0.0,
            'bid_top': 0.0, 'ask_top': 0.0
        }

    try:
        bids['price'] = pd.to_numeric(bids['price'], errors='coerce')
        bids['size'] = pd.to_numeric(bids['size'], errors='coerce')
        asks['price'] = pd.to_numeric(asks['price'], errors='coerce')
        asks['size'] = pd.to_numeric(asks['size'], errors='coerce')
    except Exception:
        return {
            'bid_density': 0.0, 'ask_density': 0.0,
            'top_n': top_n, 'depth': depth,
            'bid_total': 0.0, 'ask_total': 0.0,
            'bid_top': 0.0, 'ask_top': 0.0
        }

    bids_clean = bids.dropna(subset=['size'])
    asks_clean = asks.dropna(subset=['size'])

    bid_top = float(bids_clean.sort_values("price", ascending=False)['size'].head(top_n).sum()) if not bids_clean.empty else 0.0
    ask_top = float(asks_clean.sort_values("price", ascending=True)['size'].head(top_n).sum()) if not asks_clean.empty else 0.0
    bid_total = float(bids_clean['size'].sum()) if not bids_clean.empty else 0.0
    ask_total = float(asks_clean['size'].sum()) if not asks_clean.empty else 0.0

    bid_density = round(100 * bid_top / bid_total, 2) if bid_total > 0 else 0.0
    ask_density = round(100 * ask_top / ask_total, 2) if ask_total > 0 else 0.0

    return {
        'bid_density': bid_density,
        'ask_density': ask_density,
        'top_n': top_n,
        'depth': depth,
        'bid_total': bid_total,
        'ask_total': ask_total,
        'bid_top': bid_top,
        'ask_top': ask_top,
    }


# -----------------------
# Optional CLI / manual scan
# -----------------------
if __name__ == "__main__":
    # When executed directly, run a quick human-friendly scan and print results.
    from datetime import datetime
    # Allow overriding tokens via env var (comma-separated) or command-line fallback
    tokens_env = os.getenv("KUCOIN_OB_TOKENS")
    tokens_to_scan: List[str] = TOKENS
    if tokens_env:
        tokens_to_scan = [t.strip() for t in tokens_env.split(",") if t.strip()]

    depth = int(os.getenv("KUCOIN_OB_DEPTH", "100"))
    top_n = int(os.getenv("KUCOIN_OB_TOPN", "5"))
    timeout = int(os.getenv("KUCOIN_OB_TIMEOUT", "5"))
    retries = int(os.getenv("KUCOIN_OB_RETRIES", "1"))

    print(f"[INFO] kucoin_orderbook quick-scan started at {datetime.utcnow().isoformat()} (depth={depth}, top_n={top_n})", flush=True)
    for symbol in tokens_to_scan:
        try:
            ob = get_order_wall_delta(symbol, wall_levels=top_n, depth=depth, timeout=timeout, retries=retries)
            print(
                f"[OrderBookDeltaLog] {symbol} | ts={datetime.utcnow().isoformat()} | buy_wall={ob['buy_wall']} | sell_wall={ob['sell_wall']} | wall_delta={ob['wall_delta']} | midprice={ob['midprice']}",
                flush=True
            )
        except Exception as e:
            print(f"[OrderBookDeltaLog] {symbol} ERROR: {e}", flush=True)

    for symbol in tokens_to_scan:
        try:
            dens = get_resting_order_density(symbol, top_n=top_n, depth=depth, timeout=timeout, retries=retries)
            print(
                f"[RestingOrderDensityLog] {symbol} | ts={datetime.utcnow().isoformat()} | bid_density={dens['bid_density']}% | ask_density={dens['ask_density']}% | bid_top={dens['bid_top']} | ask_top={dens['ask_top']} | bid_total={dens['bid_total']} | ask_total={dens['ask_total']}",
                flush=True
            )
        except Exception as e:
            print(f"[RestingOrderDensityLog] {symbol} ERROR: {e}", flush=True)

    print(f"[INFO] kucoin_orderbook quick-scan completed at {datetime.utcnow().isoformat()}", flush=True)
