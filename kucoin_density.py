import pandas as pd
import requests

def get_resting_density(symbol, depth=100, levels=5):
    """
    Computes resting bid/ask density as a percentage of top N levels
    relative to total book for the given symbol.
    Returns: dict with 'bid_density', 'ask_density', 'bid_top', 'ask_top', 'bid_total', 'ask_total'
    """
    # --- Use your fetch_orderbook from earlier ---
    for sym in (symbol, symbol.replace('-', '/')):
        url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_{depth}?symbol={sym}"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json().get('data', {})
            bids = pd.DataFrame(data.get('bids', []), columns=['price', 'size']).astype(float)
            asks = pd.DataFrame(data.get('asks', []), columns=['price', 'size']).astype(float)
            break
        except Exception:
            continue
    else:
        return {
            "bid_density": 0.0, "ask_density": 0.0,
            "bid_top": 0.0, "ask_top": 0.0,
            "bid_total": 0.0, "ask_total": 0.0
        }
    bid_top = bids['size'].head(levels).sum()
    ask_top = asks['size'].head(levels).sum()
    bid_total = bids['size'].sum()
    ask_total = asks['size'].sum()
    bid_density = 100 * bid_top / bid_total if bid_total > 0 else 0.0
    ask_density = 100 * ask_top / ask_total if ask_total > 0 else 0.0
    return {
        "bid_density": bid_density,
        "ask_density": ask_density,
        "bid_top": bid_top,
        "ask_top": ask_top,
        "bid_total": bid_total,
        "ask_total": ask_total
    }
