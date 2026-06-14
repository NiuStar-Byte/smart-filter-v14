#!/usr/bin/env python3
"""
ASTERDEX Trade Fetcher - Fetch EXECUTED TRADES (not orders)
Uses /fapi/v3/trades endpoint which returns actual trade executions.

KEY FIX (Jun 8 12:26 GMT+7):
- Was: /fapi/v3/allOrders (returns order statuses)
- Now: /fapi/v3/trades (returns EXECUTED TRADES with fill data)
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Load environment variables BEFORE importing config
env_file = Path('/Users/geniustarigan/.openclaw/workspace/.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                # Handle "export KEY=VALUE" format
                line = line.replace('export ', '', 1).strip()
                key, value = line.split('=', 1)
                # Remove quotes
                value = value.strip().strip("'\"")
                os.environ[key.strip()] = value

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import auth and config
from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY

# API base
API_BASE = "https://fapi.asterdex.com"

# File locations
TRADES_FILE = Path(__file__).parent / "ASTERDEX_TRADES.jsonl"
LAST_FETCH_FILE = Path(__file__).parent / ".asterdex_trade_fetch_state.json"


def load_last_fetch_state():
    """Load when we last fetched trades."""
    if LAST_FETCH_FILE.exists():
        with open(LAST_FETCH_FILE, 'r') as f:
            return json.load(f)
    return {"last_timestamp": 0, "symbols_processed": []}


def save_fetch_state(state):
    """Save fetch state for next run."""
    with open(LAST_FETCH_FILE, 'w') as f:
        json.dump(state, f)


def fetch_account_trades_from_api(auth, symbol, start_time=None, limit=100):
    """
    Fetch EXECUTED TRADES from /fapi/v3/trades endpoint (USER_DATA).
    
    This endpoint returns actual trade execution records WITH REALIZED P&L:
    - symbol: Trading pair
    - side: BUY or SELL
    - price: Execution price
    - qty: Executed quantity
    - orderId: Order ID (crucial for matching)
    - realizedPnl: ACTUAL PROFIT/LOSS (already calculated by Asterdex!)
    - commission: Trading fee
    - time: Trade execution time
    
    This is the source of truth for performance tracking.
    """
    import requests
    
    # Convert symbol format: "BTC-USDT" → "BTCUSDT"
    symbol_asterdex = symbol.replace('-', '')
    
    params = {
        'symbol': symbol_asterdex,
        'limit': limit,
    }
    
    if start_time:
        params['startTime'] = int(start_time * 1000)  # Convert to milliseconds
    
    # Sign the request
    try:
        signed_params = auth.sign_request_v3(base_params=params)
        
        response = requests.get(
            f"{API_BASE}/fapi/v3/allOrders",  # <-- CORRECT: All filled orders (entry + exit pairs)
            params=signed_params,
            timeout=10
        )
        
        if response.status_code == 200:
            orders = response.json()
            # Filter for FILLED orders only (these represent actual executed trades)
            filled = [o for o in orders if o.get('status') == 'FILLED']
            return filled
        else:
            print(f"[ERROR] API error for {symbol}: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"[ERROR] Failed to fetch trades for {symbol}: {str(e)}")
        return []


def cache_trades(trades):
    """Append new trades to cache file (append-only)."""
    if not trades:
        return 0
    
    # Load existing trades to deduplicate by order id (orderId from /allOrders)
    existing_trade_ids = set()
    if TRADES_FILE.exists():
        with open(TRADES_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        trade = json.loads(line)
                        existing_trade_ids.add(trade.get('orderId'))
                    except:
                        pass
    
    # Append only new trades
    new_trades_count = 0
    with open(TRADES_FILE, 'a') as f:
        for trade in trades:
            order_id = trade.get('orderId')
            if order_id and order_id not in existing_trade_ids:
                # Convert symbol from API format (SOLUSDT) to display format (SOL-USDT)
                symbol = trade.get('symbol', '')
                if symbol and '-' not in symbol and symbol.endswith('USDT'):
                    # Convert SOLUSDT → SOL-USDT
                    symbol = symbol[:-4] + '-USDT'
                    trade['symbol'] = symbol
                f.write(json.dumps(trade) + '\n')
                new_trades_count += 1
    
    print(f"[INFO] Cached {new_trades_count} new trades")
    return new_trades_count


def fetch_recent_trades(symbols, hours_back=None, start_date=None):
    """
    Fetch trades from Asterdex trade history.
    Filters to only Asterdex-available symbols.
    
    Args:
        symbols: List of symbols to fetch
        hours_back: Hours back from now (default: None)
        start_date: Specific start date (ISO format or datetime) (default: Jun 7 2026)
    """
    # Initialize auth
    try:
        auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to initialize auth: {e}")
        return []
    
    # Filter to only Asterdex-available symbols
    try:
        from asterdex_available_symbols import get_available_symbols
        available_symbols = get_available_symbols(symbols)
        print(f"[INFO] Filtered to {len(available_symbols)}/{len(symbols)} Asterdex-available symbols")
    except Exception as e:
        available_symbols = symbols
        print(f"[WARN] Could not load symbol whitelist: {e}, using all symbols")
    
    state = load_last_fetch_state()
    
    # Determine start timestamp
    if start_date:
        # Parse provided start date
        try:
            if isinstance(start_date, str):
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            else:
                start_dt = start_date
        except:
            # Fallback: Jun 7 2026
            start_dt = datetime(2026, 6, 7, 0, 0, 0)
    elif hours_back:
        start_dt = datetime.now(datetime.now().astimezone().tzinfo) - timedelta(hours=hours_back)
    else:
        # Default: From Jun 7 2026 onwards (start of tracking)
        start_dt = datetime(2026, 6, 7, 0, 0, 0)
        print(f"[INFO] Fetching trades from default start date: {start_dt.isoformat()}")
    
    start_timestamp = start_dt.timestamp()
    
    all_trades = []
    
    for symbol in available_symbols:
        print(f"[INFO] Fetching trades for {symbol}...")
        trades = fetch_account_trades_from_api(auth, symbol, start_time=start_timestamp, limit=100)
        
        if trades:
            print(f"[INFO] Found {len(trades)} executed trades for {symbol}")
            # /fapi/v3/trades already has symbol, side, and realizedPnl - no enrichment needed
            all_trades.extend(trades)
    
    # Cache all trades
    cache_trades(all_trades)
    
    # Update state
    state['last_timestamp'] = datetime.now().timestamp()
    state['last_fetch'] = datetime.now().isoformat()
    state['total_trades_cached'] = sum(1 for _ in open(TRADES_FILE).readlines())
    save_fetch_state(state)
    
    return all_trades


def get_cached_trades(symbol=None, since_timestamp=None):
    """
    Load cached trades, optionally filtered.
    """
    trades = []
    
    if not TRADES_FILE.exists():
        return trades
    
    with open(TRADES_FILE, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                trade = json.loads(line)
                
                # Filter by symbol if provided
                if symbol and trade.get('symbol') != symbol:
                    continue
                
                # Filter by timestamp if provided
                if since_timestamp and trade.get('time', 0) < since_timestamp * 1000:
                    continue
                
                trades.append(trade)
            except:
                pass
    
    return trades


def main():
    """Main entry point for periodic cron execution."""
    print(f"\n[{datetime.now().isoformat()}] Starting Asterdex Trade Fetch...")
    
    # Get active symbols from config
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'smart-filter-v14-main'))
        from symbol_config_prod import get_active_symbols
        symbols = get_active_symbols()
    except:
        # Fallback to common symbols
        symbols = ['BTC-USDT', 'ETH-USDT', 'NEAR-USDT', 'SOL-USDT', 'AVAX-USDT']
        print(f"[WARN] Using fallback symbols: {symbols}")
    
    # Fetch trades from Jun 7 2026 onwards (start of real tracking)
    trades = fetch_recent_trades(symbols, start_date='2026-06-07T00:00:00Z')
    
    print(f"[INFO] Total trades fetched: {len(trades)}")
    print(f"[OK] Trade fetch complete")


if __name__ == '__main__':
    main()
