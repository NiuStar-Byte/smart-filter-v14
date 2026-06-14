#!/usr/bin/env python3
"""
ASTERDEX Account Trades Fetcher - Fetch EXECUTED TRADES (not orders)
Uses /fapi/v3/trades endpoint (USER_DATA) which returns actual trade executions.

This is different from /fapi/v3/allOrders which returns order statuses.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Load environment
env_file = Path('/Users/geniustarigan/.openclaw/workspace/.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                line = line.replace('export ', '', 1).strip()
                key, value = line.split('=', 1)
                value = value.strip().strip("'\"")
                os.environ[key.strip()] = value

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY
import requests

API_BASE = "https://fapi.asterdex.com"


def fetch_account_trades(auth, symbol, start_date='2026-06-01', limit=100):
    """
    Fetch EXECUTED TRADES from /fapi/v3/trades endpoint.
    
    This returns actual trade executions with:
    - quoteQty: Quote asset quantity at execution
    - realizedPnl: Realized profit/loss (if position closed)
    - commission: Trading fee
    - etc.
    """
    try:
        start_dt = datetime.fromisoformat(start_date)
        start_ts = int(start_dt.timestamp() * 1000)
        
        # Convert symbol
        symbol_asterdex = symbol.replace('-', '')
        
        params = {
            'symbol': symbol_asterdex,
            'startTime': start_ts,
            'limit': limit,
        }
        
        signed_params = auth.sign_request_v3(base_params=params)
        
        response = requests.get(
            f"{API_BASE}/fapi/v3/trades",  # <-- Account Trades endpoint
            params=signed_params,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[WARN] API error for {symbol}: {response.status_code} - {response.text[:100]}")
            return []
            
    except Exception as e:
        print(f"[WARN] Failed to fetch trades for {symbol}: {e}")
        return []


def main():
    """Main entry point - test fetch account trades."""
    print("\n" + "="*60)
    print("ASTERDEX ACCOUNT TRADES FETCHER TEST")
    print("Fetching EXECUTED TRADES from /fapi/v3/trades")
    print("="*60 + "\n")
    
    try:
        auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to initialize auth: {e}")
        return
    
    # Test a few symbols
    symbols = ['BTC-USDT', 'ETH-USDT', 'NEAR-USDT']
    
    for symbol in symbols:
        print(f"[INFO] Fetching account trades for {symbol}...")
        
        trades = fetch_account_trades(auth, symbol, start_date='2026-06-01', limit=100)
        
        if trades:
            print(f"  ✅ Found {len(trades)} executed trades")
            if len(trades) > 0:
                print(f"     Sample trade: {json.dumps(trades[0], indent=2)[:200]}...")
        else:
            print(f"  (no trades)")
    
    print("\n✅ Test complete")


if __name__ == '__main__':
    main()
