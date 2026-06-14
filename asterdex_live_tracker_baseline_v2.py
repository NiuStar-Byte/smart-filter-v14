#!/usr/bin/env python3
"""
Asterdex Live Tracker Baseline v2
Polls account positions every 60 seconds and displays live snapshot
"""

import json
import time
import requests
import os
from datetime import datetime
from pathlib import Path

def get_account_info():
    """Fetch account info from Asterdex"""
    from aster_v3_auth import AsterV3Auth
    from asterdex_config import (
        ASTER_MAIN_ACCOUNT,
        ASTER_API_WALLET_ADDRESS,
        ASTER_API_WALLET_PRIVATE_KEY,
        ASTERDEX_ENDPOINT,
        API_VERSION
    )
    
    auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
    
    try:
        params = {}
        signed_params = auth.sign_request_v3(params, main_account=ASTER_MAIN_ACCOUNT)
        
        url = f"{ASTERDEX_ENDPOINT}/fapi/{API_VERSION}/account"
        response = requests.get(url, params=signed_params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to fetch account info: {e}")
        return None

def format_snapshot():
    """Get and format current account snapshot"""
    data = get_account_info()
    
    if not data:
        return None
    
    positions = data.get('positions', [])
    open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
    
    snapshot = {
        'timestamp': datetime.utcnow().isoformat(),
        'total_positions': len(open_positions),
        'positions': []
    }
    
    total_pnl = 0
    for pos in open_positions:
        symbol = pos.get('symbol')
        amount = float(pos.get('positionAmt', 0))
        entry_price = float(pos.get('entryPrice', 0))
        mark_price = float(pos.get('markPrice', 0))
        pnl = float(pos.get('unrealizedProfit', 0))
        pnl_pct = (pnl / (entry_price * amount)) * 100 if entry_price * amount != 0 else 0
        
        total_pnl += pnl
        
        snapshot['positions'].append({
            'symbol': symbol,
            'side': 'LONG' if amount > 0 else 'SHORT',
            'quantity': abs(amount),
            'entry_price': entry_price,
            'mark_price': mark_price,
            'pnl_usd': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2)
        })
    
    snapshot['total_pnl'] = round(total_pnl, 2)
    return snapshot

def display_snapshot(snapshot):
    """Display snapshot in readable format"""
    print("\n" + "="*80)
    print(f"ASTERDEX LIVE TRACKER - {snapshot['timestamp']}")
    print("="*80)
    print(f"Total Open Positions: {snapshot['total_positions']}")
    print(f"Total P&L: ${snapshot['total_pnl']:.2f}")
    print("-"*80)
    
    if snapshot['total_positions'] == 0:
        print("No open positions")
    else:
        print(f"{'Symbol':<12} {'Side':<6} {'Qty':<12} {'Entry':<12} {'Mark':<12} {'P&L USD':<12} {'P&L %':<8}")
        print("-"*80)
        
        for pos in snapshot['positions']:
            print(f"{pos['symbol']:<12} {pos['side']:<6} {pos['quantity']:<12.2f} ${pos['entry_price']:<11.2f} ${pos['mark_price']:<11.2f} ${pos['pnl_usd']:<11.2f} {pos['pnl_pct']:>6.2f}%")
    
    print("="*80 + "\n")

def main():
    """Main loop - refresh every 60 seconds"""
    print("🚀 Starting Asterdex Live Tracker (60s refresh interval)")
    
    while True:
        try:
            snapshot = format_snapshot()
            
            if snapshot:
                display_snapshot(snapshot)
                
                # Save to file for reference
                output_file = "/tmp/asterdex_live_snapshot.json"
                with open(output_file, 'w') as f:
                    json.dump(snapshot, f, indent=2)
                print(f"✅ Snapshot saved to {output_file}")
            
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
