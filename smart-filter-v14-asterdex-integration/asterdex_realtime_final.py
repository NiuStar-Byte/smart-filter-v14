#!/usr/bin/env python3
"""
ASTERDEX REAL-TIME TRACKER - FINAL VERSION
Fetches open positions from /fapi/v3/account (verified 18 positions = Asterdex UI)
Continuous updates every 60 seconds
"""
import json
import requests
import time
from datetime import datetime, timezone
from pathlib import Path
from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_MAIN_ACCOUNT, ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY

class AsterdexRealtimeFinal:
    def __init__(self):
        self.auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
        self.api_base = "https://fapi.asterdex.com"
        self.main_account = ASTER_MAIN_ACCOUNT
        self.output_file = Path('ASTERDEX_POSITIONS_LIVE_FINAL.jsonl')
        
        print(f"✅ Tracker initialized")
        print(f"   Endpoint: {self.api_base}/fapi/v3/account")
        print(f"   Output: {self.output_file.name}")
        print(f"   Interval: 60 seconds")
    
    def _fetch_positions(self):
        """Fetch active positions from /fapi/v3/account"""
        try:
            params = {}
            signed = self.auth.sign_request_v3(params, main_account=self.main_account)
            
            resp = requests.get(f"{self.api_base}/fapi/v3/account", params=signed, timeout=10)
            
            if resp.status_code == 200:
                account = resp.json()
                positions = account.get('positions', [])
                
                # Filter to non-zero positions
                active = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
                return active
            else:
                print(f"⚠️ API error: {resp.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def _get_position_opened_time(self, symbol, side):
        """Fetch the time this position was opened from recent LIMIT orders"""
        try:
            params = {}
            signed = self.auth.sign_request_v3(params, main_account=self.main_account)
            
            resp = requests.get(f"{self.api_base}/fapi/v3/allOrders", params=signed, timeout=10)
            
            if resp.status_code != 200:
                return None
            
            orders = resp.json()
            if not isinstance(orders, list):
                return None
            
            # Find LIMIT orders for this symbol (entry orders)
            limit_side = 'BUY' if side == 'LONG' else 'SELL'
            matching = [o for o in orders 
                       if o.get('symbol') == symbol 
                       and o.get('side') == limit_side 
                       and o.get('type') == 'LIMIT'
                       and o.get('status') == 'FILLED']
            
            if matching:
                # Return most recent entry time
                latest = max(matching, key=lambda o: o.get('time', 0))
                return datetime.fromtimestamp(latest.get('time', 0) / 1000, tz=timezone.utc).isoformat() + 'Z'
            
            return None
        except:
            return None
    
    def _format_position(self, pos):
        """Format position for storage"""
        symbol = pos.get('symbol')
        amount = float(pos.get('positionAmt', 0))
        entry_price = float(pos.get('entryPrice', 0))
        unrealized = float(pos.get('unrealizedProfit', 0))
        
        # Determine side
        side = 'LONG' if amount > 0 else 'SHORT'
        
        # Get current mark price (from unrealizedProfit calculation)
        # unrealizedProfit = positionAmt * (markPrice - entryPrice)
        if entry_price > 0 and amount != 0:
            mark_price = (unrealized / amount) + entry_price
        else:
            mark_price = entry_price
        
        # Get opening time
        opened_time = self._get_position_opened_time(symbol, side)
        
        return {
            'symbol': symbol,
            'side': side,
            'quantity': abs(amount),
            'entry_price': entry_price,
            'current_mark_price': mark_price,
            'unrealized_pnl_usd': round(unrealized, 2),
            'unrealized_pnl_pct': round((unrealized / (entry_price * abs(amount)) * 100), 2) if entry_price > 0 and amount != 0 else 0,
            'leverage': pos.get('leverage', '10'),
            'opened': opened_time,
            'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
        }
    
    def run(self, interval_sec=60):
        """Main loop"""
        print(f"\n🚀 Starting real-time tracker (interval={interval_sec}s)")
        print(f"Expected: 18 open positions (matching Asterdex UI)\n")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                now = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
                
                positions = self._fetch_positions()
                
                if positions:
                    print(f"[{now}] ✅ {len(positions)} positions")
                    
                    # Write to file (append)
                    with open(self.output_file, 'a') as f:
                        for pos in positions:
                            formatted = self._format_position(pos)
                            f.write(json.dumps(formatted) + '\n')
                    
                    # Show summary
                    total_unrealized = sum(float(p.get('unrealizedProfit', 0)) for p in positions)
                    long_count = len([p for p in positions if float(p.get('positionAmt', 0)) > 0])
                    short_count = len([p for p in positions if float(p.get('positionAmt', 0)) < 0])
                    
                    print(f"       LONG: {long_count} | SHORT: {short_count} | Total Unrealized: ${total_unrealized:.2f}")
                else:
                    print(f"[{now}] ⚠️ No positions returned")
                
                time.sleep(interval_sec)
            
            except KeyboardInterrupt:
                print("\n⏹️ Stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    tracker = AsterdexRealtimeFinal()
    tracker.run()
