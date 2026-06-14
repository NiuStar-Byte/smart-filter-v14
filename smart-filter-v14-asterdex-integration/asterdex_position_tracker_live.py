#!/usr/bin/env python3
"""
ASTERDEX REAL-TIME POSITION TRACKER
Tracks positions opened Jun 7+ on main account (0xDad4a337c537b4590e67C53b575Fe9Fe6AacCD42)
Fetches closures continuously via authenticated /fapi/v3/allOrders API
"""
import json
import os
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# Use proven AsterV3Auth
from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_MAIN_ACCOUNT, ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY

class AsterdexPositionTrackerLive:
    def __init__(self):
        self.main_account = ASTER_MAIN_ACCOUNT
        self.signer = ASTER_API_WALLET_ADDRESS
        self.private_key = ASTER_API_WALLET_PRIVATE_KEY
        
        # Initialize auth (same as entry_poster)
        self.auth = AsterV3Auth(self.signer, self.private_key)
        
        self.api_base = "https://fapi.asterdex.com"
        self.live_file = Path('ASTERDEX_POSITIONS_LIVE.jsonl')
        
        # Jun 7 cutoff
        self.cutoff = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)
        
        # Track processed positions to avoid duplicates
        self.tracked_position_ids = set()
        self._load_tracked_positions()
        
        print(f"✅ Tracker initialized")
        print(f"   Main account: {self.main_account}")
        print(f"   Signer: {self.signer}")
        print(f"   Filter: opened >= 2026-06-07 00:00:00 UTC")
        print(f"   Tracking {len(self.tracked_position_ids)} existing positions")
    
    def _load_tracked_positions(self):
        """Load existing position IDs to avoid re-writing"""
        if self.live_file.exists():
            try:
                with open(self.live_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            pos = json.loads(line)
                            pos_id = pos.get('position_id')
                            if pos_id:
                                self.tracked_position_ids.add(pos_id)
            except Exception as e:
                print(f"⚠️ Error loading tracked positions: {e}")
    
    def _fetch_orders(self, symbol: str) -> list:
        """
        Fetch all orders for symbol using authenticated /fapi/v3/allOrders
        Uses AsterV3Auth signing (proven working with entry_poster)
        """
        try:
            url = f"{self.api_base}/fapi/v3/allOrders"
            
            # Build params (will be signed)
            params = {
                'symbol': symbol,
                'limit': '1000',
            }
            
            # Sign with same method as entry_poster
            signed_params = self.auth.sign_request_v3(params, main_account=self.main_account)
            
            # Make GET request
            response = requests.get(url, params=signed_params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
            else:
                if response.status_code != 400:  # Don't spam logs on 400s
                    print(f"⚠️ {symbol}: {response.status_code}")
            
            return []
        except Exception as e:
            print(f"❌ {symbol}: {e}")
            return []
    
    def _match_to_positions(self, orders: list, symbol: str) -> list:
        """Match entry/exit orders into positions"""
        positions = []
        filled = [o for o in orders if o.get('status') == 'FILLED']
        
        buys = [o for o in filled if o.get('side') == 'BUY']
        sells = [o for o in filled if o.get('side') == 'SELL']
        
        # LONG: BUY → SELL
        for buy in buys:
            buy_time = buy.get('time', 0) / 1000
            buy_entry_time = datetime.fromtimestamp(buy_time, tz=timezone.utc)
            
            # Skip if before Jun 7
            if buy_entry_time < self.cutoff:
                continue
            
            buy_id = buy.get('orderId')
            
            # Find first SELL after BUY
            for sell in sells:
                sell_time = sell.get('updateTime', 0) / 1000
                sell_id = sell.get('orderId')
                
                if sell_time > buy_time:
                    pos = self._create_position('LONG', buy, sell, symbol)
                    if pos:
                        positions.append(pos)
                    break
        
        # SHORT: SELL → BUY
        for sell in sells:
            sell_time = sell.get('time', 0) / 1000
            sell_entry_time = datetime.fromtimestamp(sell_time, tz=timezone.utc)
            
            # Skip if before Jun 7
            if sell_entry_time < self.cutoff:
                continue
            
            sell_id = sell.get('orderId')
            
            # Find first BUY after SELL
            for buy in buys:
                buy_time = buy.get('updateTime', 0) / 1000
                buy_id = buy.get('orderId')
                
                if buy_time > sell_time:
                    pos = self._create_position('SHORT', sell, buy, symbol)
                    if pos:
                        positions.append(pos)
                    break
        
        return positions
    
    def _create_position(self, side: str, entry: dict, exit_order: dict, symbol: str) -> dict:
        """Create position record"""
        try:
            entry_time = datetime.fromtimestamp(entry.get('time', 0) / 1000, tz=timezone.utc)
            exit_time = datetime.fromtimestamp(exit_order.get('updateTime', 0) / 1000, tz=timezone.utc)
            
            entry_price = float(entry.get('price', 0))
            exit_price = float(exit_order.get('price', 0))
            qty = float(entry.get('executedQty', 0))
            
            if entry_price == 0 or qty == 0:
                return None
            
            if side == 'LONG':
                pnl = (exit_price - entry_price) * qty
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl = (entry_price - exit_price) * qty
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            
            pos_id = f"{symbol}_{entry.get('orderId')}_{exit_order.get('orderId')}"
            
            return {
                'position_id': pos_id,
                'symbol': symbol,
                'side': side,
                'entry_price': round(entry_price, 8),
                'exit_price': round(exit_price, 8),
                'quantity': round(qty, 8),
                'entry_order_id': entry.get('orderId'),
                'exit_order_id': exit_order.get('orderId'),
                'opened': entry_time.isoformat() + 'Z',
                'closed': exit_time.isoformat() + 'Z',
                'pnl_usd': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
            }
        except Exception as e:
            return None
    
    def run(self, symbols: list = None, interval_sec: int = 60):
        """Main loop"""
        if not symbols:
            symbols = [
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                "DOGEUSDT", "ADAUSDT", "AAVUSDT", "NEARUSDT", "AVAXUSDT",
                "KASUSDT", "DOTUSDT", "ATOMUSDT", "BIOUSDT", "PORTALUSDT",
                "SEIUSDT", "INJUSDT", "PUMPUSDT", "WIFUSDT", "ONDOUSDT",
                "ARBUSDT", "POLUSDT", "BLURUSDT", "DYDXUSDT", "LAUSSDT",
            ]
        
        print(f"\n🚀 Starting position tracker")
        print(f"   Symbols: {len(symbols)}")
        print(f"   Interval: {interval_sec}s")
        print(f"   Output: {self.live_file.name}")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                now = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
                
                total_new = 0
                
                for symbol in symbols:
                    orders = self._fetch_orders(symbol)
                    if not orders:
                        continue
                    
                    positions = self._match_to_positions(orders, symbol)
                    
                    for pos in positions:
                        if pos['position_id'] not in self.tracked_position_ids:
                            with open(self.live_file, 'a') as f:
                                f.write(json.dumps(pos) + '\n')
                            self.tracked_position_ids.add(pos['position_id'])
                            total_new += 1
                
                if total_new > 0:
                    print(f"[{now}] ✅ +{total_new} positions | Total tracked: {len(self.tracked_position_ids)}")
                else:
                    print(f"[{now}] ℹ️  No new positions")
                
                time.sleep(interval_sec)
            
            except KeyboardInterrupt:
                print("\n⏹️ Stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    tracker = AsterdexPositionTrackerLive()
    tracker.run()
