#!/usr/bin/env python3
"""
ASTERDEX CLOSED POSITIONS TRACKER - FINAL VERSION
Tracks positions that have closed (entry LIMIT + exit MARKET)
Matches orders and calculates realized P&L
Updates every 60 seconds
"""
import json
import requests
import time
from datetime import datetime, timezone
from pathlib import Path
from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_MAIN_ACCOUNT, ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY

class AsterdexClosedTracker:
    def __init__(self):
        self.auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
        self.api_base = "https://fapi.asterdex.com"
        self.main_account = ASTER_MAIN_ACCOUNT
        self.output_file = Path('ASTERDEX_CLOSED_POSITIONS_LIVE.jsonl')
        self.cutoff = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)
        self.tracked_ids = set()
        
        self._load_tracked()
        
        print(f"✅ Closed tracker initialized")
        print(f"   Filter: opened >= 2026-06-07 00:00 UTC")
        print(f"   Output: {self.output_file.name}")
        print(f"   Tracked: {len(self.tracked_ids)} existing closed positions")
    
    def _load_tracked(self):
        """Load already-tracked position IDs"""
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            pos = json.loads(line)
                            pos_id = pos.get('position_id')
                            if pos_id:
                                self.tracked_ids.add(pos_id)
            except:
                pass
    
    def _fetch_orders(self):
        """Fetch all FILLED orders"""
        try:
            params = {}
            signed = self.auth.sign_request_v3(params, main_account=self.main_account)
            
            resp = requests.get(f"{self.api_base}/fapi/v3/allOrders", params=signed, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
            return []
        except Exception as e:
            print(f"❌ Fetch error: {e}")
            return []
    
    def _match_closed_positions(self, all_orders):
        """Match LIMIT entries with MARKET exits"""
        positions = []
        
        # Filter to FILLED orders from Jun 7+
        filled = [o for o in all_orders if o.get('status') == 'FILLED']
        filled_jun7 = [o for o in filled if datetime.fromtimestamp(o.get('time', 0) / 1000, tz=timezone.utc) >= self.cutoff]
        
        # Separate by type
        limits = [o for o in filled_jun7 if o.get('type') == 'LIMIT']
        markets = [o for o in filled_jun7 if o.get('type') == 'MARKET']
        
        # For each LIMIT, find matching MARKET (opposite side, same symbol, after entry)
        matched_limit_ids = set()
        
        for limit in limits:
            limit_id = limit.get('orderId')
            limit_time = limit.get('time', 0)
            limit_symbol = limit.get('symbol')
            limit_side = limit.get('side')
            opposite_side = 'SELL' if limit_side == 'BUY' else 'BUY'
            
            for market in markets:
                market_time = market.get('time', 0)
                market_symbol = market.get('symbol')
                market_side = market.get('side')
                market_id = market.get('orderId')
                
                if (market_time > limit_time and 
                    market_symbol == limit_symbol and
                    market_side == opposite_side):
                    
                    # Found matching exit!
                    pos_id = f"{limit_symbol}_{limit_id}_{market_id}"
                    
                    if pos_id not in self.tracked_ids:
                        pos = self._create_position(limit, market, pos_id)
                        if pos:
                            positions.append(pos)
                            self.tracked_ids.add(pos_id)
                    
                    matched_limit_ids.add(limit_id)
                    break
        
        return positions
    
    def _create_position(self, entry_order, exit_order, pos_id):
        """Create closed position record"""
        try:
            entry_time = datetime.fromtimestamp(entry_order.get('time', 0) / 1000, tz=timezone.utc)
            exit_time = datetime.fromtimestamp(exit_order.get('time', 0) / 1000, tz=timezone.utc)
            
            entry_price = float(entry_order.get('price', 0))
            exit_price = float(exit_order.get('price', 0))
            quantity = float(entry_order.get('executedQty', 0))
            
            if entry_price == 0 or quantity == 0:
                return None
            
            # Calculate P&L
            if entry_order.get('side') == 'BUY':
                pnl = (exit_price - entry_price) * quantity
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                side = 'LONG'
            else:
                pnl = (entry_price - exit_price) * quantity
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                side = 'SHORT'
            
            duration = (exit_time - entry_time).total_seconds() / 3600  # hours
            
            return {
                'position_id': pos_id,
                'symbol': entry_order.get('symbol'),
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'entry_order_id': entry_order.get('orderId'),
                'exit_order_id': exit_order.get('orderId'),
                'opened': entry_time.isoformat() + 'Z',
                'closed': exit_time.isoformat() + 'Z',
                'duration_hours': round(duration, 2),
                'pnl_usd': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
            }
        except:
            return None
    
    def run(self, interval_sec=60):
        """Main loop"""
        print(f"\n🚀 Starting closed positions tracker (interval={interval_sec}s)")
        print(f"Expected: 63 closed positions (growing)\n")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                now = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
                
                orders = self._fetch_orders()
                new_positions = self._match_closed_positions(orders)
                
                if new_positions:
                    print(f"[{now}] ✅ +{len(new_positions)} new closed positions")
                    
                    # Write to file
                    with open(self.output_file, 'a') as f:
                        for pos in new_positions:
                            f.write(json.dumps(pos) + '\n')
                    
                    # Calculate summary
                    wins = len([p for p in new_positions if p.get('pnl_usd', 0) > 0])
                    losses = len([p for p in new_positions if p.get('pnl_usd', 0) < 0])
                    total_pnl = sum(p.get('pnl_usd', 0) for p in new_positions)
                    
                    print(f"       {wins}W {losses}L | P&L: ${total_pnl:+.2f}")
                else:
                    print(f"[{now}] ℹ️  No new closures")
                
                print(f"       Total tracked: {len(self.tracked_ids)}")
                
                time.sleep(interval_sec)
            
            except KeyboardInterrupt:
                print("\n⏹️ Stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    tracker = AsterdexClosedTracker()
    tracker.run()
