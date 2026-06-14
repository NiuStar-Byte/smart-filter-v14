#!/usr/bin/env python3
"""
ASTERDEX AUTO UPDATER - Fetch new closed positions from API
Runs continuously to add new positions to ASTERDEX_POSITIONS_LIVE.jsonl
"""
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

class AsterdexAutoUpdater:
    def __init__(self, update_interval=60):
        self.update_interval = update_interval  # Check every 60 seconds
        self.cutoff = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)
        self.api_base = "https://fapi.asterdex.com"
        self.positions_file = Path('ASTERDEX_POSITIONS_LIVE.jsonl')
        self.last_check_time = None
    
    def load_existing_positions(self):
        """Load already-tracked positions"""
        positions = []
        if self.positions_file.exists():
            try:
                with open(self.positions_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            positions.append(json.loads(line))
            except:
                pass
        return positions
    
    def get_existing_order_ids(self, positions):
        """Get all order IDs we've already tracked"""
        existing = set()
        for pos in positions:
            entry_id = pos.get('entry_order_id')
            exit_id = pos.get('exit_order_id')
            if entry_id:
                existing.add(entry_id)
            if exit_id:
                existing.add(exit_id)
        return existing
    
    def fetch_orders_from_api(self, symbol):
        """Fetch all filled orders for a symbol using /fapi/v3/trades"""
        try:
            # Convert symbol format: BTC-USDT → BTCUSDT
            api_symbol = symbol.replace('-', '')
            
            url = f"{self.api_base}/fapi/v3/trades"
            params = {
                'symbol': api_symbol,
                'limit': 100,
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            return []
    
    def extract_position_from_orders(self, entry_order, exit_order):
        """Create position record from entry and exit orders"""
        try:
            entry_price = float(entry_order.get('avgPrice', 0))
            exit_price = float(exit_order.get('avgPrice', 0))
            quantity = float(entry_order.get('origQty', 0))
            
            pnl_usd = (exit_price - entry_price) * quantity if entry_order['side'] == 'BUY' else (entry_price - exit_price) * quantity
            pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_order['side'] == 'BUY' else ((entry_price - exit_price) / entry_price * 100)
            
            position = {
                "position_id": f"{entry_order.get('symbol', 'UNKNOWN')}_{entry_order.get('orderId')}_{exit_order.get('orderId')}",
                "symbol": entry_order.get('symbol', 'UNKNOWN'),
                "side": "LONG" if entry_order.get('side') == 'BUY' else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "entry_order_id": entry_order.get('orderId'),
                "exit_order_id": exit_order.get('orderId'),
                "opened": entry_order.get('time', ''),
                "closed": exit_order.get('time', ''),
                "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(pnl_pct, 2),
                "leverage": "10x"  # Default
            }
            return position
        except:
            return None
    
    def is_new_position(self, position, existing_ids):
        """Check if position is already tracked"""
        entry_id = position.get('entry_order_id')
        exit_id = position.get('exit_order_id')
        return entry_id not in existing_ids and exit_id not in existing_ids
    
    def is_recent_position(self, position):
        """Check if position closed after Jun 7"""
        try:
            closed_str = position.get('closed', '')
            if isinstance(closed_str, (int, float)):
                closed = datetime.fromtimestamp(closed_str / 1000, tz=timezone.utc)
            else:
                closed = datetime.fromisoformat(closed_str.replace('Z', '+00:00'))
            return closed >= self.cutoff
        except:
            return False
    
    def add_position(self, position):
        """Add position to JSONL file"""
        try:
            with open(self.positions_file, 'a') as f:
                f.write(json.dumps(position) + '\n')
            return True
        except:
            return False
    
    def run_once(self):
        """Single update cycle"""
        print(f"\n🔄 Checking for new positions... ({datetime.now().strftime('%H:%M:%S GMT+7')})")
        
        existing_positions = self.load_existing_positions()
        existing_ids = self.get_existing_order_ids(existing_positions)
        
        print(f"   Currently tracked: {len(existing_positions)} positions")
        
        # Common symbols to check
        symbols = ['IP-USDT', 'BNB-USDT', 'POL-USDT', 'BTC-USDT', 'ETH-USDT', 
                  'SOL-USDT', 'NEAR-USDT', 'AAVE-USDT', 'DOGE-USDT', 'XRP-USDT']
        
        new_positions = []
        
        for symbol in symbols:
            orders = self.fetch_orders_from_api(symbol)
            if not orders:
                continue
            
            # Group orders: find entry-exit pairs
            buy_orders = [o for o in orders if o.get('side') == 'BUY' and o.get('status') == 'FILLED']
            sell_orders = [o for o in orders if o.get('side') == 'SELL' and o.get('status') == 'FILLED']
            
            # Simple matching: pair by timestamp proximity
            for buy in buy_orders:
                for sell in sell_orders:
                    if sell.get('time', 0) > buy.get('time', 0):
                        position = self.extract_position_from_orders(buy, sell)
                        if position and self.is_new_position(position, existing_ids) and self.is_recent_position(position):
                            if self.add_position(position):
                                new_positions.append(position)
                                existing_ids.add(position.get('entry_order_id'))
                                existing_ids.add(position.get('exit_order_id'))
                                print(f"   ✅ Added: {position['symbol']} ${position['pnl_usd']:+.2f}")
                        break
            
            time.sleep(0.5)  # Rate limiting
        
        if new_positions:
            print(f"\n✨ Found {len(new_positions)} new position(s)")
        else:
            print(f"   No new positions found")
        
        return len(new_positions)
    
    def monitor_loop(self):
        """Continuous monitoring"""
        print("🟢 ASTERDEX AUTO UPDATER STARTED")
        print(f"   Update interval: {self.update_interval} seconds")
        print("   Tracking: Jun 7+ positions only")
        print("   Press Ctrl+C to stop\n")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                self.run_once()
                print(f"   Next check in {self.update_interval}s...\n")
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\n⏹️ Auto updater stopped")

def main():
    updater = AsterdexAutoUpdater(update_interval=60)  # Check every 60 seconds
    updater.monitor_loop()

if __name__ == '__main__':
    main()
