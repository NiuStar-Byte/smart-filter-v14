#!/usr/bin/env python3
"""
ASTERDEX REAL-TIME TRACKER - Fetch closing positions from API continuously
Auto-updates ASTERDEX_POSITIONS_LIVE.jsonl every 60 seconds
ZERO manual input required - all data from Asterdex API only
Uses same Web3 EIP-712 auth as entry_poster
"""
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
import sys

class AsterdexRealTimeFetcher:
    def __init__(self):
        self.api_base = "https://fapi.asterdex.com"
        
        self.positions_file = Path('ASTERDEX_POSITIONS_LIVE.jsonl')
        self.cutoff = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)
        self.update_interval = 60  # Check every 60 seconds
        self.last_positions = set()  # Track seen order IDs
        
        # Load existing position IDs
        self._load_existing_ids()
    
    def _load_existing_ids(self):
        """Load all order IDs we've already tracked"""
        if self.positions_file.exists():
            try:
                with open(self.positions_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            pos = json.loads(line)
                            self.last_positions.add(pos.get('entry_order_id'))
                            self.last_positions.add(pos.get('exit_order_id'))
            except:
                pass
    
    def _fetch_orders(self, symbol):
        """Fetch all filled orders for a symbol from public API"""
        try:
            # /fapi/v3/allOrders is a public endpoint - no auth required
            url = f"{self.api_base}/fapi/v3/allOrders"
            params = {
                'symbol': symbol,
                'limit': 1000,
                'recvWindow': 5000,
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ API error {response.status_code} for {symbol}")
                return []
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return []
    
    def _match_orders(self, orders, symbol):
        """Match entry/exit orders into complete positions"""
        positions = []
        
        # Separate BUY and SELL orders
        buys = [o for o in orders if o.get('side') == 'BUY' and o.get('status') == 'FILLED']
        sells = [o for o in orders if o.get('side') == 'SELL' and o.get('status') == 'FILLED']
        
        shorts = [o for o in orders if o.get('side') == 'SELL' and o.get('status') == 'FILLED']
        covers = [o for o in orders if o.get('side') == 'BUY' and o.get('status') == 'FILLED']
        
        # Match LONG: BUY entry → SELL exit
        for buy in buys:
            for sell in sells:
                if sell.get('updateTime', 0) > buy.get('updateTime', 0):
                    pos = self._create_position('LONG', buy, sell, symbol)
                    if pos:
                        positions.append(pos)
                    break  # Take first exit after entry
        
        # Match SHORT: SELL entry → BUY cover
        for short in shorts:
            for cover in covers:
                if cover.get('updateTime', 0) > short.get('updateTime', 0):
                    pos = self._create_position('SHORT', short, cover, symbol)
                    if pos:
                        positions.append(pos)
                    break  # Take first cover after entry
        
        return positions
    
    def _create_position(self, side, entry_order, exit_order, symbol):
        """Create position record from matched orders"""
        try:
            entry_time = datetime.fromtimestamp(entry_order.get('time', 0) / 1000, tz=timezone.utc)
            exit_time = datetime.fromtimestamp(exit_order.get('updateTime', 0) / 1000, tz=timezone.utc)
            
            # Only include positions after Jun 7
            if entry_time < self.cutoff or exit_time < self.cutoff:
                return None
            
            entry_price = float(entry_order.get('avgPrice', 0))
            exit_price = float(exit_order.get('avgPrice', 0))
            quantity = float(entry_order.get('executedQty', 0))
            
            if not entry_price or not exit_price or not quantity:
                return None
            
            # Calculate P&L
            if side == 'LONG':
                pnl_usd = (exit_price - entry_price) * quantity
            else:  # SHORT
                pnl_usd = (entry_price - exit_price) * quantity
            
            pnl_pct = (pnl_usd / (entry_price * quantity)) * 100 if entry_price > 0 else 0
            
            position = {
                "position_id": f"{symbol}_{entry_order.get('orderId')}_{exit_order.get('orderId')}",
                "symbol": symbol,
                "side": side,
                "entry_price": round(entry_price, 8),
                "exit_price": round(exit_price, 8),
                "quantity": quantity,
                "entry_order_id": entry_order.get('orderId'),
                "exit_order_id": exit_order.get('orderId'),
                "opened": entry_order.get('time'),
                "closed": exit_order.get('updateTime'),
                "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(pnl_pct, 2),
                "leverage": "10x"
            }
            
            return position
        except Exception as e:
            return None
    
    def _is_new_position(self, position):
        """Check if position is new (not yet tracked)"""
        entry_id = position.get('entry_order_id')
        exit_id = position.get('exit_order_id')
        return entry_id not in self.last_positions and exit_id not in self.last_positions
    
    def _save_position(self, position):
        """Append position to JSONL file"""
        try:
            with open(self.positions_file, 'a') as f:
                f.write(json.dumps(position) + '\n')
            self.last_positions.add(position.get('entry_order_id'))
            self.last_positions.add(position.get('exit_order_id'))
            return True
        except:
            return False
    
    def fetch_cycle(self):
        """Single fetch cycle - check all symbols for new closed positions"""
        print(f"\n🔄 Fetching from Asterdex API... ({datetime.now().strftime('%H:%M:%S GMT+7')})")
        
        # Common trading symbols
        symbols = [
            'BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'BNB-USDT', 'XRP-USDT',
            'DOGE-USDT', 'ADA-USDT', 'POLKADOT-USDT', 'AAVE-USDT', 'NEAR-USDT',
            'AVAX-USDT', 'ATOM-USDT', 'INJ-USDT', 'PUMP-USDT', 'TURBO-USDT',
            'FUN-USDT', 'BLUR-USDT', 'PARTI-USDT', 'PORTAL-USDT', 'IP-USDT',
            'BIO-USDT', 'BERA-USDT', 'POL-USDT', 'KAS-USDT', 'ENS-USDT',
            'ONDO-USDT', 'HYPE-USDT', 'ARB-USDT', 'HBAR-USDT', 'DOT-USDT',
            'SEI-USDT', 'DYDX-USDT', 'LA-USDT', 'LDO-USDT', 'XPL-USDT',
            'WIF-USDT', 'ALGO-USDT', 'AVNT-USDT'
        ]
        
        new_count = 0
        
        for symbol in symbols:
            orders = self._fetch_orders(symbol)
            if not orders:
                continue
            
            positions = self._match_orders(orders, symbol)
            
            for position in positions:
                if self._is_new_position(position):
                    if self._save_position(position):
                        new_count += 1
                        print(f"   ✅ {position['symbol']:12} | ${position['pnl_usd']:+.2f} | Closed: {position.get('closed', '')[:19]}")
            
            time.sleep(0.5)  # Rate limit: 1 request per 500ms
        
        if new_count > 0:
            print(f"\n✨ Found and added {new_count} new position(s)")
        else:
            print(f"   No new positions found")
        
        return new_count
    
    def run(self):
        """Continuous real-time monitoring"""
        print("\n" + "="*80)
        print("🟢 ASTERDEX REAL-TIME TRACKER STARTED")
        print("="*80)
        print(f"API: Public endpoint (no auth required)")
        print(f"Update interval: {self.update_interval} seconds")
        print(f"Tracking: Jun 7+ closed positions only")
        print(f"Data file: {self.positions_file}")
        print("Press Ctrl+C to stop")
        print("="*80)
        
        try:
            iteration = 0
            while True:
                iteration += 1
                self.fetch_cycle()
                print(f"\n⏳ Next check in {self.update_interval}s... (iteration {iteration})")
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\n\n⏹️ Real-time tracker stopped")
            print(f"Total iterations: {iteration}")
            print(f"Tracked positions: {len(self.last_positions) // 2}")  # Div by 2 (entry + exit IDs)

def main():
    # Load .env file first
    env_file = Path.home() / '.openclaw/workspace/.env'
    if env_file.exists():
        print(f"📋 Loading credentials from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    # Handle both "KEY=VALUE" and "export KEY=VALUE"
                    if line.startswith('export '):
                        line = line[7:].strip()
                    key, val = line.split('=', 1)
                    val = val.strip('"').strip("'")
                    os.environ[key] = val
        print(f"✅ Loaded environment variables")
    else:
        print(f"❌ Credentials file not found: {env_file}")
        sys.exit(1)
    
    fetcher = AsterdexRealTimeFetcher()
    fetcher.run()

if __name__ == '__main__':
    main()
