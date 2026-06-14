#!/usr/bin/env python3
"""
ASTERDEX CLOSURE DETECTOR - Real-time position closure tracker
Fetches closed positions from /fapi/v3/trades endpoint
Updates ASTERDEX_POSITIONS_LIVE.jsonl with new closures
"""
import json
import requests
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

API_BASE = "https://fapi.asterdex.com"
LIVE_FILE = Path('ASTERDEX_POSITIONS_LIVE.jsonl')
CUTOFF = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)

# All symbols we track (without dashes for API)
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "POLKADOTUSDT", "AAVEUSDT", "NEARUSDT", "AVAXUSDT", "ATOMUSDT",
    "INJUSDT", "LAZIOUSDT", "LDOUSDT", "ONDOUSDT", "XPLUSDT", "DYDXUSDT",
    "TURBOUSDT", "PUMPUSDT", "ENSUSDT", "INPUSDT", "KASUSDT", "SEIUSDT",
    "WIFUSDT", "DOTUSDT", "HBARLUSDT", "BLURSDT", "BERIUSDT", "PORTALUSDT",
    "PARTIUSDT", "FUNUSDT", "HYPEUSDT", "ARBUSDT", "BIOUSDT", "POLUSDT"
]

class ClosureDetector:
    def __init__(self):
        self.existing_positions = {}
        self.last_trade_times = {}
        self.load_existing()
    
    def load_existing(self):
        """Load positions already in file"""
        if LIVE_FILE.exists():
            with open(LIVE_FILE, 'r') as f:
                for line in f:
                    if line.strip():
                        pos = json.loads(line)
                        pos_id = pos.get('position_id')
                        if pos_id:
                            self.existing_positions[pos_id] = pos
    
    def fetch_trades(self, symbol):
        """Fetch recent trades for symbol"""
        try:
            response = requests.get(
                f"{API_BASE}/fapi/v3/trades",
                params={'symbol': symbol, 'limit': 100},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []
    
    def match_trades_into_positions(self, symbol, trades):
        """Match BUY+SELL trades into closed positions"""
        positions = []
        
        # Separate BUY and SELL trades
        buys = [t for t in trades if not t.get('isBuyerMaker')]  # BUY orders
        sells = [t for t in trades if t.get('isBuyerMaker')]      # SELL orders
        
        # Match each SELL (exit) with a BUY (entry) before it
        for sell in sells:
            sell_time = sell.get('time', 0)
            
            # Find matching BUY that happened before this SELL
            for buy in buys:
                buy_time = buy.get('time', 0)
                if buy_time < sell_time:
                    # Found a pair - create position record
                    entry_price = float(buy.get('price', 0))
                    exit_price = float(sell.get('price', 0))
                    quantity = float(buy.get('qty', 0))
                    
                    pnl_usd = (exit_price - entry_price) * quantity
                    pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    
                    position = {
                        "position_id": f"{symbol}_{buy.get('id')}_{sell.get('id')}",
                        "symbol": symbol.replace('USDT', '-USDT'),  # Convert back to dash format
                        "side": "LONG",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "quantity": quantity,
                        "entry_order_id": buy.get('id'),
                        "exit_order_id": sell.get('id'),
                        "opened": datetime.fromtimestamp(buy_time / 1000, tz=timezone.utc).isoformat().replace('+00:00', 'Z'),
                        "closed": datetime.fromtimestamp(sell_time / 1000, tz=timezone.utc).isoformat().replace('+00:00', 'Z'),
                        "pnl_usd": round(pnl_usd, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "leverage": "10x"
                    }
                    
                    # Check if opened on Jun 7+
                    try:
                        opened_dt = datetime.fromisoformat(position['opened'].replace('Z', '+00:00'))
                        if opened_dt >= CUTOFF:
                            positions.append(position)
                    except:
                        pass
                    
                    break  # Move to next SELL
        
        return positions
    
    def add_position(self, position):
        """Add position to LIVE file if new"""
        pos_id = position.get('position_id')
        if pos_id and pos_id not in self.existing_positions:
            try:
                with open(LIVE_FILE, 'a') as f:
                    f.write(json.dumps(position) + '\n')
                self.existing_positions[pos_id] = position
                return True
            except:
                pass
        return False
    
    def run_once(self):
        """Fetch and process new closures"""
        total_added = 0
        
        for symbol in SYMBOLS:
            trades = self.fetch_trades(symbol)
            if not trades:
                continue
            
            positions = self.match_trades_into_positions(symbol, trades)
            for pos in positions:
                if self.add_position(pos):
                    total_added += 1
                    print(f"  ✅ {pos['symbol']} {pos['side']} P&L ${pos['pnl_usd']:+.2f}")
        
        return total_added
    
    def run_continuous(self, interval=60):
        """Run continuously"""
        print("\n" + "="*80)
        print("🟢 ASTERDEX CLOSURE DETECTOR STARTED")
        print("="*80)
        print(f"Fetching closed positions from /fapi/v3/trades")
        print(f"Tracking Jun 7+ closures only")
        print(f"Update interval: {interval} seconds")
        print("="*80)
        
        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n🔄 Cycle {iteration}: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
                
                added = self.run_once()
                
                if added > 0:
                    print(f"✅ Added {added} new closed positions")
                else:
                    print(f"ℹ️ No new closures found")
                
                print(f"⏳ Next check in {interval}s...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n⏹️ Closure detector stopped")

if __name__ == '__main__':
    detector = ClosureDetector()
    detector.run_continuous(interval=60)
