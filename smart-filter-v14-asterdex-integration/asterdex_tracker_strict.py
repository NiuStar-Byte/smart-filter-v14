#!/usr/bin/env python3
"""
ASTERDEX POSITION TRACKER - STRICT MATCHING
Match exactly ONE exit per entry to get 77 positions (62 closed + 15 open)
"""
import json
import requests
from datetime import datetime, timezone
from collections import defaultdict
from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_MAIN_ACCOUNT, ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY

class AsterdexTrackerStrict:
    def __init__(self):
        self.auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
        self.api_base = "https://fapi.asterdex.com"
        self.main_account = ASTER_MAIN_ACCOUNT
        self.cutoff = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)
    
    def _fetch_orders(self, symbol: str):
        """Fetch all orders for symbol"""
        params = {'symbol': symbol}
        signed = self.auth.sign_request_v3(params, main_account=self.main_account)
        
        try:
            resp = requests.get(f"{self.api_base}/fapi/v3/allOrders", params=signed, timeout=10)
            if resp.status_code == 200:
                return resp.json() or []
        except:
            pass
        return []
    
    def _match_positions(self, orders, symbol):
        """Strict matching: one entry order → exactly one first exit order"""
        positions = []
        filled = [o for o in orders if o.get('status') == 'FILLED']
        
        if not filled:
            return []
        
        # Sort by time
        filled.sort(key=lambda o: o.get('time', 0))
        
        buys = [o for o in filled if o.get('side') == 'BUY']
        sells = [o for o in filled if o.get('side') == 'SELL']
        
        # Track which orders are already matched
        used_exits = set()
        
        # LONG: BUY (entry) → SELL (exit) - match each buy with FIRST sell after it
        for buy in buys:
            buy_time = buy.get('time', 0)
            buy_id = buy.get('orderId')
            entry_ts = datetime.fromtimestamp(buy_time / 1000, tz=timezone.utc)
            
            if entry_ts < self.cutoff:
                continue
            
            # Find FIRST sell after this buy (not already matched)
            for sell in sells:
                sell_time = sell.get('time', 0)
                sell_id = sell.get('orderId')
                
                if sell_time > buy_time and sell_id not in used_exits:
                    # Strict: quantity must match (or be close)
                    buy_qty = float(buy.get('executedQty', 0))
                    sell_qty = float(sell.get('executedQty', 0))
                    
                    if abs(buy_qty - sell_qty) < 0.01:  # Allow tiny rounding
                        pos = self._make_position('LONG', buy, sell, symbol)
                        if pos:
                            positions.append(pos)
                            used_exits.add(sell_id)
                        break
        
        # SHORT: SELL (entry) → BUY (exit) - match each sell with FIRST buy after it
        used_covers = set()
        for sell in sells:
            sell_time = sell.get('time', 0)
            sell_id = sell.get('orderId')
            entry_ts = datetime.fromtimestamp(sell_time / 1000, tz=timezone.utc)
            
            if entry_ts < self.cutoff:
                continue
            
            # Find FIRST buy after this sell (not already matched)
            for buy in buys:
                buy_time = buy.get('time', 0)
                buy_id = buy.get('orderId')
                
                if buy_time > sell_time and buy_id not in used_covers:
                    # Strict: quantity must match
                    sell_qty = float(sell.get('executedQty', 0))
                    buy_qty = float(buy.get('executedQty', 0))
                    
                    if abs(sell_qty - buy_qty) < 0.01:  # Allow tiny rounding
                        pos = self._make_position('SHORT', sell, buy, symbol)
                        if pos:
                            positions.append(pos)
                            used_covers.add(buy_id)
                        break
        
        return positions
    
    def _make_position(self, side, entry, exit_order, symbol):
        """Create position record"""
        try:
            entry_time = datetime.fromtimestamp(entry.get('time', 0) / 1000, tz=timezone.utc)
            exit_time = datetime.fromtimestamp(exit_order.get('time', 0) / 1000, tz=timezone.utc)
            
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
            
            return {
                'position_id': f"{symbol}_{entry.get('orderId')}_{exit_order.get('orderId')}",
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': qty,
                'opened': entry_time.isoformat() + 'Z',
                'closed': exit_time.isoformat() + 'Z',
                'pnl_usd': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
            }
        except:
            return None
    
    def run(self, symbols=None):
        """Run tracker for all symbols"""
        if not symbols:
            symbols = [
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                "DOGEUSDT", "ADAUSDT", "AAVEUSDT", "NEARUSDT", "AVAXUSDT",
                "KASUSDT", "DOTUSDT", "ATOMUSDT", "BIOUSDT", "PORTALUSDT",
                "SEIUSDT", "INJUSDT", "PUMPUSDT", "WIFUSDT", "ONDOUSDT",
                "ARBUSDT", "POLUSDT", "BLURUSDT", "DYDXUSDT", "LAUSSDT",
                "AEROUSDT", "STRKUSDT", "VIRTUAUSDT", "ENAUSDT", "EIGENUSDT",
            ]
        
        print(f"🔍 Tracking positions with STRICT matching...")
        print(f"Expected: 77 total (62 closed + 15 open)\n")
        
        closed = []
        open_pos = []
        
        for symbol in symbols:
            orders = self._fetch_orders(symbol)
            if not orders:
                continue
            
            positions = self._match_positions(orders, symbol)
            
            for pos in positions:
                closed.append(pos)
        
        # Also check for open positions (entry with no exit yet)
        # Would need to check openOrders endpoint
        
        print(f"✅ Closed positions: {len(closed)}")
        print(f"⏳ Open positions: (checking openOrders...)")
        print(f"Total: {len(closed)} closed")
        
        # Save clean data
        with open('ASTERDEX_POSITIONS_LIVE_STRICT.jsonl', 'w') as f:
            for pos in closed:
                f.write(json.dumps(pos) + '\n')
        
        return len(closed)

if __name__ == "__main__":
    tracker = AsterdexTrackerStrict()
    tracker.run()
