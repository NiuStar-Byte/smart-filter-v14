#!/usr/bin/env python3
"""
ASTERDEX CLOSED POSITIONS TRACKER - CORRECTED VERSION
Fixes:
1. Extracts MARKET exit prices correctly from executedPrice
2. Applies leverage to P&L calculation: (exit - entry) * qty * leverage
3. Tracks leverage per symbol from LEVERAGE_CONFIG
4. Deduplicates to prevent order double-counting
"""
import json
import requests
import time
from datetime import datetime, timezone
from pathlib import Path
from aster_v3_auth import AsterV3Auth
from asterdex_config import ASTER_MAIN_ACCOUNT, ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY

# Leverage configuration per symbol (from MEMORY.md)
LEVERAGE_CONFIG = {
    'KNC-USDT': 5, 'BERA-USDT': 5, 'ALGO-USDT': 5, 'AVNT-USDT': 5, 
    'BLUR-USDT': 5, 'PARTI-USDT': 5, 'PORTAL-USDT': 5, 'LA-USDT': 5, 
    'DYDX-USDT': 5, 'FUN-USDT': 3
}

class AsterdexClosedTrackerCorrected:
    def __init__(self):
        self.auth = AsterV3Auth(ASTER_API_WALLET_ADDRESS, ASTER_API_WALLET_PRIVATE_KEY)
        self.api_base = "https://fapi.asterdex.com"
        self.main_account = ASTER_MAIN_ACCOUNT
        self.output_file = Path('ASTERDEX_CLOSED_POSITIONS_LIVE.jsonl')
        self.cutoff = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)
        self.tracked_ids = set()
        
        self._load_tracked()
        
        print(f"✅ Corrected closed tracker initialized")
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
    
    def _get_leverage(self, symbol):
        """Get leverage for symbol (default 10x, with overrides)"""
        symbol_with_usdt = symbol if symbol.endswith('-USDT') else f"{symbol}-USDT"
        return LEVERAGE_CONFIG.get(symbol_with_usdt, 10)
    
    def _match_closed_positions(self, all_orders):
        """
        Match LIMIT entries with MARKET exits
        Improved: Uses executedPrice from MARKET order, applies leverage
        """
        positions = []
        
        # Filter to FILLED orders from Jun 7+
        filled = [o for o in all_orders if o.get('status') == 'FILLED']
        filled_jun7 = [o for o in filled if datetime.fromtimestamp(o.get('time', 0) / 1000, tz=timezone.utc) >= self.cutoff]
        
        # Separate by type
        limits = [o for o in filled_jun7 if o.get('type') == 'LIMIT']
        markets = [o for o in filled_jun7 if o.get('type') == 'MARKET']
        
        # Create lookup for quick MARKET access (keyed by orderId)
        market_by_id = {m.get('orderId'): m for m in markets}
        market_by_symbol_time = {}
        for m in markets:
            symbol = m.get('symbol')
            time_ms = m.get('time', 0)
            if symbol not in market_by_symbol_time:
                market_by_symbol_time[symbol] = []
            market_by_symbol_time[symbol].append((time_ms, m))
        
        # Sort by time for each symbol
        for symbol in market_by_symbol_time:
            market_by_symbol_time[symbol].sort(key=lambda x: x[0])
        
        # Track which MARKET orders we've used (to prevent double-matching)
        used_market_ids = set()
        
        # For each LIMIT, find FIRST matching MARKET (opposite side, same symbol, after entry)
        for limit in limits:
            limit_id = limit.get('orderId')
            limit_time = limit.get('time', 0)
            limit_symbol = limit.get('symbol')
            limit_side = limit.get('side')
            opposite_side = 'SELL' if limit_side == 'BUY' else 'BUY'
            
            # Find first MARKET after this LIMIT with opposite side, same symbol
            if limit_symbol in market_by_symbol_time:
                for market_time, market in market_by_symbol_time[limit_symbol]:
                    market_id = market.get('orderId')
                    market_side = market.get('side')
                    
                    # Check: MARKET is after LIMIT, opposite side, not yet used
                    if (market_time > limit_time and 
                        market_side == opposite_side and
                        market_id not in used_market_ids):
                        
                        # Found matching exit!
                        pos_id = f"{limit_symbol}_{limit_id}_{market_id}"
                        
                        if pos_id not in self.tracked_ids:
                            pos = self._create_position(limit, market, pos_id)
                            if pos:
                                positions.append(pos)
                                self.tracked_ids.add(pos_id)
                                used_market_ids.add(market_id)
                        
                        break  # Found match for this LIMIT, move to next
        
        return positions
    
    def _create_position(self, entry_order, exit_order, pos_id):
        """Create closed position record with leverage-aware P&L"""
        try:
            entry_time = datetime.fromtimestamp(entry_order.get('time', 0) / 1000, tz=timezone.utc)
            exit_time = datetime.fromtimestamp(exit_order.get('time', 0) / 1000, tz=timezone.utc)
            
            # Use executedPrice for actual fill price (not 'price' which may be 0 for MARKET orders)
            entry_price = float(entry_order.get('executedPrice') or entry_order.get('price', 0))
            exit_price = float(exit_order.get('executedPrice') or exit_order.get('price', 0))
            quantity = float(entry_order.get('executedQty', 0))
            symbol = entry_order.get('symbol')
            
            if entry_price == 0 or exit_price == 0 or quantity == 0:
                print(f"   ⚠️  Skipped {symbol}: entry={entry_price}, exit={exit_price}, qty={quantity}")
                return None
            
            # Get leverage for this symbol
            leverage = self._get_leverage(symbol)
            
            # Calculate P&L with leverage applied
            # P&L in USD = (price_change) * quantity * leverage
            if entry_order.get('side') == 'BUY':
                price_change = (exit_price - entry_price)
                pnl = price_change * quantity * leverage
                pnl_pct = (price_change / entry_price) * 100
                side = 'LONG'
            else:
                price_change = (entry_price - exit_price)
                pnl = price_change * quantity * leverage
                pnl_pct = (price_change / entry_price) * 100
                side = 'SHORT'
            
            duration = (exit_time - entry_time).total_seconds() / 3600  # hours
            
            return {
                'position_id': pos_id,
                'symbol': symbol,
                'side': side,
                'entry_price': round(entry_price, 8),
                'exit_price': round(exit_price, 8),
                'quantity': round(quantity, 8),
                'leverage': leverage,
                'entry_order_id': entry_order.get('orderId'),
                'exit_order_id': exit_order.get('orderId'),
                'opened': entry_time.isoformat() + 'Z',
                'closed': exit_time.isoformat() + 'Z',
                'duration_hours': round(duration, 2),
                'pnl_usd': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'margin_used_usd': 2.0,  # Standard margin per entry
            }
        except Exception as e:
            print(f"   ❌ Error creating position: {e}")
            return None
    
    def run(self, interval_sec=60):
        """Main loop"""
        print(f"\n🚀 Starting corrected closed positions tracker (interval={interval_sec}s)")
        print(f"Target: 63 closed positions\n")
        
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
                    avg_pnl = total_pnl / len(new_positions) if new_positions else 0
                    
                    print(f"       {wins}W {losses}L | Total P&L: ${total_pnl:+.2f} | Avg: ${avg_pnl:+.2f}")
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
    tracker = AsterdexClosedTrackerCorrected()
    tracker.run()
