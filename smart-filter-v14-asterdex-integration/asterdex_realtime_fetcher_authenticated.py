#!/usr/bin/env python3
"""
ASTERDEX REAL-TIME TRACKER - AUTHENTICATED VERSION
Fetch closing positions using the SAME authentication as asterdex_entry_poster.py
Uses AsterV3Auth (EIP-712 signing) to authenticate /fapi/v3/allOrders calls
"""
import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict
import logging

# Import the SAME auth class that entry_poster uses
from aster_v3_auth import AsterV3Auth
from asterdex_config import (
    ASTER_MAIN_ACCOUNT,
    ASTER_API_WALLET_ADDRESS,
    ASTER_API_WALLET_PRIVATE_KEY,
    ASTERDEX_BASE_URL,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S GMT+7'
)
logger = logging.getLogger(__name__)


class AsterdexRealTimeTrackerAuth:
    """Fetch positions from Asterdex using authenticated API calls"""
    
    def __init__(self):
        """Initialize with same credentials and auth as entry_poster"""
        self.main_account = ASTER_MAIN_ACCOUNT
        self.api_wallet = ASTER_API_WALLET_ADDRESS
        self.private_key = ASTER_API_WALLET_PRIVATE_KEY
        
        # Use the SAME AsterV3Auth that entry_poster uses
        self.auth = AsterV3Auth(self.api_wallet, self.private_key)
        
        self.api_base = ASTERDEX_BASE_URL
        self.positions_file = Path('ASTERDEX_POSITIONS_LIVE.jsonl')
        self.cutoff_date = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)
        
        # Track processed order IDs to avoid duplicates
        self.processed_order_ids = set()
        self._load_processed_positions()
        
        logger.info(f"✅ Authenticated fetcher initialized")
        logger.info(f"   Main account: {self.main_account}")
        logger.info(f"   API wallet (signer): {self.api_wallet}")
    
    def _load_processed_positions(self):
        """Load existing order IDs to avoid re-processing"""
        if self.positions_file.exists():
            try:
                with open(self.positions_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            pos = json.loads(line)
                            self.processed_order_ids.add(pos.get('entry_order_id'))
                            self.processed_order_ids.add(pos.get('exit_order_id'))
                logger.info(f"✅ Loaded {len(self.processed_order_ids)} existing order IDs")
            except Exception as e:
                logger.error(f"Error loading positions: {e}")
    
    def _fetch_all_orders(self, symbol: str) -> List[Dict]:
        """
        Fetch all orders for a symbol using AUTHENTICATED /fapi/v3/allOrders
        Uses same AsterV3Auth signing as entry_poster
        """
        try:
            url = f"{self.api_base}/fapi/v3/allOrders"
            
            # Build base parameters
            params = {
                'symbol': symbol,
                'limit': '1000',
            }
            
            # Sign with AsterV3Auth (same as entry_poster)
            signed_params = self.auth.sign_request_v3(params, main_account=self.main_account)
            
            # Make request with signed parameters (all in query string for GET)
            response = requests.get(url, params=signed_params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    logger.debug(f"✅ {symbol}: Got {len(data)} orders")
                    return data
                else:
                    logger.warning(f"⚠️ {symbol}: Unexpected response type: {type(data)}")
                    return []
            else:
                logger.warning(f"⚠️ {symbol}: API returned {response.status_code}")
                if response.status_code != 400:
                    logger.debug(f"   Response: {response.text[:200]}")
                return []
        
        except Exception as e:
            logger.error(f"❌ {symbol}: {e}")
            return []
    
    def _match_orders_to_positions(self, orders: List[Dict], symbol: str) -> List[Dict]:
        """Match entry and exit orders into complete positions"""
        positions = []
        
        # Filter to only FILLED orders
        filled_orders = [o for o in orders if o.get('status') == 'FILLED']
        
        # Separate by side
        buys = [o for o in filled_orders if o.get('side') == 'BUY']
        sells = [o for o in filled_orders if o.get('side') == 'SELL']
        
        # Match LONG positions: BUY → SELL
        for buy in buys:
            buy_time = buy.get('updateTime', 0) / 1000
            buy_order_id = buy.get('orderId')
            
            # Find first SELL after this BUY
            for sell in sells:
                sell_time = sell.get('updateTime', 0) / 1000
                sell_order_id = sell.get('orderId')
                
                if sell_time > buy_time and buy_order_id != sell_order_id:
                    pos = self._create_position('LONG', buy, sell, symbol)
                    if pos:
                        positions.append(pos)
                    break  # Take first sell after buy
        
        # Match SHORT positions: SELL → BUY
        for sell in sells:
            sell_time = sell.get('updateTime', 0) / 1000
            sell_order_id = sell.get('orderId')
            
            # Find first BUY after this SELL
            for buy in buys:
                buy_time = buy.get('updateTime', 0) / 1000
                buy_order_id = buy.get('orderId')
                
                if buy_time > sell_time and sell_order_id != buy_order_id:
                    pos = self._create_position('SHORT', sell, buy, symbol)
                    if pos:
                        positions.append(pos)
                    break  # Take first buy after sell
        
        return positions
    
    def _create_position(self, side: str, entry_order: Dict, exit_order: Dict, symbol: str) -> Dict:
        """Create position record from entry and exit orders"""
        try:
            entry_time = datetime.fromtimestamp(entry_order.get('time', 0) / 1000, tz=timezone.utc)
            exit_time = datetime.fromtimestamp(exit_order.get('updateTime', 0) / 1000, tz=timezone.utc)
            
            # Only include positions opened on or after Jun 7
            if entry_time < self.cutoff_date:
                return None
            
            entry_price = float(entry_order.get('price', 0))
            exit_price = float(exit_order.get('price', 0))
            quantity = float(entry_order.get('executedQty', 0))
            
            if entry_price == 0 or quantity == 0:
                return None
            
            # Calculate P&L
            if side == 'LONG':
                pnl_usd = (exit_price - entry_price) * quantity
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_usd = (entry_price - exit_price) * quantity
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            
            return {
                'position_id': f"{symbol}_{entry_order.get('orderId')}_{exit_order.get('orderId')}",
                'symbol': symbol,
                'side': side,
                'entry_price': round(entry_price, 8),
                'exit_price': round(exit_price, 8),
                'quantity': round(quantity, 8),
                'entry_order_id': entry_order.get('orderId'),
                'exit_order_id': exit_order.get('orderId'),
                'opened': entry_time.isoformat() + 'Z',
                'closed': exit_time.isoformat() + 'Z',
                'pnl_usd': round(pnl_usd, 2),
                'pnl_pct': round(pnl_pct, 2),
                'leverage': '10x'
            }
        except Exception as e:
            logger.debug(f"Error creating position: {e}")
            return None
    
    def run(self, symbols: List[str] = None, interval_sec: int = 60):
        """Main loop: poll API and update positions file"""
        
        # Default symbols (all that entry_poster posts to)
        if not symbols:
            symbols = [
                "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
                "DOGE-USDT", "ADA-USDT", "AAVE-USDT", "NEAR-USDT", "AVAX-USDT",
                "KAS-USDT", "DOT-USDT", "ATOM-USDT", "BIO-USDT", "PORTAL-USDT",
                "SEI-USDT", "INJ-USDT", "PUMP-USDT", "WIF-USDT", "ONDO-USDT",
                "SOL-USDT", "XRP-USDT", "BLUR-USDT", "DYDX-USDT", "LA-USDT",
                "PARTI-USDT", "PORTAL-USDT", "ALGO-USDT", "AVNT-USDT",
            ]
        
        logger.info(f"🚀 Starting real-time tracker (interval={interval_sec}s)")
        logger.info(f"   Monitoring {len(symbols)} symbols")
        logger.info(f"   Cutoff: positions opened >= 2026-06-07 00:00 GMT+7")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S GMT+7')
                logger.info(f"\n🔄 Iteration {iteration} ({now})")
                
                total_new = 0
                
                for symbol in symbols:
                    # Fetch all orders
                    orders = self._fetch_all_orders(symbol)
                    
                    if not orders:
                        continue
                    
                    # Match into positions
                    positions = self._match_orders_to_positions(orders, symbol)
                    
                    # Write new positions to file
                    new_count = 0
                    if positions:
                        with open(self.positions_file, 'a') as f:
                            for pos in positions:
                                entry_id = pos.get('entry_order_id')
                                exit_id = pos.get('exit_order_id')
                                
                                if entry_id not in self.processed_order_ids:
                                    f.write(json.dumps(pos) + '\n')
                                    self.processed_order_ids.add(entry_id)
                                    self.processed_order_ids.add(exit_id)
                                    new_count += 1
                    
                    if new_count > 0:
                        logger.info(f"   ✅ {symbol}: +{new_count} positions")
                        total_new += new_count
                
                if total_new == 0:
                    logger.info(f"   ℹ️  No new positions (checking again in {interval_sec}s)")
                else:
                    logger.info(f"   ✅ Total: +{total_new} new positions")
                
                time.sleep(interval_sec)
            
            except KeyboardInterrupt:
                logger.info("\n⏹️  Stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)


if __name__ == "__main__":
    tracker = AsterdexRealTimeTrackerAuth()
    tracker.run()
