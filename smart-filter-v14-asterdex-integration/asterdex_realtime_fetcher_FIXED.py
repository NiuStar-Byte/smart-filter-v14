#!/usr/bin/env python3
"""
ASTERDEX REAL-TIME TRACKER - FIXED VERSION
Fetch closing positions from Asterdex API with proper authentication
Uses EIP-712 signing to authenticate USER_DATA requests
"""
import json
import os
import time
import math
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
from eth_account import Account
from eth_account.messages import encode_typed_data

class AsterdexRealTimeFetcherFixed:
    def __init__(self):
        self.api_base = "https://fapi.asterdex.com"
        
        # Load credentials from environment
        self.main_account = os.getenv("ASTER_MAIN_ACCOUNT")  # user address
        self.signer = os.getenv("ASTER_API_WALLET_ADDRESS")  # API wallet
        self.private_key = os.getenv("ASTER_API_WALLET_PRIVATE_KEY")
        
        if not all([self.main_account, self.signer, self.private_key]):
            raise ValueError("Missing ASTER credentials in environment!")
        
        self.positions_file = Path('ASTERDEX_POSITIONS_LIVE.jsonl')
        self.cutoff = datetime(2026, 6, 7, 0, 0, 0, tzinfo=timezone.utc)
        self.update_interval = 60
        self.last_positions = set()
        
        # Nonce tracking
        self._last_ms = 0
        self._i = 0
        
        # EIP-712 typed data structure
        self.typed_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"}
                ],
                "Message": [
                    {"name": "msg", "type": "string"}
                ]
            },
            "primaryType": "Message",
            "domain": {
                "name": "AsterSignTransaction",
                "version": "1",
                "chainId": 1666,
                "verifyingContract": "0x0000000000000000000000000000000000000000"
            },
            "message": {
                "msg": ""  # Will be filled with encoded params
            }
        }
        
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
    
    def _get_nonce(self):
        """Generate unique monotonic nonce in microseconds"""
        now_ms = int(time.time())
        
        if now_ms == self._last_ms:
            self._i += 1
        else:
            self._last_ms = now_ms
            self._i = 0
        
        return now_ms * 1_000_000 + self._i
    
    def _sign_request(self, params):
        """Sign request using EIP-712"""
        try:
            # Build param string with URL encoding
            param_string = urllib.parse.urlencode(params)
            
            # Create typed data with param string
            typed_data = self.typed_data.copy()
            typed_data["message"]["msg"] = param_string
            
            # Sign the message
            message = encode_typed_data(typed_data)
            signed = Account.sign_message(message, private_key=self.private_key)
            
            return signed.signature.hex()
        except Exception as e:
            print(f"❌ Signing error: {e}")
            return None
    
    def _fetch_orders(self, symbol):
        """Fetch all filled orders for a symbol (USER_DATA endpoint with auth)"""
        try:
            url = f"{self.api_base}/fapi/v3/allOrders"
            
            # Build parameters
            nonce = str(self._get_nonce())
            params = {
                'symbol': symbol,
                'limit': '1000',
                'recvWindow': '5000',
                'nonce': nonce,
                'user': self.main_account,
                'signer': self.signer,
            }
            
            # Sign the request
            signature = self._sign_request(params)
            if not signature:
                print(f"⚠️ Failed to sign for {symbol}")
                return []
            
            params['signature'] = signature
            
            # Build URL with query string
            query_string = urllib.parse.urlencode(params)
            full_url = f"{url}?{query_string}"
            
            response = requests.get(full_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
                else:
                    return []
            else:
                print(f"⚠️ API error {response.status_code} for {symbol}: {response.text[:100]}")
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
        
        # Match LONG: BUY entry → SELL exit
        for buy in buys:
            for sell in sells:
                if sell.get('updateTime', 0) > buy.get('updateTime', 0):
                    pos = self._create_position('LONG', buy, sell, symbol)
                    if pos:
                        positions.append(pos)
                    break  # Take first exit after entry
        
        # Match SHORT: SELL entry → BUY cover
        for short in sells:
            for cover in buys:
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
            if entry_time < self.cutoff:
                return None
            
            entry_price = float(entry_order.get('price', 0))
            exit_price = float(exit_order.get('price', 0))
            quantity = float(entry_order.get('executedQty', 0))
            
            if entry_price == 0 or quantity == 0:
                return None
            
            # Calculate P&L
            if side == 'LONG':
                pnl_usd = (exit_price - entry_price) * quantity
            else:  # SHORT
                pnl_usd = (entry_price - exit_price) * quantity
            
            pnl_pct = ((exit_price - entry_price) / entry_price * 100) if side == 'LONG' else ((entry_price - exit_price) / entry_price * 100)
            
            return {
                'position_id': f"{symbol}_{entry_order.get('orderId')}_{exit_order.get('orderId')}",
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'entry_order_id': entry_order.get('orderId'),
                'exit_order_id': exit_order.get('orderId'),
                'opened': entry_time.isoformat() + 'Z',
                'closed': exit_time.isoformat() + 'Z',
                'pnl_usd': round(pnl_usd, 2),
                'pnl_pct': round(pnl_pct, 2),
                'leverage': '10x'
            }
        except Exception as e:
            print(f"❌ Error creating position: {e}")
            return None
    
    def run(self):
        """Main loop: fetch orders and update positions file"""
        print(f"[INFO] Starting real-time fetcher for account: {self.main_account[:10]}...")
        print(f"[INFO] API Wallet (signer): {self.signer[:10]}...")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\n🔄 Fetching from Asterdex API... (iteration {iteration})")
                
                # Get list of all symbols
                symbols = [
                    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
                    "DOGE-USDT", "ADA-USDT", "AAVE-USDT", "NEAR-USDT", "AVAX-USDT",
                    "KAS-USDT", "DOT-USDT", "ATOM-USDT", "BIO-USDT", "PORTAL-USDT",
                    # Add more symbols as needed
                ]
                
                new_positions = []
                for symbol in symbols:
                    orders = self._fetch_orders(symbol)
                    if orders:
                        positions = self._match_orders(orders, symbol)
                        new_positions.extend(positions)
                
                # Write new positions to file
                new_count = 0
                if new_positions:
                    with open(self.positions_file, 'a') as f:
                        for pos in new_positions:
                            order_id = pos.get('entry_order_id')
                            if order_id not in self.last_positions:
                                f.write(json.dumps(pos) + '\n')
                                self.last_positions.add(order_id)
                                self.last_positions.add(pos.get('exit_order_id'))
                                new_count += 1
                
                if new_count > 0:
                    print(f"✅ Added {new_count} new positions")
                else:
                    print(f"   No new positions found")
                
                print(f"⏳ Next check in {self.update_interval}s...")
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                print("\n[INFO] Stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                time.sleep(10)

if __name__ == "__main__":
    fetcher = AsterdexRealTimeFetcherFixed()
    fetcher.run()
