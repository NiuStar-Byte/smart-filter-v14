#!/usr/bin/env python3
"""
Aster Futures API v3 Client with EIP-712 Signing
Handles authentication and API requests to Aster.
"""

import os
import time
import json
import hashlib
import requests
from typing import Dict, Optional, Any
from eth_account import Account
from eth_account.messages import encode_defunct
import logging

# Try to import encode_structured_data (newer versions)
try:
    from eth_account.messages import encode_structured_data
except ImportError:
    # Fallback for older versions
    encode_structured_data = None

logger = logging.getLogger(__name__)

class AsterClient:
    def __init__(self, api_key: str = None, private_key: str = None, wallet_address: str = None, api_base_url: str = None):
        """
        Initialize Aster API client (Futures or Spot).
        
        Args:
            api_key: API key (from env: ASTER_API_KEY)
            private_key: Private key (from env: ASTER_PRIVATE_KEY)
            wallet_address: Wallet address (from env: ASTER_WALLET_ADDRESS)
            api_base_url: Override base URL (for Spot API support)
        """
        self.api_key = api_key or os.getenv('ASTER_API_KEY', '')
        self.private_key = private_key or os.getenv('ASTER_PRIVATE_KEY')
        self.wallet_address = wallet_address or os.getenv('ASTER_WALLET_ADDRESS')
        
        # Default to Futures, but allow override for Spot
        self.base_url = api_base_url or "https://fapi.asterdex.com"
        self.is_spot = "sapi" in self.base_url.lower()  # Detect if Spot API based on URL
        
        # Initialize eth_account
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            api_type = "Spot" if self.is_spot else "Futures"
            logger.info(f"✅ Aster {api_type} client initialized: {self.account.address}")
        else:
            self.account = None
            logger.warning("⚠️  No private key provided. Read-only mode only.")
    
    def _sign_request(self, params: Dict) -> str:
        """
        Sign request with EIP-712.
        
        Signing flow:
        1. Sort params alphabetically by key
        2. Create param string: key1=value1&key2=value2&...
        3. Sign with EIP-712 domain (chainId 1666) OR simple message signing
        4. Return hex signature
        """
        if not self.account:
            raise ValueError("Private key not set. Cannot sign requests.")
        
        # Add nonce and user/signer
        nonce = int(time.time() * 1_000_000)  # Current time in microseconds
        params['nonce'] = str(nonce)
        params['user'] = self.wallet_address
        params['signer'] = self.account.address
        
        # Create param string (alphabetically sorted)
        sorted_keys = sorted(params.keys())
        param_string = "&".join(f"{k}={params[k]}" for k in sorted_keys)
        
        # Try EIP-712 if available, otherwise use simple message signing
        if encode_structured_data:
            try:
                domain_data = {
                    "name": "AsterSignTransaction",
                    "version": "1",
                    "chainId": 1666,
                    "verifyingContract": "0x0000000000000000000000000000000000000000"
                }
                
                message_types = {
                    "Message": [
                        {"name": "msg", "type": "string"}
                    ]
                }
                
                message_data = {
                    "msg": param_string
                }
                
                structured_msg = encode_structured_data({
                    "types": message_types,
                    "primaryType": "Message",
                    "domain": domain_data,
                    "message": message_data
                })
                
                signed = self.account.sign_message(structured_msg)
                signature = signed.signature.hex()
            except Exception as e:
                logger.warning(f"EIP-712 signing failed: {e}. Falling back to simple signing.")
                msg = encode_defunct(text=param_string)
                signed = self.account.sign_message(msg)
                signature = signed.signature.hex()
        else:
            # Simple message signing (fallback)
            msg = encode_defunct(text=param_string)
            signed = self.account.sign_message(msg)
            signature = signed.signature.hex()
        
        logger.debug(f"Signed with nonce {nonce}, signature: {signature[:20]}...")
        
        return signature, nonce, self.wallet_address, self.account.address
    
    # ===== PUBLIC (Market Data) =====
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Fetch current price (no auth needed)."""
        try:
            r = requests.get(
                f"{self.base_url}/fapi/v3/ticker/price",
                params={"symbol": symbol},
                timeout=5
            )
            if r.status_code == 200:
                return float(r.json()['price'])
            else:
                logger.error(f"Price fetch failed: {r.status_code} {r.text}")
                return None
        except Exception as e:
            logger.error(f"Price fetch error: {e}")
            return None
    
    def get_klines(self, symbol: str, interval: str = "5m", limit: int = 100) -> Optional[list]:
        """
        Fetch klines (candlesticks).
        
        Returns: List of [time, open, high, low, close, volume, ...]
        """
        try:
            r = requests.get(
                f"{self.base_url}/fapi/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit
                },
                timeout=5
            )
            if r.status_code == 200:
                return r.json()
            else:
                logger.error(f"Klines fetch failed: {r.status_code} {r.text}")
                return None
        except Exception as e:
            logger.error(f"Klines fetch error: {e}")
            return None
    
    def get_server_time(self) -> Optional[int]:
        """Fetch server time (for clock sync)."""
        try:
            r = requests.get(f"{self.base_url}/fapi/v3/time", timeout=5)
            if r.status_code == 200:
                return r.json()['serverTime']
            return None
        except Exception as e:
            logger.error(f"Server time fetch error: {e}")
            return None
    
    # ===== SIGNED (Account & Orders) =====
    
    def get_balance(self) -> Optional[Dict]:
        """Get account balance (USDT)."""
        try:
            params = {}
            signature, nonce, user, signer = self._sign_request(params)
            params['signature'] = signature
            
            r = requests.get(
                f"{self.base_url}/fapi/v3/balance",
                params=params,
                timeout=5,
                headers={"X-MBX-APIKEY": self.api_key}
            )
            
            if r.status_code == 200:
                data = r.json()
                logger.info(f"Balance fetched: {data}")
                return data
            else:
                logger.error(f"Balance fetch failed: {r.status_code} {r.text}")
                return None
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return None
    
    def get_positions(self, symbol: Optional[str] = None) -> Optional[list]:
        """Get open positions."""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            signature, nonce, user, signer = self._sign_request(params)
            params['signature'] = signature
            
            r = requests.get(
                f"{self.base_url}/fapi/v3/positionRisk",
                params=params,
                timeout=5,
                headers={"X-MBX-APIKEY": self.api_key}
            )
            
            if r.status_code == 200:
                return r.json()
            else:
                logger.error(f"Positions fetch failed: {r.status_code} {r.text}")
                return None
        except Exception as e:
            logger.error(f"Positions fetch error: {e}")
            return None
    
    def place_order(self, symbol: str, side: str, quantity: float, price: float, 
                   order_type: str = "LIMIT", **kwargs) -> Optional[Dict]:
        """
        Place order.
        
        Args:
            symbol: e.g. "BTC-USDT"
            side: "BUY" or "SELL"
            quantity: e.g. 0.01
            price: e.g. 45000.00
            order_type: "LIMIT" or "MARKET"
        """
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': str(quantity),
            }
            
            if order_type == "LIMIT":
                params['price'] = str(price)
                params['timeInForce'] = 'GTC'
            
            # Add any extra params (newClientOrderId, etc.)
            params.update(kwargs)
            
            signature, nonce, user, signer = self._sign_request(params)
            params['signature'] = signature
            
            logger.info(f"Placing {side} order: {quantity} {symbol} @ {price}")
            
            r = requests.post(
                f"{self.base_url}/fapi/v3/order",
                data=params,
                timeout=10,
                headers={
                    "X-MBX-APIKEY": self.api_key,
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            
            if r.status_code in [200, 201]:
                order = r.json()
                logger.info(f"✅ Order placed: {order.get('orderId')}")
                return order
            else:
                logger.error(f"Order placement failed: {r.status_code} {r.text}")
                return None
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Cancel order."""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            signature, nonce, user, signer = self._sign_request(params)
            params['signature'] = signature
            
            r = requests.delete(
                f"{self.base_url}/fapi/v3/order",
                data=params,
                timeout=10,
                headers={
                    "X-MBX-APIKEY": self.api_key,
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            
            if r.status_code == 200:
                logger.info(f"✅ Order cancelled: {order_id}")
                return r.json()
            else:
                logger.error(f"Cancel failed: {r.status_code} {r.text}")
                return None
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return None
    
    def get_open_orders(self, symbol: str) -> Optional[list]:
        """Get open orders for symbol."""
        try:
            params = {'symbol': symbol}
            signature, nonce, user, signer = self._sign_request(params)
            params['signature'] = signature
            
            r = requests.get(
                f"{self.base_url}/fapi/v3/openOrders",
                params=params,
                timeout=5,
                headers={"X-MBX-APIKEY": self.api_key}
            )
            
            if r.status_code == 200:
                return r.json()
            else:
                logger.error(f"Open orders fetch failed: {r.status_code} {r.text}")
                return None
        except Exception as e:
            logger.error(f"Open orders error: {e}")
            return None
    
    # ===== SPOT TRADING METHODS (Project-4) =====
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get ticker info (price, volume, etc.) for Spot symbol.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
        
        Returns: Ticker dict with lastPrice, lastQty, etc.
        """
        try:
            if self.is_spot:
                endpoint = f"{self.base_url}/api/v3/ticker/24hr"
            else:
                endpoint = f"{self.base_url}/fapi/v1/ticker/24hr"
            
            r = requests.get(
                endpoint,
                params={'symbol': symbol},
                timeout=5
            )
            
            if r.status_code == 200:
                return r.json()
            else:
                logger.error(f"Ticker fetch failed: {r.status_code} {r.text}")
                return None
        except Exception as e:
            logger.error(f"Ticker fetch error: {e}")
            return None
    
    def place_market_order(self, symbol: str, side: str, quantity: float, **kwargs) -> Optional[Dict]:
        """
        Place market order on Spot (BUY/SELL immediately).
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            side: "BUY" or "SELL"
            quantity: Amount to buy/sell
        
        Returns: Order response with orderId, executedQty, fills, etc.
        """
        try:
            if not self.is_spot:
                logger.error("place_market_order only works for Spot API. Use place_order for Futures.")
                return None
            
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': 'MARKET',
                'quantity': str(quantity),
            }
            
            params.update(kwargs)
            
            signature, nonce, user, signer = self._sign_request(params)
            params['signature'] = signature
            
            logger.info(f"Placing MARKET {side} order: {quantity} {symbol}")
            
            r = requests.post(
                f"{self.base_url}/api/v3/order",
                data=params,
                timeout=10,
                headers={
                    "X-MBX-APIKEY": self.api_key,
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            
            if r.status_code in [200, 201]:
                order = r.json()
                logger.info(f"✅ Market order placed: {order.get('orderId')}")
                return order
            else:
                logger.error(f"Market order failed: {r.status_code} {r.text}")
                return None
        except Exception as e:
            logger.error(f"Market order error: {e}")
            return None
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get spot account balances.
        
        Returns: Dict like {'USDT': 9.5, 'BTC': 0.0001, ...}
        """
        try:
            if not self.is_spot:
                logger.error("get_account_balance (spot) should use Spot API")
                return {}
            
            params = {}
            signature, nonce, user, signer = self._sign_request(params)
            params['signature'] = signature
            
            r = requests.get(
                f"{self.base_url}/api/v3/account",
                params=params,
                timeout=5,
                headers={"X-MBX-APIKEY": self.api_key}
            )
            
            if r.status_code == 200:
                account = r.json()
                
                # Extract balances
                balances = {}
                for balance in account.get('balances', []):
                    asset = balance.get('asset')
                    free = float(balance.get('free', 0))
                    locked = float(balance.get('locked', 0))
                    total = free + locked
                    
                    if total > 0:
                        balances[asset] = total
                
                logger.debug(f"Account balances: {balances}")
                return balances
            else:
                logger.error(f"Account fetch failed: {r.status_code} {r.text}")
                return {}
        except Exception as e:
            logger.error(f"Account fetch error: {e}")
            return {}
