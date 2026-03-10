#!/usr/bin/env python3
"""
Aster API v1 Client - HMAC SHA256 Signing (API Key + Secret)
For both Futures (/fapi/v1/) and Spot (/api/v1/) trading.
"""

import os
import time
import json
import hashlib
import hmac
import requests
from typing import Dict, Optional, Any
from urllib.parse import urlencode
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env file if it exists
def _load_env():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

_load_env()

class AsterClient:
    def __init__(self, api_key: str = None, api_secret: str = None, wallet_address: str = None, api_base_url: str = None):
        """
        Initialize Aster API v1 client (HMAC SHA256 signing).
        
        Args:
            api_key: API key (from env: ASTER_API_KEY)
            api_secret: API secret (from env: ASTER_API_SECRET)
            wallet_address: Wallet address (from env: ASTER_WALLET_ADDRESS) - optional
            api_base_url: Override base URL (default: https://fapi.asterdex.com for Futures)
        """
        self.api_key = api_key or os.getenv('ASTER_API_KEY', '')
        self.api_secret = api_secret or os.getenv('ASTER_API_SECRET', '')
        self.wallet_address = wallet_address or os.getenv('ASTER_WALLET_ADDRESS', '')
        
        # Default to Futures API, but allow override for Spot
        self.base_url = api_base_url or "https://fapi.asterdex.com"
        self.is_spot = "sapi" in self.base_url.lower() or "/api/" in self.base_url.lower()
        
        api_type = "Spot" if self.is_spot else "Futures"
        logger.info(f"✅ Aster {api_type} API v1 client initialized")
        logger.info(f"   Base URL: {self.base_url}")
        logger.info(f"   API Key: {self.api_key[:8]}..." if self.api_key else "   No API key")
    
    def _get_signature(self, query_string: str) -> str:
        """
        Sign request with HMAC SHA256 (v1 method).
        
        Args:
            query_string: URL-encoded parameters (key1=value1&key2=value2&...)
        
        Returns: Hex signature
        """
        if not self.api_secret:
            raise ValueError("API secret not set. Cannot sign requests.")
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        logger.debug(f"Signed query (len={len(query_string)}): {signature[:16]}...")
        return signature
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, is_signed: bool = False) -> Optional[Dict]:
        """
        Make HTTP request to Aster API.
        
        Args:
            method: GET, POST, PUT, DELETE
            endpoint: API endpoint (e.g., /fapi/v1/account)
            params: Query/body parameters
            is_signed: Whether this is a signed request (add timestamp + signature)
        
        Returns: JSON response or None if failed
        """
        if params is None:
            params = {}
        
        # Add timestamp for signed requests
        if is_signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
        
        # Build query string
        query_string = urlencode(sorted(params.items()))
        
        # Sign if needed
        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        
        if is_signed:
            signature = self._get_signature(query_string)
            query_string += f"&signature={signature}"
        
        # Build URL and handle query string based on method
        url = f"{self.base_url}{endpoint}"
        
        # Make request
        try:
            if method == "GET":
                # For GET, append query string to URL only if there's a query string
                if query_string:
                    full_url = f"{url}?{query_string}"
                else:
                    full_url = url
                r = requests.get(full_url, headers=headers, timeout=10)
            elif method == "POST":
                # For POST, send as body
                post_headers = headers.copy()
                post_headers['Content-Type'] = 'application/x-www-form-urlencoded'
                r = requests.post(url, data=query_string, headers=post_headers, timeout=10)
            elif method == "DELETE":
                # For DELETE, send as body
                del_headers = headers.copy()
                del_headers['Content-Type'] = 'application/x-www-form-urlencoded'
                r = requests.delete(url, data=query_string, headers=del_headers, timeout=10)
            elif method == "PUT":
                # For PUT, send as body
                put_headers = headers.copy()
                put_headers['Content-Type'] = 'application/x-www-form-urlencoded'
                r = requests.put(url, data=query_string, headers=put_headers, timeout=10)
            else:
                logger.error(f"Unknown method: {method}")
                return None
            
            # Handle response
            if r.status_code in [200, 201]:
                try:
                    return r.json()
                except:
                    logger.debug(f"Response text: {r.text}")
                    return None
            else:
                logger.error(f"❌ API request failed: {r.status_code} {r.text}")
                try:
                    error_data = r.json()
                    logger.error(f"Response: {error_data}")
                except:
                    pass
                return None
        
        except Exception as e:
            logger.error(f"❌ Request error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ===== PUBLIC ENDPOINTS (No signature needed) =====
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        endpoint = "/api/v1/ticker/price" if self.is_spot else "/fapi/v1/ticker/price"
        response = self._make_request("GET", endpoint, {"symbol": symbol}, is_signed=False)
        if response and 'price' in response:
            return float(response['price'])
        return None
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24h ticker info (price, volume, etc.)."""
        # Try futures first
        if not self.is_spot:
            response = self._make_request("GET", "/fapi/v1/ticker/24hr", {"symbol": symbol}, is_signed=False)
        else:
            # Spot API - try multiple endpoint variations
            response = self._make_request("GET", "/api/v3/ticker/24hr", {"symbol": symbol}, is_signed=False)
            if not response:
                response = self._make_request("GET", "/api/v1/ticker/24hr", {"symbol": symbol}, is_signed=False)
        
        return response
    
    def get_server_time(self) -> Optional[int]:
        """Get server time in milliseconds."""
        endpoint = "/api/v1/time" if self.is_spot else "/fapi/v1/time"
        response = self._make_request("GET", endpoint, {}, is_signed=False)
        if response and 'serverTime' in response:
            return response['serverTime']
        return None
    
    def get_klines(self, symbol: str, interval: str = "5m", limit: int = 100) -> Optional[list]:
        """Get candlestick data."""
        endpoint = "/api/v1/klines" if self.is_spot else "/fapi/v1/klines"
        response = self._make_request(
            "GET",
            endpoint,
            {"symbol": symbol, "interval": interval, "limit": limit},
            is_signed=False
        )
        return response
    
    def get_depth(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Get order book depth."""
        endpoint = "/api/v1/depth" if self.is_spot else "/fapi/v1/depth"
        response = self._make_request(
            "GET",
            endpoint,
            {"symbol": symbol, "limit": limit},
            is_signed=False
        )
        return response
    
    # ===== SIGNED ENDPOINTS (Account & Trading) =====
    
    def get_account_balance(self) -> Optional[Dict]:
        """
        Get account balance.
        
        Returns: Dict with asset balances and account info
        """
        endpoint = "/api/v1/account" if self.is_spot else "/fapi/v1/account"
        response = self._make_request("GET", endpoint, {}, is_signed=True)
        
        if response:
            if self.is_spot:
                # Spot account returns balances directly
                balances = {}
                if 'balances' in response:
                    for balance in response['balances']:
                        asset = balance['asset']
                        free = float(balance['free'])
                        locked = float(balance['locked'])
                        if free > 0 or locked > 0:
                            balances[asset] = {'free': free, 'locked': locked, 'total': free + locked}
                logger.info(f"💰 Spot account balances: {list(balances.keys())}")
                return {'balances': balances}
            else:
                # Futures account returns totalWalletBalance
                wallet_balance = float(response.get('totalWalletBalance', 0))
                unrealized_pnl = float(response.get('totalUnrealizedProfit', 0))
                
                result = {
                    'total_wallet_balance': wallet_balance,
                    'total_unrealized_pnl': unrealized_pnl,
                    'can_trade': response.get('canTrade', False)
                }
                
                # Extract individual balances if available
                if 'balances' in response:
                    balances = {}
                    for balance in response['balances']:
                        asset = balance['asset']
                        free = float(balance['free'])
                        locked = float(balance['locked'])
                        if free > 0 or locked > 0:
                            balances[asset] = {'free': free, 'locked': locked, 'total': free + locked}
                    result['balances'] = balances
                
                logger.info(f"💰 Account balance: ${wallet_balance} USDT")
                return result
        else:
            logger.error(f"Failed to fetch account balance")
            return None
    
    def get_positions(self, symbol: Optional[str] = None) -> Optional[list]:
        """Get open positions (Futures only)."""
        if self.is_spot:
            logger.warning("get_positions is only available for Futures API")
            return None
        
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        response = self._make_request("GET", "/fapi/v1/positionRisk", params, is_signed=True)
        return response
    
    def place_order(self, symbol: str, side: str, order_type: str = "MARKET", 
                   quantity: float = None, price: float = None, **kwargs) -> Optional[Dict]:
        """
        Place order.
        
        Args:
            symbol: e.g., "BTCUSDT"
            side: "BUY" or "SELL"
            order_type: "MARKET" or "LIMIT"
            quantity: Order quantity
            price: Limit price (required for LIMIT orders)
        """
        if self.is_spot:
            logger.warning("Use place_market_order_spot or place_limit_order_spot for Spot API")
            return None
        
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
        }
        
        if quantity is not None:
            params['quantity'] = str(quantity)
        
        if order_type.upper() == "LIMIT" and price is not None:
            params['price'] = str(price)
            params['timeInForce'] = 'GTC'
        
        params.update(kwargs)
        
        logger.info(f"Placing {side} {order_type} order: {quantity} {symbol}")
        response = self._make_request("POST", "/fapi/v1/order", params, is_signed=True)
        
        if response:
            logger.info(f"✅ Order placed: {response.get('orderId')}")
        return response
    
    def cancel_order(self, symbol: str, order_id: str = None, orig_client_order_id: str = None) -> Optional[Dict]:
        """Cancel order."""
        endpoint = "/api/v1/order" if self.is_spot else "/fapi/v1/order"
        
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = order_id
        if orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        
        logger.info(f"Cancelling order: {order_id or orig_client_order_id}")
        response = self._make_request("DELETE", endpoint, params, is_signed=True)
        
        if response:
            logger.info(f"✅ Order cancelled")
        return response
    
    def get_open_orders(self, symbol: Optional[str] = None) -> Optional[list]:
        """Get all open orders (or for specific symbol)."""
        endpoint = "/api/v1/openOrders" if self.is_spot else "/fapi/v1/openOrders"
        
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        response = self._make_request("GET", endpoint, params, is_signed=True)
        return response
    
    def get_order_history(self, symbol: str, limit: int = 100) -> Optional[list]:
        """Get order history."""
        endpoint = "/api/v1/allOrders" if self.is_spot else "/fapi/v1/allOrders"
        
        params = {'symbol': symbol, 'limit': limit}
        response = self._make_request("GET", endpoint, params, is_signed=True)
        return response
    
    # ===== SPOT TRADING (Project-4) =====
    
    def place_market_order_spot(self, symbol: str, side: str, quantity: float = None, 
                                quoteOrderQty: float = None, **kwargs) -> Optional[Dict]:
        """
        Place market order on Spot API.
        
        Args:
            symbol: e.g., "BTC-USDT"
            side: "BUY" or "SELL"
            quantity: Amount to buy/sell (required for SELL)
            quoteOrderQty: Quote amount in USDT (required for BUY)
        """
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'MARKET',
        }
        
        if quantity is not None:
            params['quantity'] = str(quantity)
        if quoteOrderQty is not None:
            params['quoteOrderQty'] = str(quoteOrderQty)
        
        params.update(kwargs)
        
        logger.info(f"Placing MARKET {side} order: {symbol}")
        response = self._make_request("POST", "/api/v1/order", params, is_signed=True)
        
        if response:
            logger.info(f"✅ Spot order placed: {response.get('orderId')}")
        return response
    
    def get_account_balance_spot(self) -> Optional[Dict]:
        """Get spot account balance."""
        # Try v3 first, then v1
        response = self._make_request("GET", "/api/v3/account", {}, is_signed=True)
        if not response:
            response = self._make_request("GET", "/api/v1/account", {}, is_signed=True)
        
        if response:
            balances = {}
            if 'balances' in response:
                for balance in response['balances']:
                    asset = balance['asset']
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    if free > 0 or locked > 0:
                        balances[asset] = {'free': free, 'locked': locked, 'total': free + locked}
            
            if balances:
                logger.info(f"💰 Spot account balances: {balances}")
            else:
                logger.info(f"💰 No spot balances found (account empty or no assets)")
            
            return balances
        return None
