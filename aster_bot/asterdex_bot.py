#!/usr/bin/env python3
"""
Asterdex Trading Bot - API Key Authentication
Auto-trades BTC/ETH on Asterdex Futures API v3 using API key + HMAC SHA256.

Entry: Market order, 1 USDT per trade
Exit: Take profit (+3%) and Stop loss (-2%)

Uses Aster Futures API v3 (mainnet) with HMAC SHA256 signing.
"""

import os
import sys
import json
import time
import logging
import hashlib
import hmac
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import requests
from urllib.parse import urlencode

# Configuration
TRADING_PAIRS = ["BTCUSDT", "ETHUSDT"]
POSITION_SIZE_USD = 1.0  # $1 per trade
TP_PERCENT = 3.0         # +3% to exit
SL_PERCENT = 2.0         # -2% to exit
CHECK_INTERVAL = 30      # Check every 30s
ASTERDEX_FUTURES_BASE_URL = "https://fapi.asterdex.com"
API_PATH = "/fapi/v1"  # Using v1 API which supports API key HMAC auth

LOG_FILE = '/Users/geniustarigan/.openclaw/workspace/aster_bot/bot.log'
TRADE_LOG_FILE = '/Users/geniustarigan/.openclaw/workspace/aster_bot/trades.jsonl'

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AsterClient:
    """Asterdex API client with HMAC SHA256 authentication."""
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize client with API credentials.
        
        Args:
            api_key: API key from Asterdex
            api_secret: API secret from Asterdex
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = ASTERDEX_FUTURES_BASE_URL + API_PATH
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        })
        
        logger.info(f"✅ Aster Client initialized with API key")
    
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature for request (Binance-style)."""
        # Convert all params to strings first
        str_params = {k: str(v) for k, v in params.items()}
        
        # Sort params by key (ASCII order) - critical for Binance/Aster auth
        sorted_params = sorted(str_params.items())
        query_string = urlencode(sorted_params)
        
        logger.debug(f"Signing: {query_string[:100]}...")
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        logger.debug(f"Signature: {signature[:20]}...")
        return signature
    
    def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = True) -> Dict:
        """Make authenticated request to Aster API."""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        # Add timestamp and recvWindow BEFORE calculating signature
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            
            # Calculate signature from params BEFORE adding it
            signature = self._generate_signature(params)
            params['signature'] = signature
        
        try:
            logger.debug(f"{method} {endpoint}")
            
            if method == 'GET':
                response = self.session.get(url, params=params)
            elif method == 'POST':
                response = self.session.post(url, data=urlencode(params))
            elif method == 'DELETE':
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Response OK")
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    logger.error(f"Response: {e.response.json()}")
                except:
                    logger.error(f"Response: {e.response.text}")
            return {}
    
    def get_account(self) -> Dict:
        """Fetch account info."""
        return self._request('GET', '/account', {}, signed=True)
    
    def get_usdt_balance(self) -> float:
        """Get available USDT balance."""
        account = self.get_account()
        
        if not account or 'assets' not in account:
            logger.error(f"❌ No account data")
            return 0.0
        
        for asset in account['assets']:
            if asset['asset'] == 'USDT':
                available = float(asset.get('availableBalance', 0))
                logger.info(f"💰 USDT Balance: ${available:.2f}")
                return available
        
        logger.warning("⚠️  USDT not found")
        return 0.0
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        try:
            response = self.session.get(
                f"{ASTERDEX_FUTURES_BASE_URL}/fapi/v1/ticker/price",
                params={'symbol': symbol}
            )
            response.raise_for_status()
            data = response.json()
            price = float(data.get('price', 0))
            logger.debug(f"{symbol}: ${price:.2f}")
            return price
        except Exception as e:
            logger.error(f"❌ Failed to get price for {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """Place market order."""
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': str(quantity),
        }
        
        logger.info(f"📤 Placing {side} order: {quantity:.4f} {symbol}")
        result = self._request('POST', '/order', params, signed=True)
        
        if 'orderId' in result:
            logger.info(f"✅ Order placed: {result['orderId']}")
            return result
        else:
            logger.error(f"❌ Order failed: {result}")
            return None
    
    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel an order."""
        params = {
            'symbol': symbol,
            'orderId': str(order_id),
        }
        
        result = self._request('DELETE', '/order', params, signed=True)
        
        if 'orderId' in result:
            logger.info(f"✅ Order cancelled: {order_id}")
            return True
        else:
            logger.error(f"❌ Cancel failed: {result}")
            return False
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders."""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        result = self._request('GET', '/openOrders', params, signed=True)
        return result if isinstance(result, list) else []


class AsterdexTradingBot:
    """Main trading bot."""
    
    def __init__(self, client: AsterClient):
        """Initialize bot with API client."""
        self.client = client
        self.open_positions = {}
        logger.info(f"🚀 Asterdex Trading Bot initialized")
        logger.info(f"Pairs: {TRADING_PAIRS}")
        logger.info(f"Position size: ${POSITION_SIZE_USD}")
        logger.info(f"TP: +{TP_PERCENT}% | SL: -{SL_PERCENT}%")
    
    def run(self):
        """Main bot loop."""
        logger.info("🚀 Starting trading bot...")
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                logger.info(f"\n📊 CYCLE {cycle_count} - {datetime.now(timezone.utc).isoformat()}")
                
                # Check balance
                usdt_balance = self.client.get_usdt_balance()
                
                if usdt_balance < POSITION_SIZE_USD:
                    logger.warning(f"⚠️  Insufficient balance. Need ${POSITION_SIZE_USD}, have ${usdt_balance:.2f}")
                    time.sleep(CHECK_INTERVAL)
                    continue
                
                # Manage positions
                for symbol in TRADING_PAIRS:
                    if symbol in self.open_positions:
                        pos = self.open_positions[symbol]
                        current_price = self.client.get_price(symbol)
                        
                        if current_price:
                            pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                            logger.info(f"{symbol}: Entry=${pos['entry_price']:.2f}, Current=${current_price:.2f}, PnL={pnl_pct:+.2f}%")
                            
                            # Check TP
                            if pnl_pct >= TP_PERCENT:
                                logger.info(f"🎯 TP HIT on {symbol}! Closing...")
                                if self.client.cancel_order(symbol, pos['order_id']):
                                    self.client.place_market_order(symbol, 'SELL', pos['quantity'])
                                    self._log_trade(symbol, pos, current_price, 'CLOSED_TP_HIT')
                                    del self.open_positions[symbol]
                            
                            # Check SL
                            elif pnl_pct <= -SL_PERCENT:
                                logger.warning(f"🛑 SL HIT on {symbol}! Closing...")
                                if self.client.cancel_order(symbol, pos['order_id']):
                                    self.client.place_market_order(symbol, 'SELL', pos['quantity'])
                                    self._log_trade(symbol, pos, current_price, 'CLOSED_SL_HIT')
                                    del self.open_positions[symbol]
                    else:
                        # Open new position
                        current_price = self.client.get_price(symbol)
                        if current_price:
                            quantity = POSITION_SIZE_USD / current_price
                            order = self.client.place_market_order(symbol, 'BUY', quantity)
                            
                            if order:
                                self.open_positions[symbol] = {
                                    'entry_price': current_price,
                                    'quantity': quantity,
                                    'order_id': order.get('orderId'),
                                    'entry_time': datetime.now(timezone.utc).isoformat()
                                }
                
                logger.info(f"✅ Cycle complete. Sleeping {CHECK_INTERVAL}s...")
                time.sleep(CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  Bot stopped")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
    
    def _log_trade(self, symbol: str, position: Dict, exit_price: float, status: str):
        """Log trade."""
        pnl = (exit_price - position['entry_price']) * position['quantity']
        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
        
        trade = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'status': status
        }
        
        with open(TRADE_LOG_FILE, 'a') as f:
            f.write(json.dumps(trade) + '\n')
        
        logger.info(f"📝 Trade: {trade}")


if __name__ == '__main__':
    api_key = os.getenv('ASTER_API_KEY')
    api_secret = os.getenv('ASTER_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("❌ Missing ASTER_API_KEY or ASTER_API_SECRET")
        logger.error("Set them in your terminal:")
        logger.error("  export ASTER_API_KEY=your_key")
        logger.error("  export ASTER_API_SECRET=your_secret")
        sys.exit(1)
    
    client = AsterClient(api_key, api_secret)
    bot = AsterdexTradingBot(client)
    bot.run()
