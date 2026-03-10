#!/usr/bin/env python3
"""
Asterdex Spot Trading Bot v2 - MetaMask Wallet Signing
Auto-trades spot BTC/ETH on Asterdex via MetaMask wallet (EIP-712 signed).

Entry: Market order, 1 USDT per trade
Exit: Take profit (+3%) and Stop loss (-2%)

Uses Aster Spot API v1 (mainnet) with proper authentication.
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

# Add module paths
sys.path.insert(0, '/Users/geniustarigan/.openclaw/workspace/aster_bot')

# Configuration
TRADING_PAIRS = ["BTC-USDT", "ETH-USDT"]
POSITION_SIZE_USD = 1.0  # $1 per trade
TP_PERCENT = 3.0         # +3% to exit
SL_PERCENT = 2.0         # -2% to exit
CHECK_INTERVAL = 30      # Check every 30s
ASTERDEX_SPOT_BASE_URL = "https://sapi.asterdex.com"  # Mainnet
API_PATH = "/api/v1"

LOG_FILE = '/Users/geniustarigan/.openclaw/workspace/aster_bot/spot_bot_v2.log'
TRADE_LOG_FILE = '/Users/geniustarigan/.openclaw/workspace/aster_bot/spot_trades_v2.jsonl'

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


class AsterSpotBotV2:
    """Asterdex Spot Bot using API v1 with HMAC SHA256 authentication."""
    
    def __init__(self, api_key: str, api_secret: str):
        """Initialize bot with API credentials."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = ASTERDEX_SPOT_BASE_URL + API_PATH
        self.open_positions = {}  # {symbol: {entry_price, qty, order_id, ...}}
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        })
        
        logger.info(f"✅ Aster Spot Bot v2 initialized")
        logger.info(f"Pairs: {TRADING_PAIRS}")
        logger.info(f"Position size: ${POSITION_SIZE_USD}")
        logger.info(f"TP: +{TP_PERCENT}% | SL: -{SL_PERCENT}%")
    
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature for request."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = True) -> Dict:
        """Make authenticated request to Aster Spot API."""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        # Add timestamp and signature for signed requests
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            params['signature'] = self._generate_signature(params)
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params)
            elif method == 'POST':
                response = self.session.post(url, data=params)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    logger.error(f"Response: {e.response.json()}")
                except:
                    logger.error(f"Response: {e.response.text}")
            return {}
    
    def get_account_balance(self) -> Dict:
        """Fetch account balance from Asterdex."""
        return self._request('GET', '/account', {}, signed=True)
    
    def get_usdt_balance(self) -> float:
        """Get available USDT balance."""
        account = self.get_account_balance()
        
        if 'balances' not in account:
            logger.error(f"❌ No balances in response: {account}")
            return 0.0
        
        for balance in account['balances']:
            if balance['asset'] == 'USDT':
                return float(balance.get('free', 0))
        
        logger.warning("⚠️  USDT not found in account balances")
        return 0.0
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        try:
            response = self.session.get(
                f"{ASTERDEX_SPOT_BASE_URL}/api/v1/ticker/price",
                params={'symbol': symbol}
            )
            response.raise_for_status()
            data = response.json()
            return float(data.get('price', 0))
        except Exception as e:
            logger.error(f"❌ Failed to get price for {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """Place market order."""
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity,
        }
        
        logger.info(f"📤 Placing {side} order: {quantity} {symbol}")
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
            'orderId': order_id,
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
    
    def run(self):
        """Main bot loop."""
        logger.info("🚀 Starting Asterdex Spot Bot v2...")
        logger.info(f"Check interval: {CHECK_INTERVAL}s")
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                logger.info(f"\n📊 CYCLE {cycle_count} - {datetime.now(timezone.utc).isoformat()}")
                
                # Check balance
                usdt_balance = self.get_usdt_balance()
                logger.info(f"USDT Balance: ${usdt_balance:.2f}")
                
                if usdt_balance < POSITION_SIZE_USD:
                    logger.warning(f"⚠️  Insufficient balance. Need ${POSITION_SIZE_USD}, have ${usdt_balance:.2f}")
                    time.sleep(CHECK_INTERVAL)
                    continue
                
                # Check open positions and manage TP/SL
                for symbol in TRADING_PAIRS:
                    open_orders = self.get_open_orders(symbol)
                    
                    if symbol in self.open_positions:
                        pos = self.open_positions[symbol]
                        current_price = self.get_price(symbol)
                        
                        if current_price:
                            pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                            logger.info(f"{symbol}: Entry=${pos['entry_price']:.2f}, Current=${current_price:.2f}, PnL={pnl_pct:+.2f}%")
                            
                            # Check TP
                            if pnl_pct >= TP_PERCENT:
                                logger.info(f"🎯 TP HIT on {symbol}! Closing position...")
                                if self.cancel_order(symbol, pos['order_id']):
                                    self.place_market_order(symbol, 'SELL', pos['quantity'])
                                    self._log_trade(symbol, pos, current_price, 'CLOSED_TP_HIT')
                                    del self.open_positions[symbol]
                            
                            # Check SL
                            elif pnl_pct <= -SL_PERCENT:
                                logger.warning(f"🛑 SL HIT on {symbol}! Closing position...")
                                if self.cancel_order(symbol, pos['order_id']):
                                    self.place_market_order(symbol, 'SELL', pos['quantity'])
                                    self._log_trade(symbol, pos, current_price, 'CLOSED_SL_HIT')
                                    del self.open_positions[symbol]
                    else:
                        # Try to open new position if capital available
                        current_price = self.get_price(symbol)
                        if current_price:
                            quantity = POSITION_SIZE_USD / current_price
                            logger.info(f"💰 Opening position on {symbol} at ${current_price:.2f}")
                            
                            order = self.place_market_order(symbol, 'BUY', quantity)
                            if order:
                                self.open_positions[symbol] = {
                                    'entry_price': current_price,
                                    'quantity': quantity,
                                    'order_id': order.get('orderId'),
                                    'entry_time': datetime.now(timezone.utc).isoformat()
                                }
                
                logger.info(f"✅ Cycle {cycle_count} complete. Sleeping {CHECK_INTERVAL}s...")
                time.sleep(CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
    
    def _log_trade(self, symbol: str, position: Dict, exit_price: float, status: str):
        """Log trade to JSONL file."""
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
        
        logger.info(f"📝 Trade logged: {trade}")


if __name__ == '__main__':
    # Get credentials from environment
    api_key = os.getenv('ASTER_API_KEY')
    api_secret = os.getenv('ASTER_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("❌ Missing ASTER_API_KEY or ASTER_API_SECRET environment variables")
        logger.error("Set them in your terminal:")
        logger.error("  export ASTER_API_KEY=your_key")
        logger.error("  export ASTER_API_SECRET=your_secret")
        sys.exit(1)
    
    bot = AsterSpotBotV2(api_key, api_secret)
    bot.run()
