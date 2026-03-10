#!/usr/bin/env python3
"""
Asterdex Spot Trading Bot - MetaMask Wallet Signing (EIP-712)
Auto-trades spot BTC/ETH on Asterdex using MetaMask wallet.

Entry: Market order, 1 USDT per trade
Exit: Take profit (+3%) and Stop loss (-2%)

Uses Aster Spot API v1 (mainnet) with proper EIP-712 wallet signing.
"""

import os
import sys
import json
import time
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import requests
from urllib.parse import urlencode

from eth_account import Account
try:
    from eth_account.messages import encode_structured_data
except ImportError:
    # Fallback for older versions
    from web3 import Web3
    from eth_utils import to_bytes

# Configuration
TRADING_PAIRS = ["BTC-USDT", "ETH-USDT"]
POSITION_SIZE_USD = 1.0  # $1 per trade
TP_PERCENT = 3.0         # +3% to exit
SL_PERCENT = 2.0         # -2% to exit
CHECK_INTERVAL = 30      # Check every 30s
ASTERDEX_SPOT_BASE_URL = "https://sapi.asterdex.com"  # Mainnet Spot
API_PATH = "/api/v1"

LOG_FILE = '/Users/geniustarigan/.openclaw/workspace/aster_bot/spot_bot_proper.log'
TRADE_LOG_FILE = '/Users/geniustarigan/.openclaw/workspace/aster_bot/spot_trades_proper.jsonl'

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


class AsterSpotClientEIP712:
    """Asterdex Spot API client with EIP-712 wallet signing."""
    
    def __init__(self, private_key: str, wallet_address: str):
        """
        Initialize client with MetaMask wallet.
        
        Args:
            private_key: Private key (0x + 64 hex chars)
            wallet_address: Wallet address (0x + 40 hex chars)
        """
        self.private_key = private_key
        self.wallet_address = wallet_address
        self.account = Account.from_key(private_key)
        self.base_url = ASTERDEX_SPOT_BASE_URL + API_PATH
        self.session = requests.Session()
        
        logger.info(f"✅ Aster Spot Client initialized: {self.wallet_address}")
    
    def _sign_request(self, params: Dict) -> str:
        """
        Sign request with EIP-712 (or fallback to message signing).
        
        Steps:
        1. Add nonce (microseconds), user, signer to params
        2. Create param string (ASCII sorted)
        3. Sign with MetaMask wallet
        4. Return hex signature
        """
        # Add nonce, user, signer
        nonce = int(time.time() * 1_000_000)
        params['nonce'] = str(nonce)
        params['user'] = self.wallet_address
        params['signer'] = self.account.address
        
        # Create param string (ASCII sorted, all strings)
        sorted_keys = sorted(params.keys())
        param_string = "&".join(f"{k}={params[k]}" for k in sorted_keys)
        
        logger.debug(f"Param string: {param_string[:100]}...")
        
        # Try EIP-712 signing
        try:
            from eth_account.messages import encode_structured_data
            
            # EIP-712 domain
            domain_data = {
                "name": "AsterSignTransaction",
                "version": "1",
                "chainId": 1666,  # Off-chain Aster identifier
                "verifyingContract": "0x0000000000000000000000000000000000000000"
            }
            
            # Message type
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
            
            signed_msg = self.account.sign_message(structured_msg)
            signature = signed_msg.signature.hex()
            
            logger.debug(f"Signature (EIP-712): {signature[:20]}...")
            return signature
        
        except (ImportError, Exception) as e:
            # Fallback: Simple message signing
            logger.debug(f"EIP-712 not available, using message signing: {e}")
            
            from eth_account.messages import encode_defunct
            message = encode_defunct(text=param_string)
            signed_msg = self.account.sign_message(message)
            signature = signed_msg.signature.hex()
            
            logger.debug(f"Signature (message): {signature[:20]}...")
            return signature
    
    def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = True) -> Dict:
        """Make request to Aster Spot API."""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        # Sign if needed
        if signed:
            params['signature'] = self._sign_request(params)
        
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
            logger.debug(f"Response: {str(result)[:100]}...")
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    logger.error(f"Response: {e.response.json()}")
                except:
                    logger.error(f"Response: {e.response.text}")
            return {}
    
    def get_account_balance(self) -> Dict:
        """Fetch account balance."""
        return self._request('GET', '/account', {}, signed=True)
    
    def get_usdt_balance(self) -> float:
        """Get available USDT balance."""
        account = self.get_account_balance()
        
        if not account or 'balances' not in account:
            logger.error(f"❌ No balances in response: {account}")
            return 0.0
        
        for balance in account['balances']:
            if balance['asset'] == 'USDT':
                free = float(balance.get('free', 0))
                logger.info(f"💰 USDT Balance: ${free:.2f}")
                return free
        
        logger.warning("⚠️  USDT not found in account balances")
        return 0.0
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        try:
            # Market data endpoints are unsigned
            response = self.session.get(
                f"{ASTERDEX_SPOT_BASE_URL}/api/v1/ticker/price",
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
        
        logger.info(f"📤 Placing {side} order: {quantity:.6f} {symbol}")
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


class AsterdexSpotBot:
    """Main trading bot."""
    
    def __init__(self, client: AsterSpotClientEIP712):
        """Initialize bot with API client."""
        self.client = client
        self.open_positions = {}
        logger.info(f"🚀 Asterdex Spot Bot initialized")
        logger.info(f"Pairs: {TRADING_PAIRS}")
        logger.info(f"Position size: ${POSITION_SIZE_USD}")
        logger.info(f"TP: +{TP_PERCENT}% | SL: -{SL_PERCENT}%")
    
    def run(self):
        """Main bot loop."""
        logger.info("🚀 Starting bot...")
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
    private_key = os.getenv('ASTER_PRIVATE_KEY')
    wallet_address = os.getenv('ASTER_WALLET_ADDRESS')
    
    if not private_key or not wallet_address:
        logger.error("❌ Missing credentials")
        logger.error("Set in terminal:")
        logger.error("  export ASTER_PRIVATE_KEY=0xyourkey")
        logger.error("  export ASTER_WALLET_ADDRESS=0xyouraddress")
        sys.exit(1)
    
    client = AsterSpotClientEIP712(private_key, wallet_address)
    bot = AsterdexSpotBot(client)
    bot.run()
