#!/usr/bin/env python3
"""
ASTER TRADING EXECUTOR
Automatically executes smart-filter signals on Aster Finance.

Flow:
  1. Read SENT_SIGNALS.jsonl (live signals from smart-filter)
  2. For each OPEN signal: Place order via Aster API
  3. Monitor positions: Check current price vs TP/SL
  4. Close positions: Execute exit when TP/SL triggered
  5. Track P&L: Log realized profits/losses

Dependencies:
  - eth_account (pip install eth-account)
  - requests (pip install requests)
  - Aster skills must be loaded in OpenClaw

Env vars (from .env or terminal):
  ASTER_PRIVATE_KEY    = 0x... (wallet private key)
  ASTER_WALLET_ADDRESS = 0x... (wallet address)
  ASTER_CHAIN_ID       = 42161 (or your chain - usually Arbitrum)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import requests
from eth_account import Account
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/Users/geniustarigan/.openclaw/workspace/aster_trading_executor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
WORKSPACE_ROOT = "/Users/geniustarigan/.openclaw/workspace"
SENT_SIGNALS_PATH = f"{WORKSPACE_ROOT}/SENT_SIGNALS.jsonl"
TRADING_LOG_PATH = f"{WORKSPACE_ROOT}/aster_trading_log.jsonl"

ASTER_API_BASE = "https://fapi.asterdex.com/api/v3"
CHAIN_ID = 1666  # Aster off-chain signing ID (NOT 42161!)
DOMAIN_NAME = "AsterSignTransaction"
DOMAIN_VERSION = "1"

# Risk parameters
MAX_POSITION_SIZE = 1.0  # Max 1 BTC per trade (adjust as needed)
MIN_NOTIONAL = 10  # Minimum $10 USDT per order
LEVERAGE = 1  # Start with 1x (no leverage)

class AsterTradingExecutor:
    def __init__(self):
        # Load credentials from env
        self.private_key = os.getenv('ASTER_PRIVATE_KEY')
        self.wallet_address = os.getenv('ASTER_WALLET_ADDRESS')
        
        if not self.private_key or not self.wallet_address:
            raise ValueError("ASTER_PRIVATE_KEY and ASTER_WALLET_ADDRESS must be set in env")
        
        # Initialize account
        try:
            self.account = Account.from_key(self.private_key)
            logger.info(f"✅ Executor initialized for wallet: {self.account.address}")
        except Exception as e:
            logger.error(f"❌ Failed to load private key: {e}")
            raise
        
        self.nonce = self._get_nonce()
        self.open_positions = {}  # Track open positions by signal_uuid
    
    def _get_nonce(self) -> int:
        """Fetch current nonce from Aster API"""
        try:
            r = requests.get(f"{ASTER_API_BASE}/nonce", timeout=5)
            if r.status_code == 200:
                nonce = r.json().get('nonce', 0)
                logger.info(f"Current nonce: {nonce}")
                return nonce
            else:
                logger.warning(f"Failed to fetch nonce: {r.status_code}. Using system time.")
                return int(time.time() * 1_000_000)  # Fallback: system time in microseconds
        except Exception as e:
            logger.warning(f"Nonce fetch failed: {e}. Using system time.")
            return int(time.time() * 1_000_000)
    
    def _get_next_nonce(self) -> int:
        """Increment nonce for next request"""
        self.nonce += 1
        return self.nonce
    
    def _sign_message(self, params: Dict) -> str:
        """
        Sign request for Aster API
        
        Uses EIP-712 signing (ChainId 1666)
        """
        # Create param string: key=value, sorted ASCII
        param_items = sorted(params.items())
        param_str = "&".join([f"{k}={v}" for k, v in param_items])
        
        try:
            # Use Web3.py for EIP-712 signing if available
            try:
                from eth_account.messages import encode_structured_data
                
                typed_data = {
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
                        "name": DOMAIN_NAME,
                        "version": DOMAIN_VERSION,
                        "chainId": CHAIN_ID,
                        "verifyingContract": "0x0000000000000000000000000000000000000000"
                    },
                    "message": {
                        "msg": param_str
                    }
                }
                
                msg = encode_structured_data(typed_data)
            except ImportError:
                # Fallback: use simple message signing
                from eth_account.messages import encode_defunct
                msg = encode_defunct(text=param_str)
            
            signed = self.account.sign_message(msg)
            logger.debug(f"Signed: {param_str[:50]}...")
            return signed.signature.hex()
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            raise
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price from Aster market data API"""
        try:
            r = requests.get(
                f"{ASTER_API_BASE}/ticker/24hr",
                params={'symbol': symbol},
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                price = float(data.get('lastPrice', 0))
                logger.info(f"{symbol} current price: ${price}")
                return price
            else:
                logger.warning(f"Failed to fetch price for {symbol}: {r.status_code}")
                return None
        except Exception as e:
            logger.error(f"Price fetch failed: {e}")
            return None
    
    def place_order(self, signal: Dict) -> Optional[Dict]:
        """
        Place order on Aster based on signal
        
        Signal format:
        {
            "uuid": "...",
            "symbol": "BTC-USDT",
            "direction": "LONG",
            "entry_price": 50000,
            "tp_target": 51000,
            "sl_target": 49000,
            "fired_time_utc": "2026-03-10T14:30:00.000Z",
            ...
        }
        """
        try:
            symbol = signal['symbol']
            direction = signal['side']  # LONG or SHORT (field name is 'side', not 'direction')
            entry_price = float(signal['entry_price'])
            tp_target = float(signal['tp_target'])
            sl_target = float(signal['sl_target'])
            uuid = signal['uuid']
            
            # Calculate quantity (for now: fixed 0.01 BTC or equivalent)
            # TODO: Risk-based sizing
            if symbol.startswith('BTC'):
                quantity = 0.01
            elif symbol.startswith('ETH'):
                quantity = 0.1
            else:
                quantity = 1.0
            
            # Verify minimum notional
            notional = entry_price * quantity
            if notional < MIN_NOTIONAL:
                logger.warning(f"Order notional ${notional} < ${MIN_NOTIONAL}. Skipping.")
                return None
            
            # Prepare order
            nonce = self._get_next_nonce()
            
            side = "BUY" if direction == "LONG" else "SELL"
            order_type = "LIMIT"
            
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': str(quantity),
                'price': str(entry_price),
                'timeInForce': 'GTC',
                'nonce': str(nonce)
            }
            
            # Sign and submit
            signature = self._sign_message(params)
            params['signature'] = signature
            
            logger.info(f"Placing {direction} order: {quantity} {symbol} @ ${entry_price}")
            
            r = requests.post(
                f"{ASTER_API_BASE}/order",
                data=params,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=10
            )
            
            if r.status_code in [200, 201]:
                order = r.json()
                order_id = order.get('orderId')
                logger.info(f"✅ Order placed: {order_id}")
                
                # Store position tracking
                self.open_positions[uuid] = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'tp_target': tp_target,
                    'sl_target': sl_target,
                    'placed_at': datetime.now(timezone.utc).isoformat(),
                    'status': 'OPEN'
                }
                
                # Log
                self._log_trade({
                    'action': 'PLACE_ORDER',
                    'signal_uuid': uuid,
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'tp_target': tp_target,
                    'sl_target': sl_target,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                return order
            else:
                logger.error(f"❌ Order failed: {r.status_code} - {r.text}")
                return None
        
        except Exception as e:
            logger.error(f"Place order failed: {e}")
            return None
    
    def monitor_positions(self):
        """Monitor open positions and close if TP/SL triggered"""
        if not self.open_positions:
            return
        
        logger.info(f"Monitoring {len(self.open_positions)} positions...")
        
        for uuid, pos in list(self.open_positions.items()):
            if pos['status'] != 'OPEN':
                continue
            
            symbol = pos['symbol']
            current_price = self.get_current_price(symbol)
            
            if not current_price:
                continue
            
            tp = pos['tp_target']
            sl = pos['sl_target']
            side = pos['side']
            
            # Check TP/SL
            should_close = False
            exit_reason = None
            
            if side == 'BUY':  # LONG position
                if current_price >= tp:
                    should_close = True
                    exit_reason = 'TP_HIT'
                elif current_price <= sl:
                    should_close = True
                    exit_reason = 'SL_HIT'
            else:  # SHORT position
                if current_price <= tp:
                    should_close = True
                    exit_reason = 'TP_HIT'
                elif current_price >= sl:
                    should_close = True
                    exit_reason = 'SL_HIT'
            
            if should_close:
                logger.info(f"🎯 {exit_reason}: {symbol} @ ${current_price}")
                self.close_position(uuid, current_price, exit_reason)
    
    def close_position(self, uuid: str, exit_price: float, exit_reason: str):
        """Close position at market"""
        pos = self.open_positions.get(uuid)
        if not pos:
            return
        
        try:
            symbol = pos['symbol']
            quantity = pos['quantity']
            side = pos['side']
            entry_price = pos['entry_price']
            
            # Close order (opposite side, market)
            close_side = "SELL" if side == "BUY" else "BUY"
            
            nonce = self._get_next_nonce()
            
            params = {
                'symbol': symbol,
                'side': close_side,
                'type': 'MARKET',
                'quantity': str(quantity),
                'nonce': str(nonce)
            }
            
            signature = self._sign_message(params)
            params['signature'] = signature
            
            logger.info(f"Closing {symbol}: {quantity} @ ${exit_price} ({exit_reason})")
            
            r = requests.post(
                f"{ASTER_API_BASE}/order",
                data=params,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=10
            )
            
            if r.status_code in [200, 201]:
                close_order = r.json()
                close_order_id = close_order.get('orderId')
                
                # Calculate P&L
                pnl = (exit_price - entry_price) * quantity if side == "BUY" else (entry_price - exit_price) * quantity
                pnl_pct = (pnl / (entry_price * quantity)) * 100 if entry_price > 0 else 0
                
                logger.info(f"✅ Closed: {close_order_id} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                
                # Update position
                pos['status'] = 'CLOSED'
                pos['close_order_id'] = close_order_id
                pos['exit_price'] = exit_price
                pos['exit_reason'] = exit_reason
                pos['pnl_usd'] = pnl
                pos['pnl_pct'] = pnl_pct
                pos['closed_at'] = datetime.now(timezone.utc).isoformat()
                
                # Log
                self._log_trade({
                    'action': 'CLOSE_POSITION',
                    'signal_uuid': uuid,
                    'close_order_id': close_order_id,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_usd': pnl,
                    'pnl_pct': pnl_pct,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            else:
                logger.error(f"Close failed: {r.status_code} - {r.text}")
        
        except Exception as e:
            logger.error(f"Close position failed: {e}")
    
    def _log_trade(self, entry: Dict):
        """Append trade log entry"""
        try:
            with open(TRADING_LOG_PATH, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Log write failed: {e}")
    
    def run_cycle(self):
        """Execute one trading cycle"""
        logger.info("=" * 80)
        logger.info("ASTER TRADING EXECUTOR - CYCLE START")
        logger.info("=" * 80)
        
        try:
            # Read signals
            if not os.path.exists(SENT_SIGNALS_PATH):
                logger.warning(f"Signals file not found: {SENT_SIGNALS_PATH}")
                return
            
            new_signals = []
            with open(SENT_SIGNALS_PATH, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    signal = json.loads(line)
                    if signal.get('status') == 'OPEN' and signal.get('uuid') not in self.open_positions:
                        new_signals.append(signal)
            
            logger.info(f"Found {len(new_signals)} new signals to execute")
            
            # Place orders for new signals
            for signal in new_signals:
                self.place_order(signal)
            
            # Monitor and close positions
            self.monitor_positions()
            
            logger.info("=" * 80)
            logger.info("CYCLE COMPLETE")
            logger.info("=" * 80)
        
        except Exception as e:
            logger.error(f"Cycle failed: {e}")


def main():
    """Main loop"""
    logger.info("🚀 ASTER TRADING EXECUTOR starting...")
    
    try:
        executor = AsterTradingExecutor()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    
    # Run cycles every 60 seconds
    while True:
        try:
            executor.run_cycle()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        
        time.sleep(60)


if __name__ == '__main__':
    main()
