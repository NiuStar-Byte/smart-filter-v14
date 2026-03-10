#!/usr/bin/env python3
"""
Asterdex Spot Trading Bot (Project-4)
Auto-trades spot BTC/ETH on Asterdex via Binance Wallet (EIP-712 signed).

Entry: Market order, 1 USDT per trade
Exit: Take profit (+3%) and Stop loss (-2%)
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import uuid as uuid_lib
import requests

# Add module paths
sys.path.insert(0, '/Users/geniustarigan/.openclaw/workspace/aster_bot')

from aster_client import AsterClient
from spot_bot_config import (
    TRADING_PAIRS, POSITION_SIZE_USD, LEVERAGE,
    ENTRY_SIGNAL, TP_PERCENT, SL_PERCENT, CHECK_INTERVAL,
    LOG_FILE, TRADE_LOG_FILE, ASTERDEX_SPOT_BASE_URL
)

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AsterdexSpotBot:
    def __init__(self):
        """Initialize Asterdex Spot bot."""
        self.client = AsterClient(api_base_url=ASTERDEX_SPOT_BASE_URL)
        self.open_positions = {}  # Track by symbol: {symbol: {order_id, entry_price, quantity, tp_price, sl_price, ...}}
        self.trade_history = []
        self.account_balance = {}
        
        logger.info("="*80)
        logger.info("🤖 ASTERDEX SPOT BOT INITIALIZED (Project-4)")
        logger.info("="*80)
        logger.info(f"Pairs: {TRADING_PAIRS}")
        logger.info(f"Position size: ${POSITION_SIZE_USD} per trade")
        logger.info(f"Entry: Market order")
        logger.info(f"TP: +{TP_PERCENT}%, SL: -{SL_PERCENT}%")
        logger.info(f"Check interval: {CHECK_INTERVAL}s")
        logger.info("="*80)
        
        # Initialize account
        self.update_account_balance()
    
    def update_account_balance(self):
        """Fetch current USDT balance."""
        try:
            balance = self.client.get_account_balance()
            self.account_balance = balance
            usdt_balance = balance.get('USDT', 0)
            logger.info(f"💰 Account balance: ${usdt_balance} USDT")
            return usdt_balance
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
            return 0
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current spot price for symbol."""
        try:
            ticker = self.client.get_ticker(symbol)
            if ticker:
                price = float(ticker.get('lastPrice', 0))
                logger.debug(f"Price {symbol}: ${price}")
                return price
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get price for {symbol}: {e}")
            return None
    
    def place_buy_order(self, symbol: str, amount_usdt: float) -> Optional[Dict]:
        """
        Place market buy order on Asterdex Spot.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            amount_usdt: Amount in USDT
        
        Returns: Order response or None if failed
        """
        try:
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                logger.error(f"Cannot get price for {symbol}")
                return None
            
            # Calculate quantity (USDT / price, rounded to symbol precision)
            quantity = round(amount_usdt / current_price, 8)
            
            logger.info(f"📈 BUY {symbol}: {quantity} @ ${current_price} (total: ${amount_usdt})")
            
            # Place market order
            order = self.client.place_market_order(
                symbol=symbol,
                side="BUY",
                quantity=quantity
            )
            
            if order:
                order_id = order.get('orderId')
                logger.info(f"✅ Buy order placed: {order_id}")
                
                # Track position
                tp_price = current_price * (1 + TP_PERCENT / 100)
                sl_price = current_price * (1 - SL_PERCENT / 100)
                
                self.open_positions[symbol] = {
                    'order_id': order_id,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'entry_time': datetime.now(timezone.utc).isoformat(),
                    'status': 'OPEN'
                }
                
                logger.info(f"  Entry: ${current_price:.8f}")
                logger.info(f"  TP: ${tp_price:.8f} (+{TP_PERCENT}%)")
                logger.info(f"  SL: ${sl_price:.8f} (-{SL_PERCENT}%)")
                
                return order
            else:
                logger.error(f"❌ Buy order failed for {symbol}")
                return None
        
        except Exception as e:
            logger.error(f"❌ Exception placing buy order: {e}")
            return None
    
    def place_sell_order(self, symbol: str, quantity: float, reason: str = "MANUAL") -> Optional[Dict]:
        """
        Place market sell order to close position.
        
        Args:
            symbol: Trading pair
            quantity: Amount to sell
            reason: "TP_HIT", "SL_HIT", or "MANUAL"
        """
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                logger.error(f"Cannot get price for {symbol}")
                return None
            
            logger.info(f"📉 SELL {symbol}: {quantity} @ ${current_price} ({reason})")
            
            # Place market sell
            order = self.client.place_market_order(
                symbol=symbol,
                side="SELL",
                quantity=quantity
            )
            
            if order:
                order_id = order.get('orderId')
                logger.info(f"✅ Sell order placed: {order_id} ({reason})")
                
                # Mark position as closed
                if symbol in self.open_positions:
                    self.open_positions[symbol]['status'] = f'CLOSED_{reason}'
                    self.open_positions[symbol]['exit_price'] = current_price
                    self.open_positions[symbol]['exit_time'] = datetime.now(timezone.utc).isoformat()
                    
                    entry_price = self.open_positions[symbol]['entry_price']
                    pnl = (current_price - entry_price) * quantity
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    logger.info(f"  Exit: ${current_price:.8f}")
                    logger.info(f"  P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                    
                    # Save to trade log
                    self.log_trade(symbol, self.open_positions[symbol], pnl, pnl_pct)
                
                return order
            else:
                logger.error(f"❌ Sell order failed for {symbol}")
                return None
        
        except Exception as e:
            logger.error(f"❌ Exception placing sell order: {e}")
            return None
    
    def check_positions(self):
        """Check open positions and close if TP/SL hit."""
        for symbol in TRADING_PAIRS:
            if symbol not in self.open_positions:
                continue
            
            position = self.open_positions[symbol]
            if position['status'] != 'OPEN':
                continue
            
            current_price = self.get_current_price(symbol)
            if not current_price:
                continue
            
            quantity = position['quantity']
            tp_price = position['tp_price']
            sl_price = position['sl_price']
            
            # Check take profit
            if current_price >= tp_price:
                logger.warning(f"🎯 TP HIT for {symbol}: ${current_price:.8f} >= ${tp_price:.8f}")
                self.place_sell_order(symbol, quantity, reason="TP_HIT")
            
            # Check stop loss
            elif current_price <= sl_price:
                logger.warning(f"🛑 SL HIT for {symbol}: ${current_price:.8f} <= ${sl_price:.8f}")
                self.place_sell_order(symbol, quantity, reason="SL_HIT")
    
    def log_trade(self, symbol: str, position: Dict, pnl: float, pnl_pct: float):
        """Log trade to JSONL file."""
        try:
            trade_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'entry_price': position['entry_price'],
                'exit_price': position.get('exit_price'),
                'quantity': position['quantity'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'tp_price': position['tp_price'],
                'sl_price': position['sl_price'],
                'status': position['status']
            }
            
            with open(TRADE_LOG_FILE, 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
            
            logger.info(f"📝 Trade logged: {symbol} P&L=${pnl:.2f} ({pnl_pct:+.2f}%)")
        except Exception as e:
            logger.error(f"❌ Failed to log trade: {e}")
    
    def run(self):
        """Main bot loop."""
        logger.info("🚀 Starting bot loop...")
        cycle = 0
        
        try:
            while True:
                cycle += 1
                logger.info(f"\n📍 Cycle {cycle} at {datetime.now(timezone.utc).isoformat()}")
                
                # Check existing positions for TP/SL
                self.check_positions()
                
                # Try to open new positions if USDT available
                balance = self.update_account_balance()
                
                for symbol in TRADING_PAIRS:
                    # Skip if already have open position
                    if symbol in self.open_positions and self.open_positions[symbol]['status'] == 'OPEN':
                        logger.debug(f"⏭️  {symbol} already has open position, skipping")
                        continue
                    
                    # Only open if we have enough balance
                    if balance >= POSITION_SIZE_USD:
                        logger.info(f"📌 Opening new position: {symbol}")
                        self.place_buy_order(symbol, POSITION_SIZE_USD)
                        
                        # Update balance after order
                        balance = self.update_account_balance()
                    else:
                        logger.warning(f"⚠️  Insufficient balance (${balance} < ${POSITION_SIZE_USD})")
                
                # Sleep before next check
                logger.info(f"⏸️  Sleeping {CHECK_INTERVAL}s until next cycle...")
                time.sleep(CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("⛔ Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Bot crashed: {e}")
            raise

if __name__ == "__main__":
    bot = AsterdexSpotBot()
    bot.run()
