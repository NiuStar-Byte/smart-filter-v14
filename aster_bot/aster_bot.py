#!/usr/bin/env python3
"""
Aster Futures Trading Bot
Standalone bot with built-in entry/exit criteria.

Entry: Price dips below moving average
Exit: Take profit (3%) or stop loss (2%)
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

# Add module paths
sys.path.insert(0, '/Users/geniustarigan/.openclaw/workspace/aster_bot')

from aster_client import AsterClient
from bot_config import (
    TRADING_PAIRS, POSITION_SIZE_USD, LEVERAGE,
    MA_FAST, MA_SLOW, MA_INTERVAL, ENTRY_DIP_PERCENT,
    TP_PERCENT, SL_PERCENT, CHECK_INTERVAL,
    LOG_FILE, TRADE_LOG_FILE
)

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

class AsterBot:
    def __init__(self):
        """Initialize bot."""
        self.client = AsterClient()
        self.open_positions = {}  # Track by symbol: {symbol: {order_id, entry_price, tp, sl, ...}}
        self.trade_history = []
        
        logger.info("="*80)
        logger.info("🤖 ASTER TRADING BOT INITIALIZED")
        logger.info("="*80)
        logger.info(f"Pairs: {TRADING_PAIRS}")
        logger.info(f"Position size: ${POSITION_SIZE_USD}")
        logger.info(f"Entry: {ENTRY_DIP_PERCENT}% dip below {MA_SLOW}-bar MA")
        logger.info(f"TP: +{TP_PERCENT}%, SL: -{SL_PERCENT}%")
        logger.info("="*80)
    
    def calculate_ma(self, klines: List, period: int) -> Optional[float]:
        """Calculate moving average from klines."""
        if not klines or len(klines) < period:
            return None
        
        closes = [float(k[4]) for k in klines[-period:]]
        return sum(closes) / len(closes)
    
    def should_buy(self, symbol: str) -> Optional[Dict]:
        """
        Check if BUY signal triggered for symbol.
        
        Returns: {price, ma_fast, ma_slow, dip_percent} or None
        """
        try:
            # Fetch klines
            klines = self.client.get_klines(symbol, interval=MA_INTERVAL, limit=max(MA_FAST, MA_SLOW) + 5)
            if not klines or len(klines) < MA_SLOW:
                return None
            
            # Calculate MAs
            ma_fast = self.calculate_ma(klines, MA_FAST)
            ma_slow = self.calculate_ma(klines, MA_SLOW)
            
            if not ma_fast or not ma_slow:
                return None
            
            # Current price
            current_price = float(klines[-1][4])
            
            # Check if price dipped below slow MA
            dip_threshold = ma_slow * (1 - ENTRY_DIP_PERCENT / 100)
            
            if current_price < dip_threshold:
                logger.info(f"✅ BUY SIGNAL {symbol}: price ${current_price:.2f} < MA-{MA_SLOW} ${ma_slow:.2f} (dip {ENTRY_DIP_PERCENT}%)")
                return {
                    'price': current_price,
                    'ma_fast': ma_fast,
                    'ma_slow': ma_slow,
                    'dip_percent': ENTRY_DIP_PERCENT
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Buy check error for {symbol}: {e}")
            return None
    
    def place_buy_order(self, symbol: str, entry_price: float) -> Optional[Dict]:
        """Place BUY order."""
        try:
            # Calculate quantity based on position size
            quantity = POSITION_SIZE_USD / entry_price
            
            # Verify minimum quantity (simplified check)
            if quantity < 0.001:
                logger.warning(f"⚠️  Quantity too small: {quantity}. Skipping.")
                return None
            
            # Round to 4 decimals
            quantity = round(quantity, 4)
            entry_price = round(entry_price, 2)
            
            # Place LIMIT order
            order = self.client.place_order(
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                price=entry_price,
                order_type="LIMIT",
                newClientOrderId=str(uuid_lib.uuid4())[:12]
            )
            
            if order:
                # Calculate TP and SL
                tp_price = entry_price * (1 + TP_PERCENT / 100)
                sl_price = entry_price * (1 - SL_PERCENT / 100)
                
                # Store position
                self.open_positions[symbol] = {
                    'order_id': order.get('orderId'),
                    'side': 'BUY',
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'status': 'OPEN',
                    'placed_at': datetime.now(timezone.utc).isoformat()
                }
                
                logger.info(f"📊 Position tracked: {symbol} @ ${entry_price} (TP: ${tp_price:.2f}, SL: ${sl_price:.2f})")
                self._log_trade('ENTRY', symbol, entry_price, quantity, order.get('orderId'))
                
                return order
            
            return None
        
        except Exception as e:
            logger.error(f"Buy order error: {e}")
            return None
    
    def monitor_positions(self):
        """Monitor open positions for TP/SL."""
        if not self.open_positions:
            return
        
        logger.info(f"📈 Monitoring {len(self.open_positions)} position(s)...")
        
        for symbol in list(self.open_positions.keys()):
            pos = self.open_positions[symbol]
            
            if pos['status'] != 'OPEN':
                continue
            
            try:
                # Get current price
                current_price = self.client.get_price(symbol)
                if not current_price:
                    continue
                
                entry_price = pos['entry_price']
                tp_price = pos['tp_price']
                sl_price = pos['sl_price']
                pnl_percent = (current_price - entry_price) / entry_price * 100
                
                # Check TP
                if current_price >= tp_price:
                    logger.info(f"🎯 TAKE PROFIT {symbol}: ${current_price:.2f} >= ${tp_price:.2f} (+{pnl_percent:.2f}%)")
                    self._close_position(symbol, current_price, 'TP')
                
                # Check SL
                elif current_price <= sl_price:
                    logger.info(f"🛑 STOP LOSS {symbol}: ${current_price:.2f} <= ${sl_price:.2f} ({pnl_percent:.2f}%)")
                    self._close_position(symbol, current_price, 'SL')
                
                else:
                    # Still open
                    logger.debug(f"Position {symbol}: ${current_price:.2f} (entry: ${entry_price:.2f}, P&L: {pnl_percent:+.2f}%)")
            
            except Exception as e:
                logger.error(f"Monitor position error for {symbol}: {e}")
    
    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close position."""
        try:
            pos = self.open_positions[symbol]
            
            # Place SELL order
            order = self.client.place_order(
                symbol=symbol,
                side="SELL",
                quantity=pos['quantity'],
                price=exit_price,
                order_type="LIMIT",
                newClientOrderId=str(uuid_lib.uuid4())[:12]
            )
            
            if order:
                entry_price = pos['entry_price']
                pnl = (exit_price - entry_price) * pos['quantity']
                pnl_percent = (exit_price - entry_price) / entry_price * 100
                
                pos['exit_price'] = exit_price
                pos['exit_reason'] = reason
                pos['pnl'] = pnl
                pos['pnl_percent'] = pnl_percent
                pos['status'] = 'CLOSED'
                pos['closed_at'] = datetime.now(timezone.utc).isoformat()
                
                logger.info(f"✅ Position closed: {symbol} @ ${exit_price:.2f} ({reason}, P&L: ${pnl:+.2f} / {pnl_percent:+.2f}%)")
                self._log_trade('EXIT', symbol, exit_price, pos['quantity'], order.get('orderId'), pnl, pnl_percent)
                
                # Archive position
                self.trade_history.append(pos)
                del self.open_positions[symbol]
        
        except Exception as e:
            logger.error(f"Close position error: {e}")
    
    def _log_trade(self, action: str, symbol: str, price: float, quantity: float, 
                  order_id: str, pnl: Optional[float] = None, pnl_percent: Optional[float] = None):
        """Log trade to file."""
        try:
            trade = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action': action,
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'order_id': order_id,
            }
            if pnl is not None:
                trade['pnl'] = pnl
            if pnl_percent is not None:
                trade['pnl_percent'] = pnl_percent
            
            with open(TRADE_LOG_FILE, 'a') as f:
                f.write(json.dumps(trade) + '\n')
        except Exception as e:
            logger.error(f"Log trade error: {e}")
    
    def run(self):
        """Main bot loop."""
        try:
            while True:
                try:
                    # Check each pair for entry signals
                    for symbol in TRADING_PAIRS:
                        # Skip if already have position
                        if symbol in self.open_positions:
                            continue
                        
                        # Check for buy signal
                        signal = self.should_buy(symbol)
                        if signal:
                            self.place_buy_order(symbol, signal['price'])
                    
                    # Monitor existing positions
                    self.monitor_positions()
                    
                    # Sleep before next cycle
                    time.sleep(CHECK_INTERVAL)
                
                except KeyboardInterrupt:
                    logger.info("\n🛑 Bot stopped by user.")
                    break
                except Exception as e:
                    logger.error(f"Cycle error: {e}")
                    time.sleep(5)
        
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            logger.info("="*80)
            logger.info("🤖 BOT SHUTDOWN")
            logger.info(f"Trades executed: {len(self.trade_history)}")
            if self.trade_history:
                total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
                logger.info(f"Total P&L: ${total_pnl:+.2f}")
            logger.info("="*80)

if __name__ == "__main__":
    bot = AsterBot()
    bot.run()
