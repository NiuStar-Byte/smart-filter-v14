#!/usr/bin/env python3
"""
Aster Bot Configuration
"""

# ===== TRADING PARAMETERS =====

# Trading pairs to focus on (Aster Futures format: no dash)
TRADING_PAIRS = ["BTCUSDT", "ETHUSDT"]

# Position size: $1 per trade (fixed)
POSITION_SIZE_USD = 1.0

# Leverage: 1x (no leverage)
LEVERAGE = 1

# Max open positions per pair
MAX_POSITIONS_PER_PAIR = 1

# ===== ENTRY CRITERIA =====

# Entry signal type: "MA_CROSS" or "SUPPORT_RESISTANCE"
ENTRY_SIGNAL = "MA_CROSS"

# Moving average settings (if MA_CROSS)
MA_FAST = 10  # Fast MA period
MA_SLOW = 20  # Slow MA period
MA_INTERVAL = "5m"  # Kline interval

# Price threshold for entry (%)
ENTRY_DIP_PERCENT = 2.0  # Buy when price dips 2% below slow MA

# ===== EXIT CRITERIA =====

# Take profit: Close when price rises X% above entry
TP_PERCENT = 3.0

# Stop loss: Close when price drops X% below entry
SL_PERCENT = 2.0

# Check interval (seconds)
CHECK_INTERVAL = 30

# ===== LOGGING =====

LOG_FILE = "/Users/geniustarigan/.openclaw/workspace/aster_bot/bot.log"
TRADE_LOG_FILE = "/Users/geniustarigan/.openclaw/workspace/aster_bot/trades.jsonl"
