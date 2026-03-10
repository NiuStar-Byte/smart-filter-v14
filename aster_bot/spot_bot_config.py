#!/usr/bin/env python3
"""
Asterdex Spot Bot Configuration (Project-4)
"""

# ===== TRADING PARAMETERS =====

# Trading pairs to focus on (Asterdex Spot format: with dash)
TRADING_PAIRS = ["BTC-USDT", "ETH-USDT"]

# Position size: $1 per trade (fixed)
POSITION_SIZE_USD = 1.0

# Leverage: 1x (spot has no leverage)
LEVERAGE = 1

# Max open positions per pair
MAX_POSITIONS_PER_PAIR = 1

# ===== ENTRY CRITERIA =====

# Entry signal type: "MARKET" (immediate execution at market price)
ENTRY_SIGNAL = "MARKET"

# ===== EXIT CRITERIA =====

# Take profit: Close when price rises X% above entry
TP_PERCENT = 3.0

# Stop loss: Close when price drops X% below entry
SL_PERCENT = 2.0

# Check interval (seconds)
CHECK_INTERVAL = 30

# ===== ASTERDEX SPOT API =====

# Asterdex Spot API base URL (mainnet or testnet)
# Mainnet: https://sapi.asterdex.com
# Testnet: https://sapi.asterdex-testnet.com
ASTERDEX_SPOT_BASE_URL = "https://sapi.asterdex.com"

# ===== LOGGING =====

LOG_FILE = "/Users/geniustarigan/.openclaw/workspace/aster_bot/spot_bot.log"
TRADE_LOG_FILE = "/Users/geniustarigan/.openclaw/workspace/aster_bot/spot_trades.jsonl"
