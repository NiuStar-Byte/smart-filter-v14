"""
asterdex_config.py - Configuration for Asterdex Entry Poster

Status: ⏳ AWAITING API CREDENTIALS

Instructions:
1. Get Asterdex API key + secret from your account
2. Paste into ASTERDEX_API_KEY and ASTERDEX_API_SECRET below
3. Restart asterdex_entry_poster.py

DO NOT commit API keys to git or share publicly.
"""

import os
from pathlib import Path

# ============================================================================
# ASTERDEX PRO API V3 CREDENTIALS (TWO WALLETS - MAIN ACCOUNT + API WALLET)
# ============================================================================
# Pro API V3 requires TWO separate wallets:
#   1. ASTER_MAIN_ACCOUNT - Your main trading account
#   2. ASTER_API_WALLET_ADDRESS - The authorized API wallet (from Asterdex)
#   3. ASTER_API_WALLET_PRIVATE_KEY - The API wallet's private key
#
# Set these in ~/.openclaw/workspace/.env:
#   source ~/.openclaw/workspace/.env
#
# ⚠️ NEVER commit private keys to git or paste in chat!

ASTER_MAIN_ACCOUNT = os.environ.get("ASTER_MAIN_ACCOUNT")
ASTER_API_WALLET_ADDRESS = os.environ.get("ASTER_API_WALLET_ADDRESS")
ASTER_API_WALLET_PRIVATE_KEY = os.environ.get("ASTER_API_WALLET_PRIVATE_KEY")

# Validate credentials are set
if not ASTER_MAIN_ACCOUNT or not ASTER_API_WALLET_ADDRESS or not ASTER_API_WALLET_PRIVATE_KEY:
    raise ValueError(
        "❌ ASTERDEX PRO API V3 wallet credentials not found!\n"
        "Set environment variables in ~/.openclaw/workspace/.env:\n"
        "  export ASTER_MAIN_ACCOUNT='0x...' (your main trading account)\n"
        "  export ASTER_API_WALLET_ADDRESS='0x...' (authorized API wallet)\n"
        "  export ASTER_API_WALLET_PRIVATE_KEY='0x...' (API wallet private key)\n"
        "Then: source ~/.openclaw/workspace/.env && python3 asterdex_entry_poster.py"
    )

# Validate wallet address formats
if not ASTER_MAIN_ACCOUNT.startswith("0x") or len(ASTER_MAIN_ACCOUNT) != 42:
    raise ValueError(f"Invalid ASTER_MAIN_ACCOUNT format: {ASTER_MAIN_ACCOUNT}")

if not ASTER_API_WALLET_ADDRESS.startswith("0x") or len(ASTER_API_WALLET_ADDRESS) != 42:
    raise ValueError(f"Invalid ASTER_API_WALLET_ADDRESS format: {ASTER_API_WALLET_ADDRESS}")

if not ASTER_API_WALLET_PRIVATE_KEY.startswith("0x") or len(ASTER_API_WALLET_PRIVATE_KEY) != 66:
    raise ValueError(f"Invalid ASTER_API_WALLET_PRIVATE_KEY format (must be 0x + 64 hex chars)")

# ============================================================================
# ASTERDEX ENDPOINTS (PRO API V3)
# ============================================================================
ASTERDEX_BASE_URL = "https://fapi.asterdex.com"
ASTERDEX_TESTNET_URL = "https://testnet.asterdex.com"

# PRO API V3 endpoints
ASTERDEX_ENDPOINT = ASTERDEX_BASE_URL  # ✅ PRODUCTION MODE
API_VERSION = "v3"  # Using PRO API V3

# V3 EIP-712 signing parameters
EIP712_DOMAIN = {
    "name": "AsterSignTransaction",
    "version": "1",
    "chainId": 1666,  # Off-chain chainId for AsterDex (NOT Harmony)
    "verifyingContract": "0x" + "0" * 40,  # Null address
}

# ============================================================================
# POSITION SETTINGS
# ============================================================================
POSITION_SETTINGS = {
    "margin": 2.0,              # Margin per entry (USDT) - UPDATED June 2 11:52 GMT+7
    "leverage": 10,             # Leverage multiplier (10x) - Default (override per symbol if needed)
    "notional_value": 20.0,     # margin × leverage = $20 USDT per entry (2.0 × 10)
    "margin_type": "ISOLATED",  # ISOLATED or CROSS (use ISOLATED for futures)
    "order_type": "LIMIT",      # LIMIT (pending at entry_price) or MARKET
    "time_in_force": "GTC",     # GTC=Good Till Cancel, IOC=Immediate or Cancel, FOK=Fill or Kill
}

# ============================================================================
# TIER FILTERING
# ============================================================================
# Only post entries for signals with these tier values
TIER_FILTER = [1, 2, 3]  # Reject Tier-X and unassigned signals

# ============================================================================
# RATE LIMITING & POLLING
# ============================================================================
RATE_LIMIT_SETTINGS = {
    "signal_check_interval_sec": 5,    # Poll SIGNALS_MASTER every 5 seconds
    "order_check_interval_sec": 30,    # Check fills every 30 seconds
    "max_retries": 3,                  # Retry failed API calls 3 times
    "retry_delay_sec": 2,              # Wait 2 seconds between retries
}

# ============================================================================
# FILE PATHS
# ============================================================================
INTEGRATION_DIR = Path(__file__).parent
SIGNALS_MASTER_PATH = Path("/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl")  # ✅ UPDATED to workspace root (fresh signals)
ENTRY_LOG_PATH = INTEGRATION_DIR / "data" / "entry_log.jsonl"
ACTIVE_ORDERS_PATH = INTEGRATION_DIR / "data" / "active_orders.jsonl"
COOLDOWN_STATE_PATH = INTEGRATION_DIR / "data" / "cooldown_state.json"
LOG_DIR = INTEGRATION_DIR / "logs"

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S GMT+7"

# ============================================================================
# SAFETY SWITCHES - PRODUCTION MODE ENABLED
# ============================================================================
DRY_RUN_MODE = False  # ✅ REAL MODE - Posts actual orders to Asterdex
PRODUCTION_MODE = True  # ✅ Using BASE_URL (https://fapi.asterdex.com)

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================
MAX_CONCURRENT_ORDERS = 10  # Max number of open orders across all symbols
COOLDOWN_STRICT = True      # If True: strict (symbol+TF), if False: loose (symbol only)

print(f"""
[ASTERDEX_CONFIG] Loaded configuration
  ├─ Endpoint: {'TESTNET' if 'testnet' in ASTERDEX_ENDPOINT else 'PRODUCTION'}
  ├─ Dry run: {DRY_RUN_MODE}
  ├─ Margin/Leverage: ${POSITION_SETTINGS['margin']} × {POSITION_SETTINGS['leverage']}x = ${POSITION_SETTINGS['notional_value']}
  ├─ Tier filter: {TIER_FILTER}
  └─ Check interval: {RATE_LIMIT_SETTINGS['signal_check_interval_sec']}s
""")
