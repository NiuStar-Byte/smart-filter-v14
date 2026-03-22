"""
PEC Enhanced Reporter - Configuration Constants
All hardcoded values extracted for maintainability & flexibility
Date: 2026-03-20 00:31 GMT+7
"""

from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL CATEGORIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Boundary between FOUNDATION (immutable) and NEW signals
# Signals BEFORE this date → FOUNDATION (locked at 853)
# Signals AT OR AFTER this date → NEW (accumulating)
NEW_SIGNALS_CUTOFF_DATE = datetime(2026, 3, 16, tzinfo=timezone.utc)

# ═══════════════════════════════════════════════════════════════════════════════
# TIMEOUT WINDOW VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

# Expected maximum durations for each timeframe (before signal is "stale")
# Multipliers: how many periods until timeout expected
# Examples:
#   '15min': 15 periods → 15 × 15min = 225min (3.75 hours)
#   '30min': 10 periods → 10 × 30min = 300min (5 hours)
#   '1h': 5 periods → 5 × 1h = 5 hours
TIMEOUT_WINDOW_MULTIPLIERS = {
    '15min': 15,  # 15 × 15min = 225 minutes
    '30min': 10,  # 10 × 30min = 300 minutes
    '1h': 5       # 5 × 1h = 300 minutes
}

# Tolerance buffer for timing slippage (10% = 0.10)
# Used to calculate clean timeout threshold:
#   threshold = expected_max × (1 + TIMEOUT_TOLERANCE_PERCENT)
# Example: 1h window is 18,000s + 10% = 19,800s
TIMEOUT_TOLERANCE_PERCENT = 0.10

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING & LEVERAGE
# ═══════════════════════════════════════════════════════════════════════════════

# Fixed margin per trade (in USD)
# Used to calculate notional position for P&L calculations
FIXED_MARGIN_USD = 100.0

# Fixed leverage multiplier
# Used to calculate notional position
FIXED_LEVERAGE = 10.0

# Derived notional position
# Notional = Margin × Leverage (e.g., $100 × 10x = $1000 notional)
def get_notional_position():
    """Calculate total notional position size"""
    return FIXED_MARGIN_USD * FIXED_LEVERAGE

# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOL GROUPING (5D DIMENSION)
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOL_GROUPS = {
    "MAIN_BLOCKCHAIN": [
        "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT",
        "AVAX-USDT", "BNB-USDT", "XLM-USDT", "LINK-USDT", "POL-USDT"
    ],
    "TOP_ALTS": [
        "ZKJ-USDT", "ROAM-USDT", "XAUT-USDT", "SAHARA-USDT"
    ],
    "MID_ALTS": [
        "XPL-USDT", "DOT-USDT", "FUEL-USDT", "VIRTUAL-USDT", "BERA-USDT",
        "CROSS-USDT", "FUN-USDT", "ENA-USDT", "SOL-USDT", "AVAX-USDT"
    ]
}

# ═══════════════════════════════════════════════════════════════════════════════
# TIMEFRAME CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Timeframe definitions in seconds (for calculations)
TIMEFRAME_SECONDS = {
    '15min': 15 * 60,     # 900 seconds
    '30min': 30 * 60,     # 1800 seconds
    '1h': 60 * 60         # 3600 seconds
}

# ═══════════════════════════════════════════════════════════════════════════════
# VERSION TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG_VERSION = "1.0"
CONFIG_DATE = "2026-03-20 00:31 GMT+7"
CONFIG_DESCRIPTION = "All hardcoded values extracted for maintainability"

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION & HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_config():
    """Validate configuration values are sensible"""
    errors = []
    
    # Margin & Leverage checks
    if FIXED_MARGIN_USD <= 0:
        errors.append(f"FIXED_MARGIN_USD must be > 0, got {FIXED_MARGIN_USD}")
    if FIXED_LEVERAGE <= 0:
        errors.append(f"FIXED_LEVERAGE must be > 0, got {FIXED_LEVERAGE}")
    
    # Timeout multipliers check
    if not TIMEOUT_WINDOW_MULTIPLIERS:
        errors.append("TIMEOUT_WINDOW_MULTIPLIERS cannot be empty")
    for tf, mult in TIMEOUT_WINDOW_MULTIPLIERS.items():
        if mult <= 0:
            errors.append(f"TIMEOUT_WINDOW_MULTIPLIERS[{tf}] must be > 0, got {mult}")
    
    # Tolerance check
    if not (0 <= TIMEOUT_TOLERANCE_PERCENT <= 1.0):
        errors.append(f"TIMEOUT_TOLERANCE_PERCENT must be 0-1, got {TIMEOUT_TOLERANCE_PERCENT}")
    
    return errors

def get_config_summary():
    """Return human-readable config summary"""
    return f"""
PEC Reporter Configuration Summary
───────────────────────────────────
Version: {CONFIG_VERSION} ({CONFIG_DATE})

Signal Cutoff:
  Foundation/New boundary: {NEW_SIGNALS_CUTOFF_DATE.strftime('%Y-%m-%d')}

Timeout Validation:
  Multipliers: {TIMEOUT_WINDOW_MULTIPLIERS}
  Tolerance: {TIMEOUT_TOLERANCE_PERCENT*100:.0f}%

Position Sizing:
  Margin: ${FIXED_MARGIN_USD:.2f}
  Leverage: {FIXED_LEVERAGE:.1f}x
  Notional: ${get_notional_position():.2f}

Symbol Groups: {len(SYMBOL_GROUPS)} groups
Timeframes: {list(TIMEFRAME_SECONDS.keys())}
───────────────────────────────────
"""

if __name__ == "__main__":
    # Quick validation
    print(get_config_summary())
    errors = validate_config()
    if errors:
        print("❌ CONFIGURATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration valid")
