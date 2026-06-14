"""
Symbol-specific Leverage Configuration
Overrides default leverage for symbols with lower max limits
"""

# Default leverage (used if symbol not in this override dict)
DEFAULT_LEVERAGE = 10

# Per-symbol max leverage overrides
# Add symbols here if they have lower max leverage limits on Asterdex
# Verified June 12 2026 11:44 GMT+7 via Asterdex UI
SYMBOL_LEVERAGE_OVERRIDES = {
    "KNC-USDT": 5,      # CONFIRMED: KNC-USDT max 5x leverage on Asterdex
    "FUN-USDT": 5,      # UPDATED (was 2x): FUN-USDT max 5x on Asterdex (June 12 11:44 GMT+7)
    "BERA-USDT": 5,     # CONFIRMED: BERA-USDT max 5x leverage on Asterdex (June 2 00:09 GMT+7)
    "DYDX-USDT": 5,     # CONFIRMED: DYDX-USDT max 5x leverage on Asterdex (June 3 23:14 GMT+7)
    "ALGO-USDT": 5,     # NEW: ALGO-USDT max 5x on Asterdex (June 12 11:44 GMT+7)
    "AVNT-USDT": 5,     # NEW: AVNT-USDT max 5x on Asterdex (June 12 11:44 GMT+7)
    "ZORA-USDT": 5,     # NEW: ZORA-USDT max 5x on Asterdex (June 12 11:44 GMT+7)
    "ARKM-USDT": 5,     # NEW: ARKM-USDT max 5x on Asterdex (June 12 11:44 GMT+7)
    "SPK-USDT": 5,      # NEW: SPK-USDT max 5x on Asterdex (June 12 14:03 GMT+7)
    "CFX-USDT": 5,      # NEW: CFX-USDT max 5x on Asterdex (June 13 01:29 GMT+7)
}

def get_leverage_for_symbol(symbol: str) -> int:
    """
    Get the leverage to use for a specific symbol.
    
    Args:
        symbol: Trading pair (e.g., "KNC-USDT", "SOL-USDT")
    
    Returns:
        Leverage value (e.g., 10, 5, 8)
    """
    return SYMBOL_LEVERAGE_OVERRIDES.get(symbol, DEFAULT_LEVERAGE)


def validate_leverage(symbol: str, requested_leverage: int) -> tuple:
    """
    Validate that requested leverage is within limits for symbol.
    
    Args:
        symbol: Trading pair
        requested_leverage: Leverage user wants
    
    Returns:
        (is_valid: bool, actual_leverage: int, message: str)
    """
    max_leverage = get_leverage_for_symbol(symbol)
    
    if requested_leverage <= max_leverage:
        return (
            True,
            requested_leverage,
            f"✅ {symbol} leverage {requested_leverage}x OK (max {max_leverage}x)"
        )
    else:
        return (
            False,
            max_leverage,
            f"⚠️ {symbol} requested {requested_leverage}x exceeds max {max_leverage}x, using {max_leverage}x"
        )


# Test
if __name__ == "__main__":
    test_symbols = ["KNC-USDT", "SOL-USDT", "ETH-USDT", "BTC-USDT"]
    
    print("=== Symbol Leverage Configuration Test ===\n")
    for sym in test_symbols:
        lev = get_leverage_for_symbol(sym)
        is_valid, actual, msg = validate_leverage(sym, 10)
        print(f"{sym:15} → {lev}x (default: {DEFAULT_LEVERAGE}x)")
        print(f"  {msg}\n")
