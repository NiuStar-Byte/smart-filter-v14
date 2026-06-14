#!/usr/bin/env python3
"""
ASTERDEX SYMBOL CONFIGURATION
User-observed Asterdex platform configuration (Jun 8 2026 21:08 GMT+7)

Source: Direct observation from Asterdex UI by user
Verified by: User manual checking of trading pairs
Last Updated: June 8 2026 21:08 GMT+7
"""

# Symbols NOT available in Asterdex (skip posting)
# Reason: These symbols don't exist on Asterdex platform
UNAVAILABLE_SYMBOLS = {
    'ACTSOL-USDT',  # Not available
    'ZKJ-USDT',     # Not available
    'CROSS-USDT',   # Not available
    'ATH-USDT',     # Not available
    'MAGIC-USDT',   # Not available
}

# Symbol name mapping (our system → Asterdex)
# Use this when posting to Asterdex API
SYMBOL_MAPPING = {
    'XAUT-USDT': 'XAU-USDT',  # Our: XAUT-USDT, Asterdex: XAU-USDT
}

# Leverage overrides for specific symbols
# Default leverage is 10x, but these symbols require different leverage
LEVERAGE_OVERRIDES = {
    # Leverage 5x
    'KNC-USDT': 5,          # Existing override
    'BERA-USDT': 5,         # Existing override
    'ALGO-USDT': 5,         # Observed Jun 8
    'AVNT-USDT': 5,         # Observed Jun 8
    'BLUR-USDT': 5,         # Observed Jun 8 21:20
    'PARTI-USDT': 5,        # Observed Jun 8 21:20
    'PORTAL-USDT': 5,       # Observed Jun 8 21:20
    'LA-USDT': 5,           # Observed Jun 8 21:20
    'DYDX-USDT': 5,         # Observed Jun 8 21:20
    
    # Leverage 3x
    'FUN-USDT': 3,          # Updated Jun 8 21:20 (was 2x)
}

def get_asterdex_symbol(our_symbol):
    """
    Convert our symbol format to Asterdex symbol format.
    
    Args:
        our_symbol: Symbol in our system (e.g., 'XAUT-USDT')
    
    Returns:
        Symbol for Asterdex API (e.g., 'XAU-USDT')
    """
    return SYMBOL_MAPPING.get(our_symbol, our_symbol)

def is_available_in_asterdex(symbol):
    """
    Check if symbol is available for posting to Asterdex.
    
    Args:
        symbol: Symbol to check (e.g., 'ACTSOL-USDT')
    
    Returns:
        True if available, False if unavailable
    """
    return symbol not in UNAVAILABLE_SYMBOLS

def get_leverage(symbol):
    """
    Get leverage for a symbol (10x default, with overrides).
    
    Args:
        symbol: Symbol to check (e.g., 'ALGO-USDT')
    
    Returns:
        Leverage value (e.g., 5 or 10)
    """
    return LEVERAGE_OVERRIDES.get(symbol, 10)

def validate_symbol_for_posting(symbol):
    """
    Validate if a symbol can be posted to Asterdex.
    Returns symbol to use (mapped) or None if invalid.
    
    Args:
        symbol: Symbol to validate
    
    Returns:
        Mapped symbol if valid, None if unavailable
    """
    if not is_available_in_asterdex(symbol):
        return None
    return get_asterdex_symbol(symbol)

# Quick reference
SUMMARY = {
    'unavailable_count': len(UNAVAILABLE_SYMBOLS),
    'mapped_symbols': len(SYMBOL_MAPPING),
    'leverage_overrides': len(LEVERAGE_OVERRIDES),
    'last_updated': 'June 8 2026 21:08 GMT+7',
}

if __name__ == '__main__':
    print("=== ASTERDEX SYMBOL CONFIG ===")
    print(f"Unavailable: {SUMMARY['unavailable_count']} symbols")
    print(f"Mapped: {SUMMARY['mapped_symbols']} symbols")
    print(f"Leverage overrides: {SUMMARY['leverage_overrides']} symbols")
    print()
    print("Unavailable symbols:")
    for sym in sorted(UNAVAILABLE_SYMBOLS):
        print(f"  ❌ {sym}")
    print()
    print("Symbol mappings:")
    for our, asterdex in SYMBOL_MAPPING.items():
        print(f"  {our} → {asterdex}")
    print()
    print("Leverage overrides:")
    for sym, lev in sorted(LEVERAGE_OVERRIDES.items()):
        print(f"  {sym}: {lev}x")
