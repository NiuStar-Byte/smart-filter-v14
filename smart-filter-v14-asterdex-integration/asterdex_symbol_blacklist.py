"""
Asterdex Symbol Blacklist
Symbols that are NOT available on Asterdex futures trading

These symbols will be rejected BEFORE attempting to post orders.
Prevents wasted API calls and clear error messages.
"""

# Confirmed unavailable symbols on Asterdex (futures trading)
# Verified June 12 2026 11:44 GMT+7 via Asterdex UI
ASTERDEX_BLACKLIST = {
    "ACTSOL-USDT",      # Not available
    "ZKJ-USDT",         # Not available
    "CROSS-USDT",       # Not available
    "ATH-USDT",         # Not available
    "MAGIC-USDT",       # Not available
    "CVX-USDT",         # Not available
    "ICNT-USDT",        # Not available
    "VINE-USDT",        # Not available
    "LQTY-USDT",        # Not available
    "YFI-USDT",         # Not available
    "ORDER-USDT",       # Not available
    "AIN-USDT",         # Not available
}

def is_available_on_asterdex(symbol: str) -> tuple:
    """
    Check if symbol is available for trading on Asterdex futures.
    
    Args:
        symbol: Trading pair (e.g., "BTC-USDT", "ACTSOL-USDT", "XAU-USDT" mapped)
    
    Returns:
        (is_available: bool, reason: str if unavailable)
    """
    # Normalize symbol for comparison
    symbol_check = symbol.upper().replace('-USDT', '') + '-USDT' if '-USDT' not in symbol else symbol.upper()
    
    if symbol_check in ASTERDEX_BLACKLIST:
        return (
            False,
            f"❌ {symbol} is NOT available on Asterdex futures trading (blacklisted)"
        )
    
    return (True, "✅ Symbol available")


# Test
if __name__ == "__main__":
    test_symbols = ["BTC-USDT", "ACTSOL-USDT", "ETH-USDT", "SOL-USDT"]
    
    print("Testing Asterdex symbol availability:\n")
    for sym in test_symbols:
        available, reason = is_available_on_asterdex(sym)
        status = "✅" if available else "❌"
        print(f"{status} {sym:15} → {reason}")
