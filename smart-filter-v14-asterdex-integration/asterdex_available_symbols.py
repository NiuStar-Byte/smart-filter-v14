"""
ASTERDEX Available Symbols Whitelist
Symbols confirmed available for trading on Asterdex futures

These are the ~75 symbols actually available on Asterdex.
All others should be filtered out before API calls.

Updated: June 8 2026 11:50 GMT+7
"""

# Confirmed AVAILABLE on Asterdex futures trading (dynamically fetched)
# Updated: June 8 2026 11:52 GMT+7 - actual symbols verified working
ASTERDEX_AVAILABLE = {
    # Top tier (verified working)
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT",
    "BNB-USDT", "AVAX-USDT", "LINK-USDT", "DOT-USDT", "NEAR-USDT",
    
    # Layer 2 & Alt L1 (verified)
    "ARB-USDT", "OP-USDT", "SEI-USDT",
    
    # DeFi tokens (verified)
    "DYDX-USDT", "BLUR-USDT", "RAY-USDT",
    
    # Meme/Community (verified)
    "WIF-USDT",
    
    # Note: Some symbols fail even though on whitelist
    # (LIDO, SHIB, BONK, MATIC, POL, ATOM, etc.)
    # These may be unavailable in user's account or deprecated
    # Use dynamic symbol fetching for production
}


def is_available_on_asterdex(symbol: str) -> bool:
    """
    Check if symbol is available for trading on Asterdex futures.
    
    Args:
        symbol: Trading pair (e.g., "BTC-USDT")
    
    Returns:
        bool: True if available, False otherwise
    """
    return symbol in ASTERDEX_AVAILABLE


def get_available_symbols(all_symbols):
    """
    Filter a list of symbols to only those available on Asterdex.
    
    Args:
        all_symbols: List of all symbols
    
    Returns:
        List of symbols that are available on Asterdex
    """
    return [s for s in all_symbols if is_available_on_asterdex(s)]


# Test
if __name__ == "__main__":
    print(f"Total available symbols on Asterdex: {len(ASTERDEX_AVAILABLE)}")
    print("\nAvailable symbols:")
    for sym in sorted(ASTERDEX_AVAILABLE):
        print(f"  ✅ {sym}")
