"""
Asterdex Symbol Name Mapper
Maps signal symbols to Asterdex actual symbol names

Some symbols are known by different names on Asterdex vs our signal system
"""

# Symbol name mappings: signal_symbol → asterdex_symbol
# Use only when Asterdex uses a different name for the same asset
SYMBOL_MAPPINGS = {
    "XAUT-USDT": "XAU-USDT",     # Gold token: our system uses XAUT, Asterdex uses XAU (June 12 11:44 GMT+7)
    "PEPE-USDT": "1000PEPE-USDT",  # Ultra micro-cap: Asterdex displays as 1000PEPEUSDT with 1000x scaled prices (June 12 23:26 GMT+7)
    "BONK-USDT": "1000BONK-USDT",  # Ultra micro-cap: Asterdex displays as 1000BONKUSDT with 1000x scaled prices (June 12 23:26 GMT+7)
}

def get_asterdex_symbol(signal_symbol: str) -> str:
    """
    Convert signal symbol name to Asterdex symbol name.
    
    Args:
        signal_symbol: Symbol from our signal system (e.g., "XAUT-USDT", "BTC-USDT")
    
    Returns:
        Symbol name for use on Asterdex (e.g., "XAU-USDT", "BTC-USDT")
    """
    return SYMBOL_MAPPINGS.get(signal_symbol, signal_symbol)


def get_signal_symbol(asterdex_symbol: str) -> str:
    """
    Reverse mapping: convert Asterdex symbol back to signal symbol.
    Useful for tracking/logging.
    
    Args:
        asterdex_symbol: Symbol from Asterdex (e.g., "XAU-USDT")
    
    Returns:
        Symbol name in our signal system (e.g., "XAUT-USDT")
    """
    # Reverse the mapping
    for signal_sym, asterdex_sym in SYMBOL_MAPPINGS.items():
        if asterdex_sym == asterdex_symbol:
            return signal_sym
    
    # If not in mappings, return as-is
    return asterdex_symbol


# Test
if __name__ == "__main__":
    test_symbols = ["XAUT-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT"]
    
    print("=== Asterdex Symbol Mapper Test ===\n")
    for sym in test_symbols:
        asterdex_sym = get_asterdex_symbol(sym)
        back_to_signal = get_signal_symbol(asterdex_sym)
        mapping = f" → {asterdex_sym}" if sym != asterdex_sym else " (no mapping)"
        print(f"Signal: {sym:15} → Asterdex: {asterdex_sym:15} {mapping}")
