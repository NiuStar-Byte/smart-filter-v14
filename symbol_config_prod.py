"""
symbol_config_prod.py - Production Symbol Configuration
Last updated: 2026-03-05
Total symbols: 20 (TOP 20 most liquid on KuCoin perpetuals)

Reduced from 82 to 20 due to API rate limiting bottleneck.
Fast cycle = more signals = better A/B testing data.

These are the absolute most liquid + fastest API response symbols.
"""

PRODUCTION_SYMBOLS = [
    # Top 20 most liquid KuCoin perpetuals (fastest API response)
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", 
    "AVAX-USDT", "BNB-USDT", "LINK-USDT", "DOGE-USDT", "POL-USDT",
    "SUI-USDT", "UNI-USDT", "AAVE-USDT", "SHIB-USDT", "PEPE-USDT",
    "ARB-USDT", "NEAR-USDT", "ONDO-USDT", "TAO-USDT", "SEI-USDT"
]

STAGING_SYMBOLS = [
    # Once 20 is stable for 24h with <60s cycles, expand from here:
    # "LIDO-USDT", "OCEAN-USDT", "RNDR-USDT", "GMX-USDT", "JUP-USDT",
    # "YGG-USDT", "ENJ-USDT", "FLOW-USDT", "FIL-USDT", "THETA-USDT"
]

def get_active_symbols():
    """Return current active symbols (production + staging if enabled)"""
    return PRODUCTION_SYMBOLS + STAGING_SYMBOLS

print(f"[SYMBOL_CONFIG] Production symbols: {len(PRODUCTION_SYMBOLS)}")
print(f"[SYMBOL_CONFIG] Staging symbols: {len(STAGING_SYMBOLS)}")
print(f"[SYMBOL_CONFIG] Total active: {len(get_active_symbols())}")
