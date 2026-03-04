"""
symbol_config_prod.py - Production Symbol Configuration
Last updated: 2026-03-05
Total symbols: 82 (original stable baseline)
Git tag: stable-82-symbols-60s-working

Reverted to 82-symbol original set that ran reliably with 60s CYCLE_SLEEP.
This config produced stable <50s cycles and reliable JSONL updates.

CHANGELOG:
- 2026-03-04: Started with 82 symbols (stable, 60s cycle sleep works)
- 2026-03-04: Expanded to 201, then 234 (daemon crashes due to API bottleneck)
- 2026-03-05: Scaled to 153 (928s cycles - too slow even with 1000s sleep)
- 2026-03-05: Reverted to 82 (proven stable with 60s sleep)
"""

PRODUCTION_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "AVAX-USDT", 
    "XLM-USDT", "LINK-USDT", "POL-USDT", "BNB-USDT", "SKATE-USDT", "LA-USDT", 
    "SPK-USDT", "ZKJ-USDT", "IP-USDT", "AERO-USDT", "BMT-USDT", "LQTY-USDT", 
    "X-USDT", "RAY-USDT", "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", 
    "FUN-USDT", "CROSS-USDT", "KNC-USDT", "AIN-USDT", "ARK-USDT", "PORTAL-USDT", 
    "ICNT-USDT", "OMNI-USDT", "PARTI-USDT", "VINE-USDT", "ZORA-USDT", "DUCK-USDT", 
    "AUCTION-USDT", "ROAM-USDT", "FUEL-USDT", "TUT-USDT", "VOXEL-USDT", "ALU-USDT", 
    "TURBO-USDT", "PROMPT-USDT", "HIPPO-USDT", "DOGE-USDT", "ALGO-USDT", "DOT-USDT", 
    "NEWT-USDT", "SAHARA-USDT", "PEPE-USDT", "ERA-USDT", "PENGU-USDT", "CFX-USDT", 
    "ENA-USDT", "SUI-USDT", "EIGEN-USDT", "UNI-USDT", "HYPE-USDT", "TON-USDT", 
    "KAS-USDT", "HBAR-USDT", "ONDO-USDT", "VIRTUAL-USDT", "AAVE-USDT", "GALA-USDT", 
    "PUMP-USDT", "WIF-USDT", "BERA-USDT", "DYDX-USDT", "KAITO-USDT", "ARKM-USDT", 
    "ATH-USDT", "NMR-USDT", "ARB-USDT", "WLFI-USDT", "BIO-USDT", "ASTER-USDT", 
    "XPL-USDT", "AVNT-USDT", "ORDER-USDT", "XAUT-USDT",
]

STAGING_SYMBOLS = [
    # When ready to expand, add symbols here first to test
    # e.g., "OP-USDT", "TAO-USDT", "RNDR-USDT"
]

def get_active_symbols():
    """Return current active symbols (production + staging if enabled)"""
    return PRODUCTION_SYMBOLS + STAGING_SYMBOLS

print(f"[SYMBOL_CONFIG] Production symbols: {len(PRODUCTION_SYMBOLS)}")
print(f"[SYMBOL_CONFIG] Staging symbols: {len(STAGING_SYMBOLS)}")
print(f"[SYMBOL_CONFIG] Total active: {len(get_active_symbols())}")
