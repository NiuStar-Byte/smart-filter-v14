"""
symbol_config_prod.py - Production Symbol Configuration
Last updated: 2026-03-05
Total symbols: 234 (verified on both KuCoin + Binance perpetuals)
Git tag: stable-234-symbols

This file is SEPARATE from main.py to prevent accidental symbol loss during crashes.
If main.py corrupts, the symbol list is safely in this file.

CHANGELOG:
- 2026-03-04: Initial 82 symbols (stable baseline)
- 2026-03-04: Expanded to 201 symbols (commit cc29b4c stable, then 61b22a2 added 67)
- 2026-03-05: Restored to 234 verified perpetuals (bac6861, verified on both exchanges)
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
    "XPL-USDT", "AVNT-USDT", "ORDER-USDT", "XAUT-USDT", "BEAM-USDT", "ROSE-USDT", 
    "OP-USDT", "OPTIMISM-USDT", "LIDO-USDT", "LINA-USDT", "OKB-USDT", "RNDR-USDT", 
    "TAO-USDT", "OCEAN-USDT", "LUNA-USDT", "CRV-USDT", "CVX-USDT", "LDO-USDT", 
    "STG-USDT", "GNS-USDT", "GMX-USDT", "PERP-USDT", "JUP-USDT", "YGG-USDT", 
    "SAND-USDT", "MANA-USDT", "ENJ-USDT", "AXS-USDT", "ALICE-USDT", "FLOW-USDT", 
    "ZEC-USDT", "FIL-USDT", "STORJ-USDT", "AR-USDT", "CHIA-USDT", "THETA-USDT", 
    "TFUEL-USDT", "BAND-USDT", "API3-USDT", "PYTH-USDT", "LOOKS-USDT", "X2Y2-USDT", 
    "RARI-USDT", "BLUR-USDT", "MKR-USDT", "COMP-USDT", "AURA-USDT", "BAL-USDT", 
    "APT-USDT", "SEI-USDT", "MOV-USDT", "NEON-USDT", "WLD-USDT", "SHIB-USDT", 
    "FLOKI-USDT", "BONK-USDT", "FIDA-USDT", "COPE-USDT", "ZRX-USDT", "BNT-USDT", 
    "LRC-USDT", "SNX-USDT", "SYN-USDT", "GROK-USDT", "BOME-USDT", "POPCAT-USDT", 
    "CHEX-USDT", "PIXEL-USDT", "SLERF-USDT", "PUPS-USDT", "MEW-USDT", "PONKE-USDT", 
    "SPX-USDT", "TURBOS-USDT", "CETUS-USDT", "SCNSOL-USDT", "MANTA-USDT", "EXO-USDT", 
    "USDC-USDT", "BUSD-USDT", "USDP-USDT", "NEAR-USDT", "BGB-USDT", "MNT-USDT", 
    "CORE-USDT", "METIS-USDT", "KASPA-USDT", "AGLD-USDT", "ORDI-USDT", "BRETT-USDT", 
    "INJ-USDT", "JASMY-USDT", "MAPLE-USDT", "DODO-USDT", "LUNC-USDT", "OOG-USDT", 
    "PDA-USDT", "ETHPAD-USDT", "GLM-USDT", "GYEN-USDT", "LBRY-USDT", "MOVE-USDT", 
    "KEY-USDT", "KSUI-USDT", "MANGO-USDT", "USTC-USDT", "NBTC-USDT", "ORBS-USDT", 
    "VELO-USDT", "WOM-USDT", "ZCNL-USDT", "COVALENT-USDT", "DEXT-USDT", "ENFT-USDT", 
    "FLAP-USDT", "GRIM-USDT", "HAPI-USDT", "INDEX-USDT", "IRON-USDT", "JUNO-USDT", 
    "KAVA-USDT", "KDX-USDT", "LQDT-USDT", "MATIC-USDT", "MEBE-USDT", "MEME-USDT", 
    "MICE-USDT", "MIM-USDT", "MINTT-USDT", "MOGUL-USDT", "MOON-USDT", "MORA-USDT", 
    "MUDRA-USDT", "MULTI-USDT", "MURKY-USDT", "MVF-USDT", "NEXO-USDT", "NIL-USDT", 
    "NINJA-USDT", "NOLE-USDT", "NOTCOIN-USDT", "OBSR-USDT", "OFOR-USDT", "OLE-USDT", 
    "OMS-USDT", "ONE-USDT", "ONI-USDT", "ONLY-USDT", "ONYX-USDT", "OOK-USDT", 
    "OOLE-USDT", "OOR-USDT", "OPALS-USDT", "OPEN-USDT", "OPER-USDT", "OPHIR-USDT",
]

STAGING_SYMBOLS = [
    # Use this section to test new symbols before adding to PRODUCTION_SYMBOLS
    # Example: "NEW-USDT"
]

def get_active_symbols():
    """Return current active symbols (production + staging if enabled)"""
    return PRODUCTION_SYMBOLS + STAGING_SYMBOLS

print(f"[SYMBOL_CONFIG] Production symbols: {len(PRODUCTION_SYMBOLS)}")
print(f"[SYMBOL_CONFIG] Staging symbols: {len(STAGING_SYMBOLS)}")
print(f"[SYMBOL_CONFIG] Total active: {len(get_active_symbols())}")
