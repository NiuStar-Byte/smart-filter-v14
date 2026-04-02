"""
SYMBOL CLASSIFIER - Group symbols by market tier
Used by tier_lookup to match 5D/6D combo patterns
"""

def classify_symbol(symbol: str) -> str:
    """Classify symbol into group (MAIN_BLOCKCHAIN, TOP_ALTS, MID_ALTS, LOW_ALTS)
    
    Uses exact symbol matching to align with pec_post_deployment_tracker.py
    """
    if not symbol:
        return "UNKNOWN"
    
    symbol = symbol.upper().replace('-USDT', '') + '-USDT'  # Normalize
    
    # MAIN_BLOCKCHAIN (10 symbols) - Core blockchain tokens + major exchanges
    main_blockchain = [
        'BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'XRP-USDT', 'ADA-USDT', 
        'AVAX-USDT', 'BNB-USDT', 'XLM-USDT', 'LINK-USDT', 'POL-USDT'
    ]
    if symbol in main_blockchain:
        return "MAIN_BLOCKCHAIN"
    
    # TOP_ALTS (4 symbols) - Premium tier alternatives
    top_alts = ['ZKJ-USDT', 'ROAM-USDT', 'XAUT-USDT', 'SAHARA-USDT']
    if symbol in top_alts:
        return "TOP_ALTS"
    
    # MID_ALTS (8 symbols) - Mid-tier alternatives
    mid_alts = [
        'XPL-USDT', 'DOT-USDT', 'FUEL-USDT', 'VIRTUAL-USDT', 
        'BERA-USDT', 'CROSS-USDT', 'FUN-USDT', 'ENA-USDT'
    ]
    if symbol in mid_alts:
        return "MID_ALTS"
    
    # LOW_ALTS (everything else) - Lower tier tokens
    return "LOW_ALTS"
