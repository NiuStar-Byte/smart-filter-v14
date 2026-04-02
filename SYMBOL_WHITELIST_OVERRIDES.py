"""
SYMBOL WHITELIST OVERRIDES - Dynamic Release from Blacklist
Created: 2026-04-02 09:18 GMT+7
Status: ACTIVE

Purpose:
  Override blacklist decisions in real-time.
  Any symbol in this whitelist will be ALLOWED to trade,
  regardless of blacklist status.

Mechanism:
  Checked BEFORE blacklist in apply_blacklist_filter()
  If symbol is here → ALLOW (return False)
  If symbol is NOT here → Check blacklist

Usage:
  To release a blacklisted symbol:
    WHITELIST_OVERRIDES = {
        'SYMBOL-USDT': 'reason for release',
    }
  
  No restart needed - takes effect immediately on next signal.

Initial State:
  WHITELIST_OVERRIDES = {} (empty, all blacklist active)
"""

WHITELIST_OVERRIDES = {
    # Add symbols here to RELEASE from blacklist
    # Format: 'SYMBOL-USDT': 'Reason for override (date, condition, etc.)'
    
    # Example (commented out):
    # 'VOXEL-USDT': 'Testing recovery - re-enabled 2026-04-05 per manual review',
    # 'EPT-USDT': 'User-requested test - monitor closely',
}


def is_symbol_whitelisted(symbol: str) -> bool:
    """
    Check if symbol is whitelisted (override blacklist).
    
    Args:
        symbol: Symbol string (e.g., 'VOXEL-USDT')
    
    Returns:
        True if symbol is in whitelist (ALLOW trading)
        False if symbol is NOT in whitelist (check blacklist)
    """
    return symbol in WHITELIST_OVERRIDES


# ============================================================================
# STATISTICS
# ============================================================================
WHITELIST_COUNT = len(WHITELIST_OVERRIDES)
BLACKLIST_COUNT = 21  # From SYMBOL_BLACKLIST.py

print(f"""
WHITELIST STATUS
═════════════════════════════════════════════════════════════════
Whitelisted (override):     {WHITELIST_COUNT}
Blacklisted (excluded):     {BLACKLIST_COUNT}
Active trading symbols:     {98 - BLACKLIST_COUNT + WHITELIST_COUNT}

Status: READY - Whitelist is empty, all blacklist active
═════════════════════════════════════════════════════════════════
""")
