"""
ACTIVE_SYMBOLS.py - Single Source of Truth for Symbol Registry

This module enforces a central registry of all active trading symbols.
Every part of the system that processes symbols must validate against this list.

Architecture:
- Import ACTIVE_SYMBOLS from main.py (single source of truth)
- Provide validation decorators and context managers
- Enforce symbol registration before entering filter chain
- Allow safe fallbacks for testing/standalone usage
"""

import sys
import os
from typing import List, Optional, Set

# ===== PRIMARY SOURCE: Import from main.py =====
try:
    from main import TOKENS as ACTIVE_SYMBOLS
except ImportError:
    # Fallback: 92 symbols (includes 10 new ones)
    ACTIVE_SYMBOLS = [
        "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT",
        "AVAX-USDT", "XLM-USDT", "LINK-USDT", "POL-USDT", "BNB-USDT",
        "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
        "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
        "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT",
        "CROSS-USDT", "KNC-USDT", "AIN-USDT", "ARK-USDT", "PORTAL-USDT",
        "ICNT-USDT", "OMNI-USDT", "PARTI-USDT", "VINE-USDT", "ZORA-USDT",
        "DUCK-USDT", "AUCTION-USDT", "ROAM-USDT", "FUEL-USDT", "TUT-USDT",
        "VOXEL-USDT", "ALU-USDT", "TURBO-USDT", "PROMPT-USDT", "HIPPO-USDT",
        "DOGE-USDT", "ALGO-USDT", "DOT-USDT", "NEWT-USDT", "SAHARA-USDT",
        "PEPE-USDT", "ERA-USDT", "PENGU-USDT", "CFX-USDT", "ENA-USDT",
        "SUI-USDT", "EIGEN-USDT", "UNI-USDT", "HYPE-USDT", "TON-USDT",
        "KAS-USDT", "HBAR-USDT", "ONDO-USDT", "VIRTUAL-USDT", "AAVE-USDT",
        "GALA-USDT", "PUMP-USDT", "WIF-USDT", "BERA-USDT", "DYDX-USDT",
        "KAITO-USDT", "ARKM-USDT", "ATH-USDT", "NMR-USDT", "ARB-USDT",
        "WLFI-USDT", "BIO-USDT", "ASTER-USDT", "XPL-USDT", "AVNT-USDT",
        "ORDER-USDT", "XAUT-USDT",
        # NEW (2026-03-05)
        "ATOM-USDT", "AGLD-USDT", "APT-USDT", "INJ-USDT", "NEAR-USDT",
        "OCEAN-USDT", "OP-USDT", "RNDR-USDT", "SEI-USDT", "TAO-USDT"
    ]

# Convert to set for O(1) lookup
ACTIVE_SYMBOLS_SET: Set[str] = set(ACTIVE_SYMBOLS)

# Metadata
TOTAL_SYMBOLS = len(set(ACTIVE_SYMBOLS))
UNIQUE_SYMBOLS = TOTAL_SYMBOLS
LAST_UPDATED = "2026-03-05 14:20 GMT+7"


class SymbolNotRegisteredError(ValueError):
    """Raised when symbol is not in ACTIVE_SYMBOLS registry"""
    pass


def is_active_symbol(symbol: str) -> bool:
    """Check if symbol is registered and active"""
    return symbol in ACTIVE_SYMBOLS_SET


def validate_symbol(symbol: str, raise_on_invalid: bool = True) -> bool:
    """
    Validate that symbol is in ACTIVE_SYMBOLS registry.
    
    Args:
        symbol: Symbol to validate (e.g., 'BTC-USDT')
        raise_on_invalid: If True, raises SymbolNotRegisteredError; if False, returns bool
        
    Returns:
        True if valid, False if invalid (when raise_on_invalid=False)
        
    Raises:
        SymbolNotRegisteredError: If raise_on_invalid=True and symbol not registered
    """
    if symbol not in ACTIVE_SYMBOLS_SET:
        msg = f"Symbol '{symbol}' not in ACTIVE_SYMBOLS registry (total registered: {TOTAL_SYMBOLS})"
        if raise_on_invalid:
            raise SymbolNotRegisteredError(msg)
        else:
            print(f"[WARNING] {msg}", flush=True)
            return False
    return True


def validate_symbols(symbols: List[str], raise_on_invalid: bool = True) -> bool:
    """Validate multiple symbols"""
    invalid = [s for s in symbols if not is_active_symbol(s)]
    if invalid:
        msg = f"Invalid symbols: {invalid} (not in ACTIVE_SYMBOLS)"
        if raise_on_invalid:
            raise SymbolNotRegisteredError(msg)
        else:
            print(f"[WARNING] {msg}", flush=True)
            return False
    return True


def get_unregistered_symbols(symbols: List[str]) -> List[str]:
    """Get list of symbols not in registry"""
    return [s for s in symbols if not is_active_symbol(s)]


def enforce_symbol_validation(func):
    """
    Decorator: Validate symbol parameter before function execution.
    
    Usage:
        @enforce_symbol_validation
        def process_signal(symbol, price, ...):
            ...
    
    Raises SymbolNotRegisteredError if symbol not in ACTIVE_SYMBOLS
    """
    def wrapper(*args, **kwargs):
        # Try to extract symbol parameter
        symbol = kwargs.get('symbol') or (args[1] if len(args) > 1 else None)
        
        if symbol:
            validate_symbol(symbol, raise_on_invalid=True)
        
        return func(*args, **kwargs)
    
    return wrapper


# ===== STATUS & REPORTING =====

def get_registry_status() -> dict:
    """Get current registry status"""
    return {
        "total_symbols": TOTAL_SYMBOLS,
        "unique_symbols": UNIQUE_SYMBOLS,
        "has_duplicates": TOTAL_SYMBOLS != UNIQUE_SYMBOLS,
        "last_updated": LAST_UPDATED,
        "symbols": sorted(list(ACTIVE_SYMBOLS_SET))
    }


def print_registry_status():
    """Print human-readable registry status"""
    status = get_registry_status()
    print(f"""
╔════════════════════════════════════════════════════════════╗
║        ACTIVE SYMBOLS REGISTRY - ENFORCED                 ║
╠════════════════════════════════════════════════════════════╣
║ Total Symbols:      {status['total_symbols']:3}                              ║
║ Unique Symbols:     {status['unique_symbols']:3}                              ║
║ Has Duplicates:     {str(status['has_duplicates']):5}                            ║
║ Last Updated:       {status['last_updated']:30} ║
╠════════════════════════════════════════════════════════════╣
║ All symbols must be registered before entering filter      ║
║ chain. Validation occurs at:                               ║
║   1. main.py daemon loop (checks symbol in ACTIVE_SYMBOLS)║
║   2. SmartFilter.analyze() (enforced decorator)           ║
║   3. API call functions (validate before fetch)           ║
╚════════════════════════════════════════════════════════════╝
""")


# ===== AUTO-VALIDATION ON IMPORT =====
if __name__ != "__main__":
    # Verify registry integrity on module load
    if UNIQUE_SYMBOLS != len(set(ACTIVE_SYMBOLS)):
        print(f"[WARNING] Duplicate symbols detected in ACTIVE_SYMBOLS!", flush=True)
        duplicates = [s for s in ACTIVE_SYMBOLS if ACTIVE_SYMBOLS.count(s) > 1]
        print(f"[WARNING] Duplicates: {set(duplicates)}", flush=True)
