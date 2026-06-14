# ASTERDEX SYMBOL CONFIGURATION

**Status:** ✅ **Current as of June 8 2026 21:08 GMT+7**  
**Source:** Direct observation from Asterdex UI by user  
**Configuration File:** `asterdex_symbol_config.py`

---

## 📋 Configuration Summary

| Category | Count | Details |
|----------|-------|---------|
| **Unavailable Symbols** | 5 | Don't exist on Asterdex (skip posting) |
| **Symbol Mappings** | 1 | Name differs between our system and Asterdex |
| **Leverage Overrides** | 5 | Require leverage other than 10x default |

---

## ❌ Unavailable Symbols (Skip Posting)

These symbols **do NOT exist** on Asterdex. When posting, skip these entirely.

```python
UNAVAILABLE_SYMBOLS = {
    'ACTSOL-USDT',
    'ZKJ-USDT',
    'CROSS-USDT',
    'ATH-USDT',
    'MAGIC-USDT',
}
```

**Usage in asterdex_entry_poster.py:**
```python
from asterdex_symbol_config import is_available_in_asterdex

if not is_available_in_asterdex(symbol):
    logger.warning(f"Symbol {symbol} not available in Asterdex, skipping")
    continue
```

---

## 🔄 Symbol Mapping (Rename for Asterdex)

Our system uses different symbol names than Asterdex for some tokens.

```python
SYMBOL_MAPPING = {
    'XAUT-USDT': 'XAU-USDT',  # Our: XAUT-USDT → Asterdex: XAU-USDT
}
```

**Usage in asterdex_entry_poster.py:**
```python
from asterdex_symbol_config import get_asterdex_symbol

asterdex_symbol = get_asterdex_symbol(our_symbol)
# Example: get_asterdex_symbol('XAUT-USDT') → 'XAU-USDT'
```

---

## ⚙️ Leverage Overrides

Most symbols use **10x leverage** (default). These symbols require different leverage:

```python
LEVERAGE_OVERRIDES = {
    'ALGO-USDT': 5,     # New (June 8)
    'AVNT-USDT': 5,     # New (June 8)
    'BERA-USDT': 5,     # Existing
    'FUN-USDT': 2,      # Existing
    'KNC-USDT': 5,      # Existing
}
```

**Default:** 10x for all other symbols

**Usage in asterdex_entry_poster.py:**
```python
from asterdex_symbol_config import get_leverage

leverage = get_leverage(symbol)
# Example: get_leverage('ALGO-USDT') → 5
# Example: get_leverage('BTC-USDT') → 10 (default)
```

---

## 🔧 Integration Guide

### For asterdex_entry_poster.py

**Step 1: Import configuration**
```python
from asterdex_symbol_config import (
    is_available_in_asterdex,
    get_asterdex_symbol,
    get_leverage,
    validate_symbol_for_posting,
)
```

**Step 2: Validate before posting**
```python
# Method A: Simple check
if not is_available_in_asterdex(symbol):
    logger.warning(f"Symbol {symbol} unavailable")
    continue

# Method B: Get validated symbol (returns None if unavailable)
asterdex_symbol = validate_symbol_for_posting(symbol)
if not asterdex_symbol:
    logger.warning(f"Symbol {symbol} unavailable")
    continue
```

**Step 3: Use mapped symbol for API calls**
```python
# Always use the mapped symbol for Asterdex API
symbol_for_api = get_asterdex_symbol(signal_symbol)
# Example: 'XAUT-USDT' becomes 'XAU-USDT' for API call
```

**Step 4: Set correct leverage**
```python
leverage = get_leverage(signal_symbol)
# Set margin type and leverage before posting order
```

---

## 📝 Update Protocol

When user observes new symbols/changes in Asterdex:

1. **User reports changes** (like June 8 21:08 update)
2. **Update asterdex_symbol_config.py:**
   - Add/remove from UNAVAILABLE_SYMBOLS
   - Add/update SYMBOL_MAPPING
   - Add/update LEVERAGE_OVERRIDES
3. **Update MEMORY.md with timestamp**
4. **Commit to GitHub** with message: `"🔧 ASTERDEX CONFIG: Update symbol mappings (Jun X HH:MM GMT+7)"`
5. **Restart asterdex_entry_poster.py** to load new config

---

## ✅ Implementation Status

- [x] Configuration file created: `asterdex_symbol_config.py`
- [x] Functions documented with examples
- [x] Memory updated with current info
- [ ] Integration into asterdex_entry_poster.py (pending next update)
- [ ] Testing with live posting (pending next update)

---

## 📞 Quick Reference

```python
# Check if symbol is available
is_available_in_asterdex('ACTSOL-USDT')  # → False
is_available_in_asterdex('BTC-USDT')     # → True

# Get Asterdex symbol name
get_asterdex_symbol('XAUT-USDT')  # → 'XAU-USDT'
get_asterdex_symbol('BTC-USDT')   # → 'BTC-USDT' (no mapping)

# Get leverage
get_leverage('ALGO-USDT')  # → 5
get_leverage('BTC-USDT')   # → 10 (default)

# All-in-one validation
validate_symbol_for_posting('XAUT-USDT')  # → 'XAU-USDT' (valid)
validate_symbol_for_posting('ACTSOL-USDT')  # → None (unavailable)
```

---

**Created:** June 8 2026 21:08 GMT+7  
**Last Updated:** June 8 2026 21:08 GMT+7  
**Status:** Ready for integration into asterdex_entry_poster.py
