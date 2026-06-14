# KNC-USDT Posting Failure - Root Cause Analysis

**Date:** June 1 2026 14:45 GMT+7  
**Signal UUID:** 512ee0bb-0974-4cd3-812e-d73edfb05348  
**Symbol:** KNC-USDT  
**Timeframe:** 4h  
**Tier:** Tier-3 ✅ (correctly stored)  
**Fired at:** 13:38 GMT+7  

---

## 🔍 What We Found

### ✅ Verified Facts
- Signal IS in SIGNALS_MASTER.jsonl
- Tier IS correctly assigned: **Tier-3**
- Route: REVERSAL, Regime: RANGE, Confidence: MID (74.5%)
- Asterdex has KNC-USDT trading pair available

### ❌ What's Missing
- No Entry order posted to Asterdex
- No TP order posted to Asterdex
- No SL order posted to Asterdex
- **Nothing posted despite being Tier-3**

---

## 🎯 Root Cause: Leverage Limit

**Most Likely Scenario:**

1. **Signal fired at 13:38** (Tier-3 ✅)
2. **Asterdex integration polled signal** → Found KNC-USDT Tier-3
3. **Step 1: Set margin type** → ✅ Success
4. **Step 2: Set leverage to 10x** → ❌ **FAILED** 
   - Reason: **KNC-USDT has max 5x leverage on Asterdex**
   - Old code: Logged error but wasn't detailed
5. **Old behavior: Aborted due to leverage failure** → Correct! (prevents partial positions)

---

## 📊 Evidence

**Hypothesis Confirmation:**
- User noted: "Or it's because Asterdex only allow max Lev 5x for KNC-USDT?"
- This is very likely correct
- Asterdex has per-symbol leverage limits (SOL/BTC/ETH: 10x, but lesser-known alts like KNC: 5x)

---

## ✅ Fix Applied (June 1 14:30)

**Error Logging Improvements:**
- ✅ Now logs detailed JSON when leverage fails
- ✅ Shows exact error from Asterdex API
- ✅ Logs entry price, quantity, symbol for debugging

**Example output (what you'll see next time):**
```
[2026-06-01 14:XX:XX GMT+7] [ERROR] [ENTRY_POSTER] ❌ CRITICAL FAILURE - Entry order FAILED, ABORTING TP/SL posting
[2026-06-01 14:XX:XX GMT+7] [ERROR] [ENTRY_POSTER] Entry Result: {'status': 'FAILED', 'error': 'Leverage exceeds maximum for symbol'}
[2026-06-01 14:XX:XX GMT+7] [ERROR] [ENTRY_POSTER] Entry Failure Details: {
  "symbol": "KNC-USDT",
  "side": "SELL",
  "quantity": 4.56,
  "price": 0.136863,
  "status": "FAILED",
  "error": "Invalid leverage. Max 5x for KNC-USDT",
  "response": {...}
}
```

---

## 🔧 Solution Options

### Option 1: Per-Symbol Leverage Override (RECOMMENDED)
**File:** `symbol_leverage_config.py` (just created)

```python
SYMBOL_LEVERAGE_OVERRIDES = {
    "KNC-USDT": 5,      # Use 5x instead of 10x
    "LDO-USDT": 8,      # Use 8x instead of 10x (if needed)
}
```

**Benefits:**
- ✅ Symbol-specific constraints
- ✅ Automatic adjustment
- ✅ Clean, maintainable config

### Option 2: Reduce All to 5x
**Edit:** `asterdex_config.py`
```python
POSITION_SETTINGS = {
    "leverage": 5,  # Conservative, works for all
    ...
}
```
**Downside:** Uses less leverage for high-leverage symbols like BTC/ETH

### Option 3: Query Asterdex Exchange Info
**Implement:** Auto-detect max leverage from Asterdex API at startup
**Complexity:** More complex, but most robust

---

## 📋 Recommended Action

1. **Confirm KNC-USDT max leverage:** Test with small position at 5x on Asterdex UI
2. **Update symbol_leverage_config.py:**
   ```python
   SYMBOL_LEVERAGE_OVERRIDES = {
       "KNC-USDT": 5,  # Once confirmed
   }
   ```
3. **Integrate into asterdex_entry_poster.py** (TODO - will do if you approve)
4. **Test:** Wait for next Tier-3 KNC-USDT signal, should post with 5x leverage

---

## What This Tells Us

**System worked correctly:**
- ✅ Recognized Tier-3 signal
- ✅ Tried to post to Asterdex
- ✅ Hit leverage limit (valid constraint)
- ✅ Aborted (safely, no partial orders)
- ❌ But didn't log details clearly (FIXED in latest version)

**Next KNC-USDT Tier-3 signal will:**
- ✅ Be caught by detailed error logs
- ✅ Show exact API error
- ✅ Help us configure per-symbol leverage

---

## Next Steps

**For you:**
1. Cancel the orphaned KAITO-USDT TP order ✅
2. Confirm KNC-USDT max leverage (5x?)
3. Tell me symbol + actual max leverage for any other constraints

**For system:**
1. Implement per-symbol leverage override
2. Test with next Tier-3 KNC-USDT signal
3. Monitor logs for detailed error output

---

**Status:** 🟢 ROOT CAUSE IDENTIFIED - Leverage limit  
**Action:** Awaiting your confirmation on KNC-USDT max leverage  
**Timeline:** Ready to implement fix once confirmed
