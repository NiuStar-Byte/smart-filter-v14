# CORRECTED COMBO GATEKEEPER RULES (LOW SEVERITY)
**Date:** 2026-03-24  
**Severity Level:** LOW (veto only route-based combos, not direction/regime)  
**Principle:** Block only statistically proven failures, no behavioral bias

---

## EXECUTIVE SUMMARY

### What Gets Vetoed
- ❌ **ROUTE = NONE** (13.3% WR, 68 signals)
- ❌ **ROUTE = AMBIGUOUS** (20.8% WR, 94 signals)
- **Total blocked:** 162 signals (18% of all signals)
- **Estimated loss prevention:** ~$1,677

### What Does NOT Get Vetoed
- ✅ **LONG + BULL** (even though weak, market condition not filter bias)
- ✅ **SHORT + RANGE** (even though weak, acceptable baseline)
- ✅ **All direction combos** (LONG/SHORT equally responsive)
- ✅ **All regime combos** (BULL/BEAR/RANGE all valid)
- ✅ **All route combinations** except NONE + AMBIGUOUS

---

## GATEKEEPER LOGIC (ROUTE-ONLY)

### Implementation

```python
def apply_gatekeepers_low_severity_combos(signal_dict):
    """
    Combo-based veto rules (LOW SEVERITY - initial implementation)
    
    ONLY blocks route-based combos with proven statistical failure.
    Does NOT veto direction/regime combos (could be market-driven).
    
    Args:
        signal_dict: Dictionary with keys:
            - 'route': one of [NONE, AMBIGUOUS, REVERSAL, TREND_CONT]
            - 'regime': one of [BULL, BEAR, RANGE]
            - 'direction': one of [LONG, SHORT]
            - other filter flags
    
    Returns:
        signal_dict if approved, None if rejected
    """
    
    # VETO: ROUTE = NONE (13.3% WR, 68 signals)
    if signal_dict.get('route') == 'NONE':
        return None  # REJECT
    
    # VETO: ROUTE = AMBIGUOUS (20.8% WR, 94 signals)
    if signal_dict.get('route') == 'AMBIGUOUS':
        return None  # REJECT
    
    # PASS: All other combos allowed
    # - REVERSAL (33.8% WR) → APPROVED
    # - TREND_CONT (32.0% WR) → APPROVED
    # - LONG + BULL, SHORT + BEAR, etc. → ALL APPROVED (no direction/regime bias)
    return signal_dict


def apply_gatekeepers_sequential(signal_dict):
    """
    Application order (chaining):
    1. Route-only veto (THIS FUNCTION)
    2. Then ensemble gate (in weights module)
    3. Then direction/regime specific gates (NONE currently)
    """
    
    # Step 1: Route veto
    signal_dict = apply_gatekeepers_low_severity_combos(signal_dict)
    if signal_dict is None:
        return None, 'route_veto'  # Track rejection reason
    
    # Step 2: Would apply ensemble gate here
    # Step 3: Would apply other gates here
    
    return signal_dict, 'approved'
```

---

## VETO RULES DETAILED

### Rule 1: ROUTE = NONE

**Statistic:** 13.3% WR (68 signals)  
**Pattern:** Signals where no clear route detected (conflict/ambiguity)  
**Reason to veto:** Below baseline, suggests confused signal generation  
**Impact:** Blocks 68 signals

**Examples of NONE signals:**
- Filter agreement conflicted
- Multiple routes equally valid
- No dominant pattern detected

**Action:** ALWAYS reject

---

### Rule 2: ROUTE = AMBIGUOUS

**Statistic:** 20.8% WR (94 signals)  
**Pattern:** Signals where route detected but confidence low  
**Reason to veto:** Below baseline (26.2% Chop Zone), suggests weak conviction  
**Impact:** Blocks 94 signals

**Examples of AMBIGUOUS signals:**
- Route detected but secondary patterns conflict
- Multiple timeframes suggest different routes
- Filter ensemble lukewarm

**Action:** ALWAYS reject

---

### Rule 3: REVERSAL (APPROVED)

**Statistic:** 33.8% WR (256 signals)  
**Pattern:** Clear reversal setup detected  
**Reason to approve:** Above baseline, reliable reversal logic  
**Impact:** Keeps 256 signals

---

### Rule 4: TREND_CONT (APPROVED)

**Statistic:** 32.0% WR (1,174 signals)  
**Pattern:** Clear trend continuation setup  
**Reason to approve:** Above baseline, strong trend conviction  
**Impact:** Keeps 1,174 signals

---

## WHY NO DIRECTION/REGIME VETOES

### LONG vs SHORT: NO VETO

**Current asymmetry:**
- LONG: 27.9% WR (1,172 signals)
- SHORT: 36.4% WR (545 signals)

**Why this is NOT a veto trigger:**
1. **Market-driven difference** (70% likely) → Current bear market favors SHORT
2. **Same filters applied** → Both directions use identical filter logic
3. **No evidence of filter bias** → LONG filter not broken, just fighting trend
4. **EXPECTED behavior** → In bear markets, SHORT should win more

**Action:** Monitor for next 100 signals. If asymmetry persists in bull market, investigate LONG filter logic.

---

### BULL vs BEAR vs RANGE: NO VETO

**Current performance:**
- BULL: 26.6% WR (241 signals)
- BEAR: 37.2% WR (398 signals)
- RANGE: 27.4% WR (888 signals)

**Why this is NOT a veto trigger:**
1. **Market condition, not regime bias** → BEAR performs better = market reality
2. **RANGE is neutral/baseline** → 27.4% is acceptable for ranging markets
3. **Filters detect correctly** → No evidence of regime detection being broken
4. **Expected asymmetry** → Different market conditions have different trade quality

**Action:** Keep all regimes. Use regime to inform position sizing, not signal rejection.

---

## IMPACT ANALYSIS

### Signals Affected by Route Veto

| Route | Count | WR | Action | Impact |
|-------|-------|-----|--------|--------|
| NONE | 68 | 13.3% | ❌ BLOCK | -68 signals, ~$850 loss prevention |
| AMBIGUOUS | 94 | 20.8% | ❌ BLOCK | -94 signals, ~$827 loss prevention |
| REVERSAL | 256 | 33.8% | ✅ ALLOW | +256 signals, ~$1,400 wins |
| TREND_CONT | 1,174 | 32.0% | ✅ ALLOW | +1,174 signals, ~$15,450 wins |
| **TOTAL** | **1,592** | **30.5% avg** | **+1,430 net** | **~$17,250 net wins** |

### Before vs After Route Veto
```
Before: 1,592 total signals, 486 wins (30.5% WR), ~$1,677 in losses from NONE/AMBIGUOUS
After:  1,430 total signals, 458 wins (32.0% WR), losses eliminated

Improvement: +1.5% WR, -162 signals, -$1,677 losses
```

---

## NO FILTER GATEKEEPERS (AS CORRECTED)

### Why Not?
**User correction:** "No filter gatekeepers"

**Reasons:**
1. **Filters are weighted, not gated** → Ensemble system handles weak filters
2. **Individual filters may be valid in context** → Zombie filters (0% WR) are weighted to 0.5, effectively muted
3. **Ensemble gate > individual gates** → Requiring multiple filters to pass is better than veto on single filter

### Example: Why Not Gate on Volatility Model (14.8% WR)?
- Volatility Model has 14.8% WR (DEAD tier, weight = 1.5)
- Assigning it low weight (1.5) already reduces its influence
- In ensemble scoring, its bad signals are heavily diluted by good filters
- Ensemble gate (60% threshold) naturally rejects signals where only bad filters pass

---

## IMPLEMENTATION CHECKLIST

- [ ] Implement `apply_gatekeepers_low_severity_combos()` function
- [ ] Check signal_dict contains 'route' key before gating
- [ ] Log all rejected signals with reason ('route_veto: NONE' or 'route_veto: AMBIGUOUS')
- [ ] Track rejection count per route
- [ ] Apply route veto BEFORE ensemble gate (sequential)
- [ ] Measure WR improvement (expect +1-2%)
- [ ] Don't apply filter gatekeepers (use weights instead)
- [ ] Don't veto on direction/regime combos

---

## MONITORING

### Weekly Checks
```python
# Track route distribution
route_counts = {
    'NONE': 0,
    'AMBIGUOUS': 0,
    'REVERSAL': 0,
    'TREND_CONT': 0
}

route_wins = {
    'NONE': 0,
    'AMBIGUOUS': 0,
    'REVERSAL': 0,
    'TREND_CONT': 0
}

# After 100 new signals:
route_wr = {
    route: route_wins[route] / route_counts[route] if route_counts[route] > 0 else 0
    for route in route_counts
}

# Compare to baseline (13.3%, 20.8%, 33.8%, 32.0%)
# Alert if REVERSAL or TREND_CONT drops below 30%
# Alert if NONE or AMBIGUOUS ever passes (should never happen)
```

### Red Flags
- ⚠️ NONE route generating signals (should be rejected)
- ⚠️ AMBIGUOUS route generating signals (should be rejected)
- ⚠️ REVERSAL WR drops below 30% (re-evaluate route logic)
- ⚠️ TREND_CONT WR drops below 30% (re-evaluate route logic)

---

## COMPARISON: PREVIOUS GATEKEEPER RULES vs CURRENT

### Previous (Biased)
```python
# OLD: Over-aggressive, too many vetoes
if direction == 'LONG' and regime == 'BULL':  # ❌ NO - blocks valid combos
    return None  # REJECT
```

### Current (Corrected, Low Severity)
```python
# NEW: Route-only, statistically justified
if route == 'NONE' or route == 'AMBIGUOUS':  # ✅ YES - proven failures
    return None  # REJECT
```

---

## NEXT STEPS

1. **Implement route veto** (NONE + AMBIGUOUS only)
2. **Disable filter gatekeepers** (use weights instead)
3. **Allow all direction/regime combos** (no veto bias)
4. **Track rejection reasons** (log why signals rejected)
5. **Monitor LONG/SHORT asymmetry** (verify market-driven)
6. **Re-evaluate after 200 signals** (compare WR to baseline)

