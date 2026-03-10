# 🎯 ROUTE & REGIME OPTIMIZATION PLAN

**Corrected Understanding:** MIN_SCORE=12 is independent. ROUTE/REGIME optimization happens via POST-FILTER secondary gates.

**Date:** 2026-03-10 10:56 GMT+7

---

## A. OPTIMIZE ROUTE & REGIME (Post-Filter Gates)

### What We're Building

A secondary gating layer AFTER MIN_SCORE filter, using ROUTE/REGIME combos to reject bad signals.

```python
# Location: smart_filter.py, after line 850 (after filters_ok decision)

# === SECONDARY GATE: ROUTE-BASED REJECTION ===
if filters_ok and route == "NONE":
    filters_ok = False  # Hard reject, 11.1% WR is unacceptable
    print(f"[ROUTE-GATE] REJECTED: NONE route (-$12.67 avg, 11.1% WR)")

if filters_ok and route == "AMBIGUOUS" and score < 20:
    filters_ok = False  # Require extra conviction for ambiguous signals
    print(f"[ROUTE-GATE] REJECTED: AMBIGUOUS + low score (need ≥20, got {score})")

# === SECONDARY GATE: REGIME-BASED REJECTION ===
if filters_ok and regime == "BULL" and score < 14:
    filters_ok = False  # BULL regime is weak (22.3% WR), need higher bar
    print(f"[REGIME-GATE] REJECTED: BULL + low score (need ≥14, got {score})")

# === SYNERGY GATE: Worst Combos ===
if filters_ok:
    combo = f"{route}_{regime}"
    
    # Hard reject toxic combos
    toxic_combos = {
        "NONE_BULL": True,      # 7.5% WR, -$15.79 avg
        "NONE_BEAR": True,      # 13.3% WR, -$10.77 avg
        "NONE_RANGE": True,     # 18.2% WR, -$6.47 avg
        "AMBIGUOUS_BULL": True, # 19.4% WR, -$6.95 avg
    }
    
    if combo in toxic_combos:
        filters_ok = False
        print(f"[SYNERGY-GATE] REJECTED: {combo} (toxic combo)")
```

**Impact:**
- Rejects 90 NONE signals × -$12.67 = **+$1,140 saved**
- Rejects 36 AMBIGUOUS_BULL × -$6.95 = **+$250 saved**
- Total: **~$1,400+ annually**

---

## B. OPTION 1 + 2 + 4 (With Corrected Understanding)

### Option 1: Dynamic Secondary Gating (was "Dynamic Thresholds")

**What it is:** Variable rejection criteria by ROUTE/REGIME combo

```python
# In smart_filter.py, add at class initialization
ROUTE_GATING = {
    "REVERSAL": {"min_score": 16, "description": "Higher confidence for reversals"},
    "TREND_CONTINUATION": {"min_score": 12, "description": "Standard gate"},
    "AMBIGUOUS": {"min_score": 20, "description": "High conviction for mixed signals"},
    "NONE": {"min_score": 99, "description": "Hard reject"},
}

REGIME_GATING = {
    "BULL": {"extra_penalty": 2, "description": "Weak regime, tighten threshold"},
    "BEAR": {"extra_penalty": 0, "description": "Strong regime, standard gate"},
    "RANGE": {"extra_penalty": -2, "description": "Profitable regime, loosen threshold"},
}

# Then in analyze(), after MIN_SCORE filter:
if filters_ok:
    route_gate = ROUTE_GATING.get(route, {}).get("min_score", 12)
    regime_penalty = REGIME_GATING.get(regime, {}).get("extra_penalty", 0)
    final_gate = route_gate + regime_penalty
    
    if score < final_gate:
        filters_ok = False
        print(f"[DYNAMIC-GATE] {route}+{regime}: score {score} < threshold {final_gate}")
```

**Parameters:**
```
REVERSAL + RANGE:           min=16 - 2 = 14  (allow 50% WR signals)
TREND_CONTINUATION + BEAR:  min=12 + 0 = 12  (allow 35.2% WR signals)
REVERSAL + BULL:            min=16 + 2 = 18  (block weak reversal in BULL)
NONE + BULL:                min=99 + 2 = 101 (hard reject)
```

**Expected impact:** +2-3% WR by tuning to historical performance

---

### Option 2: Separate WEAK_REVERSAL (was "Weak Reversal Separation")

**What it is:** Better classification of "leaning reversal" vs "truly ambiguous"

```python
# In smart_filter.py, line 421-426 (explicit_reversal_gate)

# Current logic:
if bullish >= 2 and bearish == 0:
    return ("REVERSAL", "BULLISH")
elif bearish >= 2 and bullish == 0:
    return ("REVERSAL", "BEARISH")
elif bullish > 0 and bearish > 0:  # ← Mixed signals
    return ("AMBIGUOUS", ["BULLISH", "BEARISH"])
else:
    return ("NONE", None)

# NEW logic:
if bullish >= 2 and bearish == 0:
    return ("REVERSAL", "BULLISH")  # Strong reversal
elif bearish >= 2 and bullish == 0:
    return ("REVERSAL", "BEARISH")  # Strong reversal
elif bullish >= 2 and bearish >= 1:  # ← Leaning bullish
    return ("WEAK_REVERSAL", "BULLISH")
elif bearish >= 2 and bullish >= 1:  # ← Leaning bearish
    return ("WEAK_REVERSAL", "BEARISH")
elif bullish > 0 and bearish > 0:  # ← Truly split
    return ("AMBIGUOUS", None)
else:
    return ("NONE", None)

# Then update ROUTE_GATING:
ROUTE_GATING = {
    "REVERSAL": {"min_score": 16},        # Pure reversal
    "WEAK_REVERSAL": {"min_score": 14},   # NEW: Leaning reversal (lower bar)
    "TREND_CONTINUATION": {"min_score": 12},
    "AMBIGUOUS": {"min_score": 20},       # Truly uncertain (high bar)
    "NONE": {"min_score": 99},            # Hard reject
}
```

**Expected impact:** Better utilization of leaning signals, +200-300/year in P&L

---

### Option 4: Combo Dashboard (Unchanged)

**What it is:** Real-time visibility into ROUTE×REGIME performance

```python
# In pec_enhanced_reporter.py, add new section:

def analyze_route_regime_performance(signals):
    """Analyze which combos are profitable vs toxic"""
    
    combos = defaultdict(lambda: {
        "count": 0, "wins": 0, "losses": 0, 
        "pnls": [], "avg_pnl": 0, "wr": 0
    })
    
    for sig in signals:
        route = sig.get("route", "UNKNOWN")
        regime = sig.get("regime", "UNKNOWN")
        pnl = sig.get("pnl_usd", 0)
        combo_key = f"{route}_{regime}"
        
        combos[combo_key]["count"] += 1
        combos[combo_key]["pnls"].append(pnl)
        if pnl > 0:
            combos[combo_key]["wins"] += 1
        else:
            combos[combo_key]["losses"] += 1
    
    # Calculate metrics
    for combo, stats in combos.items():
        if stats["count"] >= 3:
            stats["avg_pnl"] = sum(stats["pnls"]) / len(stats["pnls"])
            stats["wr"] = stats["wins"] / stats["count"] * 100
    
    return combos

# Print results:
print("🏆 PROFITABLE COMBOS (>30% WR, >$0 avg):")
for combo, stats in sorted(combos.items(), key=lambda x: x[1]["avg_pnl"], reverse=True):
    if stats["wr"] > 30 and stats["avg_pnl"] > 0:
        print(f"  {combo:30s} | {stats['count']:3d}x | {stats['wr']:5.1f}% WR | ${stats['avg_pnl']:6.2f}")

print("\n💀 TOXIC COMBOS (<15% WR, <$0 avg):")
for combo, stats in sorted(combos.items(), key=lambda x: x[1]["avg_pnl"]):
    if stats["wr"] < 15 or stats["avg_pnl"] < 0:
        print(f"  {combo:30s} | {stats['count']:3d}x | {stats['wr']:5.1f}% WR | ${stats['avg_pnl']:6.2f}")
```

**Expected impact:** Real-time decision support, identifies toxic combos early

---

## Summary: A + B

### What Gets Implemented

| Item | Where | Effort | Payoff |
|------|-------|--------|--------|
| **A. Hard Reject NONE** | smart_filter.py line 850+ | 5 min | +$1,140/yr |
| **A. Synergy Gates (Toxic Combos)** | smart_filter.py line 850+ | 10 min | +$250/yr |
| **Option 1: Dynamic Secondary Gates** | smart_filter.py line 38+ | 30 min | +$200-300/yr |
| **Option 2: WEAK_REVERSAL** | smart_filter.py line 421+ | 15 min | +$200-300/yr |
| **Option 4: Dashboard** | pec_enhanced_reporter.py | 20 min | +$150/yr (visibility) |
| **TOTAL** | | **1.5-2 hours** | **~$1,900+/yr** |

---

## Implementation Order (Recommended)

### Phase 1: Quick Wins (15 min, +$1,390/yr)
```
1. Add NONE hard rejection
2. Add SYNERGY gating for toxic combos
→ Deploy immediately
```

### Phase 2: Smart Filtering (45 min, +$400-600/yr)
```
3. Implement WEAK_REVERSAL
4. Add dynamic secondary gates by ROUTE/REGIME
→ A/B test for 24h, measure impact
```

### Phase 3: Visibility (20 min, +$150/yr)
```
5. Add combo dashboard to reporter
→ Continuous monitoring
```

---

## Code Changes Required

### File 1: smart_filter.py

**Location 1 (line 38):** Add ROUTE/REGIME gating config
```python
# After MIN_SCORE = 12
ROUTE_GATING = {
    "REVERSAL": {"min_score": 16},
    "WEAK_REVERSAL": {"min_score": 14},
    "TREND_CONTINUATION": {"min_score": 12},
    "AMBIGUOUS": {"min_score": 20},
    "NONE": {"min_score": 99},
}

REGIME_GATING = {
    "BULL": {"extra_penalty": 2},
    "BEAR": {"extra_penalty": 0},
    "RANGE": {"extra_penalty": -2},
}

TOXIC_COMBOS = {
    "NONE_BULL", "NONE_BEAR", "NONE_RANGE",
    "AMBIGUOUS_BULL",
}
```

**Location 2 (line 421-426):** Update explicit_reversal_gate() for WEAK_REVERSAL
```python
# Separate WEAK_REVERSAL from AMBIGUOUS
if bullish >= 2 and bearish >= 1:
    return ("WEAK_REVERSAL", "BULLISH")
elif bearish >= 2 and bullish >= 1:
    return ("WEAK_REVERSAL", "BEARISH")
elif bullish > 0 and bearish > 0:
    return ("AMBIGUOUS", None)
# ... rest unchanged
```

**Location 3 (line 850+):** Add secondary gates after MIN_SCORE filter
```python
# After: filters_ok = (direction ...) and (score >= self.min_score)

# Secondary gates based on ROUTE/REGIME
if filters_ok:
    # Check ROUTE gating
    route_min_score = ROUTE_GATING.get(route, {}).get("min_score", 12)
    regime_penalty = REGIME_GATING.get(regime, {}).get("extra_penalty", 0)
    final_threshold = route_min_score + regime_penalty
    
    if score < final_threshold:
        filters_ok = False
        if DEBUG_FILTERS:
            print(f"[SECONDARY-GATE] {route}+{regime}: score {score} < {final_threshold}")
    
    # Check toxic combos
    if filters_ok and f"{route}_{regime}" in TOXIC_COMBOS:
        filters_ok = False
        if DEBUG_FILTERS:
            print(f"[TOXIC-COMBO] {route}+{regime} auto-rejected")
```

### File 2: pec_enhanced_reporter.py

**Add new function:** analyze_route_regime_combos()
```python
def analyze_route_regime_combos(signals):
    """Real-time ROUTE×REGIME performance tracking"""
    # See Option 4 code above
```

**Call in main report generation:**
```python
combos = analyze_route_regime_combos(signals)
print_combo_performance(combos)
```

---

## A/B Test Plan

### Before (Current)
- MIN_SCORE=12 for all
- No post-filter gating
- NONE combo: 90 trades, 11.1% WR, -$1,140 total

### After (With Optimization)
- MIN_SCORE=12 unchanged (filter layer)
- New secondary gates (ROUTE/REGIME layer)
- NONE combo: 0 trades (rejected)
- Better REVERSAL + RANGE signals (lowered threshold to 14)

### Measurement
```
Run for 24 hours:
- Signal count: Expect ~5-10% fewer signals (filtering bad combos)
- WR improvement: Expect +0.5-1 percentage point
- P&L: Expect +$50-100/day from avoided losses
```

---

## Conclusion

**Plan A (Optimize ROUTE & REGIME):** Add post-filter secondary gating based on ROUTE/REGIME combos  
**Plan B (Option 1+2+4):** Implement smart gating, weak_reversal separation, and dashboard

Both are complementary. Total effort: 1.5-2 hours. Expected annual impact: ~$1,900+.

