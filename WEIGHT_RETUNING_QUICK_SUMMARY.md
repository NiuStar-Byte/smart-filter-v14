# Quick Summary: Weight Retuning Decision
**Date:** 2026-03-24 00:03 GMT+7  
**Data:** 1,576 instrumented signals | Baseline WR: 27.3%

---

## ❌ Problem

**Current weights don't match real performance:**

| Filter | Current Weight | Actual WR | Delta | Status |
|--------|---|---|---|---|
| Momentum | 5.5 | 30.3% | +3.0pp | ✅ UNDERWEIGHTED |
| Spread Filter | 5.0 | 29.9% | +2.6pp | ✅ UNDERWEIGHTED |
| Support/Resistance | **5.0** | **0.0%** | **-27.3pp** | **❌ TOXIC** |
| Volatility Model | **3.9** | **14.8%** | **-12.6pp** | **❌ BAD** |
| ATR Momentum | **4.3** | **20.4%** | **-7.0pp** | **❌ NEGATIVE** |

---

## ✅ Solution

**Reweight filters to match real WR data:**

```
REMOVE (0.0 weight):
  - Support/Resistance (0% WR, firing junk)
  - Absorption (0% WR, dead)

CUT (reduce exposure):
  - Volatility Model: 3.9 → 2.1
  - ATR Momentum Burst: 4.3 → 3.2

BOOST (reward winners):
  - Momentum: 5.5 → 6.0
  - Spread Filter: 5.0 → 5.5

KEEP (at baseline):
  - 11 other filters
```

---

## 📊 Impact

**Before:** Total weight 89.1 (includes toxic)  
**After:** Total weight 80.2 (toxic removed, winners boosted)  
**Change:** -8.9 weight (consolidating to quality)

**Expected Win Rate:** +2-3pp improvement  
**Signal Count:** No change (gatekeeper still soft)  
**Risk:** Low (data-driven, easily reverted)

---

## ⏱️ Time to Deploy

- Edit code: **2 min**
- Commit: **1 min**
- Reload daemon: **30 sec**
- **Total: ~4 minutes**

---

## 🎯 DECISION

### **APPROVE? (YES / NO)**

If YES:
```
git edit smart_filter.py → apply weight changes
git commit → push
kill daemon → restart with new weights
monitor WR for 24h → confirm improvement
```

If NO:
```
Keep current weights as-is
Continue monitoring
Analyze further before changes
```

---

**Full proposal:** See `COMPLETE_FILTER_WEIGHT_PROPOSAL_2026_03_24.md`

