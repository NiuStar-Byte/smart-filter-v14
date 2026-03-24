# CORRECTED Weight Proposal (Based on Actual 81.5)
**Date:** 2026-03-24 00:13 GMT+7  
**Correction:** User caught error - actual current weight is 81.5 (not 86.5)

---

## ❌ MY ERROR

I calculated 86.5 based on code, but you're showing **81.5 in production**.

This means **Support/Resistance is already disabled (0.0 weight)** in the actual deployed system.

---

## ✅ CORRECTED CALCULATION

### Current State (81.5 total)
```
Total: 81.5 weight
├─ Support/Resistance: 0.0 (already disabled)
└─ All other 19 filters: 81.5 weight
```

### Proposed (Based on WR Data)
```
Remove Absorption:    2.7 → 0.0  [-2.7]  ← Dead filter (0% WR)
Cut Volatility Model: 3.9 → 2.1  [-1.8]  ← 14.8% WR
Cut ATR Momentum:     4.3 → 3.2  [-1.1]  ← 20.4% WR
Boost Momentum:       5.5 → 6.0  [+0.5]  ← 30.3% WR
Boost Spread:         5.0 → 5.5  [+0.5]  ← 29.9% WR
────────────────────────────────────
Current total: 81.5
Proposed total: 81.5 - 2.7 - 1.8 - 1.1 + 0.5 + 0.5 = 75.9
```

---

## 📊 CORRECTED COMPARISON

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Actual Current Total** | **81.5** | **75.9** | **-5.6** |
| **% Change** | 100% | 93.1% | -6.9% |
| **Toxic Removed** | 2.7 | 0.0 | -2.7 |
| **Bad Cuts** | 8.2 | 5.3 | -2.9 |
| **Winners Boosted** | 10.5 | 11.5 | +1.0 |

---

## 🎯 REVISED PROPOSAL

```
REMOVE:
└─ Absorption: 2.7 → 0.0 [-2.7]

CUT:
├─ Volatility Model: 3.9 → 2.1 [-1.8]
└─ ATR Momentum Burst: 4.3 → 3.2 [-1.1]

BOOST:
├─ Momentum: 5.5 → 6.0 [+0.5]
└─ Spread Filter: 5.0 → 5.5 [+0.5]

KEEP (unchanged):
├─ All other 14 filters: as current
└─ Support/Resistance: already 0.0

NET: 81.5 → 75.9 (remove 5.6 weight of underperformers)
```

---

## 🔍 WHAT'S ALREADY DEPLOYED?

```
✅ Support/Resistance: Already at 0.0
   └─ User already disabled it (or it was done before my analysis)
   └─ Reduces weight from 86.5 → 81.5 (-5.0)
   └─ Good call! It was firing junk signals (0% WR)
```

---

## ✅ FINAL NUMBERS

| Current | Proposed | Change |
|---------|----------|--------|
| **81.5** | **75.9** | **-5.6** |

Total weight reduction: 5.6 (down 6.9%)

---

## 📋 DECISION

**Should I deploy the revised proposal?**
- Remove Absorption (2.7 → 0.0) ← Already dead
- Cut Volatility Model (3.9 → 2.1) ← 14.8% WR is bad
- Cut ATR Momentum (4.3 → 3.2) ← 20.4% WR is bad
- Boost Momentum (5.5 → 6.0) ← 30.3% WR is best
- Boost Spread (5.0 → 5.5) ← 29.9% WR is good

Expected: Better signal quality, same count, +2-3pp WR

