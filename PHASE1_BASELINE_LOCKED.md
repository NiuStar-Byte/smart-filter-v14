# 🔒 PHASE 1 BASELINE - LOCKED AND FROZEN

**Status:** ✅ LOCKED at 1,205 signals across ALL reports  
**Locked Date:** 2026-03-03 21:22 GMT+7  
**Cutoff:** All signals BEFORE 2026-03-03 13:16 UTC (Mar 3 20:16 GMT+7)

---

## 🔒 PHASE 1 BASELINE VALUES (LOCKED - NEVER CHANGES)

```
Total Signals:     1,205
Closed Trades:     1,052
Win Rate:          29.66%
P&L (USD):         -$5,727.12
LONG WR:           27.69%
SHORT WR:          43.51%
```

**These values are FROZEN.** They will appear identically in:
- ✅ A/B Test Report (`COMPARE_AB_TEST.py`)
- ✅ Phase 3 Tracker (`PHASE3_TRACKER.py`)
- ✅ Phase 4A Tracker (when created)

---

## 📊 Applied To All Comparisons

| Comparison Tool | Script | Phase 1 (A) | Phase 2-FIXED (B) | Status |
|---------|--------|-----------|------------------|--------|
| **A/B Test** | `COMPARE_AB_TEST.py` | 1,205 (locked) | Fresh signals from 13:16 UTC | ✅ Active |
| **Phase 3 Tracking** | `PHASE3_TRACKER.py` | 1,205 (locked) | Mar 2 21:30 - Mar 3 13:16 UTC | ✅ Updated |
| **Phase 4A Tracking** | TBD | 1,205 (locked) | TBD | 📋 Pending |

---

## 🔑 Key Rules

1. **PHASE 1 BASELINE IS FROZEN**
   - Never recalculate
   - Never change cutoff dates
   - Use locked values: 1,205 signals, 29.66% WR
   
2. **ALL COMPARISONS USE SAME BASELINE**
   - No variations between reports
   - Consistency across all metrics
   - Single source of truth

3. **ONLY CHALLENGER (PHASE X) CHANGES**
   - Phase 2-FIXED: Fresh signals only (after 13:16 UTC)
   - Phase 3: Historical (Mar 2 21:30 - Mar 3 13:16 UTC)
   - Phase 4A: TBD (will be defined when created)

---

## 🎯 Cutoff Timestamp (CRITICAL)

**2026-03-03 13:16 UTC = 2026-03-03 20:16 GMT+7**

This is when:
- ✅ Critical fixes were deployed (momentum + threshold logic)
- ✅ Phase 2-FIXED fresh signals began collecting
- ✅ Phase 3 period ended (if running)
- ✅ Everything after this = NEW/FRESH data

---

## 📋 How to Verify

### Check A/B Test
```bash
python3 COMPARE_AB_TEST.py --once | grep "Total Signals\|Phase 1"
# Should show: 1,205 signals
```

### Check Phase 3 Tracker
```bash
python3 PHASE3_TRACKER.py | grep "Total Signals\|PHASE 1"
# Should show: 1,205 signals
```

### Check Locked Values in Code
```bash
grep "PHASE1_LOCKED\|PHASE1_CUTOFF" PHASE3_TRACKER.py
# Should show locked dict with: total_signals: 1205
```

---

## ⚠️ DO NOT

- ❌ Recalculate Phase 1 baseline
- ❌ Change the 13:16 UTC cutoff timestamp
- ❌ Use different baseline values in different reports
- ❌ Load Phase 2-FIXED signals into Phase 1
- ❌ Mix broken period (10:36-13:16 UTC) into any baseline

---

## ✅ DO

- ✅ Use the locked 1,205 baseline everywhere
- ✅ Use 13:16 UTC as the definitive cutoff
- ✅ Keep Phase 1 frozen and immutable
- ✅ Compare only against CHALLENGER (Phase X) data
- ✅ Document any cutoff changes in this file

---

## 📝 Locked Baseline Reference

If you ever need to verify or restore the locked values:

```python
PHASE1_LOCKED = {
    "total_signals": 1205,
    "closed_trades": 1052,
    "win_rate": 29.66,
    "pnl": -5727.12,
    "long_wr": 27.69,
    "short_wr": 43.51
}
```

---

**This baseline is the anchor point for all Phase improvements. It does not change.** 🔒
