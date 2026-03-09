# 🚀 PHASE 2-FIXED: FRESH START (2026-03-08 11:20 GMT+7)

## Reset Decision
**Date:** 2026-03-08 11:20 GMT+7  
**Decision:** Discard corrupted Mar 3-8 signals, restart clean

### Reason for Reset
- Old period (Mar 3-8): 573 total Phase 2-FIXED signals
- Problem: 371 signals never processed (raw daemon output)
- Impact: Can't calculate reliable P&L metrics
- **Action:** Start fresh from 11:20 GMT+7 onwards
- Old signals: Archived, not counted

---

## Test Parameters

### FOUNDATION (Locked Baseline)
- **Signals:** 853 (immutable, Feb 27 - Mar 3 13:16 UTC)
- **Win Rate:** 25.7%
- **Total P&L:** -$5498.59
- **Status:** IMMUTABLE - Reference only

### PHASE 2-FIXED (New Count)
- **Start Time:** 2026-03-08 11:20 GMT+7 (UTC: 2026-03-08 04:20 UTC)
- **Start Count:** 0 signals
- **Success Target:** ≥25.7% WR (match Foundation)
- **Confidence Threshold:** 100+ closed trades
- **ETA to Decision:** ~1-2 weeks

---

## What We're Testing

**The Code:** Direction-aware gates (Phase 2-FIXED implementation)

**The Gates Being Tested:**
1. Gate 1 (Momentum-Price): RSI-aware per direction
2. Gate 3 (Trend Alignment): MA-aware per regime
3. Gate 4 (Candle Structure): Pattern-aware per direction

**Expected Results:**
- BEAR SHORT WR: 0% → 25%+ (primary metric)
- BULL LONG WR: ~31% → 33%+ (no regression)
- Overall WR: 30% → 32%+

---

## Tracking Commands

```bash
# Fresh count of Phase 2-FIXED signals (only after reset point)
python3 track_phase2_fixed_fresh.py

# Compare to Foundation
python3 phase2_fixed_vs_foundation_fresh.py

# Monitor daily
python3 track_phase2_fixed_fresh.py --daily

# Check for regressions
python3 track_phase2_fixed_fresh.py --bull-long
```

---

## Daily Monitoring Schedule

### Check Every 8 Hours
```bash
python3 track_phase2_fixed_fresh.py
```

**Look for:**
1. Signal count increasing (expect 8-12/day)
2. BEAR SHORT signals appearing (should be 0 initially, then growing)
3. BULL LONG WR staying 28%+ (regression check)
4. Any errors in daemon logs

### Collect Until
- 100+ closed trades
- ~7 days elapsed (2026-03-15)
- Confidence in metrics

### Decision Point
- **Date:** ~2026-03-15 (when 100 closed trades reached)
- **Compare:** Phase 2-FIXED WR vs Foundation 25.7%
- **Approve If:** ≥25.7% + no regressions
- **Iterate If:** <25.7% or regressions detected

---

## Fresh Start Details

### Cutoff Time
```
Old Phase 2-FIXED signals:    Mar 3 13:16 UTC - Mar 8 04:20 UTC
↓↓↓ DISCARD ↓↓↓ (373 unprocessed signals)
New Phase 2-FIXED signals:    Mar 8 04:20 UTC onwards
```

### What Counts Toward Test
- **Only signals fired AFTER 2026-03-08 04:20 UTC (2026-03-08 11:20 GMT+7)**
- Must have status field (TP_HIT, SL_HIT, TIMEOUT, OPEN)
- Must be properly processed by daemon pipeline
- All exit types count (TP, SL, TIMEOUT)

### What Doesn't Count
- ❌ Mar 3-8 signals (corrupted, archived)
- ❌ FOUNDATION baseline (reference only)
- ❌ Raw unprocessed daemon output

---

## Success Criteria

**Phase 2-FIXED PASSES if:**
1. ✅ Reaches 100+ closed trades
2. ✅ Win Rate ≥ 25.7% (matches Foundation)
3. ✅ BEAR SHORT WR improves (even if from 0%)
4. ✅ BULL LONG WR stays 28%+ (no regression)
5. ✅ Zero critical errors in logs

**Phase 2-FIXED FAILS if:**
1. ❌ Closed trades < 50 after 14 days
2. ❌ Win Rate < 20% (significant miss)
3. ❌ Critical errors in daemon
4. ❌ BULL LONG drops below 25%

---

## Files & Scripts

- `PHASE2_FIXED_FRESH_START.md` ← This document
- `track_phase2_fixed_fresh.py` ← Fresh tracker (TBD)
- `phase2_fixed_vs_foundation_fresh.py` ← Comparison (TBD)

---

**Status:** READY TO COUNT  
**Reset Date:** 2026-03-08 11:20 GMT+7  
**Test Live:** YES  
**Next Review:** 2026-03-08 19:00 GMT+7 (8 hours)
