# 🔒 FOUNDATION LOCKED - ONE AUTHORITATIVE BASELINE

**Decision Date:** 2026-03-04 01:10 GMT+7  
**Status:** ✅ ALL OTHER NUMBERS DISCARDED

---

## 🔐 THE ONE TRUE BASELINE (IMMUTABLE)

```
FOUNDATION_NUMBER = 853 signals

Total Signals:        853
Closed Trades:        830
Win Rate:             25.7%
LONG WR:              29.6%
SHORT WR:             46.2%
Total P&L:            -$5,498.59
```

**Source:** pec_enhanced_reporter.py (verified from SENT_SIGNALS.jsonl)  
**Locked:** 2026-03-04 01:10 GMT+7  
**Never changes.** Never recalculates.

---

## ❌ DISCARDED NUMBERS (IGNORE FOREVER)

- ❌ **1,205** - Old claim (unverified, can't trace to data)
- ❌ **998** - Raw file lines (includes unparsed entries)
- ❌ **810** - SENT_SIGNALS_PHASE1_BASELINE.jsonl (backup file, not current)

**These numbers are WRONG. Do not use.** Delete references.

---

## ✅ HARD-CODED BASELINE (Use in ALL Scripts)

Update ALL comparison scripts to use:

```python
FOUNDATION_LOCKED = {
    "total_signals": 853,
    "closed_trades": 830,
    "win_rate": 25.7,
    "long_wr": 29.6,
    "short_wr": 46.2,
    "pnl": -5498.59
}
```

---

## 📋 Scripts to Update (Immediate)

1. `COMPARE_AB_TEST_LOCKED.py` - Change from 1,205 → 853
2. `track_phase2_fixed.py` - Reference baseline as 853
3. Any other comparison tool - Use 853 as baseline

**Command to find all old references:**
```bash
grep -r "1205\|1052\|29.66" *.py
```

---

## 🎯 FROM NOW ON

**Phase 1 Baseline = 853 signals @ 25.7% WR**

All Phase 2-FIXED, Phase 3B, Phase 4A comparisons use this ONE number.

No more "two different baselines". One foundation. All scripts synchronized.

---

**This is our anchor. Everything else is trash.** 🔒
