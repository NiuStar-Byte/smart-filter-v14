# SIGNALS_MASTER vs SIGNALS_INDEPENDENT_AUDIT Checkpoint

**Timestamp:** 2026-03-23 12:33 GMT+7
**Status:** ⚠️ DIVERGENCE DETECTED - Recovery Required

---

## File Inventory

| File | Lines | Unique UUIDs | Note |
|------|-------|--------------|------|
| SIGNALS_MASTER.jsonl | 3,952 | 3,952 | Status tracker (current state) |
| SIGNALS_INDEPENDENT_AUDIT.txt | 2,267 | 2,267 | Immutable baseline (append-only) |
| **Difference** | **1,685 lines** | **N/A** | MASTER has 1,685 more signals |

---

## UUID Overlap Analysis

```
Total unique UUIDs in both files: 2,267
├─ In BOTH (matching):           2,242 (98.9% overlap)
├─ MASTER-ONLY:                  1,710 (47.4% of MASTER)
└─ AUDIT-ONLY:                      25 (1.1% of AUDIT)

Mismatch: 1,735 diverging signals (45% of total)
```

---

## Signal Breakdown by Origin

### MASTER (3,952 total)
```
FOUNDATION (locked baseline):      2,224 (56.3%)
NEW_LIVE (current accumulation):   1,728 (43.7%)
```

### AUDIT (2,267 total)
```
FOUNDATION (locked baseline):      2,224 (98.1%) ✅
NEW_LIVE (current accumulation):      43 (1.9%)  ⚠️
```

**Issue:** AUDIT should contain all signals (FOUNDATION + NEW_LIVE), but only has 43 NEW_LIVE.

---

## Critical Findings

### 🚨 Finding #1: MASTER-ONLY Signals (1,710)
**Location:** In SIGNALS_MASTER.jsonl but NOT in SIGNALS_INDEPENDENT_AUDIT.txt
**Origin:** ALL are NEW_LIVE signals
**Status Breakdown:**
- TP_HIT: 243 (14.2% wins)
- SL_HIT: 858 (50.2% losses)
- TIMEOUT: 515 (30.2% unfinished)
- OPEN: 91 (5.3% still accumulating)
- STALE_TIMEOUT: 3

**Root Cause:** Daemon is writing to MASTER but NOT writing corresponding entries to AUDIT
**Impact:** AUDIT is incomplete - missing 1,710 recent signals
**Expected Behavior:** AUDIT should be append-only archive of ALL signals

### 🔴 Finding #2: AUDIT-ONLY Signals (25)
**Location:** In SIGNALS_INDEPENDENT_AUDIT.txt but NOT in SIGNALS_MASTER.jsonl
**Origin:** ALL are NEW_LIVE signals (OPEN status)
**Examples:**
- VIRTUAL-USDT (2 signals)
- INJ-USDT, PENGU-USDT, SUI-USDT (2 each)
- FUEL-USDT, YFI-USDT, LDO-USDT, SPK-USDT (various)

**Root Cause:** These signals were written to AUDIT but failed to write to MASTER
**Impact:** Status tracking broken - executor can't find these signals
**Expected Behavior:** All AUDIT signals should be in MASTER

---

## Architecture Check

**Current State:**
```
DAEMON (fires signal)
├─ Write to SIGNALS_MASTER.jsonl ✅ (1,710 NEW_LIVE + 2,224 FOUNDATION = 3,934)
└─ Write to SIGNALS_INDEPENDENT_AUDIT.txt ⚠️ (only 2,267 written = incomplete)

EXECUTOR (updates status)
├─ Read from SIGNALS_INDEPENDENT_AUDIT.txt ✅ (gets facts)
├─ Update SIGNALS_MASTER.jsonl ✅ (status changes)
└─ Problem: Missing 1,710 signals for backtest

REPORTER (generates reports)
├─ Read SIGNALS_INDEPENDENT_AUDIT.txt (immutable facts)
└─ Read SIGNALS_MASTER.jsonl (current status)
└─ Problem: Sees incomplete AUDIT baseline
```

**Expected State:**
```
DAEMON should write BOTH files simultaneously
AUDIT should contain ALL signals (immutable append-only)
MASTER should track current status (mutable)
```

---

## Recovery Options

### Option A: Backfill AUDIT (Recommended)
1. Extract all 1,710 MASTER-ONLY signals
2. Append to SIGNALS_INDEPENDENT_AUDIT.txt
3. Verify all signals now in both files
4. Status: Repairs incomplete AUDIT

**Pros:** 
- Completes the immutable baseline
- Makes AUDIT the true archive
- MASTER becomes status-only tracker

**Cons:**
- AUDIT no longer purely append-only (this run adds 1,710)
- 25 AUDIT-only signals still missing from MASTER

### Option B: Rebuild MASTER from AUDIT
1. Extract all signals from AUDIT (2,267)
2. Rebuild MASTER with current status from executor
3. Status: Purges 1,710 signals not in AUDIT

**Pros:**
- Enforces AUDIT as source of truth
- Removes orphaned MASTER-only signals

**Cons:**
- Loses 1,710 NEW_LIVE signals' data
- Executor loses backtest history

### Option C: Merge Both Directions (Safest)
1. **Forward:** Backfill AUDIT with 1,710 MASTER-only signals
2. **Reverse:** Merge 25 AUDIT-only signals into MASTER
3. Verify 100% alignment
4. Status: Both files complete and synced

**Pros:**
- No data loss
- Complete coverage in both files
- Full audit trail preserved

**Cons:**
- More complex recovery process
- Need to handle 25 conflicts

---

## Recommendation

**Do Option C (Merge Both Directions)** because:
1. ✅ No data loss
2. ✅ Maintains audit trail
3. ✅ Executor gets all 3,952 signals for backtest
4. ✅ Reporter gets complete AUDIT baseline
5. ✅ Verification: 3,952 signals in both files = success

**Next Steps:**
1. Extract 1,710 MASTER-only signals
2. Append to SIGNALS_INDEPENDENT_AUDIT.txt
3. Extract 25 AUDIT-only signals  
4. Append to SIGNALS_MASTER.jsonl
5. Re-run this checkpoint to verify 100% alignment
6. Commit recovery as immutable checkpoint

---

## Summary Table

| Metric | Value | Status |
|--------|-------|--------|
| Total signals (union) | 3,952 | ✅ Expected |
| MASTER signals | 3,952 | ✅ Complete |
| AUDIT signals | 2,267 | ⚠️ Incomplete (missing 1,710) |
| Perfect alignment | NO | ❌ Need recovery |
| FOUNDATION baseline | 2,224 | ✅ Locked in both |
| NEW_LIVE in MASTER | 1,728 | ✅ Tracking |
| NEW_LIVE in AUDIT | 43 | ⚠️ Only 2.5% of expected |

---

## Timeline

**Checkpoint Time:** 2026-03-23 12:33 GMT+7
**Divergence Window:** Likely since ~2026-03-21 (when daemon started accumulating NEW_LIVE)
**Recovery Priority:** HIGH - Impacts executor accuracy and reporter baseline

Next checkpoint after recovery: 2026-03-23 13:00 GMT+7 (confirm 100% alignment)
