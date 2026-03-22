# PROJECT-5: PEC (Position Entry Closure & Backtest) - Rebuild Plan

**Decision Date:** 2026-03-21 00:49 GMT+7  
**Status:** ✅ Architectural rebuild approved - Ready to execute phases  
**Scope:** Complete rebuild of PEC system from clean foundation

---

## 📋 EXECUTIVE SUMMARY

**The Problem:** For 1 month, PEC system kept breaking because:
1. Daemon stopped writing to SIGNALS_INDEPENDENT_AUDIT.txt after Mar 14
2. Files diverged (MASTER and AUDIT became out of sync)
3. No clear immutability model enforced
4. Reporter confused about which file to read

**The Solution:** 
1. Use Feb 27 - Mar 14 (2,224 signals) as NEW FOUNDATION
2. Discard all Mar 15-20 signals (contaminated period)
3. Rebuild architecture: AUDIT = immutable source, MASTER = status tracker
4. Enforce: Daemon writes to BOTH files, Executor updates MASTER only, Reporter reads both

---

## 🎯 NEW FOUNDATION (Locked)

**Period:** Feb 27 - Mar 14, 2026  
**Signal Count:** 2,224 unique signals  
**File Alignment:** Both SIGNALS_MASTER.jsonl and SIGNALS_INDEPENDENT_AUDIT.txt perfectly synced  
**Status:** ✅ CLEAN, VERIFIED

This period becomes the immutable baseline. All metrics from this period lock forever.

```
Feb 27-Mar 14: 2,224 signals (2,224 in both files)
                ├─ signal_origin = "FOUNDATION"
                ├─ Metrics locked forever
                └─ Never modified or recalculated
```

---

## 🗑️ DISCARDED PERIOD (Contaminated)

**Period:** Mar 15 - Mar 20, 2026  
**What Happened:** 
- Daemon stopped writing to SIGNALS_INDEPENDENT_AUDIT.txt
- Files split into two separate streams
- 776 signals in MASTER but not AUDIT
- 516 signals in AUDIT but not MASTER
- 707 status mismatches between files

**Decision:** Remove entirely from analysis
- Don't count toward foundation
- Don't count toward new signals  
- System restarts fresh from Mar 21

---

## 🏗️ CORRECT ARCHITECTURE

### File Responsibilities

**SIGNALS_INDEPENDENT_AUDIT.txt**
- Role: Immutable source of truth
- What: Every fired signal (from Feb 27 onwards)
- Format: Newline-delimited JSON
- Mutability: APPEND-ONLY (never modified)
- Updated by: Daemon (writes)
- Read by: Reporter (metrics), Executor (signal facts)
- Recovery use: Can rebuild MASTER from this file

**SIGNALS_MASTER.jsonl**
- Role: Current status tracker
- What: Latest state of each signal (status, actual_exit_price, pnl)
- Format: Newline-delimited JSON (same schema as AUDIT)
- Mutability: Status-only updates (OPEN → TP_HIT/SL_HIT/TIMEOUT)
- Updated by: Daemon (appends), Executor (updates status)
- Read by: Reporter (current state), Executor (finds OPEN signals)
- Rebuild use: Can be rebuilt from AUDIT anytime

### Data Flow

```
Daemon (PROJECT-3: SmartFilter fires signal)
  ↓ APPEND to SIGNALS_INDEPENDENT_AUDIT.txt (mandatory)
  ↓ APPEND to SIGNALS_MASTER.jsonl (mandatory)
  └─ Signal created with status="OPEN", signal_origin="NEW_LIVE"

Executor (backtests signal via OHLCV walking)
  ├─ READ from SIGNALS_INDEPENDENT_AUDIT.txt (get signal facts)
  ├─ UPDATE SIGNALS_MASTER.jsonl (status only)
  │  └─ status: OPEN → TP_HIT | SL_HIT | TIMEOUT
  │  └─ actual_exit_price, pnl, closed_at
  └─ NEVER modify AUDIT

Reporter (generates metrics report)
  ├─ READ from SIGNALS_INDEPENDENT_AUDIT.txt (immutable baseline)
  │  └─ Extract: signal_origin, fired_time, entry_price, tp_price, sl_price
  ├─ READ from SIGNALS_MASTER.jsonl (current status)
  │  └─ Extract: status, actual_exit_price, pnl
  ├─ MERGE both sources
  └─ OUTPUT: Metrics locked from AUDIT, status dynamic from MASTER
```

---

## 🔐 IMMUTABILITY CONTRACT

### FOUNDATION (Feb 27 - Mar 14, 2,224 signals)
```
Status: LOCKED FOREVER
├─ Location: Lines 1-2224 in SIGNALS_INDEPENDENT_AUDIT.txt
├─ Can READ: Metrics, calculations, comparisons
├─ CANNOT modify: Any field, any calculation
├─ CANNOT recalculate: Foundation metrics are fixed
└─ Rule: Violation triggers system reset to zero baseline
```

### NEW_IMMUTABLE (Completed periods after Mar 14)
```
Status: LOCKED after day ends
├─ Created daily: Each day's NEW_LIVE becomes NEW_IMMUTABLE at midnight
├─ Can READ: All historical calculations
├─ CANNOT modify: fired_time, entry_price, signal_origin
├─ CAN modify until locked: status (OPEN → TP_HIT/SL_HIT/TIMEOUT)
└─ Rule: Violation triggers system reset to zero baseline
```

### NEW_LIVE (Current day accumulating, starting fresh Mar 21)
```
Status: OPEN, GROWING
├─ Daemon appends new signals
├─ Executor updates status as signals close
├─ CAN modify: status, actual_exit_price, pnl, closed_at
├─ CANNOT modify: fired_time, entry_price, signal_origin
└─ Locks at midnight: Becomes NEW_IMMUTABLE tomorrow
```

---

## 📋 REBUILD PHASES (Step-by-Step)

### Phase 1: Extract Clean Foundation
**What:** Extract Feb 27 - Mar 14 signals from both files  
**Input:** Current SIGNALS_MASTER.jsonl + SIGNALS_INDEPENDENT_AUDIT.txt  
**Output:** 
- SIGNALS_FOUNDATION_CLEAN.jsonl (2,224 signals)
- Verify both sources have exact same signals

**Verification:** 
- [ ] Count: 2,224 unique signals
- [ ] Dates: All between Feb 27 - Mar 14 (no Mar 15-20)
- [ ] Alignment: Both files have exact same UUIDs

### Phase 2: Lock Foundation Metadata
**What:** Calculate and lock foundation metrics  
**Input:** SIGNALS_FOUNDATION_CLEAN.jsonl  
**Output:** 
- Foundation WR, P&L, avg duration
- SIGNALS_FOUNDATION_LOCKED.json (immutable metadata)
- signal_origin = "FOUNDATION" for all signals

**Metrics:** (Will be calculated, not hardcoded)
- Total: 2,224
- Closed: (count of TP_HIT + SL_HIT + TIMEOUT)
- WR: (wins / closed)
- P&L: sum of all pnl_usd
- These numbers LOCK and never change

### Phase 3: Rebuild AUDIT
**What:** Create clean SIGNALS_INDEPENDENT_AUDIT.txt  
**Input:** SIGNALS_FOUNDATION_CLEAN.jsonl  
**Output:** SIGNALS_INDEPENDENT_AUDIT.txt (only Feb 27 - Mar 14, signal_origin="FOUNDATION")

**Process:**
- [ ] Remove all Mar 15-20 signals from current AUDIT
- [ ] Keep only Feb 27 - Mar 14 signals
- [ ] Verify each signal has signal_origin field
- [ ] No other modifications

### Phase 4: Reset MASTER
**What:** Create clean SIGNALS_MASTER.jsonl  
**Input:** SIGNALS_FOUNDATION_CLEAN.jsonl + SIGNALS_INDEPENDENT_AUDIT.txt  
**Output:** SIGNALS_MASTER.jsonl (FOUNDATION section + empty NEW_LIVE ready)

**Process:**
- [ ] Keep all FOUNDATION signals from AUDIT
- [ ] Remove all Mar 15-20 entries
- [ ] Set up fresh NEW_LIVE segment starting Mar 21
- [ ] Both files now identical for FOUNDATION period

### Phase 5: Lock Reporter Template
**What:** Freeze reporter structure, make metrics dynamic  
**Input:** pec_enhanced_reporter.py  
**Output:** Updated pec_enhanced_reporter.py

**Requirements:**
- [ ] Reporter template LOCKED (never modify structure again)
- [ ] All hardcoded numbers removed (get from AUDIT)
- [ ] All metrics calculated from immutable AUDIT
- [ ] Status/current state read from MASTER
- [ ] Output: Foundation (locked) vs NEW (dynamic)

### Phase 6: Verify System
**What:** Confirm rebuild successful  
**Input:** All new files  
**Output:** Clean PEC system ready for forward operation

**Verification:**
- [ ] AUDIT and MASTER aligned for FOUNDATION period (2,224 signals)
- [ ] All signals have signal_origin field
- [ ] Foundation metrics locked and immutable
- [ ] Reporter reads AUDIT (immutable) + MASTER (current)
- [ ] Daemon configured to write to BOTH files
- [ ] Executor reads AUDIT, updates MASTER only

---

## ⏭️ NEXT STEPS (Ready to Execute)

**Do you approve proceeding with Phase 1 (Extract Clean Foundation)?**

Once approved:
1. Extract Feb 27 - Mar 14 period
2. Verify 2,224 signals in both files
3. Proceed to Phase 2 (Lock Foundation)
4. Continue through all 6 phases
5. Restart PEC system fresh from Mar 21 with clean architecture

---

## 📌 KEY PRINCIPLES (Never Violate)

1. **FOUNDATION IS IMMUTABLE** - Changes are prohibited
2. **AUDIT IS APPEND-ONLY** - Never modified, only appended
3. **MASTER IS STATUS-ONLY** - Only status updates, never historical data
4. **DUAL-WRITE MANDATORY** - Daemon writes to BOTH or nothing
5. **REPORTER IS LOCKED** - Template never changes, only metrics update
6. **SEPARATION OF CONCERNS** - AUDIT (history), MASTER (state), Reporter (reporting)

If any of these are violated, system resets to zero baseline.
