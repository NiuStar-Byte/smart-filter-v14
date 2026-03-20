# PROJECT-5: PEC (Position Entry Closure & Backtest) - DECISIONS LOCKED

**Document Date:** 2026-03-21 00:57 GMT+7  
**Status:** 🔒 ALL DECISIONS FINALIZED - NOT YET BUILT  
**Approval Status:** AWAITING FINAL BUILD CONFIRMATION

---

## 📋 EXECUTIVE SUMMARY (One Page)

**What we're doing:**
Build a reliable PEC system with immutable baseline + daily rolling accumulation.

**The problem we're solving:**
For 1 month, PEC kept breaking because daemon stopped writing to SIGNALS_INDEPENDENT_AUDIT.txt after Mar 14, causing files to diverge.

**The solution:**
- Use Feb 27 - Mar 14 (2,224 signals) as locked FOUNDATION
- Discard Mar 15-20 completely (contaminated)
- Start fresh from Mar 21 (Option A)
- AUDIT = immutable truth, MASTER = status tracker
- Reporter reads BOTH sources, template locked

**Timeline:**
- Start rebuild: 2026-03-21 (awaiting approval)
- Phase 1: Extract clean foundation
- Phase 2-6: Lock metrics, rebuild files, verify system
- Go live: Mar 21 fresh start

---

## 🔒 LOCKED DECISIONS (Do Not Change)

### Decision 1: NEW FOUNDATION
**Decision:** Use Feb 27 - Mar 14 as locked FOUNDATION (2,224 signals)  
**Approved by:** User  
**Date:** 2026-03-21 00:49 GMT+7  
**Rationale:** Both files perfectly synced for this period, clean data  
**Impact:** All metrics from this period lock forever, never recalculated  
**Proof:** Verified - 2,224 signals in BOTH files, identical UUIDs

### Decision 2: DISCARD Mar 15-20
**Decision:** Remove all Mar 15-20 signals completely from analysis  
**Approved by:** User  
**Date:** 2026-03-21 00:49 GMT+7  
**Rationale:** 
- 345 signals only in MASTER
- 516 signals only in AUDIT  
- 707 status mismatches
- Daemon stopped writing to AUDIT around Mar 15
**Impact:** Lose 6 days of signal history, but eliminate contamination  
**No recovery:** Data too split to salvage

### Decision 3: FRESH START FROM MAR 21 (Option A)
**Decision:** Start completely new from Mar 21, don't salvage Mar 15-20  
**Approved by:** User  
**Date:** 2026-03-21 00:57 GMT+7  
**Alternative considered:** Option B (salvage Mar 15-20 as NEW_IMMUTABLE) - REJECTED  
**Rationale:** Clean architecture, no contamination, easier to implement  
**Impact:** Fresh accumulation from Mar 21 forward

### Decision 4: ARCHITECTURE
**Decision:** 
- SIGNALS_INDEPENDENT_AUDIT.txt = Immutable source of truth
- SIGNALS_MASTER.jsonl = Current status tracker
- Daemon writes to BOTH
- Executor reads AUDIT, updates MASTER
- Reporter reads BOTH, template locked

**Approved by:** User  
**Date:** 2026-03-21 00:37 GMT+7  
**Rationale:** Separates immutable history from mutable state  
**Impact:** Files won't diverge again, metrics won't corrupt

### Decision 5: IMMUTABILITY LEVELS
**Decision:**
- FOUNDATION: Locked forever
- NEW_IMMUTABLE: Locked after day ends
- NEW_LIVE: Open until day-end

**Approved by:** User  
**Date:** 2026-03-21 00:49 GMT+7  
**Rationale:** Rolling daily immutability, prevents retroactive changes  
**Impact:** Clean separation of past (locked) vs present (mutable)

---

## 📐 ARCHITECTURE (Locked)

### Files & Responsibilities

**SIGNALS_INDEPENDENT_AUDIT.txt**
- Immutable source of truth
- Append-only (never modified)
- Contains every fired signal
- Read by: Reporter (metrics), Executor (facts)
- Recovery use: Rebuild MASTER anytime

**SIGNALS_MASTER.jsonl**
- Current status tracker
- Status-only updates (OPEN → TP_HIT/SL_HIT/TIMEOUT)
- Updated by: Daemon (append), Executor (status update)
- Rebuild use: Expendable, rebuild from AUDIT

**Daemon (SmartFilter input)**
- Write to SIGNALS_INDEPENDENT_AUDIT.txt (mandatory)
- Write to SIGNALS_MASTER.jsonl (mandatory)
- Both writes must succeed or fail together

**Executor (Backtest engine)**
- Read SIGNALS_INDEPENDENT_AUDIT.txt (get signal facts)
- Update SIGNALS_MASTER.jsonl (status only)
- Never modify AUDIT
- Must be idempotent (safe to retry)

**Reporter**
- Read SIGNALS_INDEPENDENT_AUDIT.txt (immutable metrics)
- Read SIGNALS_MASTER.jsonl (current status)
- Merge both sources
- Template LOCKED, content dynamic

---

## 🔐 IMMUTABILITY CONTRACT (Locked)

### FOUNDATION (Feb 27 - Mar 14)
```
Status: LOCKED FOREVER
├─ Count: 2,224 signals
├─ signal_origin: "FOUNDATION"
├─ Can READ: Yes
├─ Can MODIFY: No (ever)
├─ Rule: Violation = System reset to zero baseline
└─ Source: SIGNALS_INDEPENDENT_AUDIT.txt only
```

### NEW_IMMUTABLE (Completed days after Mar 21)
```
Status: Locked after day 23:59:59 GMT+7
├─ Created daily: Day's NEW_LIVE becomes immutable at midnight
├─ signal_origin: "NEW_IMMUTABLE"
├─ Can MODIFY before lock: status only (OPEN → closed)
├─ CANNOT MODIFY ever: fired_time_utc, entry_price, signal_origin
├─ Rule: Violation = System reset to zero baseline
└─ Source: SIGNALS_INDEPENDENT_AUDIT.txt + SIGNALS_MASTER.jsonl
```

### NEW_LIVE (Current day accumulating)
```
Status: Open, growing
├─ Created daily at: 00:00:00 GMT+7
├─ signal_origin: "NEW_LIVE"
├─ Can UPDATE: status, actual_exit_price, pnl, closed_at
├─ CANNOT MODIFY: fired_time_utc, entry_price, signal_origin
├─ Locks at: 23:59:59 GMT+7 (becomes NEW_IMMUTABLE)
└─ Source: SIGNALS_INDEPENDENT_AUDIT.txt + SIGNALS_MASTER.jsonl
```

---

## 📈 ROLLING DAILY IMMUTABILITY (Locked)

```
Day N (e.g., Mar 21):
├─ 00:00:00 GMT+7
│  ├─ Fresh NEW_LIVE segment opens
│  └─ FOUNDATION remains locked
├─ During day
│  ├─ Daemon appends signals to BOTH files
│  └─ Executor backtests, updates MASTER status
├─ 23:59:59 GMT+7
│  └─ All signals finalized
└─ Midnight transition
   └─ Day N's NEW_LIVE → NEW_IMMUTABLE (LOCKED)

Day N+1 (e.g., Mar 22):
├─ 00:00:00 GMT+7
│  ├─ Day N now immutable (cannot change)
│  ├─ Fresh NEW_LIVE segment opens
│  └─ Report shows:
│     ├─ FOUNDATION (Mar 27-14, locked)
│     ├─ NEW_IMMUTABLE (Mar 21, locked)
│     └─ NEW_LIVE (Mar 22, accumulating)
└─ Repeat forever
```

---

## 📊 REPORTER OUTPUT (Locked)

### Structure
```
FOUNDATION BASELINE (Feb 27 - Mar 14)
├─ Total: 2,224 signals (LOCKED)
├─ Closed: X
├─ Win Rate: Y%
├─ P&L: $Z
├─ Avg duration TP: MM:SS
├─ Avg duration SL: MM:SS
└─ [LOCKED - Never changes]

NEW SIGNALS (Mar 21+, rolling daily)
├─ Mar 21
│  ├─ Total: N signals
│  ├─ Closed: M
│  ├─ WR: X%
│  ├─ P&L: $Y
│  └─ Status: Accumulating (locked at day-end)
├─ Mar 22
│  ├─ Previous day locked
│  ├─ Total: N signals
│  ├─ Closed: M
│  ├─ WR: X%
│  ├─ P&L: $Y
│  └─ Status: Accumulating (locks at day-end)
└─ [Continues daily]

COMPARISON
├─ Foundation WR vs NEW WR (progress metric)
└─ Foundation P&L vs NEW P&L (impact metric)
```

---

## 🏗️ BUILD PHASES (Awaiting Approval)

### Phase 1: Extract Clean Foundation
- Input: Current SIGNALS_MASTER.jsonl + SIGNALS_INDEPENDENT_AUDIT.txt
- Output: SIGNALS_FOUNDATION_CLEAN.jsonl (2,224 signals)
- Verify: Both files have exact same UUIDs for Feb 27 - Mar 14
- Status: AWAITING APPROVAL

### Phase 2: Lock Foundation Metadata
- Calculate: WR, P&L, avg duration for FOUNDATION
- Output: SIGNALS_FOUNDATION_LOCKED.json
- Tag: signal_origin = "FOUNDATION" for all 2,224 signals
- Status: Depends on Phase 1

### Phase 3: Rebuild AUDIT
- Remove: All Mar 15-20 signals from current AUDIT
- Keep: Only Feb 27 - Mar 14 signals
- Verify: Count = 2,224
- Output: Clean SIGNALS_INDEPENDENT_AUDIT.txt
- Status: Depends on Phase 1

### Phase 4: Reset MASTER
- Keep: FOUNDATION section
- Remove: Mar 15-20 section
- Prepare: Empty NEW_LIVE ready for Mar 21+
- Output: Clean SIGNALS_MASTER.jsonl
- Status: Depends on Phase 1

### Phase 5: Lock Reporter Template
- Freeze: Reporter structure (never change again)
- Dynamic: All metrics calculated from AUDIT
- Output: Updated pec_enhanced_reporter.py
- Status: Depends on Phase 1

### Phase 6: Verify System
- Check: AUDIT and MASTER alignment for FOUNDATION
- Check: All signals have signal_origin field
- Check: Foundation metrics locked
- Check: Daemon/Executor/Reporter configured correctly
- Status: Depends on Phase 1

---

## ❓ READY TO BUILD?

**Status:** All decisions locked, awaiting final approval

**To proceed:**
1. Confirm this summary is complete and accurate
2. Approve Phase 1 (extract clean foundation)
3. Execute all 6 phases sequentially
4. Go live: Mar 21 fresh start

**Do you approve proceeding with the build?**
