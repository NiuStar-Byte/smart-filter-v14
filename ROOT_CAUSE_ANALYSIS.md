# 🚨 ROOT CAUSE: 818 STUCK OPEN SIGNALS (Foundation 32.54% vs New 28.10% WR)

## The Real Problem (Not What We Thought)

### What's Happening
**818 OPEN signals are stuck in the system** — fired between Mar 1-18 but NEVER CLOSED.
- Oldest signal: **March 1, 13:54 UTC** (401+ hours ago / 16+ days old)
- Newest signal: March 18, 07:11 UTC (just 7 hours old)
- **TP/SL targets are identical** (e.g., 0.18/0.18), suggesting calculation broken

### Why This Invalidates WR Comparison
```
Foundation WR = 32.54% = (347 TP + 90 TIMEOUT_WIN) / 1,343 CLOSED
New WR = 28.10% = (86 TP + ... ) / 306 CLOSED

PROBLEM: Foundation signals have CLOSED. New signals haven't.
RESULT: Apples-to-oranges comparison.
SOLUTION: Fix signal closing mechanism FIRST, then re-measure.
```

---

## Symptom Evidence

### 1. **818 OPEN Signals (Should Be ~10-20 Max)**
```
Total OPEN signals: 818
Oldest: ARK-USDT 30min from 2026-03-01 13:54 (401+ hours = 16+ days old)
Newest: BIO-USDT 1h from 2026-03-18 07:11 (7 hours old)
```

### 2. **TP/SL Bug: Identical Targets**
```
ARK-USDT 30min: TP=0.18, SL=0.18 ← Same price! Can't be right.
ARK-USDT 1h: TP=0.18, SL=0.18 ← Same price!
ARK-USDT 30min: TP=0.17, SL=0.18 ← Backwards? SL > TP for LONG?
```

### 3. **Score Distribution (Only OPEN)**
```
OPEN signals only: score ranges 12-15 (min_score = 12)
NO closed signals from new batch (Mar 16+)
```

---

## Why This Matters

### Hypothesis A: TP/SL Calculation Broken
- calculate_tp_sl() is returning wrong values
- Targets are unreachable (or identical, or inverted)
- Positions never hit TP or SL → stuck OPEN forever

### Hypothesis B: Trade Execution Tracking Broken
- Signals fire correctly
- Trades execute
- But status update fails (never marks as TP/SL)
- Daemon loses connection to exchange or doesn't track closure

### Hypothesis C: RR Filter Blocking New Signals
- New signals have correct TP/SL
- But achieved_rr is too low
- RR_FILTER rejects before firing
- Signal never gets sent → never tracked

---

## Impact on WR Analysis

**Current Situation:**
- Foundation: 1,343 signals, mostly CLOSED → WR = 32.54% (VALID)
- New (Mar 16+): 306 CLOSED + 818 OPEN → WR = 28.10% (INVALID)

**Why Invalid:**
- 818 OPEN signals haven't had chance to hit TP/SL
- Some may eventually TP (raising WR)
- Some may SL (lowering WR)
- Current 28.10% is **premature** and not comparable

**True New WR Unknown Until:**
1. All 818 signals close (natural market movement)
2. OR signals timeout (typically 4-6h for intraday)
3. OR system manually closes them for analysis

---

## What's Wrong in the Code

### File: tp_sl_retracement.py
```python
def calculate_tp_sl(df, entry_price, signal_type, regime=None):
    # Issue: Returns dict with tp/sl targets
    # Bug: tp and sl might be calculated wrong
    # Evidence: TP=0.18, SL=0.18 (identical = bug)
```

### File: main.py (signal closure tracking)
```python
# Issue: No code to UPDATE signal status from OPEN to TP/SL/TIMEOUT
# Bug: status stays OPEN forever even after trade closes on exchange
# Evidence: 16-day-old signals still marked OPEN
```

### File: signal_sent_tracker.py or execution tracking
```python
# Issue: Should check exchange order status (filled? cancelled?)
# Bug: No periodic polling or webhook to detect order closure
# Evidence: Only way signals are CLOSED is if PEC marks them manually
```

---

## Immediate Actions Required

### 1. **DIAGNOSTIC: Check tp_sl_retracement.py**
```bash
# Are TP/SL targets being calculated correctly?
# Sample a few signals:
python3 << 'SCRIPT'
import json
signals_with_bad_targets = 0
with open('SENT_SIGNALS.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        if d['status'] == 'OPEN' and abs(d['tp_target'] - d['sl_target']) < 0.001:
            signals_with_bad_targets += 1
            if signals_with_bad_targets <= 5:
                print(f"{d['symbol']} {d['timeframe']}: TP={d['tp_target']:.8f}, SL={d['sl_target']:.8f}")
print(f"Total signals with TP≈SL: {signals_with_bad_targets}")
SCRIPT
```

### 2. **CRITICAL: Implement Signal Closure Tracking**
```python
# In main.py or new file:
# - Poll exchange order status every 30 minutes
# - Update SENT_SIGNALS.jsonl status from OPEN → TP/SL
# - Log closure events
# - Mark old signals as TIMEOUT after 6h
```

### 3. **FALLBACK: Force-Close Old Signals for Analysis**
```python
# For signals older than 6 hours with OPEN status:
# - Mark as TIMEOUT_CLOSE (treat as TP for conservative WR)
# - Re-calculate WR on closed-only signals
# - Compare Foundation vs New on same basis
```

### 4. **RESET: Clear Bad Signal Data**
```python
# Once fix is implemented:
# - Delete/archive old 818 signals (can't trust them)
# - Start fresh counting from Mar 18 14:00 with fixed system
# - Run for 48 hours, then compare WR
```

---

## Decision Tree

**Option A: Quick Fix (2 hours)**
```
1. Implement basic signal closure polling
2. Mark old OPEN signals as TIMEOUT
3. Recalculate WR on CLOSED signals only
4. Compare Foundation vs New fairly
Result: Get reliable WR comparison
Risk: May lose 818 signals data (acceptable)
```

**Option B: Deep Fix (6-8 hours)**
```
1. Debug tp_sl_retracement.py calculation
2. Fix TP/SL target calculation
3. Implement persistent trade tracking (exchange orders)
4. Auto-update signal status on closure
5. Backfill old 818 signals with correct closure data
Result: Preserve all data + future signals auto-close
Risk: High (complex debugging)
Value: Worth it if we can save the 818 signals
```

**Option C: Restart Clean (1 hour)**
```
1. Kill daemon
2. Archive old signal data
3. Restart daemon with fresh SENT_SIGNALS.jsonl
4. Run for 48h with new clean baseline
5. Compare WR on new data
Result: Clean comparison, but lose 818 signals
Risk: Low (safe reset)
```

---

## Recommended Path

**START WITH OPTION A (Quick Fix):**
1. Implement closure polling (30 min timeout = 6h old = TIMEOUT)
2. Force-close 818 old signals as TIMEOUT
3. Recalculate Foundation vs New WR fairly
4. Identify if Phase 2 is actually the problem
5. IF gates are still issue → then apply Option B (deep fix)
6. IF gates were false alarm → gates are actually OK

**Then proceed to Option B IF NEEDED** (after diagnosing gates properly)

---

## Why This Happened

**Timeline:**
- Mar 1-10: Daemon accumulating signals (mixed closed/open)
- Mar 10: Daemon crashes, watchdog doesn't restart it
- Mar 16: Daemon restarted manually
- Mar 16+: New signals fire BUT closure tracking broken
- Mar 18: We have 818 stuck OPEN signals

**Root:** Signal closure tracking was never implemented. Signals only "close" when PEC system marks them. No automation.

---

## Next Step

Run Option A diagnostic now (30 min), then decide deep fix vs restart.
