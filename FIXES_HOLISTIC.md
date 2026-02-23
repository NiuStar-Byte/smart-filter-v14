# HOLISTIC FIXES - Debug Spam + Duplicate Signals

## ROOT CAUSES IDENTIFIED

### 1. DEBUG FILE SPAM (10+ files)
**Problem:** The `valid_debugs` list has multiple gates `if len(valid_debugs) < 2` but they're spread across:
- 15min branch (line 381)
- 30min branch (line 467)
- 1h branch (line 625)

Each branch INDEPENDENTLY checks `if len(valid_debugs) < 2`, but they all ADD to the SAME list in ONE run_cycle() call.

**Example:** If all 3 timeframes fire in same cycle:
- 15min: len(valid_debugs)=0, adds 1 debug → now len=1
- 30min: len(valid_debugs)=1, adds 1 debug → now len=2
- 1h: len(valid_debugs)=2, gate BLOCKS (len NOT < 2) → stays len=2

But if multiple SYMBOLS fire in sequence:
- Symbol 1 15min: adds debug → len=1
- Symbol 1 30min: adds debug → len=2
- Symbol 2 15min: gate BLOCKS (len NOT < 2) ✓ WORKING
- Symbol 2 30min: gate BLOCKS ✓ WORKING

**ACTUAL ISSUE:** The gate is working per-symbol, but if signals are being fired from DIFFERENT runs or if CYCLE_SLEEP is not being enforced, multiple cycles can run quickly and each adds 2 debugs = 10+ files total.

### 2. DUPLICATE SIGNALS (Same signal twice)
**Problem:** Same signal (78.A ASTER-USDT 15min) fired twice with slightly different entry prices (0.691391 vs 0.691791).

**Hypothesis:** Either:
1. SmartFilter.analyze() is being called twice per bar
2. get_live_entry_price() is called twice and returns different prices
3. run_cycle() is being called multiple times per CYCLE_SLEEP interval
4. Signal deduplication is missing (same symbol+tf+bar can fire twice)

---

## FIXES TO IMPLEMENT

### FIX 1: Debug File Gate (Global Level)
**Location:** main.py, inside run_cycle()
**Change:** Move the 2-file gate to AFTER all timeframe processing, globally
- Current: Each TF branch checks independently
- New: Global gate BEFORE adding ANY debug object

```python
# CURRENT (inside 15min/30min/1h blocks):
if len(valid_debugs) < 2:
    valid_debugs.append({...})

# NEW (once per run_cycle):
def run_cycle():
    valid_debugs = []
    GLOBAL_DEBUG_LIMIT = 2
    # ... process all TFs ...
    # Before appending ANY debug:
    if len(valid_debugs) < GLOBAL_DEBUG_LIMIT:
        valid_debugs.append({...})
```

### FIX 2: Signal Deduplication (signal_store.py)
**Location:** signal_store.py, append_signal()
**Change:** Check if signal with same (symbol, timeframe, entry_price) already exists in current cycle
- Use a bloom filter or simple check: last 10 signals
- Skip if exact duplicate (symbol+tf+entry within last 60 seconds)

### FIX 3: CYCLE_SLEEP Enforcement (main.py)
**Location:** main.py, run() function
**Change:** Verify CYCLE_SLEEP is being respected
- Add timestamp logging to detect if cycles run faster than CYCLE_SLEEP
- Add lock mechanism to prevent overlapping cycles

### FIX 4: Entry Price Consistency (main.py)
**Location:** main.py, 15min/30min/1h blocks
**Change:** Cache entry_price to ensure get_live_entry_price() is called ONCE per signal
- Move get_live_entry_price() call OUTSIDE of conditional logic
- Store result and reuse throughout signal processing

---

## IMPLEMENTATION ORDER

1. **FIX 3 first (Enforcement):** Add logging to verify CYCLE_SLEEP is working
2. **FIX 4 next (Entry Price):** Ensure consistent entry price throughout signal processing
3. **FIX 1 then (Debug Gate):** Move gate to global level
4. **FIX 2 last (Deduplication):** Add signal deduplication in signal_store

---

## CRITICAL CODE SECTIONS

### Current Problematic Code (main.py, 15min block)
```python
if len(valid_debugs) < 2:  # ← GATE IS HERE
    valid_debugs.append({...})

# ... later in same TF block ...

entry_price_raw = get_live_entry_price(...)  # ← CALLED ONCE
# ... coercion ...
entry_price = float(entry_price_raw)

# ... even later ...
send_telegram_alert(..., price=entry_price, ...)  # ← USES CACHED VALUE ✓
```

**Suspicious patterns:**
- Three separate TF blocks each with own debug gate
- If all fire: 2+2+2 = 6 signals (but limit is 2)
- Unless there's branching that adds more...

### Need to check:
1. Are 30min/1h blocks ACTUALLY commented out?
2. Is there a retry/exception handler that re-runs the block?
3. Is run_cycle() called more than once per CYCLE_SLEEP?

---

## NEXT STEPS

1. Add extensive logging to run_cycle() to detect duplicates
2. Check if 30min/1h sends are truly disabled
3. Implement global debug gate
4. Implement signal deduplication in signal_store.py
5. Push to Railway and test
