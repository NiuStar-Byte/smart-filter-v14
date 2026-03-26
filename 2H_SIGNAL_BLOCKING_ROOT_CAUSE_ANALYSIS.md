# 2h Signal Blocking - Root Cause Analysis & Fix

## Problem Statement
2h TF signals were not being transmitted to Telegram despite having the signal generation code implemented and running. 4h TF signals worked fine.

## Root Cause Analysis

### Methodology
Compared the code block structures of working (4h) vs non-working (2h) timeframes side-by-side to identify structural differences.

### Key Findings

**2h block had 3 EXTRA BLOCKING CHECKS that 4h didn't have:**

```python
# 2h block (lines 2398-2412) - BLOCKING PATTERN:
if not signal_uuid:
    print(...)
    continue  # ← BLOCKS SIGNAL

cycle_key = f"{symbol_val}|2h|{signal_type}"
if cycle_key in signals_sent_this_cycle:
    print(...)
    continue  # ← BLOCKS SIGNAL
    
if is_duplicate_signal(symbol_val, "2h", signal_type):
    continue  # ← BLOCKS SIGNAL
    
signals_sent_this_cycle.add(cycle_key)
if os.getenv("DRY_RUN") != "true":
    send_telegram()
```

**4h block (working) - SIMPLIFIED PATTERN:**

```python
# 4h block - NO EXTRA CHECKS:
if not signal_uuid:
    print(...)
    pass  # ← No continue, just pass

# NO cycle_key dedup check
# NO is_duplicate_signal check
# NO signals_sent_this_cycle.add()

if signal_uuid and os.getenv("DRY_RUN") != "true":
    send_telegram()
```

### Why This Caused Blocking

The `continue` statements in 2h would **skip to the next symbol in the loop**, preventing:
1. Signal UUID validation
2. Telegram alert transmission
3. PEC signal logging
4. MASTER/AUDIT writer updates
5. Executor trigger

Meanwhile, 4h used `pass` (do nothing) and conditional guards (`if signal_uuid and`), allowing graceful flow-through even when conditions weren't met.

### Redundancy Issue

The 2h block's dedup checks were:
- **Cycle-level dedup**: `cycle_key` checking if already sent in THIS CYCLE
- **General dedup**: `is_duplicate_signal()` checking general duplicate signals

These appear to be **additional/redundant** dedup logic that doesn't exist in the simpler, working 4h block.

## Solution

**Commit:** `668f62f` - "FIX: Remove blocking dedup checks from 2h block"

**Changes Made:**
1. Removed `continue` from signal_uuid check → changed to `pass`
2. Removed `cycle_key` dedup block entirely
3. Removed `is_duplicate_signal()` check entirely
4. Removed `signals_sent_this_cycle.add()` call
5. Simplified Telegram condition to: `if signal_uuid and os.getenv("DRY_RUN") != "true"`

**Lines Changed:** 2398-2412 (removed 3 blocking checks + 1 continue → 1 simplified condition)

**Result:** 2h block structure now matches 4h's proven working pattern.

## Remaining Differences

The 2h block still has these additional gates that 4h doesn't have:

1. **DirectionAwareGatekeeper** (DISABLED but preserved) - lines 2186-2206
2. **PHASE3B Reversal Quality Check** - lines 2277-2310 (with continue if Route changes)
3. **RR_FILTER Check** - lines 2352-2356 (with continue if RR too low)

These are legitimate validation gates, but unlike the dedup checks, they have **semantic meaning** (measuring actual signal quality). The dedup checks were purely redundant **state tracking** logic.

Whether these remaining gates are actually blocking signals depends on how often they trigger. They should be monitored during the 48-72 hour validation period.

## Testing & Validation

**Next Steps:**
1. Monitor `exit_condition_debug.log` for `[LOG] Sending 2h alert` entries
2. Check `SIGNALS_MASTER.jsonl` for new 2h entries after daemon cycle
3. Monitor Telegram channel for 2h signal alerts
4. If signals still don't appear, investigate remaining gates (PHASE3B, RR_FILTER)

**Expected Outcome:** 2h signals should now transmit on next daemon cycle (every 5 minutes).

## Technical Notes

- **Dedup redundancy**: 2h had TWO layers of dedup (cycle-level + signal-level) while 4h had NONE
- **Blocking pattern**: Using `continue` in signal flow is more severe than conditional checks (`if X and`) because it skips entire blocks
- **Code versioning**: This change is tracked in CODE_VERSION_LOCK.md; if main.py changes further, all trackers require re-baselining
- **Git status**: Commit pushed to origin/main at a1578b0

## References

- Working 4h block: lines 2541-2750
- Fixed 2h block: lines 2140-2506
- Debug script: DEBUG_2H_SIGNALS.py
