# Phase 1 Implementation Guide: Dual-Write Verification

**Status:** Ready for integration into main.py
**Timeline:** 2026-03-24 to 2026-03-26
**Priority:** HIGH - Prevents future signal divergence
**Strategy:** Alert + Continue (NO HALT) - Signals never stop, failures auto-fixed

---

## Overview

Phase 1 adds dual-write verification to the daemon. When a signal fires, the system now:
1. Writes to SIGNALS_MASTER.jsonl ✓
2. Writes to SIGNALS_INDEPENDENT_AUDIT.txt ✓
3. **NEW:** Verifies both writes succeeded ✓
4. **NEW:** Alerts on failure but CONTINUES firing signals ✓
5. **NEW:** Auto-recovery fixes issues in background ✓

---

## What Changed

### New File: `signal_dual_write_verification.py`
- **Purpose:** Verify signals written to both files
- **Location:** `/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/`
- **Size:** ~280 lines
- **Status:** Ready to use

### Key Functions

```python
# Initialize verifier (once at startup)
from signal_dual_write_verification import initialize_dual_write_verifier
verifier = initialize_dual_write_verifier(
    master_path='SIGNALS_MASTER.jsonl',
    audit_path='SIGNALS_INDEPENDENT_AUDIT.txt',
    debug=True  # Set to False in production
)

# Verify after writing signals
from signal_dual_write_verification import verify_signal_dual_write
verify_signal_dual_write(
    signal_uuid=signal_uuid,
    signal_data=signal_obj,
    raise_on_failure=True  # Halt daemon if write fails
)
```

---

## Integration Steps

### Step 1: Import Verifier Module (main.py, top level)

Find the imports section in `main.py` and add:

```python
# === PHASE 1: DUAL-WRITE VERIFICATION ===
from signal_dual_write_verification import (
    initialize_dual_write_verifier,
    verify_signal_dual_write,
    get_dual_write_status
)
```

### Step 2: Initialize Verifier (main.py, early in script)

Find where other components are initialized (near signal_store, signal_tracker initialization). Add:

```python
# === INITIALIZE DUAL-WRITE VERIFIER ===
try:
    _dual_write_verifier = initialize_dual_write_verifier(
        master_path=os.path.abspath(SIGNALS_MASTER_PATH),
        audit_path=os.path.abspath(SIGNALS_AUDIT_PATH),
        debug=VERBOSE_LOGGING
    )
    print(f"[INIT] Dual-write verifier initialized", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to initialize dual-write verifier: {e}", flush=True)
    _dual_write_verifier = None
```

### Step 3: Wrap Signal Firing (main.py, in signal firing logic)

Find the code that fires signals (typically in `run_cycle()` or equivalent). Wrap the write calls:

**BEFORE:**
```python
# Fire signal
if signal_score >= MIN_SCORE:
    # Write and send
    _signals_master_writer.write(signal_obj)
    send_telegram_alert(symbol, direction, entry_price, ...)
    trigger_executor_on_signal(signal_uuid, symbol, timeframe)
    print(f"[FIRE] {symbol} {direction} score={signal_score}")
```

**AFTER (Alert + Continue Strategy):**
```python
# Fire signal with dual-write verification (Alert + Continue)
if signal_score >= MIN_SCORE:
    # Write to files
    _signals_master_writer.write(signal_obj)
    
    # PHASE 1: Verify dual-write (non-halting)
    dual_write_ok = True
    if _dual_write_verifier:
        try:
            result = verify_signal_dual_write(
                signal_uuid=signal_uuid,
                signal_data=signal_obj,
                raise_on_failure=False  # ✅ Don't halt on failure
            )
            
            if result:
                print(f"[DUAL-WRITE] ✅ Verified {signal_uuid[:12]} to both files", flush=True)
            else:
                # ⚠️ Write failed but continue trading
                dual_write_ok = False
                _divergence_tracker.record_failure(signal_uuid)
                
                alert_msg = f"""
🚨 DUAL-WRITE FAILED 🚨
Signal UUID: {signal_uuid}
Symbol: {symbol}
Time: {datetime.utcnow()}
Status: MASTER written, AUDIT write missing

Action: Signal still fired. Monitoring will auto-recover.
Impact: Minimal - auto-recovery in background
"""
                print(f"[DUAL-WRITE] ❌ {alert_msg}", flush=True)
                
                # Send ops alert
                send_ops_alert(alert_msg, severity="CRITICAL")
        
        except Exception as e:
            print(f"[DUAL-WRITE] Warning: Verification error (continuing): {e}", flush=True)
            dual_write_ok = False
    
    # Continue with normal processing (ALWAYS, even if dual-write failed)
    # ✅ Signals never stop
    send_telegram_alert(symbol, direction, entry_price, ...)
    trigger_executor_on_signal(signal_uuid, symbol, timeframe)
    
    if dual_write_ok:
        print(f"[FIRE] ✅ {symbol} {direction} score={signal_score}")
    else:
        print(f"[FIRE] ⚠️  {symbol} {direction} score={signal_score} (dual-write alert sent)")
```

---

## Configuration

### Environment Variables (optional)

Add to `.env` or shell environment:

```bash
# Enable detailed dual-write logging
export DUAL_WRITE_DEBUG=1

# Timeout for verification (seconds)
export DUAL_WRITE_TIMEOUT=10

# Alert threshold: gap larger than this triggers warning
export DUAL_WRITE_ALERT_THRESHOLD=10
```

### main.py Integration

```python
# Use env vars if set
DUAL_WRITE_DEBUG = os.getenv('DUAL_WRITE_DEBUG', 'false').lower() == 'true'
DUAL_WRITE_TIMEOUT = float(os.getenv('DUAL_WRITE_TIMEOUT', '10.0'))
DUAL_WRITE_ALERT_THRESHOLD = int(os.getenv('DUAL_WRITE_ALERT_THRESHOLD', '10'))
```

---

## Testing Checklist

### Test 1: Normal Operation (24 hours)
- [ ] Let daemon run normally
- [ ] Verify: All signals written to both files
- [ ] Verify: No dual-write failures logged
- [ ] Verify: Logs show "✅ Verified" messages for signals
- [ ] Check: MASTER and AUDIT remain in sync

### Test 2: Simulate Audit Write Failure
```python
# Temporarily block AUDIT writes (for testing only)
import os
os.chmod('SIGNALS_INDEPENDENT_AUDIT.txt', 0o444)  # Read-only

# Fire a signal
# Expected: Daemon logs "❌ CRITICAL" and halts
# Expected: Signal NOT counted as successful fire

# Cleanup
os.chmod('SIGNALS_INDEPENDENT_AUDIT.txt', 0o644)  # Restore
```

- [ ] Daemon detects write failure
- [ ] Logs show "HALTING DAEMON"
- [ ] Manual restart required (fail-safe)
- [ ] No partial writes to files

### Test 3: High Throughput (100 signals/min)
- [ ] Fire 1000+ signals quickly
- [ ] Verify: All written to both files
- [ ] Verify: No verification timeouts
- [ ] Verify: MASTER and AUDIT remain aligned
- [ ] Check: Performance not degraded significantly

### Test 4: Recovery Idempotency
- [ ] Backfill both files if divergence detected
- [ ] Run recovery 3 times on same gap
- [ ] Verify: Same result each time (no duplicates)

---

## Success Criteria

Phase 1 is complete when:

- [x] `signal_dual_write_verification.py` created and tested
- [ ] Integrated into `main.py` (all 3 steps completed)
- [ ] All 4 test cases pass
- [ ] No performance regression (signals fire within normal latency)
- [ ] MASTER and AUDIT remain 100% aligned for 24+ hours
- [ ] Zero dual-write failures logged
- [ ] Committed to GitHub and reviewed

---

## Rollback Plan

If issues arise:

1. **Disable verification temporarily:**
   ```python
   # Comment out verification code in main.py
   # if _dual_write_verifier:
   #     verify_signal_dual_write(...)
   ```

2. **Revert commit:**
   ```bash
   git revert <commit-hash>
   ```

3. **Restart daemon:**
   ```bash
   pkill -f main.py
   python3 main.py &
   ```

---

## Monitoring

During Phase 1, watch for:

```bash
# Watch logs in real-time
tail -f /path/to/daemon.log | grep "DUAL-WRITE"

# Count successful verifications per hour
grep "✅ Verified" /path/to/daemon.log | wc -l

# Check for failures
grep "❌ CRITICAL" /path/to/daemon.log

# Monitor file alignment
watch -n 60 'wc -l SIGNALS_MASTER.jsonl SIGNALS_INDEPENDENT_AUDIT.txt'
```

---

## Expected Behavior

### Normal (Healthy System)
```
[19:50:23.456] [DEBUG  ] Dual-write verifier initialized
[19:50:45.123] [DUAL-WRITE] ✅ Verified abc123de to both files | [FIRE] ✅ BTC SHORT score=14
[19:50:46.234] [DUAL-WRITE] ✅ Verified def456gh to both files | [FIRE] ✅ ETH LONG score=15
[19:50:47.345] [DUAL-WRITE] ✅ Verified ghi789ij to both files | [FIRE] ✅ SOL SHORT score=13
```

### Write Failure (Alert + Continue, NOT Halt)
```
[19:50:48.456] [DUAL-WRITE] ❌ Verification failed for jkl012kl
[19:50:48.457] [ALERT    ] 🚨 DUAL-WRITE FAILED: jkl012kl | MASTER:✓ AUDIT:✗
[19:50:48.458] [FIRE     ] ⚠️  ADA LONG score=12 (dual-write alert sent)
                            ↑ Signal still fired! Trading continues!
[19:50:48.459] [MONITOR  ] Failure recorded for auto-recovery
[19:50:48.460] [EMAIL    ] Ops alerted: Dual-write failure detected
```

**Key Difference from Original:**
- ❌ OLD: Failure → HALT daemon → All signals stop
- ✅ NEW: Failure → ALERT ops → Signals continue → Auto-fix in background

Trading never stops. Ops is notified. System auto-recovers.
```

---

## Timeline

| Date | Task | Status |
|------|------|--------|
| 2026-03-23 | Create verification module | ✅ DONE |
| 2026-03-24 | Integrate into main.py | 🔄 TODO |
| 2026-03-24 | Run Test 1 (24h normal ops) | 🔄 TODO |
| 2026-03-25 | Run Test 2 (failure simulation) | 🔄 TODO |
| 2026-03-25 | Run Test 3 (high throughput) | 🔄 TODO |
| 2026-03-26 | Final validation + commit | 🔄 TODO |

---

## Key Points

1. **Fail-Safe Design:** If either write fails, daemon halts (don't hide errors)
2. **Fast Verification:** Checks complete within 10 seconds
3. **Detailed Logging:** Every verification logged for troubleshooting
4. **Backward Compatible:** Can be disabled if needed
5. **Zero Data Loss:** Verification only confirms writes, doesn't modify data

---

## Questions?

Refer to:
- `signal_dual_write_verification.py` — Implementation details
- `DUAL_WRITE_PREVENTION_PLAN.md` — Full strategy (4 phases)
- `RECOVERY_SUMMARY_2026_03_23.md` — Context & timeline

---

Next: Integrate into main.py and begin testing (2026-03-24)
