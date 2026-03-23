# Dual-Write Verification & Prevention Plan

**Document Created:** 2026-03-23 18:55 GMT+7
**Recovery Completed:** YES (Perfect alignment restored at 4,672 signals)
**Status:** Implementation plan for preventing future failures

---

## Root Cause Analysis

### What Failed
The daemon (`main.py`) writes signals to two files simultaneously:
```python
# Expected behavior:
_signals_master_writer.write(signal)       # → SIGNALS_MASTER.jsonl ✅
_audit_writer.write(signal)                # → SIGNALS_INDEPENDENT_AUDIT.txt ⚠️ FAILED
```

### Evidence of Failure
- **Start date:** ~2026-03-21 (when NEW_LIVE began)
- **Duration:** 48+ hours
- **Impact:** 1,710 signals written to MASTER but NOT to AUDIT
- **Gap:** AUDIT missing 66% of NEW_LIVE signals (1,710 of 2,447)

### Why It Happened
1. **No error handling** — both writes silently fail separately
2. **No verification** — no check if both writes succeeded
3. **No logging** — failures are invisible
4. **No retry** — failed writes just disappear
5. **No monitoring** — no alert when files diverge

---

## Prevention Strategy

### Layer 1: Dual-Write Verification (Immediate)
**File:** `main.py` signal firing logic
**Change:** Verify BOTH writes succeed before confirming signal

```python
def fire_signal_safe(signal_uuid, signal_obj):
    """
    Fire signal with guaranteed dual-write verification
    """
    try:
        # Write to MASTER
        write_master_result = _signals_master_writer.write(signal_obj)
        if not write_master_result:
            raise Exception("MASTER write failed")
        
        # Write to AUDIT
        write_audit_result = _audit_writer.write(signal_obj)
        if not write_audit_result:
            raise Exception("AUDIT write failed")
        
        # BOTH succeeded
        print(f"[DUAL-WRITE] ✅ Signal {signal_uuid} written to BOTH files")
        return True
    
    except Exception as e:
        print(f"[DUAL-WRITE] ❌ FAILURE for {signal_uuid}: {e}")
        print(f"[DUAL-WRITE] ALERT: Partial write detected - check files for corruption")
        
        # DO NOT CONTINUE - fail safely
        raise RuntimeError(f"Dual-write failure: {e}")
```

**Implementation:**
1. Add try-catch around both write calls
2. Check return values (not just exceptions)
3. Log all write attempts
4. Raise error if either fails
5. Halt daemon on dual-write failure (fail-safe)

---

### Layer 2: Write Verification Function
**File:** `signal_store.py` or new `signal_verification.py`

```python
def verify_dual_write(signal_uuid, timeout_sec=10):
    """
    Verify that a signal was written to BOTH files
    Returns: (in_master, in_audit, match_content)
    """
    import time
    
    start = time.time()
    while time.time() - start < timeout_sec:
        # Check MASTER
        with open(SIGNALS_MASTER_PATH, 'r') as f:
            master_uuids = set()
            for line in f:
                try:
                    sig = json.loads(line.strip())
                    if sig.get('signal_uuid') == signal_uuid:
                        master_uuids.add(signal_uuid)
                        master_sig = sig
                        break
                except:
                    pass
        
        # Check AUDIT
        with open(SIGNALS_AUDIT_PATH, 'r') as f:
            audit_uuids = set()
            for line in f:
                try:
                    sig = json.loads(line.strip())
                    if sig.get('signal_uuid') == signal_uuid:
                        audit_uuids.add(signal_uuid)
                        audit_sig = sig
                        break
                except:
                    pass
        
        # Verify both exist and match
        if signal_uuid in master_uuids and signal_uuid in audit_uuids:
            # Check key fields match (uuid, symbol, direction, entry_price, etc.)
            match = (
                master_sig.get('symbol') == audit_sig.get('symbol') and
                master_sig.get('direction') == audit_sig.get('direction') and
                master_sig.get('entry_price') == audit_sig.get('entry_price')
            )
            return True, True, match
        
        # If not found, retry
        time.sleep(0.5)
    
    # Timeout or not found
    return signal_uuid in master_uuids, signal_uuid in audit_uuids, False

# Usage in fire_signal_safe():
write_master_result = _signals_master_writer.write(signal_obj)
write_audit_result = _audit_writer.write(signal_obj)

# Verify within 10 seconds
in_master, in_audit, match = verify_dual_write(signal_uuid, timeout_sec=10)
if not (in_master and in_audit and match):
    raise RuntimeError(f"Dual-write verification failed: MASTER={in_master}, AUDIT={in_audit}, Match={match}")
```

**Benefits:**
- Catches silent failures
- Verifies content matches
- Retries on transient failures
- Fails loudly if something's wrong

---

### Layer 3: Real-Time Divergence Monitoring
**File:** New `signal_sync_monitor.py`

```python
def monitor_signal_sync(check_interval_sec=300):
    """
    Monitor MASTER/AUDIT files for divergence
    Runs every 5 minutes in background
    Alerts if gap detected
    """
    import threading
    import time
    
    def check_sync():
        while True:
            try:
                master_count = count_file_lines('SIGNALS_MASTER.jsonl')
                audit_count = count_file_lines('SIGNALS_INDEPENDENT_AUDIT.txt')
                
                diff = master_count - audit_count
                pct = (diff / audit_count * 100) if audit_count > 0 else 0
                
                if diff > 10:  # Threshold: more than 10 signals divergence
                    log_alert(f"""
                    🚨 DUAL-WRITE DIVERGENCE DETECTED 🚨
                    Timestamp: {datetime.now()}
                    MASTER: {master_count} lines
                    AUDIT: {audit_count} lines
                    Gap: {diff} signals ({pct:.1f}%)
                    
                    Action: Review daemon logs and signal writes
                    Severity: HIGH - gap is growing
                    """)
                else:
                    log_debug(f"✅ Files in sync (MASTER={master_count}, AUDIT={audit_count})")
                
                time.sleep(check_interval_sec)
            except Exception as e:
                log_error(f"Monitor error: {e}")
                time.sleep(60)
    
    # Run in background thread
    monitor_thread = threading.Thread(target=check_sync, daemon=True)
    monitor_thread.start()
```

**Benefits:**
- Detects divergence in real-time
- Alerts before gap gets large
- Runs continuously in background
- Can trigger auto-recovery if needed

---

### Layer 4: Automatic Recovery Trigger
**File:** `signal_sync_monitor.py` (extended)

```python
def auto_recover_if_diverged(max_allowed_gap=50):
    """
    If divergence detected > threshold, auto-trigger recovery
    """
    master_count = count_file_lines('SIGNALS_MASTER.jsonl')
    audit_count = count_file_lines('SIGNALS_INDEPENDENT_AUDIT.txt')
    gap = master_count - audit_count
    
    if gap > max_allowed_gap:
        log_alert(f"Gap {gap} exceeds threshold {max_allowed_gap} - triggering auto-recovery")
        
        # Execute recovery script
        subprocess.run([
            'python3',
            '/Users/geniustarigan/.openclaw/workspace/scripts/auto_recover_dual_write.py'
        ])
        
        log_info("Auto-recovery completed")
        return True
    
    return False
```

---

### Layer 5: Cron Job Checkpoint (Hourly)
**File:** New cron job - `check_dual_write_sync.sh`

```bash
#!/bin/bash
# Run hourly to verify alignment

MASTER_COUNT=$(wc -l < SIGNALS_MASTER.jsonl)
AUDIT_COUNT=$(wc -l < SIGNALS_INDEPENDENT_AUDIT.txt)
GAP=$((MASTER_COUNT - AUDIT_COUNT))

LOG_FILE="/tmp/dual_write_sync.log"

if [ $GAP -gt 5 ]; then
    echo "[$(date)] 🚨 DIVERGENCE: MASTER=$MASTER_COUNT AUDIT=$AUDIT_COUNT GAP=$GAP" >> $LOG_FILE
    
    # Send alert
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🚨 Dual-write divergence: gap=$GAP signals\"}" \
        $SLACK_WEBHOOK_URL  # Optional: send to Slack
else
    echo "[$(date)] ✅ SYNC OK: MASTER=$MASTER_COUNT AUDIT=$AUDIT_COUNT" >> $LOG_FILE
fi
```

**Cron entry:**
```
0 * * * * /path/to/check_dual_write_sync.sh
```

---

## Implementation Roadmap

### Phase 1: Immediate (This Week - 2026-03-24 to 2026-03-26)
- [ ] Add dual-write verification function to `main.py`
- [ ] Create `signal_verification.py` with verify_dual_write()
- [ ] Test on live signals for 24 hours
- [ ] Verify no regressions
- [ ] Commit to GitHub

**Timeline:** ~2 hours dev + 24h testing = 26 hours

### Phase 2: Monitoring (2026-03-27 to 2026-03-28)
- [ ] Implement `signal_sync_monitor.py`
- [ ] Add background thread to daemon startup
- [ ] Create alert mechanism (log file / Slack)
- [ ] Test divergence detection
- [ ] Commit to GitHub

**Timeline:** ~1 hour dev + 12h testing = 13 hours

### Phase 3: Automation (2026-03-29 to 2026-03-31)
- [ ] Add auto-recovery trigger to monitor
- [ ] Create `auto_recover_dual_write.py` script
- [ ] Implement hourly checkpoint cron job
- [ ] Test full recovery flow
- [ ] Commit to GitHub

**Timeline:** ~2 hours dev + 12h testing = 14 hours

### Phase 4: Hardening (April)
- [ ] Add detailed logging to all write operations
- [ ] Create dashboard for dual-write health
- [ ] Set up automated tests for dual-write failures
- [ ] Document procedures for manual recovery
- [ ] Create runbook for operations team

---

## Success Criteria

After implementation, verify:

1. **Write Verification Works**
   - Every signal write verified within 10 seconds
   - Failures cause daemon to alert/halt (fail-safe)
   - No silent failures

2. **Real-Time Monitoring Active**
   - Divergence detected within 5 minutes
   - Alerts generated and logged
   - Background thread runs continuously

3. **Automatic Recovery Functional**
   - Gap > 50 signals triggers auto-recovery
   - Recovery completes in < 1 minute
   - Files re-align automatically

4. **Hourly Checkpoint Running**
   - Cron job executes every hour
   - Results logged with timestamp
   - Alerts sent if gap detected

5. **No Future Divergence**
   - MASTER and AUDIT stay in sync
   - Gap never exceeds 5 signals
   - Zero data loss going forward

---

## Testing Plan

### Test #1: Normal Operation
```bash
# Let daemon run for 24 hours
# Monitor should show: ✅ Files in sync
# Checkpoint should show: Gap = 0-2 signals
```

### Test #2: Simulate Write Failure
```python
# Temporarily make AUDIT writes fail
# Verify: Daemon alerts and halts
# Verify: Monitor detects divergence
# Verify: Auto-recovery triggers
```

### Test #3: High Throughput
```bash
# Fire 1000 signals in short burst
# Verify: All 1000 written to both files
# Verify: No partial writes
# Verify: Verification succeeds for all
```

### Test #4: Recovery Idempotency
```bash
# Intentionally create 200-signal gap
# Run auto-recovery 3 times
# Verify: Same result each time (idempotent)
# Verify: No duplicate signals created
```

---

## Monitoring Dashboard (Future)

Create a simple dashboard showing:
```
DUAL-WRITE HEALTH STATUS
═════════════════════════
Last 24 Hours: ✅ HEALTHY

Files:
  SIGNALS_MASTER.jsonl:         4,672 signals
  SIGNALS_INDEPENDENT_AUDIT.txt: 4,672 signals
  Alignment:                     100% (gap: 0)

Writes (last hour):
  Master successful: 142 ✅
  Audit successful:  142 ✅
  Verification pass: 142 ✅
  Verification fail: 0 ✅

Alerts (last 7 days):
  Divergence detected: 0 (no issues)
  Auto-recovery runs:  0 (no issues)
  Manual recovery:     0 (no issues)

Last checkpoint: 2026-03-23 18:00 GMT+7
Next checkpoint: 2026-03-23 19:00 GMT+7
```

---

## Rollback Plan

If implementation causes issues:

1. **Disable verification:** Comment out verify_dual_write() calls
2. **Stop monitor:** Kill monitor thread
3. **Disable cron:** Comment out checkpoint job
4. **Revert code:** `git revert <commit-hash>`
5. **Manual recovery:** Run backfill script again

---

## Summary

| Component | Status | Timeline |
|-----------|--------|----------|
| Recovery (completed) | ✅ DONE | 2026-03-23 18:55 GMT+7 |
| Dual-write verification | 🔄 TODO | Phase 1 (2026-03-24-26) |
| Real-time monitoring | 🔄 TODO | Phase 2 (2026-03-27-28) |
| Auto-recovery | 🔄 TODO | Phase 3 (2026-03-29-31) |
| Full hardening | 🔄 TODO | Phase 4 (April) |

**Recovery is complete. Prevention implementation begins tomorrow.**

Next step: Code review of `main.py` dual-write logic for Phase 1 implementation.
