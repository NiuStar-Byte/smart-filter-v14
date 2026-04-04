# Tier Assignment Monitoring Integration Checklist

## What Just Happened

✅ **Created tier assignment validator** that monitors the quality loop
✅ **Integrated into pec_master_controller.py** to check health every 60 seconds
✅ **Checks if tier assignments are being written to signals**

## Current Status

🚨 **Validator detects: 0% tier assignments in SIGNALS_MASTER.jsonl**

This is **not a validator bug** — it's **accurately detecting a missing workflow step**.

## Quality Loop Diagram

```
1. main.py fires signal
                 ↓
2. pec_executor_persistent.py processes it
                 ↓
3. tier_lookup.py assigns tier (using SIGNAL_TIERS.json patterns)
                 ↓
4. ❌ MISSING: Write tier back to signal
                 ↓
5. Update signal in SIGNALS_MASTER.jsonl
                 ↓
6. Next run: validator confirms tier ✅
                 ↓
7. Loop repeats with proven-quality signals
```

## What Needs to Be Done

### In pec_executor.py

After tier assignment (around line ~220-270 where you assign tier):

```python
# Current code (somewhere in pec_executor_persistent.py):
tier = tier_lookup.get_tier(signal)  # or similar

# Add these lines:
signal['tier'] = tier  # ← Write tier to signal dict
# Then when writing signal back to SIGNALS_MASTER.jsonl:
json.dump(signal, f)  # Make sure this happens for ALL signals
```

**Key requirement:** Every signal written to SIGNALS_MASTER.jsonl must have the `tier` field populated (even if tier=None for non-tiered signals).

## How the Monitoring Works

Once tier assignments are wired:

### ✅ **Healthy State**
```
[15:25:02] [TIER_OK] ✅ TIER ASSIGNMENT HEALTH: 156/160 (97.5%) | T1:41 | T2:62 | T3:53
```
Meaning: 156 of last 160 closed signals have tier assignments
- T1: 41 signals assigned to Tier-1
- T2: 62 signals assigned to Tier-2
- T3: 53 signals assigned to Tier-3

### ⚠️ **Warning State**
```
[15:40:02] [TIER_WARN] ⚠️ TIER ASSIGNMENT WARNING: 73.0% success | 27 pattern mismatches
```
Meaning: Only 73% of signals got tier assignments (below 70% threshold)
- 27 signals matched tier patterns but tier field wasn't written

### 🚨 **Critical State**
```
[15:40:02] [TIER_CRITICAL] 🚨 TIER ASSIGNMENT FAILURE: Only 0.0% assigned (threshold: 70%)
```
Meaning: Assignment mechanism is broken
- System ALERT: manual review needed
- Could trigger automatic executor restart for healing

## Manual Validation

To check if tier assignments are working:

```bash
# Run validator manually
python3 ~/.openclaw/workspace/tier_assignment_validator.py

# Or check recent signals for tier field
python3 << 'EOF'
import json
with open('SIGNALS_MASTER.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i < 10:
            sig = json.loads(line)
            print(f"Signal {i}: tier={sig.get('tier')}")
EOF
```

## Integration Timeline

1. **Now** ← You are here (Validator is ready)
2. **Next:** Wire tier writing into pec_executor.py
3. **Then:** Restart pec_master_controller
4. **Then:** Validator will automatically start reporting success rates
5. **Then:** Quality loop runs autonomously, self-healing on failures

## Files Created

- `tier_assignment_validator.py` - Monitors tier assignment health
- `pec_master_controller.py` - Updated to call validator every 60s
- `TIER_ASSIGNMENT_INTEGRATION_CHECKLIST.md` - This document

## Success Metrics

**Target:** ≥90% of closed signals have tier assignments within 60 seconds of closing

When achieved:
- Quality loop is operating correctly
- Signals with better WR automatically get tiered
- System self-heals on failures
- Ready to scale toward 51% overall WR

## Commands

```bash
# Monitor tier health in real-time
tail -f pec_controller.log | grep TIER

# Manual validation run
python3 tier_assignment_validator.py

# Check if controller is running with validator
ps aux | grep pec_master_controller
grep "Tier Assignment Validator" pec_controller.log
```

## Notes

- Validator checks **last 100 closed signals** every 60 seconds
- Threshold: **≥70% assignment success rate** to stay green
- Tier patterns loaded from **SIGNAL_TIERS.json (latest entry)**
- If critical failure, logs detailed diagnostics for manual recovery
