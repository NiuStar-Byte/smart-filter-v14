# Hourly Cron Setup - Zero Daemon Architecture

## Overview

Replace unreliable daemon with hourly cron job that:
- Runs at :00 every UTC hour
- Analyzes signals from previous hour
- Locks hour (immutable)
- Appends to SIGNALS_LEDGER_IMMUTABLE.jsonl
- Zero daemon, zero crashes, zero missed hours

---

## How It Works

### Hourly Cron Job
```
:00 UTC every hour
  ↓
Run: hourly_signal_analysis.py
  ↓
Load signals_fired.jsonl for that hour
  ↓
Deduplicate (check against existing ledger)
  ↓
Append new signals to SIGNALS_LEDGER_IMMUTABLE.jsonl
  ↓
Lock that hour (cannot change)
```

### Result in PEC Reporter
```
Date breakdown (auto-summed from hours):
  2026-02-27: 80 signals [LOCKED - complete date]
  2026-03-05: 227 signals [OPEN - still accumulating]

Hour breakdown (for today):
  2026-03-05 00:00-00:59 UTC: 77 signals [LOCKED]
  2026-03-05 01:00-01:59 UTC: 19 signals [LOCKED]
  2026-03-05 21:00-21:59 UTC: [OPEN - still accumulating]
```

---

## Setup Options

### Option 1: System Crontab (macOS/Linux)

```bash
# Edit crontab
crontab -e

# Add this line (runs at :00 every UTC hour):
0 * * * * cd /Users/geniustarigan/.openclaw/workspace && /usr/bin/python3 hourly_signal_analysis.py >> hourly_analysis.log 2>&1

# Verify
crontab -l
```

### Option 2: OpenClaw Cron (Built-in)

Already created job: `Hourly Signal Analysis - Lock Hours`
- ID: 54c47fa0-0882-43bb-99f9-ad0d1733585a
- Schedule: 0 * * * * UTC
- Payload: systemEvent

### Option 3: Shell Script Wrapper

Use provided `run_hourly_analysis.sh`:
```bash
chmod +x run_hourly_analysis.sh
crontab -e
# Add: 0 * * * * /Users/geniustarigan/.openclaw/workspace/run_hourly_analysis.sh
```

---

## Files

### Core
- `hourly_signal_analysis.py` - Analyzes signals, locks hours
- `SIGNALS_LEDGER_IMMUTABLE.jsonl` - Immutable ledger (appended hourly)

### Reporter
- `pec_immutable_ledger_reporter.py` - Reads from ledger (unchanged template)
- `PEC_IMMUTABLE_LEDGER_REPORT.txt` - Output (generated anytime)

### Logging
- `hourly_analysis.log` - Cron execution log (created if using shell script)

---

## What Changed

**OLD (Daemon):**
```
main.py daemon (crashes)
  ↓
SENT_SIGNALS.jsonl (moving, conflicting)
  ↓
PEC Reporter (confused, biased)
```

**NEW (Hourly Cron):**
```
hourly_cron (reliable, scheduled)
  ↓
SIGNALS_LEDGER_IMMUTABLE.jsonl (locked, clean)
  ↓
PEC Reporter (same template, immutable input)
```

---

## What Didn't Change

✓ PEC Reporter template (8 sections)  
✓ PEC Executor logic  
✓ PEC Watchdog logic  
✓ Backtest tools  
✓ All dependent processes  

**Only change:** Input source (from daemon files → immutable ledger)

---

## Verification

### Test hourly script
```bash
python3 hourly_signal_analysis.py
# Output: Hour lock status, signals appended
```

### Check ledger
```bash
wc -l SIGNALS_LEDGER_IMMUTABLE.jsonl
# Should grow by ~30 signals per hour
```

### Generate report
```bash
python3 pec_immutable_ledger_reporter.py
# Output: Same template, clean input data
```

---

## Benefits

✅ No daemon (no crashes, no memory leaks)  
✅ Immutable ledger (clean data, no gaps)  
✅ Hourly locks (integrity per hour)  
✅ Same reporter template (no logic changes)  
✅ Reliable scheduling (cron vs daemon)  
✅ Complete signal history (nothing lost)  

---

## Next Steps

1. Set up system crontab (Option 1) - recommended, most reliable
2. Verify job runs at :00 UTC (check hourly_analysis.log)
3. Monitor ledger growth (~30 signals/hour)
4. Run PEC reporter anytime (reads clean data)

---

## Maintenance

### Check cron is running
```bash
tail -f hourly_analysis.log
# Should see new entries at each :00 UTC hour
```

### View scheduled jobs
```bash
crontab -l
# Should show: 0 * * * * cd /Users/geniustarigan/...
```

### Test next run
```bash
python3 hourly_signal_analysis.py
# Manually locks the current/previous hour
```

---

**Status: ✅ READY TO DEPLOY**

Hourly cron is set up. Just activate system crontab and go live.
