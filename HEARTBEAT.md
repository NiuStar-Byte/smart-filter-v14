# HEARTBEAT.md - Daily Operations & Startup Verification

## 🚀 STARTUP VERIFICATION (Automatic on Mac startup)
**File:** `startup_verification.py`
**LaunchAgent:** `com.trading.startup-verification.plist`
**Runs at:** Mac startup + Every 60 minutes
**Checks:**
  ✅ All 10 critical tracker files exist (100% guaranteed in GitHub)
  ✅ All 6 critical services running:
     1. main.py (Signal Generation)
     2. pec_executor_persistent.py (Position Closure)
     3. asterdex_realtime_fetcher.py (Data Fetcher)
     4. asterdex_entry_poster.py (Entry Posting)
     5. xaut_daemon.py (XAUT Tributary)
     6. caffeinate (Mac Keep-Awake)
  ✅ Quick health check on validate_tiers_all.py
  ✅ Quick health check on filter_effectiveness_analyzer_detailed.py
  
**If anything fails:** Attempts auto-restart, logs to `.startup_verification.log` and `.startup_verification_error.log`
**GUARANTEED:** After Mac restart, all 6 services + all 10 tracker files will be running within 60 seconds!

---

## 📊 SIGNAL GROWTH MONITORING (2026-06-18 BASELINE)

**Baseline Locked: 544 unique signals as of 2026-06-18 12:16 GMT+7**

**Target:** Signal count must NEVER DECREASE. Only grow.

**Monitor:**
- Daily total signal count (must be >= 544 by end of 2026-06-18)
- No data loss or signal drops
- COMPLETE_SIGNALS.jsonl is the ONLY source of truth

**Alert Triggers:**
- Signal count drops below 544
- Any attempt to use other signal files (SIGNALS_MASTER, SENT_SIGNALS, etc.)
- Code changes without explicit approval

---

## ✅ **CRITICAL SYSTEMS VERIFIED (2026-06-18 12:22-12:41 GMT+7)**

**PEC Executor Persistent:**
- ✅ Properly targeting OPEN signals in COMPLETE_SIGNALS.jsonl
- ✅ Generating full closures (status, exit_price, P&L, closed_at)
- ✅ Atomic write-back to COMPLETE_SIGNALS.jsonl confirmed
- ✅ Field completeness: symbol_group, confidence_level, all closure fields present

**MTF Alignment Tracker v2:**
- ✅ Fixed open-ended window filtering (end=None handling)
- ✅ Removed old comparison logic
- ✅ Fresh start baseline: 643 fired, 282 closed, 48.09% WR
- ✅ Tracker running successfully

**Status: ALL SYSTEMS OPERATIONAL**

---

# HEARTBEAT CHECKLIST - 2026-06-18 ONWARDS

## 📌 CRITICAL - Check on every heartbeat
- [ ] **Signal growth:** `grep -c '"status": "OPEN"' COMPLETE_SIGNALS.jsonl` should ALWAYS be >= last check
  - **Rule:** OPEN count can only STAY SAME or GROW, NEVER DECREASE
  - If it decreases: STOP main.py immediately, investigate data corruption
  
- [ ] **Executor status:** `pgrep -c pec_executor_persistent` should be running (3+ processes normal)
  - If 0: Restart with: `python3 pec_executor_persistent.py &`

- [ ] **Single source of truth:** COMPLETE_SIGNALS.jsonl is the ONLY file being read/written
  - Never touch: SIGNALS_MASTER.jsonl, SENT_SIGNALS.jsonl, SIGNALS_CANONICAL.jsonl, ALL_SIGNALS.jsonl (banned forever)

## 🔍 Quick Status Command
```bash
# Copy-paste this for live monitoring (every 5 seconds):
watch -n 5 /Users/geniustarigan/.openclaw/workspace/monitor_pec.sh
```

## 📊 What to Look For in Monitor Output
- 🔵 **OPEN signals:** Must be stable or growing (never decrease)
- ✅ **TP_HIT:** Profitable closures - target is 50%+ of closed trades
- ❌ **SL_HIT:** Stop loss closures - monitor ratio (TP:SL ideally > 1:1)
- ⏱️  **PEC-TIMEOUT:** Should stay low (closed signals expired naturally, not executed)
- ✅ **Processes:** Should show "3 process(es)" running normally

## 🚨 Alert Conditions
1. **OPEN count decreased:** Data corruption, STOP main.py, investigate
2. **No running processes:** Executor crashed, check logs, restart
3. **Reading from banned files:** Code violation, revert changes
4. **Tier assignment not matching LOCKED_COMBOS_TODAY:** Tier assignment broken
