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

## 🚨 **CRITICAL: TIER LOOKUP MODULE CACHE BUG - FIXED (2026-06-18 13:32 GMT+7)**

**Issue:** Tier-3 signals firing when only Tier-1 locked today  
**Root Cause:** Python module import cache (old LOCKED_COMBOS_TODAY still in memory)  
**Status:** ✅ FIXED with importlib.reload() in tier_lookup.py  
**Action Required:** RESTART main.py to clear old cache  
```bash
pkill -f "main.py"
python3 /Users/geniustarigan/.openclaw/workspace/main.py &
```

---

## 📊 SIGNAL GROWTH MONITORING & TODAY'S LOCKED COMBOS (2026-06-18)

**Baseline Locked: 544 unique signals as of 2026-06-18 12:16 GMT+7**

**Target:** Signal count must NEVER DECREASE. Only grow.

**Monitor:**
- Daily total signal count (must be >= 544 by end of 2026-06-18)
- No data loss or signal drops
- COMPLETE_SIGNALS.jsonl is the ONLY source of truth

**Today's Locked Combos (2026-06-18 13:22 GMT+7):**
- Generated from: PEC_POST_DEPLOYMENT_TRACKER_v2_2026-06-17_23-44-14.txt (newest yesterday report)
- Expires: 2026-06-19 00:00:00 GMT+7
- **TOTAL: 1 COMBO**
  - **Tier-1:** 1 combo (6D SHORT)
    - `30min_SHORT_TREND CONTINUATION_BEAR_LOW_ALTS_HIGH`
  - **Tier-2:** 0 combos
  - **Tier-3:** 0 combos

**Alert Triggers:**
- Signal count drops below 544
- Any attempt to use other signal files (SIGNALS_MASTER, SENT_SIGNALS, etc.)
- Code changes without explicit approval

---

## ✅ **CRITICAL SYSTEMS VERIFIED (2026-06-18 13:07 GMT+7) - FULLY OPERATIONAL**

**PEC Executor Persistent:**
- ✅ **3 processes running** (PIDs: 9651, 9653, 60554)
- ✅ **Properly targeting OPEN signals** in COMPLETE_SIGNALS.jsonl
- ✅ **Generating full closures** (status, exit_price, P&L, closed_at, pnl_usd)
- ✅ **Atomic write-back confirmed** - All closures written to COMPLETE_SIGNALS.jsonl successfully
- ✅ **Field completeness 100%** - symbol_group, confidence_level, tier all present
- ✅ **No timeout issues** - 0 PEC-TIMEOUT, 0 STALE_TIMEOUT (clean execution)
- ✅ **Win rate 47.2%** (128 TP_HIT vs 143 SL_HIT - acceptable)
- ✅ **Closure tracking verified** - Recent closures show: symbol, timeframe, closed_at, pnl_usd
- **Status: FULLY OPERATIONAL & VERIFIED**

**MTF Alignment Tracker v2:**
- ✅ Fixed open-ended window filtering (end=None handling)
- ✅ Removed old comparison logic
- ✅ Fresh start baseline: 643 fired, 282 closed, 48.09% WR
- ✅ Tracker running successfully

**Monitor.pec.sh (Live Monitoring Script):**
- ✅ Created and verified working
- ✅ Tracks OPEN, TP_HIT, SL_HIT, timeouts, process status

**Hourly Backup Scripts (2026-06-18 13:16 GMT+7):**
- ✅ `hourly_backup_complete_signals.sh` - Creates dual-format backups:
  - `.jsonl` backups: Last 24 hourly snapshots (full history for long-term recovery)
  - `.txt` backups: Last 5 hourly snapshots only (sliding window for quick access)
  - Format: `COMPLETE_SIGNALS_hourly_backup_YYYY-MM-DD_HHMM.txt`
  - Example pattern: `COMPLETE_SIGNALS_hourly_backup_2026-06-18_1400.txt` → `1500` → `1600` → `1700` → `1800` → (next creates 1900, deletes 1400)
- ✅ All 743 signals backed up with complete record attributes (100% field completeness)
- ✅ Rotation mechanism tested & verified: Creates at top of each hour, auto-deletes oldest when >5 .txt files exist
- ✅ Both backup types preserve: symbol, timeframe, tier, status, symbol_group, confidence_level, closed_at, pnl_usd, entry_price, tp_target, sl_target, all filters

**Status: ALL SYSTEMS OPERATIONAL & VERIFIED STABLE**

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
