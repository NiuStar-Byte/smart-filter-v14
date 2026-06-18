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

# Keep this file empty (or with only comments) to skip heartbeat API calls.

# Add tasks below when you want the agent to check something periodically.
