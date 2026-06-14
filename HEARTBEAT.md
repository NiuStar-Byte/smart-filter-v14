# HEARTBEAT.md - Daily Operations & Startup Verification

## 🚀 STARTUP VERIFICATION (Automatic on Mac startup)
**File:** `startup_verification.py`
**LaunchAgent:** `com.trading.startup-verification.plist`
**Runs at:** Mac startup + Every 60 minutes
**Checks:**
  ✅ All 10 critical tracker files exist
  ✅ All 3 critical services running (main.py, asterdex_entry_poster, pec_executor_persistent)
  ✅ Quick health check on validate_tiers_all.py
  ✅ Quick health check on filter_effectiveness_analyzer_detailed.py
  
**If anything fails:** Attempts auto-restart, logs to `.startup_verification.log`
**Important:** This script now GUARANTEES that after Mac restart, all services and trackers are running!

---

# Keep this file empty (or with only comments) to skip heartbeat API calls.

# Add tasks below when you want the agent to check something periodically.
