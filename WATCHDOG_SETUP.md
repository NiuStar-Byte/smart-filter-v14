# PEC Watchdog Setup

**Status:** ✅ **ACTIVE & PERSISTENT**

---

## 📋 What Is This?

The **PEC Watchdog** is an auto-restart daemon that:
- ✅ Monitors the PEC Executor process every 30 seconds
- ✅ Automatically restarts it if it crashes
- ✅ Logs all activity for audit trail
- ✅ Persists across system reboots (via macOS Launch Agent)
- ✅ Gives up gracefully after 3 consecutive startup failures

---

## 🚀 Current Setup

### Running Processes
```
PEC Watchdog   (PID 1621) → Via launchctl (system-managed)
PEC Executor   (PID 1736) → Monitored & auto-restarted
Main.py        (PID 27031) → Signal generation (separate)
```

### Files
- **Script:** `pec_watchdog.py` (130 lines)
- **Launch Agent:** `~/Library/LaunchAgents/com.smartfilter.pec-watchdog.plist`
- **Logs:** `pec_watchdog.log` (main), `pec_watchdog_error.log` (errors)

### Configuration
- **Check Interval:** 30 seconds
- **Max Startup Retries:** 3 (gives up after 3 failures in a row)
- **Restart Behavior:** KeepAlive=true (system respects this)

---

## 📊 How It Works

### Detection Cycle
1. Watchdog checks if `pec_executor.py` is running
2. If dead → **Restart immediately**, log event
3. If alive → Verify no crashes occurred
4. Sleep 30 seconds, repeat

### Log Format
```
[2026-03-02 15:23:13] ⚠️  PEC Executor died (crash #1). Restarting...
[2026-03-02 15:23:15] ✅ PEC Executor started (PID will auto-continue)
```

---

## ✅ Testing & Verification

### Kill PEC to trigger restart:
```bash
kill $(pgrep -f pec_executor.py)
# Watchdog detects within 30 seconds and restarts automatically
```

### Check watchdog health:
```bash
launchctl list | grep pec-watchdog
# Should show: [PID]  0  com.smartfilter.pec-watchdog
```

### View recent logs:
```bash
tail -20 /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/pec_watchdog.log
```

---

## 🔄 Persistence Across Reboots

The **macOS Launch Agent** ensures watchdog auto-starts:
1. System boots
2. launchctl loads `com.smartfilter.pec-watchdog`
3. Watchdog starts → Checks PEC status
4. If PEC dead → Starts it automatically
5. Continuous monitoring begins

**To unload (disable) watchdog:**
```bash
launchctl unload ~/Library/LaunchAgents/com.smartfilter.pec-watchdog.plist
```

**To reload (re-enable) watchdog:**
```bash
launchctl load ~/Library/LaunchAgents/com.smartfilter.pec-watchdog.plist
```

---

## 📈 Expected Behavior

### Before Watchdog
- PEC crashes → Data collection stops
- No signals processed
- Stale OPEN signals pile up
- Manual restart required

### After Watchdog
- PEC crashes → Auto-detected & restarted within 30 seconds
- No manual intervention needed
- Data collection continuous
- Auditable crash log available

---

## 🚨 Troubleshooting

### PEC keeps crashing repeatedly?
- Check `pec_executor.log` for errors
- Watchdog will log all attempts: `tail -20 pec_watchdog.log`
- If 3 startup failures in a row → watchdog exits (needs manual restart)

### Watchdog not responding?
```bash
# Check if watchdog is actually running
launchctl list | grep pec-watchdog

# Force reload launch agent
launchctl unload ~/Library/LaunchAgents/com.smartfilter.pec-watchdog.plist
launchctl load ~/Library/LaunchAgents/com.smartfilter.pec-watchdog.plist
```

### Verify both processes alive:
```bash
ps aux | grep -E "(pec_executor|pec_watchdog)" | grep -v grep
```

---

## 📝 Changelog

**2026-03-02 15:22 GMT+7**
- ✅ Created `pec_watchdog.py` (auto-restart logic)
- ✅ Created launch agent plist (persistence)
- ✅ Loaded into launchctl (system-managed)
- ✅ Verified crash detection & restart capability
- ✅ Tested & confirmed working

---

**Bottom Line:** PEC data collection is now **bulletproof**. Crashes are detected & fixed automatically. No stale signals will pile up.
