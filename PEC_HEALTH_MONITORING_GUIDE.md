# 🏥 PEC SYSTEM HEALTH MONITORING GUIDE

**Status:** ✅ **AUTOMATED ERROR DETECTION & LOGGING**  
**Deployment:** 2026-03-05 12:41 GMT+7  
**Purpose:** Catch signal accumulation failures before they happen

---

## 📋 QUICK CHECK - Two Modes

### **Mode 1: Single Check (Quick)**
```bash
python3 pec_system_health_monitor.py
```

**Output:**
```
🟢 SYSTEM HEALTHY - All components operational

OR

🔴 SYSTEM DEGRADED - See errors above, check pec_system_health.log for details
```

### **Mode 2: Watch Mode (Continuous - 30 seconds auto-refresh)**
```bash
# Default: refresh every 30 seconds
python3 pec_system_health_monitor.py --watch

# Or custom interval (e.g., 10 seconds):
python3 pec_system_health_monitor.py --watch 10
```

**Output updates automatically every 30 seconds:**
```
🔄 PEC SYSTEM HEALTH MONITOR (auto-refresh every 30s) - 2026-03-05 12:50:15 GMT+7

  ✅ Daemon (main.py)              → Running & firing signals
  ✅ Executor (pec_executor.py)    → Running & processing
  ✅ Watchdog (pec_watchdog.py)    → Running & monitoring
  ✅ File Access (SENT_SIGNALS)   → Readable & writable
  ✅ Signal Pipeline              → Accumulating normally

🟢 SYSTEM HEALTHY - All components operational

⏳ Next update in 30 seconds... (Ctrl+C to stop)
```

Stop with: **`Ctrl+C`**

---

## 🔴 ERROR LOG - Where Problems Are Reported

**Live error log:**
```bash
tail -f pec_system_health.log
```

**Example error messages:**
```
[2026-03-05 12:41:49 GMT+7] 🔴 EXECUTOR   | NOT_RUNNING    | pec_executor.py process not found
[2026-03-05 12:42:15 GMT+7] 🔴 EXECUTOR   | RUNTIME_ERROR  | SENT_SIGNALS.jsonl corrupted: Expecting value
[2026-03-05 12:43:00 GMT+7] 🟡 FLOW       | BACKLOG        | 47 signals still OPEN - executor may be behind
[2026-03-05 12:44:30 GMT+7] 🟡 DAEMON     | STALE_DATA     | SENT_SIGNALS.jsonl not updated for 15+ minutes
```

---

## 🟢 WHAT'S BEING MONITORED

### **Component 1: DAEMON (main.py)**
- ✅ Process running
- ✅ Signals being fired
- ✅ File updated regularly (< 10 min old)

**If fails:**
- 🔴 `DAEMON | NOT_RUNNING` → Daemon crashed, need restart
- 🟡 `DAEMON | STALE_DATA` → Daemon not firing signals (stuck cycle?)

---

### **Component 2: EXECUTOR (pec_executor.py)**
- ✅ Process running
- ✅ Updating signal statuses
- ✅ Writing closed_at timestamps

**If fails:**
- 🔴 `EXECUTOR | NOT_RUNNING` → Executor crashed, watchdog should restart
- 🔴 `EXECUTOR | RUNTIME_ERROR` → Code error in executor (see message)
- 🔴 `EXECUTOR | FILE_MISSING` → SENT_SIGNALS.jsonl not found

---

### **Component 3: WATCHDOG (pec_watchdog.py)**
- ✅ Process running
- ✅ Monitoring executor
- ✅ Auto-restarting on crash

**If fails:**
- 🔴 `WATCHDOG | NOT_RUNNING` → Watchdog crashed (manual restart needed)
- 🔴 `WATCHDOG | START_FAILED` → Watchdog can't start executor (path issue?)
- 🔴 `WATCHDOG | SCRIPT_MISSING` → pec_executor.py not found

---

### **Component 4: FILE ACCESS**
- ✅ SENT_SIGNALS.jsonl exists
- ✅ Readable and writable

**If fails:**
- 🔴 `FILE | MISSING` → File deleted or moved
- 🔴 `FILE | NOT_WRITABLE` → Permission issue (chmod?)

---

### **Component 5: SIGNAL FLOW**
- ✅ Signals accumulating (recent activity)
- ✅ Signals being closed (executor processing)
- ✅ No backlog (< 50 open signals)

**If fails:**
- 🔴 `FLOW | NO_DATA` → No signals in file
- 🟡 `FLOW | NO_RECENT` → No signals in last hour
- 🟡 `FLOW | BACKLOG` → 47+ signals still open (executor slow)

---

## ⚡ TROUBLESHOOTING BY ERROR

| Error | Meaning | Fix |
|-------|---------|-----|
| `DAEMON NOT_RUNNING` | main.py crashed | Restart daemon: `cd smart-filter-v14-main && nohup python3 main.py > ../main_daemon.log 2>&1 &` |
| `EXECUTOR NOT_RUNNING` | pec_executor crashed | Restart watchdog: `pkill -f pec_watchdog && sleep 2 && nohup python3 pec_watchdog.py > pec_watchdog.log 2>&1 &` |
| `EXECUTOR RUNTIME_ERROR` | Code error in executor | Check error message, review pec_executor.py code |
| `WATCHDOG NOT_RUNNING` | Watchdog crashed | Restart: `nohup python3 pec_watchdog.py > pec_watchdog.log 2>&1 &` |
| `FILE NOT_WRITABLE` | Permission issue | `chmod 644 SENT_SIGNALS.jsonl` |
| `FLOW BACKLOG` | Executor slow | Check KuCoin API, network latency, increase CYCLE_SLEEP |
| `DAEMON STALE_DATA` | No new signals | Check if daemon hung, review main_daemon.log |

---

## 🔄 AUTOMATED MONITORING (Coming Soon)

Health monitor can run automatically via cron:

```bash
# Run health check every 10 minutes
*/10 * * * * cd /Users/geniustarigan/.openclaw/workspace && python3 pec_system_health_monitor.py >> pec_system_health.log 2>&1
```

---

## 📊 WHAT YOU DON'T HAVE TO DO ANYMORE

✅ No more manual diagnosis scripts  
✅ No more "is it running?" checks  
✅ No more confusion about what broke  
✅ Errors logged automatically  
✅ One command to see everything  

**Instead of:**
```
Is daemon running? Check ps aux | grep main
Is executor running? Check ps aux | grep pec_executor
Is file being updated? Check ls -la SENT_SIGNALS.jsonl
Is executor processing? Check tail SENT_SIGNALS.jsonl
What went wrong? Dig through logs...
```

**Now just:**
```
python3 pec_system_health_monitor.py
```

**Done. All answers in 2 seconds.** 🎯

---

## 📝 FILES CREATED/MODIFIED

- **NEW:** `pec_system_health_monitor.py` - Health check tool
- **NEW:** `pec_system_health.log` - Error log (auto-created on first error)
- **UPDATED:** `pec_watchdog.py` - Now logs errors to health monitor
- **UPDATED:** `pec_executor.py` - Now logs runtime errors to health monitor
- **UPDATED:** `main.py` (daemon) - [Optional] can be enhanced with error logging in future

---

## 🎯 INTEGRATION WITH TRACKERS LOCK

This monitoring system is:
- ✅ **Non-intrusive** - Only adds logging, doesn't change functionality
- ✅ **Read-only diagnostic** - Doesn't modify signal data
- ✅ **Backward compatible** - Works with all existing trackers
- ✅ **Approved** - Enhancement to detect issues during accumulation phase

**No change to:**
- ✅ COMPARE_AB_TEST_LOCKED.py
- ✅ PHASE3_TRACKER.py
- ✅ track_rr_comparison.py
- ✅ pec_enhanced_reporter.py
- ✅ SENT_SIGNALS.jsonl

---

**Last Updated:** 2026-03-05 12:41 GMT+7  
**Commit:** 0aed69b
