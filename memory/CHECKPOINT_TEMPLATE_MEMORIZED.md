# 🧠 CHECKPOINT TEMPLATE (MEMORIZED)

**Use this structure EVERY TIME user says "checkpoint"**

---

## 📋 CHECKPOINT STRUCTURE (Always Follow This)

### Header Section
```markdown
# 📋 CHECKPOINT - YYYY-MM-DD HH:MM GMT+7

## 🎯 SESSION SUMMARY
**Date:** YYYY-MM-DD
**Session Type:** [Type of work done]
**Status:** ✅ [Overall status]
**GitHub:** ✅ Synced (Commit XXXXX)
**Daemon:** ✅ Running (PID XXXXX, fresh restart)
```

### Main Sections (In Order)

1. **🔄 MAJOR ACCOMPLISHMENTS**
   - List what was done with status checkmarks
   - Include before/after metrics
   - Show impact

2. **📊 CURRENT OPERATIONAL STATUS**
   
   **Daemon**
   - PID: [current]
   - Code Version: Commit [X]
   - DEBUG_FILTERS: [true/false]
   - Symbols: [count] (validated [X] perpetuals)
   - Cycle Time: [X] average
   
   **Signal Generation**
   - Daily Snapshot: [X] signals
   - Closed: [Y] signals
   - Open: [Z] signals
   - Hour Rate: ~[X] signals/hour
   - Status: ✅ [Status]
   
   **Signal File Sync (MASTER vs AUDIT)** ← ALWAYS INCLUDE THIS
   - SIGNALS_MASTER.jsonl: [X] lines (daemon primary source)
   - SIGNALS_INDEPENDENT_AUDIT.txt: [Y] lines (audit trail)
   - Divergence: [Z] signals (difference)
   - Lag Type: ✅ [Normal/Monitor/Critical]
   - Root Cause: [Explanation]
   - Status: ✅ [Assessment]
   - Trend: [Converging/Stable/Diverging]
   - Action: [What to do, if needed]
   
   **Interpretation:**
   - MASTER = daemon writes immediately after Telegram send
   - AUDIT = backup verification log (may lag slightly)
   - Lag 1-8 signals = normal (async update, accumulation)
   - Lag >15 signals = warning (possible MASTER stuck)
   - MASTER > AUDIT = critical (data loss risk)
   
   **GitHub Sync**
   - Workspace: Commit [X] ✅ Synced
   - Submodule: Commit [X] ✅ Synced
   - Divergence: Zero ✅

3. **🎯 NEXT STEPS (Pending)**
   - List what's coming
   - Include timeline if applicable
   - Add monitoring items

4. **📁 KEY FILES & LOCATIONS**
   - Table with File | Purpose | Path

5. **💾 CRITICAL GITHUB COMMITS**
   - Table with Commit | Change | Type

6. **🔒 VALIDATION SUMMARY**
   - Code Quality
   - Data Integrity
   - System Health

7. **📋 SESSION STATISTICS**
   - Time, commits, symbols, filters, features, etc.

8. **🎓 KEY LESSONS LEARNED**
   - What was discovered/improved

9. **✅ FINAL STATUS**
   - What's running
   - What's next
   - Overall health

---

## CRITICAL SECTION: Signal File Sync (Never Omit)

**ALWAYS include this in EVERY checkpoint:**

```markdown
### Signal File Sync (MASTER vs AUDIT)
**Critical Integrity Check**

- **SIGNALS_MASTER.jsonl:** [X] lines (daemon primary source)
- **SIGNALS_INDEPENDENT_AUDIT.txt:** [Y] lines (backup audit trail)
- **Divergence:** [Z] signals ([Y] - [X] = [Z])
- **Direction:** [AUDIT ahead / MASTER ahead / Equal]
- **Lag Type:** [✅ Normal / ⚠️ Monitor / 🔴 Critical]
- **Root Cause:** [Explanation of why lag/divergence exists]
- **Status:** [Assessment]
- **Trend:** [Converging/Stable/Diverging]
- **Verification:** [Last UUID match / UUID mismatch / etc.]
- **Action:** [What to monitor or do, if anything]

**Interpretation:**
- MASTER = daemon writes immediately after Telegram send
- AUDIT = backup verification log (may lag slightly)
- Lag 1-8 signals = normal (async update, accumulation)
- Lag >15 signals = warning (possible MASTER stuck)
- MASTER > AUDIT = critical (data loss risk)
```

---

## Quick Check Command (When Generating Checkpoint)

```bash
cd /Users/geniustarigan/.openclaw/workspace

# Get MASTER count
MASTER=$(wc -l < SIGNALS_MASTER.jsonl)

# Get AUDIT count
AUDIT=$(wc -l < SIGNALS_INDEPENDENT_AUDIT.txt)

# Calculate divergence
DIVERGENCE=$((AUDIT - MASTER))

# Check daemon PID
ps aux | grep "python3.*main.py" | grep -v grep | awk '{print $2}'

# Check latest cycle time
tail -20 main_daemon.log | grep "Cycle took" | tail -1

echo "MASTER: $MASTER"
echo "AUDIT: $AUDIT"
echo "Divergence: $DIVERGENCE"
```

---

## Footer Section (Always End With)

```markdown
---

**Prepared by:** Nox (Personal Assistant)
**Date:** YYYY-MM-DD HH:MM GMT+7
**Status:** ✅ CHECKPOINT COMPLETE
**Next Checkpoint:** [When]
```

---

## When User Says "Checkpoint"

1. ✅ Gather current metrics:
   - File sync (MASTER vs AUDIT)
   - Daemon status
   - Signal counts
   - GitHub sync status
   - Recent commits

2. ✅ Follow template structure above (in order)

3. ✅ Include Signal File Sync section (CRITICAL - never skip)

4. ✅ Save to `/Users/geniustarigan/.openclaw/workspace/memory/YYYY-MM-DD-checkpoint.md`

5. ✅ Also save summary to Desktop (if requested)

6. ✅ Always include interpretation guide for divergence

---

**MEMORIZED:** 2026-03-09 23:09 GMT+7  
**Will use this template:** Every checkpoint from now on  
**Key requirement:** Always include SIGNALS_MASTER.jsonl vs SIGNALS_INDEPENDENT_AUDIT.txt comparison
