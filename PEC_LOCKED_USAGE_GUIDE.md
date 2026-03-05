# 🔒 PEC Enhanced Reporter - LOCKED USAGE GUIDE

**Status:** IMMUTABLE | **Locked:** 2026-03-05 09:54 GMT+7 | **Hash:** cd27d30...

---

## ✅ WHAT YOU CAN DO

### Run Reports
```bash
cd /Users/geniustarigan/.openclaw/workspace
python3 pec_enhanced_reporter.py
```

### Save Output
```bash
python3 pec_enhanced_reporter.py > PEC_REPORT_2026-03-05.txt
```

### Verify Integrity (DO THIS BEFORE EACH USE)
```bash
sha256sum pec_enhanced_reporter.py
```
**Expected output:**
```
cd27d3045991153d2c7fce73c1859e602c24b3dcc1f8c5b4599ce6cfc0389d1a  pec_enhanced_reporter.py
```

### Check Git Status
```bash
git log --oneline -1
# Should show: 5ae94a2 🔒 [FINAL LOCK] PEC Template IMMUTABLE
```

---

## ❌ WHAT YOU CANNOT DO

- ❌ Edit `pec_enhanced_reporter.py`
- ❌ Change any formula or calculation
- ❌ Modify section structure
- ❌ Adjust dimension definitions
- ❌ Touch the code in any way

**These are not suggestions. They are rules.**

---

## 🛡️ IF YOU SUSPECT CHANGES

### Step 1: Check Hash
```bash
sha256sum pec_enhanced_reporter.py
```
If hash ≠ `cd27d3045991153d2c7fce73c1859e602c24b3dcc1f8c5b4599ce6cfc0389d1a`:
- 🛑 File has been modified
- ❌ Do NOT use it
- 🔄 Proceed to Step 2

### Step 2: Emergency Revert
```bash
# Option A: Surgical revert (just the file)
git checkout 368b184 -- pec_enhanced_reporter.py

# Option B: Full rollback (reset entire repo)
git reset --hard 368b184
```

### Step 3: Verify Success
```bash
sha256sum pec_enhanced_reporter.py
# Must match: cd27d3045991153d2c7fce73c1859e602c24b3dcc1f8c5b4599ce6cfc0389d1a
```

### Step 4: Test Run
```bash
python3 pec_enhanced_reporter.py 2>&1 | head -50
```

---

## 📊 DAILY CHECKLIST

Before running reports each day:

- [ ] Verify hash: `sha256sum pec_enhanced_reporter.py`
- [ ] Check git: `git log --oneline -1`
- [ ] Confirm commit: Should show `5ae94a2` or earlier
- [ ] Test run: `python3 pec_enhanced_reporter.py | grep "SUMMARY" -A 5`

---

## 🔐 PROTECTION MECHANISMS IN PLACE

1. **SHA256 Hash Checksum** ✅
   - Expected: `cd27d3045991153d2c7fce73c1859e602c24b3dcc1f8c5b4599ce6cfc0389d1a`
   - Verify: `sha256sum pec_enhanced_reporter.py`
   - Detects: Any accidental modification

2. **Git Commit Lock** ✅
   - Frozen at commit: `368b184`
   - Rollback command: `git checkout 368b184 -- pec_enhanced_reporter.py`
   - Protection: Full commit history preserved

3. **Read-Only Backup** ✅
   - File: `pec_enhanced_reporter_LOCKED_FROZEN.py`
   - Purpose: Reference copy to compare against
   - Use: `diff pec_enhanced_reporter.py pec_enhanced_reporter_LOCKED_FROZEN.py`

4. **Lock Manifest** ✅
   - File: `PEC_REPORTER_LOCK_MANIFEST.json`
   - Contains: Official hash, file size, line count
   - Use: Machine-readable verification

---

## 🎯 WORKFLOW (From Now On)

### Every Morning
```bash
# Verify template integrity
sha256sum pec_enhanced_reporter.py | grep "cd27d3045991153d2c7fce73c1859e602c24b3dcc1f8c5b4599ce6cfc0389d1a"
# If no match → ALERT: File modified
```

### Generate Daily Report
```bash
python3 pec_enhanced_reporter.py > PEC_REPORT_$(date +%Y-%m-%d).txt
```

### Monitor Signals
```bash
# Daemon automatically updates SENT_SIGNALS.jsonl
# Daily cron creates SENT_SIGNALS_CUMULATIVE_YYYY-MM-DD.jsonl at 23:00 GMT+7
# Reporter auto-uses latest cumulative file
```

### No Code Changes
```bash
# If you want to change something:
# STOP
# Ask Jetro first
# Wait for approval
# Full review required
```

---

## 📞 IF YOU NEED TO CHANGE THE TEMPLATE

**This is critical: Changes are not allowed without explicit user approval.**

### Process
1. **Don't make changes** - Stop immediately
2. **Document the reason** - Why is change needed?
3. **Contact user** - Explain the issue
4. **Wait for approval** - User must explicitly authorize
5. **Full review** - Every change gets reviewed
6. **Test thoroughly** - Verify nothing breaks
7. **Update hash** - New checksum for locked version
8. **Create new lock** - Re-freeze with new hash
9. **Document change** - Add to immutable changelog

### Who can approve changes?
- **Jetro (User)** ✅ Only person who can authorize

### What happens after approval?
1. Changes documented in detail
2. New hash generated
3. Commit message explains why
4. This guide updated
5. New lock manifest created
6. Old version archived

---

## 📋 REFERENCE: LOCKED COMPONENTS

### Code (Immutable)
- ✅ All 1,562 lines of pec_enhanced_reporter.py
- ✅ All 8 report sections
- ✅ All calculations and formulas
- ✅ All dimension definitions
- ✅ All column formats

### Data (Always Updated)
- ✅ SENT_SIGNALS.jsonl (live, growing)
- ✅ SENT_SIGNALS_CUMULATIVE_*.jsonl (daily snapshots)
- ✅ Signal metrics (TP, SL, TIMEOUT, etc. - calculated each run)

### Structure (Frozen)
- ✅ Column order and names
- ✅ Sorting rules
- ✅ Calculation formulas
- ✅ Timezone handling (GMT+7)
- ✅ Summary metrics (14 values)

---

## ✅ SIGN-OFF

**This template is locked and protected.**

- Hash verified ✅
- Git protected ✅
- Backup created ✅
- Manifest generated ✅
- Documentation complete ✅

**Do not change. Ever. Without explicit approval.**

---

**Locked by:** Nox (Agent)  
**For:** Jetro (User)  
**Date:** 2026-03-05 09:54 GMT+7  
**Reason:** Foundation of backtest system - zero tolerance for changes  

🔒 **IMMUTABLE**
