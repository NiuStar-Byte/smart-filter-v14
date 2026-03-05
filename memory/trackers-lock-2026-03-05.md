# 🔒 TRACKERS LOCK DOCUMENTATION (2026-03-05 11:53-11:59 GMT+7)

## CRITICAL: All 4 Live Trackers Locked Until Mar 10

**Status:** ✅ IMMUTABLE - NO MODIFICATIONS ALLOWED  
**Period:** Mar 5 11:53 GMT+7 → Mar 10 14:30 GMT+7  
**Git Commit:** 8bf435a (TRACKERS LOCK) + 04a0413 (MEMORY)  
**Git Tag:** `stable-trackers-locked-2026-03-05`  
**Verification:** `python3 verify_trackers_lock.py` (daily)

---

## 📋 LOCKED TRACKERS (4 Active)

### **1. COMPARE_AB_TEST_LOCKED.py**
- **Purpose:** Track Phase 2-FIXED vs FOUNDATION baseline (25.7% WR)
- **SHA256:** `424d239eadb70a1026ab105c9d9602bae44c8257b78a5cebbb12d95e51211ba3`
- **Backup:** `COMPARE_AB_TEST_LOCKED_LOCKED_FROZEN.py` (immutable reference)
- **What's LOCKED:**
  - Signal composition explanation
  - Comparison table format
  - Success criterion (≥25.7%)
  - Decision date (Mar 10 14:30 GMT+7)
  - Foundation baseline (853 signals)
- **What's DYNAMIC:**
  - Phase 2-FIXED count (accumulates)
  - Closed trade count (accumulates)
  - Win rate % (recalculated)
  - LONG/SHORT WR (recalculated)
- **Crash-Safe:** Reads SENT_SIGNALS.jsonl with `signal_origin='NEW'` filter
- **Run:** `python3 COMPARE_AB_TEST_LOCKED.py --once` or with live watch mode

---

### **2. PHASE3_TRACKER.py**
- **Purpose:** Track REVERSAL signals only (Phase 3B reversal quality gates)
- **SHA256:** `a64a12b7bb2a32534030e2b4bb37ac8e7350e3a01061646bf13e6475361ab329`
- **Backup:** `PHASE3_TRACKER_LOCKED_FROZEN.py` (immutable reference)
- **What's LOCKED:**
  - Deployment info (2026-03-03 19:31 GMT+7)
  - Focus statement (REVERSAL route only)
  - Metrics table structure
  - Regime breakdown format
  - Foundation baseline (25.7% WR, -$5,498.59 P&L)
  - Decision framework
- **What's DYNAMIC:**
  - Total REVERSAL signals (accumulates)
  - Closed REVERSAL trades (accumulates)
  - Win rate % (recalculated)
  - P&L total (recalculated)
  - Regime performance breakdown (recalculated)
- **Crash-Safe:** Filters by `route='REVERSAL'` and `fired_time >= 2026-03-03T19:31:00Z`
- **Run:** `python3 PHASE3_TRACKER.py --once` or with live watch mode

---

### **3. track_rr_comparison.py**
- **Purpose:** Compare 2.0:1 PROD (fixed period) vs 1.5:1 NEW RR (accumulating)
- **SHA256:** `ad27f98315f14cfb38f27ea7b65a99f62ff396da39aa12c2bdcc7a04d9c8e056`
- **Backup:** `track_rr_comparison_LOCKED_FROZEN.py` (immutable reference)
- **What's LOCKED:**
  - RR variant labels (2.0:1 PROD vs 1.5:1 NEW RR)
  - Cutoff time explanation (Feb 27 15:55 UTC → Mar 4 17:51 UTC boundary)
  - Table structure with F (FIXED) vs D (DYNAMIC) labels
  - 3.0:1 RR exclusion note (legacy, pre-optimization era)
  - 2.0:1 RR fixed count (895 signals, locked period)
  - Impact analysis format
- **What's DYNAMIC:**
  - 1.5:1 RR total count (accumulates as new signals fire)
  - 1.5:1 RR closed count (accumulates)
  - Win rates, P&L, durations (recalculated)
  - Signal volume change (recalculated)
- **Crash-Safe:** Excludes `achieved_rr=3.0`, filters by `achieved_rr in [2.0, 1.5]`
- **Run:** `python3 track_rr_comparison.py --once` or with live watch mode

---

### **4. pec_enhanced_reporter.py**
- **Purpose:** Comprehensive signal performance analysis (8 sections)
- **SHA256:** `02913b861199f99dbd9bfe822d42619e8e4bcc500adcfd5fc3ca46d9981dced2`
- **Backup:** `pec_enhanced_reporter_LOCKED_FROZEN.py` (immutable reference)
- **Status:** Already locked since 2026-03-05 10:30 GMT+7
- **What's LOCKED:**
  - All 8 section structure (header, aggregates, multi-dimensional, detailed list, summary, hierarchy, signal tiers, auto-detection)
  - Format and layout
  - Calculation formulas (all 14 metrics)
  - Foundation baseline reference (853 signals, 25.7% WR)
- **What's DYNAMIC:**
  - All signal counts (total, closed, TP, SL, TIMEOUT)
  - All win rates and P&L
  - All duration metrics
  - Auto-detects latest CUMULATIVE or LIVE file
- **Crash-Safe:** Auto-detects SENT_SIGNALS_CUMULATIVE or SENT_SIGNALS.jsonl
- **Run:** `python3 pec_enhanced_reporter.py` or `python3 pec_enhanced_reporter.py --once`

---

## 🛡️ CRASH RECOVERY PROCEDURES (DO NOT SKIP)

### **Scenario 1: Daemon Crash / Signals Stop Accumulating**
**Symptoms:**
- No new signals for >30 minutes
- `SENT_SIGNALS.jsonl` timestamp not advancing
- Trackers show stale data

**Recovery:**
```bash
# Check daemon status
ps aux | grep "main.py"

# Check latest signal timestamp
tail -1 SENT_SIGNALS.jsonl | jq '.fired_time_utc'

# If stale: Restart daemon
pkill -f "python3.*main.py"
sleep 2
cd smart-filter-v14-main
nohup python3 main.py > ../main_daemon.log 2>&1 &

# Verify recovery
tail -f ../main_daemon.log | grep "Telegram alert sent"
```

---

### **Scenario 2: API Rate Limit / Network Timeout**
**Symptoms:**
- Cycle slowdown (>120 seconds per cycle)
- Signals accumulate in queue
- Timeouts in logs: `timeout`, `rate limit`, `connection refused`

**Recovery:**
```bash
# Check current CYCLE_SLEEP
grep "CYCLE_SLEEP" smart-filter-v14-main/main.py

# If <1000, increase to 1000s (16.7 min cycles to accommodate API slowdown)
# Edit: smart-filter-v14-main/main.py
# Change: CYCLE_SLEEP = 1000

# Restart daemon
pkill -f "python3.*main.py"
sleep 2
nohup python3 smart-filter-v14-main/main.py > main_daemon.log 2>&1 &

# Monitor cycle speed
tail -f main_daemon.log | grep "CYCLE"
```

---

### **Scenario 3: File Corruption / Invalid JSON**
**Symptoms:**
- Trackers fail to load with JSON error
- Numbers don't match expected
- `python3 -c "import json; ..."` fails

**Recovery:**
```bash
# Test file integrity
python3 << 'EOF'
import json
try:
    with open('SENT_SIGNALS.jsonl') as f:
        for line in f:
            json.loads(line.strip())
    print("✅ File OK")
except Exception as e:
    print(f"❌ Corruption: {e}")
EOF

# If corrupted: Restore from git
git checkout SENT_SIGNALS.jsonl

# Verify FOUNDATION baseline still = 853
python3 << 'EOF'
import json
foundation_count = sum(1 for line in open('SENT_SIGNALS.jsonl') 
                       if json.loads(line.strip()).get('signal_origin') == 'FOUNDATION')
print(f"FOUNDATION signals: {foundation_count}")
assert foundation_count == 853, "Baseline corrupted!"
print("✅ Baseline intact")
EOF
```

---

### **Scenario 4: Tracker Code Modification (DETECTION)**
**Symptoms:**
- Hash verification fails: `❌ MODIFIED!`
- Tracker structure looks different
- Need to know: WHO changed it and WHY

**Recovery:**
```bash
# Run daily verification
python3 verify_trackers_lock.py

# If hash mismatch on any tracker:
# Step 1: Check what changed
git diff COMPARE_AB_TEST_LOCKED.py

# Step 2: Review changes (DO NOT APPROVE unless user requested)
# Step 3: Restore if unauthorized
git checkout COMPARE_AB_TEST_LOCKED.py

# Step 4: Verify hash restored
python3 verify_trackers_lock.py
```

---

## ⚡ QUICK RESPONSE GUIDE

| Issue | Detection | First Action |
|-------|-----------|--------------|
| **Signals stuck** | No new data >30 min | Restart daemon (Scenario 1) |
| **Slow cycles** | Cycle >120s | Increase CYCLE_SLEEP (Scenario 2) |
| **Tracker errors** | JSON parse fail | `git checkout SENT_SIGNALS.jsonl` (Scenario 3) |
| **Hash mismatch** | `verify_trackers_lock.py` fails | `git checkout <tracker>` (Scenario 4) |
| **Unknown state** | Unsure what's wrong | Run verify script + check logs |

---

## 📊 DAILY MONITORING (NO CHANGES REQUIRED)

**Each day (Mar 5-10):**
```bash
# Morning check
python3 COMPARE_AB_TEST_LOCKED.py --once

# Midday check  
python3 PHASE3_TRACKER.py --once

# Evening check
python3 track_rr_comparison.py --once

# Optional: Full report
python3 pec_enhanced_reporter.py --once

# Daily verification (automated recommended)
python3 verify_trackers_lock.py
```

**Expected behavior:**
- Signal counts increase (accumulating)
- Win rates fluctuate (variance with small sample)
- No code changes (all frozen)
- Hash verification passes (all intact)

---

## 🎯 DECISION POINT (Mar 10 14:30 GMT+7)

**What to evaluate:**
1. Phase 2-FIXED WR vs 25.7% baseline (APPROVE if ≥25.7%)
2. Phase 3B REVERSAL quality (track trend)
3. 1.5:1 RR impact vs 2.0:1 baseline
4. Overall system performance trend

**What NOT to change:**
- ❌ Tracker code (all frozen)
- ❌ Tracker structure (all locked)
- ❌ Signal format (all immutable)
- ❌ Hash values (all verified daily)

---

## 🔐 FILE LOCATIONS & PROTECTION

**Locked Trackers:**
- `/Users/geniustarigan/.openclaw/workspace/COMPARE_AB_TEST_LOCKED.py`
- `/Users/geniustarigan/.openclaw/workspace/PHASE3_TRACKER.py`
- `/Users/geniustarigan/.openclaw/workspace/track_rr_comparison.py`
- `/Users/geniustarigan/.openclaw/workspace/pec_enhanced_reporter.py`

**Frozen Backups (immutable references):**
- `/Users/geniustarigan/.openclaw/workspace/COMPARE_AB_TEST_LOCKED_LOCKED_FROZEN.py`
- `/Users/geniustarigan/.openclaw/workspace/PHASE3_TRACKER_LOCKED_FROZEN.py`
- `/Users/geniustarigan/.openclaw/workspace/track_rr_comparison_LOCKED_FROZEN.py`
- `/Users/geniustarigan/.openclaw/workspace/pec_enhanced_reporter_LOCKED_FROZEN.py`

**Documentation:**
- `/Users/geniustarigan/.openclaw/workspace/TRACKERS_LOCK_FINAL.md` (comprehensive)
- `/Users/geniustarigan/.openclaw/workspace/memory/trackers-lock-2026-03-05.md` (this file)

**Verification:**
- `/Users/geniustarigan/.openclaw/workspace/verify_trackers_lock.py` (daily check)

---

## ✅ COMMIT HISTORY

- **8bf435a:** 🔒 [TRACKERS LOCK] Lock all 4 live trackers - IMMUTABLE until Mar 10
- **04a0413:** [MEMORY] Document TRACKERS LOCK - all immutable until Mar 10

**Git Tag:** `stable-trackers-locked-2026-03-05`

---

**REMINDER:** From now until Mar 10, this is ACCUMULATION mode only. No code changes. No format changes. Just let signals accumulate and trackers auto-update. If something breaks, follow the crash recovery procedures above. ✅
