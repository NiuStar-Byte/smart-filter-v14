# 🔒 TRACKERS LOCK - IMMUTABLE (2026-03-05 11:53 GMT+7)

**Status:** ✅ **ALL LIVE TRACKERS LOCKED & FROZEN**  
**Decision:** NO MODIFICATIONS without explicit user confirmation  
**Purpose:** Ensure data integrity during accumulation phase (Mar 5 → Mar 10 decision)  
**Git Commit:** (pending - current session)  
**Git Tag:** `stable-trackers-locked-2026-03-05`

---

## 🔒 LOCKED TRACKERS (4 Active Live Trackers)

### **1. COMPARE_AB_TEST_LOCKED.py** - Phase 2-FIXED A/B Test
- **Purpose:** Track Phase 2-FIXED vs FOUNDATION baseline
- **SHA256:** `424d239eadb70a1026ab105c9d9602bae44c8257b78a5cebbb12d95e51211ba3`
- **Structure:** LOCKED ✅
  - Signal composition explanation (foundation + Phase 2-PRIOR + Phase 2-FIXED)
  - Comparison table (FOUNDATION A vs PHASE 2-FIXED B)
  - Success criterion display
  - Decision date (Mar 10 14:30 GMT+7)
- **What's Dynamic:** Only the signal counts (Phase 2-FIXED accumulates)
- **What's FIXED:** Foundation baseline (853), cutoff dates, success thresholds
- **Crash-Safe:** Reads from SENT_SIGNALS.jsonl with signal_origin filter

---

### **2. PHASE3_TRACKER.py** - Reversal Quality Gates (Phase 3B)
- **Purpose:** Track REVERSAL signals only (Phase 3B quality validation)
- **SHA256:** `a64a12b7bb2a32534030e2b4bb37ac8e7350e3a01061646bf13e6475361ab329`
- **Structure:** LOCKED ✅
  - Deployment info (2026-03-03 19:31 GMT+7)
  - Focus statement (REVERSAL signals only)
  - Metrics table (total, closed, WR, LONG/SHORT, regime breakdown)
  - Performance summary vs Foundation
  - Decision framework
- **What's Dynamic:** Signal counts, WR%, P&L (accumulates)
- **What's FIXED:** Deployment time, Foundation baseline (25.7% WR)
- **Crash-Safe:** Filters by route='REVERSAL' and fired_time >= Mar 3 19:31 UTC

---

### **3. track_rr_comparison.py** - RR Variant Comparison
- **Purpose:** Compare 2.0:1 PROD (fixed) vs 1.5:1 NEW (accumulating)
- **SHA256:** `ad27f98315f14cfb38f27ea7b65a99f62ff396da39aa12c2bdcc7a04d9c8e056`
- **Structure:** LOCKED ✅
  - RR variant labels (2.0:1 PROD vs 1.5:1 NEW RR)
  - Cutoff time explanation (Feb 27 15:55 UTC → Mar 4 17:51 UTC boundary)
  - Table with F (FIXED) vs D (DYNAMIC) labels
  - 3.0:1 RR exclusion note (legacy, pre-optimization)
  - Impact analysis (WR, P&L, duration changes)
- **What's Dynamic:** 1.5:1 RR counts, metrics
- **What's FIXED:** 2.0:1 RR at 895 signals (locked period)
- **Crash-Safe:** Excludes 3.0 RR, filters by achieved_rr field

---

### **4. pec_enhanced_reporter.py** - PEC Enhanced Report (ALREADY LOCKED)
- **Purpose:** Comprehensive signal performance analysis
- **SHA256:** `02913b861199f99dbd9bfe822d42619e8e4bcc500adcfd5fc3ca46d9981dced2`
- **Status:** ✅ LOCKED since 2026-03-05 10:30 GMT+7
- **Protection:** Multi-layered (hash, git tag, backup frozen copy)
- **Structure:** IMMUTABLE - 8 sections, all dynamic calculations
- **Crash-Safe:** Auto-detects latest CUMULATIVE or LIVE file

---

## 🛡️ CRASH RECOVERY SAFEGUARDS

### **Scenario 1: Daemon Crash / Signal Accumulation Stops**
**Problem:** Daemon dies, signals stop being written to SENT_SIGNALS.jsonl  
**Detection:** Trackers show no new signals for >30 minutes  
**Recovery:**
```bash
# Step 1: Check daemon status
ps aux | grep "main.py"

# Step 2: Check latest signal timestamp
tail -1 /Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl | jq '.fired_time_utc'

# Step 3: Restart daemon
pkill -f "python3.*main.py"
sleep 2
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main
nohup python3 main.py > /Users/geniustarigan/.openclaw/workspace/main_daemon.log 2>&1 &

# Step 4: Verify recovery (check logs for new signals)
tail -f /Users/geniustarigan/.openclaw/workspace/main_daemon.log | grep "Telegram alert sent"
```

---

### **Scenario 2: API Rate Limit / Network Timeout**
**Problem:** KuCoin/Binance API limits cause cycle slowdown, signals queue up  
**Detection:** CYCLE_SLEEP timeout increases, signals delayed  
**Recovery:**
```bash
# Step 1: Check current CYCLE_SLEEP setting
grep "CYCLE_SLEEP" /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/main.py

# Step 2: If <1000s, increase to 1000s (16.7 min cycles)
# Edit main.py, set CYCLE_SLEEP = 1000

# Step 3: Restart daemon
pkill -f "python3.*main.py"
sleep 2
nohup python3 /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/main.py > /Users/geniustarigan/.openclaw/workspace/main_daemon.log 2>&1 &

# Step 4: Monitor cycle speed
tail -f /Users/geniustarigan/.openclaw/workspace/main_daemon.log | grep "CYCLE"
```

---

### **Scenario 3: File Corruption / Stale Data**
**Problem:** SENT_SIGNALS.jsonl becomes corrupted or contains invalid JSON  
**Detection:** Trackers fail to load or show incorrect numbers  
**Recovery:**
```bash
# Step 1: Verify file integrity
python3 -c "import json; [json.loads(line) for line in open('SENT_SIGNALS.jsonl')]"

# Step 2: If fails, restore from latest git commit
git checkout SENT_SIGNALS.jsonl

# Step 3: Verify FOUNDATION baseline still = 853
python3 << 'EOF'
import json
foundation_count = 0
with open('SENT_SIGNALS.jsonl') as f:
    for line in f:
        sig = json.loads(line.strip())
        if sig.get('signal_origin') == 'FOUNDATION':
            foundation_count += 1
print(f"FOUNDATION signals: {foundation_count}")
if foundation_count == 853:
    print("✅ FOUNDATION baseline intact")
else:
    print(f"❌ FOUNDATION corrupted: {foundation_count} (should be 853)")
EOF
```

---

### **Scenario 4: Tracker Modification (PREVENTS CHANGE)**
**Problem:** Someone modifies a tracker file accidentally  
**Detection:** Daily hash verification (automated)  
**Recovery:**
```bash
# Step 1: Daily verification (add to cron or manual check)
python3 << 'EOF'
import hashlib
import os

trackers = {
    'COMPARE_AB_TEST_LOCKED.py': '424d239eadb70a1026ab105c9d9602bae44c8257b78a5cebbb12d95e51211ba3',
    'PHASE3_TRACKER.py': 'a64a12b7bb2a32534030e2b4bb37ac8e7350e3a01061646bf13e6475361ab329',
    'track_rr_comparison.py': 'ad27f98315f14cfb38f27ea7b65a99f62ff396da39aa12c2bdcc7a04d9c8e056',
    'pec_enhanced_reporter.py': '02913b861199f99dbd9bfe822d42619e8e4bcc500adcfd5fc3ca46d9981dced2',
}

for tracker, expected_hash in trackers.items():
    with open(tracker, 'rb') as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    
    if actual_hash == expected_hash:
        print(f"✅ {tracker} - INTACT")
    else:
        print(f"❌ {tracker} - MODIFIED!")
        print(f"   Expected: {expected_hash}")
        print(f"   Actual:   {actual_hash}")
        print(f"   ACTION: git checkout {tracker}")
EOF

# Step 2: If hash mismatch, restore immediately
git checkout <tracker_name>
```

---

## 📋 ACCUMULATION PHASE (Mar 5-10)

### **What Happens During Accumulation:**
1. ✅ Daemon fires new signals → appended to SENT_SIGNALS.jsonl
2. ✅ Trackers read live file → counts update automatically
3. ✅ No code modifications → only data accumulates
4. ❌ NO tracker changes → all structures frozen

### **Daily Monitoring (No Action Required):**
```bash
# View Phase 2-FIXED progress
python3 COMPARE_AB_TEST_LOCKED.py --once

# View Phase 3B progress
python3 PHASE3_TRACKER.py --once

# View RR variant comparison
python3 track_rr_comparison.py --once

# View comprehensive PEC report
python3 pec_enhanced_reporter.py --once
```

### **Decision Point (Mar 10 14:30 GMT+7):**
- Compare Phase 2-FIXED WR vs baseline (25.7%)
- Evaluate Phase 3B performance
- Assess 1.5 RR impact
- Make go/no-go decisions

---

## 🔐 FILE LOCATIONS & IMMUTABILITY

| File | Purpose | Hash | Status | Modifications |
|------|---------|------|--------|---|
| COMPARE_AB_TEST_LOCKED.py | Phase 2-FIXED tracker | 424d23... | 🔒 LOCKED | NONE allowed |
| PHASE3_TRACKER.py | Phase 3B tracker | a64a12... | 🔒 LOCKED | NONE allowed |
| track_rr_comparison.py | RR comparison | ad27f9... | 🔒 LOCKED | NONE allowed |
| pec_enhanced_reporter.py | PEC report | 029138... | 🔒 LOCKED | NONE allowed |
| SENT_SIGNALS.jsonl | Signal data | — | 📝 APPEND ONLY | Accumulate only |
| main.py (daemon) | Signal generation | — | ⏸️ FROZEN | NO changes until Mar 10 |

---

## ⚡ QUICK REFERENCE - CRASH RESPONSE

| Issue | Check | Action |
|-------|-------|--------|
| Signals stop accumulating | `tail -1 SENT_SIGNALS.jsonl \| jq '.fired_time_utc'` | Restart daemon (see Scenario 1) |
| Tracker shows old data | `python3 <tracker> --once` | Check SENT_SIGNALS.jsonl for new lines |
| Hash mismatch warning | `shasum -a 256 <tracker>` | `git checkout <tracker>` (Scenario 4) |
| API timeout errors | `tail -f main_daemon.log \| grep timeout` | Increase CYCLE_SLEEP to 1000s (Scenario 2) |
| JSON parse error | `python3 -c "import json; [json.loads(line) for line in open('SENT_SIGNALS.jsonl')]"` | `git checkout SENT_SIGNALS.jsonl` (Scenario 3) |

---

## 🎯 COMMITMENT

**From now on (Mar 5 11:53 GMT+7 onwards):**

✅ **ONLY ACCUMULATE** - New signals append to file, trackers auto-update  
✅ **NO TRACKER CHANGES** - All 4 trackers frozen, no modifications  
✅ **CRASH-SAFE** - Recovery procedures documented above  
✅ **HASH-VERIFIED** - Daily verification prevents accidental changes  
✅ **GIT-TAGGED** - `stable-trackers-locked-2026-03-05` marks this checkpoint  

**NO further modifications until Mar 10 decision point.**

---

## 📝 Change Log (If User Requests Changes After This Point)

Any modifications must be:
1. Explicitly requested by user
2. Documented with reason
3. Hash recalculated
4. New git tag created
5. TRACKERS_LOCK_FINAL.md updated

**Current Frozen State:** 2026-03-05 11:53 GMT+7

---
