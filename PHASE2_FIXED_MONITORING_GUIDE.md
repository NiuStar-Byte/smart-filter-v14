# 🦇 PHASE 2-FIXED MONITORING GUIDE

**Deployment:** 2026-03-03 17:36 GMT+7 (Commit: `cf0ca4f`)  
**Status:** ✅ LIVE (Daemon restarted with new code)  
**Objective:** Monitor SHORT signal recovery in BEAR regime

---

## 📊 QUICK MONITORING COMMANDS

### Daily Report (5-minute read)
```bash
cd ~/.openclaw/workspace/smart-filter-v14-main
python3 track_phase2_fixed.py          # Full performance summary
```

### Real-Time Log Watching (Live)
```bash
cd ~/.openclaw/workspace/smart-filter-v14-main
python3 track_phase2_fixed.py --watch  # Follow [PHASE2-FIXED] tags in main_daemon.log
```

### SHORT Signals Only (Focus)
```bash
python3 track_phase2_fixed.py --short  # SHORT performance by regime
```

### BEAR Regime Only (Key Metric)
```bash
python3 track_phase2_fixed.py --bear   # BEAR LONG vs BEAR SHORT detailed breakdown
```

---

## 🎯 KEY METRICS TO WATCH

### 1. BEAR SHORT WR (PRIMARY METRIC)
**What:** Win rate of SHORT signals in BEAR market regime  
**Baseline (Phase 2 original):** 0.0% (1/1 signals, completely broken)  
**Target:** 25%+ (healthy recovery)  
**Success threshold:** > 15% (shows recovery is working)  
**Current:** Tracking live since 2026-03-03 17:36

**Command:**
```bash
python3 track_phase2_fixed.py --bear
# Look for: BEAR > SHORT > WR: XX.X%
```

### 2. BULL LONG WR (REGRESSION CHECK)
**What:** Ensure we didn't break LONG signals in BULL  
**Baseline (Phase 2 original):** ~31%  
**Target:** Maintain 30%+ (no regression)  
**Alert threshold:** < 28% (significant drop)

**Command:**
```bash
python3 track_phase2_fixed.py
# Look for: BULL > LONG > WR: XX.X%
```

### 3. Overall SHORT WR (System Level)
**What:** All SHORT signals across all regimes  
**Baseline (Phase 2 original):** 13.3%  
**Target:** 20%+ (system improvement)  
**Current tracking:** Live

**Command:**
```bash
python3 track_phase2_fixed.py --short
# Sum all SHORT WR: BULL + BEAR + RANGE
```

### 4. Signal Volume (Health Check)
**What:** Total signals being fired  
**Baseline (Phase 1):** ~11-12/day  
**Expected (Phase 2-FIXED):** 9-11/day (slight reduction from stricter gates)  
**Alert:** < 5/day (too restrictive)

**Command:**
```bash
# Check last 24 hours
grep "fired_time_utc" SENT_SIGNALS.jsonl | tail -500 | wc -l
```

---

## 📅 MONITORING SCHEDULE

### Daily (Every 8 hours)
```bash
# 3 times per day at:
# - 08:00 GMT+7
# - 16:00 GMT+7
# - 00:00 GMT+7

python3 track_phase2_fixed.py

# Compare to previous run:
# 1. Is BEAR SHORT WR improving? (0% → 5% → 10% → 15%+)
# 2. Is BULL LONG WR staying healthy? (>28%)
# 3. Any [ERROR] tags in main_daemon.log?
```

### Real-Time (When Monitoring)
```bash
# In a terminal, watch live daemon:
tail -f main_daemon.log | grep "PHASE2-FIXED"

# You should see:
# [PHASE2-FIXED] 15min BTC-USDT SHORT ✓ ALL GATES PASS (BEAR)
# [PHASE2-FIXED] 30min ETH-USDT LONG ✓ ALL GATES PASS (BULL)
# [PHASE2-FIXED-THRESHOLD] ... score=76.5 | threshold=14 | ...
# [PHASE2-FIXED-REJECT] ... 12.3 < 18 (below REVERSAL threshold)
```

### Weekly (Day 7 - Mar 10 14:30 GMT+7)
```bash
# Run comprehensive report
python3 pec_enhanced_reporter.py > phase2_fixed_day7_report.txt

# Compare metrics:
# - BEAR SHORT: Compare Day 1 vs Day 7
# - BULL LONG: Check for regressions
# - Overall WR: Should be 30%+
# - Signal count: Should be stable

# Final decision:
# ✅ APPROVE: If BEAR SHORT > 15% and BULL LONG > 28%
# ⚠️  INVESTIGATE: If metrics marginal, need more data
# ❌ ROLLBACK: If failures, debug issues
```

---

## 🔍 WHAT THE LOGS TELL YOU

### Good Sign (Gates Working Correctly)
```
[PHASE2-FIXED] 15min BTC-USDT SHORT ✓ ALL GATES PASS (BEAR)
[PHASE2-FIXED] 30min ETH-USDT LONG ✓ ALL GATES PASS (BULL)
[PHASE2-FIXED-THRESHOLD] ... score=78 | threshold=14 | ... PASS
```

**Interpretation:**
- SHORT signals are passing gates in BEAR ✅
- LONG signals are passing in BULL ✅
- Thresholds are direction-aware ✅

### Warning Sign (Possible Regression)
```
[PHASE2-FIXED] 15min BTC-USDT SHORT REJECTED - failed: ['trend_alignment']
[PHASE2-FIXED] 15min BTC-USDT SHORT REJECTED - failed: ['candle_structure']
```

**Interpretation:**
- SHORT signals failing Gate 3 or 4 (candles/trend)
- May indicate gate calibration too strict
- Monitor daily - expect some rejections

### Error Sign (Code Problem)
```
[PHASE2-FIXED] Error checking gates for BTC: ModuleNotFoundError
[PHASE2-FIXED] Error in 30min threshold: AttributeError
```

**Interpretation:**
- Code bug, not design issue
- **Action:** Check main_daemon.log full context
- **Rollback:** `git revert cf0ca4f`

---

## 📈 EXPECTED PROGRESSION (If Working Correctly)

### Day 1 (Mar 3, Evening)
- BEAR SHORT WR: 0-5% (still recovering)
- Total signals: Normal (9-12/day)
- BULL LONG WR: >28% (maintained)
- **Status:** Early data, normal variation

### Day 3 (Mar 5)
- BEAR SHORT WR: 5-15% (showing improvement)
- ~30 closed BEAR SHORT trades accumulated
- BULL LONG WR: >28% (maintained)
- **Status:** Trend visible

### Day 5 (Mar 7)
- BEAR SHORT WR: 10-20% (good progress)
- ~50+ closed BEAR SHORT trades
- BULL LONG WR: >28% (maintained)
- **Status:** Confidence building

### Day 7 (Mar 9)
- BEAR SHORT WR: 15-25%+ (success!)
- ~70+ closed BEAR SHORT trades
- BULL LONG WR: 30-33% (improved)
- **Status:** DECISION TIME

---

## 🚨 RED FLAGS & ACTIONS

### RED FLAG 1: BEAR SHORT Still Below 10% After 3 Days
**Indicates:** Gates still too restrictive for SHORT  
**Action:** 
1. Check log tags for which gates are blocking SHORT most
2. Review Gate 3 (Trend) and Gate 4 (Candles) logic
3. Minor threshold adjustment may be needed
4. Don't panic - normal tuning

### RED FLAG 2: BULL LONG Drops Below 28%
**Indicates:** New gates broken LONG signals  
**Action:**
1. Check which gates failing LONG in BULL
2. Likely issue: Gate 3 or 4 logic
3. **Consider:** Revert and debug, OR
4. Adjust thresholds slightly

### RED FLAG 3: Frequent [PHASE2-FIXED] Error Tags
**Indicates:** Code bug, not design  
**Action:**
1. Check full error message in main_daemon.log
2. Fix code issue
3. Restart daemon
4. **Or:** Rollback: `git revert cf0ca4f`

### RED FLAG 4: Signal Rate Drops Below 5/Day
**Indicates:** Gates filtering too aggressively  
**Action:**
1. Expected: 9-12/day like Phase 1
2. If <5/day: adjust thresholds or gate criteria
3. Loosen base thresholds by 1-2 points

---

## 📊 HISTORICAL COMPARISON

### Phase 2 Original (Broken)
```
BEAR REGIME:
  LONG:  1 signal,  0% WR  ← Ultra-rare, super filtered
  SHORT: 1 signal,  0% WR  ← 99% reduction from Phase 1 (111→1)
  
BULL REGIME:
  LONG:  ~300 signals, 31% WR
  SHORT: 1 signal,  0% WR
```

### Phase 2-FIXED (Expected)
```
BEAR REGIME:
  LONG:  ~10-15 signals, ~25-30% WR (easier criteria)
  SHORT: ~10-15 signals, ~25%+ WR (TE: PRIMARY RECOVERY)
  
BULL REGIME:
  LONG:  ~250-300 signals, ~30-32% WR (maintained)
  SHORT: ~5-10 signals, ~10-15% WR (still penalized, as designed)
```

---

## 🛠️ ADVANCED DEBUGGING

### If SHORT Still Not Recovering After 3 Days

**Step 1: Check which gate is blocking SHORT**
```bash
# Count gate failures in logs
grep "PHASE2-FIXED.*SHORT.*REJECTED" main_daemon.log | wc -l
grep "PHASE2-FIXED.*SHORT.*failed" main_daemon.log | grep trend_alignment | wc -l
grep "PHASE2-FIXED.*SHORT.*failed" main_daemon.log | grep candle_structure | wc -l
grep "PHASE2-FIXED.*SHORT.*failed" main_daemon.log | grep momentum_alignment | wc -l
```

**Step 2: Analyze specific failing patterns**
```bash
# Show recent SHORT rejections
grep "PHASE2-FIXED.*SHORT.*REJECTED" main_daemon.log | tail -20
```

**Step 3: Check regime detection**
```bash
# Are BEAR signals being detected?
grep "\[SUMMARY\] Market regime detected: BEAR" main_daemon.log | wc -l
```

### If BULL LONG Regressed

**Step 1: Verify LONG passing gates**
```bash
grep "PHASE2-FIXED.*LONG.*APPROVED\|PASS" main_daemon.log | tail -20
```

**Step 2: Check for systematic rejections**
```bash
grep "PHASE2-FIXED.*LONG.*REJECTED" main_daemon.log | tail -10
```

---

## ✅ SUCCESS CHECKLIST (Day 7)

- [ ] BEAR SHORT WR ≥ 15% (or higher)
- [ ] BULL LONG WR ≥ 28% (no regression)
- [ ] Total signal rate 8-12/day (normal)
- [ ] Zero [ERROR] tags in logs (or minor, unrelated)
- [ ] ~70+ closed BEAR SHORT trades collected
- [ ] Overall system WR ≥ 30%
- [ ] Phase 4A still working independently

If ALL checked ✅ → **APPROVE Phase 2-FIXED permanently**

If some ⚠️ → **Minor tuning or revert**

---

## 📞 SUPPORT & ROLLBACK

### View Full Details
```bash
# Complete log analysis
tail -100 main_daemon.log

# Full performance report
python3 pec_enhanced_reporter.py | grep -A 50 "BY REGIME"
```

### Rollback (If Needed)
```bash
cd ~/.openclaw/workspace/smart-filter-v14-main

# Revert to Phase 2 original
git revert cf0ca4f

# Restart daemon
pkill -f "python3 main.py"
sleep 2
nohup python3 main.py > main_daemon.log 2>&1 &
```

### Quick Health Check
```bash
# Is daemon running?
ps aux | grep "python3 main.py" | grep -v grep

# Recent log entries
tail -30 main_daemon.log

# Syntax errors?
python3 -m py_compile main.py && echo "✅ OK" || echo "❌ ERROR"
```

---

**Last Updated:** 2026-03-03 17:36 GMT+7  
**Status:** LIVE - Monitoring Phase 2-FIXED deployment  
**Next Review:** 2026-03-10 14:30 GMT+7 (Day 7 Decision)
