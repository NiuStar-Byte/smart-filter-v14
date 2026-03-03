# 📊 HOW TO TRACK PHASE 3B - REVERSAL QUALITY GATES

**Status:** ✅ Phase 3B is INTEGRATED but INACTIVE (no REVERSAL signals yet)  
**Last Update:** 2026-03-03 21:33 GMT+7

---

## 🎯 What is Phase 3B?

**Phase 3B = Reversal Quality Gate** (deployed Mar 3 19:31 GMT+7)

4-gate validation for REVERSAL signals:
- **RQ1:** Detector Consensus (2-3 detectors agree)
- **RQ2:** Momentum Alignment (RSI/MACD thresholds)
- **RQ3:** Trend Strength (ADX assessment)
- **RQ4:** Direction-Regime Match (SHORT in BEAR, LONG in BULL)

If all gates pass → REVERSAL approved  
If gates fail → REVERSAL rejected or routed to TREND_CONTINUATION

---

## 🔍 How to Check Phase 3B Activity

### Method 1: Real-Time Daemon Log (BEST)
```bash
tail -f main_daemon.log | grep "PHASE3B"
```

**Output lines you'll see:**
```
[PHASE3B-RQ] 15min BTC LONG: RQ1✓ RQ2✓ RQ3✓ RQ4✗ → Strength=75%
[PHASE3B-APPROVED] 15min BTC: REVERSAL approved (4/4 gates)
[PHASE3B-FALLBACK] 15min BTC: REVERSAL rejected (RQ4 fails) → Routed to TREND_CONTINUATION
[PHASE3B-SCORE] 15min BTC: +25 REVERSAL (score: 65%)
```

### Method 2: Auto-Refresh Monitoring Script (WATCH MODE)
```bash
python3 track_phase3b.py --watch
```

Shows real-time REVERSAL signal processing:
```
📊 PHASE 3B PERFORMANCE REPORT
Last updated: 2026-03-03 21:35:42 UTC

📈 REVERSALS OVERVIEW:
  Total Checked: 12
  ✅ Approved: 8 (66.7%)
  ❌ Rejected: 4 (33.3%)

🚪 GATE PERFORMANCE:
  RQ1 Pass Rate: 12 (100%)
  RQ2 Pass Rate: 11 (91.7%)
  RQ3 Pass Rate: 10 (83.3%)
  RQ4 Pass Rate: 9 (75.0%)
```

### Method 3: Full Phase 3B Report
```bash
python3 track_phase3b.py
```

Shows:
- Reversal approval rate
- Gate pass rates
- By-direction performance (SHORT vs LONG)
- By-combo analysis (BEAR+SHORT, BULL+LONG, etc.)

### Method 4: Daemon Log Grep (HISTORICAL)
```bash
grep "PHASE3B" main_daemon.log | head -20
```

Shows all Phase 3B actions from daemon startup.

---

## 🎯 Why Phase 3B Shows 0 Right Now

**Current Status:** 0 reversals checked (normal)

**Why?**
- Phase 2-FIXED only fired 3 fresh signals (13:16 UTC onwards)
- Those 3 signals are likely TREND_CONTINUATION route (not REVERSAL)
- REVERSAL signals haven't arrived yet in the fresh period

**When Phase 3B will activate:**
- ✅ As more signals accumulate (50+ per day)
- ✅ When REVERSAL route signals start firing
- ✅ Expected: Tomorrow (Mar 4) or Mar 5

---

## 📊 Example Phase 3B Output (When Active)

### Real-Time Log (tail -f)
```
[PHASE3B-RQ] 15min MATIC LONG: RQ1✓ RQ2✓ RQ3✓ RQ4✓ → Strength=92%
[PHASE3B-APPROVED] 15min MATIC: REVERSAL approved (4/4 gates pass)
[PHASE3B-SCORE] 15min MATIC: +25 REVERSAL_LONG_BULL (score: 85%)

[PHASE3B-RQ] 30min SOL SHORT: RQ1✓ RQ2✓ RQ3✗ RQ4✓ → Strength=65%
[PHASE3B-FALLBACK] 30min SOL: REVERSAL rejected (RQ3 fails, ADX < 20) → Routed to TREND_CONTINUATION
[PHASE3B-SCORE] 30min SOL: +0 TREND_CONTINUATION (fallback)

[PHASE3B-RQ] 1h ETH LONG: RQ1✓ RQ2✗ RQ3✓ RQ4✓ → Strength=70%
[PHASE3B-APPROVED] 1h ETH: REVERSAL approved (3/4 gates, favorable combo)
[PHASE3B-SCORE] 1h ETH: +15 REVERSAL_LONG_BULL (score: 72%)
```

**Reading the tags:**
- `RQ1✓` = Gate 1 passed
- `RQ2✗` = Gate 2 failed
- `Strength=70%` = Overall signal quality score (0-100%)
- `APPROVED` = Signal sent to trader
- `FALLBACK` = Rejected, routed elsewhere
- `SCORE` = Final route recommendation

---

## 🔑 Key Metrics to Watch (When Active)

### Approval Rate
- **Target:** 60-80% (not too strict, not too lenient)
- **Too strict (<40%):** Gates killing good reversals
- **Too lenient (>90%):** Gates not filtering bad ones

### Gate Pass Rates
- **RQ1 (Detector):** Should be 85%+ (consensus mostly there)
- **RQ2 (Momentum):** Should be 70-80% (RSI/MACD reasonable)
- **RQ3 (Trend):** Should be 65-75% (ADX varies)
- **RQ4 (Direction):** Should be 80%+ (mostly aligned)

### By-Direction Performance
- **REVERSAL SHORT WR:** Should improve vs Phase 1 (target: 15%+)
- **REVERSAL LONG WR:** Should maintain (target: 22%+)

---

## 🚨 Troubleshooting Phase 3B

### Problem: track_phase3b.py shows "Total Checked: 0"

**Possible causes:**
1. ✅ Normal — not enough reversals fired yet (Mar 3 23:33, only fresh signals)
2. ⚠️ Phase 3B code not running — check daemon status
3. ⚠️ No log tags matching `[PHASE3B]` — grep to verify

**Fix:**
```bash
# Check if daemon is running
ps aux | grep "python3 main.py"

# Check if Phase 3B tags exist in logs
grep "[PHASE3B" main_daemon.log | wc -l

# If 0 matches, Phase 3B hasn't fired yet (normal for early stage)
# If >0 matches, monitoring script has data to read
```

### Problem: track_phase3b.py hangs

```bash
# Kill and restart
pkill -f "track_phase3b.py"
python3 track_phase3b.py --watch
```

---

## 📈 When Will Phase 3B Have Data?

| Timeline | Signal Count | Phase 3B Status |
|----------|-------------|-----------------|
| **Today (Mar 3)** | 3 signals | 0% collected (normal) |
| **Tomorrow (Mar 4)** | ~50-60 signals | ~5-10 reversals (expect data) |
| **Mar 5-7** | ~100-150 signals | ~15-30 reversals (good sample) |
| **Mar 10 (Decision)** | ~200+ signals | ~40-50 reversals (decision-ready) |

---

## 🎯 Recommended Monitoring Plan

### Daily (Every 8-12 hours)
```bash
# Quick check
python3 track_phase3b.py --short

# Full report
python3 track_phase3b.py
```

### Real-Time (While trading)
```bash
# Live log tail (watch for PHASE3B tags)
tail -f main_daemon.log | grep "PHASE3B"
```

### Decision Time (Mar 10)
```bash
# Final comprehensive report
python3 track_phase3b.py
python3 COMPARE_AB_TEST.py --once
python3 pec_enhanced_reporter.py
```

---

## ✅ Current Status

| Aspect | Status | Notes |
|--------|--------|-------|
| **Phase 3B Code** | ✅ Integrated | All 3 timeframes (15min, 30min, 1h) |
| **Monitoring Script** | ✅ Ready | `track_phase3b.py` --watch/--once/--short |
| **Current Activity** | ⏳ Waiting | 0 reversals (normal, collect 50+ signals first) |
| **Expected Activation** | Mar 4-5 | When reversal signals arrive |
| **Decision Readiness** | Mar 10 | 40-50 reversal samples for evaluation |

---

**Next:** Monitor Phase 2-FIXED signal accumulation. Phase 3B will activate automatically as REVERSAL signals arrive.
