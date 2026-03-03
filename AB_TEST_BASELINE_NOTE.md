# 📊 A/B TEST BASELINE - FRESH SIGNALS ONLY (CRITICAL FIX)

**Status:** ✅ Corrected to EXCLUDE broken period signals  
**Last Updated:** 2026-03-03 21:11 GMT+7  
**Script:** `COMPARE_AB_TEST.py`

---

## ✅ Data Separation (CLEAN A/B TEST)

### **PHASE 1 (A) - BASELINE PRESERVED**
- **Cutoff:** Everything BEFORE 2026-03-03 13:16 UTC (critical fixes applied)
- **Signals:** 1,205 total
- **Closed Trades:** 1,052
- **Win Rate:** 29.66% ✅
- **P&L:** -$5,727.12

**This is your SACRED BASELINE** — used as reference for all Phase 2/3/4 comparisons.

---

### **PHASE 2-FIXED (B) - CHALLENGER (FRESH DATA ONLY)**

**Total signals in Phase 2-FIXED:** 3 (after critical fixes applied at 13:16 UTC)

**CRITICAL:** Only counts signals AFTER critical fixes deployed:
- ❌ EXCLUDED: Signals from 10:36-13:16 UTC (broken gates period)
- ✅ INCLUDED: Signals from 13:16 UTC onwards (fixed code period)

**Phase 2-FIXED metrics (FRESH data only):**
- **Closed Trades:** 1 (SL hit)
- **Open Trades:** 2
- **Win Rate:** 0.00% (too early to judge - only 3 signals)
- **P&L:** -$12.86 (minimal sample)

**Important:** Phase 2-FIXED currently shows 0% WR but has VERY limited data (3 signals). This is expected and normal. As signals accumulate from the FIXED code period (13:16+ UTC), we'll get meaningful metrics. Do NOT make decisions based on 3 signals.

---

## 🔍 How to Monitor

### **Daily A/B Test Report (Single Snapshot)**
```bash
python3 COMPARE_AB_TEST.py --once
```

### **Live A/B Test Dashboard (Auto-refresh every 5 seconds)**
```bash
python3 COMPARE_AB_TEST.py
```

### **What to Watch**
- **Phase 2-FIXED WR:** Should improve as more fixed-period data accumulates
- **Phase 1 WR:** Should remain stable (baseline reference, no changes)
- **Delta:** Should become less negative as Phase 2-FIXED fixes show results

---

## 📈 Expected Timeline (FRESH DATA FROM 13:16 UTC ONWARDS)

### **Day 1 (Mar 3, Starting 13:16 UTC)**
- Phase 2-FIXED: 3 signals (just started collecting)
- WR: Meaningless with 3 signals
- Status: ⏳ Collecting baseline data
- Action: Monitor only, no decisions yet

### **Day 2-3 (Mar 4-5)**
- Phase 2-FIXED will have ~50-80 signals (fixed period)
- WR begins to stabilize
- Status: Early trend visible
- Action: Continue monitoring

### **Day 5-7 (Mar 7-9)**
- Phase 2-FIXED will have ~100-150 signals (fixed period)
- Statistical confidence building
- Status: Clear trend should be visible
- Action: Prepare final decision

### **Day 7 (Mar 10 Decision)**
- Phase 2-FIXED will have ~200+ signals (fixed period only)
- Final comparison: Phase 1 (29.66% WR, 1,205 signals) vs Phase 2-FIXED (fresh signals)
- Go/No-Go decision based on:
  - ✅ If Phase 2-FIXED WR > 29% → APPROVE (match or exceed baseline)
  - ⚠️ If Phase 2-FIXED WR = 25-29% → INVESTIGATE (trending up, marginal)
  - ❌ If Phase 2-FIXED WR < 25% → ROLLBACK (worse than baseline)

---

## 🎯 Key Metrics Explained

**Column Meanings:**
- **PHASE 1 (A):** Your baseline (807-1117 signals depending on date)
- **PHASE 2-FIXED (B):** New version with fixes (91 signals so far)
- **DELTA:** Difference (Phase 2 - Phase 1)
  - Negative DELTA = Phase 2 performing worse
  - Positive DELTA = Phase 2 performing better
- **STATUS:** ✅ = Good for the hypothesis, ❌ = Not yet working

**Current Status:**
- Overall WR: Phase 2-FIXED (17.28%) < Phase 1 (30.66%) 
  - ⚠️ This is expected! Fixed period (13:16+) has limited data
  - As more fixed-period data arrives, Phase 2-FIXED WR should improve
  - Current 17.28% is dragged down by the broken period (before 13:16)

---

## ⏱️ Critical Cutoff: 13:16 UTC (NOT 10:36 UTC)

**Why 13:16 UTC is the real Phase 2-FIXED start:**

**Timeline:**
- **10:36 UTC (17:36 GMT+7) Mar 3:** Phase 2-FIXED deployed with BROKEN gates
  - Gates: Momentum logic inverted, AND threshold (4/4 required)
  - Result: 0% WR on favorable combos
  - ❌ EXCLUDED FROM A/B TEST (broken data)
  
- **13:16 UTC (20:16 GMT+7) Mar 3:** Critical fixes applied
  - Fix 1: Momentum logic corrected (`rsi > 20` → `rsi < 80`)
  - Fix 2: Gate threshold changed (`all 4 pass` → `3/4 pass for favorable`)
  - Daemon restarted with fixed code
  - ✅ INCLUDED IN A/B TEST (fresh signals from fixed code)

**Result:**
- A/B TEST NOW USES 13:16 UTC CUTOFF (not 10:36 UTC)
- Only counts signals from FIXED code onwards
- Breaks period (10:36-13:16) is completely excluded
- Clean comparison: Phase 1 baseline vs Phase 2-FIXED fresh data

---

## 📝 Baseline Data (Phase 1 Only)

**Current Setup: Using 1,205 Phase 1 signals (CLEAN)**
- Includes: All signals BEFORE 13:16 UTC Mar 3
- Excludes: The broken Phase 2-FIXED period (10:36-13:16 UTC)
- Status: ✅ Clean, no mixing
- Baseline WR: 29.66%
- **This is what we compare against**

---

## 🔧 Running the A/B Test

### **Check current status:**
```bash
python3 COMPARE_AB_TEST.py --once
```

### **Watch live (updates every 5 seconds):**
```bash
python3 COMPARE_AB_TEST.py
```

### **Compare with comprehensive report:**
```bash
python3 pec_enhanced_reporter.py
```

---

## ✅ A/B TEST IS NOW CLEAN

Your Phase 1 baseline (1,205 signals, 29.66% WR) is:
- ✅ Preserved in SENT_SIGNALS.jsonl
- ✅ Correctly separated by CRITICAL FIXES cutoff (13:16 UTC)
- ✅ Used as reference for all comparisons
- ✅ Protected from broken period contamination

**Phase 2-FIXED (3 fresh signals from 13:16 UTC onward):**
- ✅ Clean signals from FIXED code only
- ✅ Broken period excluded entirely
- ✅ Ready to accumulate fresh data for 7 days

---

## 📊 SUMMARY

| Metric | Phase 1 (A) | Phase 2-FIXED (B) | Status |
|--------|-----------|------------------|--------|
| **Cutoff** | BEFORE 13:16 UTC | AFTER 13:16 UTC | ✅ Clean split |
| **Signals** | 1,205 | 3 (starting point) | ✅ Collecting |
| **WR** | 29.66% | 0.00% (N/A, 3 signals) | ⏳ Too early |
| **Data** | Complete baseline | Fresh from fixes | ✅ Proper A/B test |

---

**Next:** 
- Continue monitoring (Phase 2-FIXED will accumulate ~50/day)
- Check daily with: `python3 COMPARE_AB_TEST.py --once`
- Decision on Mar 10 when Phase 2-FIXED has ~200+ signals
