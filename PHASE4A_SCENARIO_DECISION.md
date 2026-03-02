# Phase 4A Scenario Decision Framework

**Status:** Backtest in progress with real KuCoin candle data  
**Decision Date:** Upon backtest completion  
**Implementation:** Immediate upon decision

---

## 🎯 Decision Criteria

Based on backtest_multitf_alignment.py results, use this framework:

### **WINNING SCENARIO = Highest WR + Viability**

```
Viability = (Signals >= 5/day) AND (WR > Baseline)
```

**Baseline Metrics (from Phase 3 clean data):**
- Overall WR: 73.08%
- Signals/day: ~11.2
- P&L: +$811.77 (52 trades)

**Success Targets (Phase 4A):**
- WR > 73% preferred (maintain Phase 3 level)
- WR > 72% acceptable (minor decline ok if signal quality up)
- Signals > 5/day required (minimum viable)

---

## 🔄 Scenario Comparison

### **Scenario 2: 15min + 30min Alignment**
- Filter: 15min signal direction = 30min trend
- Signal reduction: ~60-70% (aggressive filtering)
- Signals/day: ~3-4 (borderline low)
- Expected WR improvement: +2-3%
- **Decision:** If WR > 75%, APPROVE. If signals < 5/day, ⚠️ CAUTION

### **Scenario 3: 15min + 1h Alignment**
- Filter: 15min signal direction = 1h trend
- Signal reduction: ~75-80% (very aggressive)
- Signals/day: ~2-3 (might be too low)
- Expected WR improvement: +4-5%
- **Decision:** If WR > 77%, APPROVE. If signals < 5/day, REJECT (too selective)

### **Scenario 5: Triple Confirmation**
- Filter: 15min = 30min = 1h (all three agree)
- **STATUS:** Likely redundant to Scenario 3 (preliminary)
- **Decision:** Confirm redundancy. If so, SKIP (no added value)

---

## 📊 Decision Tree

```
Run: python3 backtest_multitf_alignment.py

Results:
├─ Scenario 2
│  ├─ Signals >= 5/day AND WR > 73.08%
│  │  └─ ✅ SCENARIO 2 APPROVED
│  │     (Implement 15min + 30min filter in main.py)
│  │
│  └─ Signals < 5/day OR WR <= 73.08%
│     └─ ❌ SCENARIO 2 REJECTED
│
├─ Scenario 3
│  ├─ Signals >= 5/day AND WR > 73.08%
│  │  └─ ✅ SCENARIO 3 APPROVED
│  │     (Implement 15min + 1h filter in main.py)
│  │
│  └─ Signals < 5/day OR WR <= 73.08%
│     └─ ❌ SCENARIO 3 REJECTED
│
└─ Scenario 5
   └─ If identical to S3, skip (no added value)
   └─ If different, evaluate same as S3
```

---

## 🚀 Implementation Plan (Upon Decision)

### **If Scenario 2 Approved:**

**File:** `main.py`  
**Changes:**
1. Add function: `check_multitf_alignment_15_30(symbol, signal_type, tf_main)`
2. Before sending each signal, call function
3. Only send if 15min+30min alignment = TRUE
4. Log tag: `[PHASE4A-S2-FILTERED]` when signal blocked

**Expected Impact:**
- 60-70% fewer signals
- 2-3% WR improvement
- Signals/day: 3-4 (low but acceptable)

### **If Scenario 3 Approved:**

**File:** `main.py`  
**Changes:**
1. Add function: `check_multitf_alignment_15_1h(symbol, signal_type, tf_main)`
2. Before sending each signal, call function
3. Only send if 15min+1h alignment = TRUE
4. Log tag: `[PHASE4A-S3-FILTERED]` when signal blocked

**Expected Impact:**
- 75-80% fewer signals
- 4-5% WR improvement
- Signals/day: 2-3 (very selective)

### **Rollback Plan:**

If results don't meet criteria:
```bash
git checkout main.py
pkill -f "python3 main.py"
sleep 2
nohup python3 main.py > main_daemon.log 2>&1 &
```

---

## 📋 Post-Decision Tasks

1. ✅ Finalize backtest results
2. ✅ Make scenario selection
3. ✅ Code implementation (if applicable)
4. ✅ Deploy Phase 4A to production
5. ✅ Monitor with PHASE3_TRACKER.py (compare to Phase 3 baseline)
6. ✅ Collect 5-7 days of data
7. ✅ Day 7 evaluation: Approve, Refine, or Revert

---

## 📝 Notes

- **Real Data:** Backtest uses real KuCoin OHLCV data (not synthetic)
- **Validation:** Results validated against Phase 3 baseline (73.08% WR)
- **Conservative Approach:** If unsure between S2 and S3, choose S2 (safer filtering)
- **Reversal Window:** All Phase 4 changes are reversible via git

---

**Decision Status:** ⏳ PENDING (awaiting backtest results)  
**Target Decision Time:** 2026-03-03 00:30 GMT+7  
**Deployment Window:** Immediate upon approval

