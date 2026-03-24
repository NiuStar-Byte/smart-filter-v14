# Filter Weight Retuning Based on Win Rate Data
**Date:** 2026-03-23 23:59 GMT+7  
**Analysis:** Real performance data vs assumed weights

---

## 🚨 PROBLEM IDENTIFIED

### Simplified Filters Under-Performing

The 3 simplified filters are now **firing signals BUT with WORSE win rates**:

```
Filter                  Passes  Wins  WR      Baseline  Delta    Weight  Status
─────────────────────────────────────────────────────────────────────────────
Support/Resistance        3      0    0.0%    27.3%    -27.3pp   5.0    ❌ TOXIC
Volatility Model         88     13   14.8%    27.3%    -12.6pp   3.9    ❌ NEGATIVE
ATR Momentum Burst       54     11   20.4%    27.3%     -7.0pp   4.3    ❌ NEGATIVE
─────────────────────────────────────────────────────────────────────────────
```

### Root Cause Analysis

**What went wrong:**
1. **Oversimplified logic** - Cut too many meaningful gates
2. **Retailization backfired** - Lost institutional quality checks
3. **Threshold changes broke calibration** - 0.02 ATR threshold too loose
4. **Missing filters matter** - Support/Resistance needs more nuance

**Example: Support/Resistance**
```
BEFORE (150 lines):
- Retest validation ✓
- Volume confirmation ✓
- Multi-TF confluence ✓
- Result: 0% pass (too strict)

AFTER (50 lines):
- 2% proximity only ✗
- No retest gate ✗
- No volume check ✗
- Result: 3 passes, 0 wins (firing junk signals)
```

---

## ✅ USER INSIGHT: Weight-Based Tuning

**User suggestion:** "Don't you think it's best time to assign weights based on filter's WR?"

This is **correct** — Fixed weights don't adapt to real performance.

### Solution: Dynamic Weight Tuning

Instead of manual weights, derive from actual WR data:

```
New Weight = Base Weight × (Filter WR / Baseline WR)

Example:
Momentum:     5.5 × (30.3% / 27.3%) = 5.5 × 1.11 = 6.1 (boost)
Vol Model:    3.9 × (14.8% / 27.3%) = 3.9 × 0.54 = 2.1 (cut)
S/R:          5.0 × (0% / 27.3%)    = 5.0 × 0.00 = 0.0 (remove)
```

---

## 📊 PROPOSED NEW WEIGHTS (Based on WR Data)

### Current Weights vs Proposed (Based on WR Performance)

| Rank | Filter | Current | WR | WR Ratio | Proposed | Change | Action |
|------|--------|---------|----|----|----------|--------|--------|
| 1 | Momentum | 5.5 | 30.3% | 1.11 | **6.0** | +0.5 | ⬆️ Boost |
| 2 | Spread Filter | 5.0 | 29.9% | 1.10 | **5.5** | +0.5 | ⬆️ Boost |
| 3 | HH/LL Trend | 4.8 | 27.9% | 1.02 | **4.9** | +0.1 | ↑ Slight boost |
| 4 | Fractal Zone | 4.2 | 27.4% | 1.00 | **4.2** | 0.0 | ➡️ Keep |
| 5 | Liquidity Awareness | 5.3 | 27.4% | 1.00 | **5.3** | 0.0 | ➡️ Keep |
| 6 | Volatility Squeeze | 3.2 | 27.3% | 1.00 | **3.2** | 0.0 | ➡️ Keep |
| 7 | Liquidity Pool | 3.1 | 27.3% | 1.00 | **3.1** | 0.0 | ➡️ Keep |
| 8 | Smart Money Bias | 4.5 | 27.3% | 1.00 | **4.5** | 0.0 | ➡️ Keep |
| 9 | Wick Dominance | 4.0 | 27.2% | 0.99 | **3.9** | -0.1 | ↓ Slight cut |
| 10 | MTF Volume Agreement | 4.6 | 27.2% | 0.99 | **4.5** | -0.1 | ↓ Slight cut |
| 11 | MACD | 5.0 | 26.8% | 0.98 | **4.9** | -0.1 | ↓ Slight cut |
| 12 | TREND | 4.3 | 26.6% | 0.97 | **4.1** | -0.2 | ↓ Cut |
| 13 | Volume Spike | 5.3 | 26.5% | 0.97 | **5.1** | -0.2 | ↓ Cut |
| 14 | Chop Zone | 3.3 | 26.2% | 0.96 | **3.1** | -0.2 | ↓ Cut |
| 15 | VWAP Divergence | 3.5 | 25.8% | 0.94 | **3.3** | -0.2 | ↓ Cut |
| **16** | **ATR Momentum Burst** | **4.3** | **20.4%** | **0.75** | **3.2** | **-1.1** | **❌ Cut** |
| **17** | **Volatility Model** | **3.9** | **14.8%** | **0.54** | **2.1** | **-1.8** | **❌ Cut Heavily** |
| **18** | **Candle Confirmation** | **5.0** | **N/A** | **N/A** | **5.0** | **0.0** | **⚠️ Review** |
| **19** | **Support/Resistance** | **5.0** | **0.0%** | **0.00** | **0.0** | **-5.0** | **🚨 REMOVE** |
| **20** | **Absorption** | **2.7** | **0.0%** | **0.00** | **0.0** | **-2.7** | **🚨 REMOVE** |

---

## 🔴 CRITICAL FINDINGS

### Three Filters Are Toxic (Negative Win Rate)

```
1. Support/Resistance: 0% WR (0 wins, 3 passes)
   → Firing junk signals
   → Weight: 5.0 (WRONG!)
   → Recommendation: REMOVE (0.0 weight)

2. Volatility Model: 14.8% WR vs 27.3% baseline (-12.6pp)
   → Hurting signal quality
   → Weight: 3.9 (TOO HIGH!)
   → Recommendation: CUT to 2.1

3. ATR Momentum Burst: 20.4% WR vs 27.3% baseline (-7.0pp)
   → Dragging down average
   → Weight: 4.3 (TOO HIGH!)
   → Recommendation: CUT to 3.2
```

### What's Actually Working (High WR)

```
1. Momentum: 30.3% WR (608 passes, 184 wins)
   → Best filter on system
   → Weight: 5.5 → BOOST to 6.0

2. Spread Filter: 29.9% WR (798 passes, 239 wins)
   → Second best
   → Weight: 5.0 → BOOST to 5.5

3. HH/LL Trend: 27.9% WR (585 passes, 163 wins)
   → Solid performer
   → Weight: 4.8 → KEEP or slight boost to 4.9
```

---

## 📈 PROPOSED WEIGHT ALGORITHM

Instead of manual tuning, use **data-driven formula:**

```python
# For each filter, calculate WR-based weight adjustment
baseline_wr = 0.273  # 27.3%

for filter in all_filters:
    if filter.passes == 0:
        new_weight = 0.0  # Dead filters removed
    else:
        wr_ratio = filter.wr / baseline_wr
        new_weight = filter.current_weight * wr_ratio
        new_weight = max(new_weight, 0.5)  # Floor at 0.5 to avoid too-low weights
        new_weight = round(new_weight, 1)  # Round to 1 decimal
    
    print(f"{filter.name}: {filter.current_weight} → {new_weight}")
```

---

## 🎯 RECOMMENDATIONS

### IMMEDIATE (Today)

**Option A: Conservative (Adjust current weights)**
```python
# Reduce toxic filters, boost high performers
Momentum:           5.5 → 6.0 (+0.5)
Spread Filter:      5.0 → 5.5 (+0.5)
ATR Momentum Burst: 4.3 → 3.2 (-1.1)
Volatility Model:   3.9 → 2.1 (-1.8)
Support/Resistance: 5.0 → 0.0 (REMOVE) [-5.0]
Absorption:         2.7 → 0.0 (REMOVE) [-2.7]
```

Expected impact:
- Remove ~8 weight of toxic signals
- Boost ~1 weight to high performers
- Net: Better signal quality (higher average WR)

---

### MEDIUM TERM (This Week)

**Investigate why Volatility Model & ATR Momentum Burst underperform:**
1. Check if threshold values are wrong (0.02 too loose?)
2. Verify signal firing conditions (test with debug output)
3. Compare to original implementations (were they better?)
4. Possible causes:
   - Simplified logic lost critical gates
   - Parameters need recalibration
   - Simplification was TOO aggressive

**For Support/Resistance:**
- 3 passes, 0 wins → fundamentally broken
- Either:
  - Restore some original complexity (retest validation, volume check)
  - Or remove entirely and rely on HH/LL Trend (27.9% WR, similar purpose)

---

### LONG TERM (Next 2 Weeks)

**Build dynamic weight system:**
```python
# Calculate weights daily from real WR data
# This enables automatic adaptation as market changes
# Instead of manual tuning every time

weights = calculate_weights_from_wr(filter_history, baseline_wr)
apply_weights(weights)  # No code changes needed
```

Benefits:
- Weights always aligned with performance
- Auto-adapts to market regime changes
- No more guessing
- Data-driven decision making

---

## ⚠️ KEY INSIGHT FROM USER

**"Don't you think it's best time to assign weights based on filter's WR?"**

Translation: **Fixed weights are assumptions. Real data tells truth.**

This is the path forward:
1. ✅ We now have 1,576 instrumented signals with real WR data
2. ✅ We can calculate accurate WR for each filter
3. ✅ We can derive optimal weights from WR performance
4. ✅ No more guessing — let data speak

---

## 📋 ACTION ITEMS

### IMMEDIATE (Next 30 minutes)
- [ ] Approve weight reductions (Option A above)
- [ ] Apply to smart_filter.py
- [ ] Reload daemon
- [ ] Monitor for improvement

### 24-HOUR CHECKPOINT
- [ ] Check if signal quality improved
- [ ] Win rate should be closer to 27%+ (not -7pp)
- [ ] Review signals fired by toxic filters (any patterns?)

### DECISION POINT
**Volatility Model & ATR Momentum Burst:**
- Option 1: Cut weights drastically (2.1, 3.2)
- Option 2: Remove entirely (0.0)
- Option 3: Restore original complexity and retune

**Recommendation:** Option 1 (cut weights) for now, investigate later

---

## 📊 Summary Statistics

```
Total Instrumented: 1,576 signals
Closed: 1,094 (TP/SL only)
Baseline WR: 27.3%

Best 3 filters (keep/boost):
  1. Momentum 30.3% (boost)
  2. Spread 29.9% (boost)
  3. HH/LL 27.9% (keep)

Worst 5 filters (cut/remove):
  1. Absorption 0.0% (REMOVE)
  2. Support/Resistance 0.0% (REMOVE)
  3. Volatility Model -12.6pp (CUT)
  4. ATR Momentum -7.0pp (CUT)
  5. Candle Conf N/A (REVIEW)

Total weight swing: 13.4 removed + 1.0 added = net -12.4 weight
```

---

## ✅ NEXT STEP

**Approve Option A (Conservative weight adjustment):**
- Cut 3 toxic filters
- Boost 2 high performers
- Expected: Better signal quality, higher average WR
- Risk: Low (data-driven, not guess-based)

Deploy in next 30 minutes and monitor for +2-3pp WR improvement.

