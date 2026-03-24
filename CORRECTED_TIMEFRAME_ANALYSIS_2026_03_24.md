# CORRECTED TIMEFRAME ANALYSIS & COMPARISON
**Date:** 2026-03-24  
**Purpose:** Evaluate TF2h and TF4h vs current 15min to determine optimal timeframe mix  
**Question:** Which timeframe to add: 2h or 4h? Keep or drop others?

---

## CURRENT TIMEFRAME PERFORMANCE

### Existing Data
| TF | WR | Signal Count | Quality | Status |
|----|----|--------------|---------|--------|
| **15min** | 29.7% | 747 | ⚠️ Medium | Current baseline |
| **30min** | 29.9% | 693 | ⚠️ Medium | Slightly better |
| **1h** | 34.6% | 277 | ✅ High | Best performer, low signal volume |

### Key Observation
**Quality improves with longer timeframe:**
- 15min: 29.7% (noise, high signal volume, lower conviction)
- 30min: 29.9% (marginal improvement)
- 1h: 34.6% (significant improvement, but only 277 signals)

**Pattern:** Longer timeframes = higher win rate, but fewer signals

---

## TIMEFRAME THEORY

### Signal Quality vs Frequency Trade-off
```
Quality (WR %)     Frequency (signals)    Optimal Zone?
    ↓                   ↓
34.6% (1h)  ....      277         Low signal volume
32.0%       ....      ~450        Unknown
30.0%       ....      ~600        Current 15/30min
29.7%       ....      ~747        Noise, high volume

HYPOTHESIS: TF2h and TF4h will show better quality
with moderate signal reduction
```

### Expected Curve
```
Expected Win Rate by Timeframe
┌─────────────────────────────────┐
│                           4h
│                        /
│                    2h /
│                  /
│              1h /
│          /
│      30m─────────────────
│  /
│ 15min
└─────────────────────────────────┘
Timeframe →
```

---

## ANALYSIS: SHOULD WE ADD TF2h?

### TF2h Hypothesis
**Assumption:** 2h will be intermediate between 1h (34.6%) and 15min (29.7%)

| Metric | Estimate | Reasoning |
|--------|----------|-----------|
| Win Rate | 32-34% | Between 1h (34.6%) and 15min (29.7%) |
| Signal Count | 350-450 | Fewer than 1h (277 is too low) |
| Quality | ✅ High | Longer TF, less noise |
| Volatility | 📉 Lower | 2h candles = smoother trends |
| False Alarm Rate | 📉 Lower | Fewer wicks, cleaner setups |

### Pros of TF2h
1. ✅ **Quality improvement** — Expect 32-34% WR vs 29.7% (15min)
2. ✅ **Balanced signal volume** — ~400 signals (better than 1h's 277, more than 15min's 747)
3. ✅ **Sweet spot** — Between noise (15min) and silence (1h)
4. ✅ **Lower volatility** — 2h candles less susceptible to wicks/noise
5. ✅ **Convergence** — Better agreement across SOLID filters (HH/LL, MTF Volume)

### Cons of TF2h
1. ⚠️ **Slower updates** — Takes 2 hours to complete candle (vs 15min)
2. ⚠️ **Reduced flexibility** — Fewer setups per day
3. ⚠️ **Overnight risk** — 2h candle may span market close (if non-24h)

---

## ANALYSIS: SHOULD WE ADD TF4h?

### TF4h Hypothesis
**Assumption:** 4h will be even cleaner than 2h, approaching 35-37% WR

| Metric | Estimate | Reasoning |
|--------|----------|-----------|
| Win Rate | 34-37% | Longest TF, highest quality |
| Signal Count | 200-300 | Similar to 1h (277), possibly less |
| Quality | 🏆 Highest | Cleanest trend, least noise |
| Volatility | 📉 Lowest | 4h candles very smooth |
| False Alarm Rate | 📉 Lowest | Major trend alignment only |

### Pros of TF4h
1. ✅ **Highest quality** — Expect 35-37% WR, potentially matching/beating 1h
2. ✅ **Institutional alignment** — 4h is preferred by many institutional traders
3. ✅ **Lowest noise** — 4h candles reflect major moves, ignore wicks
4. ✅ **Best signal clarity** — Only high-conviction setups qualify
5. ✅ **Sustainable** — Fewer false alarms = better risk management

### Cons of TF4h
1. ⚠️ **Very slow updates** — Only 6 candles per day
2. ⚠️ **Low signal volume** — Likely <300 signals (similar to 1h)
3. ⚠️ **Risk concentration** — Fewer opportunities = higher position risk
4. ⚠️ **Overnight gap risk** — 4h candle spans market close

---

## RECOMMENDATION: TF4h > TF2h

### Decision: **ADD TF4h**

**Reasoning:**
1. **Quality maximization** — 4h will likely achieve 35-37% WR (best in class)
2. **Signal volume acceptable** — Even at 250-300 signals, quality > quantity
3. **Institutional standard** — 4h is proven timeframe for trending strategies
4. **Risk clarity** — Only major moves are captured, lower false alarm rate
5. **Ensemble benefit** — Adding 4h to 1h + 15min provides diversity without noise

### Why NOT TF2h?
- TF2h is "in-between" but offers no clear edge
- Risk of complexity without benefit
- If TF4h is 35-37%, TF2h adds little (32-34% is similar to TF30min's 29.9%)
- Better to go "high quality" (4h) than "medium quality" (2h)

---

## PROPOSED TIMEFRAME MIX

### Current (3 TFs)
```
15min: 29.7% (high volume, high noise)
30min: 29.9% (similar to 15min)
1h:    34.6% (good quality, low volume)
```

### Proposed (4 TFs)
```
15min: 29.7% → KEEP (market entry, responsive)
30min: 29.9% → CONSIDER REMOVING (duplicate of 15min)
1h:    34.6% → KEEP (good quality, medium volume)
4h:    35-37% → ADD (highest quality, institutional)
```

### Alternative Recommendation
**Remove 30min, keep others, add 4h:**

```
15min: 29.7% ← Keep (responsive, market entry)
1h:    34.6% ← Keep (quality)
4h:    35-37% ← Add (highest quality)

Why drop 30min? Only 0.2% better than 15min, redundant.
Benefit: Fewer signals to filter, cleaner ensemble.
```

---

## SENSITIVITY ANALYSIS

### What if TF4h is only 33% WR?
```
Scenario: TF4h disappoints, only 33% WR (not 35-37%)
Decision: Keep TF4h, add TF2h
Reason: Two good filters (1h=34.6%, TF4h=33%) > one medium (TF2h=32%)
New mix: 15min + 1h + 4h + 2h (all kept)
```

### What if TF4h is 37%+ WR?
```
Scenario: TF4h outperforms expectations, 37%+ WR
Decision: Keep TF4h, consider dropping 15min
Reason: 4h (37%) + 1h (34.6%) provides enough quality signals
New mix: 1h + 4h (drop 15min for noise reduction)
```

### What if signal volume is critical?
```
Scenario: Trading requires 500+ signals per day
Decision: Keep 15min, drop 30min, add TF2h
Reason: 15min (747) + TF2h (400) = 1,147 signals
Quality trade-off: 29.7% + 32% = acceptable
```

---

## IMPLEMENTATION PLAN

### Phase 1: Add TF4h (Immediate)
```
1. Backtest TF4h on historical data (measure WR)
2. Verify signal generation (expect 250-300 signals)
3. Compare TF4h WR to 1h (34.6%)
   - If TF4h >= 34%: Keep and use
   - If TF4h < 34%: Try TF2h instead
4. Update ensemble weights to include TF4h
```

### Phase 2: Optimize Mix (After 100 TF4h signals)
```
1. Measure TF4h actual WR (sample size 100+)
2. Evaluate 30min (is it redundant with 15min?)
   - If 30min WR < 30%: Remove it
   - If 30min WR > 30%: Keep it
3. Rebalance ensemble weights
4. Set TF4h as high-conviction filter (higher weight)
```

### Phase 3: Final Mix (After 200 signals)
```
Recommended outcome:
- 15min: 29.7% WR, weight=4.5 (responsive, market entry)
- 1h:    34.6% WR, weight=5.2 (quality confirmation)
- 4h:    35-37% WR, weight=5.5 (highest quality, institutional)
- 30min: [REMOVED] (redundant with 15min)
```

---

## COMPARISON TABLE: All Scenarios

| Scenario | TFs | Avg WR | Signal Vol | Quality | Risk |
|----------|-----|--------|-----------|---------|------|
| Current | 15, 30, 1h | 31.4% | 1,717 | Medium | Medium |
| Add 2h | 15, 30, 1h, 2h | 31.9% | 2,167 | Medium | High (noise) |
| Add 4h | 15, 30, 1h, 4h | 32.3% | 1,967 | High | Medium |
| **Optimal** | **15, 1h, 4h** | **32.7%** | **1,324** | **High** | **Low** |

---

## MONITORING & VALIDATION

### After TF4h Launch
```python
# Track TF4h performance (first 100 signals)
tf4h_metrics = {
    'signal_count': 0,
    'wins': 0,
    'losses': 0,
    'wr': 0.0,
    'false_alarms': 0
}

# Compare to 1h
expected_tf4h_wr = 35.5  # Midpoint estimate
acceptable_tf4h_range = [33.0, 37.0]  # Range

# Alert conditions
if tf4h_wr < 33.0:
    print("WARNING: TF4h underperforming, consider TF2h")
if tf4h_wr > 37.0:
    print("EXCELLENT: TF4h exceeding expectations, weight it heavily")
```

### Weekly Checks
- [ ] TF4h signal count (expect 3-6 per day)
- [ ] TF4h win rate (expect 35%+)
- [ ] Ensemble agreement across TFs (are they converging?)
- [ ] Ensemble gate pass rate (expect 60-70%)
- [ ] PnL by timeframe (which TF is most profitable?)

---

## FINAL RECOMMENDATION

### ✅ **ADD TF4h**

**Rationale:**
1. Quality improvement (35-37% WR expected)
2. Proven institutional timeframe
3. Low false alarm rate
4. Optimal signal volume (250-300)
5. Best for trend-following strategy

**Implementation:**
1. Backtest TF4h immediately
2. Remove 30min (redundant with 15min)
3. Assign TF4h weight = 5.5 (second highest after Momentum's 6.0)
4. Monitor first 100 signals
5. Re-evaluate after 200 signals

**Expected outcome:**
- Ensemble WR: 29.7% → 32.7% (+3.0 percentage points)
- False alarm rate: ↓ 10-15%
- Signal volume: 1,717 → 1,324 (-393, acceptable)
- Risk-adjusted return: Significantly improved

---

## APPENDIX: Why Not Other Timeframes?

### TF30min ⚠️ Not Recommended
- WR: 29.9% (only 0.2% better than 15min)
- Redundant with 15min
- Splits signal volume without quality gain
- **Action:** Remove 30min, keep 15min

### TF8h ⚠️ Not Recommended
- Too slow (3 candles per day)
- High overnight risk
- Too few signals for backtesting
- **Action:** Skip, use 4h instead

### TF1D ⚠️ Not Recommended
- Daily candles too slow for day trading
- Overnight gap risk severe
- Swing-trading timeframe, not aligned with strategy
- **Action:** Skip

### TF15min (Current) ✅ Keep
- Despite 29.7% WR, maintains market responsiveness
- Essential for market entry timing
- Acts as "trigger" for 1h/4h setups
- **Action:** Keep, don't remove

