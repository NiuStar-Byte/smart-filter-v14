# FILTER TUNING RECORD - 2026-03-23

## Timing & Execution

**Start Time:** 2026-03-23 12:22 GMT+7 (Mon)
**Tuning Duration:** ~5 minutes
**Implementation:** Manual parameter edits in `smart_filter.py`
**Git Commit:** `3af2022` (2026-03-23 12:22:15 GMT+7)

---

## Changes Applied

### 1. ATR Momentum Burst Filter
- **Parameter:** `volume_mult`
- **Before:** 1.5 (requires 50% above average volume)
- **After:** 1.2 (requires 20% above average volume)
- **Rationale:** Real momentum spikes don't always achieve 50% volume multiples; 20% is more realistic
- **Expected Impact:** Increase pass rate by loosening volume gate
- **Line Changed:** Function definition line ~2800

### 2. Volatility Model Filter
- **Parameter:** `atr_expansion_pct`
- **Before:** 0.08 (requires 8% ATR expansion above MA)
- **After:** 0.05 (requires 5% ATR expansion above MA)
- **Rationale:** Real market expansions are 5-7%, 8% is upper end of typical range
- **Expected Impact:** Increase pass rate significantly (this was blocking most signals)
- **Line Changed:** Function definition line ~2910

### 3. Candle Confirmation Filter
- **Parameter:** `min_pin_wick_ratio`
- **Before:** 1.5 (wick must be 1.5× body size)
- **After:** 1.3 (wick must be 1.3× body size)
- **Rationale:** Real pin bars in live markets are 1.3-1.5× ratio; 1.3 is acceptable lower bound
- **Expected Impact:** Increase pin bar acceptance, more signals pass
- **Line Changed:** Function definition line ~3250

### 4. Absorption Filter - Proximity Buffer
- **Parameter:** `price_proximity_pct`
- **Before:** 0.02 (price within 2% of 25-bar extreme)
- **After:** 0.03 (price within 3% of 25-bar extreme)
- **Rationale:** For small-cap altcoins with price in cents, 2% buffer is too tight
- **Expected Impact:** Catch absorption signals missed due to tight proximity gate
- **Line Changed:** Function definition line ~2025

### 5. Absorption Filter - Volume Threshold
- **Parameter:** `volume_threshold`
- **Before:** 1.1 (requires 10% above average)
- **After:** 1.05 (requires 5% above average)
- **Rationale:** Genuine absorption doesn't always show massive volume spikes; 5% above avg is meaningful
- **Expected Impact:** More volume confirmations pass threshold
- **Line Changed:** Function definition line ~2026

---

## Testing Plan

**Baseline Metrics (Before Tuning):**
```
Total Instrumented Signals: 655
ATR Momentum Burst Passes: 0 (0%)
Volatility Model Passes: 1 (0.2%, 0 wins)
Candle Confirmation Passes: 0 (0%)
Absorption Passes: 0 (0%)
Support/Resistance Passes: 0 (0%)
```

**Post-Facto Testing Method:**
1. Run filter_effectiveness_analyzer on 655 instrumented signals
2. Compare pass rates before vs after
3. Document any increases in pass rates
4. Check if any filters now show non-zero wins

**Test Command:**
```bash
python3 /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/filter_effectiveness_analyzer_detailed.py
```

---

## Post-Facto Results

### Baseline Snapshot (Pre-Tuning Live Signals)
**Time:** 2026-03-23 12:23 GMT+7
**Source:** Signals fired BEFORE tuning applied at 12:22 GMT+7
**Dataset:** 660 instrumented signals (437 closed)

```
BEFORE TUNING (Historical signals):
Filter                    Passes   Wins    WR       FA         Status
────────────────────────────────────────────────────────────────────
ATR Momentum Burst        0        0       N/A      0.00%      ○ (0 pass)
Candle Confirmation       0        0       N/A      0.00%      ○ (0 pass)
Support/Resistance        0        0       N/A      0.00%      ○ (0 pass)
Absorption                0        0       N/A      0.00%      ○ (0 pass)
Volatility Model          1        0       0.0%     0.23%      · Low
────────────────────────────────────────────────────────────────────
Baseline WR (all):        437      127     29.1%    (baseline)
```

### Live Signal Monitoring (Starting 12:22 GMT+7)

**Strategy:** Track new signals fired AFTER tuning applied. These will use NEW tuned parameters.

**Tracking Period:** 2026-03-23 12:22 GMT+7 onwards (continuous)

**Expected Improvement Window:** 12-24 hours
- Tonight: Initial data collection (4-8 new signals?)
- Tomorrow morning (6am-8am GMT+7): Significant sample size
- Tomorrow afternoon (12pm GMT+7): Full statistical significance

**Success Criteria:**
- ✅ At least 1 pass on ATR Momentum Burst
- ✅ At least 1 pass on Candle Confirmation
- ✅ At least 1 pass on Support/Resistance OR Absorption
- ✅ Volatility Model improves from 1 → 3+ passes

**How to Track:**
```bash
# Run this tomorrow to compare:
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main
python3 filter_effectiveness_analyzer_detailed.py
```

Compare new results against baseline above.

---

## Notes

- All changes are **parameter tuning only** - no logic changes
- All 5 filters remain in place; no removal or architecture change
- Changes are conservative (not excessive loosening)
- If tuning insufficient, next phase will involve logic audit + potential parameter scaling
- Commit message documents full rationale for traceability

---

## Next Steps

1. ✅ Applied tuning (DONE)
2. ⏳ Run effectiveness analyzer to check post-facto results
3. ⏳ If improvement: document results and deploy
4. ⏳ If no improvement: move to Phase 2 (deep logic audit)
5. ⏳ Consider symbol-specific parameter scaling (e.g., min_atr for small-caps)
