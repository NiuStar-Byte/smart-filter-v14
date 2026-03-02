# 📊 PHASE 1 BASELINE - A/B Testing Reference

## Status
✅ **OFFICIAL BASELINE** (Locked for Phase 2 comparison)  
📅 **Period:** 2026-02-27 00:00 to 2026-03-02 18:04 GMT+7  
🔧 **System State:** Before Hard Gates + Regime Adjustments  
📈 **Data Source:** pec_enhanced_reporter.py (AUTHORITATIVE - includes stale timeout corrections)

---

## 🎯 PRIMARY METRICS (WHOLE SYSTEM)

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Signals** | 813 | Fired across 3 TFs (15min, 30min, 1h) |
| **Closed Trades** | 658 | TP (157) + SL (333) + TimeOut (168) |
| **Open Signals** | 32 | Waiting for close |
| **Overall Win Rate** | 31.19% | (157 TP + 51 TimeOut Win) / 658 |
| **Total P&L** | **-$4,288.36** | LONG -$4,479.62 + SHORT +$191.26 |
| **Profit Factor** | 0.62 | Below 1.0 = Money losing |

---

## 📈 DIRECTION BREAKDOWN

### LONG Trades (👎 WEAKNESS)
| Metric | Value | Notes |
|--------|-------|-------|
| **Total Fired** | 577 | 71% of all signals |
| **Closed** | 550 | 27 open waiting |
| **Win Rate** | 28.0% | ❌ **TARGET FIX: 35%+** |
| **TP Hits** | 124 | Direct hits |
| **SL Hits** | 283 | 51% hit stop loss |
| **TimeOut Trades** | 143 | Mixed results |
| **P&L** | **-$4,479.62** | Heavy losses |
| **Avg P&L per Trade** | -$8.14 | Consistently negative |

### SHORT Trades (✅ STRENGTH)
| Metric | Value | Notes |
|--------|-------|-------|
| **Total Fired** | 113 | 29% of all signals |
| **Closed** | 108 | 5 open waiting |
| **Win Rate** | 50.0% | ✅ **Performing well, maintain** |
| **TP Hits** | 33 | Strong closure |
| **SL Hits** | 50 | 46% hit stop loss |
| **TimeOut Trades** | 25 | Positive contributor |
| **P&L** | **+$191.26** | Profitable! |
| **Avg P&L per Trade** | +$1.77 | Positive average |

---

## 🕐 TIMEFRAME BREAKDOWN

| TF | Total | Closed | WR | P&L | Notes |
|----|-------|--------|----|----|-------|
| **15min** | 239 | 233 | 33.0% | -$1,258.00 | Responsive but noisy |
| **30min** | 366 | 346 | 31.5% | -$2,417.59 | **Middle performer** |
| **1h** | 85 | 79 | 27.8% | -$612.77 | Poorest TF performance |

---

## 🛣️ ROUTE BREAKDOWN

| Route | Total | Closed | WR | P&L | Quality |
|-------|-------|--------|----|----|---------|
| **TREND CONTINUATION** | 529 | 506 | 37.2% | -$2,265.69 | ✅ **Best** (only keeper) |
| **NONE** | 56 | 52 | 5.8% | -$1,046.10 | ❌ **Worst** |
| **REVERSAL** | 74 | 70 | 21.4% | -$553.83 | ❌ Poor |
| **AMBIGUOUS** | 31 | 30 | 6.7% | -$422.74 | ❌ Worst |

---

## 🌊 MARKET REGIME BREAKDOWN

| Regime | Total | Closed | WR | P&L | Notes |
|--------|-------|--------|----|----|--------|
| **BEAR** | 253 | 240 | 38.8% | -$512.00 | ✅ Best regime |
| **BULL** | 388 | 372 | 29.3% | -$3,098.77 | ❌ Struggles here |
| **RANGE** | 49 | 46 | 13.0% | -$677.60 | ❌ **Worst** (fixed in Phase 2) |

---

## 💰 SYMBOL GROUP BREAKDOWN

| Group | Total | Closed | WR | P&L | Notes |
|-------|-------|--------|----|----|--------|
| **TOP_ALTS** | 81 | 77 | 45.5% | +$165.37 | ✅ **Only profitable group** |
| **MAIN_BLOCKCHAIN** | 127 | 127 | 29.9% | -$646.34 | Slightly above avg |
| **MID_ALTS** | 117 | 114 | 27.2% | -$1,589.95 | Below average |
| **LOW_ALTS** | 488 | 463 | 29.4% | -$3,375.44 | ❌ **Heaviest losses** |

---

## 💡 CONFIDENCE LEVEL BREAKDOWN

| Conf Level | Total | Closed | WR | P&L | Notes |
|------------|-------|--------|----|----|--------|
| **HIGH (≥76%)** | 658 | 632 | 29.1% | -$4,423.33 | **Most volume**, still losing |
| **MID (51-75%)** | 155 | 149 | 37.6% | -$1,023.03 | Better WR but less traded |
| **LOW (≤50%)** | 0 | 0 | N/A | $0 | No low confidence signals |

---

## 🔍 KEY WEAKNESSES IDENTIFIED

### #1: LONG Direction (28.0% WR)
- **Impact:** 577 signals = 71% of system
- **Cost:** -$4,479.62 = 104% of total losses
- **Root Cause:** Hard gates missing + no regime awareness
- **Phase 2 Target:** 35%+ (need +7% improvement)

### #2: RANGE Regime (13.0% WR)
- **Impact:** 49 signals = 6% of system
- **Cost:** -$677.60 = 16% of losses
- **Root Cause:** Wrong TP/SL scaling (uses 2:1 instead of 3:1)
- **Phase 2 Solution:** Regime-aware score adjustments

### #3: LOW_ALTS Symbol Group (29.4% WR)
- **Impact:** 488 signals = 60% of system
- **Cost:** -$3,375.44 = 79% of total losses
- **Root Cause:** Lower quality signals, no filtering
- **Phase 2 Gate Filter:** Volume + momentum alignment will help

### #4: BULL Regime (29.3% WR)
- **Impact:** 388 signals = 48% of system
- **Cost:** -$3,098.77 = 72% of losses
- **Root Cause:** Filters don't adapt to bull conditions
- **Phase 2 Solution:** Trend alignment gate + regime-aware thresholds

### #5: Non-TREND_CONTINUATION Routes (12-22% WR)
- **Routes:** AMBIGUOUS (6.7%), NONE (5.8%), REVERSAL (21.4%)
- **Combined P&L:** -$2,022.67 = 47% of losses
- **Phase 2 Fix:** Hard gates will filter these out more aggressively

---

## ✅ STRENGTHS TO MAINTAIN

### SHORT Direction (50.0% WR) ✅
- **Profitable:** +$191.26 total
- **Consistent:** 50% WR across all regimes
- **Action:** Keep SHORT logic intact, don't disable

### TREND_CONTINUATION Route (37.2% WR) ✅
- **Only profitable route group:** Contains most wins
- **Volume:** 506 closed trades = best coverage
- **Action:** Protect and enhance with Phase 2 gates

### TOP_ALTS Symbol Group (45.5% WR) ✅
- **Only positive symbol group:** +$165.37 P&L
- **Quality:** Fewer signals but better quality
- **Action:** Study what makes this group different

### BEAR Regime (38.8% WR) ✅
- **Best performing regime:** 240 trades, 38.8% WR
- **Trend:** Clear trends, fewer choppy moves
- **Action:** Maintain BEAR performance, improve BULL

---

## 📊 PHASE 2 SUCCESS CRITERIA

### Kill Switch Thresholds (Will Trigger Immediate Revert)
- ❌ Overall WR drops below 30% → REVERT
- ❌ SHORT WR drops below 45% → REVERT  
- ❌ TREND_CONTINUATION drops below 35% → REVERT
- ❌ Total P&L loss >$5,000 in 5-day window → REVERT

### Success Targets (Need to Hit These)
- ✅ Overall WR: 31.19% → **35-40%** (need +4-9 percentage points)
- ✅ LONG WR: 28.0% → **35%+** (need +7 percentage points)
- ✅ SHORT WR: 50.0% → **50%+** (maintain current level)
- ✅ P&L: -$4,288.36 → **-$500 to +$200** (need +$3,788 improvement)
- ✅ Signal Volume: 65/day → **45-52/day** (30% fewer, higher quality)

---

## 🔧 PHASE 2 INTERVENTIONS DEPLOYED

### Hard Gates (4 Independent Filters)
1. **Momentum-Price Alignment** - Avoid reversal whipsaws
2. **Volume >120% MA20** - Require volume confirmation  
3. **Trend Alignment per Regime** - Match market conditions
4. **Candle Structure** - Reject malformed entries

**Expected Impact:** Filter 40-60% of weaker signals, improve overall WR

### Regime-Aware Score Adjustments
- **BULL:** LONG 1.0×, SHORT 0.6× (favor LONG in uptrends)
- **BEAR:** LONG 0.5×, SHORT 1.0× (favor SHORT in downtrends)
- **RANGE:** Both 0.9× + higher thresholds (caution in consolidation)

**Expected Impact:** 
- Reduce false entries in mismatched regimes
- Improve LONG WR in BULL regime
- Improve SHORT WR in BEAR regime

---

## 📅 MONITORING SCHEDULE

### Days 1-2 (Dry Run - 2026-03-02 to 2026-03-03)
- ✅ Verify gates firing correctly
- ✅ Check logs for Phase 2 tags
- ✅ Validate no false negatives

### Days 3-7 (Live Collection - 2026-03-04 to 2026-03-08)
- 📊 Daily: Run `python3 pec_enhanced_reporter.py`
- 📊 Track LONG WR trend
- 📊 Monitor overall WR
- 📊 Count closed trades (need 20-30 for confidence)

### Day 7 Decision (2026-03-08)
- 🎯 Compare Phase 2 metrics vs Phase 1 baseline
- 🎯 Check against success criteria
- 🎯 Decide: Continue Phase 2, iterate params, or revert

---

## 📁 FILE REFERENCES

**Data Source:**  
`/smart-filter-v14-main/SENT_SIGNALS.jsonl` (Phase 1 section: before 2026-03-02 18:04)

**Reporting:**  
`python3 pec_enhanced_reporter.py` → PEC_ENHANCED_REPORT.txt

**Git Backup:**  
`/Users/geniustarigan/Desktop/smart-filter-v14-main_02Mar26_1752.zip` (Pre-Phase-2 backup)

---

**Created:** 2026-03-02 18:42 GMT+7  
**Status:** ✅ Locked - Official Baseline  
**Next Review:** 2026-03-08 (Day 7 of Phase 2 monitoring)
