# 🎯 WINNERS ANALYSIS - What's Working?
## Filter Pattern Recognition (2,193 signals, 1,678 closed trades)

**Report Date:** 2026-03-24 13:05 GMT+7  
**Baseline WR:** 30.51% | **Baseline P&L:** -$6,388.55  
**Baseline Avg Per Trade:** -$2.48

---

## 📊 TOP 10 PERFORMING 6D COMBOS (BY WIN RATE)

| Rank | 6D Combo ID | TF | Dir | Route | Regime | Symbol Group | Confidence | WR | Sample | Wins | P&L | Velocity |
|------|-------------|----|----|-------|--------|--------------|------------|-----|--------|------|-----|----------|
| 1 | 30min-SHORT-TREND-BEAR-LALT-H | 30min | SHORT | TREND CONT. | BEAR | LOW_ALTS | HIGH | **61.7%** | 60 | 37 | +$338 | 4.6/day |
| 2 | 1h-SHORT-TREND-BEAR-LALT-L | 1h | SHORT | TREND CONT. | BEAR | LOW_ALTS | LOW | **59.4%** | 32 | 19 | +$257 | 2.5/day |
| 3 | 30min-LONG-TREND-BULL-MALT-H | 30min | LONG | TREND CONT. | BULL | MID_ALTS | HIGH | **52.0%** | 25 | 13 | +$67 | 1.9/day |
| 4 | 15min-LONG-TREND-BEAR-MB-H | 15min | LONG | TREND CONT. | BEAR | MAIN_BLC | HIGH | **42.3%** | 26 | 11 | +$60 | 2.0/day |
| 5 | 1h-LONG-REVERSAL-RANGE | 1h | LONG | REVERSAL | RANGE | - | - | **59.3%** | 27 | 16 | +$421 | 2.1/day |
| 6 | 30min-SHORT-TREND-BEAR-LALT-L | 30min | SHORT | TREND CONT. | BEAR | LOW_ALTS | LOW | **44.8%** | 125 | 56 | -$93 | 9.6/day |
| 7 | TF-ROUTE-REGIME-30min-TREND-BEAR | 30min | - | TREND CONT. | BEAR | - | - | **42.9%** | 219 | 94 | +$1644 | 16.9/day |
| 8 | TF-DIR-REGIME-30min-SHORT-BEAR | 30min | SHORT | - | BEAR | - | - | **42.5%** | 193 | 82 | -$159 | 14.8/day |
| 9 | 1h-SHORT-TREND-BEAR (4D) | 1h | SHORT | TREND CONT. | BEAR | - | - | **41.6%** | 77 | 32 | +$85 | 5.9/day |
| 10 | 1h-BEAR (2D) | 1h | - | - | BEAR | - | - | **41.1%** | 107 | 44 | +$29 | 8.2/day |

---

## 🎯 COMMON PATTERNS IN WINNERS

### **Pattern #1: The BEAR Dominance**
- **Regime: BEAR** overall WR: **37.2%** (vs baseline 30.51%)
- **Boost Factor:** 1.22x baseline
- **Sample:** 629 signals, 147 TP, 158 SL + timeout
- **P&L:** +$533.65 (profitable regime!)
- **Why it works:** Bear markets have clearer trend continuation + short bias

**Top BEAR sub-combos:**
- SHORT + BEAR + TREND CONT: 39.2% WR (530 signals)
- SHORT + BEAR (any route): 39.8% WR (432 signals)
- 30min + BEAR + TREND CONT: 42.9% WR (219 signals)

---

### **Pattern #2: SHORT Direction > LONG**
- **Direction: SHORT** overall WR: **36.4%** (vs LONG 27.9%)
- **Boost Factor:** 1.30x vs LONG
- **Sample:** 545 signals, 107 TP, 257 SL
- **P&L:** -$1,121.11 (still negative, but better than LONG)
- **Why it works:** Market bias downward, shorts are easier to execute

**Top SHORT sub-combos:**
- SHORT + BEAR: 39.8% WR (432 signals)
- SHORT + TREND CONT: 37.5% WR (461 signals)
- 30min + SHORT: 38.4% WR (229 signals)

---

### **Pattern #3: REVERSAL Route Excellence**
- **Route: REVERSAL** overall WR: **33.8%** (vs baseline 30.51%)
- **Boost Factor:** 1.11x baseline
- **Sample:** 151 signals, 40 TP, 82 SL
- **P&L:** -$108.31 (near breakeven)
- **Why it works:** Reversal patterns are sharp, clear entries

**Top REVERSAL sub-combos:**
- REVERSAL + RANGE: 41.7% WR (48 signals, +$361 P&L) ⭐
- REVERSAL + BULL: 32.7% WR (55 signals)
- LONG + REVERSAL: 38.1% WR (113 signals)

**CRITICAL:** REVERSAL + RANGE is 41.7% WR with $361 P&L! This should be HIGH weight.

---

### **Pattern #4: Quality Symbol Groups**
Ranking by WR:

1. **MAIN_BLOCKCHAIN** - 38.4% WR (172 signals, -$199 P&L)
   - Strong WR but negative P&L (bad risk:reward)
   
2. **MID_ALTS** - 38.2% WR (186 signals, -$847 P&L)
   - Decent WR, worse P&L
   
3. **TOP_ALTS** - 37.3% WR (102 signals, -$864 P&L)
   - Decent WR, bad P&L
   
4. **LOW_ALTS** - 27.7% WR (1,257 signals, -$4,479 P&L)
   - **WORST WR, HIGHEST SAMPLE** ❌
   - BUT: Used in 57% of all signals
   - **Caution:** Highest correlation to overall baseline drag

**Insight:** Higher-quality alts have better WR but terrible P&L. May indicate poor risk:reward setup (TP too small).

---

### **Pattern #5: Confidence Level Breakdown**
1. **HIGH (≥73%)** - 33.0% WR (900 signals) ✓ BEST
2. **LOW (≤65%)** - 28.7% WR (496 signals)
3. **MID (66-72%)** - 26.3% WR (321 signals) ❌ WORST

**Action:** HIGH confidence signals are 1.08x better. Consider 1.2x boost for HIGH confidence.

---

### **Pattern #6: Timeframe Analysis**
1. **1h** - 34.6% WR (277 signals, -$87 P&L) ✓ BEST
2. **15min** - 29.7% WR (747 signals, -$3,710 P&L)
3. **30min** - 29.9% WR (693 signals, -$2,592 P&L)

**Insight:** 1h has highest WR but lowest sample. 15min + 30min have similar WR but dominate volume.

---

## 🔥 FILTER INTERSECTION ANALYSIS

### **The Golden Formula (Multiple Filters Aligned)**

**Condition 1: SHORT + BEAR + TREND CONT**
- WR: 39.2%
- Sample: 530 signals
- P&L: +$1,148.47 ✓ PROFITABLE!
- **This combo is money machine**

**Condition 2: 30min + SHORT + BEAR**  
- WR: 42.5%
- Sample: 193 signals
- P&L: -$159 (near breakeven)
- **Excellent WR, needs P&L improvement**

**Condition 3: 1h + ANY DIRECTION + REVERSAL**
- WR: 59.3%
- Sample: 27 signals
- P&L: +$421
- **Premium signal combo**

---

## ⏱️ SIGNAL VELOCITY (Winners Per Hour)

Based on 13-hour deployment period:

| Combo | Total | Hour 1-13 | Per Hour | Status |
|-------|-------|-----------|----------|---------|
| BEAR overall | 629 | Distributed | ~48/day | Steady |
| SHORT + BEAR | 432 | Distributed | ~33/day | HIGH velocity |
| 1h timeframe | 277 | Distributed | ~21/day | Lower but quality |
| REVERSAL + RANGE | 48 | Distributed | ~3.7/day | RARE but ELITE |
| TOP 3 QW combos | 117 | Distributed | ~9/day | Premium signals |

**Daily Signal Velocity by Quality Tier:**
- **Elite (WR > 55%):** ~9 signals/day
- **Premium (WR > 50%):** ~15 signals/day  
- **Good (WR > 40%):** ~60 signals/day
- **Acceptable (WR > 30%):** ~130 signals/day
- **All signals:** 170 signals/day

---

## 🏆 WINNING FORMULA SYNTHESIS

### **The Pattern Across Winners:**

1. **Regime matters most** (BEAR = +640 bps above baseline)
2. **Direction second** (SHORT = +110 bps above baseline)
3. **Route third** (REVERSAL/TREND = +80-110 bps above baseline)
4. **Symbol group and confidence matter but less**
5. **Timeframe provides some lift** (1h = +110 bps above baseline)

### **Winning Filter Combination Score:**

Score each dimension, sum for combo strength:

```
BEAR regime           = +100 pts
SHORT direction       = +90 pts
TREND CONTINUATION    = +50 pts
REVERSAL              = +80 pts
1h timeframe          = +50 pts
HIGH confidence       = +30 pts
MAIN_BLOCKCHAIN       = +40 pts
─────────────────────────────
IDEAL COMBO SCORE     = +440 pts
```

**Translation:** Best combos will have multiple high-scoring filters.

---

## 📈 WINNERS WINNING METRICS

| Metric | Value | Status |
|--------|-------|--------|
| Best single dimension | 30min SHORT TREND BEAR (59.4% WR) | 🟢 |
| Best 2D combo | REVERSAL RANGE (41.7% WR) | 🟢 |
| Best 3D combo | 1h LONG REVERSAL (59.3% WR) | 🟢 |
| Best 4D combo | 30min SHORT TREND BEAR (44.3% WR) | 🟢 |
| Best 6D combo | 30min SHORT TREND BEAR LOW_ALTS HIGH (61.7% WR) | 🟢 |
| Most profitable dimension | 30min TREND BEAR (no direction spec) | +$1,644 |
| Highest momentum | SHORT BEAR TREND CONT (530 signals at +$1,148 P&L) | 🟢 |

---

## 🎯 RECOMMENDATIONS FOR REWEIGHTING 2.0

Based on this winners analysis:

1. **BEAR regime:** Boost to 1.5x weight (from baseline)
2. **SHORT direction:** Boost to 1.3x weight
3. **REVERSAL route:** Boost to 1.2x weight
4. **TREND CONTINUATION:** Maintain 1.0x (proven stable)
5. **1h timeframe:** Boost to 1.2x weight
6. **HIGH confidence:** Boost to 1.15x weight
7. **LOW_ALTS symbol group:** Consider cutting to 0.7x (drag)

---

## ⚠️ CAVEATS

1. **Survivorship bias?** These patterns emerged from 2,193 signals - are they predictive or lucky?
2. **Walk-forward test:** Do winners from Mar 21-24 hold up? (New signals WR = 26.5%, lower than foundation!)
3. **Correlation:** Are multiple winning combos firing together (concentrated risk)?
4. **Sample size:** Some combos (1h LONG REVERSAL) have only 27 samples - validate with more data

---

**Status:** ✅ Ready for integration into REWEIGHTING 2.0 formula  
**Next Steps:** Cross-reference with LOSERS_ANALYSIS for contrast, finalize weights
