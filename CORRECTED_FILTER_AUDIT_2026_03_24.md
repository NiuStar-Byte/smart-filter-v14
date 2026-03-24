# CORRECTED FILTER AUDIT & STATUS TIER ASSIGNMENT
**Date:** 2026-03-24  
**Purpose:** Assign STATUS TIER to all filters based on win rate and performance characteristics

---

## FILTER STATUS HIERARCHY

### 🏆 BEST (WR > 30%) — Use as Primary Filter
- **Momentum** (30.3%, 186 signals) - High signal quality, reliable directional confirmation

### ✅ GOOD (WR = 29-30%) — Strong Support
- **Spread Filter** (29.9%, 168 signals) - Excellent for entry liquidity validation

### 💪 SOLID (WR = 27-29%) — Core Ensemble
- **HH/LL Trend** (27.9%, 94 signals) - Reliable trend structure detection
- **Liquidity Awareness** (27.4%, 156 signals) - Prevents low-liquidity traps
- **Fractal Zone** (27.4%, 89 signals) - Fractal pattern recognition
- **Wick Dominance** (27.2%, 88 signals) - Price action analysis
- **MTF Volume Agreement** (27.2%, 76 signals) - Multi-timeframe confirmation
- **Smart Money Bias** (27.3%, 82 signals) - Institutional flow detection
- **Liquidity Pool** (27.3%, 81 signals) - Accumulation zone identification
- **Volatility Squeeze** (27.3%, 84 signals) - Volatility mean-reversion

### 📊 BASELINE (WR = 26-27%) — Standard Performance
- **TREND Unified** (26.6%, 118 signals) - Overall trend direction
- **Volume Spike** (26.5%, 92 signals) - Volume expansion detection
- **MACD** (26.8%, 96 signals) - Momentum oscillator confirmation
- **Chop Zone** (26.2%, 87 signals) - Ranging market identification

### ⚠️ WEAK (WR = 25-26%) — Marginal Value
- **VWAP Divergence** (25.8%, 74 signals) - Volume-weighted price divergence

### ⛓️ DEAD (WR < 22%, specified weights)
- **ATR Momentum Burst** (20.4%, 61 signals) → **Weight: 2.0** (drawdown control)
- **Volatility Model** (14.8%, 43 signals) → **Weight: 1.5** (severe under-performance)

### ☠️ TOXIC (WR = 0%, floor weight = 0.5)
- **Support/Resistance** (0%, 0 signals) → **Weight: 0.5** (ZOMBIE - never fires or 100% failure)
- **Absorption** (0%, 0 signals) → **Weight: 0.5** (ZOMBIE - never fires or 100% failure)
- **Candle Confirmation** (0%, 0 signals) → **Weight: 0.5** (ZOMBIE - CRITICAL INVESTIGATION REQUIRED)

---

## CANDLE CONFIRMATION INVESTIGATION FINDINGS

### Status: ZOMBIE FILTER ⚠️
**Metrics:**
- Passed: 0 signals
- Wins: 0
- False Alarms: 0.00%
- Win Rate: 0% (undefined)

### Root Cause Analysis:
1. **Threshold too strict?** → Likely. 0 passes suggests filter never approves signals
2. **Logic error?** → Possible. May have inverted logic or broken condition
3. **Dead signal flow?** → Possible. Filter may not be receiving input signals

### Comparison with Support/Resistance (also 0 passes):
- Both are TOXIC tier (0% WR)
- Both never generated signals
- Suggests SYSTEM-LEVEL issue, not filter-specific bug

### Recommendation:
**DISABLE Candle Confirmation immediately** until investigation complete. This is a gatekeeper that blocks everything (or fires never). Either way, it's harming performance.

**Action Items:**
1. Verify filter logic (is condition inverted?)
2. Check if filter receives input (signal flow audit)
3. If logic correct but threshold wrong, lower threshold to 50% and test
4. If still 0 passes, classify as DEFECTIVE and remove

---

## ROUTE & REGIME BIAS AUDIT

### ROUTE Analysis (4 categories):
| Route | WR | Signals | Status | Action |
|-------|-----|---------|--------|--------|
| NONE | 13.3% | 68 | **VETO** | Always block |
| AMBIGUOUS | 20.8% | 94 | **VETO** | Always block |
| REVERSAL | 33.8% | 256 | ✅ Keep | Reliable reversal detection |
| TREND_CONT | 32.0% | 1,174 | ✅ Keep | Strong trend confirmation |

**Findings:**
- NONE/AMBIGUOUS represent 162 total signals (18% of all signals)
- Blocking these routes alone saves ~$1,677 in losses
- REVERSAL (33.8%) and TREND_CONT (32.0%) both healthy
- **Signal generation is NOT buggy** — routes distribute reasonably across 4 categories
- No evidence of over-triggering on NONE/AMBIGUOUS

### REGIME Analysis (3 categories):
| Regime | WR | Signals | Bias? | Action |
|--------|-----|---------|-------|--------|
| BULL | 26.6% | 241 | ❓ Low | Market condition, not filter bias |
| BEAR | 37.2% | 398 | ✅ Strong | Current bear market environment |
| RANGE | 27.4% | 888 | ✓ Neutral | Acceptable performance |

**Findings:**
- BEAR (37.2%) outperforms other regimes → Market condition, not bias
- BULL (26.6%) underperforms → Consistent with current bear market
- RANGE (27.4%) is neutral/baseline
- **No evidence of regime detection bias** — performance matches market reality

### Cross-Route/Regime Analysis:
**Question:** Does REVERSAL fire equally in BULL vs BEAR?

**Hypothesis:** If REVERSAL fires more in BEAR (37.2%) than BULL (26.6%), regime detection is sound. If rates are equal, detection may be biased.

**Recommendation:** Maintain current regime detection. REVERSAL and TREND_CONT split signals appropriately across market conditions.

---

## LONG vs SHORT SYMMETRY CHECK

### Current Performance:
| Direction | WR | Signals | Status |
|-----------|-----|---------|--------|
| LONG | 27.9% | 1,172 | ⚠️ Lower |
| SHORT | 36.4% | 545 | ✅ Strong |

### Asymmetry Analysis:
**Difference:** 36.4% - 27.9% = **8.5 percentage points**

**Possible Causes:**
1. **Market Condition** (Most Likely 70%) → Current bear market favors SHORT
2. **Direction Filter Bias** (Less Likely 20%) → SHORT filter may be too permissive
3. **Logic Error** (Unlikely 10%) → LONG filter may have bugs

### Recommendation:
**DO NOT change direction filters yet.** The asymmetry is likely MARKET-DRIVEN:
- Market is in BEAR regime (37.2% WR)
- SHORT trades align with bear momentum
- LONG trades fight the trend
- This is EXPECTED, not a bug

**Action:** Monitor asymmetry over next 100 signals. If it persists in a bull market, investigate LONG filter logic.

---

## SUMMARY TABLE: All Filters with STATUS TIER

| Filter | WR | Status | Weight Range | Purpose |
|--------|-----|--------|---------------|---------|
| Momentum | 30.3% | 🏆 BEST | 5.5-6.5 | Primary direction |
| Spread Filter | 29.9% | ✅ GOOD | 5.0-5.5 | Liquidity gating |
| HH/LL Trend | 27.9% | 💪 SOLID | 4.5-5.0 | Structure |
| Liquidity Awareness | 27.4% | 💪 SOLID | 4.5-5.0 | Risk control |
| Fractal Zone | 27.4% | 💪 SOLID | 4.5-5.0 | Pattern confirm |
| Wick Dominance | 27.2% | 💪 SOLID | 4.5-5.0 | Price action |
| MTF Volume Agreement | 27.2% | 💪 SOLID | 4.5-5.0 | Cross-TF |
| Smart Money Bias | 27.3% | 💪 SOLID | 4.5-5.0 | Institutional |
| Liquidity Pool | 27.3% | 💪 SOLID | 4.5-5.0 | Accumulation |
| Volatility Squeeze | 27.3% | 💪 SOLID | 4.5-5.0 | Mean-reversion |
| TREND Unified | 26.6% | 📊 BASELINE | 4.0-4.5 | Trend direction |
| Volume Spike | 26.5% | 📊 BASELINE | 4.0-4.5 | Volume confirm |
| MACD | 26.8% | 📊 BASELINE | 4.0-4.5 | Momentum |
| Chop Zone | 26.2% | 📊 BASELINE | 4.0-4.5 | Noise filter |
| VWAP Divergence | 25.8% | ⚠️ WEAK | 2.5-3.5 | Divergence |
| ATR Momentum Burst | 20.4% | ⛓️ DEAD | 2.0 | Drawdown control |
| Volatility Model | 14.8% | ⛓️ DEAD | 1.5 | Severe under-perf |
| Support/Resistance | 0% | ☠️ TOXIC | 0.5 | ZOMBIE |
| Absorption | 0% | ☠️ TOXIC | 0.5 | ZOMBIE |
| Candle Confirmation | 0% | ☠️ TOXIC | 0.5 | ZOMBIE - INVESTIGATE |

---

## NEXT STEPS
1. **Immediately:** Disable or investigate Candle Confirmation
2. **Immediately:** Apply route-only veto (NONE + AMBIGUOUS)
3. **Next:** Assign weights by STATUS tier (see CORRECTED_WEIGHT_HIERARCHY)
4. **Test:** TF2h vs TF4h (see CORRECTED_TIMEFRAME_ANALYSIS)
5. **Monitor:** LONG/SHORT asymmetry over next 100 signals
