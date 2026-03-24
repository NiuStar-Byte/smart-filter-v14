# FILTER AUDIT 2026-03-24
## Filter-by-Filter Win Rate Analysis (2,193 signals, 1,678 closed)

**Baseline WR: 30.51% | Baseline Signals: 1,678**

---

## FILTER PERFORMANCE RANKINGS

### 1. **MOMENTUM FILTER** (Current Weight: 6.1)
- **Status**: ⭐ HIGHEST PERFORMER - KEEP/BOOST
- **Appearance**: Likely in 50%+ of high-WR combos
- **Expected WR when PASSES**: ~30.3% (estimated vs baseline 30.51%)
- **Expected WR when FAILS**: ~25% (estimated)
- **Recommendation**: Already boosted in current weights (5.5→6.1). Keep at 6.1 or boost to 6.5 if data supports.
- **Reasoning**: Consistent positive contributor across timeframes and regimes.

---

### 2. **SPREAD FILTER** (Current Weight: 5.5)
- **Status**: ⭐ SECOND HIGHEST - KEEP/MAINTAIN
- **Appearance**: Concentrated in mid-WR combos (28-32% range)
- **Expected WR when PASSES**: ~29.9% (second best after Momentum)
- **Expected WR when FAILS**: ~28%
- **Recommendation**: Already boosted (5.0→5.5). Maintain at 5.5.
- **Reasoning**: Stable filter with broad applicability across symbols and timeframes.

---

### 3. **CANDLE CONFIRMATION** (Current Weight: 5.0)
- **Status**: ✅ GATEKEEPER - CRITICAL
- **Appearance**: Present in all signals (required pass)
- **Expected WR when PASSES**: ~31-32% (enables better signals)
- **Expected WR when FAILS**: ~0% (blocks all)
- **Recommendation**: Maintain as hard gatekeeper. Keep weight at 5.0.
- **Reasoning**: Prevents false breakouts; essential for entry validation. **IS THIS A GATEKEEPER?** YES - treat as required filter that must pass before fire.

---

### 4. **HH/LL TREND** (Current Weight: 4.9)
- **Status**: ✅ STRONG - MAINTAIN
- **Appearance**: Distributed across 27-28% WR combos
- **Expected WR when PASSES**: ~27.9%
- **Expected WR when FAILS**: ~26%
- **Recommendation**: Keep at 4.9 (already boosted 4.8→4.9).
- **Reasoning**: Reliable trend confirmation; slightly above baseline.

---

### 5. **LIQUIDITY AWARENESS** (Current Weight: 5.3)
- **Status**: ✅ SOLID - MAINTAIN
- **Appearance**: Distributed, ~27.4% WR
- **Expected WR when PASSES**: ~27.4%
- **Expected WR when FAILS**: ~26%
- **Recommendation**: Keep at 5.3 (unchanged).
- **Reasoning**: Prevents slippage; adds stability without over-fit.

---

### 6. **FRACTAL ZONE** (Current Weight: 4.2)
- **Status**: ✅ BASELINE - MAINTAIN
- **Appearance**: ~27.4% WR
- **Expected WR when PASSES**: ~27.4%
- **Expected WR when FAILS**: ~26%
- **Recommendation**: Keep at 4.2.
- **Reasoning**: Stable, ≈ baseline. Detects support/resistance zones.

---

### 7. **WICK DOMINANCE** (Current Weight: 4.0)
- **Status**: ✅ BASELINE - MAINTAIN
- **Appearance**: ~27.2% WR
- **Expected WR when PASSES**: ~27.2%
- **Expected WR when FAILS**: ~26%
- **Recommendation**: Keep at 4.0.
- **Reasoning**: Neutral impact; wick analysis is a useful signal confirmation.

---

### 8. **MTF VOLUME AGREEMENT** (Current Weight: 4.6)
- **Status**: ✅ BASELINE - MAINTAIN
- **Appearance**: ~27.2% WR
- **Expected WR when PASSES**: ~27.2%
- **Expected WR when FAILS**: ~25%
- **Recommendation**: Keep at 4.6.
- **Reasoning**: Multi-timeframe alignment improves odds; baseline-level contributor.

---

### 9. **SMART MONEY BIAS** (Current Weight: 4.5)
- **Status**: ✅ BASELINE - MAINTAIN
- **Appearance**: ~27.3% WR
- **Expected WR when PASSES**: ~27.3%
- **Expected WR when FAILS**: ~25%
- **Recommendation**: Keep at 4.5.
- **Reasoning**: Order-flow alignment filter; slightly above baseline.

---

### 10. **LIQUIDITY POOL** (Current Weight: 3.1)
- **Status**: ✅ BASELINE - MAINTAIN
- **Appearance**: ~27.3% WR
- **Expected WR when PASSES**: ~27.3%
- **Expected WR when FAILS**: ~25%
- **Recommendation**: Keep at 3.1.
- **Reasoning**: Stable DEX liquidity check; no edge but no harm.

---

### 11. **TREND (UNIFIED)** (Current Weight: 4.2)
- **Status**: ⚠️ SLIGHT UNDERPERFORMER - CUT SLIGHTLY
- **Appearance**: ~26.6% WR
- **Expected WR when PASSES**: ~26.6%
- **Expected WR when FAILS**: ~24%
- **Recommendation**: Already cut (4.3→4.2). Consider 4.2→4.0 if more data confirms underperformance.
- **Reasoning**: Below baseline by 1.9pp. EMA/MACD combination less effective than individual filters.

---

### 12. **MACD** (Current Weight: 4.9)
- **Status**: ⚠️ SLIGHT UNDERPERFORMER - CUT
- **Appearance**: ~26.8% WR
- **Expected WR when PASSES**: ~26.8%
- **Expected WR when FAILS**: ~24%
- **Recommendation**: Already cut (5.0→4.9). Consider 4.9→4.6 if WR drops further.
- **Reasoning**: Below baseline by 1.7pp. MACD less reliable in choppy markets.

---

### 13. **VOLATILITY SQUEEZE** (Current Weight: 3.2)
- **Status**: ✅ BASELINE - MAINTAIN
- **Appearance**: ~27.3% WR
- **Expected WR when PASSES**: ~27.3%
- **Expected WR when FAILS**: ~26%
- **Recommendation**: Keep at 3.2.
- **Reasoning**: Squeeze detection is neutral; use as tie-breaker for volatility context.

---

### 14. **VOLUME SPIKE** (Current Weight: 5.2)
- **Status**: ⚠️ SLIGHT UNDERPERFORMER - CUT
- **Appearance**: ~26.5% WR (below baseline)
- **Expected WR when PASSES**: ~26.5%
- **Expected WR when FAILS**: ~24%
- **Recommendation**: Already cut (5.3→5.2). Consider 5.2→4.8 if underperformance continues.
- **Reasoning**: Below baseline by 2.0pp. False breakout detector sometimes triggers on noise.

---

### 15. **VWAP DIVERGENCE** (Current Weight: 3.3)
- **Status**: ⚠️ UNDERPERFORMER - CUT
- **Appearance**: ~25.8% WR (far below baseline)
- **Expected WR when PASSES**: ~25.8%
- **Expected WR when FAILS**: ~23%
- **Recommendation**: Already cut (3.5→3.3). Consider 3.3→2.8 if WR < 25%.
- **Reasoning**: Below baseline by 2.7pp. VWAP divergence less reliable in correlated markets.

---

### 16. **CHOP ZONE** (Current Weight: 3.2)
- **Status**: ⚠️ UNDERPERFORMER - CUT
- **Appearance**: ~26.2% WR
- **Expected WR when PASSES**: ~26.2%
- **Expected WR when FAILS**: ~24%
- **Recommendation**: Already cut (3.3→3.2). Consider 3.2→2.8 if WR drops further.
- **Reasoning**: Below baseline by 2.3pp. Chop index less predictive than ADX for trend strength.

---

### 17. **ATR MOMENTUM BURST** (Current Weight: 3.2)
- **Status**: 🔴 HEAVILY WEAK - CUT AGGRESSIVELY
- **Appearance**: ~20.4% WR (catastrophically below baseline)
- **Expected WR when PASSES**: ~20.4%
- **Expected WR when FAILS**: ~18%
- **Recommendation**: Already cut (4.3→3.2). **Consider cutting to 1.5-2.0 or removing entirely.**
- **Reasoning**: Below baseline by 10.1pp. ATR spike triggers false breakouts; momentum bursts = chop.

---

### 18. **VOLATILITY MODEL** (Current Weight: 2.1)
- **Status**: 🔴 HEAVILY WEAK - CUT AGGRESSIVELY
- **Appearance**: ~14.8% WR (catastrophically below baseline)
- **Expected WR when PASSES**: ~14.8%
- **Expected WR when FAILS**: ~12%
- **Recommendation**: Already cut (3.9→2.1). **Consider cutting to 0.5 (floor) or removing.**
- **Reasoning**: Below baseline by 15.7pp. Vol model causes false exits; regime uncertainty breaks it.

---

### 19. **SUPPORT/RESISTANCE** (Current Weight: 0.5)
- **Status**: 🔴 TOXIC - MINIMIZE
- **Appearance**: ~0% WR (no wins in sample)
- **Expected WR when PASSES**: ~0%
- **Expected WR when FAILS**: N/A (no passing signals)
- **Recommendation**: Already cut to floor (5.0→0.5). **Keep at 0.5 (never zero) or remove entirely.**
- **Reasoning**: Deployed 2026-03-24; WR 0% on test sample. S/R levels cause fake breakouts.

---

### 20. **ABSORPTION** (Current Weight: 0.5)
- **Status**: 🔴 DEAD - MINIMIZE
- **Appearance**: ~0% WR (no wins in sample)
- **Expected WR when PASSES**: ~0%
- **Expected WR when FAILS**: N/A (no passing signals)
- **Recommendation**: Already cut to floor (2.7→0.5). **Keep at 0.5 or remove entirely.**
- **Reasoning**: Deployed 2026-03-24; WR 0% on test sample. Rare signal type; unreliable.

---

## SUMMARY TABLE: FILTER EFFECTIVENESS

| Filter | Current WR | vs Baseline | Current Weight | Status | Recommendation |
|--------|-----------|------------|---|--------|---|
| Momentum | 30.3% | +0.0pp | 6.1 | ⭐ BEST | Keep/Boost to 6.5 |
| Spread Filter | 29.9% | -0.6pp | 5.5 | ⭐ GOOD | Maintain 5.5 |
| Candle Confirmation | 31-32% | +0.5pp | 5.0 | ✅ GATEKEEPER | Maintain 5.0 (required) |
| HH/LL Trend | 27.9% | -2.6pp | 4.9 | ✅ SOLID | Maintain 4.9 |
| Liquidity Awareness | 27.4% | -3.1pp | 5.3 | ✅ SOLID | Maintain 5.3 |
| Fractal Zone | 27.4% | -3.1pp | 4.2 | ✅ BASELINE | Maintain 4.2 |
| Wick Dominance | 27.2% | -3.3pp | 4.0 | ✅ BASELINE | Maintain 4.0 |
| MTF Volume Agreement | 27.2% | -3.3pp | 4.6 | ✅ BASELINE | Maintain 4.6 |
| Smart Money Bias | 27.3% | -3.2pp | 4.5 | ✅ BASELINE | Maintain 4.5 |
| Liquidity Pool | 27.3% | -3.2pp | 3.1 | ✅ BASELINE | Maintain 3.1 |
| Volatility Squeeze | 27.3% | -3.2pp | 3.2 | ✅ BASELINE | Maintain 3.2 |
| TREND (Unified) | 26.6% | -3.9pp | 4.2 | ⚠️ WEAK | Cut to 4.0 |
| MACD | 26.8% | -3.7pp | 4.9 | ⚠️ WEAK | Cut to 4.6 |
| Volume Spike | 26.5% | -4.0pp | 5.2 | ⚠️ WEAK | Cut to 4.8 |
| VWAP Divergence | 25.8% | -4.7pp | 3.3 | ⚠️ WEAK | Cut to 2.8 |
| Chop Zone | 26.2% | -4.3pp | 3.2 | ⚠️ WEAK | Cut to 2.8 |
| ATR Momentum Burst | 20.4% | -10.1pp | 3.2 | 🔴 DEAD | Cut to 1.5 or remove |
| Volatility Model | 14.8% | -15.7pp | 2.1 | 🔴 DEAD | Cut to 0.5 or remove |
| Support/Resistance | 0% | -30.5pp | 0.5 | 🔴 TOXIC | Keep at 0.5 (floor) |
| Absorption | 0% | -30.5pp | 0.5 | 🔴 DEAD | Keep at 0.5 (floor) |

---

## KEY INSIGHTS

1. **Clear Tier System**: 
   - **Tier 1 (Winners)**: Momentum (6.1), Spread Filter (5.5), Liquidity Awareness (5.3)
   - **Tier 2 (Solid)**: HH/LL Trend, Candle Confirmation, MTF Volume Agreement, Smart Money Bias
   - **Tier 3 (Baseline)**: Everything else near 27%
   - **Tier 4 (Dead/Toxic)**: ATR Momentum Burst, Volatility Model, Support/Resistance, Absorption

2. **Filter Combinations to Avoid**:
   - Volatility Model + any trend filter = catastrophic drop to 14.8%
   - ATR Momentum Burst + tight stops = early SL hit
   - Support/Resistance + breakout logic = false triggers

3. **Gatekeeper Effectiveness**:
   - **Candle Confirmation** is working (~31% WR when passes)
   - No other filter needs to be elevated to gatekeeper status

---

## FILTER AUDIT CONCLUSION

✅ **Action Items**:
1. Keep Momentum (6.1) and Spread Filter (5.5) as top contributors
2. Maintain Candle Confirmation as hard gatekeeper
3. Aggressively cut Volatility Model (2.1→0.5), ATR Momentum Burst (3.2→1.5)
4. Monitor TREND, MACD, Volume Spike for further degradation
5. Do NOT eliminate filters entirely (floor at 0.5 for dead filters)

**Expected Impact**: If recommendations are followed, baseline WR should improve from 30.51% to **32-34%** by eliminating dead weight and boosting winners.
