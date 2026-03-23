# FILTER AUDIT: 5 Non-Passing Filters

## Overview
- **Total Instrumented Signals:** 655
- **5 Filters with 0 passes:** ATR Momentum Burst, Candle Confirmation, Support/Resistance, Absorption, Volatility Model (1 pass)
- **Question:** Intentional gatekeepers OR broken logic?

---

## FILTER #1: ATR Momentum Burst (0/655 = 0%)

### Logic
```
Requires BOTH:
1. ≥2 momentum bars where:
   - atr_ratio = |pct_move| / ATR > 0.15 (15% of ATR threshold)
   - volume > volume_MA × 1.5 (50% above average)
   
2. Directional consistency:
   - For LONG: bullish_count ≥ 2 AND bullish_count > bearish_count
   - For SHORT: bearish_count ≥ 2 AND bearish_count > bullish_count
```

### Key Parameters
- `threshold_ratio = 0.15` (15% of ATR) - MODERATE
- `volume_mult = 1.5` (50% above avg) - STRICT
- `lookback = 3` (last 3 bars) - MODERATE
- `min_momentum_bars = 2` - MODERATE
- `min_atr = 0.5` - GATE (skips low-volatility symbols)

### Why It Never Passes
**Hypothesis #1: Parameter Too Strict (Volume)**
- Requires volume > 1.5× MA on MULTIPLE bars simultaneously
- Real-world: Volume spikes don't always align with momentum bars
- **Fix Needed:** Loosen `volume_mult` from 1.5 → 1.2 (20% above avg)

**Hypothesis #2: Min ATR Gate Too Strict**
- `min_atr = 0.5` is absolute minimum
- Many altcoins have ATR < 0.5 (price in cents)
- **Fix Needed:** Scale ATR requirement by symbol tier (LOW_ALTS = 0.05, MID = 0.1, TOP = 0.5)

**Hypothesis #3: Looks for Perfect Storm**
- Needs BOTH momentum bars + directional consistency
- Markets rarely have 2+ bars with >15% ATR move + >50% volume spike
- **Action:** Run 100-bar lookback analysis to see if momentum ever occurs

---

## FILTER #2: Candle Confirmation (0/655 = 0%)

### Logic
```
Requires ≥2 of:
1. Bullish/Bearish Engulfing (body > prev_body × 1.05)
2. Pin Bar (wick > 1.5× body)
3. Close > Close_prev (LONG) or Close < Close_prev (SHORT)
4. Volume confirmation (optional)

Plus: long_met > short_met (strict majority)
```

### Key Parameters
- `min_pin_wick_ratio = 1.5` - MODERATE (real pin bars are 1.3-1.5×)
- `require_volume_confirm = False` - GOOD (disabled by default)
- Engulfing body threshold = `prev_body × 1.05` (5% larger) - MODERATE

### Why It Never Passes
**Hypothesis #1: Engulfing Too Strict**
- Requires previous candle to be opposite direction
- Requires current body > prev_body × 1.05
- Real engulfings on 15/30min are rare → 5% body-size match even rarer
- **Fix Needed:** Loosen engulfing to `prev_body × 1.02` (2% threshold) OR enable volume as easy pass

**Hypothesis #2: Pin Bar Definition Too Tight**
- Wick/Body > 1.5 is textbook definition
- But also requires either engulfing OR 2+ other conditions
- Pin bars alone aren't passing
- **Fix Needed:** Make pin bar alone sufficient (`if bullish_pin_bar: return "LONG"`)

**Hypothesis #3: Close Movement Alone Not Enough**
- Even if close > close_prev, still need 1 more condition
- Single-bar up-closes are common but not signals
- **Action:** Check how many bars are close > close_prev AND close < close_prev (should be ~50%)

---

## FILTER #3: Support/Resistance (0/655 = 0%)

### Logic
```
Requires ≥min_cond of:
1. Price within adaptive margin of support/resistance
2. Support/resistance tested ≥min_retest_touches times
3. Volume > MA (absorption at level)
4. Multi-TF confluence (optional, bonus)

Plus: long_met > short_met (strict majority)
```

### Key Parameters
- `window = 20` (rolling high/low) - MODERATE
- `use_atr_margin = True` - DYNAMIC (uses volatility scaling)
- `atr_multiplier = 0.75` - CONSERVATIVE
- `min_retest_touches = 0` - LOOSE ✓
- `min_cond = 1` (code doesn't specify, likely default 2) - **ISSUE**
- `volume_at_level_check = True` - STRICT

### Why It Never Passes
**Hypothesis #1: Min Condition Default Likely Too Strict**
- Code has `min_cond: int = 1` parameter but default application unknown
- If actual default is 2 or 3, very restrictive
- Needs 2+ of: proximity + retest + volume
- **Fix Needed:** Explicitly set `min_cond = 1` for easier pass

**Hypothesis #2: Volume at Level Too Strict**
- Requires `volume > volume_MA` at support/resistance
- But support/resistance are past extremes, volume there = institutional activity
- Current candle volume might not be high
- **Fix Needed:** Remove `require_volume_confirm` OR lower threshold

**Hypothesis #3: ATR Margin Calculation Bug?**
- Formula: `margin = (ATR × 0.75) / support_level`
- For BTC ($60k): margin = (300 × 0.75) / 60000 = 0.375% (good)
- For small alt ($0.50): margin = (0.01 × 0.75) / 0.50 = 1.5% (too wide)
- **Action:** Check if margin calculation is correct for small-cap symbols

---

## FILTER #4: Absorption (0/655 = 0%)

### Logic
```
Requires ≥min_cond of:
1. Price near support (within 2%) OR near high (within 2%)
2. Volume > avg_volume × 1.1 (10% above avg) - LOOSENED
3. Directional pressure ≥ 0.0 (any positive)
4. Placeholder (always True)

Plus: long_met > short_met
```

### Key Parameters
- `window = 25` - MODERATE
- `price_proximity_pct = 0.02` (2%) - MODERATE
- `volume_threshold = 1.1` (10% above avg) - **VERY LOOSE**
- `momentum_threshold = 0.0` (any positive) - **VERY LOOSE**
- `min_cond = 1` - **VERY LOOSE**

### Why It Never Passes (Despite Loose Parameters)
**Hypothesis #1: Directional Pressure Calculation Wrong**
```python
pressure_long = (close_prev - close_now) / close_prev   # Moving DOWN into support
pressure_short = (close_now - close_prev) / close_prev  # Moving UP into resistance
```
- For LONG absorption: need pressure_long ≥ 0.0 (price moving DOWN)
  - BUT also need cond1_long = price_near_low = `close ≤ low × 1.02`
  - So pressure_long & price_near_low both checking DOWN movement
  - **Problem:** Close might not be near low if pressure just started
  
**Hypothesis #2: Proximity Too Strict for Small-Caps**
- `close ≤ low × (1 + 0.02)` = within 2% above 25-bar low
- For PUMP coins (BONK, PEPE), the 25-bar low could be 50% lower
- Current close might not be within 2% of extreme
- **Fix Needed:** Make proximity relative to ATR instead of fixed 2%

**Hypothesis #3: Volume Threshold Wrong**
- Even though 1.1× is very loose, real volume spikes are 2-3×
- Current bar volume might be normal even if it's 10% above MA
- **Fix Needed:** Increase to 1.3× (30% above) for real absorption signal

---

## FILTER #5: Volatility Model (1/655 = 0.2%, 0 wins)

### Logic
```
Requires ALL:
1. ATR > min_atr (0.5 minimum)
2. ATR expansion > 8% above MA ← KEY GATE
3. ≥1 bar in lookback showing expansion
4. ≥2 of [ATR↑, Price↑/↓, Volume↑]
5. ATR direction matches signal (cond_atr_up for LONG, cond_atr_down for SHORT)

LONG: long_conditions ≥ 2 AND cond_atr_up
SHORT: short_conditions ≥ 2 AND cond_atr_down
```

### Key Parameters
- `atr_expansion_pct = 0.08` (8% above MA) - **STRICT GATE**
- `lookback = 2` - MODERATE
- `volume_mult = 1.15` (15% above avg) - MODERATE
- `min_atr = 0.5` - GATE
- `direction_threshold = 2` (of 3 conditions) - MODERATE

### Why Only 1 Pass (0 Wins)
**Hypothesis #1: ATR Expansion Gate Too Strict**
- Requires ATR > MA × 1.08 (8% expansion)
- Real ATR expansions happen but not consistently
- Most bars: ATR ≈ MA (no expansion)
- **Fix Needed:** Lower to 0.05 (5% expansion) OR remove this gate entirely

**Hypothesis #2: Double-Directional Check Too Strict**
- Must have ATR_direction PLUS 2 other conditions
- That's 3 gates total (expansion + direction + conditions)
- **Fix Needed:** Simplify to just expansion + 1 other condition

**Hypothesis #3: Volume Confirmation Not Helping**
- Even with lenient 1.15× volume, still needs ATR expansion
- Volume alone doesn't help if ATR not expanding
- **Action:** Check if 1-passing signal had actual volume spike

---

## Summary Table

| Filter | Passes | Root Issue | Severity | Fix Priority |
|--------|--------|-----------|----------|--------------|
| **ATR Momentum Burst** | 0/655 | Volume × 1.5 too strict + min_atr = 0.5 gates out small caps | HIGH | Scale parameters by symbol tier |
| **Candle Confirmation** | 0/655 | Engulfing × 1.05 too tight; pin bar needs 2nd condition | CRITICAL | Loosen engulfing to 1.02; allow pin bar solo |
| **Support/Resistance** | 0/655 | min_cond likely defaulting to 2-3; volume at level too strict | HIGH | Set min_cond=1 explicitly; remove volume gate |
| **Absorption** | 0/655 | Proximity 2% for 25-bar extreme; pressure calculation wrong | MEDIUM | Use ATR-relative proximity; fix pressure logic |
| **Volatility Model** | 1/655 | ATR expansion 8% gate too strict | HIGH | Lower to 5% OR remove + simplify logic |

---

## Recommended Actions (Order of Priority)

### PHASE 1: Quick Fixes (Tomorrow)
1. **Candle Confirmation:** Change `min_pin_wick_ratio = 1.5` → `1.3` (real pin bars)
2. **Volatility Model:** Change `atr_expansion_pct = 0.08` → `0.05` (5% is more realistic)
3. **ATR Momentum Burst:** Change `volume_mult = 1.5` → `1.2` (volume spikes aren't always perfect)

### PHASE 2: Logic Fixes (This Week)
4. **Support/Resistance:** Explicitly test if `min_cond` is defaulting wrong; add debug logging
5. **Absorption:** Rewrite proximity as `price_near_low = close <= (support + 3*atr)` instead of fixed 2%

### PHASE 3: Deeper Review (Next Week)
6. **All 5 Filters:** Cross-validate against 655 instrumented signals
   - Find examples where filters ALMOST passed
   - Identify the 1 condition blocking each signal
   - Decide: adjust OR discard filter

---

## Data Needed for Full Audit

To run the audit, I need:
1. ✅ Filter definitions (got them)
2. ❓ 655 instrumented signals with detailed per-filter scoring
3. ❓ Sample signals that came closest to passing (e.g., ATR Momentum Burst got 1.8 momentum bars instead of 2)
4. ❓ Historical market data for those 655 signals (to manually backtest fixes)

**Do you want me to:**
- [ ] Start with PHASE 1 quick fixes (change 3 parameters)?
- [ ] Run detailed logging to see why signals almost pass?
- [ ] Build a "filter repair" simulation (test looser params on historical data)?
