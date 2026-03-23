# Filter Treatment Proposal: 5 Non-Passing Filters (1-by-1 Review)

**Date:** 2026-03-23 20:28 GMT+7
**Status:** Parameter tuning showed pass-rate improvement but 0% win rate → requires deeper investigation

---

## Current Status (After ~3.5 hours of live testing)

| Filter | Before Tuning | After Tuning | Wins | WR | Assessment |
|--------|---|---|---|---|---|
| ATR Momentum Burst | 0 passes | **3 passes** | 0 | 0.0% | ⚠️ Loose but losing |
| Volatility Model | 1 pass | **30 passes** | 0 | 0.0% | ⚠️ Much looser but losing |
| Candle Confirmation | 0 passes | 0 passes | 0 | N/A | ❌ Still too strict |
| Support/Resistance | 0 passes | 0 passes | 0 | N/A | ❌ Still too strict |
| Absorption | 0 passes | 0 passes | 0 | N/A | ❌ Still too strict |

**Key Finding:** Tuning relaxed some gates (ATR, Volatility) but signals that pass are **losing trades**. This suggests:
1. Gates were too tight (FIXED ✓)
2. But filter logic is selecting LOSING signals (NEW PROBLEM)

---

## Treatment Proposal: Decision Tree

For each filter, we need to decide:

```
Does filter have HIGH pass rate?
├─ YES (ATR=3, Volatility=30)
│   └─ Are signals WINNING?
│       ├─ YES → Keep it, monitor
│       ├─ NO → Logic is inverted/broken (DISABLE or FIX LOGIC)
│       └─ MIXED → Selective tuning needed
│
└─ NO (Candle, Support/Resistance, Absorption = 0)
    └─ Loosen further OR investigate if they should be GATEKEEPERS
```

---

# FILTER-BY-FILTER REVIEW

---

## FILTER #1: ATR Momentum Burst

### Current Status
- **Before:** 0 passes
- **After:** 3 passes
- **Wins:** 0 (100% losing)
- **Assessment:** Tuning worked (3x more passes) but logic is picking LOSING signals

### Root Cause Analysis

**Hypothesis 1: Logic is Reversed**
- Filter detects momentum (volatility + volume spike)
- But high momentum = **reversal risk** (pull-back after big move)
- Maybe it's detecting EXHAUSTION, not ENTRY

**Hypothesis 2: Parameters Still Wrong**
- `volume_mult = 1.2` (was 1.5): Catching too many signals
- `threshold_ratio = 0.15`: Maybe 15% move is too extreme (overextended)
- Signals passing might be at reversal points, not continuations

**Hypothesis 3: Directional Consistency Check Weak**
- Needs ≥2 bullish/bearish bars in last 3
- But 2/3 bars same direction ≠ strong signal
- Should require 3/3 or NONE of them?

### Treatment Proposal: Choose One

**Option A: DISABLE (Recommended)**
- Filter is detecting exhaustion, not opportunity
- 0 wins = confirmed losing signals
- Cost of keeping: -1-2% WR drag
- Recommendation: **DISABLE** (don't remove code, just exclude from scoring)

**Option B: INVERT LOGIC (Risky)**
- Use as contrarian indicator (reversal detector)
- But would need full rewrite and testing
- High risk, uncertain payoff

**Option C: REVERT PARAMETERS (Unlikely to help)**
- Tighten again (volume_mult 1.2 → 1.5)
- But we already know 0 passes was bad
- More tuning likely won't fix losing-signal problem

### **RECOMMENDATION: DISABLE**
- Status: Exclude from filter scoring
- Keep code: In case future value found
- Rationale: 0 wins = confirmed ineffective

---

## FILTER #2: Volatility Model

### Current Status
- **Before:** 1 pass
- **After:** 30 passes (30x improvement!)
- **Wins:** 0 (100% losing)
- **Assessment:** Tuning worked spectacularly for pass rate but signals are LOSING

### Root Cause Analysis

**Hypothesis 1: Expansion != Entry Signal**
- Filter detects ATR expansion (>5% above MA)
- But ATR expansion can mean:
  - ✅ Breakout starting (good)
  - ❌ Reversal coming (bad)
  - ❌ News/gap (uncertain)
- Just because volatility expanded doesn't mean trade direction is good

**Hypothesis 2: Threshold Still Too Loose**
- `atr_expansion_pct = 0.05` (was 0.08): Very loose gate
- Catching almost every expansion, including false breakouts
- 30 passes = too many, quality degraded

**Hypothesis 3: Missing Confirmation**
- Filter checks ATR + 2 of 3 conditions (price, volume, direction)
- But without trend confirmation, just volatility is noise
- Should require TREND ALIGNMENT (is price moving WITH volatility?)

### Treatment Proposal: Choose One

**Option A: TIGHTEN BACK UP (Likely)**
- Increase `atr_expansion_pct = 0.05 → 0.07` (middle ground)
- Or add TREND requirement (price must also show momentum)
- Target: Reduce 30 passes → 5-10 with better quality

**Option B: ADD TREND FILTER (Better)**
- Keep `atr_expansion_pct = 0.05` (loose)
- BUT add check: "Is price moving in expected direction?"
- This filters out whipsaws
- Example: ATR↑ but price going opposite way = REJECT

**Option C: DISABLE (Conservative)**
- Same as ATR: 0 wins = ineffective
- Cost: -0.5% WR (small, only 30 signals)
- Simple solution

### **RECOMMENDATION: OPTION B (Add Trend Filter)**
- Keep loose `atr_expansion_pct` (helps with gate)
- Add: "Price must move >0.5% in expected direction when ATR expands"
- Expected result: 30 → 10-15 passes with better WR
- Rationale: Volatility is entry opportunity, but needs price confirmation

---

## FILTER #3: Candle Confirmation

### Current Status
- **Before:** 0 passes
- **After:** 0 passes (no change)
- **Wins:** 0 (no data)
- **Assessment:** Tuning did NOT help (still too strict)

### Root Cause Analysis

**Hypothesis 1: Definition Conflict**
- Requires ≥2 of 4 conditions (engulfing, pin bar, close movement, volume)
- BUT these are mutually exclusive often:
  - Engulfing = large body move
  - Pin bar = small body + big wick
  - Close movement = simple direction
  - These don't usually happen together

**Hypothesis 2: Previous Candle Requirement**
- Engulfing needs opposite direction previous candle
- But in trending markets, you rarely get reversals before continuing
- This is a COUNTER-TREND filter (good for reversals, bad for continuations)

**Hypothesis 3: Tight Validation**
- `min_pin_wick_ratio = 1.3` (tuned from 1.5)
- Engulfing `body > prev_body × 1.05`
- Even loosened, still too restrictive when combined

### Treatment Proposal: Choose One

**Option A: SIMPLIFY (Recommended)**
- Remove the "2 of 4" logic
- Instead: Check EACH condition separately, return if ANY is TRUE
- Pseudo-code: `if bullish_engulfing OR bullish_pin_bar: return "LONG"`
- Result: Much simpler, more signals pass

**Option B: SEPARATE INTO TWO (Complex)**
- Create "Engulfing Reversal" filter (rare, high quality)
- Create "Pin Bar Entry" filter (more common)
- Use each separately in scoring
- But adds complexity

**Option C: DISABLE (Conservative)**
- 0 passes = can't test effectiveness yet
- Tuning didn't help
- Cost: -0% WR (no data to lose)
- Can reintroduce later if simplified

### **RECOMMENDATION: OPTION A (Simplify)**
- Change logic: ANY condition passes (not 2 of 4)
- Loosen pin bar: 1.3 → 1.2
- Loosen engulfing: 1.05 → 1.02
- Expected result: 0 → 10-20 passes
- Then test for effectiveness

---

## FILTER #4: Support/Resistance

### Current Status
- **Before:** 0 passes
- **After:** 0 passes (no change)
- **Wins:** 0 (no data)
- **Assessment:** Tuning did NOT help (still too strict)

### Root Cause Analysis

**Hypothesis 1: Rolling Extremes Are Poor S/R**
- Window=20: Looking at last 20 bars for high/low
- On 15min = 300 minutes = 5 hours
- These aren't institutional S/R levels, they're just recent extremes
- Real S/R comes from longer-term (daily, weekly, monthly)

**Hypothesis 2: Proximity Gate Too Strict**
- `min_retest_touches = 0`: Loose (good)
- But ATR-based margin may be too tight
- `support_margin = (atr × 0.75) / support_level`
- For small-cap altcoins (price = $0.05), this is VERY small buffer

**Hypothesis 3: Volume Requirement Impossible**
- Requires `volume > MA` at support level
- But historical volume at S/R doesn't match current volume
- This is unfair gate

### Treatment Proposal: Choose One

**Option A: INCREASE WINDOW (Better Foundation)**
- Change `window = 20 → 100` (get longer-term levels)
- Or calculate S/R from higher timeframes
- Example: Get 1h S/R, apply to 15min
- Result: More institutional-grade levels

**Option B: USE EXTERNAL S/R (Best)**
- Don't calculate rolling extremes
- Use fixed S/R from technical analysis (round numbers: 10, 50, 100, etc.)
- Or use Multi-TF confluence (5 min + 15 min + 30 min agree on level)
- Result: Higher quality, fewer but better signals

**Option C: DISABLE (Conservative)**
- 0 passes = can't measure effectiveness
- Tuning didn't help
- Likely a gatekeeper (meant to block most signals)
- Cost: 0% (no signals passing anyway)

### **RECOMMENDATION: DISABLE**
- Current implementation is too sophisticated for live data
- Needs redesign to use proper S/R (external or multi-TF)
- Not a quick fix
- Can reintroduce after redesign

---

## FILTER #5: Absorption

### Current Status
- **Before:** 0 passes
- **After:** 0 passes (no change)
- **Wins:** 0 (no data)
- **Assessment:** Tuning did NOT help (still too strict)

### Root Cause Analysis

**Hypothesis 1: Proximity Check Misses Real Absorption**
- `price_proximity_pct = 0.03` (tuned from 0.02)
- Requires price within 3% of 25-bar extreme
- But real absorption happens at support/resistance, not recent extremes
- These are different concepts

**Hypothesis 2: Volume Threshold Wrong**
- `volume_threshold = 1.05` (tuned from 1.1)
- Requires only 5% above MA for "absorption"
- But true absorption needs 2-3x volume (not 1.05x)
- Too loose threshold = catching noise

**Hypothesis 3: Directional Pressure Wrong**
- `pressure_long = (close_prev - close_now) / close_prev`
- Requires movement down into support
- But what if there's NO movement? (consolidation before move)
- This might reject valid absorption signals

### Treatment Proposal: Choose One

**Option A: REDEFINE ABSORPTION (Complex)**
- Real absorption = price trades at level + volume spike + reversal
- Current filter doesn't check for reversal (no TP/SL needed)
- Needs to detect: price at level + volume up + next bar reverses
- Full rewrite needed

**Option B: INCREASE THRESHOLDS (Risky)**
- Raise `volume_threshold = 1.05 → 1.3` (back to reasonable)
- Keep `proximity_pct = 0.03`
- This defeats the tuning purpose (was trying to loosen)
- Unlikely to help

**Option C: DISABLE (Recommended)**
- 0 passes = absorption pattern not found in current market
- Could be rare pattern or poor definition
- Cost: 0% (no signals anyway)
- Can study absorption separately later

### **RECOMMENDATION: DISABLE**
- Absorption is a rare pattern
- Current definition not matching market behavior
- Tuning didn't help = logic may be wrong
- Can research and redesign later

---

## Summary: Treatment Matrix

| Filter | Current | Tuning Result | Effectiveness | Treatment |
|--------|---------|---|---|---|
| ATR Momentum Burst | 0→3 | ✓ Pass rate up | ❌ 0 wins | **DISABLE** |
| Volatility Model | 1→30 | ✓✓ Pass rate up 30x | ❌ 0 wins | **ADD TREND FILTER** |
| Candle Confirmation | 0→0 | ❌ No change | ❌ No data | **SIMPLIFY & RETRY** |
| Support/Resistance | 0→0 | ❌ No change | ❌ No data | **DISABLE** |
| Absorption | 0→0 | ❌ No change | ❌ No data | **DISABLE** |

---

## Action Plan (Prioritized)

### Immediate (Today)
1. **DISABLE 3 filters** (ATR, Support/Resistance, Absorption)
   - Keep code, exclude from scoring
   - Commit as "Disable ineffective filters"
   
2. **APPLY TREND FILTER to Volatility** (5 min implementation)
   - Add: "Price must move >0.5% in signal direction"
   - Test: Should reduce 30 → 10-15 with better WR
   - Commit as "Add trend confirmation to volatility"

3. **SIMPLIFY Candle Confirmation** (10 min implementation)
   - Change "2 of 4" → "ANY condition passes"
   - Loosen thresholds (1.3 → 1.2, 1.05 → 1.02)
   - Test: Should enable signals to pass
   - Commit as "Simplify candle confirmation logic"

### Monitor (Next 24 hours)
- Volatility Model with trend filter: Does WR improve?
- Candle Confirmation simplified: Any passes with good WR?
- ATR/Support/Resistance/Absorption disabled: Any impact on overall WR?

### Later (This Week)
- Full redesign of Support/Resistance (multi-TF approach)
- Research absorption pattern separately
- Keep ATR disabled unless new hypothesis found

---

## Expected Impact on System

**If all changes applied:**
- Remove 3 ineffective filters (ATR, S/R, Absorption)
- Enhance Volatility with trend confirmation
- Simplify Candle Confirmation
- Expected: Better WR, fewer false signals, cleaner filter suite

**Estimated WR impact:**
- Current baseline: ~32.7% WR (FOUNDATION)
- Remove -0% from disabling (these 3 had 0 wins anyway)
- Potential +1-2% from volatility trend filter
- Potential +2-3% from cleaner candle logic
- Net: Slight improvement if changes work as expected

---

## Recommendation Summary

| Filter | Treatment | Confidence | Complexity |
|--------|-----------|---|---|
| **ATR Momentum Burst** | **DISABLE** | 95% | Low |
| **Volatility Model** | **Add Trend Filter** | 80% | Low-Medium |
| **Candle Confirmation** | **Simplify** | 70% | Low |
| **Support/Resistance** | **DISABLE** | 90% | Low |
| **Absorption** | **DISABLE** | 90% | Low |

**Overall:** 3 definite DISABLES, 1 enhancement, 1 simplification. Low risk, clear next steps.

Shall we proceed with these changes?
