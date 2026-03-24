# Filter Simplification Proposal
**Date:** 2026-03-23 21:05 GMT+7  
**Status:** PROPOSAL - AWAITING APPROVAL BEFORE IMPLEMENTATION

---

## PROBLEM STATEMENT

### 1. Gatekeeper Structure Broken
Currently in smart_filter.py:
```python
self.gatekeepers_long = ["Candle Confirmation"]
self.gatekeepers_short = ["Candle Confirmation"]
self.soft_gatekeepers = ["Candle Confirmation"]
```

**Issue:** Candle Confirmation is listed as BOTH hard gatekeeper AND soft gatekeeper
- **Result:** In `get_signal_direction()`, it fails to block signals when it should
- **Logic:** If filter is soft, it can't veto; if it's hard, signal dies when it fails
- **Current behavior:** It passes, but doesn't gate anything (both AND conditions true = pass)

### 2. Three Filters Dead (0 Passes in 295 Signals)
- **Support/Resistance** (weight 5.0): Designed for institutional confluence, threshold too tight
- **Volatility Model** (weight 3.9): Requires 5% ATR expansion, unrealistic
- **ATR Momentum Burst** (weight 4.3): Requires 0.15 ATR ratio + 1.2x volume simultaneously, impossible

**Root cause:** All three are "institutional-grade" checks. They work for hedge fund detection, not for retail trading signals.

### 3. No Real Gatekeeper
- Support/Resistance was supposed to be gatekeeper but moved to filter-level
- Candle Confirmation can't block (it's soft)
- Result: **Any signal with min_score=12 passes**, even if quality is bad

---

## PROPOSAL: Simplified Filter Suite

### PART A: Fix Gatekeeper Structure

**Status Quo:**
```
Hard Gatekeepers: [Candle Confirmation]  ← Listed as hard but treated as soft
Soft Gatekeepers: [Candle Confirmation]  ← Redundant, contradictory
```

**Proposal:**
```
Hard Gatekeepers (MUST PASS): [Candle Confirmation]
Soft Gatekeepers: []  ← Remove this redundancy
```

**Implementation:**
- Remove "Candle Confirmation" from `soft_gatekeepers` list
- Keep it ONLY in `gatekeepers_long` and `gatekeepers_short`
- This makes it a true hard gatekeeper (must pass to fire signal)

---

### PART B: Simplify 4 Filters

#### **FILTER #1: Candle Confirmation (Weight 5.0) - HARD GATEKEEPER**

**Current Logic:** Complex multi-condition check (pin bar, engulfing, reversal patterns)

**Proposal - SIMPLIFIED:**
```
LONG: close > open AND close > ema20
      (Bullish candle + above key MA)

SHORT: close < open AND close < ema20
       (Bearish candle + below key MA)

min_cond: 1 of 1 (just needs to be directional)
```

**Purpose:** Gate signals to only directional bars (no doji, no sideways)
**Why it works:** Simple, meaningful, high pass rate (40-50% of bars)
**Gate function:** Blocks neutral/sideways bars

---

#### **FILTER #2: Support/Resistance (Weight 5.0) - FILTER (not gatekeeper)**

**Current Logic:** 
- Institutional confluence detection
- ATR-based dynamic margins
- Multi-TF retest validation
- Complexity: 200+ lines

**Proposal - SIMPLIFIED:**
```
window = 20  # Last 20 candles

recent_support = lowest_low(20)
recent_resistance = highest_high(20)
close = current_close

LONG: close >= recent_support 
      AND close < recent_support * 1.02
      (Within 2% above recent support)

SHORT: close <= recent_resistance 
       AND close > recent_resistance * 0.98
       (Within 2% below recent resistance)

min_cond: 1 of 1 (just proximity check)
```

**Purpose:** Detect bounces off recent extremes (retail-level S/R, not institutional)
**Why it works:** Simple calculation, quick filter, high pass rate (25-35% of bars)
**Expected result:** More passes, less institutional filtering

---

#### **FILTER #3: Volatility Model (Weight 3.9) - FILTER**

**Current Logic:**
- Complex ATR expansion (5% above MA)
- Lookback period validation
- Direction threshold (2 of 3 conditions)
- Complexity: 100+ lines

**Proposal - SIMPLIFIED:**
```
current_atr = atr[-1]
atr_ma_20 = atr_sma(20)

# Simple: Is volatility higher than baseline?
VOLATILITY_UP = current_atr > atr_ma_20

LONG: volatility_up AND close > close_prev
      (Volatility rising + price moving up)

SHORT: volatility_up AND close < close_prev
       (Volatility rising + price moving down)

min_cond: 1 of 1 (just volatility confirmation)
```

**Purpose:** Confirm moves with expanding volatility (institutional entry often causes ATR spike)
**Why it works:** Single gate (ATR > MA), directional check, high pass rate (40-50%)
**Expected result:** 50+ passes instead of 0

---

#### **FILTER #4: ATR Momentum Burst (Weight 4.3) - FILTER**

**Current Logic:**
- 0.15 ATR ratio requirement (15% move per bar)
- Simultaneous volume + volatility gates
- 2-bar minimum confirmation
- Complexity: 80+ lines

**Proposal - SIMPLIFIED:**
```
current_atr = atr[-1]
current_atr_prev = atr[-2]
current_close = close[-1]
current_close_prev = close[-2]

ATR_EXPANDING = current_atr > current_atr_prev

# Momentum: How strong is the move?
pct_move = abs((close - close_prev) / close_prev) * 100
momentum_strong = pct_move > current_atr * 0.02  # 2% of ATR (not 15%)

LONG: atr_expanding AND close > close_prev AND momentum_strong
      (Volatility up + price up + real move)

SHORT: atr_expanding AND close < close_prev AND momentum_strong
       (Volatility up + price down + real move)

min_cond: 1 of 1 (all three conditions must be true)
```

**Purpose:** Detect volatility-backed moves (real momentum, not noise)
**Why it works:** Lower threshold (2% vs 15%), no volume gate (removes dual-gate), high pass rate (30-40%)
**Expected result:** 100+ passes instead of 0

---

## PART C: Gatekeeper vs Filter Architecture

**Current (Broken):**
```
Hard Gatekeepers: Candle Confirmation (soft)  ← Doesn't gate
Filters: 20 filters including dead ones
Result: No real gating, signals pass with min_score only
```

**Proposed (Fixed):**
```
Hard Gatekeepers: Candle Confirmation (hard)  ← GATES directional bars only
Filters: 
  - Support/Resistance (simplified, high pass rate)
  - Volatility Model (simplified, high pass rate)
  - ATR Momentum Burst (simplified, high pass rate)
  - 17 other filters unchanged

Result: Only directional bars can fire (Candle gate), then score-based selection within those
```

**Logic Flow:**
```
Incoming bar
  ↓
Candle Confirmation hard gate → MUST have close > open (LONG) or close < open (SHORT)
  ↓
If FAILS → Signal blocked (no score check needed)
  ↓
If PASSES → Run all 20 filters
  ↓
Sum non-GK filter weights
  ↓
If score >= min_score (12) → Signal fires
  ↓
ELSE → Signal blocked
```

---

## PART D: Expected Impact

### Passing Rates (Current vs Proposed)

| Filter | Current Pass Rate | Current Status | Proposed Pass Rate | Proposed Status |
|--------|------------------|---|------------------|---|
| Candle Confirmation | 50% | Soft (non-gating) | 50% | **Hard (gating)** |
| Support/Resistance | 0% | Dead | 30% | Alive ✓ |
| Volatility Model | 0% | Dead | 45% | Alive ✓ |
| ATR Momentum Burst | 0% | Dead | 35% | Alive ✓ |

### Signal Quality Impact

**Hypothesis:** 
- Candle gate will **reduce total signals by ~30%** (filter out sideways bars)
- Simplified filters will **increase signal diversity** (no dead weight filters)
- Net result: **Fewer but higher-quality signals**

**Expected outcome:**
- 300 signals/day → 210 signals/day (gatekeeper effect)
- But more diverse (no longer blocked by institutional thresholds)
- Better win rate (directional bars have higher close% than sideways bars)

---

## PART E: Code Changes Required

### File: smart_filter.py

**Change 1: Remove soft_gatekeepers redundancy**
```python
# BEFORE:
self.gatekeepers_long = ["Candle Confirmation"]
self.gatekeepers_short = ["Candle Confirmation"]
self.soft_gatekeepers = ["Candle Confirmation"]  # ← Remove this line

# AFTER:
self.gatekeepers_long = ["Candle Confirmation"]
self.gatekeepers_short = ["Candle Confirmation"]
# (soft_gatekeepers becomes empty list or unused)
```

**Change 2: Simplify _check_candle_confirmation()**
- Replace 200+ line institutional logic with simple: `close > open AND close > ema20`

**Change 3: Simplify _check_support_resistance()**
- Replace 150+ line confluence logic with simple: `close >= support * 1.02`

**Change 4: Simplify _check_volatility_model()**
- Replace 100+ line expansion logic with simple: `atr > atr_ma AND directional`

**Change 5: Simplify _check_atr_momentum_burst()**
- Replace 80+ line dual-gate logic with simple: `atr_up AND move > 2% of atr`

---

## PART F: Risk Assessment

### Low Risk
✓ Simplification only (logic becomes simpler, not inverted)  
✓ Candle gate is conservative (only gates, doesn't amplify)  
✓ New filters higher pass rate = more signals, but all gated by Candle  

### Medium Risk
⚠ Might see 20-30% decrease in signal count (gatekeeper effect)  
⚠ Win rate may temporarily shift (need 24-48h to validate)  

### Mitigation
- Deploy to staging first (1h testing)
- Monitor signal counts vs baseline (236/hr from 20:00-21:00 hour)
- If win rate drops >5%, rollback and adjust parameters

---

## PART G: Approval Checklist

Before implementation, please confirm:

- [ ] **Gatekeeper fix**: OK to move Candle Confirmation to hard-only (remove soft)?
- [ ] **Support/Resistance**: OK to simplify to 2% proximity check?
- [ ] **Volatility Model**: OK to simplify to ATR > MA + directional?
- [ ] **ATR Momentum Burst**: OK to simplify to 2% move threshold (not 15%)?
- [ ] **Rollout**: Staging 1h → Production if signal counts stable?

---

## Timeline

**If approved:**
- **2026-03-23 21:15** - Code changes (all 5 filter methods)
- **2026-03-23 21:30** - Unit tests (verify each filter logic)
- **2026-03-23 21:45** - Staging deployment (daemon reload)
- **2026-03-24 00:00** - Review first hour metrics
- **2026-03-24 06:00** - 24-hour validation complete
- **2026-03-24 12:00** - Go/no-go decision

---

## Code Examples (Ready to Copy-Paste)

See attached simplified implementations below (after approval).

