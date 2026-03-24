# CORRECTED WEIGHT HIERARCHY BY STATUS TIER
**Date:** 2026-03-24  
**Method:** Normalize by STATUS TIER, not win rate directly  
**Principle:** Better filters get proportionally higher weights

---

## WEIGHT ASSIGNMENT FRAMEWORK

### Weight Tier Ranges (Proposed)
- **BEST:** 5.5-6.5 (highest confidence)
- **GOOD:** 5.0-5.5 (strong support)
- **SOLID:** 4.5-5.0 (core ensemble)
- **BASELINE:** 4.0-4.5 (standard performance)
- **WEAK:** 2.5-3.5 (marginal value)
- **DEAD:** 1.0-2.0 (specified values, drawdown control)
- **TOXIC:** 0.5 (floor, zombie filters)

### Multiplier Structure
```
BEST status:     weight = X (highest)
GOOD status:     weight = X * 0.95
SOLID status:    weight = X * 0.90
BASELINE status: weight = X * 0.85
WEAK status:     weight = X * 0.70
DEAD status:     use specified values
TOXIC status:    0.5 (floor)
```

---

## DETAILED WEIGHT ASSIGNMENT

### 🏆 TIER 1: BEST (WR > 30%)
**Base weight:** 6.0  
**Rationale:** Highest signal quality, use as primary confirmation

| Filter | WR | Assigned Weight | Notes |
|--------|-----|-----------------|-------|
| **Momentum** | 30.3% | **6.0** | Primary direction filter, best performer |

### ✅ TIER 2: GOOD (WR = 29-30%)
**Multiplier:** 6.0 × 0.95 = 5.7  
**Rationale:** Strong support, nearly as reliable as BEST

| Filter | WR | Assigned Weight | Notes |
|--------|-----|-----------------|-------|
| **Spread Filter** | 29.9% | **5.7** | Liquidity validation, high confidence |

### 💪 TIER 3: SOLID (WR = 27-29%)
**Multiplier:** 6.0 × 0.90 = 5.4  
**Rationale:** Core ensemble filters, reliable across conditions

| Filter | WR | Assigned Weight | Notes |
|--------|-----|-----------------|-------|
| **HH/LL Trend** | 27.9% | **5.4** | Trend structure, consistent |
| **Liquidity Awareness** | 27.4% | **5.4** | Risk control, highly valuable |
| **Fractal Zone** | 27.4% | **5.4** | Pattern recognition, reliable |
| **Wick Dominance** | 27.2% | **5.4** | Price action, quality signals |
| **MTF Volume Agreement** | 27.2% | **5.4** | Multi-TF confirmation, strong |
| **Smart Money Bias** | 27.3% | **5.4** | Institutional flow, valuable |
| **Liquidity Pool** | 27.3% | **5.4** | Accumulation zones, consistent |
| **Volatility Squeeze** | 27.3% | **5.4** | Mean-reversion setup, reliable |

### 📊 TIER 4: BASELINE (WR = 26-27%)
**Multiplier:** 6.0 × 0.85 = 5.1  
**Rationale:** Standard performance, acceptable as confirming filters

| Filter | WR | Assigned Weight | Notes |
|--------|-----|-----------------|-------|
| **MACD** | 26.8% | **5.1** | Momentum oscillator |
| **TREND Unified** | 26.6% | **5.1** | Overall trend direction |
| **Volume Spike** | 26.5% | **5.1** | Volume expansion |
| **Chop Zone** | 26.2% | **5.1** | Ranging market filter |

### ⚠️ TIER 5: WEAK (WR = 25-26%)
**Multiplier:** 6.0 × 0.70 = 4.2  
**Rationale:** Marginal value, use as optional confirmation only

| Filter | WR | Assigned Weight | Notes |
|--------|-----|-----------------|-------|
| **VWAP Divergence** | 25.8% | **4.2** | Divergence detection |

### ⛓️ TIER 6: DEAD (WR < 22%, severe under-performance)
**Weights:** User-specified, fixed per user correction  
**Rationale:** Control drawdown, prevent worst performers from harmful influence

| Filter | WR | Assigned Weight | Notes |
|--------|-----|-----------------|-------|
| **ATR Momentum Burst** | 20.4% | **2.0** | User spec: drawdown control |
| **Volatility Model** | 14.8% | **1.5** | User spec: severe under-perf |
| **Support/Resistance** | 0% | **1.0** | User spec: zombie filter |
| **Absorption** | 0% | **0.5** | User spec: zombie filter |

### ☠️ TIER 7: TOXIC (WR = 0%, broken/zombie)
**Weight:** 0.5 (floor)  
**Rationale:** Prevent negative influence while awaiting repair/removal

| Filter | WR | Assigned Weight | Action |
|--------|-----|-----------------|--------|
| **Candle Confirmation** | 0% | **0.5** | ⚠️ INVESTIGATE: 0 passes, disable or fix |

---

## WEIGHT DISTRIBUTION SUMMARY

### Count by Tier
| Tier | Count | Total Weight | Avg Weight |
|------|-------|--------------|-----------|
| BEST | 1 | 6.0 | 6.0 |
| GOOD | 1 | 5.7 | 5.7 |
| SOLID | 8 | 43.2 | 5.4 |
| BASELINE | 4 | 20.4 | 5.1 |
| WEAK | 1 | 4.2 | 4.2 |
| DEAD | 4 | 5.0 | 1.25 |
| TOXIC | 1 | 0.5 | 0.5 |
| **TOTAL** | **20** | **85.0** | **4.25** |

### Weight Profile
- **High Performance (BEST + GOOD + SOLID):** 54.9 total weight (64.6%)
- **Standard (BASELINE + WEAK):** 24.6 total weight (28.9%)
- **Drawdown Control (DEAD + TOXIC):** 5.5 total weight (6.5%)

---

## APPLICATION LOGIC

### Ensemble Scoring
```python
def calculate_filter_score(signal_dict, weights_dict):
    """
    Calculate ensemble score weighted by STATUS tier
    """
    total_score = 0
    total_weight = 0
    
    for filter_name, filter_passed in signal_dict.items():
        if filter_name in weights_dict:
            weight = weights_dict[filter_name]
            score = 1 if filter_passed else 0
            total_score += score * weight
            total_weight += weight
    
    # Normalize to 0-1 range
    normalized_score = total_score / total_weight if total_weight > 0 else 0
    return normalized_score, total_score, total_weight

# Example:
weights = {
    'Momentum': 6.0,           # BEST
    'Spread Filter': 5.7,      # GOOD
    'HH/LL Trend': 5.4,        # SOLID
    'MACD': 5.1,               # BASELINE
    'ATR Momentum Burst': 2.0,  # DEAD (drawdown control)
    'Candle Confirmation': 0.5, # TOXIC (zombie)
}
```

### Gating Logic
```python
def apply_ensemble_gate(signal_dict, weights_dict, min_threshold=0.60):
    """
    Apply ensemble gate: signal must pass majority of high-tier filters
    
    BEST + GOOD: 11.7 total (27.4% of high-tier)
    SOLID: 43.2 total (72.6% of high-tier)
    
    Gate: Signal must pass >= 60% of filters by weight
    """
    score, total_score, total_weight = calculate_filter_score(signal_dict, weights_dict)
    
    if score >= min_threshold:
        return signal_dict  # PASS
    else:
        return None  # REJECT
```

---

## KEY IMPROVEMENTS OVER PREVIOUS WEIGHTS

### Previous Problems
- All filters weighted 0.5-6.1 with no clear hierarchy
- No distinction between SOLID performers and BASELINE performers
- Zombie filters (0% WR) treated same as legitimate DEAD filters
- LONG/SHORT bias not addressed in weights (symmetric application)

### Current Corrections
✅ **Tier-based normalization** — filters grouped by performance category  
✅ **Dead/Toxic separation** — zombie filters (0% WR) isolated at floor (0.5)  
✅ **Specified dead weights** — ATR (2.0), Vol Model (1.5), S/R (1.0), Absorption (0.5)  
✅ **High-tier clustering** — BEST/GOOD/SOLID compressed (5.4-6.0) for cohesion  
✅ **Weak/Baseline separation** — WEAK filters (4.2) clearly distinguished  
✅ **LONG/SHORT symmetric** — weights applied equally to both directions  

---

## NEXT STEPS

1. **Implement weights** in ensemble calculation
2. **Set ensemble gate threshold** to 0.60 (60% weighted pass rate)
3. **Apply route veto** before ensemble gate (NONE + AMBIGUOUS always blocked)
4. **Monitor DEAD/TOXIC filters** — flag any improvement for re-classification
5. **Track weight impact** — measure WR before/after weight adjustment

---

## TRANSITION CHECKLIST

- [ ] Update filter weight configuration
- [ ] Verify Momentum (6.0) as primary direction filter
- [ ] Disable or investigate Candle Confirmation (0.5 floor)
- [ ] Apply route-only veto (NONE + AMBIGUOUS)
- [ ] Test ensemble gate threshold (recommend 0.60)
- [ ] Log weight changes for performance tracking
- [ ] Measure improvement after 100-200 signals
