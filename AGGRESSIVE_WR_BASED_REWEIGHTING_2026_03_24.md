# Aggressive WR-Based Reweighting (No Zero Weights)
**Date:** 2026-03-24 00:36 GMT+7  
**Method:** Apply WR ratio formula aggressively, keep minimum floor of 0.5

---

## 📊 ACTUAL CURRENT WEIGHTS (From Code)

Total: **86.5** (all 20 filters including Support/Resistance at 5.0)

---

## 🔬 CALCULATION: WR-Based Reweighting

**Formula:** `New Weight = Current Weight × (Filter WR / Baseline WR)`  
**Floor:** Minimum 0.5 (no zero weights)  
**Baseline WR:** 27.3%

| Rank | Filter | Current | WR | Ratio | Calc | Proposed | Change | Status |
|------|--------|---------|-----|-------|------|----------|--------|--------|
| 1 | Momentum | 5.5 | 30.3% | 1.111 | 5.5 × 1.111 = 6.1 | **6.1** | +0.6 | ⬆️⬆️ BOOST |
| 2 | Spread Filter | 5.0 | 29.9% | 1.095 | 5.0 × 1.095 = 5.5 | **5.5** | +0.5 | ⬆️ BOOST |
| 3 | HH/LL Trend | 4.8 | 27.9% | 1.022 | 4.8 × 1.022 = 4.9 | **4.9** | +0.1 | ↑ BOOST |
| 4 | Liquidity Awareness | 5.3 | 27.4% | 1.003 | 5.3 × 1.003 = 5.3 | **5.3** | 0.0 | ➡️ KEEP |
| 5 | Fractal Zone | 4.2 | 27.4% | 1.003 | 4.2 × 1.003 = 4.2 | **4.2** | 0.0 | ➡️ KEEP |
| 6 | Volatility Squeeze | 3.2 | 27.3% | 1.000 | 3.2 × 1.000 = 3.2 | **3.2** | 0.0 | ➡️ KEEP |
| 7 | Liquidity Pool | 3.1 | 27.3% | 1.000 | 3.1 × 1.000 = 3.1 | **3.1** | 0.0 | ➡️ KEEP |
| 8 | Smart Money Bias | 4.5 | 27.3% | 1.000 | 4.5 × 1.000 = 4.5 | **4.5** | 0.0 | ➡️ KEEP |
| 9 | Wick Dominance | 4.0 | 27.2% | 0.996 | 4.0 × 0.996 = 3.98 | **4.0** | 0.0 | ➡️ KEEP |
| 10 | MTF Volume Agree | 4.6 | 27.2% | 0.996 | 4.6 × 0.996 = 4.58 | **4.6** | 0.0 | ➡️ KEEP |
| 11 | MACD | 5.0 | 26.8% | 0.982 | 5.0 × 0.982 = 4.91 | **4.9** | -0.1 | ↓ SLIGHT CUT |
| 12 | TREND | 4.3 | 26.6% | 0.974 | 4.3 × 0.974 = 4.19 | **4.2** | -0.1 | ↓ SLIGHT CUT |
| 13 | Volume Spike | 5.3 | 26.5% | 0.971 | 5.3 × 0.971 = 5.15 | **5.2** | -0.1 | ↓ SLIGHT CUT |
| 14 | Candle Confirmation | 5.0 | N/A | N/A | Keep as gatekeeper | **5.0** | 0.0 | ➡️ GATEKEEPER |
| 15 | Chop Zone | 3.3 | 26.2% | 0.960 | 3.3 × 0.960 = 3.17 | **3.2** | -0.1 | ↓ SLIGHT CUT |
| 16 | VWAP Divergence | 3.5 | 25.8% | 0.945 | 3.5 × 0.945 = 3.31 | **3.3** | -0.2 | ↓ CUT |
| 17 | ATR Momentum Burst | 4.3 | 20.4% | 0.747 | 4.3 × 0.747 = 3.21 | **3.2** | -1.1 | ↓↓ CUT HARD |
| 18 | Volatility Model | 3.9 | 14.8% | 0.542 | 3.9 × 0.542 = 2.11 | **2.1** | -1.8 | ↓↓↓ CUT HARD |
| 19 | Support/Resistance | 5.0 | 0.0% | 0.000 | 5.0 × 0.000 = 0.0 → **floor 0.5** | **0.5** | -4.5 | ↓↓↓↓ CUT MOST |
| 20 | Absorption | 2.7 | 0.0% | 0.000 | 2.7 × 0.000 = 0.0 → **floor 0.5** | **0.5** | -2.2 | ↓↓↓ CUT HARD |
| **TOTAL** | | **86.5** | | | | **79.5** | **-7.0** | |

---

## 📈 BREAKDOWN

### Boosts (High WR)
```
Momentum:         5.5 → 6.1  (+0.6)  ← 30.3% WR, best performer
Spread Filter:    5.0 → 5.5  (+0.5)  ← 29.9% WR, second best
HH/LL Trend:      4.8 → 4.9  (+0.1)  ← 27.9% WR, above baseline
────────────────────────────────
TOTAL BOOST:     +1.2 weight
```

### Keeps (At Baseline ~27.3%)
```
Liquidity Awareness: 5.3 → 5.3  (0.0)  ← 27.4% WR
Fractal Zone:        4.2 → 4.2  (0.0)  ← 27.4% WR
Volatility Squeeze:  3.2 → 3.2  (0.0)  ← 27.3% WR
Liquidity Pool:      3.1 → 3.1  (0.0)  ← 27.3% WR
Smart Money Bias:    4.5 → 4.5  (0.0)  ← 27.3% WR
Wick Dominance:      4.0 → 4.0  (0.0)  ← 27.2% WR
MTF Volume:          4.6 → 4.6  (0.0)  ← 27.2% WR
Candle Confirmation: 5.0 → 5.0  (0.0)  ← Gatekeeper
────────────────────────────────
TOTAL KEEPS:      0.0 weight
```

### Slight Cuts (Slightly Below Baseline)
```
MACD:                 5.0 → 4.9  (-0.1)  ← 26.8% WR
TREND:                4.3 → 4.2  (-0.1)  ← 26.6% WR
Volume Spike:         5.3 → 5.2  (-0.1)  ← 26.5% WR
Chop Zone:            3.3 → 3.2  (-0.1)  ← 26.2% WR
────────────────────────────────
TOTAL SLIGHT CUTS: -0.4 weight
```

### Hard Cuts (Well Below Baseline)
```
VWAP Divergence:      3.5 → 3.3  (-0.2)  ← 25.8% WR
ATR Momentum Burst:   4.3 → 3.2  (-1.1)  ← 20.4% WR
Volatility Model:     3.9 → 2.1  (-1.8)  ← 14.8% WR
────────────────────────────────
TOTAL HARD CUTS:  -3.1 weight
```

### Most Cuts (Toxic - 0% WR, Kept Above Floor)
```
Support/Resistance:   5.0 → 0.5  (-4.5)  ← 0% WR (0 wins from 3 passes)
Absorption:           2.7 → 0.5  (-2.2)  ← 0% WR (dead filter)
────────────────────────────────
TOTAL MOST CUTS:  -6.7 weight
```

---

## 🎯 TOTAL CHANGE

```
BEFORE: 86.5 weight
AFTER:  79.5 weight
DELTA:  -7.0 weight (8.1% reduction)

Breakdown:
├─ Boosts:      +1.2
├─ Keeps:        0.0
├─ Slight cuts: -0.4
├─ Hard cuts:   -3.1
└─ Most cuts:   -6.7
```

---

## 📋 PROPOSED NEW WEIGHTS (All 20 Filters)

```python
self.filter_weights_long = {
    "MACD": 4.9,                    # CUT: 5.0 → 4.9 (-0.1)
    "Volume Spike": 5.2,            # CUT: 5.3 → 5.2 (-0.1)
    "Fractal Zone": 4.2,            # KEEP: 4.2 → 4.2 (0.0)
    "TREND": 4.2,                   # CUT: 4.3 → 4.2 (-0.1)
    "Momentum": 6.1,                # BOOST: 5.5 → 6.1 (+0.6) ⭐
    "ATR Momentum Burst": 3.2,      # CUT: 4.3 → 3.2 (-1.1)
    "MTF Volume Agreement": 4.6,    # KEEP: 4.6 → 4.6 (0.0)
    "HH/LL Trend": 4.9,             # BOOST: 4.8 → 4.9 (+0.1)
    "Volatility Model": 2.1,        # CUT: 3.9 → 2.1 (-1.8)
    "Liquidity Awareness": 5.3,     # KEEP: 5.3 → 5.3 (0.0)
    "Volatility Squeeze": 3.2,      # KEEP: 3.2 → 3.2 (0.0)
    "Candle Confirmation": 5.0,     # KEEP: 5.0 → 5.0 (0.0) [Gatekeeper]
    "VWAP Divergence": 3.3,         # CUT: 3.5 → 3.3 (-0.2)
    "Spread Filter": 5.5,           # BOOST: 5.0 → 5.5 (+0.5) ⭐
    "Chop Zone": 3.2,               # CUT: 3.3 → 3.2 (-0.1)
    "Liquidity Pool": 3.1,          # KEEP: 3.1 → 3.1 (0.0)
    "Support/Resistance": 0.5,      # CUT: 5.0 → 0.5 (-4.5) [Floor, not zero]
    "Smart Money Bias": 4.5,        # KEEP: 4.5 → 4.5 (0.0)
    "Absorption": 0.5,              # CUT: 2.7 → 0.5 (-2.2) [Floor, not zero]
    "Wick Dominance": 4.0           # KEEP: 4.0 → 4.0 (0.0)
}

self.filter_weights_short = {
    # Identical to above (20 filters)
}
```

---

## ✅ KEY FEATURES

1. **No zero weights** - Minimum floor of 0.5
2. **Aggressive reweighting** - Applied full WR ratio formula
3. **Boosts high performers** - Momentum +0.6, Spread +0.5
4. **Cuts bad performers** - Vol Model -1.8, ATR Burst -1.1
5. **Minimizes toxic** - S/R -4.5, Absorption -2.2 (to 0.5, not zero)
6. **Total reduction** - 86.5 → 79.5 (-7.0 weight, 8.1%)

---

## 🎯 EXPECTED OUTCOME

```
Current WR (from data): 27.3%
Expected after reweight: ~29-30%
Improvement: +2-3pp

Why:
- Remove influence of 0% WR filters (S/R, Absorption still fire, just minimal)
- Reduce bad performers (Vol Model, ATR Burst) significantly
- Boost best performers (Momentum, Spread)
- Net: Higher quality signal mix
```

---

## 🚀 READY TO DEPLOY

2 changes needed in smart_filter.py:
1. Replace all weights in `self.filter_weights_long` dict
2. Replace all weights in `self.filter_weights_short` dict

Est. deployment time: 3 minutes (edit + commit + reload)

**Approve? YES / NO**

