# 🔧 PROACTIVE UPSTREAM ANALYSIS - Framework & Initial Findings
**Generated:** 2026-03-24 13:46 GMT+7  
**Data Source:** 2,193 signals, 1,678 closed trades, 30.51% baseline WR  
**Goal:** Understand WHY signals fail/win, then FIX the generation itself

---

## 📊 OVERVIEW: Problem Areas Identified

From the 2,193 signals, **4 critical upstream problems** emerged:

| Problem | Impact | Root Cause (Hypothesis) | Fix Strategy |
|---------|--------|------------------------|--------------|
| **BULL regime losing** | -10.6pp WR vs BEAR | Regime detection too loose? Thresholds off? | Tighten BULL filters or exclude entirely |
| **LOW_ALTS underperforming** | -10.7pp WR vs MAIN_BLOCKCHAIN | High volatility? Poor TP:SL ratio? Symbol liquidity? | Adjust position sizing, TP/SL by symbol group |
| **LONG direction weak** | -8.5pp WR vs SHORT | Directional bias in market? Filters better for shorts? | Add LONG-specific gatekeeper, retune entry logic |
| **AMBIGUOUS/NONE routes toxic** | -17.3pp WR vs REVERSAL | Catch-all dumping ground? Should veto? | Convert to explicit gatekeeper rule |

---

## 🎯 AUDIT 1: FILTER LOGIC
### Thesis: Not all 20 filters equally contribute to wins

**Current State:** All 20 filters active with weights (PROJECT-8 reweighting deployed)

**Key Questions:**
1. Which filters NEVER pass? (Potential gatekeepers?)
2. Which filters pass but generate losses?
3. Are filter thresholds correct for different regimes?

**Preliminary Data (from PROJECT-8 discovery):**
- **5 filters never/rarely passing:**
  - Support/Resistance: 0% WR (0 wins from 3 passes)
  - Absorption: 0% WR (rare pattern)
  - Volatility Model: 14.8% WR vs 27.3% baseline
  - ATR Momentum Burst: 20.4% WR vs 27.3% baseline
  - Candle Confirmation: 5.0 weight (GATEKEEPER - should it be?)

**Action:** Deep-dive which filter combos appear in winners (Tier-1) vs losers (Tier-X)

---

## 🌊 AUDIT 2: REGIME LOGIC
### Thesis: BEAR works (+640bps), BULL fails (-320bps), should we segment by regime?

**Current Data:**
```
BEAR:  37.2% WR (629 signals) → +$533.65 P&L ✓ PROFITABLE
BULL:  26.6% WR (912 signals) → -$6,478.77 P&L ✗ LOSING
RANGE: 27.4% WR (176 signals) → -$443.43 P&L ✗ LOSING
```

**Breaking Down:**
```
Direction × Regime Performance:

LONG + BULL:   26.4% WR ← Worst combo (-5,354pp loss)
LONG + BEAR:   30.9% WR ← Better but not great
SHORT + BEAR:  39.8% WR ← Best combo (432 signals, winning)
SHORT + BULL:  34.8% WR ← Decent
```

**Critical Finding:** 
- **SHORT + BEAR = Money machine** (39.8% WR, 432 signals)
- **LONG + BULL = Toxic** (26.4% WR, 879 signals) → Should we veto this entirely?

**Action:** Propose regime-specific gatekeeper rules

---

## 🛣️ AUDIT 3: ROUTE LOGIC
### Thesis: REVERSAL is elite, TREND_CONT stable, AMBIGUOUS/NONE are catch-all garbage

**Current Data:**
```
REVERSAL:          33.8% WR (151 signals, -$108 P&L) ← Clearest entries
TREND CONTINUATION: 32.0% WR (1,364 signals, -$4,211 P&L) ← Most volume
AMBIGUOUS:         20.8% WR (72 signals, -$525 P&L) ← Worst
NONE:              13.3% WR (92 signals, -$1,152 P&L) ← TOXIC
```

**But drilling deeper:**
```
REVERSAL + RANGE:        41.7% WR (48 signals, +$361 P&L) ← ELITE
REVERSAL + BULL:         32.7% WR (55 signals)
TREND CONT + BEAR:       39.2% WR (530 signals, +$1,148 P&L) ← ELITE

NONE + BULL:              9.6% WR (52 signals, -$842 P&L) ← POISON
AMBIGUOUS + BULL:        20.9% WR (43 signals, -$294 P&L)
NONE + BEAR:             12.5% WR (16 signals, -$172 P&L)
```

**Critical Finding:**
- **NONE route should be a VETO** (13.3% WR overall, below 30% baseline)
- **AMBIGUOUS is questionable** (20.8% WR, below 30% baseline)
- **REVERSAL needs regime filter** (41.7% in RANGE, but 32.7% in BULL)

**Action:** Add route-specific veto rules

---

## 👮 AUDIT 4: GATEKEEPER RULES (WHAT SHOULD VETO A SIGNAL?)
### Thesis: Some dimension combos are so toxic they should never fire

**Current Veto Candidates:**

| Rule | Reason | Sample | WR | Impact |
|------|--------|--------|-----|---------|
| VETO: ROUTE == "NONE" | Below baseline, always loses | 92 | 13.3% | Prevent 92 bad signals |
| VETO: ROUTE == "AMBIGUOUS" + BULL | Toxic combo | 43 | 20.9% | Prevent 43 bad signals |
| VETO: DIRECTION == "LONG" + REGIME == "BULL" | Worst combo | 879 | 26.4% | Prevent 879 losing signals |
| VETO: SYMBOL_GROUP == "LOW_ALTS" + WR < 27% | Dragging down | ? | 27.7% | Need thresholds |
| REQUIRE: REGIME == "BEAR" + DIRECTION == "SHORT" | Only fire elite combos | 432 | 39.8% | Only ~2.5/hour |

**Cost-Benefit:**
```
If we add VETO rules:
- Prevent 1,014 low-WR signals (46% of total)
- Keep only 1,179 signals (54% of total)
- Expected WR improvement: +2-4pp (need to validate)
- Drawback: Lower signal velocity (110 → 55 signals/day)
```

---

## 💰 AUDIT 5: SYMBOL GROUP LOGIC
### Thesis: LOW_ALTS is a TP:SL problem, not a filter problem

**Current Data:**
```
MAIN_BLOCKCHAIN: 38.4% WR (172 sig) but -$199 P&L ← Good WR, bad P&L
MID_ALTS:        38.2% WR (186 sig) but -$847 P&L ← Good WR, bad P&L
TOP_ALTS:        37.3% WR (102 sig) but -$864 P&L ← Good WR, bad P&L
LOW_ALTS:        27.7% WR (1,257 sig) and -$4,479 P&L ← Bad WR, bad P&L
```

**Hypothesis:** 
- Higher alts have better WR but LOSSES because TP targets are too small relative to volatility
- LOW_ALTS has lower WR AND losses (compounding problem)

**Actions to Test:**
1. Adjust TP:SL ratio by symbol group (wider for lower alts)
2. Reduce LOW_ALTS allocation (weight down)
3. Increase position size for MAIN_BLOCKCHAIN (higher Sharpe)

---

## 🔧 CONFIDENCE LEVEL AUDIT
### Thesis: HIGH confidence filters work, MID/LOW are weak

**Current Data:**
```
HIGH (≥73%):   33.0% WR (900 signals) ← Best
MID (66-72%):  26.3% WR (321 signals) ← Worst
LOW (≤65%):    28.7% WR (496 signals) ← Middle
```

**Action:** Boost HIGH confidence signals, suppress MID/LOW (or add gatekeeper for MID <26%)

---

## 📋 TIMEFRAME ANALYSIS
### Thesis: 1h is highest quality, 15min is highest volume but lowest quality

**Current Data:**
```
1h:    34.6% WR (277 sig) — BEST, limited velocity
15min: 29.7% WR (747 sig) — LOWEST WR, highest velocity
30min: 29.9% WR (693 sig) — LOWEST WR, medium velocity
```

**Action:** 
- Consider boosting 1h weight (if gatekeeper-friendly)
- Suppress 15min/30min (or add regime filter: only BEAR + SHORT on fast TFs)

---

## 🎯 PROPOSED GATEKEEPER RULES (PHASE 1)

Based on upstream analysis, recommend adding these VETO rules to smart_filter.py:

```python
def apply_gatekeepers(signal_dict):
    """
    UPSTREAM VETO RULES - prevent bad signals before they fire
    """
    
    # RULE 1: Block NONE route (13.3% WR, always loses)
    if signal_dict['route'] == 'NONE':
        return None  # VETO
    
    # RULE 2: Block LONG + BULL combo (26.4% WR, 879 losing signals)
    if signal_dict['direction'] == 'LONG' and signal_dict['regime'] == 'BULL':
        return None  # VETO
    
    # RULE 3: Block MID confidence in certain conditions
    if signal_dict['confidence'] < 0.66 and signal_dict['regime'] != 'BEAR':
        return None  # VETO (only allow LOW confidence in BEAR)
    
    # RULE 4: Block AMBIGUOUS in BULL (20.9% WR)
    if signal_dict['route'] == 'AMBIGUOUS' and signal_dict['regime'] == 'BULL':
        return None  # VETO
    
    # RULE 5: Prefer SHORT + BEAR (39.8% WR, elite combo)
    # Don't veto others, but weight this higher
    
    # RULE 6: Block 15min if not BEAR + SHORT (too risky)
    if signal_dict['timeframe'] == '15min':
        if not (signal_dict['direction'] == 'SHORT' and signal_dict['regime'] == 'BEAR'):
            return None  # VETO
    
    return signal_dict  # PASS
```

---

## ⚠️ NEXT STEPS (Waiting for Tier Analysis)

1. ✅ This framework identifies upstream problems
2. ⏳ Tier analysis will show **which filters/combos** dominate Tier-1 signals
3. 🎯 Combine insights to propose final gatekeeper rules + reweighting

---

**Status:** Framework ready, awaiting tier analysis results to finalize recommendations

