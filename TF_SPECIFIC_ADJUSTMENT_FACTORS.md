# Timeframe-Specific Adjustment Factors Beyond Volatility

## Summary: What Needs Per-TF Adjustment?

Based on analysis of ETH-USDT across 5 timeframes:

### ✅ YES - Needs Adjustment
1. **Volatility** (ATR, StdDev, % change)
2. **Support/Resistance Proximity** (2% threshold broken for 4h)
3. **Lookback Period/Moving Average Length** (20-bar = 5h on 15min, 80h on 4h)
4. **Reversal Frequency** (59% on 30min/2h, 45% on 4h)

### ❌ NO - Same for All TFs
- Mean Reversion vs Trend Following (all ~0.0 autocorrelation = neutral)
- Noise Level ratio (all ~2.2, consistent)
- Trend Duration (only slight variation 1.7-2.2)

---

## 1️⃣ VOLATILITY (Already Identified)

```
TF     | ATR   | % Change | Adjustment Needed
────────────────────────────────────────────────
15min  | 11.81 | 0.20%    | Baseline
30min  | 14.65 | 0.28%    | +1.4x
1h     | 16.14 | 0.44%    | +2.2x
2h     | 25.89 | 0.64%    | +3.2x
4h     | 45.39 | 0.88%    | +4.4x
```

**Thresholds to adjust:**
- Volatility Squeeze: 0.05 → 0.12 (for 2h/4h)
- VWAP Divergence: 0.005 → 0.02 (for 2h/4h)

---

## 2️⃣ SUPPORT/RESISTANCE PROXIMITY

**Current threshold:** price_proximity_pct = 3% (0.03)

```
TF     | Avg Range | Near 3% S/R? | Problem
────────────────────────────────────────────────
15min  | 0.55%     | ❌ Rare      | ✅ Works
30min  | 0.68%     | ❌ Rare      | ✅ Works
1h     | 0.75%     | ❌ Rare      | ✅ Works
2h     | 1.20%     | ❌ Rare      | ✅ Works
4h     | 2.10%     | ⚠️  Sometimes| ⚠️  Marginal
```

**Issue:** On 4h, S/R hits are common (2.1% avg range) but threshold is 3% → filter rarely passes

**Fix needed:**
```python
if tf == "4h":
    price_proximity_pct = 0.06  # 6% for 4h (wider acceptance)
elif tf == "2h":
    price_proximity_pct = 0.04  # 4% for 2h
else:
    price_proximity_pct = 0.03  # 3% for shorter TFs
```

---

## 3️⃣ LOOKBACK PERIOD / MOVING AVERAGE LENGTH

**Current:** All TFs use 20-bar moving average (EMA20)

```
TF     | 20-bar = ? Hours | Adequacy
────────────────────────────────────────────────
15min  | 5 hours          | ⚠️  Too short (noise)
30min  | 10 hours         | ✅ Good
1h     | 20 hours         | ✅ Good
2h     | 40 hours         | ✅ Good
4h     | 80 hours         | ✅ Good (maybe too long)
```

**Issue:** 15min with 5-hour lookback is too noisy; 4h with 80-hour lookback might miss recent momentum

**Fix needed:**
```python
if tf == "15min":
    ma_period = 60         # 60-bar = 15 hours (better noise filtering)
elif tf == "4h":
    ma_period = 14         # 14-bar = 56 hours (sharper response)
else:
    ma_period = 20         # 20-bar = standard
```

---

## 4️⃣ REVERSAL FREQUENCY

**Current:** All TFs use same reversal detection logic

```
TF     | Reversals per 100 candles | Implication
────────────────────────────────────────────────────
15min  | 55%                       | Whipsaws common
30min  | 59%                       | Whipsaws common ⚠️
1h     | 49%                       | Moderate
2h     | 59%                       | Whipsaws common ⚠️
4h     | 45%                       | Rare, reliable
```

**Issue:** 30min/2h have VERY HIGH reversal rates (59%) = many false signals in reversal patterns

**Fix needed:**
```python
# For high-reversal-frequency TFs (30min, 2h):
# Require higher bar count before considering it a reversal pattern
if tf in ["30min", "2h"]:
    reversal_bars_required = 5   # Stronger confirmation
else:
    reversal_bars_required = 3   # Normal
```

---

## 5️⃣ TREND CHARACTERISTICS (Minor, Skip for Now)

Trend duration is similar across TFs (1.7-2.2 bars), so this doesn't need adjustment.

---

## RECOMMENDED IMPLEMENTATION APPROACH

**Option 2A: Volatility + Proximity + Lookback Normalization**

```python
def get_tf_specific_params(tf):
    """Return adjusted thresholds based on timeframe"""
    
    # Detect volatility
    current_vol = self.df['close'].std()
    
    params = {
        "vwap_deviation_pct": 0.005,      # Default
        "price_proximity_pct": 0.03,       # Default
        "ma_period": 20,                   # Default
        "reversal_bars": 3,                # Default
        "min_squeeze_diff": 0.05,          # Default
    }
    
    # Adjust based on TF + volatility
    if tf == "4h" or current_vol > 1500:
        params["vwap_deviation_pct"] = 0.02   # 2.0%
        params["price_proximity_pct"] = 0.06  # 6.0%
        params["ma_period"] = 14              # Shorter lookback
        params["reversal_bars"] = 3           # Same
        params["min_squeeze_diff"] = 0.12     # Looser squeeze
        
    elif tf == "2h" or current_vol > 1200:
        params["vwap_deviation_pct"] = 0.015  # 1.5%
        params["price_proximity_pct"] = 0.04  # 4.0%
        params["ma_period"] = 20              # Normal
        params["reversal_bars"] = 5           # Higher (59% reversal rate)
        params["min_squeeze_diff"] = 0.10     # Looser squeeze
        
    elif tf == "15min" or current_vol < 800:
        params["ma_period"] = 60              # Longer lookback (less noise)
        params["reversal_bars"] = 3           # Normal
        
    return params
```

---

## Summary Table: What to Adjust Per TF

```
Parameter                | 15min | 30min | 1h    | 2h    | 4h
──────────────────────────────────────────────────────────────────
Volatility Sensitivity   | 1.0x  | 1.4x  | 2.2x  | 3.2x  | 4.4x
VWAP Divergence %        | 0.5%  | 0.5%  | 0.7%  | 1.5%  | 2.0%
S/R Proximity %           | 3%    | 3%    | 3%    | 4%    | 6%
MA Lookback (bars)       | 60    | 20    | 20    | 20    | 14
Reversal Confirmation    | 3     | 5     | 3     | 5     | 3
Squeeze Diff Threshold   | 0.05  | 0.05  | 0.05  | 0.10  | 0.12
```

---

## RECOMMENDATION

**Implement a 3-Factor Normalization:**
1. **Volatility-based thresholds** (already identified)
2. **S/R Proximity adjustment** (for 4h)
3. **Reversal frequency adjustment** (for 30min/2h)

Skip lookback period adjustment for now - it's a secondary refinement.

This will fix 2h/4h signal generation without breaking 15min/30min/1h.
