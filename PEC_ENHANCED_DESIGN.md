# PEC ENHANCED DESIGN - PROPER ANALYSIS

**Date:** 2026-02-22 17:15 GMT+7  
**Status:** Based on actual code investigation  
**Author:** Nox

---

## 1️⃣ TP/SL TARGETS - ANSWER FOUND IN `tp_sl_retracement.py`

### **Current Implementation: DYNAMIC**

TP/SL is NOT static 1.5%/-1.0%. It's **DYNAMIC per signal**, calculated by:

#### **TP Target (Fibonacci-Based):**
```
1. Scans recent 100 bars for swing high/low
2. Builds Fibonacci retracement levels (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
3. For LONG: Selects deepest fib level ABOVE entry (prefers 0.786, 0.618, 0.5...)
4. For SHORT: Selects deepest fib level BELOW entry
5. Result: TP varies by signal, based on recent price structure
```

**Example:**
```
LONG Signal:
- Entry: $42,500
- Recent Swing High: $43,000
- Recent Swing Low: $41,500
- Fib Levels: [41500, 41930, 42190, 42250, 42440, 42750, 43000]
- TP Selected: $42,750 (0.618 ratio, deepest fib above entry)
- Achieved R:R: (42750-42500)/(42500-41500) = 250/1000 = 1:4 RR
```

#### **SL Target (ATR-Based):**
```
1. Calculates Average True Range (14 bars)
2. SL = Entry ± (ATR × ATR_Multiplier)
3. ATR Multiplier: default 1.0 (tightened)
4. Result: SL varies by volatility
```

**Example (same LONG):**
```
- ATR (14-bar): 80 points
- ATR Multiplier: 1.0
- SL = 42500 - (80 × 1.0) = 42420
- Risk Distance: 42500 - 42420 = 80 points
- Achieved R:R: 250 / 80 = 1:3.125
```

### **Conclusion on Question 1:**
✅ **TP/SL is DYNAMIC** (fibonacci + ATR)  
✅ **Each signal has different targets** based on market structure & volatility  
✅ **Smart Filter already implements this properly**  

**For PEC:** Store actual TP/SL from `calculate_tp_sl()` output, don't hardcode 1.5%/-1.0%

---

## 2️⃣ EXIT BARS - RECOMMENDATION

### **Current Issue:**
- `pec_engine.py` uses hardcoded `max_bars=20` (same for all timeframes)
- This doesn't make sense for HTF (High Time Frame)

### **Your Proposal (Makes Sense):**
```
15min timeframe: 15 bars × 15min = 225 min ≈ 3.75 hours
30min timeframe: 10 bars × 30min = 300 min ≈ 5 hours
1h timeframe:     5 bars × 1h    = 5 hours
```

### **My Analysis & Recommendation:**

**Option A: Hold Time Based (Your Proposal)**
```python
max_bars_by_tf = {
    "3min": 20,    # 60 min = 1 hour
    "5min": 12,    # 60 min = 1 hour
    "15min": 15,   # 225 min = 3.75 hours
    "30min": 10,   # 300 min = 5 hours
    "1h": 5,       # 5 hours
}
```
✅ **Pros:** Consistent hold time across TF, makes sense operationally  
✅ **Cons:** None really  

**Option B: Bars Based on Market Strength**
```python
# Adaptive: if RR is high (>1:3), use fewer bars
# if RR is low (<1:2), use more bars
if achieved_rr >= 3:
    max_bars = 5    # Confident signal, exit quick
elif achieved_rr >= 2:
    max_bars = 10
else:
    max_bars = 15   # Lower RR, give it more time
```
✅ **Pros:** Matches quality of signal (high RR = quicker exit)  
❓ **Cons:** More complex, might be overthinking

**Option C: Hybrid (Best for Smart Filter)**
```python
# Start with TF-based, but cap based on RR
if tf == "15min":
    base_bars = 15
elif tf == "30min":
    base_bars = 10
elif tf == "1h":
    base_bars = 5
else:
    base_bars = 20

# Adjustment: if RR is poor, extend hold
if achieved_rr < 1.5:
    max_bars = base_bars + 3  # More time for weak signals
else:
    max_bars = base_bars
```

### **My Recommendation for Smart Filter:**
**Use Option A (Your Proposal)** - it's clean and makes operational sense.

Rationale:
- Keeps hold time ~3.75-5 hours regardless of TF
- Aligns with daily/swing trading mindset
- Easy to understand and track
- Reduces "noise" from short-term chop

**Implementation:**
```python
MAX_BARS_BY_TF = {
    "3min": 20,
    "5min": 12,
    "15min": 15,
    "30min": 10,
    "1h": 5
}

max_bars = MAX_BARS_BY_TF.get(timeframe, 20)
```

---

## 3️⃣ RISK-REWARD RATIO - DISCUSSION

### **Current State:**
- Smart Filter calculates `achieved_rr` from fibonacci TP + ATR SL
- Example: 1:3.125 RR (from Fibonacci)
- This is **per-signal based on market conditions**

### **Your Question:**
Should we:
- **Option A:** Accept dynamic RR (what happens now)?
- **Option B:** Enforce minimum RR threshold (e.g., only fire if RR ≥ 1:2)?
- **Option C:** Target fixed RR ratio (e.g., all signals aim for 1:3)?

### **Analysis:**

**Option A: Accept Dynamic (Current)**
```
Pro:
- More signals fired
- Captures opportunities with lower RR
- Market dictates the ratio

Con:
- Inconsistent profitability
- Low RR = higher win rate needed (1:1 needs 50%+ wins)
```

**Option B: Enforce Minimum Threshold (Recommended)**
```
Example: Min RR 1:2

if achieved_rr < 2.0:
    DO_NOT_FIRE_SIGNAL()
    
Pro:
- Higher quality signals
- Better profitability buffer
- Reduces noise

Con:
- Fewer signals = longer wait for batch completion
```

**Option C: Target Fixed RR (Not Realistic)**
```
Pro:
- Consistent targets
Con:
- Market won't cooperate
- Fibonacci calculations don't guarantee exact RR
- Would require constant TP/SL adjustment
- NOT practical
```

### **My Strong Recommendation:**

**Option B + Monitoring:**

```python
MIN_ACCEPTED_RR = 1.5  # Can be tuned

# In SmartFilter, after calculate_tp_sl():
achieved_rr = result.get('achieved_rr')

if achieved_rr is None or achieved_rr < MIN_ACCEPTED_RR:
    # Don't fire - quality too low
    skip_signal()
    print(f"[FILTERED] RR too low: {achieved_rr} < {MIN_ACCEPTED_RR}")
else:
    # Fire signal - quality acceptable
    fire_signal()
    print(f"[FIRED] RR acceptable: {achieved_rr}")
```

**Why?**
1. Profitability improves with higher RR
2. Win rate can be 50% with RR 1:2 and still be profitable
3. Reduces false positives (filters out weak setups)
4. Aligns with "Smart Filter" philosophy (quality over quantity)

---

## 📋 PROPOSED PEC PARAMETERS (FINAL)

### **For Storage in signals_fired.jsonl:**
```json
{
  "uuid": "23dee4d1-...",
  "symbol": "BTC-USDT",
  "timeframe": "15min",
  "signal_type": "LONG",
  "fired_time_utc": "2026-02-22T15:30:00Z",
  "entry_price": 42500.00,
  "tp_target": 42750.00,
  "sl_target": 42420.00,
  "tp_pct": 0.588,         # (42750-42500)/42500 = 0.588%
  "sl_pct": -0.188,        # (42420-42500)/42500 = -0.188%
  "achieved_rr": 3.125,    # (42750-42500)/(42500-42420) = 250/80 = 3.125
  "fib_ratio": 0.618,      # Which fib level was TP at
  "atr_value": 80.0,       # ATR at signal time
  "score": 18,
  "confidence": 90.0,
  "route": "LONG_CONFIRMED",
  "regime": "UPTREND"
}
```

### **For PEC Backtesting:**
```python
# Dynamic params by timeframe
PEC_CONFIG = {
    "15min": {
        "max_bars": 15,
        "min_rr": 1.5,
    },
    "30min": {
        "max_bars": 10,
        "min_rr": 1.5,
    },
    "1h": {
        "max_bars": 5,
        "min_rr": 1.5,
    }
}

# When running PEC:
for signal in signals:
    tf = signal["timeframe"]
    max_bars = PEC_CONFIG[tf]["max_bars"]
    
    # Use actual TP/SL from signal (not hardcoded)
    tp_target = signal["tp_target"]
    sl_target = signal["sl_target"]
    
    # Run validation
    result = validate_signal_pec(
        signal=signal,
        tp_target=tp_target,
        sl_target=sl_target,
        max_bars=max_bars
    )
```

---

## ✅ FINAL RECOMMENDATIONS (Summary)

### **1. TP/SL Targets**
✅ **Keep DYNAMIC** (Fibonacci + ATR)  
✅ **Store actual TP/SL** from `calculate_tp_sl()`  
✅ **Use in PEC** - don't hardcode percentages

### **2. Exit Bars (Max Hold)**
✅ **Use TF-based approach** (your proposal):
- 15min: 15 bars (3.75 hrs)
- 30min: 10 bars (5 hrs)
- 1h: 5 bars (5 hrs)

### **3. Risk-Reward Ratio**
✅ **Enforce minimum RR threshold** (1.5:1 minimum)  
✅ **Filter out weak RR signals** before firing  
✅ **Log when filtered** (track false positives prevented)

### **4. Signal Storage**
✅ **Create signals_fired.jsonl** with all complete data  
✅ **Include:** achieved_rr, fib_ratio, atr_value  
✅ **Use in PEC:** Load actual TP/SL, don't recalculate

---

## 🔧 FILES TO MODIFY

1. **`signal_store.py`** (NEW)
   - Store signals to JSONL with all fields

2. **`main.py`** (MODIFY)
   - After `calculate_tp_sl()`, add RR check
   - If RR < MIN_ACCEPTED_RR, skip signal
   - Store complete signal to JSONL

3. **`pec_engine.py`** (MODIFY)
   - Accept max_bars parameter (not hardcoded)
   - Accept tp/sl from signal (not calculate from percentages)

4. **`pec_backtest.py`** (MODIFY)
   - Load from signals_fired.jsonl
   - Use PEC_CONFIG for timeframe-based max_bars
   - Report achieved_rr stats

---

## 🎯 NEXT STEP

Before I code, confirm:
1. **Min RR threshold:** 1.5? Or different? (I recommend 1.5)
2. **Max bars approach:** Use your TF-based proposal?
3. **Storage location:** `/smart-filter-v14-main/signals_fired.jsonl`?

Once you confirm, I'll build the enhanced PEC system properly. 🚀

