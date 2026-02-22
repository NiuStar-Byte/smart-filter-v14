# SuperGK (Super Gatekeeper) - ANALYSIS & ISSUES

**Date:** 2026-02-22 17:50 GMT+7  
**Status:** 🔴 BROKEN - Currently Disabled  
**Author:** Nox

---

## 🔴 CRITICAL ISSUE

**SuperGK is blocking 100% of signals** when enabled.

**Symptom:** All signals rejected, none reach Telegram  
**Root Cause:** Logic flaw in composite score calculation  
**Current Status:** Disabled (bypassed)

---

## 📋 HOW SuperGK WORKS (Current Implementation)

### **Step 1: Fetch Market Data**
```python
buy_wall = orderbook_result['buy_wall']      # BUY orders at best bid
sell_wall = orderbook_result['sell_wall']    # SELL orders at best ask
bid_density = density_result['bid_density']  # % of volume in top bids
ask_density = density_result['ask_density']  # % of volume in top asks
```

### **Step 2: Calculate Bias from Data**
```python
wall_delta = buy_wall - sell_wall
wall_pct = abs(wall_delta) / (buy_wall + sell_wall) × 100

density_diff = bid_density - ask_density
density_pct = abs(density_diff)

# Determine sign (LONG or SHORT or NEUTRAL)
wall_sign = "LONG" if wall_delta > 0 else "SHORT" if wall_delta < 0 else "NEUTRAL"
density_sign = "LONG" if density_diff > 0 else "SHORT" if density_diff < 0 else "NEUTRAL"
```

### **Step 3: Compare to Signal Bias**
```python
# Check if market data supports signal direction
wall_factor = sign_to_factor(wall_sign, signal_bias)
# Returns: 1.0 if aligned, -1.0 if opposite, 0.0 if neutral

density_factor = sign_to_factor(density_sign, signal_bias)
# Same logic: 1.0 aligned, -1.0 opposite, 0.0 neutral
```

### **Step 4: Calculate Composite Score**
```python
wall_score = wall_pct / 100.0        # 0.0 to 1.0
density_score = density_pct / 100.0  # 0.0 to 1.0

composite = (
    wall_weight * wall_factor * wall_score +
    density_weight * density_factor * density_score
)
# Defaults: wall_weight=0.25, density_weight=0.75

# Return: composite >= composite_threshold (0.01)
```

---

## 🔍 WHY IT'S BROKEN

### **Problem 1: Opposite Indicator Kills Signal**

```
Scenario: LONG signal fires
- SmartFilter says: LONG (strong filters)
- Wall: Shows SHORT (sell_wall > buy_wall) → wall_factor = -1.0
- Density: Shows NEUTRAL (bid/ask similar) → density_factor = 0.0

Calculation:
composite = 0.25 × (-1.0) × 0.8 + 0.75 × 0.0 × 0.2
composite = -0.20 + 0.0
composite = -0.20

Result: -0.20 >= 0.01? NO → SIGNAL REJECTED ❌
```

**The Problem:** One conflicting indicator completely kills the signal, even if:
- SmartFilter is very confident
- The other indicator is neutral
- Both indicators are just momentary noise

### **Problem 2: Default Thresholds Are Wrong**

**wall_pct_threshold = 1.0%** = Very high bar
- Most markets have ±0.5-2% wall imbalance constantly
- A 1% threshold means only extreme wall imbalance counts
- This makes wall effectively useless

**composite_threshold = 0.01** = Impossibly low
- With wall_weight=0.25, density_weight=0.75
- Any single neutral factor makes reaching 0.01 impossible
- Example: 0.25×1.0×0.01 + 0.75×0.0×1.0 = 0.0025 < 0.01 ❌

### **Problem 3: Weights Are Imbalanced**

```
Wall weight: 0.25 (25%)
Density weight: 0.75 (75%)
```

This means:
- Density dominates (75% of score)
- Wall contribution is tiny (25%)
- But they're weighted equally in logic (both can be -1.0 or +1.0)

**Better approach:** Equal weight (0.5 each) or make weights adjustable by TF

### **Problem 4: Zero Factor is Too Harsh**

When an indicator is NEUTRAL (sign_to_factor returns 0.0):
```python
wall_weight × 0.0 × wall_score = 0.0  # Contributes nothing
density_weight × 0.0 × density_score = 0.0  # Contributes nothing
```

**Result:** Neutral is treated as "adds nothing" instead of "doesn't hurt"

This is backwards. Neutral should = no penalty, not no contribution.

---

## 📊 EXAMPLE: WHY 100% SIGNALS FAIL

### **Real Scenario (Common Market Conditions)**

```
BTC-USDT Market at 10:30 UTC:
- Buy wall: $1,200,000
- Sell wall: $1,100,000
- Wall delta: +$100,000 (shows LONG bias)
- Wall %: 4.3% (shows alignment)

But bid density: 22%
And ask density: 28%
Density delta: -6% (shows SHORT bias) ← OPPOSITE!

Signal from SmartFilter: LONG (very strong, score 19/20)

SuperGK Calculation:
- wall_sign = "LONG" → wall_factor = +1.0
- density_sign = "SHORT" → density_factor = -1.0  ← PROBLEM!
- composite = 0.25×(+1.0)×0.043 + 0.75×(-1.0)×0.06
- composite = +0.01075 - 0.045
- composite = -0.034

Result: -0.034 >= 0.01? NO → REJECTED ❌

Even though:
- SmartFilter is 95% confident (19/20)
- Orderbook shows LONG (4.3% imbalance)
- Density shows slight SHORT (6% imbalance)
  
Rejected because of ONE metric!
```

---

## ✅ OPTIONS TO FIX

### **Option A: Disable Permanently (RECOMMENDED)**

```python
super_gk_ok = True  # Always bypass
```

**Pros:**
- SmartFilter filters are already quite good
- No more 100% signal rejection
- Simpler system (fewer gatekeepers)

**Cons:**
- Lose orderbook/density validation entirely
- No real-time market structure checks

### **Option B: Fix SuperGK Logic**

```python
# Fix 1: Remove negative factors (use 0 or 1, never -1)
def sign_to_factor(sign, desired):
    if sign == desired:
        return 1.0
    else:
        return 0.0  # Neutral OR opposite = no help (not penalty)

# Fix 2: Lower thresholds significantly
composite_threshold = -0.25  # Allow one opposed metric + neutral other

# Fix 3: Rebalance weights
wall_weight = 0.5  # 50%
density_weight = 0.5  # 50%

# Fix 4: Add logging to debug
print(f"[SuperGK] wall={wall_factor} density={density_factor} composite={composite:.3f}")
```

**Pros:**
- Keeps orderbook/density checks
- More forgiving (won't kill good signals)
- Can tune via env vars

**Cons:**
- Requires careful testing
- Thresholds need optimization
- More complex

### **Option C: Rewrite SuperGK from Scratch**

```python
# New approach: Voting system instead of composite
def supergk_voting(bias, orderbook_result, density_result):
    """
    Simple voting system:
    - Each metric votes: YES, NO, or ABSTAIN
    - Need >50% YES (ignoring ABSTAIN)
    - Much simpler logic
    """
    votes = []
    
    # Orderbook vote (less weight: market makers manipulate)
    if orderbook_result.get('wall_pct', 0) > 2.0:
        wall_vote = "LONG" if orderbook_result.get('wall_delta', 0) > 0 else "SHORT"
        votes.append(("orderbook", wall_vote == bias, weight=0.3))
    else:
        votes.append(("orderbook", None, weight=0.3))  # ABSTAIN
    
    # Density vote (more weight: real trader sentiment)
    bid = density_result.get('bid_density', 0)
    ask = density_result.get('ask_density', 0)
    if abs(bid - ask) > 10.0:  # Clear bias
        density_vote = "LONG" if bid > ask else "SHORT"
        votes.append(("density", density_vote == bias, weight=0.7))
    else:
        votes.append(("density", None, weight=0.7))  # ABSTAIN
    
    # Simple majority
    yes_votes = sum(w for _, v, w in votes if v is True)
    return yes_votes >= 0.5  # Need 50% YES
```

**Pros:**
- Much simpler logic
- Harder to break
- Easy to understand and debug

**Cons:**
- Completely different approach
- Needs rewriting + testing
- More time investment

---

## 🎯 RECOMMENDATION

**For immediate deployment:** Option A (disable permanently)

**Reasoning:**
1. SmartFilter is already doing heavy lifting (60+ filters)
2. SuperGK adds complexity without obvious benefit
3. Perfect is enemy of good - get signals flowing first
4. Can add SuperGK back after testing

**Timeline:**
- Disable (DONE): 2 minutes
- Test signals flow: 30 minutes (wait for a batch)
- Fix SuperGK properly: 2-3 hours (Option B or C)

---

## 📝 NEXT STEPS

**Immediate:**
- ✅ Disabled SuperGK (dde003c commit)
- ⏳ Monitor signals flowing to Telegram
- ⏳ Verify signal rate returns to normal

**Next 1-2 hours:**
- [ ] Collect fresh signals
- [ ] Verify RR filtering works
- [ ] Start Batch 1 backtest

**Later (optional):**
- [ ] Investigate Option B (fix logic)
- [ ] Or implement Option C (rewrite)
- [ ] Add back with proper tuning

---

## 🔍 DEBUGGING CHECKLIST (When Re-enabling)

If you ever want to re-enable SuperGK, enable with logging:

```python
# In main.py, 15min block:
super_gk_ok = super_gk_aligned(
    bias, orderbook_result, density_result
)

# Add after:
print(f"[SuperGK] Signal={bias} | Orderbook={orderbook_result} | Density={density_result}", flush=True)
print(f"[SuperGK] Result={super_gk_ok}", flush=True)
```

Then check logs for patterns:
- Which signals are being rejected?
- What orderbook state causes rejection?
- What density patterns cause rejection?

This will help identify the exact problem.

---

## ✨ CONCLUSION

**SuperGK is well-intentioned but fundamentally broken.**

Current implementation:
- ❌ Blocks opposite indicators as hard failures
- ❌ Thresholds are unrealistic
- ❌ Weights are imbalanced
- ❌ Zero contribution = penalty (backwards logic)

**Best path forward:** Keep disabled, build proper version later.

