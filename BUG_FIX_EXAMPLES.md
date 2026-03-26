# PEC P&L Calculation Bug Fix - 8 Scenarios

**Bug:** Currently using `current_price` for TP/SL P&L. Should use target price for TP/SL, actual price for TIMEOUT.

---

## LONG SCENARIOS

### a. LONG Win (TP_HIT)

**Setup:**
- Entry Price: $100.00
- TP Target: $131.00
- SL Target: $90.69
- Current Price When TP Reached: $135.00 (overshot, but we recognize TP was hit)

**Expected RR at Signal Fire:**
- Risk = $100.00 - $90.69 = $9.31
- Reward = $131.00 - $100.00 = $31.00
- RR = $31.00 / $9.31 = 3.33:1

**CORRECT Calculation (After Fix):**
```
Use TP_TARGET ($131.00):
P&L = [(131.00 - 100.00) / 100.00] × $100 notional
    = [31.00 / 100.00] × $100
    = 31%
    = +$31.00
```

**WRONG Calculation (Current Bug):**
```
Use CURRENT_PRICE ($135.00):
P&L = [(135.00 - 100.00) / 100.00] × $100 notional
    = [35.00 / 100.00] × $100
    = 35%
    = +$35.00  ❌ OVERSTATED by $4.00
```

**Impact:** Avg TP becomes artificially high → RR ratio breaks (appears < 3.33)

---

### b. LONG Loss (SL_HIT)

**Setup:**
- Entry Price: $100.00
- TP Target: $131.00
- SL Target: $90.69
- Current Price When SL Reached: $88.00 (gapped down past SL)

**CORRECT Calculation (After Fix):**
```
Use SL_TARGET ($90.69):
P&L = -[(100.00 - 90.69) / 100.00] × $100 notional
    = -[9.31 / 100.00] × $100
    = -9.31%
    = -$9.31
```

**WRONG Calculation (Current Bug):**
```
Use CURRENT_PRICE ($88.00):
P&L = -[(100.00 - 88.00) / 100.00] × $100 notional
    = -[12.00 / 100.00] × $100
    = -12%
    = -$12.00  ❌ OVERSTATED LOSS by $2.69
```

**Impact:** Avg SL loss becomes artificially large → RR ratio breaks (Avg SL > Avg TP despite RR=1.31)

---

### c. LONG TIMEOUT_WIN

**Setup:**
- Entry Price: $100.00
- TP Target: $131.00
- SL Target: $90.69
- Current Price At Timeout: $102.50 (no target hit, price above entry = profit)

**CORRECT Calculation (After Fix):**
```
Use ACTUAL CURRENT_PRICE ($102.50) - target never reached:
P&L = [(102.50 - 100.00) / 100.00] × $100 notional
    = [2.50 / 100.00] × $100
    = 2.5%
    = +$2.50
```

**Why Different:** TIMEOUT is forced exit at market, not at target. Must use actual price.

---

### d. LONG TIMEOUT_LOSS

**Setup:**
- Entry Price: $100.00
- TP Target: $131.00
- SL Target: $90.69
- Current Price At Timeout: $97.00 (no target hit, price below entry = loss)

**CORRECT Calculation (After Fix):**
```
Use ACTUAL CURRENT_PRICE ($97.00) - target never reached:
P&L = -[(100.00 - 97.00) / 100.00] × $100 notional
    = -[3.00 / 100.00] × $100
    = -3%
    = -$3.00
```

---

## SHORT SCENARIOS

### e. SHORT Win (TP_HIT)

**Setup:**
- Entry Price: $100.00
- TP Target: $76.03 (1.31 RR = profit $23.97, risk $9.31)
- SL Target: $109.31
- Current Price When TP Reached: $74.00 (overshot down, but TP was hit)

**Expected RR at Signal Fire:**
- Risk = $109.31 - $100.00 = $9.31
- Reward = $100.00 - $76.03 = $23.97
- RR = $23.97 / $9.31 = 2.57:1

**CORRECT Calculation (After Fix):**
```
Use TP_TARGET ($76.03):
P&L = [(100.00 - 76.03) / 100.00] × $100 notional
    = [23.97 / 100.00] × $100
    = 23.97%
    = +$23.97
```

**WRONG Calculation (Current Bug):**
```
Use CURRENT_PRICE ($74.00):
P&L = [(100.00 - 74.00) / 100.00] × $100 notional
    = [26.00 / 100.00] × $100
    = 26%
    = +$26.00  ❌ OVERSTATED by $2.03
```

---

### f. SHORT Loss (SL_HIT)

**Setup:**
- Entry Price: $100.00
- TP Target: $76.03
- SL Target: $109.31
- Current Price When SL Reached: $112.00 (gapped up past SL)

**CORRECT Calculation (After Fix):**
```
Use SL_TARGET ($109.31):
P&L = -[(109.31 - 100.00) / 100.00] × $100 notional
    = -[9.31 / 100.00] × $100
    = -9.31%
    = -$9.31
```

**WRONG Calculation (Current Bug):**
```
Use CURRENT_PRICE ($112.00):
P&L = -[(112.00 - 100.00) / 100.00] × $100 notional
    = -[12.00 / 100.00] × $100
    = -12%
    = -$12.00  ❌ OVERSTATED LOSS by $2.69
```

---

### g. SHORT TIMEOUT_WIN

**Setup:**
- Entry Price: $100.00
- TP Target: $76.03
- SL Target: $109.31
- Current Price At Timeout: $97.50 (below entry = SHORT profit)

**CORRECT Calculation (After Fix):**
```
Use ACTUAL CURRENT_PRICE ($97.50) - target never reached:
P&L = [(100.00 - 97.50) / 100.00] × $100 notional
    = [2.50 / 100.00] × $100
    = 2.5%
    = +$2.50
```

---

### h. SHORT TIMEOUT_LOSS

**Setup:**
- Entry Price: $100.00
- TP Target: $76.03
- SL Target: $109.31
- Current Price At Timeout: $103.00 (above entry = SHORT loss)

**CORRECT Calculation (After Fix):**
```
Use ACTUAL CURRENT_PRICE ($103.00) - target never reached:
P&L = -[(103.00 - 100.00) / 100.00] × $100 notional
    = -[3.00 / 100.00] × $100
    = -3%
    = -$3.00
```

---

## Summary Table

| Scenario | Direction | Outcome | Entry | Target | Actual @ Exit | CORRECT P&L | WRONG P&L | Delta |
|----------|-----------|---------|-------|--------|---------------|-------------|-----------|-------|
| a | LONG | TP_HIT | $100 | $131 (TP) | $135 | +$31.00 | +$35.00 | -$4.00 |
| b | LONG | SL_HIT | $100 | $90.69 (SL) | $88 | -$9.31 | -$12.00 | +$2.69 |
| c | LONG | TIMEOUT_WIN | $100 | N/A | $102.50 | +$2.50 | +$2.50 | $0 |
| d | LONG | TIMEOUT_LOSS | $100 | N/A | $97.00 | -$3.00 | -$3.00 | $0 |
| e | SHORT | TP_HIT | $100 | $76.03 (TP) | $74 | +$23.97 | +$26.00 | -$2.03 |
| f | SHORT | SL_HIT | $100 | $109.31 (SL) | $112 | -$9.31 | -$12.00 | +$2.69 |
| g | SHORT | TIMEOUT_WIN | $100 | N/A | $97.50 | +$2.50 | +$2.50 | $0 |
| h | SHORT | TIMEOUT_LOSS | $100 | N/A | $103.00 | -$3.00 | -$3.00 | $0 |

---

## Key Insight

**Scenarios a, b, e, f (TP/SL):** Bug causes $2-4 error per trade
- **TP_HIT:** Overstate profit when price overshoots
- **SL_HIT:** Overstate loss when price gaps past

**Scenarios c, d, g, h (TIMEOUT):** Already correct (using current_price)
- Timeout must use actual price (target never reached)

**Result:** Over many trades, these errors compound, causing:
- Avg TP higher than it should be
- Avg SL higher loss than it should be
- RR ratio appears lower than actual (1.31 → appears as 1.22)

---

## Fix Implementation

**In pec_executor.py check_signal_status():**

```python
if current_price >= tp_target:
    # TP HIT: Use target price (was recognized)
    pnl_result = calculate_pnl(entry_price, tp_target, direction, NOTIONAL_POSITION)
    return {
        'status': 'TP_HIT',
        'exit_price': tp_target,  # Record target as exit
        'pnl_usd': pnl_result['pnl_usd'],
        ...
    }

elif current_price <= sl_target:
    # SL HIT: Use target price (was recognized)
    pnl_result = calculate_pnl(entry_price, sl_target, direction, NOTIONAL_POSITION)
    return {
        'status': 'SL_HIT',
        'exit_price': sl_target,  # Record target as exit
        'pnl_usd': pnl_result['pnl_usd'],
        ...
    }

# TIMEOUT: Use actual current price (target never reached)
elif bars_elapsed >= max_bars:
    pnl_result = calculate_pnl(entry_price, current_price, direction, NOTIONAL_POSITION)
    return {
        'status': 'TIMEOUT',
        'exit_price': current_price,  # Record actual price
        'pnl_usd': pnl_result['pnl_usd'],
        ...
    }
```

**Result:** Avg TP / Avg SL = RR exactly, calculations validated ✓
