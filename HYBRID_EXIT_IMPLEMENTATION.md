# HYBRID EXIT MECHANISM - Implementation Guide

**This document shows EXACTLY what to change for hybrid exit tracking in Batch 1.**

---

## Phase 1: Add Exit Reason Tracking (IMMEDIATE)

### Change 1: pec_engine.py - Return exit_reason

**Current code (pec_engine.py, find_realistic_exit function):**
```python
return {
    'exit_price': final_close,
    'exit_reason': 'TIMEOUT',  # ← Already there!
    'exit_idx': entry_idx + len(df_future),
    'mfe': mfe,
    'mae': mae
}
```

✅ **Already implemented!** The engine already tracks:
- `TP` (Take Profit hit)
- `SL` (Stop Loss hit)
- `TIMEOUT` (Max bars exceeded)

### Change 2: pec_backtest_v2.py - Include exit_reason in CSV

**Modify run_pec_backtest_v2():**

```python
# In the result dict being appended to results list, ADD:

result = {
    'symbol': symbol,
    'timeframe': tf,
    'signal_type': signal_type,
    'entry_price': entry_price,
    'tp_target': tp_price,
    'sl_target': sl_price,
    'exit_price': exit_result['exit_price'],
    'exit_reason': exit_result['exit_reason'],  # ← ADD THIS
    'pnl_pct': pnl_pct,
    'result': 'WIN' if is_win else 'LOSS',
    'fired_time': signal.get('fired_time_utc'),
    'achieved_rr': signal.get('achieved_rr'),
    'score': signal.get('score'),
    'confidence': signal.get('confidence'),
    'mfe': exit_result.get('mfe'),  # ← ADD THIS
    'mae': exit_result.get('mae'),  # ← ADD THIS
}
```

**This gives you CSV columns:**
```
...exit_price, exit_reason, pnl_pct, mfe, mae, result...
```

---

## Phase 2: Optional - Switch to ATR-Based TP/SL

### If Batch 1 shows <50% win rate, apply this change:

**File: tp_sl_retracement.py**

**REPLACE this block** (around line 160-200):

```python
# OLD: Fibonacci-based TP selection
if price_range > 0 and fib_levels is not None:
    if dir_up:
        candidates = [(r, fib_levels[f"{r:.3f}"]) for r in fib_ratios_preferred_deep 
                      if fib_levels.get(f"{r:.3f}", float('nan')) > entry_f]
        if candidates:
            chosen_ratio, chosen_val = max(candidates, key=lambda x: x[1])
            tp = float(chosen_val)
            source = "fib_deep"
    # ... more Fibonacci logic ...
```

**WITH this block** (ATR-based):

```python
# NEW: ATR-based 2:1 RR targeting (standard algo trading)
atr_mult_tp = 2.0  # Always 2:1 ratio from entry

if dir_up:  # LONG
    sl_candidate = entry_f - (atr_multiplier * atr) if atr > 0 else entry_f * (1 - 0.01)
    sl = float(max(recent_low, sl_candidate))  # SL can't be below recent low
    
    tp = entry_f + (atr_mult_tp * atr)  # TP = Entry + (2 × ATR)
    source = "atr_based_2to1"
    chosen_ratio = 2.0
    
elif dir_down:  # SHORT
    sl_candidate = entry_f + (atr_multiplier * atr) if atr > 0 else entry_f * (1 + 0.01)
    sl = float(min(recent_high, sl_candidate))  # SL can't be above recent high
    
    tp = entry_f - (atr_mult_tp * atr)  # TP = Entry - (2 × ATR)
    source = "atr_based_2to1"
    chosen_ratio = 2.0
```

**Key changes:**
- SL distance = 1 × ATR (risk amount)
- TP distance = 2 × ATR (2:1 reward)
- Always results in 2:1 RR by design
- Volatility adaptive (high ATR = bigger targets)

---

## Phase 3: Add Exit Reason Statistics to Batch 1 Report

### Create this script: `batch1_analysis.py`

```python
#!/usr/bin/env python3
"""
batch1_analysis.py
Analyze Batch 1 results focusing on exit reason distribution.
"""

import pandas as pd
import sys

def analyze_batch1(csv_file='batch1_results.csv'):
    """Analyze exit reasons and profitability."""
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Results file not found: {csv_file}")
        return
    
    total = len(df)
    if total == 0:
        print("❌ No results to analyze")
        return
    
    # Overall metrics
    wins = (df['result'] == 'WIN').sum()
    losses = (df['result'] == 'LOSS').sum()
    win_rate = wins / total * 100
    avg_pnl = df['pnl_pct'].mean()
    total_pnl = df['pnl_pct'].sum()
    
    # Exit reason breakdown
    exit_reasons = df['exit_reason'].value_counts()
    
    # By exit reason
    tp_trades = df[df['exit_reason'] == 'TP']
    sl_trades = df[df['exit_reason'] == 'SL']
    timeout_trades = df[df['exit_reason'] == 'TIMEOUT']
    
    print("\n" + "="*70)
    print("BATCH 1 RESULTS ANALYSIS")
    print("="*70)
    
    # Overall
    print(f"\n📊 Overall Performance:")
    print(f"  Total Signals: {total}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg PnL: {avg_pnl:+.2f}%")
    print(f"  Total PnL: {total_pnl:+.2f}%")
    
    # Exit reasons
    print(f"\n🎯 Exit Reason Distribution:")
    for reason, count in exit_reasons.items():
        pct = count / total * 100
        reason_trades = df[df['exit_reason'] == reason]
        reason_wr = (reason_trades['result'] == 'WIN').sum() / count * 100
        reason_avg = reason_trades['pnl_pct'].mean()
        
        print(f"  {reason:8s}: {count:3d} trades ({pct:5.1f}%) | WR: {reason_wr:5.1f}% | Avg PnL: {reason_avg:+.2f}%")
    
    # Target metrics
    print(f"\n✅ Target Metrics:")
    tp_pct = len(tp_trades) / total * 100 if len(tp_trades) > 0 else 0
    sl_pct = len(sl_trades) / total * 100 if len(sl_trades) > 0 else 0
    timeout_pct = len(timeout_trades) / total * 100 if len(timeout_trades) > 0 else 0
    
    print(f"  TP exits: {tp_pct:.1f}% (target: 60-70%) {'✓' if 60 <= tp_pct <= 70 else '✗'}")
    print(f"  SL exits: {sl_pct:.1f}% (target: 20-30%) {'✓' if 20 <= sl_pct <= 30 else '✗'}")
    print(f"  TIMEOUT:  {timeout_pct:.1f}% (target: 5-10%) {'✓' if 5 <= timeout_pct <= 10 else '✗'}")
    
    # By timeframe
    print(f"\n📈 By Timeframe:")
    for tf in df['timeframe'].unique():
        tf_trades = df[df['timeframe'] == tf]
        tf_wins = (tf_trades['result'] == 'WIN').sum()
        tf_wr = tf_wins / len(tf_trades) * 100
        tf_avg = tf_trades['pnl_pct'].mean()
        print(f"  {tf:6s}: {len(tf_trades):3d} trades | WR: {tf_wr:5.1f}% | Avg: {tf_avg:+.2f}%")
    
    # Verdict
    print(f"\n" + "="*70)
    if win_rate >= 55:
        print(f"✅ BATCH 1 PASSED - Win rate {win_rate:.1f}% >= 55% target")
        print(f"   Recommendation: Approve for Batch 2 (150 signals)")
    elif win_rate >= 50:
        print(f"⚠️  BATCH 1 MARGINAL - Win rate {win_rate:.1f}% is borderline")
        print(f"   Recommendation: Review TP/SL logic, consider optimization")
    else:
        print(f"❌ BATCH 1 FAILED - Win rate {win_rate:.1f}% < 50% threshold")
        print(f"   Recommendation: Switch to ATR-based TP/SL, retry Batch 1")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'batch1_results.csv'
    analyze_batch1(csv_file)
```

---

## Exact Batch 1 Execution Steps

### When 50 signals accumulated:

```bash
cd smart-filter-v14-main

# 1. Run backtest
python3 -c "
from pec_backtest_v2 import run_pec_backtest_v2
from data_fetcher import get_ohlcv_kucoin, get_local_wib

result = run_pec_backtest_v2(
    get_ohlcv=get_ohlcv_kucoin,
    get_local_wib=get_local_wib,
    output_csv='batch1_results.csv'
)
print(f'Backtest complete: {result[\"output_file\"]}')
"

# 2. Analyze results
python3 batch1_analysis.py batch1_results.csv

# 3. Review CSV
open batch1_results.csv  # Columns: symbol, timeframe, entry_price, tp_target, 
                         # sl_target, exit_price, exit_reason, pnl_pct, result, ...
```

---

## Expected Batch 1 CSV Format

```csv
symbol,timeframe,signal_type,entry_price,tp_target,sl_target,exit_price,exit_reason,pnl_pct,result,fired_time,achieved_rr,score,confidence,mfe,mae
BTC-USDT,15m,LONG,42500.50,42750.25,42420.00,42750.25,TP,0.59,WIN,2026-02-22T15:30:00Z,3.125,18,90.0,0.78,-0.35
BTC-USDT,15m,SHORT,42450.00,42150.00,42550.00,42550.00,SL,-0.47,LOSS,2026-02-22T15:45:00Z,2.857,17,88.0,0.24,-0.58
ETH-USDT,30m,LONG,2520.50,2560.75,2480.00,2560.75,TP,1.60,WIN,2026-02-22T16:00:00Z,2.500,16,85.0,1.75,-0.42
SOL-USDT,1h,LONG,155.50,160.25,151.50,154.50,TIMEOUT,-0.64,LOSS,2026-02-22T17:00:00Z,1.875,15,80.0,0.33,-1.28
```

---

## Decision Tree

**After Batch 1 Results:**

```
┌─────────────────────────────────────────────────┐
│ Run Batch 1 (50 signals)                        │
├─────────────────────────────────────────────────┤
│ Check: Win Rate?                                │
│                                                 │
├─→ WR ≥ 55% ✅                                  │
│   └─ PASS: Fibonacci is working                │
│      Action: Proceed to Batch 2 (150 signals)  │
│                                                 │
├─→ 50% ≤ WR < 55% ⚠️                            │
│   └─ MARGINAL: Need optimization               │
│      Check: TP/SL distribution reasonable?     │
│      Action: Switch to ATR-based, retry Batch1 │
│                                                 │
└─→ WR < 50% ❌                                  │
    └─ FAIL: Fibonacci not working for this set  │
       Check: Exit reason distribution            │
       Action: Switch to ATR-based, retry Batch 1│
```

---

## Summary: 3-Step Implementation

### ✅ Step 1: Minimal Change (Phase 1)
Modify pec_backtest_v2.py to add exit_reason to CSV.
**Time:** 10 minutes
**Result:** Track TP/SL/TIMEOUT distribution

### ⚠️ Step 2: Conditional (Phase 2)
If Batch 1 <55% WR, modify tp_sl_retracement.py to use ATR-based.
**Time:** 20 minutes
**Result:** Switch to industry-standard TP/SL

### 📊 Step 3: Analysis
Run batch1_analysis.py to validate exit reason distribution.
**Time:** 5 minutes
**Result:** Clear verdict on profitability

---

**Ready to implement Phase 1 immediately?** I can modify pec_backtest_v2.py right now to add exit_reason tracking to Batch 1 results.

[[reply_to_current]]
