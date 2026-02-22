# BATCH 1 OPERATIONS GUIDE

**Status:** ✅ READY TO GO (as of 2026-02-22 10:44 GMT+7)

---

## Goal
Execute first validation backtest on 50 fired signals with:
- RR ≥ 1.25:1 (filtered)
- Dynamic max_bars per timeframe (15m=15, 30m=10, 1h=5)
- Signal's stored TP/SL prices
- Complete audit trail in CSV

**Expected Timeline:** 1-3 days (accumulate 50 signals)

---

## Monitor Signal Accumulation

### Real-time monitoring:
```bash
tail -f smart-filter-v14-main/signals_fired.jsonl | jq .
```

### Check signal count:
```bash
wc -l smart-filter-v14-main/signals_fired.jsonl
# Should show 50 or more when ready for Batch 1
```

### Sample signal inspection:
```bash
head -1 smart-filter-v14-main/signals_fired.jsonl | jq .
# Shows: uuid, symbol, timeframe, signal_type, entry_price, tp_target, sl_target, achieved_rr, etc.
```

### Daily summary:
```bash
python3 smart-filter-v14-main/count_signals_by_tf.py
# Shows: 15min: 15 signals, 30min: 18 signals, 1h: 20 signals (example)
```

---

## Trigger Batch 1 (When Signal Count ≥ 50)

### 1. Validate readiness:
```bash
cd smart-filter-v14-main
python3 validate_batch1_ready.py
# Should output: ✅ BATCH 1 READY TO GO!
```

### 2. Run backtest:
```bash
cd smart-filter-v14-main
python3 -c "
from pec_backtest_v2 import run_pec_backtest_v2
from data_fetcher import get_ohlcv_kucoin, get_local_wib

# Run with KuCoin API
result = run_pec_backtest_v2(
    get_ohlcv=get_ohlcv_kucoin,
    get_local_wib=get_local_wib,
    output_csv='batch1_results.csv'
)

print('\n=== BATCH 1 RESULTS ===')
print(f'Processed: {result[\"processed\"]}/{result[\"total_signals\"]}')
print(f'Wins: {result[\"wins\"]} | Losses: {result[\"losses\"]}')
print(f'Win Rate: {(result[\"wins\"] / result[\"processed\"] * 100):.1f}%')
print(f'Total PnL: {result[\"total_pnl\"]:+.2f}%')
print(f'Output: {result[\"output_file\"]}')
"
```

### 3. Review results:
```bash
# CSV report
open batch1_results.csv

# Quick stats
python3 -c "
import pandas as pd
df = pd.read_csv('batch1_results.csv')
print(f'Total signals: {len(df)}')
print(f'Wins: {(df[\"result\"] == \"WIN\").sum()}')
print(f'Losses: {(df[\"result\"] == \"LOSS\").sum()}')
print(f'Win rate: {((df[\"result\"] == \"WIN\").sum() / len(df) * 100):.1f}%')
print(f'Avg PnL: {df[\"pnl_pct\"].mean():+.2f}%')
print(f'\nBy timeframe:')
print(df.groupby('timeframe')[['result', 'pnl_pct']].agg({
    'result': 'count',
    'pnl_pct': 'mean'
}).round(2))
"
```

---

## Expected Metrics (Target Baseline)

| Metric | Target | Notes |
|--------|--------|-------|
| **Win Rate** | 55%+ | Minimum for profitability |
| **Avg PnL** | +0.5% to +1.0% per trade | Quality signal validation |
| **By Timeframe** | Even distribution | No bias to specific TF |
| **TP Hits** | 60%+ | Primary exit |
| **SL Hits** | 20-30% | Stop loss protection working |
| **Timeout** | 10-20% | Extended holds acceptable |

---

## Signal Filtering Details

**Applied Filters (all signals in Batch 1):**
- ✅ RR ≥ 1.25:1 (enforced in main.py + pec_backtest_v2.py)
- ✅ Score ≥ 16 (set in main.py)
- ✅ SmartFilter confidence ≥ 75% (typical for fired signals)
- ✅ All gatekeepers passed (required for fire)

**Output includes:**
- Entry/exit prices
- TP/SL levels (from signal storage)
- Exit reason (TP/SL/TIMEOUT)
- PnL per trade
- MFE/MAE (max favorable/adverse excursion)
- Signal metadata (score, confidence, route, regime)

---

## Files Involved

| File | Purpose | Status |
|------|---------|--------|
| `signal_store.py` | JSONL append/query | ✅ Working |
| `pec_config.py` | Config (RR, max_bars) | ✅ Loaded |
| `main.py` | Signal firing + RR filter | ✅ Live |
| `pec_engine.py` | Dynamic TP/SL + max_bars | ✅ Ready |
| `pec_backtest_v2.py` | JSONL backtest runner | ✅ Ready |
| `signals_fired.jsonl` | Signal storage | ✅ Accumulating |
| `validate_batch1_ready.py` | Pre-flight check | ✅ Passing |

---

## Troubleshooting

### No signals accumulating
- Check: `tail -f signals_fired.jsonl` (should append in real-time)
- Verify: main.py is running with RR filter active
- Check logs for: `create_and_store_signal()` calls

### Backtest fails with "not enough bars"
- Signals near end of OHLCV data may not have future bars
- Expected: ~5-10% signals skipped due to data edge
- Solution: None needed, this is normal

### PnL calculations unrealistic
- Check: Entry/exit prices in CSV
- Verify: TP/SL levels make sense (tp > entry for LONG)
- Review: MFE/MAE columns (should match movement)

### CSV export missing columns
- Ensure pec_backtest_v2.py completed without errors
- Check: batch1_results.csv exists and has rows
- Verify: Columns include: symbol, timeframe, result, pnl_pct, etc.

---

## After Batch 1 Completes

1. **Analysis:** Review win rate, avg PnL, distribution by TF
2. **Decision:** Approve live trading if metrics > targets
3. **Batch 2:** Accumulate 150 signals (optimization phase)
4. **Documentation:** Update SMART_FILTER_DEVELOPMENT_LOG.md

---

## Key Contacts

**Signal Firing:** main.py (SmartFilter core)
**Backtest Runner:** pec_backtest_v2.py (with get_ohlcv dependency)
**Config:** pec_config.py (MIN_ACCEPTED_RR, MAX_BARS_BY_TF)
**Nox Oversight:** Daily monitoring + results reporting

---

**Last Updated:** 2026-02-22 10:44 GMT+7
**Status:** ✅ BATCH 1 UNBLOCKED - Awaiting 50 signals
