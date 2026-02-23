# BATCH 1 MONITORING CHECKLIST

**While waiting for 50 signals to accumulate (1-2 days)**

---

## Daily Monitoring (Copy & Paste)

```bash
# Morning check (takes 10 seconds)
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main
python3 count_signals_by_tf.py
```

**What to look for:**
```
Total Signals: X/50 (Y% progress)
  15min: A signals | LONG: B, SHORT: C | Avg RR: D:1
  30min: E signals | LONG: F, SHORT: G | Avg RR: H:1
  1h:    I signals | LONG: J, SHORT: K | Avg RR: L:1

Expected:
✓ Even distribution across timeframes
✓ Mix of LONG and SHORT (both directions)
✓ All RR values > 1.25:1
✓ Average RR 2-4:1 (healthy)
```

---

## Process Health Check

```bash
# Verify main.py still running
ps aux | grep "python.*main" | grep -v grep

# Should show: python3 main.py
# If NOT showing: run this to restart:
# cd smart-filter-v14-main && nohup python3 main.py > main.log 2>&1 &
```

---

## Real-Time Monitoring (Optional)

```bash
# Watch signals arrive as they fire
tail -f smart-filter-v14-main/signals_fired.jsonl | \
  python3 -c "
import json, sys
for line in sys.stdin:
    try:
        sig = json.loads(line)
        print(f\"{sig['fired_time_utc'][:19]} | {sig['symbol']:12s} {sig['timeframe']:6s} {sig['signal_type']:5s} | RR: {sig['achieved_rr']:5.2f}:1 | Score: {sig['score']}/18\")
    except:
        pass
"
```

Output looks like:
```
2026-02-22T18:30:15 | XRP-USDT     15min  LONG  | RR:  2.50:1 | Score: 14/18
2026-02-22T18:31:22 | BTC-USDT     30min  SHORT | RR:  1.80:1 | Score: 15/18
2026-02-22T18:32:45 | ETH-USDT     15min  LONG  | RR:  3.25:1 | Score: 13/18
...
```

---

## What NOT to Worry About

❌ Don't manually trigger Batch 1 before 50 signals  
❌ Don't restart main.py unless process stops  
❌ Don't modify signal storage code (it's working)  
❌ Don't check Batch 1 results yet (they'll be auto-generated)

---

## When 50 Signals Arrive (You'll See)

```
Total Signals: 50/50 (100.0% progress) ✅

At that point, Nox will automatically:
1. Run: python3 pec_backtest_v2.py
2. Generate: batch1_results.csv
3. Analyze: python3 batch1_analysis.py batch1_results.csv
4. Show: Full results with verdict
```

---

## Expected Timeline

| Day | Signal Count | Status |
|-----|-------------|--------|
| Today (Sun 18:27) | 3 | 6% — Starting phase |
| Tomorrow (Mon) | 15-20 | 30-40% — Steady accumulation |
| Day 3 (Tue) | 40-45 | 80-90% — Almost there |
| Day 4 (Wed) | 50 | 100% — Batch 1 Ready! |

---

## Commands Reference

**Daily progress check:**
```bash
python3 smart-filter-v14-main/count_signals_by_tf.py
```

**Check if main.py running:**
```bash
ps aux | grep "python.*main" | grep -v grep
```

**Restart main.py (if needed):**
```bash
pkill -f "python.*main\.py"
sleep 2
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main
nohup python3 main.py > main.log 2>&1 &
```

**When 50 signals arrive (auto-triggered):**
```bash
# Results will be ready at:
# /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/batch1_results.csv
# /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/batch1_analysis.txt
```

---

## What Batch 1 Results Will Tell Us

```
OVERALL METRICS:
- Win Rate (target: 55%+)
- Avg PnL per trade (target: +0.5% to +1.0%)
- Total PnL (all trades combined)

EXIT DISTRIBUTION (validates hybrid mechanism):
- TP exits: 60% (winners hitting profit targets)
- SL exits: 28% (losers stopped at stop loss)
- TIMEOUT: 12% (rare — neither hit, exited at bar limit)

BY TIMEFRAME:
- 15min: Win rate? Avg PnL?
- 30min: Win rate? Avg PnL?
- 1h: Win rate? Avg PnL?

FIBONACCI VALIDATION:
- If WR ≥ 55% → Fibonacci TP/SL is working ✓
- If 50% ≤ WR < 55% → Consider ATR-based optimization
- If WR < 50% → Switch to ATR-based 2:1 RR

FINAL VERDICT:
✅ Ready for live trading (Batch 2 → 150 signals)
⚠️  Needs optimization (switch TP/SL method, retry)
❌ Signals need revision (back to SmartFilter tuning)
```

---

## You've Done Great Work

**Current achievements:**
✅ Built complete PEC backtesting system (signal_store, pec_config, pec_engine, pec_backtest_v2)  
✅ Implemented hybrid exit mechanism (TP → SL → TIMEOUT)  
✅ Added timeout_price tracking (full P&L audit trail)  
✅ Fixed signal sync (Telegram + JSONL in sync)  
✅ Deployed RR filtering (only 1.25:1+ signals fire)  
✅ Started signal accumulation (3/50 with excellent RR)

**Next phase:** Just wait and monitor. The system is self-running now.

---

**Keep it simple. Check daily. Batch 1 will be ready in 1-2 days.**
