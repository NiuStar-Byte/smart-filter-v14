# SMART FILTER SYSTEM - BATCH BACKTEST SCHEDULE

**Owner:** Nox (AI Agent)  
**Last Updated:** 2026-02-22 16:01 GMT+7  
**Status:** 🟡 PENDING (awaiting first signals)

---

## 📋 BATCH STRATEGY

**Trigger:** Only start backtesting when system fires signals on 15m, 30m, 1h HTF  
**Mode:** Sequential batches (not continuous)  
**Validation Window:** Full market session (UTC)  
**Metrics Tracked:** Win rate, false positives, P&L, filter effectiveness

---

## 📊 BATCH SCHEDULE (Trigger-Based)

### **Batch 1: Initial Validation (50 signals fired)**
- **Trigger:** When 50 total signals have been generated
- **Scope:** All 3 timeframes (15m + 30m + 1h)
- **Run Command:** `python pec_backtest.py --start <date> --end <date> --batch 1`
- **Expected Output:** 
  - Win rate baseline
  - Filter effectiveness report
  - Top 3 failing filters (if any)
- **Time:** ~30 min runtime
- **Report:** Saved to `BATCH_1_VALIDATION_REPORT.md`

### **Batch 2: Optimization Window (150 signals fired)**
- **Trigger:** When 150 total signals have been generated (100 additional)
- **Scope:** Deep analysis on all 3 timeframes
- **Run Command:** `python pec_backtest.py --start <date> --end <date> --batch 2`
- **Expected Output:**
  - Win rate trend (compared to Batch 1)
  - Filter combination effectiveness
  - Recommended parameter tuning
- **Time:** ~45 min runtime
- **Report:** Saved to `BATCH_2_OPTIMIZATION_REPORT.md`

### **Batch 3: Monthly Review (300 signals fired)**
- **Trigger:** When 300 total signals have been generated (150 additional)
- **Scope:** Comprehensive analysis + recommendations for next month
- **Run Command:** `python pec_backtest.py --start <date> --end <date> --batch 3 --comprehensive`
- **Expected Output:**
  - Monthly performance summary
  - Strength/weakness by timeframe
  - Profitability projection for live trading
  - Filter tuning recommendations
- **Time:** ~60 min runtime
- **Report:** Saved to `BATCH_3_MONTHLY_REVIEW.md` + uploaded to Desktop

---

## 🚀 HOW TO TRIGGER BATCHES

Once signal generation starts, monitor `/polymarket-copytrade/data/buy_store.jsonl` line count:

```bash
# Monitor signal count
wc -l /polymarket-copytrade/data/buy_store.jsonl

# When count reaches trigger, run batch:
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main
python pec_backtest.py --batch 1
```

---

## 📊 BATCH OUTPUT STRUCTURE

Each batch generates:
1. **CSV Export:** `pec_backtest_results_BATCH_N.csv`
2. **Excel Export:** `pec_backtest_results_BATCH_N.xlsx` (with color-coded results)
3. **Markdown Report:** `BATCH_N_ANALYSIS_REPORT.md`
4. **Summary Card:** Posted in daily summary

---

## ✅ CURRENT STATUS

| Item | Status | Notes |
|------|--------|-------|
| Backtest Script | ✅ Ready | `run_pec_backtest.py` operational |
| PEC Engine | ✅ Ready | `pec_engine.py` validated |
| Excel Export | ✅ Ready | openpyxl installed |
| Timestamp Tracking | ✅ Ready | signal_fired_utc, entry_time_utc, exit_time_utc active |
| Signal Generation | 🟡 PENDING | Awaiting first 15m/30m/1h signals |

---

## 🔔 NOTIFICATION RULES

- **Batch 1 Ready:** Alert user when 50 signals fired
- **Batch 2 Ready:** Alert user when 150 signals fired
- **Batch 3 Ready:** Alert user when 300 signals fired
- **Daily Summary:** Brief update on signal count + system health

---

## 📝 NOTES

- Batches are **NOT cumulative** - each uses new data window
- Backtest can be re-run on same batch anytime (non-destructive)
- Reports are archived in `/BATCH_REPORTS/` directory
- Results inform next enhancement cycle

