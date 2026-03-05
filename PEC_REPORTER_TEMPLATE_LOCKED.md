# PEC Enhanced Reporter - TEMPLATE LOCKED ✅
**Locked Date:** 2026-03-05 09:21 GMT+7  
**Status:** ✅ FINAL & APPROVED  
**Template Version:** 1.0  

---

## 📋 Template Overview

The PEC Enhanced Reporter generates comprehensive signal performance analysis with dynamic calculations based on actual signal data. All numbers are calculated from real signals—nothing is hardcoded.

**Key Files:**
- `pec_enhanced_reporter.py` — Active/working script (reads latest CUMULATIVE file)
- `pec_enhanced_reporter_TEMPLATE_LOCKED.py` — Frozen reference template
- `PEC_ENHANCED_REPORT.txt` — Generated report output (1760 lines)

---

## 📊 SUMMARY Section Structure (LOCKED)

The SUMMARY section contains dynamically calculated metrics:

### Foundation Baseline (Reference Only)
```
🔒 FOUNDATION BASELINE (IMMUTABLE - Locked at commit c535c34)
Total Signals: 853 | Closed: 830 | WR: 25.7%
LONG WR: 29.6% | SHORT WR: 46.2% | P&L: $-5498.59
```

### Total Signals Breakdown (Dynamic)
```
Total Signals (Foundation + New): {total_signals}
(Count Win = {total_tp}; Count Loss = {total_sl}; Count TimeOut = {total_timeout}; Count Open = {total_open}; Stale Timeouts Excluded = {stale_timeout_count})
```

### Closed Trades Analysis (Dynamic)
```
Closed Trades (Clean Data): {closed_signals}
(TP: {total_tp}, SL: {total_sl}; TimeOut Win = {timeout_wins}; TimeOut Loss = {timeout_losses})
```

### Win Rate Calculation (Dynamic)
```
Overall Win Rate: {overall_wr:.2f}%
>> [ (count TP + Count TimeOut Win) / (Closed Trades) ] = [ ({total_tp}+{timeout_wins}) / {closed_signals} ]
```

### P&L Metrics (Dynamic)
```
Total P&L (Clean Data): ${total_pnl:+.2f}
(Avg P&L per Signal = ${avg_pnl_per_signal:+.2f}; Avg P&L per Closed Trade = ${avg_pnl_per_trade:+.2f})
```

### Duration Metrics (Dynamic)
```
Avg TP Duration (Clean): {avg_tp_duration_summary}
Avg SL Duration (Clean): {avg_sl_duration_summary}
Max TIMEOUT Window: 15min=3h 45m | 30min=5h 0m | 1h=5h 0m
```

### Total Fired per Date (Dynamic - Per-Date Breakdown)
```
Total Fired per Date:
  {YYYY-MM-DD}: {count} fired | Beginning Fired Time: {HH:MM:SS} | Last Fired Time: {HH:MM:SS} |
  {YYYY-MM-DD}: {count} fired | Beginning Fired Time: {HH:MM:SS} | Last Fired Time: {HH:MM:SS} |
  ... (each date on own line)
```

### Data Quality Note (Dynamic)
```
⚠️  DATA QUALITY NOTE: {stale_timeout_count} signals marked as 'STALE_TIMEOUT' (closed >150% past deadline)
These are EXCLUDED from all above metrics to preserve backtest accuracy. See DETAILED SIGNAL LIST for individual stale timeout records.
```

---

## 🔧 Dynamic Calculations (Code-Level)

All SUMMARY numbers are calculated from actual signals:

```python
# 1. Total signals loaded
total_signals = len(self.signals)

# 2. Count by status (excluding stale timeouts)
total_tp = sum(1 for s in self.signals if s.get('status') == 'TP_HIT' and not STALE)
total_sl = sum(1 for s in self.signals if s.get('status') == 'SL_HIT' and not STALE)
total_timeout = sum(1 for s in self.signals if s.get('status') == 'TIMEOUT')
total_open = sum(1 for s in self.signals if s.get('status') == 'OPEN')

# 3. Separate TIMEOUT into wins/losses based on P&L calculation
# (iterates all signals, checks actual_exit_price vs entry_price)
timeout_wins = ...
timeout_losses = ...

# 4. Closed trades calculation
closed_signals = total_tp + total_sl + timeout_wins + timeout_losses

# 5. Win rate
overall_wr = (total_tp + timeout_wins) / closed_signals * 100

# 6. P&L calculation (notional position $1,000)
total_pnl = sum(pnl_calc for each signal)
avg_pnl_per_signal = total_pnl / total_signals
avg_pnl_per_trade = total_pnl / closed_signals

# 7. Duration metrics
avg_tp_duration = calculate_avg(TP_HIT signals)
avg_sl_duration = calculate_avg(SL_HIT signals)

# 8. Per-date breakdown
for each date in signals:
    count = signals on that date
    beginning_time = earliest signal time on that date
    last_time = latest signal time on that date
```

---

## 📂 File Auto-Detection

The reporter automatically detects and uses the latest CUMULATIVE file:

```python
def __init__(self, sent_signals_file=None):
    if sent_signals_file is None:
        # Auto-detect latest SENT_SIGNALS_CUMULATIVE_YYYY-MM-DD.jsonl
        cumulative_files = sorted([f for f in os.listdir(...) 
                                  if f.startswith('SENT_SIGNALS_CUMULATIVE_')])
        if cumulative_files:
            sent_signals_file = f"/path/{cumulative_files[-1]}"  # Latest file
```

**How it works:**
- Daily at 23:00 GMT+7, a cron job creates `SENT_SIGNALS_CUMULATIVE_YYYY-MM-DD.jsonl`
- Reporter auto-uses the latest file
- Each file contains: Foundation (853) + New signals accumulated that day

---

## ✅ Validation Checklist

**All SUMMARY metrics validated:**

| # | Metric | Type | Dynamic | Validated |
|---|--------|------|---------|-----------|
| 1 | Total Signals (Foundation + New) | Count | ✅ YES | ✅ YES |
| 2 | Count Win (TP_HIT, excl. stale) | Count | ✅ YES | ✅ YES |
| 3 | Count Loss (SL_HIT, excl. stale) | Count | ✅ YES | ✅ YES |
| 4 | Count TimeOut | Count | ✅ YES | ✅ YES |
| 5 | Count Open | Count | ✅ YES | ✅ YES |
| 6 | Stale Timeouts Excluded | Count | ✅ YES | ✅ YES |
| 7 | TimeOut Win (P&L > 0) | Calculated | ✅ YES | ✅ YES |
| 8 | TimeOut Loss (P&L < 0) | Calculated | ✅ YES | ✅ YES |
| 9 | Closed Trades | Calculated | ✅ YES | ✅ YES |
| 10 | Overall Win Rate | Calculated | ✅ YES | ✅ YES |
| 11 | Total P&L | Calculated | ✅ YES | ✅ YES |
| 12 | Avg P&L per Signal | Calculated | ✅ YES | ✅ YES |
| 13 | Avg P&L per Trade | Calculated | ✅ YES | ✅ YES |
| 14 | Avg TP Duration | Calculated | ✅ YES | ✅ YES |
| 15 | Avg SL Duration | Calculated | ✅ YES | ✅ YES |
| 16 | Total Fired per Date | Breakdown | ✅ YES | ✅ YES |

---

## 🚀 Usage

**Generate report:**
```bash
cd /Users/geniustarigan/.openclaw/workspace
python3 pec_enhanced_reporter.py > PEC_ENHANCED_REPORT.txt
```

**Output:**
- `PEC_ENHANCED_REPORT.txt` — Full report (1760+ lines)
- Console output with all sections

**Auto-Update:**
- Runs manually as needed
- Will use latest CUMULATIVE file (Mar 5 → Mar 6 → Mar 7, etc.)
- All calculations updated dynamically

---

## 🔒 Template Lock Details

**What's locked:**
- ✅ SUMMARY section structure and layout
- ✅ Calculation formulas and logic
- ✅ Output format (per-date per-line for fired times)
- ✅ Metric names and descriptions
- ✅ Foundation baseline reference values

**What's NOT locked:**
- ❌ Actual numbers (always dynamic from real signals)
- ❌ CUMULATIVE file source (auto-detects latest)
- ❌ Aggregation sections (remain flexible for analysis)

**Future Changes:**
- If new metrics are needed: add to SUMMARY calculation section
- If layout needs adjustment: update template code only
- If formulas change: update calculation logic (document why)

---

## 📝 Change Log

**2026-03-05 09:21 GMT+7 - TEMPLATE LOCKED v1.0**
- Finalized SUMMARY section structure
- All metrics validated as dynamic (0 hardcoded values)
- Per-date fired time breakdown implemented
- Template frozen for production use
- Reference files created: `pec_enhanced_reporter_TEMPLATE_LOCKED.py`

---

## 🎯 Next Steps

1. ✅ Template locked and saved
2. ⏳ Deploy for daily use (runs manually or via cron)
3. ⏳ Monitor first week of outputs (Mar 5-12)
4. ⏳ Compare against baseline metrics as needed

---

**Template Status:** 🔒 LOCKED - Ready for Production
