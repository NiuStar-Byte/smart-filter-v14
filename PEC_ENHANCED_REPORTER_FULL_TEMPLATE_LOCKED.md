# PEC Enhanced Reporter - FULL TEMPLATE LOCKED ✅
**Locked Date:** 2026-03-05 09:23 GMT+7  
**Status:** ✅ COMPLETE & FROZEN  
**Template Version:** 1.0 (Full)  
**Scope:** Entire reporter (all 8 sections)

---

## 📋 Complete Template Structure

The PEC Enhanced Reporter consists of **8 major sections**, all now LOCKED:

### 1. Header Section
```
========================================================================================================================================================================================================
📊 PEC ENHANCED REPORTER - SIGNAL PERFORMANCE ANALYSIS
========================================================================================================================================================================================================
```

### 2. Aggregates Section (7 breakdowns)
- 🕐 BY TIMEFRAME
- 📈 BY DIRECTION
- 🛣️ BY ROUTE
- 🌊 BY REGIME
- 💰 BY SYMBOL GROUP
- 💡 BY CONFIDENCE LEVEL
- (Each with: Total | TP | SL | TIMEOUT | Closed | Open | WR | P&L | Avg TP Duration | Avg SL Duration)

### 3. Multi-Dimensional Aggregates Section (9 combinations)
- 🕐💰 BY TIMEFRAME x SYMBOL_GROUP
- 📈💰 BY DIRECTION x SYMBOL_GROUP
- 🕐📈 BY TIMEFRAME x DIRECTION
- 🕐🌊 BY TIMEFRAME x REGIME
- 📈🌊 BY DIRECTION x REGIME
- 📈🛣️ BY DIRECTION x ROUTE
- 🛣️🌊 BY ROUTE x REGIME
- 🛣️💰 BY ROUTE x SYMBOL_GROUP
- 🌊💰 BY REGIME x SYMBOL_GROUP

### 4. 3D & Higher Dimensional Aggregates (6 combinations)
- 🕐📈🛣️ BY TIMEFRAME x DIRECTION x ROUTE
- 🕐📈🌊 BY TIMEFRAME x DIRECTION x REGIME
- 📈🛣️🌊 BY DIRECTION x ROUTE x REGIME
- 🕐🛣️🌊 BY TIMEFRAME x ROUTE x REGIME
- 🕐📈💡 BY TIMEFRAME x DIRECTION x CONFIDENCE
- 🛣️💰 BY ROUTE x SYMBOL_GROUP

### 5. Detailed Signal List Section
- Headers: Symbol | TF | Dir | Route | Regime | Conf | Sym Grp | Status | Entry | Exit | PnL | Fired Time | Exit Time | Duration | Data Quality
- Row per signal (sorted by fired time, descending)
- Status indicators: TP_HIT | SL_HIT | TIMEOUT | OPEN
- Data Quality flags: ✓ CLEAN | ⚠️ STALE_TIMEOUT

### 6. Summary Section (LOCKED - See SUMMARY Template Below)
- Foundation Baseline (Reference)
- Total Signals (Foundation + New)
- Closed Trades Analysis
- Win Rate Calculation
- P&L Metrics
- Duration Metrics
- Total Fired per Date (per-date breakdown)
- Data Quality Note

### 7. Hierarchy Ranking Section (5D/4D/3D/2D)
- 📊 5-DIMENSIONAL COMBOS (TimeFrame × Direction × Route × Regime × Symbol_Group)
  - Top 10 by WR
- 📊 4-DIMENSIONAL COMBOS (TimeFrame × Direction × Route × Regime)
  - Top 5 by WR
- 📊 3-DIMENSIONAL COMBOS (4 types)
  - Top 5 by WR per type
- 📊 2-DIMENSIONAL COMBOS (5 types)
  - Top 8 by WR per type

### 8. Tier Information (if generated)
- Signal tiers appended to SIGNAL_TIERS.json
- Tier-1, Tier-2, Tier-3, Tier-X classification

---

## 🔧 Code Structure (LOCKED)

### Class: PECEnhancedReporter

**Initialization:**
```python
class PECEnhancedReporter:
    def __init__(self, sent_signals_file=None):
        # Auto-detect latest CUMULATIVE file if not specified
        if sent_signals_file is None:
            cumulative_files = sorted([f for f in os.listdir(...) 
                                     if f.startswith('SENT_SIGNALS_CUMULATIVE_')])
            if cumulative_files:
                sent_signals_file = f"/path/{cumulative_files[-1]}"
```

**Key Methods:**

| Method | Purpose | Dynamic |
|--------|---------|---------|
| `load_signals()` | Load all signals from JSONL file | ✅ YES |
| `get_symbol_group()` | Map symbol to group (MAIN_BLOCKCHAIN, TOP_ALTS, MID_ALTS, LOW_ALTS) | ✅ YES |
| `get_gmt7_time()` | Convert UTC to GMT+7 timestamp | ✅ YES |
| `_calculate_pnl_usd()` | Calculate P&L for $1,000 notional position | ✅ YES |
| `_format_duration_hms()` | Format duration as HH:MM:SS | ✅ YES |
| `_format_duration_hm()` | Format duration as Xh Ym (human readable) | ✅ YES |
| `_calculate_avg_duration_by_status()` | Average duration for status (TP_HIT/SL_HIT) | ✅ YES |
| `_calculate_avg_timeout_by_timeframe()` | Average timeout duration by TF | ✅ YES |
| `_generate_detailed_signal_list()` | Generate detailed signal list section | ✅ YES |
| `generate_report()` | Generate entire report with all sections | ✅ YES |
| `_aggregate_by()` | Aggregate stats by single dimension | ✅ YES |
| `_aggregate_by_dimensions()` | Aggregate stats by multiple dimensions | ✅ YES |
| `_aggregate_by_dimensions_with_symbol()` | 5D aggregation with symbol group | ✅ YES |
| `_check_tier_qualification()` | Check tier qualification thresholds | ✅ YES |
| `generate_signal_tiers()` | Generate tier classification (Tier-1/2/3/X) | ✅ YES |
| `_generate_hierarchy_ranking()` | Generate 5D/4D/3D/2D rankings | ✅ YES |
| `save_signal_tiers()` | Save tiers to SIGNAL_TIERS.json | ✅ YES |

---

## 📊 SUMMARY Section (LOCKED)

### Structure (Locked Format)

```
📊 SUMMARY
══════════════════════════════════════════════════════════════════════════════════════════════════════

🔒 FOUNDATION BASELINE (IMMUTABLE - Locked at commit c535c34)
Total Signals: 853 | Closed: 830 | WR: 25.7%
LONG WR: 29.6% | SHORT WR: 46.2% | P&L: $-5498.59

Total Signals (Foundation + New): {total_signals}
(Count Win = {total_tp}; Count Loss = {total_sl}; Count TimeOut = {total_timeout}; Count Open = {total_open}; Stale Timeouts Excluded = {stale_timeout_count})

Closed Trades (Clean Data): {closed_signals}
(TP: {total_tp}, SL: {total_sl}; TimeOut Win = {timeout_wins}; TimeOut Loss = {timeout_losses})

Overall Win Rate: {overall_wr:.2f}%
>> [ (count TP + Count TimeOut Win) / (Closed Trades) ] = [ ({total_tp}+{timeout_wins}) / {closed_signals} ]

Total P&L (Clean Data): ${total_pnl:+.2f}
(Avg P&L per Signal = ${avg_pnl_per_signal:+.2f}; Avg P&L per Closed Trade = ${avg_pnl_per_trade:+.2f})

Avg TP Duration (Clean): {avg_tp_duration_summary}
Avg SL Duration (Clean): {avg_sl_duration_summary}

Max TIMEOUT Window: 15min={max_timeout_15min} | 30min={max_timeout_30min} | 1h={max_timeout_1h}
(CALCULATED from actual clean timeout signals, maximum duration per timeframe)

Total Fired per Date:
  {YYYY-MM-DD}: {count} fired | Beginning Fired Time: {HH:MM:SS} | Last Fired Time: {HH:MM:SS} |
  {YYYY-MM-DD}: {count} fired | Beginning Fired Time: {HH:MM:SS} | Last Fired Time: {HH:MM:SS} |
  [... additional dates ...]

⚠️  DATA QUALITY NOTE: {stale_timeout_count} signals marked as 'STALE_TIMEOUT' (closed >150% past deadline)
These are EXCLUDED from all above metrics to preserve backtest accuracy. See DETAILED SIGNAL LIST for individual stale timeout records.
```

### Calculation Logic (LOCKED)

**All calculations are 100% dynamic:**

1. **Total Signals:** `len(self.signals)` ✅
2. **Count Win:** Sum of TP_HIT (excluding stale) ✅
3. **Count Loss:** Sum of SL_HIT (excluding stale) ✅
4. **Count TimeOut:** Sum of TIMEOUT ✅
5. **Count Open:** Sum of OPEN ✅
6. **Stale Timeouts:** Count with `data_quality_flag='STALE_TIMEOUT'` ✅
7. **TimeOut Win:** TIMEOUT signals with P&L > 0 ✅
8. **TimeOut Loss:** TIMEOUT signals with P&L < 0 ✅
9. **Closed Trades:** TP + SL + TW + TL ✅
10. **Win Rate:** `(TP + TW) / Closed * 100%` ✅
11. **Total P&L:** Sum of all signal P&L calculations ✅
12. **Avg P&L/Signal:** Total P&L / Total Signals ✅
13. **Avg P&L/Trade:** Total P&L / Closed Trades ✅
14. **TP Duration:** Average of all TP_HIT signal durations ✅
15. **SL Duration:** Average of all SL_HIT signal durations ✅
16. **Max TIMEOUT per TF:** Maximum duration of TIMEOUT signals per timeframe (clean only, excluding stale) ✅
17. **Fired per Date:** Per-date breakdown with min/max times ✅

---

## 📈 Aggregates Section (LOCKED)

### Format (Each Aggregate Type)

```
{EMOJI} BY {DIMENSION}
─────────────────────────────────────────────────────────────────────────────────────────────────────
{Key:<width} | Total  | TP   | SL   | TIMEOUT  | Closed  | Open   | WR       | P&L        | Avg TP Duration   | Avg SL Duration
─────────────────────────────────────────────────────────────────────────────────────────────────────
{row1}
{row2}
[... additional rows ...]
```

### Columns (Locked)
- **Key:** Dimension value (TimeFrame, Direction, Route, Regime, Symbol Group, Confidence)
- **Total:** Total signals in group
- **TP:** TP_HIT count (excluding stale)
- **SL:** SL_HIT count (excluding stale)
- **TIMEOUT:** Total TIMEOUT count
- **Closed:** Total closed signals (TP + SL + TW + TL)
- **Open:** Open signals
- **WR:** Win rate % = (TP + TW) / Closed
- **P&L:** Total P&L for group
- **Avg TP Duration:** Average duration of TP_HIT signals
- **Avg SL Duration:** Average duration of SL_HIT signals

### Grouping Rules (LOCKED)

**By TimeFrame:**
- Values: 15min, 30min, 1h

**By Direction:**
- Values: LONG, SHORT

**By Route:**
- Values: AMBIGUOUS, NONE, REVERSAL, TREND_CONTINUATION, TREND CONTINUATION

**By Regime:**
- Values: BEAR, BULL, RANGE

**By Symbol Group:**
- MAIN_BLOCKCHAIN: BTC, ETH, SOL, XRP, ADA, AVAX, BNB, XLM, LINK, POL
- TOP_ALTS: ZKJ, ROAM, XAUT, SAHARA
- MID_ALTS: XPL, DOT, FUEL, VIRTUAL, BERA, CROSS, FUN, ENA
- LOW_ALTS: Everything else

**By Confidence:**
- HIGH (≥76%)
- MID (51-75%)
- LOW (≤50%)

---

## 🎯 Multi-Dimensional Aggregates (LOCKED)

### Format

```
{EMOJI} BY {DIMENSION1} x {DIMENSION2}
─────────────────────────────────────────────────────────────────────────────────────────────────
{Dim1:<width} | {Dim2:<width} | Total | TP | SL | TIMEOUT | Closed | Open | WR   | P&L    | Avg TP Duration | Avg SL Duration
─────────────────────────────────────────────────────────────────────────────────────────────────
{row1}
[... additional rows ...]
```

### 2D Combinations (Locked)
- TF × Symbol Group (row: 3 TFs × 4 groups = 12 combos)
- Direction × Symbol Group (row: 2 dirs × 4 groups = 8 combos)
- TF × Direction (row: 3 × 2 = 6 combos)
- TF × Regime (row: 3 × 3 = 9 combos)
- Direction × Regime (row: 2 × 3 = 6 combos)
- Direction × Route (row: 2 × 5 routes = 10 combos)
- Route × Regime (row: 5 × 3 = 15 combos)
- Route × Symbol Group (row: 5 × 4 = 20 combos)
- Regime × Symbol Group (row: 3 × 4 = 12 combos)

### 3D+ Combinations (Locked)
- TF × Direction × Route (row: 3 × 2 × 5 = 30 combos)
- TF × Direction × Regime (row: 3 × 2 × 3 = 18 combos)
- Direction × Route × Regime (row: 2 × 5 × 3 = 30 combos)
- TF × Route × Regime (row: 3 × 5 × 3 = 45 combos)
- TF × Direction × Confidence (row: 3 × 2 × 3 = 18 combos)

---

## 🎯 Hierarchy Ranking Section (LOCKED)

### Format (Locked)

```
========================================================================================================================================================================================================
🎯 HIERARCHY RANKING - 5D / 4D / 3D / 2D PERFORMANCE TRACKING
========================================================================================================================================================================================================

📊 5-DIMENSIONAL COMBOS (TimeFrame × Direction × Route × Regime × Symbol_Group)
────────────────────────────────────────────────────────────────────────────────

  Top 10 5D Combos by WR:
    ✓ {TF}_{DIR}_{ROUTE}_{REGIME}_{SYMBOL_GROUP} | WR: {wr}% | P&L: ${pnl} | Avg: ${avg_pnl} | Closed: {count}
    [... 9 more ...]

📊 4-DIMENSIONAL COMBOS (TimeFrame × Direction × Route × Regime)
────────────────────────────────────────────────────────────────────────────────

  Top 5 4D Combos by WR:
    ✓ {TF}_{DIR}_{ROUTE}_{REGIME} | WR: {wr}% | P&L: ${pnl} | Avg: ${avg_pnl} | Closed: {count}
    [... 4 more ...]

📊 3-DIMENSIONAL COMBOS (Sample: TF_DIR_ROUTE, TF_DIR_REGIME, DIR_ROUTE_REGIME, TF_ROUTE_REGIME)
────────────────────────────────────────────────────────────────────────────────

  Top 5 3D Combos by WR:
    ✓ {TYPE}_{DIM1}_{DIM2}_{DIM3} | WR: {wr}% | P&L: ${pnl} | Avg: ${avg_pnl} | Closed: {count}
    [... 4 more ...]

📊 2-DIMENSIONAL COMBOS (TF_DIR, TF_REGIME, DIR_REGIME, DIR_ROUTE, ROUTE_REGIME)
────────────────────────────────────────────────────────────────────────────────

  Top 8 2D Combos by WR:
    ✓ {TYPE}_{DIM1}_{DIM2} | WR: {wr}% | P&L: ${pnl} | Avg: ${avg_pnl} | Closed: {count}
    [... 7 more ...]
```

### Ranking Criteria (Locked)
- **Minimum trade threshold:** 25 closed trades (configurable in code)
- **Sorting:** By WR (descending), then by P&L (descending)
- **Display:** Top N combos for each dimension level
- **Metrics shown:** WR%, Total P&L, Avg P&L, Closed trade count

---

## 📋 Detailed Signal List Section (LOCKED)

### Format (Locked)

```
========================================================================================================================================================================================================
📋 DETAILED SIGNAL LIST: FIXED POSITION SIZE $100, LEVERAGE 10x
Report Generated: {YYYY-MM-DD HH:MM:SS GMT+7}
Total Signals Loaded: {count}

⚠️  NOTE: Signals marked 'STALE_TIMEOUT' in Data Quality column are EXCLUDED from Aggregates/Summary/Hierarchy
    (They appear here for audit trail only, but don't affect backtest P&L calculations)

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Symbol       | TF      | Dir    | Route               | Regime | Conf  | Sym Grp    | Status     | Entry      | Exit       | PnL       | Fired Time  | Exit Time/TimeOut  | Duration    | Data Quality
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
{row1}
{row2}
[... signal row per line ...]
```

### Columns (Locked)

| Column | Format | Notes |
|--------|--------|-------|
| Symbol | {SYM:<12} | Symbol name, truncated to 11 chars |
| TF | {TF:<8} | TimeFrame (15min, 30min, 1h) |
| Dir | {DIR:<6} | Direction (LONG, SHORT) |
| Route | {ROUTE:<20} | Trade route (TREND CONTINUATION, REVERSAL, etc.) |
| Regime | {REGIME:<6} | Regime (BULL, BEAR, RANGE) |
| Conf | {CONF:<6} | Confidence as % (e.g., "85%") |
| Sym Grp | {GROUP:<12} | Symbol group mapping |
| Status | {STATUS:<10} | Status (TP_HIT, SL_HIT, TIMEOUT, OPEN) |
| Entry | {ENTRY:<12} | Entry price (6 decimals) |
| Exit | {EXIT:<12} | Exit price (6 decimals) or "N/A" |
| PnL | {PNL:<10} | P&L in USD or "OPEN" |
| Fired Time | {FIRED:<12} | Fired timestamp (GMT+7, HH:MM:SS) |
| Exit Time | {EXIT_TIME:<18} | Closure time (GMT+7) or dash for OPEN |
| Duration | {DURATION:<12} | Duration (HH:MM:SS) or dash for OPEN |
| Data Quality | {QUALITY:<28} | ✓ CLEAN or ⚠️ STALE_TIMEOUT_{Xh} |

### Sorting (Locked)
- Sorted by **fired_time_utc** (most recent first, descending)

### Signal Status Values (Locked)
- **TP_HIT:** Take-profit target reached
- **SL_HIT:** Stop-loss triggered
- **TIMEOUT:** Max bars reached without TP/SL
- **OPEN:** Signal still open (no exit)

---

## 🔄 Auto-Detection Logic (LOCKED)

```python
def __init__(self, sent_signals_file=None):
    if sent_signals_file is None:
        # Auto-detect latest CUMULATIVE file
        cumulative_files = sorted([f for f in os.listdir(workspace_path) 
                                  if f.startswith('SENT_SIGNALS_CUMULATIVE_') 
                                  and f.endswith('.jsonl')])
        if cumulative_files:
            sent_signals_file = f"{workspace_path}/{cumulative_files[-1]}"
            # Latest file is used automatically
        else:
            sent_signals_file = f"{workspace_path}/SENT_SIGNALS.jsonl"
```

**Behavior:**
- ✅ Searches for files matching: `SENT_SIGNALS_CUMULATIVE_*.jsonl`
- ✅ Sorts alphabetically (by date)
- ✅ Uses the LAST (most recent) file
- ✅ Falls back to `SENT_SIGNALS.jsonl` if no CUMULATIVE files found
- ✅ Automatic progression: Mar 4 → Mar 5 → Mar 6 (no manual changes)

---

## 📊 Current Example Output

**File:** `PEC_REPORTER_SUMMARY_EXAMPLE.txt`  
**Data Source:** `SENT_SIGNALS_CUMULATIVE_2026-03-04.jsonl`  
**Total Signals:** 1,321 (853 Foundation + 468 New)  
**Metrics:**
- Total Closed: 1,113
- Win Rate: 33.33%
- P&L: $-3,688.15
- Dates: Feb 27 - Mar 4

---

## 🔒 What's Locked (Complete List)

✅ **All 8 sections locked:**
1. Header format (constant)
2. Aggregates structure (7 types, fixed columns)
3. Multi-dimensional aggregates (9 2D combos, 6 3D+ combos)
4. Detailed signal list (fixed columns, GMT+7 timezone)
5. Summary section (14 metrics, all dynamic)
6. Hierarchy ranking (5D/4D/3D/2D breakdowns)
7. Data quality flags (stale timeout handling)
8. Auto-detection logic (latest CUMULATIVE file)

✅ **Calculation formulas locked:**
- Win Rate: `(TP + TW) / Closed * 100%`
- P&L: `[(exit - entry) / entry] * $1,000` (LONG), `[(entry - exit) / exit] * $1,000` (SHORT)
- Timeout classification: based on P&L calculation
- Duration: seconds converted to HH:MM:SS or Xh Ym format
- Per-date breakdown: min/max fired times per date

✅ **Symbol grouping locked:**
- MAIN_BLOCKCHAIN: 10 symbols
- TOP_ALTS: 4 symbols
- MID_ALTS: 10 symbols
- LOW_ALTS: All others

✅ **Dimension definitions locked:**
- TimeFrame: 15min, 30min, 1h
- Direction: LONG, SHORT
- Route: 5 types (AMBIGUOUS, NONE, REVERSAL, TREND_CONTINUATION, TREND CONTINUATION)
- Regime: BULL, BEAR, RANGE
- Confidence: HIGH (≥76%), MID (51-75%), LOW (≤50%)

---

## ❌ What's NOT Locked

❌ Actual metric values (always dynamic from signals)  
❌ CUMULATIVE file source (auto-detects latest)  
❌ Number of rows displayed (depends on actual data)  
❌ Visualization/color formatting (can be customized)  

---

## 📂 Reference Files

| File | Purpose | Size | Status |
|------|---------|------|--------|
| `pec_enhanced_reporter.py` | Active script (v1.0) | 79K | ✅ Locked |
| `pec_enhanced_reporter_TEMPLATE_LOCKED.py` | Frozen copy | 79K | 🔒 Reference |
| `PEC_ENHANCED_REPORTER_FULL_TEMPLATE_LOCKED.md` | This doc | - | 📖 Reference |
| `PEC_REPORTER_TEMPLATE_LOCKED.md` | Summary-only doc | 7.3K | 📖 Reference |
| `PEC_REPORTER_SUMMARY_EXAMPLE.txt` | Live output example | 1.9K | 📊 Example |
| `PEC_ENHANCED_REPORT.txt` | Full report output | 335K | 📄 Output |

---

## 🚀 Usage

**Generate full report:**
```bash
cd /Users/geniustarigan/.openclaw/workspace
python3 pec_enhanced_reporter.py > PEC_ENHANCED_REPORT.txt
```

**All outputs:**
- Console: Full 1,700+ line report
- Aggregates: 7 dimensional breakdowns
- Multi-dimensional: 15 combination analyses
- Detailed signals: All signals with full metrics
- Summary: 14 dynamic metrics
- Hierarchy: 5D/4D/3D/2D rankings
- Tiers: Appended to SIGNAL_TIERS.json (if generated)

---

## ✅ Sign-Off

**Template Status:** 🔒 **LOCKED v1.0 (COMPLETE)**  
**Scope:** Entire PEC Enhanced Reporter (all 8 sections)  
**Validation:** ✅ All calculations verified as dynamic  
**Deployment:** ✅ Ready for production  
**Date Locked:** 2026-03-05 09:23 GMT+7  
**Git Commit:** 0c3b302 + (new)

**Next Steps:**
- Daily monitoring (Mar 5-12)
- Generate new reports as CUMULATIVE files are created
- Compare metrics over time
- Adjust if formula changes needed (document why)

---

**🔒 COMPLETE TEMPLATE LOCKED FOR PRODUCTION**
