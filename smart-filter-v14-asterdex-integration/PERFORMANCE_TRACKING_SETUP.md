# ASTERDEX PERFORMANCE TRACKING - COMPLETE SETUP

**Status:** ✅ **BUILT & READY TO DEPLOY** (Jun 8 11:30 GMT+7)

---

## 🎯 What This Does

**Complete isolated performance tracking system for Asterdex trading:**

1. **Logs posted entries** → `ASTERDEX_POSTED_ENTRIES.jsonl`
2. **Fetches executed trades** → `ASTERDEX_TRADES.jsonl`
3. **Correlates entries ↔ trades** → `ASTERDEX_PERFORMANCE_CORRELATED.jsonl`
4. **Calculates metrics** → `ASTERDEX_PERFORMANCE_ANALYSIS.json`
5. **Generates daily reports** → Text + JSON

**Key insight:** Matches trades WITHOUT using UUID (not available in Asterdex)
- Uses: symbol + side + time window (±10 min) + price proximity (±2%)
- Confidence scoring for match quality

---

## 📦 New Files Created

### Core Modules (3 independent components)

1. **`asterdex_trade_fetcher.py`** (6.6 KB)
   - Fetches from `/fapi/v3/allOrders` endpoint
   - Caches trades locally every 5 minutes
   - Deduplicates by order ID

2. **`asterdex_performance_matcher.py`** (7.1 KB)
   - Matches POSTED entries to EXECUTED trades
   - No UUID matching strategy
   - Confidence scoring per match

3. **`asterdex_performance_analytics.py`** (7.9 KB)
   - Calculates WR, P&L, metrics
   - Breaks down by: Tier, Symbol, Timeframe, MTF band, Route
   - Generates formatted reports

### Support Modules

4. **`asterdex_entry_logger.py`** (1.5 KB)
   - Fire-and-forget logging of posted entries
   - ZERO impact on posting logic
   - Silent failures

5. **`asterdex_performance_system.py`** (2.9 KB)
   - Master orchestrator
   - Runs complete pipeline in order
   - Can be called from cron

### Data Files

6. **`ASTERDEX_POSTED_ENTRIES.jsonl`**
   - Append-only log of posted entries
   - Created by entry poster when posting succeeds
   - One entry per line (JSON)

7. **`ASTERDEX_TRADES.jsonl`**
   - Cache of executed trades from Asterdex
   - Fetched every 5 minutes
   - Deduplicated by order ID

8. **`ASTERDEX_PERFORMANCE_CORRELATED.jsonl`**
   - Matched entry ↔ trade correlation
   - P&L data for each entry
   - Used for analytics

9. **`ASTERDEX_PERFORMANCE_ANALYSIS.json`**
   - JSON output from analytics
   - Structured metrics by dimension
   - Updated every run

### Documentation

10. **`ASTERDEX_TRACKING_ARCHITECTURE.md`**
    - Complete system design
    - Data flow diagrams
    - Matching strategy explained

11. **`PERFORMANCE_TRACKING_SETUP.md`** (this file)
    - Setup instructions
    - Usage examples
    - Troubleshooting

---

## 🚀 How to Use

### Manual Test

```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration

# Run complete pipeline
python3 asterdex_performance_system.py
```

**Output:**
```
============================================================
ASTERDEX PERFORMANCE TRACKING SYSTEM
Start: 2026-06-08T11:30:45.123456

[STEP 1] Fetching trades from Asterdex API...
✅ Fetched 12 recent trades

[STEP 2] Correlating posted entries with executed trades...
✅ New correlations: 5

[STEP 3] Analyzing performance metrics...
✅ Analysis complete

[STEP 4] Performance Report
------------------------------------------------------------
Total Trades Analyzed: 5

Performance by TIER:
Tier-2       |  WR:  60.0% | Trades:  3 | Avg P&L: $+15.50
Tier-1       |  WR:  100.0% | Trades:  2 | Avg P&L: $+45.20
...
============================================================
```

### Automatic (Cron every 5 minutes)

Add to crontab:
```bash
*/5 * * * * cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration && python3 asterdex_performance_system.py >> /tmp/asterdex_perf.log 2>&1
```

### Integration with Telegram Reports

```bash
# Optional: Send daily summary to Telegram
python3 asterdex_performance_system.py && \
  python3 send_daily_report.py  # (create wrapper if needed)
```

---

## 📊 Output Examples

### ASTERDEX_POSTED_ENTRIES.jsonl
```json
{
  "signal_uuid": "f740120d-8e5c-4235-90f6-9225c0050461",
  "symbol": "NEAR-USDT",
  "side": "LONG",
  "entry_price": 2.6050,
  "quantity": 768.0,
  "timeframe": "4h",
  "tier": 2,
  "mtf_alignment_band": "strong",
  "route": "TREND_CONTINUATION",
  "confidence_level": "MID",
  "posted_timestamp": "2026-06-08T09:47:12.123456Z",
  "status": "POSTED"
}
```

### ASTERDEX_PERFORMANCE_CORRELATED.jsonl
```json
{
  "signal_uuid": "f740120d-8e5c-4235-90f6-9225c0050461",
  "symbol": "NEAR-USDT",
  "tier": 2,
  "mtf_alignment_band": "strong",
  "route": "TREND_CONTINUATION",
  "timeframe": "4h",
  "side": "LONG",
  "posted_price": 2.6050,
  "executed_price": 2.6048,
  "quantity": 768.0,
  "posted_timestamp": "2026-06-08T09:47:12Z",
  "executed_timestamp": "2026-06-08T09:47:45Z",
  "realized_pnl_usd": 45.20,
  "fee_usd": -0.30,
  "net_pnl_usd": 44.90,
  "win": true,
  "status": "CLOSED",
  "match_confidence": 0.98,
  "match_method": "symbol_side_time_price"
}
```

### ASTERDEX_PERFORMANCE_ANALYSIS.json
```json
{
  "total_trades": 15,
  "last_updated": "2026-06-08T11:30:45.123456",
  "by_tier": {
    "Tier-1": {
      "wr": 75.0,
      "count": 4,
      "total_pnl": 180.80,
      "avg_pnl": 45.20
    },
    "Tier-2": {
      "wr": 60.0,
      "count": 10,
      "total_pnl": 155.00,
      "avg_pnl": 15.50
    }
  },
  "by_symbol": {
    "NEAR-USDT": {
      "wr": 70.0,
      "count": 5,
      "total_pnl": 123.45
    }
  }
}
```

---

## 🔍 Key Features

### ✅ Zero Impact on Posting
- Logging happens AFTER successful posting
- Failures in logging don't stop trading
- Completely separate codebase

### ✅ Smart Matching (No UUID needed)
- Symbol + Side + Time + Price proximity
- Confidence scoring (0.0 to 1.0)
- Fallback logic if no exact match

### ✅ Dimension Analytics
- **By Tier:** Which tiers actually win?
- **By Symbol:** Which symbols are profitable?
- **By Timeframe:** Which TFs work best?
- **By MTF Band:** Strong vs neutral performance
- **By Route:** Which entry types succeed?

### ✅ Extensible
- Easy to add new dimensions
- Easy to create custom reports
- Foundation for A/B testing

---

## 📈 Using This for Trading Decisions

### After 5-10 days of data:

1. **Identify best performers:**
   ```bash
   python3 asterdex_performance_analytics.py | grep "Tier\|Symbol\|MTF"
   ```

2. **Check which combos are winning:**
   - High WR = good signal timing
   - High Avg P&L = good risk/reward
   - Both high = excellent combo

3. **Rotation strategy:**
   - Keep high-WR combos
   - Increase allocation to winners
   - Decrease or remove losers
   - Test new combos with small allocation

4. **A/B testing:**
   - Compare Tier-1 vs Tier-2 vs Tier-3 WR
   - Compare MTF Strong vs other alignments
   - Decide which filters to use

---

## 🔒 Safety Guarantees

✅ **Completely isolated** - Zero changes to posting code  
✅ **Fault tolerant** - Errors in logging don't impact trading  
✅ **Reversible** - Can disable/remove without affecting trades  
✅ **Append-only** - No data loss, pure logging  
✅ **Deduplication** - No duplicate entries via order ID  

---

## 🛠️ Troubleshooting

### "No trading data available yet"
- **Cause:** No entries posted yet, or no trades matched
- **Fix:** Wait for at least 1 entry to be posted and executed
- **Timeline:** Usually 5-15 minutes after first entry post

### Matches not found for posted entries
- **Cause:** Trade hasn't executed yet, or price mismatch
- **Fix:** Wait longer, check if entry order was actually filled
- **Debug:** Review ASTERDEX_TRADES.jsonl to see fetched trades

### Missing symbol in analysis
- **Cause:** Symbol hasn't generated any posted entries yet
- **Fix:** Wait for signals in that symbol, or post manually
- **Status:** Only tracks symbols that have posted entries

### High time window (>30 min) in matches
- **Cause:** Delay between posting and execution
- **Normal:** Usually < 5 min, up to 30 min acceptable
- **Alert if:** > 2 hours suggests posting/execution lag

---

## 📝 Next Steps

1. ✅ **Deploy:** Run performance system
2. ⏳ **Collect data:** Let it run for 5-10 days
3. 📊 **Analyze:** Review WR and P&L by dimension
4. 🎯 **Optimize:** Rotate combos based on performance
5. 🔄 **Iterate:** Repeat every week

---

## 🎓 How Matching Works (Technical Details)

### Matching Algorithm

For each POSTED entry, we find the corresponding EXECUTED trade using:

```
Matching Score = (symbol match + side match) × (time score × price score)

Criteria:
1. Symbol: Exact (NEAR-USDT == NEAR-USDT) ✓
2. Side: Mapping (LONG → BUY, SHORT → SELL) ✓
3. Time: Window (posted ±10 min, expandable to ±30 min)
4. Price: Proximity (±2% of entry price, expandable to ±5%)

Example:
- Posted: NEAR-USDT LONG @ 2.6050 at 09:47:12
- Found:  NEAR-USDT BUY  @ 2.6048 at 09:47:45
- Time diff: 33 seconds (within 10 min window) ✓
- Price diff: 0.0002 = 0.008% (within 2%) ✓
- Confidence: 0.98/1.0 ✓ MATCH!
```

### Why No UUID?

Asterdex API doesn't return the signal UUID in executed trades. Instead:
- We match by trading characteristics (symbol, side, entry, time, price)
- This is actually more robust - works even if signals system changes
- Requires careful handling of price/time rounding/delays
- Confidence scoring tells us match quality

---

## 📞 Support

If any issues:
1. Check `/tmp/asterdex_perf.log` for errors
2. Verify API is reachable: `ping fapi.asterdex.com`
3. Check credentials in `.env` file
4. Run manual test: `python3 asterdex_performance_system.py`

---

**Built Jun 8 2026 - Completely isolated from posting logic - Zero impact on trading**
