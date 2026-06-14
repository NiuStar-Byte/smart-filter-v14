# ASTERDEX PERFORMANCE TRACKING SYSTEM
**Completely isolated from posting logic. Zero impact on entry placement.**

---

## 📊 System Overview

```
┌─ asterdex_entry_poster.py (UNCHANGED)
│  └─ Posts entries → logs to ASTERDEX_POSTED_ENTRIES.jsonl
│
├─ asterdex_performance_tracker_system.py (NEW - isolated)
│  ├─ asterdex_trade_fetcher.py
│  │  └─ Fetches executed trades from Asterdex API every 5 min
│  │
│  ├─ asterdex_performance_matcher.py
│  │  └─ Matches POSTED entries ↔ EXECUTED trades
│  │
│  └─ asterdex_performance_analytics.py
│     └─ Calculates WR, P&L, performance by dimension
│
└─ Output: Daily performance reports → Telegram
```

---

## 🗂️ Data Files

### Input: ASTERDEX_POSTED_ENTRIES.jsonl
**One line per posted entry (appended by posting code)**

```json
{
  "signal_uuid": "f740120d-8e5c-4235-90f6-9225c0050461",
  "symbol": "TURBO-USDT",
  "side": "SHORT",
  "entry_price": 0.000866,
  "quantity": 1048.0,
  "timeframe": "4h",
  "tier": 2,
  "mtf_alignment_band": "strong",
  "route": "TREND_CONTINUATION",
  "confidence_level": "MID",
  "posted_timestamp": "2026-06-08T09:47:12.123456Z",
  "status": "POSTED"
}
```

### Working: ASTERDEX_TRADES.jsonl
**Fetched from Asterdex API, cached locally**

```json
{
  "symbol": "TURBO-USDT",
  "side": "SELL",
  "executedPrice": 0.000866,
  "executedQty": 1048.0,
  "realizedProfit": -0.00835,
  "fee": -0.00084,
  "time": "2026-06-08T09:47:45.000Z",
  "orderId": "99999"
}
```

### Output: ASTERDEX_PERFORMANCE_CORRELATED.jsonl
**Matched POSTED entry ↔ EXECUTED trade**

```json
{
  "signal_uuid": "f740120d-8e5c-4235-90f6-9225c0050461",
  "symbol": "TURBO-USDT",
  "tier": 2,
  "mtf_alignment_band": "strong",
  "route": "TREND_CONTINUATION",
  "timeframe": "4h",
  "side": "SHORT",
  "posted_price": 0.000866,
  "executed_price": 0.000866,
  "quantity": 1048.0,
  "posted_timestamp": "2026-06-08T09:47:12Z",
  "executed_timestamp": "2026-06-08T09:47:45Z",
  "realized_pnl_usd": -0.00835,
  "fee_usd": -0.00084,
  "net_pnl_usd": -0.00919,
  "win": false,
  "status": "CLOSED",
  "match_confidence": 0.99,
  "match_method": "symbol_side_time_price"
}
```

---

## 🔍 Matching Strategy (No UUID)

For each POSTED entry, find corresponding EXECUTED trade by:

1. **Symbol** (exact): `TURBO-USDT`
2. **Side** (exact): `SHORT` ↔ `SELL`
3. **Time window** (±10 min): posted 09:47:12 → executed 09:47:45 ✓
4. **Price proximity** (±2%): posted 0.000866 vs executed 0.000866 ✓
5. **Quantity** (exact or close): 1048 qty ✓

**Match score:** All 5 criteria met = **High confidence (0.95+)**

**Fallback:** If no exact match found:
- Extend time window to ±30 min
- Relax price to ±5%
- Mark with lower confidence score

---

## 📈 Analytics Output

### Performance by Tier
```
Tier-1:  WR=75%, Avg P&L=$+45.23, Trades=4
Tier-2:  WR=60%, Avg P&L=$+12.50, Trades=15
Tier-3:  WR=45%, Avg P&L=-$2.10, Trades=8
```

### Performance by MTF Alignment
```
Strong:  WR=68%, Avg P&L=$+28.50, Trades=19
Neutral: WR=40%, Avg P&L=-$5.20, Trades=8
```

### Performance by Symbol
```
NEAR-USDT:  WR=70%, P&L=$+65.40
SOL-USDT:   WR=65%, P&L=$+42.10
AVAX-USDT:  WR=55%, P&L=$-8.30
```

---

## 🔄 Operational Flow

### Every 5 minutes (cron or daemon)
1. **Fetch trades** from Asterdex API (`/fapi/v3/allOrders`)
   - Only new trades since last check
   - Cache in ASTERDEX_TRADES.jsonl

2. **Match entries** against trades
   - For each POSTED entry with status="POSTED"
   - Find corresponding trade
   - Update status → "MATCHED"/"CLOSED"
   - Store P&L in ASTERDEX_PERFORMANCE_CORRELATED.jsonl

3. **Calculate analytics**
   - Aggregate by Tier, Symbol, MTF, Timeframe, Route
   - Generate daily summary
   - Send to Telegram if significant changes

### Daily (00:00 GMT+7)
1. Generate daily performance report
2. Send to Telegram channel
3. Archive correlated data

---

## 🚀 Implementation Timeline

| Module | Est. Time | Status |
|--------|-----------|--------|
| `asterdex_trade_fetcher.py` | 45 min | Building |
| `asterdex_performance_matcher.py` | 60 min | Next |
| `asterdex_performance_analytics.py` | 45 min | Next |
| `Integration + testing` | 30 min | Final |
| **Total** | **3 hours** | In progress |

---

## 🔒 Safety Guarantees

✅ **ZERO impact on posting code** — only reads and logs  
✅ **Isolated files** — separate from trading files  
✅ **Failure tolerant** — if tracker dies, posting continues  
✅ **No authentication changes** — uses existing API credentials  
✅ **Reversible** — can disable/remove without affecting trades  

---

## 📝 Next Steps (User Approval)

Once this is live with 5-10 days of data:
1. Review Tier performance (which tiers are actually profitable?)
2. Rotate underperforming combos out
3. Increase allocation to high-WR combos
4. A/B test new combos with small allocation
5. Iterate to optimize signal quality

**This becomes the foundation for continuous improvement.**
