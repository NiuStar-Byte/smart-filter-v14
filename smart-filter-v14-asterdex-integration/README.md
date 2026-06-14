# Smart Filter v14 → Asterdex Integration

**Purpose:** Automatically post real trading entries to Asterdex based on signals from smart-filter-v14.

**Status:** 🚧 UNDER CONSTRUCTION

---

## Architecture Overview

```
smart-filter-v14-asterdex-integration/
├── asterdex_entry_poster.py        # Main: Listen to SIGNALS_MASTER → Post to Asterdex
├── asterdex_rate_limiter.py        # Prevent duplicate entries (symbol+TF cooldown)
├── asterdex_order_tracker.py       # Track posted orders, sync with fills
├── asterdex_config.py              # API keys, leverage, margin settings
├── asterdex_utils.py               # Helper functions (request signing, etc.)
├── requirements.txt                # Dependencies
├── README.md                        # This file
├── logs/                            # Execution logs
│   └── asterdex_poster_YYYY-MM-DD.log
└── data/
    ├── entry_log.jsonl             # All entry attempts (success + failed)
    ├── active_orders.jsonl         # Currently open orders
    └── cooldown_state.json         # (symbol, TF) → last entry timestamp
```

---

## Core Components

### 1. **asterdex_entry_poster.py** (Main Engine)
```
Workflow:
1. Poll SIGNALS_MASTER.jsonl every 5-10 seconds
2. For each NEW signal:
   a. Check tier: Only Tier 1, 2, 3
   b. Check cooldown: (symbol, timeframe) not in cooldown
   c. Fetch entry price + TP/SL from signal
   d. Post entry to Asterdex (10x leverage, $1 margin)
   e. Log order details
   f. Start cooldown timer
3. Loop forever
```

**Key Features:**
- ✅ Real-time signal monitoring
- ✅ Tier-based filtering
- ✅ Cooldown enforcement (per symbol+TF)
- ✅ Asterdex order posting (market or limit)
- ✅ Automatic TP/SL placement
- ✅ Error handling & retry logic

### 2. **asterdex_rate_limiter.py** (Cooldown Manager)
```
Tracks: (symbol, timeframe) → last_entry_timestamp

Rules:
- BTC-USDT + 2h + entered at 10:00
- Cannot enter BTC-USDT + 2h again until 12:00 (after 2 hours)
- BTC-USDT + 4h is separate (can enter independently)
- SOL-USDT + 2h is separate (can enter independently)

Storage: cooldown_state.json (persistent across restarts)
```

### 3. **asterdex_order_tracker.py** (Order Monitoring)
```
Workflow:
1. Poll Asterdex /fapi/v1/openOrders every 30 seconds
2. For each active order:
   a. Check if TP/SL hit (by checking Asterdex balance/positions)
   b. If TP_HIT: Log as successful, mark order closed
   c. If SL_HIT: Log as loss, mark order closed
   d. Update entry_log.jsonl with result
3. Sync with SENT_SIGNALS.jsonl (if needed for reporting)
```

### 4. **asterdex_config.py** (Settings)
```python
ASTERDEX_API_KEY = "YOUR_KEY"
ASTERDEX_API_SECRET = "YOUR_SECRET"
ASTERDEX_BASE_URL = "https://fapi.asterdex.com"

POSITION_SETTINGS = {
    "margin": 1.0,          # $1 per entry
    "leverage": 10,         # 10x → $10 notional
    "order_type": "MARKET", # MARKET (instant) or LIMIT (pending)
}

RATE_LIMIT_SETTINGS = {
    "check_interval_sec": 5,      # Poll signals every 5 seconds
    "order_check_interval_sec": 30, # Check fills every 30 seconds
}

TIER_FILTER = [1, 2, 3]  # Only post Tier 1, 2, 3
```

### 5. **asterdex_utils.py** (Helpers)
```
Functions:
- sign_request() → HMAC-SHA256 signature for API calls
- parse_signal() → Extract symbol, direction, TP, SL from signal dict
- check_tier() → Validate Tier 1/2/3
- calculate_order_size() → Convert $10 notional to contracts
- format_log_entry() → Standardized logging
```

---

## Signal Flow (Detailed)

```
SIGNALS_MASTER.jsonl (from main.py)
        ↓
   asterdex_entry_poster.py reads new signals
        ↓
   Filter: Tier == 1, 2, or 3? → YES
        ↓
   Check cooldown: (symbol, TF) allowed? → YES
        ↓
   Extract: symbol, direction, TP_price, SL_price, entry_price
        ↓
   Calculate: quantity = ($10 notional) / entry_price
        ↓
   Sign request (HMAC-SHA256 using Asterdex API key)
        ↓
   POST /fapi/v1/order (MARKET entry)
        ↓
   Asterdex returns: Order ID, fill price, status
        ↓
   Log entry_log.jsonl + Set cooldown timer
        ↓
   asterdex_order_tracker.py polls fills every 30 seconds
        ↓
   If TP/SL hit: Update order status, log result
```

---

## Data Files

### entry_log.jsonl (Immutable Ledger)
```json
{
  "timestamp": "2026-05-29T13:50:00Z",
  "signal_uuid": "abc-123-def",
  "symbol": "BTC-USDT",
  "direction": "LONG",
  "timeframe": "4h",
  "tier": 2,
  "entry_price": 67500.50,
  "tp_price": 69000.00,
  "sl_price": 66000.00,
  "quantity": 0.00148,
  "notional": 10.00,
  "asterdex_order_id": "12345678",
  "status": "ENTERED",
  "fill_price": 67501.00,
  "fill_time": "2026-05-29T13:50:05Z"
}
```

### cooldown_state.json (Persistent)
```json
{
  "BTC-USDT_4h": "2026-05-29T13:50:00Z",
  "ETH-USDT_2h": "2026-05-29T12:30:00Z",
  "SOL-USDT_1h": "2026-05-29T13:00:00Z"
}
```

### active_orders.jsonl
```json
{
  "asterdex_order_id": "12345678",
  "signal_uuid": "abc-123-def",
  "symbol": "BTC-USDT",
  "status": "OPEN",
  "entry_price": 67501.00,
  "tp_price": 69000.00,
  "sl_price": 66000.00,
  "entry_time": "2026-05-29T13:50:05Z"
}
```

---

## Execution Flow (Timeline)

```
13:50:00 - Signal arrives in SIGNALS_MASTER.jsonl
           symbol: BTC-USDT, direction: LONG, tier: 2, timeframe: 4h
           
13:50:01 - asterdex_entry_poster.py polls SIGNALS_MASTER
           ✅ Tier 2 matches filter
           ✅ BTC-USDT_4h not in cooldown
           
13:50:02 - Sign API request, POST to Asterdex
           
13:50:05 - Asterdex confirms fill
           Log entry_log.jsonl with order_id=12345678
           Set cooldown: BTC-USDT_4h → 2026-05-29T13:50:00 (expires 2026-05-29T17:50:00)
           
13:50:06 onwards - asterdex_order_tracker.py monitors order
                   Polls every 30 seconds
                   
15:00:00 - Price hits TP (69000)
           Asterdex closes position automatically (TP order executed)
           Order tracker detects TP_HIT
           Log result in entry_log.jsonl
           
17:50:00 - Cooldown expires, can enter BTC-USDT_4h again
```

---

## Safety Constraints

✅ **Read-Only:** Never modifies SIGNALS_MASTER.jsonl  
✅ **Immutable Logs:** entry_log.jsonl is append-only  
✅ **Cooldown Lock:** Prevents duplicate entries per (symbol, TF)  
✅ **Tier Filter:** Only 1/2/3, rejects X/unassigned  
✅ **Position Limits:** Fixed $10 notional per entry (no margin spirals)  
✅ **Error Resilience:** Failed API calls don't crash system, logged for review  

---

## TODO Before Launch

- [ ] asterdex_entry_poster.py - Main engine
- [ ] asterdex_rate_limiter.py - Cooldown logic
- [ ] asterdex_order_tracker.py - Order monitoring
- [ ] asterdex_utils.py - Helper functions
- [ ] asterdex_config.py - Configuration (awaiting API keys)
- [ ] Unit tests for cooldown logic
- [ ] Integration test with Asterdex testnet
- [ ] Live deployment with your API keys

---

## Next Steps

1. Show this structure ✅
2. Code the core files (asterdex_entry_poster.py first)
3. Test with Asterdex testnet
4. Request Asterdex API key + secret
5. Deploy to production

