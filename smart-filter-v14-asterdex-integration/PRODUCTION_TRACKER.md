# Production Tracking - Smart Filter v14 → Asterdex Integration

**Status:** 🔴 **LIVE PRODUCTION MODE ENABLED**

---

## **Production Configuration**

| Setting | Value | Status |
|---------|-------|--------|
| **Endpoint** | https://fapi.asterdex.com | ✅ PRODUCTION |
| **API Key Source** | Environment variable | ✅ SECURE |
| **DRY_RUN_MODE** | FALSE | ✅ REAL ORDERS |
| **PRODUCTION_MODE** | TRUE | ✅ LIVE |

---

## **Tracking Metrics**

### Real-Time Tracking Files

| File | Purpose | Format |
|------|---------|--------|
| `data/entry_log.jsonl` | All entry attempts (immutable) | JSONL |
| `data/active_orders.jsonl` | Currently open orders | JSONL |
| `data/cooldown_state.json` | (symbol, TF) cooldown tracking | JSON |
| `logs/asterdex_poster_YYYY-MM-DD.log` | Execution logs (production) | Text |

---

## **Entry Log Schema (Production)**

Every real entry includes:

```json
{
  "timestamp": "2026-05-29T15:18:00Z",
  "signal_uuid": "abc-123-def",
  "symbol": "BTC-USDT",
  "direction": "LONG",
  "timeframe": "4h",
  "tier": 2,
  
  "entry_direction": "LONG",
  "entry_price": 67500.5,
  "tp_price": 69000.0,
  "sl_price": 66000.0,
  "margin_usdt": 1.0,
  "leverage": 10,
  "margin_mode": "ISOLATED",
  "order_type": "LIMIT",
  
  "quantity": 0.00014815,
  "notional": 10.0,
  
  "status": "COMPLETE",
  "entry_order_id": "12345678",
  "tp_order_id": "12345679",
  "sl_order_id": "12345680",
  
  "asterdex_response": {
    "orderId": 12345678,
    "symbol": "BTCUSDT",
    "status": "NEW",
    "type": "LIMIT",
    "avgPrice": "0",
    "executedQty": "0",
    "cumQuote": "0"
  }
}
```

---

## **Production Checklist**

✅ **API Authentication**
- [ ] ASTERDEX_API_KEY set in environment
- [ ] ASTERDEX_API_SECRET set in environment
- [ ] Both keys from "AutoTradingBot" API

✅ **Configuration**
- [ ] ASTERDEX_ENDPOINT = https://fapi.asterdex.com
- [ ] DRY_RUN_MODE = False
- [ ] PRODUCTION_MODE = True

✅ **Monitoring**
- [ ] Entry logs being written to data/entry_log.jsonl
- [ ] Logs rotating daily (logs/asterdex_poster_YYYY-MM-DD.log)
- [ ] Cooldown state persisting (data/cooldown_state.json)
- [ ] Active orders tracked (data/active_orders.jsonl)

---

## **How to Start Production**

### 1. Set Environment Variables

```bash
# Option A: Set in shell
export ASTERDEX_API_KEY="your_key_from_AutoTradingBot_API"
export ASTERDEX_API_SECRET="your_secret_from_AutoTradingBot_API"

# Option B: Create .env file (don't commit!)
echo 'export ASTERDEX_API_KEY="..."' >> ~/.openclaw/workspace/.env
echo 'export ASTERDEX_API_SECRET="..."' >> ~/.openclaw/workspace/.env
chmod 600 ~/.openclaw/workspace/.env
source ~/.openclaw/workspace/.env
```

### 2. Verify Configuration

```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration

# Test that config loads (will fail if API keys missing)
python3 -c "from asterdex_config import ASTERDEX_API_KEY, ASTERDEX_ENDPOINT; print(f'✅ Endpoint: {ASTERDEX_ENDPOINT}')"
```

### 3. Start Production Service

```bash
# Option A: Run directly
python3 asterdex_entry_poster.py

# Option B: Run in background (with logging)
nohup python3 asterdex_entry_poster.py > logs/asterdex_poster_background.log 2>&1 &

# Option C: Use LaunchAgent (auto-restart on failure)
# See PRODUCTION_SETUP.md for LaunchAgent instructions
```

### 4. Monitor in Real-Time

```bash
# Watch logs as entries are posted
tail -f logs/asterdex_poster_*.log

# Check entry log
tail -10 data/entry_log.jsonl | python3 -m json.tool

# Check cooldown state
cat data/cooldown_state.json | python3 -m json.tool
```

---

## **Daily Metrics Summary**

Create a daily summary script to track:

```json
{
  "date": "2026-05-29",
  "signals_fired": 1598,
  "entries_posted": 45,
  "entry_success_rate": "100%",
  "tier_distribution": {
    "tier_1": 15,
    "tier_2": 20,
    "tier_3": 10
  },
  "symbols_traded": 32,
  "total_notional_usd": 450,
  "margin_used_usdt": 45,
  "orders_pending_tp": 12,
  "orders_pending_sl": 8
}
```

---

## **⚠️ Critical Safeguards**

**DO:**
- ✅ Keep API keys in environment variables only
- ✅ Rotate API keys monthly
- ✅ Monitor logs daily
- ✅ Track entry_log.jsonl immutability
- ✅ Validate all 8 infos are captured
- ✅ Test with small margin first ($1 USDT)

**DON'T:**
- ❌ Commit API keys to git
- ❌ Share API secrets in chat
- ❌ Increase margin without testing
- ❌ Disable TP/SL orders
- ❌ Run multiple instances (cooldown conflict)

---

## **Troubleshooting**

### Error: "ASTERDEX API credentials not found"
- Check environment variables: `echo $ASTERDEX_API_KEY`
- Verify .env file is sourced: `source ~/.openclaw/workspace/.env`

### Error: "API error: 401 Unauthorized"
- API key/secret mismatch
- Verify AutoTradingBot API credentials are correct
- Check for extra whitespace in env vars

### Error: "Order failed: Insufficient balance"
- Check Asterdex account has USDT balance
- Verify balance >= margin amount ($1 per entry)
- Futures account might be separate from spot

### Orders not executing
- Check if LIMIT orders are waiting for price
- Verify TP/SL prices are realistic
- Check cooldown not blocking new entries

---

## **Status Dashboard**

Last Updated: 2026-05-29 15:18 GMT+7

| Metric | Value | Trend |
|--------|-------|-------|
| Configuration | ✅ PRODUCTION | Locked |
| API Keys | ✅ ENV VARS | Secure |
| Endpoint | ✅ LIVE | Active |
| DRY-RUN Mode | ✅ OFF | Real Orders |
| Entry Log | ✅ ENABLED | Immutable |
| Tracking | ✅ COMPLETE | All 8 Infos |

---

**Ready for live trading. Monitor logs closely for first 24 hours.** 🚀
