# Aster Trading Bot 🤖

Standalone automated trading bot for Aster Futures with built-in entry/exit criteria.

**No smart-filter integration. No complex signals. Just simple, automated trading.**

---

## Architecture

- **aster_client.py** — Aster API client with EIP-712 signing
- **bot_config.py** — Bot settings (pairs, position size, entry/exit criteria)
- **aster_bot.py** — Main bot loop (entry detection → order placement → monitoring)
- **START_BOT.sh** — Startup script

---

## Trading Criteria

### Entry Signal
- **Price dips below moving average** (Simple MA Cross)
- MA-Fast: 10-bar
- MA-Slow: 20-bar
- Trigger: Price drops **2% below MA-20**
- Interval: 5-minute candles

### Exit Signal
- **Take Profit**: +3% above entry
- **Stop Loss**: -2% below entry

### Position Management
- Fixed position size: **$1 USD per trade** (easy to scale up)
- Pairs: **BTC-USDT, ETH-USDT** (can add more)
- Max 1 position per pair at a time
- Leverage: 1x (no leverage)

---

## Setup

### 1. Prerequisites

```bash
pip3 install eth-account requests
```

### 2. Set Environment Variables

```bash
# In terminal
export ASTER_PRIVATE_KEY=0x...your_private_key...
export ASTER_WALLET_ADDRESS=0x...your_wallet_address...
export ASTER_API_KEY=...optional...  # For higher rate limits
```

**Or in ~/.zshrc or ~/.bash_profile:**

```bash
export ASTER_PRIVATE_KEY=0x...
export ASTER_WALLET_ADDRESS=0x...
export ASTER_API_KEY=...
```

Then:
```bash
source ~/.zshrc
```

### 3. Verify Credentials

```bash
python3 -c "
import os
from eth_account import Account

key = os.getenv('ASTER_PRIVATE_KEY')
account = Account.from_key(key)
print(f'✅ Account: {account.address}')
print(f'✅ Matches ASTER_WALLET_ADDRESS: {account.address == os.getenv(\"ASTER_WALLET_ADDRESS\")}')
"
```

---

## Running the Bot

### Option A: Direct

```bash
cd /Users/geniustarigan/.openclaw/workspace/aster_bot
python3 aster_bot.py
```

### Option B: Via Script

```bash
cd /Users/geniustarigan/.openclaw/workspace/aster_bot
chmod +x START_BOT.sh
./START_BOT.sh
```

### Option C: Background (with nohup)

```bash
nohup python3 aster_bot.py > aster_bot.log 2>&1 &
```

---

## Monitoring

### View Live Logs

```bash
tail -f /Users/geniustarigan/.openclaw/workspace/aster_bot/bot.log
```

### View Trade History

```bash
cat /Users/geniustarigan/.openclaw/workspace/aster_bot/trades.jsonl | jq '.'
```

### Count Trades

```bash
wc -l /Users/geniustarigan/.openclaw/workspace/aster_bot/trades.jsonl
```

---

## Configuration

Edit **bot_config.py** to customize:

```python
# Trading pairs
TRADING_PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

# Position size (USD)
POSITION_SIZE_USD = 1.0  # Change to 5, 10, 50, etc.

# Entry criteria
ENTRY_DIP_PERCENT = 2.0  # Dip below MA-20 (%)
MA_SLOW = 20  # MA period

# Exit criteria
TP_PERCENT = 3.0  # Take profit
SL_PERCENT = 2.0  # Stop loss

# Check interval (seconds)
CHECK_INTERVAL = 30  # Check every 30s
```

---

## Log Output Example

```
[2026-03-10 15:00:00] INFO: ================================================================================
[2026-03-10 15:00:00] INFO: 🤖 ASTER TRADING BOT INITIALIZED
[2026-03-10 15:00:00] INFO: Pairs: ['BTC-USDT', 'ETH-USDT']
[2026-03-10 15:00:00] INFO: Position size: $1
[2026-03-10 15:00:00] INFO: Entry: 2.0% dip below 20-bar MA
[2026-03-10 15:00:00] INFO: TP: +3%, SL: -2%
[2026-03-10 15:00:00] INFO: ================================================================================

[2026-03-10 15:01:30] INFO: ✅ BUY SIGNAL BTC-USDT: price $45000.00 < MA-20 $45909.00 (dip 2%)
[2026-03-10 15:01:31] INFO: 📊 Position tracked: BTC-USDT @ $45000.00 (TP: $46350.00, SL: $44100.00)

[2026-03-10 15:05:45] INFO: 🎯 TAKE PROFIT BTC-USDT: $46350.01 >= $46350.00 (+3.00%)
[2026-03-10 15:05:46] INFO: ✅ Position closed: BTC-USDT @ $46350.01 (TP, P&L: +$0.02 / +3.00%)
```

---

## Trade Log Format (trades.jsonl)

```json
{"timestamp": "2026-03-10T15:01:31Z", "action": "ENTRY", "symbol": "BTC-USDT", "price": 45000.00, "quantity": 0.0001, "order_id": "12345"}
{"timestamp": "2026-03-10T15:05:46Z", "action": "EXIT", "symbol": "BTC-USDT", "price": 46350.01, "quantity": 0.0001, "order_id": "12346", "pnl": 0.02, "pnl_percent": 3.00}
```

---

## API Calls

The bot calls:

**Public (no auth):**
- `GET /api/v3/ticker/price` — Current price
- `GET /api/v3/klines` — Historical candles (for MA calculation)

**Signed (auth required):**
- `POST /api/v3/order` — Place order
- `DELETE /api/v3/order` — Cancel order
- `GET /api/v3/openOrders` — Check open orders
- `GET /api/v3/positionRisk` — Check positions (optional)

---

## Troubleshooting

### "ASTER_PRIVATE_KEY not set"
```bash
export ASTER_PRIVATE_KEY=0x...
export ASTER_WALLET_ADDRESS=0x...
```

### "No orders placed"
- Check logs: `tail -f bot.log`
- Verify API key is correct
- Verify wallet has USDT balance
- Check Aster API status: https://fapi.asterdex.com/api/v3/ping

### "Orders fail with '404'"
- Server time mismatch: Bot will use system time as fallback
- Check: `curl https://fapi.asterdex.com/api/v3/time`

### "Signature invalid"
- Private key format incorrect (must start with `0x`)
- ChainId 1666 is hardcoded — do not change

---

## What's Next?

Once the bot is running and placing trades:

1. **Monitor for 24-48h** — Verify entry/exit logic works
2. **Check P&L** — Review `trades.jsonl` for profitability
3. **Refine criteria** — Adjust MA periods, TP%, SL%, entry dip% based on results
4. **Scale position size** — Increase from $1 to $5, $10, $100 as confidence grows
5. **Add more pairs** — Expand to SOL, ARB, OP, etc. once stable
6. **Consider advanced signals** — Later integrate with smart-filter if profitable

---

## Key Points

✅ **Standalone** — No smart-filter integration  
✅ **Simple** — MA cross + fixed TP/SL  
✅ **Testable** — Run with $1 position size first  
✅ **Scalable** — Easy to increase position size  
✅ **Logged** — Full audit trail in trades.jsonl  
✅ **Safe** — Direct EIP-712 signing, no MetaMask  

---

**Status: Ready to trade 🚀**
