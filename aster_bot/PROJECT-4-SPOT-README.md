# 🚀 Project-4: Asterdex Spot Trading Bot

Auto-trades BTC/ETH on Asterdex Spot (mainnet) using your Binance Wallet.

**Current Status:** ✅ Built, ready to run

---

## 📋 Overview

| Parameter | Value |
|-----------|-------|
| **Trading Pairs** | BTC-USDT, ETH-USDT |
| **Entry Strategy** | Market order, 1 USDT per trade |
| **Exit TP** | +3% |
| **Exit SL** | -2% |
| **Starting Balance** | 9 USDT |
| **Check Interval** | 30s |

---

## 🔧 Setup

### 1. **Environment Variables** (.env file)

Create `/Users/geniustarigan/.openclaw/workspace/aster_bot/.env`:

```bash
# Binance Wallet / EIP-712 Signing
ASTER_PRIVATE_KEY=<your_private_key>
ASTER_WALLET_ADDRESS=<your_wallet_address>

# API Key (optional, for read-only access)
ASTER_API_KEY=<optional_api_key>
```

**To get your private key from Binance Wallet:**

1. Open Binance Wallet extension
2. Click "Account" → "Export Private Key"
3. Enter password
4. Copy the hex string (usually starts with `0x`)

**Your wallet address:** You can see it in Binance Wallet at the top

### 2. **Install Dependencies** (if not already done)

```bash
cd /Users/geniustarigan/.openclaw/workspace/aster_bot
pip install -r requirements.txt
```

Or manually:

```bash
pip install requests eth-account
```

### 3. **Test the Setup**

```bash
cd /Users/geniustarigan/.openclaw/workspace/aster_bot
python3 -c "from spot_bot_config import *; print(f'Config loaded: {TRADING_PAIRS}, TP={TP_PERCENT}%, SL={SL_PERCENT}%')"
```

---

## ▶️ Running the Bot

### **Option A: Foreground (Testing)**

```bash
cd /Users/geniustarigan/.openclaw/workspace/aster_bot
python3 asterdex_spot_bot.py
```

Watch logs in real-time:
```
[2026-03-10 15:00:00] INFO: 🤖 ASTERDEX SPOT BOT INITIALIZED (Project-4)
[2026-03-10 15:00:00] INFO: Pairs: ['BTC-USDT', 'ETH-USDT']
[2026-03-10 15:00:00] INFO: 💰 Account balance: $9.0 USDT
[2026-03-10 15:00:30] INFO: 📍 Cycle 1 at 2026-03-10T15:00:30Z
[2026-03-10 15:00:30] INFO: 📌 Opening new position: BTC-USDT
```

Stop: `Ctrl+C`

### **Option B: Background (Production)**

```bash
cd /Users/geniustarigan/.openclaw/workspace/aster_bot
nohup python3 asterdex_spot_bot.py > spot_bot_output.log 2>&1 &
```

Find process:
```bash
ps aux | grep asterdex_spot_bot
```

Stop:
```bash
kill <PID>
```

---

## 📊 Log Files

**Real-time log:**
```bash
tail -f /Users/geniustarigan/.openclaw/workspace/aster_bot/spot_bot.log
```

**Trade history (JSONL):**
```bash
cat /Users/geniustarigan/.openclaw/workspace/aster_bot/spot_trades.jsonl | jq
```

Example trade record:
```json
{
  "timestamp": "2026-03-10T15:05:30.123456Z",
  "symbol": "BTC-USDT",
  "entry_price": 45000.50,
  "exit_price": 46350.00,
  "quantity": 0.0000222,
  "pnl": 0.0300,
  "pnl_pct": 3.00,
  "tp_price": 46350.515,
  "sl_price": 44100.49,
  "status": "CLOSED_TP_HIT"
}
```

---

## 🔍 How It Works

### **Cycle (30s interval)**

1. **Check existing positions**
   - If TP hit (+3%): Close with market SELL order
   - If SL hit (-2%): Close with market SELL order

2. **Try to open new positions**
   - For each pair (BTC-USDT, ETH-USDT):
     - Skip if already OPEN
     - If USDT balance ≥ $1: Place market BUY order
     - Log position (entry price, TP, SL)

3. **Sleep 30s, repeat**

### **Order Flow**

```
BUY (Market)
  ↓
Entry Price: $45,000.50
TP: $46,350.52 (45000.50 * 1.03)
SL: $44,100.49 (45000.50 * 0.98)
  ↓
[Monitor every 30s]
  ↓
Price ≥ $46,350.52? → SELL (TP_HIT)
Price ≤ $44,100.49? → SELL (SL_HIT)
```

---

## ⚙️ Configuration

Edit `/Users/geniustarigan/.openclaw/workspace/aster_bot/spot_bot_config.py`:

```python
# Change trading pairs
TRADING_PAIRS = ["BTC-USDT", "ETH-USDT"]  # Add/remove symbols

# Change position size
POSITION_SIZE_USD = 1.0  # Max $1 per trade

# Change TP/SL
TP_PERCENT = 3.0  # Take profit at +3%
SL_PERCENT = 2.0  # Stop loss at -2%

# Change check interval
CHECK_INTERVAL = 30  # Seconds between cycles

# Change API endpoint
ASTERDEX_SPOT_BASE_URL = "https://sapi.asterdex.com"  # Mainnet
# ASTERDEX_SPOT_BASE_URL = "https://sapi.asterdex-testnet.com"  # Testnet (not supported yet)
```

After editing, restart the bot.

---

## 🚨 Troubleshooting

### **"Operation not permitted" when reading private key**

Solution: Make sure `.env` file is in the aster_bot directory:
```bash
ls -la /Users/geniustarigan/.openclaw/workspace/aster_bot/.env
```

### **"No private key provided. Read-only mode only."**

Solution: `.env` file is missing or ASTER_PRIVATE_KEY is not set. Create it with your private key.

### **"Failed to get price for BTC-USDT"**

Possible causes:
- Network issue (check internet connection)
- Symbol not found (BTC-USDT vs BTCUSDT format mismatch)
- Asterdex API down

### **Orders not closing at TP/SL**

Possible causes:
- API key doesn't have TRADE permission
- Private key is incorrect
- Market moved too fast (SL triggered but not filled)

### **Balance stays at $9.0, no orders placed**

Possible causes:
- Bot waiting for 30s between cycles
- Symbol pair doesn't exist on Asterdex Spot
- Insufficient API permissions

---

## 📝 Files

| File | Purpose |
|------|---------|
| `asterdex_spot_bot.py` | Main bot logic |
| `spot_bot_config.py` | Configuration |
| `aster_client.py` | API client (extended with Spot methods) |
| `spot_bot.log` | Real-time logs |
| `spot_trades.jsonl` | Trade history |

---

## 🔐 Security Notes

✅ **Safe to run:**
- Private key loaded from `.env` (stays local)
- All signing happens client-side (never sent to server)
- Bot only reads/writes to local files
- No sensitive data logged

❌ **Do NOT:**
- Commit `.env` to Git
- Share your private key
- Run bot on untrusted machines
- Leave private key in code

---

## 📊 Example Session

```bash
$ python3 asterdex_spot_bot.py

[2026-03-10 15:00:00] INFO: ════════════════════════════════════════════════
[2026-03-10 15:00:00] INFO: 🤖 ASTERDEX SPOT BOT INITIALIZED (Project-4)
[2026-03-10 15:00:00] INFO: ════════════════════════════════════════════════
[2026-03-10 15:00:00] INFO: Pairs: ['BTC-USDT', 'ETH-USDT']
[2026-03-10 15:00:00] INFO: Position size: $1.0 per trade
[2026-03-10 15:00:00] INFO: Entry: Market order
[2026-03-10 15:00:00] INFO: TP: +3.0%, SL: -2.0%
[2026-03-10 15:00:00] INFO: Check interval: 30s
[2026-03-10 15:00:00] INFO: ════════════════════════════════════════════════
[2026-03-10 15:00:00] INFO: 💰 Account balance: $9.0 USDT

[2026-03-10 15:00:00] INFO: 🚀 Starting bot loop...

[2026-03-10 15:00:00] INFO: 📍 Cycle 1 at 2026-03-10T15:00:00Z
[2026-03-10 15:00:01] INFO: 📌 Opening new position: BTC-USDT
[2026-03-10 15:00:02] DEBUG: Price BTC-USDT: $45123.50
[2026-03-10 15:00:03] INFO: 📈 BUY BTC-USDT: 0.00002213 @ $45123.50 (total: $1.0)
[2026-03-10 15:00:04] INFO: ✅ Buy order placed: 1234567890
[2026-03-10 15:00:04] INFO:   Entry: $45,123.50
[2026-03-10 15:00:04] INFO:   TP: $46,477.20 (+3.0%)
[2026-03-10 15:00:04] INFO:   SL: $44,221.03 (-2.0%)
[2026-03-10 15:00:05] INFO: 💰 Account balance: $8.0 USDT

[2026-03-10 15:00:05] INFO: 📌 Opening new position: ETH-USDT
[2026-03-10 15:00:06] DEBUG: Price ETH-USDT: $2850.30
[2026-03-10 15:00:07] INFO: 📈 BUY ETH-USDT: 0.00035046 @ $2850.30 (total: $1.0)
[2026-03-10 15:00:08] INFO: ✅ Buy order placed: 1234567891
[2026-03-10 15:00:08] INFO:   Entry: $2,850.30
[2026-03-10 15:00:08] INFO:   TP: $2,936.81 (+3.0%)
[2026-03-10 15:00:08] INFO:   SL: $2,793.29 (-2.0%)
[2026-03-10 15:00:09] INFO: 💰 Account balance: $7.0 USDT

[2026-03-10 15:00:09] INFO: ⏸️  Sleeping 30s until next cycle...

[2026-03-10 15:00:39] INFO: 📍 Cycle 2 at 2026-03-10T15:00:39Z
[2026-03-10 15:00:40] DEBUG: Price BTC-USDT: $46,500.00
[2026-03-10 15:00:40] INFO: 🎯 TP HIT for BTC-USDT: $46,500.00 >= $46,477.20
[2026-03-10 15:00:41] INFO: 📉 SELL BTC-USDT: 0.00002213 @ $46,500.00 (TP_HIT)
[2026-03-10 15:00:42] INFO: ✅ Sell order placed: 1234567892 (TP_HIT)
[2026-03-10 15:00:42] INFO:   Exit: $46,500.00
[2026-03-10 15:00:42] INFO:   P&L: $1.03 (+3.07%)
[2026-03-10 15:00:43] INFO: 📝 Trade logged: BTC-USDT P&L=$1.03 (+3.07%)
[2026-03-10 15:00:43] INFO: 💰 Account balance: $8.03 USDT

...continuing...
```

---

## 🎯 Next Steps

1. **Create `.env` file** with your private key
2. **Test foreground:** `python3 asterdex_spot_bot.py` (watch logs for 2-3 cycles)
3. **Verify trades:** Check `spot_trades.jsonl` file
4. **Run background:** `nohup python3 asterdex_spot_bot.py > spot_bot_output.log 2>&1 &`
5. **Monitor daily:** `tail -f spot_bot.log` to track P&L

---

**Questions?** Check the logs or ask Nox. 🙌
