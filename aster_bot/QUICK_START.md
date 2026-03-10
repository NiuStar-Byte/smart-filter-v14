# 🚀 Quick Start - Asterdex Spot Bot (Project-4)

**5 minutes to auto-trading.**

---

## Step 1: Get Your Private Key

1. Open **Binance Wallet** extension (in Chrome/Edge)
2. Click your **Account** (top left)
3. Click **"Export Private Key"**
4. Enter your password
5. Copy the hex string (starts with `0x`)

---

## Step 2: Create .env File

Create a file: `/Users/geniustarigan/.openclaw/workspace/aster_bot/.env`

```
ASTER_PRIVATE_KEY=0xyourprivatekeyhere
ASTER_WALLET_ADDRESS=0xyourwalletaddress
```

**Save and close.**

---

## Step 3: Test the Bot (Foreground)

```bash
cd /Users/geniustarigan/.openclaw/workspace/aster_bot
python3 asterdex_spot_bot.py
```

**You should see:**
```
[2026-03-10 15:00:00] INFO: 🤖 ASTERDEX SPOT BOT INITIALIZED (Project-4)
[2026-03-10 15:00:00] INFO: 💰 Account balance: $9.0 USDT
[2026-03-10 15:00:00] INFO: 📍 Cycle 1 at 2026-03-10T15:00:00Z
[2026-03-10 15:00:00] INFO: 📌 Opening new position: BTC-USDT
```

**Wait 60 seconds** to see if orders place and execute.

**Stop:** Press `Ctrl+C`

---

## Step 4: Check Trades

```bash
cat /Users/geniustarigan/.openclaw/workspace/aster_bot/spot_trades.jsonl
```

You should see entries like:
```json
{"timestamp": "...", "symbol": "BTC-USDT", "pnl": 0.03, "pnl_pct": 3.00, "status": "CLOSED_TP_HIT"}
```

---

## Step 5: Run in Background

```bash
cd /Users/geniustarigan/.openclaw/workspace/aster_bot
nohup python3 asterdex_spot_bot.py > spot_bot_output.log 2>&1 &
```

**Bot now runs 24/7.**

---

## Monitor It

**Real-time logs:**
```bash
tail -f /Users/geniustarigan/.openclaw/workspace/aster_bot/spot_bot.log
```

**Stop the bot:**
```bash
ps aux | grep asterdex_spot_bot
kill <PID>
```

**Trade history:**
```bash
tail -20 /Users/geniustarigan/.openclaw/workspace/aster_bot/spot_trades.jsonl | jq
```

---

## Configuration

All settings in: `/Users/geniustarigan/.openclaw/workspace/aster_bot/spot_bot_config.py`

```python
TRADING_PAIRS = ["BTC-USDT", "ETH-USDT"]  # What to trade
POSITION_SIZE_USD = 1.0                    # $1 per trade
TP_PERCENT = 3.0                           # Close at +3%
SL_PERCENT = 2.0                           # Close at -2%
CHECK_INTERVAL = 30                        # Check every 30s
```

Change any, restart bot.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No private key" | Check `.env` file exists and has ASTER_PRIVATE_KEY |
| "Failed to get price" | Check internet, Asterdex API status |
| "Operation not permitted" | Make sure `.env` is in `aster_bot/` folder |
| Orders not closing | Check logs: `tail -f spot_bot.log` |

---

**That's it!** Your bot is now trading BTC/ETH on Asterdex Spot with your 9 USDT. 🎉

For details, see `PROJECT-4-SPOT-README.md`.
