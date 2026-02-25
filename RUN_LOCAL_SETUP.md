# SmartFilter Local Development Setup

## Goal
Run SmartFilter directly on your Mac so code changes take effect immediately—no git push delays.

## Prerequisites
- Python 3.9+
- Your KuCoin API keys
- Telegram bot token + chat ID
- Git (for pulling changes)

## Step 1: Clone the Repository

```bash
cd ~/Projects  # or your preferred location
git clone https://github.com/NiuStar-Byte/smart-filter-v14.git
cd smart-filter-v14
```

## Step 2: Install Dependencies

```bash
pip install pandas numpy requests pytz python-dateutil
```

## Step 3: Configure Environment Variables

Create a `.env` file in the repo root. **No KuCoin API credentials needed!** SmartFilter uses only public endpoints.

```bash
cat > .env << 'ENVEOF'
# Telegram Bot Token & Chat ID (required for alerts)
# Get BOT_TOKEN from @BotFather on Telegram
# Get CHAT_ID from https://api.telegram.org/bot<TOKEN>/getUpdates after sending a message
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# SmartFilter Config
CYCLE_SLEEP=60
DRY_RUN=false
DEBUG_FILTERS=false
ENVEOF
```

**Why no KuCoin API keys needed?** SmartFilter fetches from public KuCoin endpoints:
- OHLCV: `api.kucoin.com/api/v1/market/candles` (public)
- Orderbook: `api.kucoin.com/api/v1/market/orderbook` (public)
- No authentication required!

## Step 4: Run SmartFilter Locally

### Option A: Simple Direct Run
```bash
cd smart-filter-v14
python3 main.py
```

### Option B: Run with Output to File (Recommended)
```bash
cd smart-filter-v14
python3 main.py > smartfilter_run.log 2>&1 &
```

Then monitor logs:
```bash
tail -f smartfilter_run.log | grep -E "^\[SIGNAL\]|^\[✅ SENT\]|ERROR|CYCLE SUMMARY"
```

## Step 5: Quick Restart Script

Create `restart.sh` for fast restarts when you make changes:

```bash
#!/bin/bash
echo "Killing existing SmartFilter process..."
pkill -f "python.*main.py" 2>/dev/null || true
sleep 2

echo "Starting SmartFilter..."
cd /path/to/smart-filter-v14
python3 main.py > smartfilter_run.log 2>&1 &

echo "✅ SmartFilter started. PID: $!"
echo "Watch logs: tail -f smartfilter_run.log"
```

Make it executable:
```bash
chmod +x restart.sh
```

Then use:
```bash
./restart.sh
```

## Step 6: Efficient Workflow

### When You Make Code Changes:

1. **Edit the file** (e.g., `main.py`, `smart_filter.py`)
2. **Restart immediately**:
   ```bash
   ./restart.sh
   ```
3. **Watch logs**:
   ```bash
   tail -f smartfilter_run.log | grep "^\[SIGNAL\]\|^\[✅ SENT\]"
   ```

### No git push needed—changes take effect instantly! ✅

## Step 7: Monitor Signals in Real-Time

Watch only the important logs:

```bash
# Show signals as they're generated and sent
tail -f smartfilter_run.log | grep -E "^\[SIGNAL\]|^\[✅ SENT\]|^\[⛔ REJECT\]"

# Show full summary at end of cycle
tail -f smartfilter_run.log | grep "CYCLE SUMMARY" -A 5
```

## Step 8: Sync Latest Changes (When Needed)

If you want to pull latest code from GitHub:

```bash
git pull origin main
./restart.sh
```

---

## Example Workflow (Development)

```bash
# Terminal 1: Watch logs
tail -f smartfilter_run.log | grep "^\[SIGNAL\]\|SENT\|CYCLE"

# Terminal 2: Make changes & restart
cd smart-filter-v14
nano main.py          # Make your fix
./restart.sh          # Restart immediately
# Changes take effect in <2 seconds!
```

## Troubleshooting

**"Module not found"** error:
```bash
pip install -r requirements.txt  # if it exists
# Or manually: pip install pandas numpy requests pytz
```

**Port/PID already in use:**
```bash
pkill -9 -f "python.*main.py"  # Force kill
sleep 2
./restart.sh
```

**Check if running:**
```bash
ps aux | grep "python.*main.py" | grep -v grep
```

---

## Summary

✅ **Local running** = changes take effect instantly  
✅ **No git delays** = development is fast  
✅ **Easy restarts** = one command: `./restart.sh`  
✅ **Live logs** = `tail -f` shows everything  

Ready to develop efficiently! 🚀
