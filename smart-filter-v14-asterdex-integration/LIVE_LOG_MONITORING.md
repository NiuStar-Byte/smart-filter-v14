# 🚀 Asterdex Entry Poster - Live Log Monitoring Guide

**Status:** ✅ OPERATIONAL (June 1 2026)

---

## Quick Start

### Run with Live Logs (Terminal)
```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration
./run_with_logs.sh
```

This will:
1. ✅ Source credentials from `.env`
2. ✅ Kill any existing process
3. ✅ Start fresh asterdex_entry_poster.py
4. ✅ Display live logs in real-time

### View Logs Only (If Already Running)
```bash
./tail_logs.sh
```

This will:
- Show last 50 lines
- Then stream new logs live
- Press `Ctrl+C` to stop

---

## All Available Commands

| Command | Purpose |
|---------|---------|
| `./run_with_logs.sh` | Start fresh + live logs |
| `./run_with_logs.sh --attach` | Attach to existing running instance |
| `./tail_logs.sh` | View live logs |
| `./tail_logs.sh last` | Show last 100 lines only |
| `./tail_logs.sh search ERROR` | Search for keyword in logs |
| `./tail_logs.sh search "POSTED"` | Search for successful posts |

---

## How It Works

### Problem (Before)
```bash
# This failed with "credentials not found" error:
python3 asterdex_entry_poster.py
# ❌ ValueError: ASTERDEX PRO API V3 wallet credentials not found!
```

### Solution (Now)
The `run_with_logs.sh` script:
1. **Sources .env file** → Environment variables loaded
2. **Verifies credentials** → Check if ASTER_* variables exist
3. **Starts asterdex_entry_poster.py** → With credentials available
4. **Pipes output to log file** → `/tmp/asterdex_entry_poster_live.log`
5. **Tails log file** → Shows real-time updates

---

## Log File Details

| Property | Value |
|----------|-------|
| **Location** | `/tmp/asterdex_entry_poster_live.log` |
| **Cleared on** | Each fresh `./run_with_logs.sh` start |
| **Retained when** | Using `./tail_logs.sh` to view existing instance |
| **Format** | `[TIMESTAMP] [LOG_LEVEL] Message` |

---

## What the Logs Show

### ✅ Success Log Examples
```
[2026-06-01 13:40:57 GMT+7] [INFO] ✅ Web3 account initialized: 0x78e3809797d29db9d494873bde595aaef55ddd61
[2026-06-01 13:41:35 GMT+7] [INFO] ✅ [TIER_FILTER] SOL-USDT Tier-1 found (entry: 82.97)
[2026-06-01 13:41:36 GMT+7] [INFO] ✅ [ENTRY_POSTER] SOL-USDT posted to Asterdex (Order ID: 12345)
[2026-06-01 13:41:40 GMT+7] [INFO] ✅ [TP/SL] Stop Market order placed @ 2.51 (SL)
[2026-06-01 13:41:41 GMT+7] [INFO] ✅ [TP/SL] Take Profit Market order placed @ 2.722 (TP)
```

### ⛔ Rejection Log Examples (Normal - Low Tier Signals)
```
[2026-06-01 13:50:24 GMT+7] [WARNING] [ENTRY_POSTER] ⛔ REJECTED: DOGE-USDT 30min - Tier Tier-X not in filter [1, 2, 3]
[2026-06-01 13:50:24 GMT+7] [WARNING] [ENTRY_POSTER] ⛔ REJECTED: ENA-USDT 1h - Tier Tier-X not in filter [1, 2, 3]
```
(These are normal - Tier-X signals are filtered out, only Tier 1/2/3 post to Asterdex)

### 🔴 Error Examples (Would Show Problems)
```
[2026-06-01 13:41:50 GMT+7] [ERROR] [ENTRY_POSTER] ❌ Failed to post SOL-USDT: Rate limit exceeded
[2026-06-01 13:41:55 GMT+7] [ERROR] [ENTRY_POSTER] ❌ Symbol SOL-USDT not found on Asterdex
```

---

## Monitoring Checklist

When viewing live logs, check for:

- ✅ **Initialization:** "Web3 account initialized" appears on start
- ✅ **Polling:** "Polling SIGNALS_MASTER" every 5 seconds
- ✅ **Tier Filtering:** See appropriate Tier-1/2/3 detections
- ✅ **Rate Limiting:** No same (symbol, timeframe) posted twice in cooldown period
- ❌ **Rejections:** Tier-X and low-tier signals rejected (NORMAL)
- ❌ **No Errors:** Should see no [ERROR] level messages (or only temporary API issues)

---

## Permanent Background Operation

The Asterdex integration runs as a **background process** on Mac startup via LaunchAgent:
- Process automatically restarts on crash
- Runs independently of terminal
- Logs still written to `/tmp/asterdex_entry_poster_live.log`

**To check background status:**
```bash
pgrep -f "asterdex_entry_poster.py"
# Returns PID if running, empty if stopped
```

---

## Environment Variables (.env Setup)

The `.env` file should contain:
```bash
export ASTER_MAIN_ACCOUNT="0xDad4a337c537b4590e67C53b575Fe9Fe6AacCD42"
export ASTER_API_WALLET_ADDRESS="0x78e3809797d29db9d494873bde595aaef55ddd61"
export ASTER_API_WALLET_PRIVATE_KEY="0x5ab9d118ab9c8e5d1d1206e3d856e742c5f4e60ff0ff530c3ecba7aeef1e68b4"
```

Location: `/Users/geniustarigan/.openclaw/workspace/.env`

---

## Troubleshooting

### Scripts don't work (permission denied)
```bash
chmod +x run_with_logs.sh tail_logs.sh
```

### Log file not found
```bash
# Logs only appear after first `./run_with_logs.sh` start
./run_with_logs.sh
```

### Process keeps crashing
```bash
# Check last error
tail -50 /tmp/asterdex_entry_poster_live.log | grep ERROR

# Restart with full environment:
source /Users/geniustarigan/.openclaw/workspace/.env
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration
./run_with_logs.sh
```

---

## Summary

| Action | Command |
|--------|---------|
| **Run fresh + see logs** | `./run_with_logs.sh` |
| **View logs (existing)** | `./tail_logs.sh` |
| **Attach to running** | `./run_with_logs.sh --attach` |
| **Check process status** | `pgrep -f "asterdex_entry_poster.py"` |
| **Search logs** | `./tail_logs.sh search "POSTED"` |

---

**Last Updated:** June 1 2026, 13:49 GMT+7
**Status:** ✅ Ready for permanent monitoring
