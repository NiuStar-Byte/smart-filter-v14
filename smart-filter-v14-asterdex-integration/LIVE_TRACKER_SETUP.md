# ASTERDEX LIVE TRACKER - Setup & Usage

## Overview
- **asterdex_live_tracker_final.py** - Display all tracked positions with complete metrics
- **asterdex_auto_updater.py** - Auto-fetch and update new closed positions (cron-friendly)

## One-Time Setup

### 1. Initialize Tracker Database
```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration
python3 asterdex_live_tracker_final.py
```

This creates `ASTERDEX_POSITIONS_LIVE.jsonl` with all 37 verified baseline positions.

### 2. Set Up Auto-Update Cron Job (Optional)
Run every 30 minutes to auto-fetch new closed positions:

```bash
# Edit crontab
crontab -e

# Add this line:
*/30 * * * * cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration && python3 asterdex_auto_updater.py >> /tmp/asterdex_auto_update_cron.log 2>&1
```

Or for every hour:
```bash
0 * * * * cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration && python3 asterdex_auto_updater.py >> /tmp/asterdex_auto_update_cron.log 2>&1
```

## Daily Usage

### Display Current Tracker (Manual)
```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration
python3 asterdex_live_tracker_final.py
```

**Output includes:**
- All 37+ positions in chronological order
- Complete metrics table (entry/exit price, qty, P&L, duration)
- Summary with:
  - Total positions count
  - Win/Loss/Breakeven breakdown
  - **Overall WR: Wins / (Wins + Losses)**
  - Total and average P&L
  - **Avg Risk:Reward ratio**
  - **Avg TP Duration (hours)**
  - **Avg SL Duration (hours)**

### Auto-Update (With Cron)
Cron runs every 30/60 minutes, fetches new closed positions, and updates database:

```bash
# Check logs
tail -50 /tmp/asterdex_auto_update_cron.log

# Manual run
python3 asterdex_auto_updater.py
```

## Files

| File | Purpose |
|------|---------|
| `asterdex_live_tracker_final.py` | Display all positions + metrics |
| `asterdex_auto_updater.py` | Auto-fetch + update new positions |
| `ASTERDEX_POSITIONS_LIVE.jsonl` | Persistent position database |
| `asterdex_auto_update.log` | Update log file |

## Metrics Explained

### Overall WR (Win Rate)
```
WR = Wins / (Wins + Losses)
Example: 15 / (15 + 19) = 44.1%
```

### Avg Risk:Reward (RR)
Average profit/loss ratio per trade
```
RR = (Avg Profit) / (Avg Loss)
Example: 0.03:1.0 means avg profit $0.03 per $1.0 of loss
```

### Avg TP Duration
Average time (hours) from entry to TP hit
```
Example: 4.42 hours = ~0.18 days
Shows how quick winning trades close
```

### Avg SL Duration
Average time (hours) from entry to SL hit
```
Example: 2.42 hours = ~0.10 days
Shows how quick losing trades close
(Faster SL = better risk management)
```

## Data Persistence

All positions saved to `ASTERDEX_POSITIONS_LIVE.jsonl`:
- One JSON object per line
- Sorted chronologically
- Appended (never deleted)
- Can be imported to Excel/Google Sheets

## Troubleshooting

### "ModuleNotFoundError: No module named 'aster_v3_auth'"
Add PYTHONPATH:
```bash
PYTHONPATH=/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main:$PYTHONPATH python3 asterdex_live_tracker_final.py
```

### Cron not working?
Check if Python paths are absolute:
```bash
which python3
# Use full path in crontab, e.g.: /usr/bin/python3
```

### New positions not showing?
Run updater manually:
```bash
python3 asterdex_auto_updater.py
tail -20 asterdex_auto_update.log
```

Then display tracker:
```bash
python3 asterdex_live_tracker_final.py
```
