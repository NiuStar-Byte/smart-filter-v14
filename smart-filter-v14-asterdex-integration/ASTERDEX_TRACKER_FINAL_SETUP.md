# ASTERDEX FINAL TRACKER - JUN 7+ ONLY

**Status:** ✅ **PRODUCTION READY** (Jun 8 2026 20:23 GMT+7)

## Overview

Two-part tracking system for positions opened Jun 7, 2026 onwards (cleanest entry period):
- **One-time report:** Quick snapshot of current performance
- **Live monitor:** Real-time tracking with 5-minute auto-refresh

### Why Jun 7+?
- Earlier entries (Jun 1-4) had experimental configurations
- Jun 7+ entries are clean, stable, and production-ready
- Contains 58 verified positions (39 from manual validation + system deduped entries)

---

## 📊 Current Performance (58 Positions - Jun 7+)

| Metric | Value | Status |
|--------|-------|--------|
| **Total Positions** | 58 | ✅ Growing |
| **Win Rate** | 50.9% | ✓ Breakeven+ |
| **Total P&L** | -$6.59 USD | ⚠️ Recent losses from outliers |
| **Avg P&L/trade** | -$0.11 USD | ⚠️ High impact from PORTAL, PARTI, FUN |

### Top Winners
- 🥇 BIO-USDT: +$1.03 (+26%)
- 🥈 KAS-USDT: +$0.32 (+1.7%)
- 🥉 INJ-USDT: +$0.30 (+1.5%)

### Worst Losers
- 📉 PORTAL-USDT: -$2.25 (-11.4%)
- 📉 PARTI-USDT: -$1.86 (-4.8%)
- 📉 FUN-USDT: -$1.66 (-8.3%)

---

## 🚀 COMMAND LINES (Copy & Paste)

### A. ONE-TIME REPORT (Quick Snapshot)

Run this once to see current performance:

```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration && \
python3 asterdex_tracker_final_jun7.py
```

**Output:** Summary metrics, symbol breakdown, top 10 winners, bottom 10 losers

**Run time:** ~1 second

**Use case:** Daily check-in, performance verification, quick review

---

### B. LIVE MONITOR (Real-Time Tracking - 5 Minute Updates)

Run this to monitor positions continuously:

```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration && \
python3 asterdex_live_monitor_jun7.py
```

**Output:** 
- Auto-refreshes every 300 seconds (5 minutes)
- Shows new positions as they appear
- Real-time P&L, WR, and symbol metrics
- Press `Ctrl+C` to stop

**Run time:** Continuous (until you stop it)

**Use case:** Leave running in terminal/screen/tmux for live monitoring

---

## 🔧 Running Both Simultaneously

### Option 1: Two Terminal Windows

**Terminal 1 - Live Monitor:**
```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration && \
python3 asterdex_live_monitor_jun7.py
```

**Terminal 2 - Manual Reports (run report anytime):**
```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration && \
python3 asterdex_tracker_final_jun7.py
```

### Option 2: Screen Session (Recommended)

```bash
# Create screen session with live monitor
screen -S asterdex-monitor
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration
python3 asterdex_live_monitor_jun7.py

# Detach: Press Ctrl+A then D
# Reattach: screen -r asterdex-monitor
# Stop: screen -X -S asterdex-monitor quit
```

### Option 3: Tmux Session (Alternative)

```bash
# Create tmux session
tmux new-session -d -s asterdex
tmux send-keys -t asterdex "cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration && python3 asterdex_live_monitor_jun7.py" Enter

# View: tmux attach -t asterdex
# Detach: Ctrl+B then D
# Kill: tmux kill-session -t asterdex
```

---

## 📈 What to Monitor

### Live Monitor Updates Every 5 Minutes
- **New positions detected** ✨ (shows when positions added)
- **Running totals** (positions, wins, losses, P&L)
- **Recent closed positions** (last 5 with P&L)
- **Win rate trending** (should aim for >55%)

### Expected Behavior
- Positions open randomly throughout the day
- Closed positions appear 1-24 hours after opening
- P&L should accumulate (hopefully positive!)
- WR should stabilize after 50+ positions

---

## 🔄 Adjusting Update Interval

If 5 minutes is too frequent or too slow:

### Edit update interval in `asterdex_live_monitor_jun7.py`:

```python
monitor = AsterdexLiveMonitor(update_interval=300)  # Change 300 to your interval in seconds

# Examples:
# update_interval=60    # 1 minute
# update_interval=180   # 3 minutes
# update_interval=300   # 5 minutes (default)
# update_interval=600   # 10 minutes
```

Then restart the monitor.

---

## 📁 Files Involved

| File | Purpose | Size |
|------|---------|------|
| `asterdex_tracker_final_jun7.py` | One-time report script | 5.6 KB |
| `asterdex_live_monitor_jun7.py` | Live monitor script | 6.5 KB |
| `ASTERDEX_POSITIONS_LIVE.jsonl` | Position database (auto-updated) | Grows daily |

---

## 🎯 Next Steps

1. **Start live monitor now:**
   ```bash
   cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration && \
   python3 asterdex_live_monitor_jun7.py
   ```

2. **Check report once daily:**
   ```bash
   cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration && \
   python3 asterdex_tracker_final_jun7.py
   ```

3. **Monitor outliers:**
   - PORTAL, PARTI, FUN are losing badly
   - Need to investigate why these symbols are underperforming
   - Consider filtering these combos from entry posting

4. **Adjust interval if needed:**
   - If CPU/RAM impact is high: increase to 600s (10 min)
   - If you want more frequent updates: decrease to 120s (2 min)

---

## 🐛 Troubleshooting

### "ASTERDEX_POSITIONS_LIVE.jsonl not found"
**Solution:** Make sure you're in the correct directory:
```bash
cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration
ls ASTERDEX_POSITIONS_LIVE.jsonl
```

### Live monitor not updating
**Solution:** Check that asterdex_entry_poster.py is still running and adding positions

### Screen/Tmux session lost
**Solution:** Recreate with the commands above, it will read the same position file

---

## 📞 Quick Reference

| Need | Command |
|------|---------|
| **One-time report** | `python3 asterdex_tracker_final_jun7.py` |
| **Live monitor** | `python3 asterdex_live_monitor_jun7.py` |
| **Stop monitor** | `Ctrl+C` (in terminal) or `screen -X -S asterdex-monitor quit` |
| **View old report** | Run `python3 asterdex_tracker_final_jun7.py` again |
| **Check file sync** | `wc -l ASTERDEX_POSITIONS_LIVE.jsonl` |
| **Last 10 lines** | `tail -10 ASTERDEX_POSITIONS_LIVE.jsonl` |

---

**Created:** Jun 8 2026 20:23 GMT+7  
**Scope:** Jun 7 onwards (58 verified positions)  
**Status:** ✅ Production Ready
