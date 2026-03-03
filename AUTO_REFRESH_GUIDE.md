# ✅ AUTO-REFRESH MONITORING (Every 5 Seconds)

## 🚀 Quick Start

Just run this one command and watch the live updates:

```bash
python3 track_phase2_fixed.py
```

**That's it!** The script will:
- ✅ Display full Phase 2-FIXED performance metrics
- ✅ Update every 5 seconds automatically
- ✅ Clear the screen and show fresh data
- ✅ Show timestamp of last update
- ✅ Track BEAR SHORT recovery (key metric)
- ✅ Press Ctrl+C to stop

---

## 📊 What You'll See

### Header (Updates Every 5 Seconds)
```
====================================================================================================
📊 PHASE 2-FIXED PERFORMANCE TRACK (PHASE 2-FIXED ONLY)
   🔄 AUTO-REFRESH (every 5 seconds)
   Last updated: 2026-03-03 12:41:50 UTC
====================================================================================================
```

### Performance by Regime & Direction
```
🌊 BEAR REGIME
  LONG     | Total: 3    | Closed: 2    | TP: 1   SL: 1   TO: 0   | WR:   50.0% ✅
  SHORT    | Total: 2    | Closed: 1    | TP: 0   SL: 1   TO: 0   | WR:    0.0% 🚨

🌊 BULL REGIME
  LONG     | Total: 3    | Closed: 0    | TP: 0   SL: 0   TO: 0   | WR:    0.0% ⚠️

🌊 RANGE REGIME
  LONG     | Total: 1    | Closed: 0    | TP: 0   SL: 0   TO: 0   | WR:    0.0% ⚠️
```

### Key Metrics
```
📈 SUMMARY
  Total signals (Phase 2-FIXED): 24
  Deployment time: 2026-03-03 10:36:00 UTC
  Current time: 2026-03-03 12:41:50 UTC
  Elapsed: 2.1 hours

🎯 KEY METRIC: BEAR SHORT WR
   Current: 0.0% (target: 25%+)
   Signals: 2 | Closed: 1
   ⚠️  Need more data or adjustment
```

### Status Indicators
```
⏱️  Next update in 5 seconds... (press Ctrl+C to stop)
```

---

## 📋 All Commands

| Command | What It Does | Update Freq |
|---------|----------|----------|
| `python3 track_phase2_fixed.py` | Full report with all regimes | Auto (5s) |
| `python3 track_phase2_fixed.py --once` | Single snapshot (no auto-refresh) | Once |
| `python3 track_phase2_fixed.py --short` | SHORT signals only | Auto (5s) |
| `python3 track_phase2_fixed.py --bear` | BEAR regime only | Auto (5s) |
| `python3 track_phase2_fixed.py --watch` | Live daemon logs (PHASE2-FIXED tags) | Real-time |

---

## 🎯 How to Use for Monitoring

### **Best Practice: Run in a Terminal Window**

Keep this running in a dedicated terminal while Phase 2-FIXED is active:

```bash
cd ~/.openclaw/workspace/smart-filter-v14-main
python3 track_phase2_fixed.py
```

**Watch for:**
- ✅ BEAR SHORT WR increasing (target: 15%+ by Day 3, 25%+ by Day 7)
- ✅ Total signal count growing (expect ~8-12/day)
- ✅ LONG WR staying stable (no regression)
- 🚨 Any dramatic drops in WR (signal quality issue)

### **Quick Snapshots: Use --once**

If you just want a snapshot without the endless auto-refresh:

```bash
python3 track_phase2_fixed.py --once
```

### **Focus on SHORT: Use --short**

To watch only SHORT signals (the key focus area):

```bash
python3 track_phase2_fixed.py --short
```

### **Watch Raw Logs: Use --watch**

To see raw daemon logs with PHASE2-FIXED tags:

```bash
python3 track_phase2_fixed.py --watch
```

---

## 📊 Reading the Metrics

### **Columns Explained**
- **Total:** Number of signals fired for this combo
- **Closed:** How many signals resolved (TP, SL, or TIMEOUT)
- **TP:** Target Profit hit (winner)
- **SL:** Stop Loss hit (loser)
- **TO:** Timeout (signal expired without reaching either)
- **WR:** Win Rate % (TP / Closed)
- **P&L:** Total P&L in USD

### **Status Emojis**
- ✅ **Green** = WR > 20% (good)
- ⚠️ **Yellow** = WR 15-20% (okay, needs monitoring)
- 🚨 **Red** = WR < 15% for SHORT (concerning)

### **Examples**
```
SHORT in BEAR | WR: 25% ✅     = Good (favorable combo working)
SHORT in BULL | WR: 10% 🚨     = Bad (counter-trend not working)
LONG in BULL  | WR: 30% ✅     = Excellent (favorable combo strong)
```

---

## 💡 Tips

1. **Run continuously during monitoring periods**
   - Let it refresh every 5 seconds
   - Don't restart the script
   - Keep window visible

2. **Check daily snapshots**
   - Once/day at fixed times (e.g., 08:00, 14:00, 20:00 GMT+7)
   - Record the BEAR SHORT WR trend
   - Note total signal count growth

3. **Use --watch for debugging**
   - If WR drops suddenly, watch raw logs
   - Look for gate failures or routing changes
   - Check for daemon errors

4. **Stop with Ctrl+C**
   - Press Ctrl+C at any time
   - Script will exit gracefully
   - No data loss

---

## 🔄 How Auto-Refresh Works

1. **Initial Load** (First run)
   - Loads all signals from SENT_SIGNALS.jsonl
   - Parses metrics
   - Displays report with auto-refresh header

2. **Wait 5 Seconds**
   - Shows countdown: "Next update in 5 seconds..."

3. **Screen Clear** (Iteration 2+)
   - Clears terminal (fresh start)
   - Reloads signals from file
   - Recalculates metrics
   - Shows updated report with new timestamp

4. **Loop** (Repeats)
   - Continues until you press Ctrl+C

---

## 📝 Example: Tracking BEAR SHORT Recovery

**Watch this metric over 7 days:**

```
Day 1: BEAR SHORT WR = 0%   (0/1 closed)      → Phase 2-FIXED just deployed
Day 2: BEAR SHORT WR = 10%  (1/10 closed)     → Slight improvement
Day 3: BEAR SHORT WR = 15%  (5/33 closed)     → Trend confirmed
Day 5: BEAR SHORT WR = 20%  (10/50 closed)    → Good progress
Day 7: BEAR SHORT WR = 25%+ (15/60 closed)    → Phase 2-FIXED working! ✅
```

If you see this trend, Phase 2-FIXED is succeeding. If it stays at 0-5%, there's an issue.

---

## 🚨 Red Flags to Watch

| Flag | What It Means | Action |
|------|---------|--------|
| BEAR SHORT stuck at 0% WR | SHORT recovery not working | Check daemon logs, verify code deployment |
| Total signals dropping | Gate filtering too aggressive | Check Phase 2-FIXED gate thresholds |
| BULL LONG WR drops below 20% | Regression in baseline signals | Investigate gate changes, check for bugs |
| Frequent timeouts | Price not moving enough | Check market volatility, normal in choppy markets |

---

**Enjoy live monitoring! Leave the script running and watch your Phase 2-FIXED performance improve over the next 7 days.** 🎯
