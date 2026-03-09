# Dual Tracker (Immutable + Dynamic) - Command Line Reference

## 📊 Version 1: One-Time Report (Single Run)

Shows immutable baseline, dynamic tracker, and comparison once, then exits.

### Direct Command:
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 dual_tracker_immutable_dynamic.py --save
```

**What it does:**
- Loads all signals
- Stratifies by 2026-03-05 cutoff
- Calculates immutable baseline (pre-enhancement)
- Calculates dynamic tracker (post-enhancement)
- Generates comparison report
- Saves all 3 to JSON files
- Prints formatted report to terminal

**Time to Complete:** ~2-3 seconds

**Output Files:**
- `FILTERS_BASELINE_IMMUTABLE.json` - Locked reference
- `FILTERS_TRACKER_POST_ENHANCEMENT.json` - Live tracker
- `FILTERS_COMPARISON_REPORT.json` - Comparison & analysis

---

## 🔄 Version 2: Live Monitoring (Updates Every 60 Seconds)

Continuously monitors and refreshes trackers every 60 seconds. Shows real-time impact as signals accumulate.

### Direct Command:
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 dual_tracker_immutable_dynamic.py --watch --save
```

**What it does:**
- Loads all signals
- Stratifies by 2026-03-05 cutoff
- Calculates baseline, tracker, comparison
- Prints formatted report
- Updates JSON files
- Waits 60 seconds
- Repeats infinitely
- Press `Ctrl+C` to stop

**Time to Complete:** Runs continuously (Ctrl+C to exit)

**Output Files:**
- Updates `FILTERS_BASELINE_IMMUTABLE.json` every 60s (actually locked, minimal change)
- Updates `FILTERS_TRACKER_POST_ENHANCEMENT.json` every 60s (grows as signals arrive)
- Updates `FILTERS_COMPARISON_REPORT.json` every 60s (delta updates)

---

## 🚀 Quick Reference

| Version | Command | Purpose | Duration |
|---------|---------|---------|----------|
| **Once** | `python3 dual_tracker_immutable_dynamic.py --save` | Single report + save | ~3 seconds |
| **Live** | `python3 dual_tracker_immutable_dynamic.py --watch --save` | Continuous monitoring | ∞ (Ctrl+C) |

---

## 📋 Copy-Paste Ready Commands

### One-Time Report:
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 dual_tracker_immutable_dynamic.py --save
```

### Live Monitoring (60s Refresh):
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 dual_tracker_immutable_dynamic.py --watch --save
```

---

## 📊 What You'll See

### Report Output (Both Versions)

```
====================================================================================================
DUAL TRACKER REPORT: IMMUTABLE BASELINE vs DYNAMIC POST-ENHANCEMENT
====================================================================================================

📌 IMMUTABLE BASELINE (Pre-Enhancement - Locked Forever)
----------------------------------------------------------------------------------------------------
Cutoff: All signals BEFORE 2026-03-05T00:00:00
Signal count: 998 (locked, final)
Avg score: 14.64 / 20
Pass rate: 73.2%
Failure rate: 26.8%
Status: ✓ IMMUTABLE (reference point)

📊 DYNAMIC TRACKER (Post-Enhancement - Live)
----------------------------------------------------------------------------------------------------
Cutoff: All signals FROM 2026-03-05T00:00:00 onwards
Signal count: 605 (growing, updates live)
Avg score: 12.51 / 20
Pass rate: 62.5%
Failure rate: 37.5%
Enhancement coverage: 12/20 filters (60%)
Status: ⏳ DYNAMIC (updates continuously)

🔍 COMPARISON & ANALYSIS
----------------------------------------------------------------------------------------------------
Score change: -2.13 filters per signal (-14.5%)
Pass rate change: -10.7%

Interpretation: ⚠️ Potential degradation - enhancements may be too strict (high gate rate)

💡 Key Insights:
   • enhancement_coverage: 12/20 filters (60%)
   • no_need_to_wait: Partial enhancement is meaningful - no need to wait for 20/20
   • gate_rate_note: Higher failure rate ≠ bad. May indicate better selectivity filtering weak signals.
   • quality_metric: Compare P&L and win rate (WR) to confirm if delta = good or bad
   • next_action: Deploy Phase 3/4 with same dual-tracker methodology

====================================================================================================
[OK] Saved immutable baseline: /Users/geniustarigan/.openclaw/workspace/FILTERS_BASELINE_IMMUTABLE.json
[OK] Updated dynamic tracker: /Users/geniustarigan/.openclaw/workspace/FILTERS_TRACKER_POST_ENHANCEMENT.json
[OK] Updated comparison: /Users/geniustarigan/.openclaw/workspace/FILTERS_COMPARISON_REPORT.json
```

---

## 🔍 Understanding the Output

### IMMUTABLE BASELINE
- **Locked forever** - Never changes (reference point)
- Pre-enhancement baseline (998 signals before 2026-03-05)
- Shows: 14.64 avg score, 73.2% pass rate
- Used to compare against post-enhancement

### DYNAMIC TRACKER
- **Updates live** - Grows as signals arrive (605+ signals)
- Post-enhancement with 12/20 filters (60%)
- Shows: 12.51 avg score, 62.5% pass rate
- Real-time impact tracking

### COMPARISON
- **Score delta:** -2.13 (lower = more strict filtering)
- **Pass rate delta:** -10.7% (might be good gate rate)
- **Interpretation:** Could be selectivity or strictness - check P&L/WR to confirm

---

## 🎯 Recommended Workflow

### Step 1: Check Baseline Once
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 dual_tracker_immutable_dynamic.py --save
```
→ See immutable baseline established

### Step 2: Start Live Monitoring
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 dual_tracker_immutable_dynamic.py --watch --save
```
→ Watch signal accumulation and delta changes every 60s
→ Keep running in background while deploying Phase 3/4

### Step 3: Check Signal Quality (External)
- Run signal analysis to check P&L and win rate
- Determine if -2.13 delta = good selectivity or bad strictness

### Step 4: Deploy Phase 3/4
- While live monitoring still running
- Tracker automatically updates with Phase 3/4 signals
- Compare new enhancements against immutable baseline

---

## 💡 Tips

- **Version 1 (Once):** Use for quick status checks between deployments
- **Version 2 (Live):** Use while deploying Phase 3/4, keep terminal visible
- **JSON Files:** Can view directly with `cat` or editor for detailed analysis
- **Ctrl+C:** Stop live monitoring anytime (doesn't delete files)
- **Background:** Can run live monitoring in separate terminal tab

---

## 📁 Output Files Location

```
/Users/geniustarigan/.openclaw/workspace/
├── FILTERS_BASELINE_IMMUTABLE.json          # Locked reference (998 pre-signals)
├── FILTERS_TRACKER_POST_ENHANCEMENT.json    # Live tracker (605+ post-signals)
└── FILTERS_COMPARISON_REPORT.json           # Delta & interpretation
```

View any file:
```bash
cat /Users/geniustarigan/.openclaw/workspace/FILTERS_COMPARISON_REPORT.json | python3 -m json.tool
```

---

## Example Terminal Session

```bash
# Terminal 1: Start live monitoring
$ cd /Users/geniustarigan/.openclaw/workspace && python3 dual_tracker_immutable_dynamic.py --watch --save

[INFO] Dual tracking mode (60s refresh). Press Ctrl+C to stop.

====================================================================================================
DUAL TRACKER REPORT: IMMUTABLE BASELINE vs DYNAMIC POST-ENHANCEMENT
====================================================================================================
...
[INFO] Next update in 60s... (Ctrl+C to stop)

# ... 60 seconds pass ...

====================================================================================================
DUAL TRACKER REPORT: IMMUTABLE BASELINE vs DYNAMIC POST-ENHANCEMENT
====================================================================================================
...
[INFO] Next update in 60s... (Ctrl+C to stop)

# Keep this running while deploying Phase 3/4
# Ctrl+C to stop when done
```

---

## 🚀 Ready to Go?

**One-time report:**
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 dual_tracker_immutable_dynamic.py --save
```

**Live monitoring:**
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 dual_tracker_immutable_dynamic.py --watch --save
```

**Both commands are ready to copy-paste into terminal.** ✅
