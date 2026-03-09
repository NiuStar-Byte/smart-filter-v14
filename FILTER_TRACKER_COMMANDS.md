# Filter Failure Tracker - Command Line Reference

## 📊 Version 1: One-Time Report (Single Run)

Shows complete filter failure analysis once and exits.

```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 filter_failure_inference.py
```

**Output:**
- Filter bottleneck ranking (all 20 filters)
- Top 8 priority targets
- Enhanced vs not-enhanced breakdown
- Expected impact after enhancements
- Exported CSV report

**Time to Complete:** ~2-3 seconds

---

## 🔄 Version 2: Live Monitoring (Updates Every 5 Seconds)

Continuously monitors filter failure rates and refreshes automatically.

```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 filter_failure_inference.py --watch
```

**Features:**
- Auto-updates every 5 seconds
- Shows latest signal counts
- Real-time bottleneck ranking
- Press `Ctrl+C` to stop

**Time to Complete:** Runs continuously (Ctrl+C to exit)

---

## 📊 With CSV Export

Add `--export` flag to automatically save results to CSV:

### Version 1 with CSV:
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 filter_failure_inference.py --export
```

### Version 2 with CSV (Live + Export every 5s):
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 filter_failure_inference.py --watch --export
```

**CSV Location:** `/Users/geniustarigan/.openclaw/workspace/filter_inference_analysis.csv`

---

## 🎯 Quick Reference

| Command | Purpose | Duration | Output |
|---------|---------|----------|--------|
| `python3 filter_failure_inference.py` | Single report | ~3s | Terminal + Optional CSV |
| `python3 filter_failure_inference.py --watch` | Live monitoring | ∞ (Ctrl+C) | Terminal (refreshing) |
| `python3 filter_failure_inference.py --export` | Report + CSV | ~3s | Terminal + CSV file |
| `python3 filter_failure_inference.py --watch --export` | Live + CSV | ∞ (Ctrl+C) | Terminal (refreshing) + CSV updates every 5s |

---

## 📈 Expected Output Structure

Both versions show:
1. **Signal Score Distribution** - Basic stats (avg score: 13.5, failures: 6.5)
2. **Filter Bottleneck Ranking** - All 20 filters ranked by failure rate
3. **Top 8 Priority Targets** - Detailed view of highest-failure filters
4. **Enhanced Filters Performance** - Status of your 6 enhancements
5. **Strategy Breakdown** - Next enhancement recommendations
6. **Expected Impact** - Projected WR improvement after all 20 enhanced

---

## 🚀 Copy-Paste Ready Commands

**One-time (quickest):**
```
cd /Users/geniustarigan/.openclaw/workspace && python3 filter_failure_inference.py
```

**Live monitoring (best for ongoing tracking):**
```
cd /Users/geniustarigan/.openclaw/workspace && python3 filter_failure_inference.py --watch
```

**Export to CSV for spreadsheet analysis:**
```
cd /Users/geniustarigan/.openclaw/workspace && python3 filter_failure_inference.py --export
```

---

## 💡 Usage Tips

- **For quick status check:** Use Version 1 (one-time)
- **For continuous monitoring during enhancement work:** Use Version 2 (live)
- **For record-keeping:** Add `--export` to save CSV snapshot
- **For detailed analysis:** Pipe to file with `> report.txt`

Example:
```bash
cd /Users/geniustarigan/.openclaw/workspace && python3 filter_failure_inference.py > filter_report_$(date +%Y%m%d_%H%M%S).txt
```

This saves timestamped report files for comparison over time.
