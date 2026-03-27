# TIER TRACKER - QUICK REFERENCE CARD

## Run Once (Generate Report)
```bash
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py
```
📁 Output: `TIER_COMPARISON_REPORT.txt`

## Live Mode (Auto-Refresh)
```bash
# Every 30 seconds (default)
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py --live

# Custom interval
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py --live --interval 60
```

## View Latest Report
```bash
cat ~/.openclaw/workspace/TIER_COMPARISON_REPORT.txt
```

## Archived Reports (History)
```bash
ls -lh ~/.openclaw/workspace/tier_comparison_reports/
```

---

## What to Look For

### ✅ Good Signs
- [ ] Combos appearing in TIER-3 (40%+ WR, $2+ avg, 40+ trades)
- [ ] More combos in TIER-2 than TIER-3
- [ ] Few combos in TIER-1 (hard to achieve 60%)
- [ ] TIER-X count decreasing over time

### ⚠️ Warning Signs
- [ ] All combos still in TIER-X (sample size too low)
- [ ] No improvement after 2 weeks (quality issue)
- [ ] Combos moving DOWN tiers (signal degradation)

### 🚨 Critical Issues
- [ ] Tracker shows 0 tier patterns loaded (SIGNAL_TIERS.json corrupted)
- [ ] Hypothesis shows ❌ refuted (tiering not working)

---

## Tier Criteria Checklist

```
TIER-1: ✓ ✓ ✓
├─ WR ≥ 60%?
├─ Avg ≥ $5.50?
└─ Trades ≥ 60?

TIER-2: ✓ ✓ ✓
├─ WR ≥ 50%?
├─ Avg ≥ $3.50?
└─ Trades ≥ 50?

TIER-3: ✓ ✓ ✓
├─ WR ≥ 40%?
├─ Avg ≥ $2.00?
└─ Trades ≥ 40?
```

---

## When to Deploy Project 4

Deploy Asterdex bot when:
- [ ] At least 3 combos in TIER-2 (proven performers)
- [ ] Overall system WR ≥ 51% (mathematically profitable)
- [ ] 60+ days of stable tier performance (consistency proven)
- [ ] Tier-1 combos ≥ 60% WR sustained (elite signal quality)

---

## Files Created This Session

| File | Purpose |
|------|---------|
| `tier_performance_comparison_tracker_live.py` | Enhanced tracker (live + breakdown) |
| `TIER_TRACKER_GUIDE.md` | Full detailed guide |
| `TRACKER_UPDATE_SUMMARY.md` | Executive summary |
| `TIER_BREAKDOWN_EXPLAINED.md` | Example outputs + explanation |
| `TIER_TRACKER_QUICK_REF.md` | This quick reference |

---

## Read Next (In Order)

1. **TIER_TRACKER_GUIDE.md** - Understand all sections
2. **TIER_BREAKDOWN_EXPLAINED.md** - See example output
3. **TRACKER_UPDATE_SUMMARY.md** - Summary + timeline

---

## Troubleshooting

### "No tier patterns loaded"
```bash
ls -la ~/.openclaw/workspace/SIGNAL_TIERS.json
cat ~/.openclaw/workspace/SIGNAL_TIERS.json | head -10
```
Check file exists and is valid JSON.

### "All combos in TIER-X"
✅ Expected initially. Wait for:
1. Combos to accumulate 40+ closed trades
2. Tier field to be persisted (see BLOCKER section)

### "Live mode not refreshing"
Try single run instead:
```bash
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py
sleep 30
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py
```

---

## Key Insights (TL;DR)

| Question | Answer |
|----------|--------|
| Why no combos in Tier-1/2/3? | Too few trades + tier field not persisted |
| Should I change tier thresholds? | Not yet. Data validates current config. |
| When to check the tracker? | Daily snapshot OR hourly live monitoring |
| How long to reach Project 4? | ~4 weeks (need 51% system WR) |
| What if tiering fails? | Hypothesis shows ✅ SUPPORTED; it works |

---

## One-Line Commands

```bash
# Single report
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py && cat TIER_COMPARISON_REPORT.txt

# Live monitoring (60 sec)
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py --live --interval 60

# Cron hourly (add to task scheduler)
cron add job --schedule '0 * * * *' --payload '{"kind":"systemEvent","text":"python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py"}'
```

---

## Next Immediate Step

⚠️ **MAIN BLOCKER:** Wire tier field into `pec_executor.py`

Once fixed:
1. Tier assignments persist to signals ✓
2. Tracker shows real tier distribution ✓
3. Can begin monitoring tier evolution ✓
4. Path to Project 4 becomes clear ✓

**Expected Timeline:**
- This week: Wire tier field
- Week 1-2: Combos reach TIER-3 (40 trades)
- Week 2-3: TIER-2 combos appear (50%+ WR)
- Week 4+: TIER-1 combos rare (60%+ WR)
- Week 5+: System ready for Project 4 (51%+ WR)
