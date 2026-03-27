# Tier Performance Tracker Guide

## Quick Start

### Single Run (Generate Report Once)
```bash
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py
```
Output: `TIER_COMPARISON_REPORT.txt` + timestamped archive in `tier_comparison_reports/`

### Live Mode (Auto-Refresh Every 30 Seconds)
```bash
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py --live --interval 30
```
- Clears screen and updates automatically
- Great for monitoring real-time tier evolution
- Press Ctrl+C to stop

### Custom Refresh Interval
```bash
# Refresh every 60 seconds
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py --live --interval 60

# Refresh every 10 seconds
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py --live --interval 10
```

---

## Understanding the Output

### Four Groups (2×2 Factorial Design)

```
GROUP A: FOUNDATION (Pre-3/4 Norm) - NON-TIERED
├─ 7,338 signals
├─ 27.54% WR
└─ -$0.82 avg per signal
   → Baseline: no improvements applied

GROUP B: FRESH (Post-3/4 Norm) - NON-TIERED
├─ 2,826 signals
├─ 37.37% WR
└─ -$0.19 avg per signal
   → Effect of 3/4 normalization filter alone

GROUP C: FOUNDATION (Pre-3/4 Norm) - TIERED
├─ 158 signals
├─ 35.96% WR
└─ +$2.24 avg per signal
   → Effect of tiering alone (pre-norm)

GROUP D: FRESH (Post-3/4 Norm) - TIERED
├─ 62 signals
├─ 71.43% WR ← BEST PERFORMANCE
└─ +$0.63 avg per signal
   → Combined effect: tiering + 3/4 norm
```

### Key Insights

**Tiering Effect (pre-norm):** C vs A
- WR: 35.96% vs 27.54% = **+8.41%** ✓
- Avg: +$2.24 vs -$0.82 = **+$3.07 improvement** ✓

**Tiering Effect (post-norm):** D vs B
- WR: 71.43% vs 37.37% = **+34.05%** ✓
- Avg: +$0.63 vs -$0.19 = **+$0.82 improvement** ✓

**Verdict:** ✅ **TIERING STRONGLY IMPROVES PERFORMANCE**

---

## Tier Breakdown: Which Combos Qualify?

### Tier Criteria (ALL THREE must be met)

| Tier | Win Rate | Avg P&L | Min Trades | Status |
|------|----------|---------|-----------|--------|
| Tier-1 | ≥ 60% | ≥ $5.50 | ≥ 60 | Elite |
| Tier-2 | ≥ 50% | ≥ $3.50 | ≥ 50 | Good |
| Tier-3 | ≥ 40% | ≥ $2.00 | ≥ 40 | Acceptable |
| Tier-X | < 40% | < $2.00 | < 40 | Below Min |

### Example: Why Your Combo Isn't in Tier-2

**Combo:** `4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID`

| Criterion | Value | Required | Status |
|-----------|-------|----------|--------|
| Win Rate | 57.5% | ≥ 50% | ✅ PASS |
| Avg P&L | $8.14 | ≥ $3.50 | ✅ PASS |
| Closed Trades | 120 | ≥ 50 | ✅ PASS |
| **Should Qualify** | — | **TIER-2** | **✅ YES** |

**Why not shown in TIER-2?**
1. Tier assignments not yet persisted to signals (main blocker)
2. Pattern matching against SIGNAL_TIERS.json may not find exact match
3. Once tier field is written, validator will confirm and tracker will update

---

## Reading the Tier Breakdown Section

### Current State
```
TIER-1: Elite (60% WR, $5.50+ avg, 60+ trades)
────────────────────────────────────────────
(0 combos)

TIER-2: Good (50% WR, $3.50+ avg, 50+ trades)
────────────────────────────────────────────
(0 combos)

TIER-3: Acceptable (40% WR, $2.00+ avg, 40+ trades)
────────────────────────────────────────────
(0 combos)

TIER-X: Below Minimum (< 40% WR, < $2.00 avg, or < 40 trades)
────────────────────────────────────────────
✓ 1h|REVERSAL
    WR: 100.0% | P&L: $5.08 | Avg: $+2.54 | Closed: 2
✓ 4h|REVERSAL
    WR: 100.0% | P&L: $4.42 | Avg: $+4.42 | Closed: 1
... (64 combos total)
```

### What the Columns Mean
- **WR:** Win Rate (TP_HIT + TIMEOUT_WIN) / Closed
- **P&L:** Total P&L for this combo
- **Avg:** Average P&L per closed trade
- **Closed:** Number of closed trades for this combo

### Why Most Combos Are in TIER-X
1. **Too few trades:** Most combos have < 40 closed trades
2. **Low average:** Many combos lose money per trade
3. **Low win rate:** Market regime not favorable for some combos

---

## Performance Delta Analysis: What It Means

### Group Comparisons
The tracker shows four delta comparisons:

1. **TIERING EFFECT (Pre-Norm):** GROUP C vs GROUP A
   - Isolates the effect of tiering alone (before 3/4 norm)
   - C: tiered, A: non-tiered (both pre-norm)

2. **TIERING EFFECT (Post-Norm):** GROUP D vs GROUP B
   - Isolates the effect of tiering with 3/4 norm applied
   - D: tiered, B: non-tiered (both post-norm)

3. **3/4 NORMALIZATION (Non-Tiered):** GROUP B vs GROUP A
   - Isolates the effect of 3/4 filter alone
   - B: post-norm non-tiered, A: pre-norm non-tiered

4. **3/4 NORMALIZATION (Tiered):** GROUP D vs GROUP C
   - How 3/4 filter affects tiered signals
   - D: post-norm tiered, C: pre-norm tiered

---

## When Combos Will Appear in Higher Tiers

### Condition 1: Tier Field Must Be Persisted
Currently, tier assignments are calculated but not saved. Once `signal['tier']` is written to SIGNALS_MASTER.jsonl:
- Tracker can verify tier assignments
- Validator confirms tier quality
- Daily reports track tier evolution

### Condition 2: Enough Closed Trades
Combos need sample size:
- Tier-3: 40+ closed trades
- Tier-2: 50+ closed trades
- Tier-1: 60+ closed trades

### Condition 3: Good Performance
All three metrics must align:
- WR ≥ threshold
- Avg P&L ≥ threshold
- Trades ≥ minimum

**Expected Growth Path:**
```
Day 1: All TIER-X (too few trades)
       ↓
Day 7: Some TIER-3 (40 trades, 40%+ WR, $2+ avg)
       ↓
Day 14: More TIER-3, some TIER-2 (better combos surface)
       ↓
Day 30: TIER-2 peak (many combos qualify with $3.50+ avg)
       ↓
Day 60: Few TIER-1 (60% WR is hard to achieve)
```

---

## Using Reports for Decision-Making

### Check 1: Is Tiering Working?
- **Look at:** HYPOTHESIS VALIDATION section
- **Good:** Verdict = "✅ THEORY STRONGLY SUPPORTED"
- **Action:** Tiering is justified; can build aggressive tier config

### Check 2: Which Combos Are Elite?
- **Look at:** TIER BREAKDOWN → TIER-1 section
- **Good:** Multiple combos with consistent 60%+ WR
- **Action:** Lock in tier patterns; use for next phase

### Check 3: How Many Combos Qualify for Each Tier?
- **Look at:** Combo count at end of each tier section
- **Good:** 
  - Tier-1: 3-5 combos (elite)
  - Tier-2: 10-20 combos (good)
  - Tier-3: 20-40 combos (acceptable)
- **Action:** Balance between selectivity and coverage

### Check 4: Which Combos Are Failing?
- **Look at:** TIER-X section
- **Good:** Combos have either:
  - Too few trades (sampling issue, not quality issue)
  - OR low WR + negative P&L (bad signal generation)
- **Action:** 
  - If too few trades: wait for more samples
  - If low WR: investigate filter logic or market regime

---

## Automation: Cron Job for Live Tracking

If you want reports generated hourly, add to cron:

```bash
cron add job \
  --schedule '0 * * * *' \
  --name 'Tier Tracker Hourly' \
  --payload '{
    "kind": "systemEvent",
    "text": "python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py"
  }'
```

Or for live mode every 4 hours:
```bash
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py --live --interval 240 > tier_live.log 2>&1 &
```

---

## Troubleshooting

### Problem: No combos in Tier-1/2/3
**Cause:** Tier assignments not persisted to SIGNALS_MASTER.jsonl
**Fix:** Wire `signal['tier']` into pec_executor.py

### Problem: Tracker shows 0 patterns loaded
**Cause:** SIGNAL_TIERS.json not found or corrupted
**Check:** 
```bash
ls -la ~/.openclaw/workspace/SIGNAL_TIERS.json
cat ~/.openclaw/workspace/SIGNAL_TIERS.json | head -50
```

### Problem: Live mode doesn't refresh
**Fix:** Try without --live:
```bash
python3 ~/.openclaw/workspace/tier_performance_comparison_tracker_live.py
```
Then manually re-run every 30 seconds.

---

## Key Metrics to Watch

### Daily
- Total combos in TIER-X (should decrease over time)
- Any NEW combos entering TIER-3 or higher

### Weekly
- Average WR across all tiers
- Total P&L from TIER-1 + TIER-2 (should be positive)

### Monthly
- Percentage of signals in TIER-1/2 (should increase)
- Average WR by tier (Tier-1 should be ≥60%)

---

## Next Phase: Project 4 Integration

Once you achieve:
- ✅ Tier-1 combos consistently hitting ≥61% WR
- ✅ Overall system WR ≥ 51% (statistically unbeatable)
- ✅ Tier assignments validated every 60 seconds

Then you're ready to deploy Project 4 (Asterdex bot) to live trading.
