# CRITICAL: TIER ASSIGNMENT LOGIC IS BACKWARDS

## Evidence

**Timeline Truth:**
- Tier-2 and Tier-3 both created on 2026-03-05 (SAME DAY, same time)
- User's Telegram observation: Tier-3 appeared first, then Tier-2
- This doesn't match "Tier-3 just added today" explanation

**Performance Truth:**
- Tier-3: 66.67% WR (12 closed trades)
- Tier-2: 56.04% WR (91 closed trades)  
- **Tier-3 outperforms Tier-2 by 10.63%**

**Tier Hierarchy Violation:**
```
CORRECT LOGIC:
  Tier-1 = Highest threshold (>60% WR, >$5+ avg) ← Best performers
  Tier-2 = Good threshold     (>50% WR, >$3.50)
  Tier-3 = Acceptable         (>40% WR, >$2+)    ← Worst performers

ACTUAL DATA:
  Tier-3: 66.67% ← Should be Tier-1!
  Tier-2: 56.04% ← Correct tier
  Tier-X: 38.52% ← Baseline (should be non-qualifying)
```

## Root Cause

The `sync_tier_patterns.py` or `get_signal_tier()` logic is assigning combos to the WRONG tier levels.

This could be:
1. **Backwards threshold comparison** (using < instead of >)
2. **Inverted tier priority** (assigning high-performers to Tier-3)
3. **Incorrect combo pattern matching** (wrong patterns in SIGNAL_TIERS.json)

## Impact

- Tier-3 signals are actually BEST quality (contradicts intent)
- Tier-2 signals are medium quality (correct position)
- Tier structure is unreliable for quality assessment until fixed

## What to Do

Before fixing pec_tier_dimensional_report, need to:
1. Audit `sync_tier_patterns.py` logic
2. Check tier_config.py thresholds
3. Verify SIGNAL_TIERS.json combo assignments
4. Fix the assignment logic to respect hierarchy (Tier-1 > Tier-2 > Tier-3)

## Recommendation

**Don't** finalize pec_tier_dimensional_report format until tier logic is correct.
Current report will show wrong conclusions if tier assignments remain inverted.
