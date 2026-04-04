# COMPREHENSIVE TIER ASSIGNMENT AUDIT
**Date:** 2026-03-28 01:00 GMT+7
**Status:** INVESTIGATION IN PROGRESS

---

## PART 1: TIER HIERARCHY DEFINITION (Correct)

From `tier_config.py`:
```
Tier-1: WR >= 60%, Avg >= $5.50, Trades >= 60
Tier-2: WR >= 50%, Avg >= $3.50, Trades >= 50  
Tier-3: WR >= 40%, Avg >= $2.00, Trades >= 40
Tier-X: Everything else (WR < 40% or negative P&L)
```

This hierarchy is **CORRECT** (Tier-1 > Tier-2 > Tier-3).

---

## PART 2: ACTUAL TIER ASSIGNMENTS (From SIGNAL_TIERS.json)

### Current Tier-2 Combos (2 total):
1. `DIR_ROUTE_SHORT_NONE` 
2. `TF_REGIME_4h_RANGE`

### Current Tier-3 Combos (8 total):
1. `1h_SHORT_TREND CONTINUATION_BULL`
2. `4h_LONG_TREND CONTINUATION_BULL_LOW_ALTS`
3. `DIR_ROUTE_REGIME_LONG_REVERSAL_RANGE`
4. `ROUTE_REGIME_REVERSAL_RANGE`
5. `TF_DIR_2h_SHORT`
6. `TF_DIR_REGIME_1h_SHORT_BULL`
7. `TF_DIR_ROUTE_2h_SHORT_TREND CONTINUATION`
8. `TF_REGIME_1h_RANGE`

---

## PART 3: SIGNAL FIRING DATA (From pec_enhanced_reporter.py)

### Tier-2 Signals Fired (408 total, 91 closed = 22.3% close rate):
- **Tier-2 Performance**: 56.04% WR, +$0.34 avg per closed, +$31.09 total P&L
- Status: OUTPERFORMS baseline (Tier-X: 38.52% WR)

### Tier-3 Signals Fired (77-88 total, 7-12 closed = 13.6-15.6% close rate):
- **Tier-3 Performance**: 66.67% WR, +$0.74 avg per closed, +$8.82 total P&L
- Status: MASSIVELY OUTPERFORMS all tiers!

### Tier-X (Non-tier) Signals Fired (13,077 total, 6,950 closed = 53.2% close rate):
- **Tier-X Performance**: 38.52% WR, -$0.88 avg per closed, -$6,104.99 total P&L
- Status: Baseline (expected)

---

## PART 4: THE CORE PROBLEM

**Tier-3 (66.67% WR) should be Tier-1** according to the hierarchy definition.

This means ONE of these is wrong:

**Option A: The Tier Thresholds Are Wrong**
- Perhaps Tier-3 threshold should be higher?
- But the code clearly says Tier-3 = >=40% WR

**Option B: The Tier-3 Combos Are Incorrectly Assigned**
- These 8 combos don't actually meet ANY tier criteria
- They're being force-assigned to Tier-3 despite poor historical performance
- Then NEW signals that match those patterns get labeled Tier-3
- Then those NEW signals close with 66.67% WR (proving the historical data was stale!)

**Option C: The Assignment Logic Has A Bug**
- `sync_tier_patterns.py` might be using inverted logic
- Or `tier_lookup.py` might have backwards matching
- Or something else is inversing the tier assignment

---

## PART 5: TIMELINE EVIDENCE

**SIGNAL_TIERS.json history:**
- Created: 2026-03-05 20:00:25 GMT+7 (Entry #0)
- Both Tier-2 and Tier-3 existed from DAY 1 with the same combos
- Latest: 2026-03-28 00:15:30 GMT+7 (Entry #947)
- Tier composition has remained mostly constant

**This means:**
- Tier-3 combos were NEVER "just added today"
- They've been Tier-3 since Mar 5 (~23 days ago)
- Yet signals matching those combos NOW have 66.67% WR
- The combos must have improved dramatically, OR the old assignment was based on stale data

---

## PART 6: HYPOTHESIS

**The tier assignments were made on Mar 5 based on HISTORICAL performance at that time.**

But:
1. The market changed
2. The signal logic improved (3/4 factor norm on Mar 21, filters on Mar 25)
3. The same combos now perform MUCH better
4. The tier assignments haven't caught up

**This is why sync_tier_patterns.py is supposed to CONTINUOUSLY update tiers based on latest performance.**

But something in that sync logic is broken:

---

## PART 7: CRITICAL QUESTION

**Has sync_tier_patterns.py ever successfully extracted and promoted high-performers?**

Look for evidence:
- Did Tier-3 combos at 66.67% WR get promoted to Tier-2 or Tier-1?
- Or are they stuck in Tier-3 permanently?
- What's the last sync that actually CHANGED tier assignments?

---

## PART 8: FILES TO AUDIT

1. **sync_tier_patterns.py**
   - Line ~50-80: `extract_qualifying_combos()` function
   - Does it correctly parse WR, Avg, Closed from PEC_ENHANCED_REPORT.txt?
   - Does the AND logic work correctly?

2. **tier_lookup.py**
   - How does `get_signal_tier()` match signals to combos?
   - Is there any inversion or backwards logic?
   - Does it correctly read from SIGNAL_TIERS.json?

3. **SIGNAL_TIERS.json**
   - Why are Tier-3 combos not promoted despite 66.67% WR?
   - When was the last time tier assignments actually changed?
   - Is sync_tier_patterns.py even being called?

4. **pec_master_controller.py**
   - Does it call sync_tier_patterns.py?
   - How often? Every 5 minutes as documented?
   - Is it actually running?

---

## NEXT STEPS

1. Check if sync_tier_patterns.py is being called
2. Check sync logs in tier_sync.log
3. Verify extraction logic in extract_qualifying_combos()
4. Check tier_lookup.py for any inverted logic
5. Trace a single combo from pec_enhanced_reporter → SIGNAL_TIERS.json → tier assignment

---

## CRITICAL INSIGHT

The quality loop is broken at the **FEEDBACK LOOP** stage:
- ✅ Tier-2/3 combos are firing correctly
- ✅ They're closing with good performance
- ✅ pec_enhanced_reporter shows their metrics
- ❌ BUT those metrics aren't feeding back to update SIGNAL_TIERS.json
- ❌ So tier assignments stay stale
- ❌ New signals get assigned outdated tier levels

**Fix the sync loop and the quality loop comes alive.**
