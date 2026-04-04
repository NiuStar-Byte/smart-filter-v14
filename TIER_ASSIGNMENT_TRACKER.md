# TIER ASSIGNMENT TRACKING & SYNCHRONIZATION

**Generated:** 2026-03-27 20:56 GMT+7

## THE PROBLEM: Quality Loop Broken at Sync Point

```
pec_enhanced_reporter identifies actual performing combos ✓
  ↓
  SIGNAL_TIERS.json NOT updated with these combos ❌
  ↓
  main.py fires signals with WRONG tier patterns ❌
  ↓
  Signals close with misaligned tier labels ❌
  ↓
  Quality loop BROKEN: next iteration can't use correct performance data
```

## ACTUAL TIER-QUALIFYING COMBOS (From Closed Signal Performance)

### 🥈 TIER-2 (3 combos - 50% WR, $3.50+ avg, 50+ trades)

| Combo Pattern | WR | Avg | Closed | Status |
|---|---|---|---|---|
| `TF_REGIME_4h_RANGE` | 70.0% | $5.34 | 60 | ✅ PROVEN |
| `DIR_ROUTE_SHORT_NONE` | 64.9% | $4.75 | 57 | ✅ PROVEN |
| `4h\|LONG\|TREND CONTINUATION\|BULL\|LOW_ALTS\|MID` | 57.5% | $8.14 | 120 | ✅ PROVEN |

### 🥉 TIER-3 (4 combos - 40% WR, $2.00+ avg, 40+ trades)

| Combo Pattern | WR | Avg | Closed | Status |
|---|---|---|---|---|
| `TF_DIR_2h_SHORT` | 64.2% | $2.64 | 215 | ✅ PROVEN |
| `TF_DIR_ROUTE_REGIME_1h_SHORT_TREND CONTINUATION_BULL` | 59.5% | $2.61 | 42 | ✅ PROVEN |
| `30min\|SHORT\|TREND CONTINUATION\|BEAR\|LOW_ALTS\|HIGH` | 57.1% | $2.99 | 140 | ✅ PROVEN |
| `TF_DIR_ROUTE_REGIME_SG_4h_LONG_TREND CONTINUATION_BULL_LOW_ALTS` | 49.4% | $2.15 | 263 | ✅ PROVEN |

## WHAT'S CURRENTLY IN SIGNAL_TIERS.json vs WHAT SHOULD BE

### PROBLEM SIGNALS Being Fired (Wrong Patterns)

**Current Tier-2 assignment:** `30min|TREND CONTINUATION|BEAR`
- **Status:** 240 signals fired, all OPEN, zero closed performance
- **Should be:** Not in Tier-2 (no historical proof)
- **Why:** Pattern `30min_SHORT_TREND CONTINUATION_BEAR` exists in SIGNAL_TIERS.json tier2, but actual performance is unknown

**Correct Tier-2 patterns should be:**
1. `TF_REGIME_4h_RANGE` (70% WR proven)
2. `DIR_ROUTE_SHORT_NONE` (64.9% WR proven)
3. `4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID` (57.5% WR proven)

---

## THE QUALITY LOOP SYNC POINT

For quality loop to work:

```
STEP 1: pec_enhanced_reporter runs
  ↓
  Identifies: "4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID has 57.5% WR, $8.14 avg, 120 closed"
  
STEP 2: ⚠️ MISSING: TIER_QUALIFYING_COMBOS.json extracted
  ↓
  Tier-2 Patterns Extracted:
  - TF_REGIME_4h_RANGE
  - DIR_ROUTE_SHORT_NONE
  - 4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID ← Add this!
  
STEP 3: ⚠️ MISSING: SIGNAL_TIERS.json synchronized
  ↓
  Update tier2 list with extracted patterns

STEP 4: main.py restart
  ↓
  Next signals fired with CORRECT tier patterns

STEP 5: pec_executor persists tier field
  ↓
  Signals close with CORRECT tier labels

STEP 6: Next iteration: pec_enhanced_reporter validates performance
  ↓
  "Tier-2 combos achieved 57.5% WR - thesis validated"
  
STEP 7: ✅ Quality loop continues autonomously
```

---

## WHAT NEEDS TO HAPPEN NOW

**Priority 1: Manual Sync (Today)**
1. ✅ Extract tier-qualifying combos from pec_enhanced_report → `TIER_QUALIFYING_COMBOS.json`
2. ⚠️ TODO: Update `SIGNAL_TIERS.json` with these proven patterns
3. ⚠️ TODO: Restart `main.py` to fire signals with correct patterns

**Priority 2: Automation (Next)**
1. Create cron job that runs `extract_tier_combos.py` after each pec_enhanced_reporter generation
2. Auto-sync SIGNAL_TIERS.json with new tier-qualifying combos
3. Soft-restart main.py to pick up new patterns

**Priority 3: Validation (Ongoing)**
1. Create tracker showing: tier assignment source → signal fired → closed performance
2. Validate tier assignments match actual performance
3. Alert if tier patterns diverge from performance

---

## FILE INVENTORY

**Generated Files:**
- `TIER_QUALIFYING_COMBOS.json` - Actual tier-qualifying combos (from performance)
- `TIER_ASSIGNMENT_TRACKER.md` - This doc (sync status)

**Files to Update:**
- `SIGNAL_TIERS.json` - Sync with `TIER_QUALIFYING_COMBOS.json` patterns

**To Monitor:**
- `PEC_ENHANCED_REPORT.txt` - Source of truth for combo performance
- `SIGNALS_MASTER.jsonl` - Signals with tier field (must match assignment source)

---

## KEY INSIGHT

The quality loop doesn't break from pec_executor, validator, or tracker.

**It breaks at the SYNC POINT between:**
- What performance data says (pec_enhanced_reporter)
- What patterns fire signals (SIGNAL_TIERS.json → main.py)

Until these are synchronized, tier assignments are **meaningless noise** not signal quality feedback.

---

**Next step:** Sync SIGNAL_TIERS.json with TIER_QUALIFYING_COMBOS.json and restart main.py.
