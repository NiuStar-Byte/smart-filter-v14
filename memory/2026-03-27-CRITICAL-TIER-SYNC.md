# 2026-03-27 20:56 - CRITICAL: Tier Sync Break in Quality Loop

## THE ISSUE YOU IDENTIFIED

User correctly identified: **Quality loop is broken at tier assignment sync point**

### What's Happening (Wrong)

```
pec_enhanced_reporter says: "4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID = Tier-2"
                ↓
    SIGNAL_TIERS.json does NOT have this pattern
                ↓
    main.py fires signals with WRONG pattern: "30min|TREND CONTINUATION|BEAR"
                ↓
    240 signals fired with Tier-2 label, all OPEN, no performance validation
                ↓
    Quality loop BROKEN - tier assignments don't reflect actual performance
```

### Actual Data

**TIER-2 signals currently firing:** `30min|TREND CONTINUATION|BEAR`
- Count: 240 signals
- Status: All OPEN (0 closed)
- Performance: Unknown (can't validate)
- Should be: Not tier-2 (unproven)

**ACTUAL tier-2 combos from performance:**
1. `TF_REGIME_4h_RANGE` - 70.0% WR, $5.34 avg, 60 closed ✅
2. `DIR_ROUTE_SHORT_NONE` - 64.9% WR, $4.75 avg, 57 closed ✅
3. `4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID` - 57.5% WR, $8.14 avg, 120 closed ✅

## THE FIX

**Three-step synchronization:**

1. **Extract** actual tier-qualifying combos from pec_enhanced_reporter ✅ DONE
   - File: `TIER_QUALIFYING_COMBOS.json`
   - Contains: 3 Tier-2 + 4 Tier-3 combos with proven performance

2. **Sync** SIGNAL_TIERS.json with these proven patterns ⚠️ PENDING
   - Add tier-2 patterns: TF_REGIME_4h_RANGE, DIR_ROUTE_SHORT_NONE, 4h|LONG|...
   - Add tier-3 patterns: TF_DIR_2h_SHORT, 30min|SHORT|TREND CONTINUATION|..., etc.
   - Remove non-performing patterns

3. **Restart** main.py to fire signals with correct tier assignments ⚠️ PENDING
   - Next signals will use validated patterns
   - Quality loop can now track performance → quality feedback → tier update

## FILES CREATED

- `TIER_QUALIFYING_COMBOS.json` - Source of truth (from performance data)
- `TIER_ASSIGNMENT_TRACKER.md` - Full documentation of the issue
- `extract_tier_combos.py` - Script to extract tier combos from reports

## NEXT IMMEDIATE ACTION

**Option 1: Auto-sync (Recommended)**
- Create script to sync SIGNAL_TIERS.json from TIER_QUALIFYING_COMBOS.json
- Update SIGNAL_TIERS.json now
- Restart main.py
- Verify tier assignments match actual combos being fired

**Option 2: Manual Review**
- Review TIER_QUALIFYING_COMBOS.json
- Decide if tier patterns are correct
- Then run auto-sync

## CRITICAL POINT

User is 100% correct: "if you don't have it, you won't be able to run the exact performance"

Without this sync, tier field in signals is **meaningless**. The quality loop needs:
1. Performance data (pec_enhanced_reporter) ✓
2. Tier assignment source (SIGNAL_TIERS.json) ← BROKEN, not synced
3. Tier persistence (pec_executor) ✓
4. Tier validation (validator) ✓

Item #2 is the missing link preventing the quality loop from closing.
