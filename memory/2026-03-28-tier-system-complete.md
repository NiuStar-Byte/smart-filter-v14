# 2026-03-28 - Tier System Complete & Signal Enrichment

## Session Goal: Fix Tier Assignment Blocker
**Status:** ✅ COMPLETE

## Root Cause (Investigated & Resolved)
- **Initial Suspicion:** Tier logic inversion (WRONG)
- **Actual Problem:** Signals lacked `symbol_group` and `confidence_level` fields
- **Result:** Signals couldn't match 6D/5D dimensional combos, only 4D
- **Solution:** Enriched all 13,607 signals with missing fields

## What Changed

### 1. SIGNALS_MASTER.jsonl Enhanced
- **Added field:** `symbol_group` (MAIN_BLOCKCHAIN | MID_ALTS | LOW_ALTS)
  - BTC, ETH → MAIN_BLOCKCHAIN
  - BNB, SOL, ADA, AVAX, LINK, DOGE, MATIC → MID_ALTS
  - All others → LOW_ALTS
- **Added field:** `confidence_level` (HIGH | MID | LOW)
  - HIGH: confidence ≥73
  - MID: 66-73
  - LOW: <66
- All 13,603 signals now have both fields

### 2. SIGNAL_TIERS.json Regenerated
- **Format:** Append-only list with timestamps (for audit trail)
- **Current Entry:** 2026-03-28 07:41:14 GMT+7
- **Config Version:** "B (AGREED)" - ORIGINAL, NO DEVIATIONS
- **Qualified Combos:**
  - **Tier-1 (4 combos):** 60% WR, $5.50+ avg, 60+ trades
    - TF_DIR_ROUTE_REGIME_SG_4h_SHORT_TREND CONTINUATION_RANGE_LOW_ALTS
    - TF_DIR_ROUTE_REGIME_SG_2h_SHORT_TREND CONTINUATION_BULL_LOW_ALTS
    - 4h_SHORT_TREND CONTINUATION_RANGE
    - 2h_SHORT_TREND CONTINUATION_BULL
  - **Tier-2 (5 combos):** 50% WR, $3.50+ avg, 50+ trades
  - **Tier-3 (6 combos):** 40% WR, $2.00+ avg, 40+ trades
  - **Tier-X:** All non-qualifying combos

## Critical Notes
1. **Tier Config is LOCKED:** Only user can authorize changes
2. **Full Dimensional Support:** 6D → 5D → 4D → 3D cascade evaluation
3. **Timestamp Trail:** Each SIGNAL_TIERS.json entry has timestamp for audit
4. **Signal Assignment:** `tier_lookup.py` reads latest entry (-1 index), assigns tiers to signals
5. **Immutable Config:** Reverted unauthorized changes (60→30, 50→25, 40→15 trades); restored to strict original

## How It Works Now
1. Signal closes → `pec_executor.py` calls `get_signal_tier(tf, direction, route, regime, symbol_group)`
2. `tier_lookup.py` loads latest SIGNAL_TIERS.json entry
3. Matches signal combo against tier1/tier2/tier3/tierx lists
4. Writes tier to signal's `tier` field in SIGNALS_MASTER.jsonl
5. Next Telegram alert shows tier badge (🥇 Tier-1, 🥈 Tier-2, 🥉 Tier-3, ⚙️ Tier-X)

## Commits Today
- Signal enrichment: Added symbol_group + confidence_level to 13,603 signals
- Tier config revert: Locked to original (NO DEVIATIONS without user approval)
- Tier regeneration: 4/5/6 combos for Tier-1/2/3 with original strict criteria

## Next Steps
- Signals fired today and tomorrow will use enriched fields + new tier assignments
- Monitor executor cycle (5-min) for tier badge appearance in Telegram
- Track how many signals get assigned Tier-1/2/3 vs Tier-X
