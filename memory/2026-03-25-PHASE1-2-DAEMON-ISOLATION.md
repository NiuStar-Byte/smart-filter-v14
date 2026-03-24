# 2026-03-25 — PHASE 1-2 DAEMON ISOLATION & VALIDATION

## Critical Discovery
**Root cause of tracker failures:** 8 competing daemon processes running simultaneously, causing:
- Signal loss (4h signals never reaching SIGNALS_MASTER.jsonl)
- Route veto bypass (NONE/AMBIGUOUS leaks despite upstream fix)
- Direction parsing failure (LONG/SHORT counts = 0)
- Processing conflicts & race conditions

**Resolution:** Kill all competing daemons, restart single fresh instance, re-validate all three fixes.

## Phase 1 Deployed (2026-03-24 18:52 GMT+7)
**File:** `smart_filter.py` (lines 960-1000)  
**Change:** Route-only gatekeeper upstream in filters_ok calculation  
**Logic:** Block NONE (13.3% WR) + AMBIGUOUS (20.8% WR) routes before signal passes  
**Verification required:** Tracker should show 0 NONE/AMBIGUOUS signals (was 12 leaked due to 8 daemons)

## Phase 2 Deployed (2026-03-24 19:07 GMT+7)
**File:** `main.py` (lines 2108-2345)  
**Changes:**
1. Added TF4h to COOLDOWN (1200), EMA_CONFIG (200), DEDUP_WINDOWS (14400)
2. Simplified 4h block: REMOVED DirectionAwareGatekeeper, alignment filters, dedup cycle checks
3. Fixed tp_pct_val/sl_pct_val: Calculate from prices, not dict extraction
4. Extract market regime from SmartFilter for 4h signals

**Signal flow:** df4h → SmartFilter → route veto → TP/SL calc → Telegram send  
**Verification required:** Tracker should show >0 TF4h signals (was 0 due to 8 daemons)

## Critical Fixes Applied
- ✅ `signal_logger.py`: Added "4h" to dictionaries (commit e9ca43d)
- ✅ `calculate_tp_sl()` call: Uses df4h + regime (commit e1dffd5)
- ✅ Regime extraction: Populated from SmartFilter (commit c16abb4)
- ✅ TP/SL percentage: `((tp - entry_price) / entry_price * 100)` (commit 876a694)

## Validation Framework
**Tracker:** `phase1_phase3_phase2_tracker.py` (5 sections)
1. Foundation baseline: 2,224 signals, 32.7% WR (locked Feb 27-Mar 19)
2. NEW signals: Fired from single daemon (18:52 GMT+7 → NOW)
3. Timeframe breakdown: 15min, 30min, 1h, 4h with identical metrics
4. Route veto: NONE/AMBIGUOUS count (target: 0)
5. Decision checklist: LONG/SHORT, regime population, storage verification

## Next Validation Steps
1. **Verify daemon logs** (5 min): `tail -100 /tmp/daemon.log` → Look for 4h processing, no errors
2. **Run tracker** (15 min): Should show >0 TF4h, 0 NONE/AMBIGUOUS, LONG/SHORT both >0
3. **Monitor 1-2 hours**: Every 30 min, verify leak rate = 0, TF4h growing, regime populated
4. **Deploy Phase 3**: Route/regime audit after single daemon validation complete

## Performance Target
**Current (before Phase 1-2 with single daemon):** 26.04% WR (Foundation 32.7%)  
**Target:** 51% WR by tuning upstream signal generation (not downstream filtering)  
**Mechanism:** Better combo gatekeepers (route veto) + timeframe ensemble (4h) + status-tier weights

## Key Constraints
- **No directional veto:** LONG/SHORT equally valid (bear market asymmetry is market-driven)
- **Route veto only:** Block NONE + AMBIGUOUS upstream; no combo/regime/direction gates
- **Single daemon only:** Multiple daemons = signal loss; always restart cleanly
- **Keep all TFs:** Don't remove 15min/30min/1h when adding 4h
- **Simplify > optimize:** Mirror working 1h pattern for 4h

## Files Modified (Commits 2026-03-24 22:00-23:20 GMT+7)
- `smart-filter-v14-main/smart_filter.py` - Route veto (commit 5d9e213)
- `smart-filter-v14-main/main.py` - TF4h + simplify (commit 876a694)
- `signal_logger.py` - Add 4h tracking (commit e9ca43d)
- `phase1_phase3_phase2_tracker.py` - Validation framework (created)

## Daemon Restart
**Status:** Single fresh daemon instance started after killing all 8 competing processes  
**Time:** 2026-03-24 23:20 GMT+7  
**Log:** `/tmp/daemon.log`  
**PID:** Check with `pgrep -f 'python.*main.py'`

## What NOT to Do
- ❌ Run multiple daemons simultaneously
- ❌ Revert 4h simplification (DirectionAwareGatekeeper blocks all 4h signals)
- ❌ Extract TP/SL from dict (causes None when missing)
- ❌ Add direction/regime/combo gates to route veto (only NONE/AMBIGUOUS)

---

**Status:** 🟢 **CODE FIXES COMPLETE** — Awaiting fresh daemon validation with single instance
