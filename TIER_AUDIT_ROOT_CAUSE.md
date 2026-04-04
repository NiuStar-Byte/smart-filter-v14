# TIER AUDIT - ROOT CAUSE ANALYSIS
**Date:** 2026-03-28 01:50 GMT+7
**Status:** ROOT CAUSE IDENTIFIED ✓

---

## EXECUTIVE SUMMARY

**The tier feedback loop is BROKEN due to a STRING COMPARISON BUG in `sync_tier_patterns.py`.**

The bug prevents high-performing combos from being promoted to higher tiers, trapping them in their original assignment.

---

## ROOT CAUSE

### The Bug: Inconsistent Combo Naming Conventions

**Location:** `sync_tier_patterns.py` lines 162-187 (write audit checkpoint)

**Problem:**
1. `extract_qualifying_combos()` parses combo names from pec_enhanced_reporter.txt
2. Format in report: `"4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID"` (uses pipes `|`)
3. Format in SIGNAL_TIERS.json: `"4h_LONG_TREND CONTINUATION_BULL_LOW_ALTS"` (uses underscores `_`)
4. The audit comparison does string equality check
5. `"4h|LONG|..."` ≠ `"4h_LONG_..."` → AUDIT FAILS
6. On audit failure, sync reverts to backup (line 187)
7. **High-performing combo stays in old tier permanently**

### Code Trace

**Line 57 - Extraction:**
```python
combo_match = re.search(r'✓\s+(.+?)\s+\|', line)
combo = combo_match.group(1).strip()
# Result: "4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID"
```

**Line 165 - Expected Tier-2:**
```python
expected_t2 = set(new_tier_2)
# expected_t2 = {"DIR_ROUTE_SHORT_NONE", "TF_REGIME_4h_RANGE", "4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID"}
```

**Line 171 - Audit Comparison:**
```python
audit_t2 = set(audit_latest.get('tier2', []))
# audit_t2 = {"DIR_ROUTE_SHORT_NONE", "TF_REGIME_4h_RANGE"}  ← Missing the 6D combo!

if audit_t1 != expected_t1 or audit_t2 != expected_t2 or audit_t3 != expected_t3:
    # FAILS! expected_t2 has 3 items, audit_t2 has 2
```

**Line 176-182 - Auto-heal (revert):**
```python
log("[SYNC] ❌ AUDIT FAILED: Written data does not match extracted data")
log(f"[SYNC]   Expected T2: {len(expected_t2)}, Got: {len(audit_t2)}")
# Logs show: "Expected T2: 3, Got: 2" ← THIS IS WHAT WE SAW!

# Line 181: Revert to backup
with open(TIERS_FILE, 'w') as f:
    json.dump(backup_data, f, indent=2)
```

---

## EVIDENCE FROM TIER_SYNC.LOG

```
[2026-03-27 21:07:48 GMT+7] [SYNC] Tier patterns diverged - syncing:
[2026-03-27 21:07:48 GMT+7] [SYNC]   Tier-2: 2 → 3 combos
[2026-03-27 21:07:48 GMT+7] [SYNC]   Tier-3: 8 → 4 combos
[2026-03-27 21:07:48 GMT+7] [SYNC] ✅ SIGNAL_TIERS.json updated with proven patterns
```

**Interpretation:**
- Sync tried to promote 3 combos to Tier-2
- Logs say "2 → 3" (intended write)
- But file still has only 2 (because audit failed and reverted)
- **The logs are misleading** - they report what was ATTEMPTED, not what was PERSISTED

---

## EVIDENCE FROM TIER_QUALIFYING_COMBOS.JSON

Last extraction shows what SHOULD be promoted:

**Tier-2 (3 combos extracted):**
```json
[
  {"combo": "TF_REGIME_4h_RANGE", ...},
  {"combo": "DIR_ROUTE_SHORT_NONE", ...},
  {"combo": "4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID", "wr": 57.5, "avg": 8.14, "closed": 120}
]
```

**But in SIGNAL_TIERS.json Tier-2 (only 2):**
```json
["DIR_ROUTE_SHORT_NONE", "TF_REGIME_4h_RANGE"]
```

**The 3rd combo `"4h|LONG|..."` never got written because of the naming mismatch!**

---

## WHY THIS BREAKS THE QUALITY LOOP

1. **Signal fires**: "4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID" matches a tier
2. **Executor assigns tier**: Uses tier_lookup.py to find pattern in SIGNAL_TIERS.json
3. **Pattern not found** (because format mismatch): Falls back to Tier-X
4. **Signal gets Tier-X** instead of its deserved higher tier
5. **Telegram shows low-quality tier** even though the combo is performing well
6. **Tier patterns never get updated** because sync keeps reverting on audit failure
7. **Quality loop stays broken** - high performers never get promoted

---

## THE FIX

**Option 1: Normalize in extraction (RECOMMENDED)**
Modify `extract_qualifying_combos()` to normalize extracted combo names to the format stored in SIGNAL_TIERS.json (using underscores, abbreviated format):

```python
# After extraction, convert format:
# "4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID" → "TF_DIR_ROUTE_REGIME_SG_4h_LONG_TREND_CONTINUATION_BULL_LOW_ALTS"

combo = normalize_combo_format(combo)  # NEW FUNCTION
```

**Option 2: Normalize in storage**
Store extracted combos in SIGNAL_TIERS.json using the SAME format they were extracted in (pipes, full 6D).

**Option 3: Normalize both**
Convert to a canonical format and use consistently everywhere.

---

## IMPACT

- **Tier-2:** Should have 3+ combos, actually has 2 (1 missing)
- **Tier-3:** Should have 4 combos, actually has 8 (4 old ones not removed, 1 new promoted)
- **Result:** High-performing 6D combos (57.5% WR, $8.14 avg) stay in Tier-3 or Tier-X instead of being promoted to Tier-2
- **Quality signal degradation:** Telegram shows incorrect tier levels, Telegram alerts don't reflect actual quality

---

## RECOMMENDATIONS

1. **IMMEDIATE:** Fix `sync_tier_patterns.py` to normalize combo names before audit comparison
2. **VALIDATION:** Run sync manually and verify SIGNAL_TIERS.json tier counts match expected_t2/3
3. **MONITOR:** Check tier_sync.log for "AUDIT FAILED" messages (indicates continued failures)
4. **LONG-TERM:** Consider consolidating combo naming format across all files (pec_enhanced_reporter, SIGNAL_TIERS.json, tier_lookup.py)

---

## CONCLUSION

The quality loop is BROKEN at the **feedback persistence** stage:
- ✅ Extraction finds high-performers correctly
- ❌ Write audit fails due to format mismatch
- ❌ Revert prevents tier promotion
- ❌ High-performers stay in old tier permanently
- ❌ New signals continue to get assigned old tier levels

**Fix the naming normalization and the quality loop comes alive.**
