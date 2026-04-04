# ROOT CAUSE: TIER FEEDBACK LOOP BROKEN AT SIGNAL_TIERS.json WRITE

**Severity:** CRITICAL
**Impact:** Quality loop cannot close; tier assignments forever stale
**Root Cause:** sync_tier_patterns.py file write bug

---

## THE BUG

### Extraction (Correct ✓)
`TIER_QUALIFYING_COMBOS.json` shows extraction correctly found:
```
Tier-2 Combos (3 total):
1. TF_REGIME_4h_RANGE (70.0% WR, $5.34 avg, 60 closed) ✓
2. DIR_ROUTE_SHORT_NONE (64.9% WR, $4.75 avg, 57 closed) ✓
3. 4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID (57.5% WR, $8.14 avg, 120 closed) ✓
```

### Sync Logs (Claims Success ✓)
```
[2026-03-27 23:03:21 GMT+7] [SYNC] Tier-2: 2 → 3 combos
[2026-03-27 23:03:21 GMT+7] [SYNC] ✅ SIGNAL_TIERS.json updated with proven patterns (validated)
[2026-03-27 23:03:21 GMT+7] [SYNC] New tier composition (VERIFIED):
[2026-03-27 23:03:21 GMT+7] [SYNC]   Tier-2: 3 combos ← CLAIMS 3
```

### File Reality (Only 2 Written ✗)
`SIGNAL_TIERS.json` actual Tier-2 list:
```
Tier-2 Combos (2 total):
1. DIR_ROUTE_SHORT_NONE ✓
2. TF_REGIME_4h_RANGE ✓
3. [MISSING: 4h|LONG|TREND CONTINUATION|BULL|LOW_ALTS|MID] ✗
```

**The 3rd combo is MISSING from the file despite logs claiming it was written!**

---

## CONSEQUENCE

Tier-3 combo `4h_LONG_TREND CONTINUATION_BULL_LOW_ALTS`:
- **Actual Performance:** 57.5% WR, $8.14 avg per closed, $977.17 total P&L ← **TIER-2 QUALITY**
- **Assigned Tier:** Tier-3 (stays there)
- **New Signals:** Get assigned Tier-3 due to pattern match
- **Result:** High-quality signals labeled as medium-quality tier

This is why:
- Tier-3 has 66.67% WR (it includes Tier-2-quality combos!)
- Tier-2 only has 56.04% WR (the best combos got trapped in Tier-3!)
- The hierarchy is INVERTED

---

## WHERE THE BUG LIVES

In `sync_tier_patterns.py`, the `sync_tiers()` function around line 150+:

**Likely issues:**
1. **List slicing** - Only first 2 combos written?
2. **JSON append vs replace** - Old entry not fully replaced?
3. **File flush** - Write succeeds but not flushed to disk?
4. **Validation logic** - Overwrites the file after writing?

The sync logs show validation passed: "✓ Audit passed: Written data verified"

But the file doesn't actually contain what was claimed to be verified!

---

## THE FIX

**Step 1:** Check sync_tier_patterns.py write logic (lines 150-200)
- How is the JSON being written?
- Is the file being truncated properly?
- Is the latest entry being appended or replaced?

**Step 2:** Add sanity check after write
```python
# After writing SIGNAL_TIERS.json
# Immediately read it back
with open(TIERS_FILE, 'r') as f:
    written_data = json.load(f)
    
latest_written = written_data[-1]
assert len(latest_written['tier2']) == len(new_tier_2), \
    f"Write mismatch: expected {len(new_tier_2)}, got {len(latest_written['tier2'])}"
```

**Step 3:** Re-run sync after fix to properly promote Tier-2 qualifying combos

---

## EVIDENCE SUMMARY

| Component | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Extraction | 3 T2 combos | 3 T2 combos | ✓ CORRECT |
| Sync logs | "Tier-2: 2 → 3" | Claims 3 written | ✓ CLAIMS SUCCESS |
| File verify logs | All data verified | Claims verified | ✓ CLAIMS VERIFIED |
| **Actual file** | **3 T2 combos** | **2 T2 combos** | **✗ BUG** |

The bug is in the write implementation, not the logic.

---

## IMPACT ON QUALITY LOOP

Current broken state:
```
Combo performance improves → Extraction finds it → Sync reports writing it →
File validation claims success → BUT FILE DOESN'T ACTUALLY HAVE IT!
→ Signal stays in old tier → New signals get stale tier → Performance metrics stay mixed
```

After fix:
```
Combo performance improves → Extraction finds it → Sync writes it → File has it →
Signals assigned to new tier → Quality improves → Metrics clearly separate
```

**This is the CRITICAL MISSING LINK in the quality loop.**
