# TIER INTEGRATION ARCHITECTURE
## Answering Q1 & Q2 While Respecting Code Immutability

**Date:** 2026-03-28 00:03 GMT+7
**Status:** DESIGN (ready for implementation)

---

## THREE CORE REQUIREMENTS

### **Req 1: Code Immutability (LOCKED Trackers)**
6 trackers have IMMUTABLE code:
- `pec_enhanced_reporter.py` (LOCKED)
- `pec_post_deployment_tracker.py` (LOCKED)
- 4 others (LOCKED)

**Solution:** Do NOT modify locked code. Instead:
- ✅ Create NEW `pec_tier_dimensional_report.py`
- ✅ Reads from same SIGNALS_MASTER.jsonl source
- ✅ Adds tier breakdown without changing locked tracker logic

### **Req 2: Both Trackers Need Tier Breakdown**
Both `pec_enhanced_reporter` and `pec_post_deployment_tracker` metrics need tier segmentation.

**Solution:** New tier-dimensional report generates:
- **SECTION A:** Tier breakdown of pec_enhanced_reporter's metrics
  - WITH_TIER (Tier-2, Tier-3, Tier-X): WR%, P&L, averages
  - WITHOUT_TIER: Baseline comparison
  
- **SECTION B:** Tier breakdown of pec_post_deployment_tracker's metrics
  - Same dimensional analysis, different metric source

### **Req 3: Tier Persisted in BOTH Sources**

**Current State:**
```
SIGNALS_MASTER.jsonl:           tier field ✓ POPULATED (13,480/13,480)
SIGNALS_INDEPENDENT_AUDIT.txt:  tier field ✗ NULL (needs update)
```

**Data Source Usage:**
- Both trackers read from: `SIGNALS_MASTER.jsonl` (single source of truth)
- Audit file is secondary (verification/audit trail, not used by trackers)

**Solution:** Ensure tier is in BOTH files:
1. Retroactively sync tier from SIGNALS_MASTER.jsonl → SIGNALS_INDEPENDENT_AUDIT.txt
2. Modify main.py to ALWAYS write tier field to both files when firing signals
3. Verify no divergence between sources

---

## IMPLEMENTATION PLAN

### **Step 1: Create Tier-Dimensional Report (NEW CODE - Not Locked)**

**File:** `pec_tier_dimensional_report.py`

```python
Purpose:
  Answer Q1: How do signals WITH tier vs WITHOUT tier perform?
  Answer Q2: How do 6D/5D/4D/3D/2D combos differ by tier status?

Input:
  SIGNALS_MASTER.jsonl (same source as locked trackers)

Output:
  PEC_TIER_DIMENSIONAL_REPORT.txt with:
  
  SECTION A: Tier Breakdown of pec_enhanced_reporter Metrics
    - Total signals: 13,480
    - WITH_TIER: 478 (Tier-2: 406 + Tier-3: 72)
      - WR%: X%
      - P&L: $X
      - Avg per closed: $X
      - 6D combos top 10
    - WITHOUT_TIER: 13,002 (non-matching)
      - WR%: Y%
      - P&L: $Y
      - Avg per closed: $Y
      - 6D combos top 10
    - DELTA (Tier Advantage): WR +/- Z%
  
  SECTION B: Tier Breakdown of pec_post_deployment_tracker Metrics
    - (Same structure, different metric source)

Code Status: NEW (not modifying locked trackers)
```

### **Step 2: Sync Tier to Audit File (Retroactive)**

**File:** `sync_tier_to_audit.py`

```python
Purpose:
  Populate tier field in SIGNALS_INDEPENDENT_AUDIT.txt from SIGNALS_MASTER.jsonl

Process:
  1. Load all signals from SIGNALS_MASTER.jsonl (tier already populated)
  2. Load all signals from SIGNALS_INDEPENDENT_AUDIT.txt (tier=NULL)
  3. Match by signal_uuid
  4. Copy tier from MASTER → AUDIT
  5. Rewrite AUDIT file with tier populated

Result:
  Both files now have tier field in sync
```

### **Step 3: Modify main.py (Write Tier to Both Files)**

**Changes to main.py:**
- Tier already written to SIGNALS_MASTER ✓
- ADD: Write tier to SIGNALS_INDEPENDENT_AUDIT.txt when firing
- Verify both writes succeed before confirming signal fire

**Code change:** Minimal, adds tier to audit write operation

---

## VERIFICATION MATRIX

### **Code Immutability**
| Tracker | Status | Change? | Rationale |
|---------|--------|---------|-----------|
| pec_enhanced_reporter.py | LOCKED | No | Immutable code |
| pec_post_deployment_tracker.py | LOCKED | No | Immutable code |
| pec_tier_dimensional_report.py | NEW | - | New file, doesn't violate constraint |
| main.py | Active | Minimal | Only add tier→audit write |

### **Data Source Consistency**
| Source | Tier Coverage | Used By Trackers | Audit Trail |
|--------|---------------|------------------|-------------|
| SIGNALS_MASTER.jsonl | 100% (13,480) | ✓ Both trackers | Primary |
| SIGNALS_INDEPENDENT_AUDIT.txt | 0% (needs sync) | ✗ Not used | Secondary |

**After implementation:**
| Source | Tier Coverage | Status |
|--------|---------------|--------|
| SIGNALS_MASTER.jsonl | 100% | ✓ Ready |
| SIGNALS_INDEPENDENT_AUDIT.txt | 100% | ✓ Synced |

### **Answer Q1 & Q2**

| Question | How Answered | Source |
|----------|-------------|--------|
| Q1: Performance WITH tier vs WITHOUT tier? | pec_tier_dimensional_report.py SECTION A | SIGNALS_MASTER.jsonl |
| Q2: 6D/5D/4D combos by tier status? | pec_tier_dimensional_report.py SECTION A (combos by tier) | SIGNALS_MASTER.jsonl |

---

## SUMMARY OF CONSTRAINTS RESPECTED

✅ **Code Immutability:** 6 locked trackers untouched
✅ **Both Trackers:** Tier metrics available for both via new report
✅ **Data Integrity:** Tier synced to BOTH SIGNALS_MASTER and SIGNALS_INDEPENDENT_AUDIT
✅ **Single Source of Truth:** Trackers read from SIGNALS_MASTER.jsonl (consistent)
✅ **New Code vs Locked:** Tier-dimensional report is NEW (not modifying locked)

---

## NEXT STEP

Proceed with implementation:
1. Create `pec_tier_dimensional_report.py` (new code)
2. Create `sync_tier_to_audit.py` (retroactive sync)
3. Modify `main.py` (write tier to audit on signal fire)
4. Verify tier coverage 100% in both files
5. Run new report and verify Q1 & Q2 answered correctly
