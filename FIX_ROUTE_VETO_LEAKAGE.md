# ROOT CAUSE: Route Veto Leakage (118 NONE/AMBIGUOUS signals)

## Problem
Phase 1 route veto is integrated into smart_filter.py but **Phase 3B code changes the route AFTER veto check**:

```python
# Line 850-868 (15min block, repeated in 30min & 1h)
if Route == "REVERSAL":
    quality_check = check_reversal_quality(...)
    if not quality_check["allowed"]:
        Route = quality_check["recommendation"]  # <-- BYPASSES VETO!
```

The recommendation can be "NONE" or "AMBIGUOUS", creating 118 leaked signals.

## Solution
Apply veto AGAIN after all route modifications:

```python
# AFTER Phase 3B changes route, BEFORE creating signal:

# === RE-APPLY PHASE 1 VETO (post-Phase 3B route changes) ===
if Route in ["NONE", "AMBIGUOUS"]:
    print(f"[VETO-POST-3B] {symbol_val} {tf_val}: Route changed to {Route} by Phase 3B. REJECTING.")
    continue  # Skip signal entirely
```

## Implementation
Add this check in 3 places (15min, 30min, 1h blocks) right AFTER the Phase 3B route modification code but BEFORE signal is created:

**Location 1:** main.py ~line 869 (15min)
```python
# Right after: Route = quality_check["recommendation"]
# Add re-veto check
```

**Location 2:** main.py ~line 1332 (30min)
```python
# Right after: Route = quality_check["recommendation"]
# Add re-veto check
```

**Location 3:** main.py ~line 1823 (1h)
```python
# Right after: Route = quality_check["recommendation"]
# Add re-veto check
```

**Location 4:** main.py ~line 2166 (4h) - NEW!
```python
# Right after: Route = quality_check["recommendation"] (if added)
# Add re-veto check
```

## Expected Result
- ✅ 118 NONE/AMBIGUOUS signals blocked
- ✅ NEW WR improves (NONE/AMBIGUOUS had worse WR than baseline)
- ✅ Phase 1 veto finally effective
