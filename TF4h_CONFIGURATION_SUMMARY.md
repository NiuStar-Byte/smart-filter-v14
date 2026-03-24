# TF4h CONFIGURATION SUMMARY (2026-03-25)

## Question 1: RR Configuration for TF4h

### Answer: YES - TF4h uses market-driven RR (same as 15min/30min/1h)

**Current Implementation:**
- **Strategy:** Market-driven S&R (actual support/resistance levels)
- **Cap:** Maximum 2.5:1 RR (prevents overleveraging)
- **Fallback:** 1.25:1 ATR-based when no S&R available
- **Source:** `tp_sl_retracement.py` (PHASE 3 FIX)

**For TF4h:** ✅ Uses SAME market-driven logic as all other timeframes
- No minimum RR assigned
- Market structure determines actual RR
- Caps at 2.5:1 across all timeframes

**Recommended Min-Max RR for Future Tuning:**
- Min RR: 1.0:1 (equal risk/reward)
- Target RR: 1.5-2.0:1 (conservative for 4h candles)
- Max RR: 2.5:1 (current cap applies)

**Note:** If market-driven RR is producing poor results (not capturing full 4h moves), consider adding explicit min/max RR constraints in future Phase:
```python
# Future enhancement (not yet implemented):
tf4h_min_rr = 1.5  # Ensure 4h signals have at least 1.5:1 RR
tf4h_max_rr = 3.0  # Allow larger RR for volatile 4h pairs
```

---

## Question 2: Timeout Duration for TF4h

### Answer: FIXED - TF4h timeout now set to 20 hours (was missing, now configured)

**Implementation (pec_executor.py, 2026-03-25):**

**Champion Model (MAX_BARS):**
```python
"4h": 5  # 5 bars × 4h = 1,200 minutes = 20 hours
```

**Challenger Model (Tier-Based):**
```python
"4h": {
    "TIER-1": 24 * 60,     # 24h (full day for high conviction)
    "TIER-2": 18 * 60,     # 18h (mid-range)
    "TIER-3": 12 * 60,     # 12h (quick exit for losing combos)
}
```

**Comparison with 1h:**
- **1h:** 5h (Champion), 4h-6h (Challenger)
- **4h:** 20h (Champion), 12h-24h (Challenger)
- **Ratio:** 4h gets 4x longer timeout (4 bars/hour × 4 hours/candle = 4x slower)

**Rationale:**
- 4h candles are 4x larger than 1h
- Take 4x longer to reach TP/SL targets
- 20h timeout prevents premature timeout exits
- Tier-based windows reward high-conviction signals with 24h runway

---

## Question 3: Tracker Start Time (Dynamic)

### Answer: FIXED - Tracker now uses deployment time (2026-03-24T18:52:00 GMT+7)

**Implementation (phase123_tracker_v2.py):**

**Deployed Configuration:**
```python
PHASE1_2_START_TIME = "2026-03-24T18:52:00"  # When fixes were deployed
```

**Why Dynamic:**
- Fixes deployed at 18:52 GMT+7 (not 00:00 next day)
- Tracker measures performance from actual fix deployment
- Shows true impact of Phase 1 & 2 changes

**Current Tracker Status (2026-03-25 00:11 GMT+7):**
- NEW signals: 3,691 (from 18:52 UTC onwards)
- Closed: 2,465 (66.8%)
- WR: 24.9% (vs Foundation 24.7%, +0.2pp)
- TF4h: 10 signals (barely meets >10 threshold)

---

## Summary of Fixes Applied

| Item | Status | Implementation | Date |
|------|--------|-----------------|------|
| TF4h RR Config | ✅ CONFIRMED | Market-driven S&R with 2.5:1 cap | 2026-03-25 |
| TF4h Timeout | ✅ FIXED | 20h (Champion), 12h-24h (Challenger) | 2026-03-25 |
| Tracker Start Time | ✅ FIXED | Dynamic deployment time 18:52 GMT+7 | 2026-03-25 |

---

## Outstanding Issues

### Critical:
1. **Route Veto:** Still 118 NONE/AMBIGUOUS signals leaked (veto fix not working as expected)
2. **TF4h Signal Count:** Only 10 signals in 5.5 hours (should be >50)

### Next Steps:
1. Debug route veto failure (post-Phase 3B route changes still bypassing veto)
2. Diagnose TF4h signal generation (gates or availability issue)
3. Monitor 24-48h for stabilization
4. Re-evaluate if min-max RR constraints needed for TF4h

---

**Tracker Command:** `python3 /Users/geniustarigan/.openclaw/workspace/phase123_tracker_v2.py`
