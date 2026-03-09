# A/B TEST CUTOFF AGREEMENT - 2026-03-08 10:27 GMT+7

## Test Design: Champion vs Challenger (TIMEOUT-Based Comparison)

### **Sample Size Agreement**
- **TIMEOUT signals must be equal for both groups: 50 each**
- Exit type distribution (TP_HIT, SL_HIT, OPEN) may differ between groups — this is allowed
- Only TIMEOUT counts toward the cutoff threshold

### **Current Status (2026-03-08 10:27 GMT+7)**
- Champion TIMEOUT: 8 / 50 (need 42 more)
- Challenger TIMEOUT: 5 / 50 (need 45 more)
- **Progress:** 13 / 100 signals collected

### **Comparison Framework (When 50/50 Reached)**
1. Load all TIMEOUT signals for Champion (n=50)
2. Load all TIMEOUT signals for Challenger (n=50)
3. Compare head-to-head:
   - Win Rate (% of TIMEOUT trades that profit)
   - Total P&L (aggregate across 50 TIMEOUT signals)
   - Avg P&L per signal
   - P&L distribution by timeframe (15min/30min/1h)

4. Declare winner based on P&L + Win Rate
5. If Challenger wins: Deploy as new standard
6. If Champion wins: Keep current strategy

### **Other Exit Paths (Tracked Separately, Not Part of Cutoff)**
- TP_HIT: Early profit-taking (before timeout)
- SL_HIT: Stop loss hit (before timeout)
- OPEN: Still accumulating (not yet resolved)

These will be analyzed **after** the TIMEOUT comparison, to understand full performance profile.

### **Estimated Timeline**
At current firing rate (~190 signals/day total across 92 symbols, 3 timeframes):
- Average signals per group per day: ~95 / 3 groups = ~32 signals/day per group
- Time to 50 TIMEOUT each: ~1-2 weeks (depending on TIMEOUT % of all signals)

### **Monitoring Command**
```bash
python3 ab_test_cutoff_monitor.py
```
Outputs current progress toward 50/50 and ETA.

---

**Decision Made By:** User (geniustarigan)  
**Date Agreed:** 2026-03-08 10:27 GMT+7  
**Test Status:** SILENT (traders unaware)
