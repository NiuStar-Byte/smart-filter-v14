# CODE VERSION LOCK - Tracker Stability Reference

**Lock Date:** 2026-03-25 19:36 GMT+7

## FROZEN CODE STATE (Trackers depend on this)

### main.py (Latest: 350814a)
- **Status:** DirectionAwareGatekeeper DISABLED (all TFs)
- **2h TF:** LIVE integration (lines 2125-2497)
- **COOLDOWN:** 15min:120s, 30min:300s, 1h:420s, 2h:600s, 4h:900s
- **Decision:** Gatekeeper disabled because 4h (no gate) = 56.2% WR vs 1h (with gate) = 35.9% WR
- **⚠️  WARNING:** Any changes to these sections will affect ALL tracker results

### calculations.py (Latest: 1b7067a)
- **RR Calculation:** Uses entry_price (not current_price)
- **Impact:** 16 EXTREME_RR signals flagged, excluded from metrics
- **⚠️  WARNING:** RR formula changes affect WR/P&L calculations

### pec_config.py (Latest: frozen)
- **MAX_BARS_BY_TF:** {15min:8, 30min:6, 1h:4, 2h:3, 4h:2}
- **Timeout Windows:** 15min:2h, 30min:3h, 1h:4h, 2h:6h, 4h:8h
- **⚠️  WARNING:** These are the basis for timeout metrics in reports

## TRACKER BASELINE (As of 2026-03-25 19:36)

```
SIGNALS_MASTER.jsonl: 7,223 signals (synced from AUDIT)
FOUNDATION baseline: 2,224 (locked at 2026-03-14T23:59:59.999999)
NEW signals: 4,999 (from 2026-03-21+)
Overall WR: 30.94%
Total P&L: -$13,165.15
```

## ⚡ CRITICAL RULE

**IF YOU CHANGE:**
- main.py (signal generation logic)
- calculations.py (RR/P&L calculation)
- pec_config.py (timeout architecture)

**THEN YOU MUST:**
1. Re-baseline all 4 trackers with new data
2. Document the change impact
3. Update this VERSION file
4. Notify user that baseline has shifted
5. DO NOT make changes to tracker code/templates
6. Run full validation before deployment

