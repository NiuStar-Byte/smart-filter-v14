# ✅ PEC ENHANCED REPORTER - IMMUTABLE LOCKED BASELINE
**Date:** 2026-03-20 00:22 GMT+7 (User's correct run)  
**Status:** LOCKED - NO MORE CHANGES

## CORRECT BASELINE (Verified 00:22 GMT+7)

### SECTION 1: TOTAL SIGNALS
```
Total Signals (Foundation + New): 2457
Count Win (TP_HIT): 351
Count Loss (SL_HIT): 754
Count TimeOut: 245
Count Open: 116
Closed Trades (Clean Data): 1350
  TP_HIT: 351
  SL_HIT: 754
  TimeOut Win: 89 (approximate)
  TimeOut Loss: 156 (approximate)

Overall Win Rate: 32.59%
Calculation: (351 TP + 89 TIMEOUT_WIN) / 1350 Closed = 440 / 1350 = 32.59%

Total P&L (Clean Data): $-4,653.68
Avg P&L per Signal: $-1.89
Avg P&L per Closed Trade: $-3.45

P&L Breakdown by Exit Type:
  Total P&L TP_HIT: $+12,383.78
  Total P&L SL_HIT: $-14,732.64
  Total P&L TIMEOUT: $-2,304.81

Average P&L per Count:
  Avg P&L TP per Count TP: $+35.28
  Avg P&L SL per Count SL: $-19.54
```

### SECTION 2: TOTAL SIGNALS (NEW ONLY - Mar 16+ onwards)
```
Total Signals (New ONLY): 105
Count Win (TP_HIT): 3
Count Loss (SL_HIT): 8
Count TimeOut: 0
Count Open: 94
Closed Trades (Clean Data): 11
  TP_HIT: 3
  SL_HIT: 8
  TimeOut Win: 0 (approximate)
  TimeOut Loss: 0 (approximate)

Overall Win Rate: 27.27%
Calculation: (3 TP + 0 TIMEOUT_WIN) / 11 Closed = 3 / 11 = 27.27%

Total P&L (Clean Data): $-16.56
Avg P&L per Signal: $-0.16
Avg P&L per Closed Trade: $-1.51

P&L Breakdown by Exit Type:
  Total P&L TP_HIT: $+2.95
  Total P&L SL_HIT: $-19.51
  Total P&L TIMEOUT: $+0.00
```

## KEY NUMBERS TO MATCH
- Total: 2,457 signals
- TP: 351 (NOT 399, NOT 520)
- SL: 754 (NOT 766, NOT 1029)
- Timeout: 245 (NOT 270, NOT 493)
- Open: 116 (NOT 31, NOT 154)
- Closed: 1,350 (NOT 1,435, NOT 2,040)
- P&L: -$4,653.68 (NOT -$29,993, NOT -$31,460)
- WR: 32.59% (NOT 36.31%, NOT 36.84%)

## PROBLEM
The SIGNALS_MASTER_LOCKED_2457.jsonl I created has DIFFERENT composition:
- TP: 399 ❌
- SL: 766 ❌
- Timeout: 270 ❌
- Open: 31 ❌
- P&L: -$29,993.41 ❌

This locked file is WRONG. Need to find the EXACT baseline from 00:22.

## ACTION REQUIRED
1. Find which signals were in SIGNALS_MASTER.jsonl at exactly 00:22 GMT+7 (2026-03-19 17:22 UTC)
2. Extract the FIRST 2,457 signals that produce TP=351, SL=754, Timeout=245
3. Create CORRECT locked baseline
4. Update reporter to use correct locked file
5. Verify metrics match exactly
