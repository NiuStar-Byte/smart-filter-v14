# CRITICAL DIAGNOSTIC FINDINGS - 2026-04-05 01:18-01:35 GMT+7

## THE MICRO-LEVEL ISSUE UNCOVERED

### **Problem 1: Symbol_group Field Missing (NOW MOSTLY FIXED)**
- **Symptom:** 22% of recent signals missing `symbol_group` and `confidence_level` fields
- **Root Cause:** Variables were undefined in 30min and 1h timeframe blocks
- **Fix Applied:** Added `symbol_group = classify_symbol(symbol_val)` to both blocks
- **Result:** Completion improved 22% → 95% (95 out of 100 recent signals now have complete fields)

### **Problem 2: Zombie Processes Blocking Signal Generation**
- **Symptom:** Old main.py instances running since 12:47 AM, 12:52 AM (2+ hours old)
- **Impact:** Multiple processes competing for file writes, signal generation stalled
- **Action Taken:** Killed all 5 zombie processes (PIDs: 27574, 28427, 32198, 32710, 31812)
- **New Instance:** Fresh main.py started (PID: 33052) - single, clean instance

### **Problem 3: BERA-USDT Tier-2 Signal Not Persisted (CRITICAL)**
- **Observation:** BERA-USDT fired at 00:56 GMT+7 with Tier-2 assignment shown in Telegram alert
- **Combo Matched:** `TF_DIR_ROUTE_REGIME_4h_SHORT_REVERSAL_BEAR` (4D, 68.5% WR) - correctly identified as Tier-2
- **ISSUE:** Signal is NOT in SIGNALS_MASTER.jsonl file (never persisted)
- **Status:** Tier matching works (Telegram shows correct tier), but write_signal() is failing
- **Hypothesis:** Signal may be processed by a different main.py instance (possibly in smart-filter-v14-main directory)

### **Key Insight from User**
The user correctly emphasized that tier assignment IS WORKING (Telegram shows Tier-2 for BERA correctly). The problem is NOT tier matching - it's signal persistence. The signal fires with correct Tier-2 detection but isn't being written to the master file.

## CURRENT SYSTEM STATE (Post-Fix)
| Metric | Status | Details |
|--------|--------|---------|
| Field Completion Rate | 95% ✅ | Up from 22% (original) via symbol_group fixes |
| Zombie Processes | 0 ✅ | All killed, fresh instance running |
| Tier Matching | ✓ Working | Telegram shows correct tiers (Tier-2 for BERA detected) |
| Signal Persistence | ❌ Issue | BERA not in SIGNALS_MASTER.jsonl despite Telegram alert |
| Recent Tier-2 Signals | 1 of 200 | Only 1 old signal from 2026-03-28 in last 200 signals |

## REMAINING INVESTIGATION NEEDED
1. **Which process sends the BERA alert to Telegram?** (main.py in workspace vs smart-filter-v14-main)
2. **Why is BERA not persisted?** (Silent exception, missing _master_writer_ready, file lock, etc.)
3. **Are there other "ghost" signals fired to Telegram but not persisted?**

## NEXT STEP FOR USER
Run the following to identify which main.py is sending BERA alerts:
```bash
grep -r "BERA" /Users/geniustarigan/.openclaw/workspace/*main*.log | grep -i "tier"
```
This will show which process detected BERA and when. Then we can trace why that signal wasn't persisted.
