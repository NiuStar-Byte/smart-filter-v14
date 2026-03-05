# PEC Immutable Ledger System

## Problem Solved

**4-day investigation** revealed daemon crashes creating expanding gaps:
- Early Mar-05 (00:00-05:59 UTC): 197 signals ✓
- Gap 1 (06:00-11:59 UTC): **11-hour DEAD** ✗
- Mid Mar-05 (12:00-13:59 UTC): 30 signals ✓
- Gap 2 (14:00+ UTC): **Daemon hung** ✗

Reporter appeared to show "118 fired" due to timezone shifts + corrupted data, but only 47 real signals existed.

## Solution: Immutable Ledger Design

### Files

```
SIGNALS_LEDGER_IMMUTABLE.jsonl
├─ 1,557 signals consolidated from all sources
├─ Sorted by fired_time_utc
└─ Single source of truth (not dependent on daemon)

pec_immutable_reporter.py
├─ Reads ONLY from SIGNALS_LEDGER_IMMUTABLE.jsonl
├─ No connection to daemon
├─ Locks past dates (immutable)
└─ Accumulates new signals for today only

PEC_IMMUTABLE_REPORT.txt
├─ Daily output (human-readable)
├─ Hourly breakdown for today
└─ Shows gaps clearly (hours with 0 signals)
```

### Immutability Rules

**Past Dates (Feb-27 through Mar-04):**
- LOCKED: Cannot change signal counts
- Once a date closes, its metrics are final
- Example: Feb-27 = 80 signals, IMMUTABLE

**Today (Mar-05):**
- OPEN: Accumulates new signals as they arrive
- Only changes allowed = new signal accumulation
- Once today becomes "past", it freezes (6-hour windows lock hourly)

### Data Flow

```
Daemon fires signals → signals_fired.jsonl
                  ↓
    SIGNALS_LEDGER_IMMUTABLE.jsonl
                  ↓
    pec_immutable_reporter.py
                  ↓
    PEC_IMMUTABLE_REPORT.txt
```

### Hourly Breakdown (Mar-05 Example)

```
⏰ TOTAL FIRED TODAY HOURLY (UTC)
00:00-00:59 UTC: 77 fired | Begin: 00:02:56 | Last: 00:59:09 | [LOCKED at 01:00 UTC]
01:00-01:59 UTC: 19 fired | Begin: 01:08:04 | Last: 01:59:15 | [LOCKED at 02:00 UTC]
...
06:00-06:59 UTC: 0 fired | [DAEMON DEAD - visible in report]
07:00-07:59 UTC: 0 fired | [DAEMON DEAD]
...
12:00-12:59 UTC: 16 fired | Begin: 12:47:01 | Last: 12:59:58 | [Daemon restart visible]
13:00-13:59 UTC: 14 fired | Begin: 13:24:30 | Last: 13:57:26 | [OPEN - still accumulating]
```

## Key Advantages

1. **No daemon dependency** - Reporter reads immutable file
2. **Gaps visible** - Hourly breakdown shows exactly when daemon died
3. **Tamper-proof** - Past data cannot change once locked
4. **Accumulation only** - No complex recalculations or timezone shifts
5. **Clear audit trail** - Every signal has fired_time_utc in ledger

## Running the Reporter

```bash
cd /Users/geniustarigan/.openclaw/workspace
python3 pec_immutable_reporter.py
# Output: PEC_IMMUTABLE_REPORT.txt
```

## Signal Status Calculation

Signals are classified as:
- **TP_HIT**: Target profit hit
- **SL_HIT**: Stop loss hit
- **TIMEOUT_WIN**: Timeout with positive P&L
- **TIMEOUT_LOSS**: Timeout with negative P&L
- **OPEN**: Still running (no exit)

Win Rate = (TP_HIT + TIMEOUT_WIN) / Closed Trades

## Current Stats (as of Mar-5 14:00 UTC)

- **Total Signals**: 1,557
- **TP_HIT**: 213 (13.7%)
- **SL_HIT**: 445 (28.6%)
- **TIMEOUT_WIN**: 68 (4.4%)
- **TIMEOUT_LOSS**: 249 (16.0%)
- **OPEN**: 582 (37.4%)
- **Closed**: 975 (62.6%)
- **Win Rate**: 28.8%

## Future: Recovery from Daemon Crashes

Once ledger is complete (today ends), that day's data freezes. The daemon can:
- Crash multiple times
- Restart at any hour
- But it won't affect past days (immutable)

Only today's hourly buckets update as new signals arrive.
