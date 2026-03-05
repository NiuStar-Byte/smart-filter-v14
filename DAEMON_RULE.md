# 🔐 DAEMON RULE - CRITICAL CONSTRAINT

## THE RULE (NON-NEGOTIABLE)

**ONLY 1 DAEMON INSTANCE AT A TIME**

```
❌ NEVER: Multiple daemons (causes race conditions + lost signals)
✅ ALWAYS: Single daemon instance
```

## Why This Matters

When multiple daemons write to `SENT_SIGNALS.jsonl` simultaneously:
- Daemon A writes signal X
- Daemon B writes signal Y
- Daemon A's write gets overwritten
- **Result: Lost signals!**

**Real Example (Mar 5):** 3 daemons running → 12-hour gap of lost signals (07:16-19:47)

## How to Check & Fix

**Check current status:**
```bash
pgrep -f "smart-filter-v14-main/main.py" | wc -l
```

**If > 1, kill all and restart:**
```bash
pkill -f "smart-filter-v14-main/main.py"
sleep 3
bash ensure_single_daemon.sh
```

**To start daemon safely:**
```bash
bash ensure_single_daemon.sh
```
(This checks for existing instances before starting)

## The Violation (2026-03-05)

```
Commitment: 1 daemon only
Reality:    3 daemons running
Impact:     Lost signals 07:16-19:47 (12-hour gap)
Status:     ✅ FIXED at 20:15 with single fresh daemon
```

## From Now On

Every daemon restart MUST:
1. ✅ Kill all existing instances first
2. ✅ Verify only 0 running
3. ✅ Start 1 fresh instance
4. ✅ Verify only 1 running

**NO EXCUSES. NO EXCEPTIONS. ONE DAEMON ONLY.**
