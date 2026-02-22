#!/usr/bin/env python3
"""
diagnose_signal_sync.py

Diagnose why signals fire to Telegram but don't appear in signals_fired.jsonl

Checks:
1. Is main.py running?
2. What signals are in logs.txt?
3. What signals are in signals_fired.jsonl?
4. Are they the same?
"""

import os
import json
import subprocess
from datetime import datetime

print("\n" + "="*70)
print("SIGNAL SYNC DIAGNOSTIC")
print("="*70)

# 1. Check if main.py is running
print("\n[CHECK 1] Is main.py running?")
result = subprocess.run(["pgrep", "-f", "main.py"], capture_output=True, text=True)
if result.stdout.strip():
    print(f"  ✓ main.py IS running (PID: {result.stdout.strip()})")
else:
    print(f"  ✗ main.py NOT running")
    print(f"    → create_and_store_signal() is never called")
    print(f"    → signals_fired.jsonl remains empty")

# 2. Check logs.txt
print("\n[CHECK 2] Signals in logs.txt?")
if os.path.exists("logs.txt"):
    with open("logs.txt", "r") as f:
        lines = [l for l in f if "[FIRED]" in l]
    print(f"  ✓ logs.txt exists with {len(lines)} fired signals")
    if lines:
        print(f"    Last signal: {lines[-1][:80]}...")
else:
    print(f"  ✗ logs.txt not found")

# 3. Check signals_fired.jsonl
print("\n[CHECK 3] Signals in signals_fired.jsonl?")
if os.path.exists("signals_fired.jsonl"):
    with open("signals_fired.jsonl", "r") as f:
        jsonl_lines = [l.strip() for l in f if l.strip()]
    print(f"  ✓ signals_fired.jsonl exists with {len(jsonl_lines)} signals")
    if jsonl_lines:
        last = json.loads(jsonl_lines[-1])
        print(f"    Last signal: {last['symbol']} {last['timeframe']} @ {last['fired_time_utc']}")
else:
    print(f"  ✗ signals_fired.jsonl NOT found")
    print(f"    → Never created")
    print(f"    → create_and_store_signal() never called")

# 4. Are they in sync?
print("\n[CHECK 4] Sync Status?")
logs_count = len(lines) if 'lines' in locals() and lines else 0
jsonl_count = len(jsonl_lines) if 'jsonl_lines' in locals() and jsonl_lines else 0

if logs_count > 0 and jsonl_count == 0:
    print(f"  ✗ SYNC BROKEN")
    print(f"    logs.txt: {logs_count} signals")
    print(f"    signals_fired.jsonl: {jsonl_count} signals")
    print(f"    Gap: {logs_count} signals missing from JSONL")
    print(f"    ")
    print(f"  Reason: main.py is not running with new create_and_store_signal() code")
elif logs_count > 0 and jsonl_count > 0:
    if logs_count == jsonl_count:
        print(f"  ✓ IN SYNC ({jsonl_count} signals in both)")
    else:
        print(f"  ⚠ PARTIAL SYNC")
        print(f"    logs.txt: {logs_count} signals")
        print(f"    signals_fired.jsonl: {jsonl_count} signals")
        print(f"    Gap: {logs_count - jsonl_count} signals missing")
elif logs_count == 0 and jsonl_count == 0:
    print(f"  ⚠ NO SIGNALS ANYWHERE")
    print(f"    main.py has not been run yet (or recently)")
else:
    print(f"  ⚠ UNEXPECTED STATE")
    print(f"    logs.txt: {logs_count} signals")
    print(f"    signals_fired.jsonl: {jsonl_count} signals")

# 5. Recommendation
print("\n[RECOMMENDATION]")
print("="*70)

if not result.stdout.strip():
    print("\n❌ ACTION REQUIRED: Start main.py")
    print("\nRun this command to start main.py:")
    print("  cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main")
    print("  nohup python3 main.py > main.log 2>&1 &")
    print("\nThen monitor:")
    print("  tail -f main.log            # See what's happening")
    print("  tail -f signals_fired.jsonl # See new signals")
    print("  python3 count_signals_by_tf.py  # Daily progress")
else:
    print("\n✓ main.py is running")
    print("  Monitor progress with:")
    print("    python3 count_signals_by_tf.py")
    print("    tail -f signals_fired.jsonl")

print("\n" + "="*70 + "\n")
