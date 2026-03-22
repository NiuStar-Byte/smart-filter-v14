#!/usr/bin/env python3
"""
PEC Hourly Backtest + Reporter
1. Runs PEC Executor to process NEW OPEN signals
2. Runs PEC Reporter to generate latest metrics
3. Tracks progress
"""

import subprocess
import os
import sys
from datetime import datetime, timezone, timedelta

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*80}")
    print(f"[HOURLY] {description}")
    print(f"[HOURLY] Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, cwd='/Users/geniustarigan/.openclaw/workspace', capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"[HOURLY] ❌ Command failed with code {result.returncode}")
            if result.stderr:
                print(f"[HOURLY] STDERR:\n{result.stderr[:500]}")
            return False
        else:
            print(f"[HOURLY] ✅ {description} completed successfully")
            # Print last few lines of output
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    print(f"[HOURLY] {line}")
            return True
    except subprocess.TimeoutExpired:
        print(f"[HOURLY] ❌ Timeout after 300s")
        return False
    except Exception as e:
        print(f"[HOURLY] ❌ Error: {e}")
        return False

def main():
    now_gmt7 = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7)))
    timestamp = now_gmt7.strftime('%Y-%m-%d %H:%M:%S GMT+7')
    
    print(f"\n{'#'*80}")
    print(f"# PEC HOURLY BACKTEST + REPORT")
    print(f"# Time: {timestamp}")
    print(f"{'#'*80}\n")
    
    results = []
    
    # Step 1: Run PEC Executor (process NEW OPEN signals)
    print("\n[STEP 1] Running PEC Executor to process OPEN signals...")
    executor_ok = run_command(
        ['python3', 'pec_executor.py'],
        "PEC Executor (closes OPEN signals via backtest)"
    )
    results.append(('PEC Executor', executor_ok))
    
    # Step 2: Run Reporter
    print("\n[STEP 2] Running PEC Reporter...")
    reporter_ok = run_command(
        ['python3', 'pec_reporter_hourly.py'],
        "PEC Reporter Hourly (generates snapshots)"
    )
    results.append(('PEC Reporter', reporter_ok))
    
    # Step 3: Track Progress
    print("\n[STEP 3] Tracking progress...")
    progress_ok = run_command(
        ['python3', 'track_pec_progress.py'],
        "Progress Tracker (shows hourly metrics)"
    )
    results.append(('Progress Tracker', progress_ok))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"[HOURLY] SUMMARY ({timestamp})")
    print(f"{'='*80}")
    for name, ok in results:
        status = "✅ OK" if ok else "❌ FAILED"
        print(f"[HOURLY] {name}: {status}")
    
    all_ok = all(ok for _, ok in results)
    if all_ok:
        print(f"\n[HOURLY] ✅ All hourly tasks completed successfully")
        sys.exit(0)
    else:
        print(f"\n[HOURLY] ⚠️  Some tasks failed - see above for details")
        sys.exit(1)

if __name__ == '__main__':
    main()
