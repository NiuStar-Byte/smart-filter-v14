#!/usr/bin/env python3
"""
count_signals_by_tf.py

Quick utility to count signals by timeframe and show progress toward Batch 1 (50 signals).
Usage: python3 count_signals_by_tf.py
"""

import json
import os
from collections import defaultdict
from pec_config import SIGNALS_JSONL_PATH

def count_signals():
    """Count signals by timeframe from signals_fired.jsonl."""
    
    if not os.path.exists(SIGNALS_JSONL_PATH):
        print(f"⚠️  Signal file not found: {SIGNALS_JSONL_PATH}")
        print(f"   Signals not yet firing. Check if main.py is running.")
        return
    
    tf_counts = defaultdict(int)
    type_counts = defaultdict(int)
    total_rr = defaultdict(float)
    signal_count = 0
    
    try:
        with open(SIGNALS_JSONL_PATH, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    signal = json.loads(line)
                    signal_count += 1
                    
                    tf = signal.get('timeframe', 'UNKNOWN')
                    signal_type = signal.get('signal_type', 'UNKNOWN')
                    rr = signal.get('achieved_rr', 0)
                    
                    tf_counts[tf] += 1
                    type_counts[f"{tf}_{signal_type}"] += 1
                    total_rr[tf] += rr
                
                except json.JSONDecodeError as e:
                    print(f"⚠️  Line {line_num}: JSON parse error: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Error reading signals: {e}")
        return
    
    # Display results
    print("\n" + "="*60)
    print("BATCH 1 SIGNAL ACCUMULATION STATUS")
    print("="*60)
    print(f"\nTotal Signals: {signal_count}/50 ({(signal_count/50*100):.1f}% progress)")
    print(f"File: {SIGNALS_JSONL_PATH}")
    
    if signal_count == 0:
        print("\n⚠️  No signals yet. Check that main.py is running and firing signals.")
        return
    
    # Progress bar
    filled = int((signal_count / 50) * 30)
    bar = "█" * filled + "░" * (30 - filled)
    print(f"\n[{bar}] {signal_count}/50")
    
    # By timeframe
    print("\nBy Timeframe:")
    print("-" * 60)
    for tf in sorted(tf_counts.keys()):
        count = tf_counts[tf]
        avg_rr = total_rr[tf] / count if count > 0 else 0
        long_count = type_counts.get(f"{tf}_LONG", 0)
        short_count = type_counts.get(f"{tf}_SHORT", 0)
        print(f"  {tf:6s}: {count:3d} signals | L:{long_count:2d} S:{short_count:2d} | Avg RR: {avg_rr:.2f}:1")
    
    # Summary stats
    total_rr_avg = sum(total_rr.values()) / signal_count if signal_count > 0 else 0
    print("\n" + "-" * 60)
    print(f"Overall Avg RR: {total_rr_avg:.2f}:1")
    
    # Ready for batch?
    if signal_count >= 50:
        print("\n✅ BATCH 1 READY!")
        print("   Run: python3 -c \"from pec_backtest_v2 import run_pec_backtest_v2; ...\"")
    else:
        remaining = 50 - signal_count
        print(f"\n⏳ Waiting for {remaining} more signals ({(remaining/50*100):.1f}% to go)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    count_signals()
