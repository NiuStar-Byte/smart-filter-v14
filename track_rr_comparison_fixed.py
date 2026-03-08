#!/usr/bin/env python3
"""
📊 RR COMPARISON - FIXED (Dual-Source)

Reads from TWO sources:
- 2.0:1 baseline: SENT_SIGNALS_CUMULATIVE_2026-03-08.jsonl (immutable, locked Feb 27 - Mar 4)
- 1.5:1 test: SIGNALS_MASTER.jsonl (live, daemon-updated)

This provides accurate comparison between locked baseline and live test.
"""

import json
import os
import sys

def load_baseline_2_0():
    """Load 2.0:1 baseline from immutable cumulative file"""
    stats = {'total': 0, 'closed': 0, 'open': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'pnl': 0}
    baseline_file = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS_CUMULATIVE_2026-03-08.jsonl"
    
    if not os.path.exists(baseline_file):
        return stats
    
    try:
        with open(baseline_file) as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    rr = sig.get('achieved_rr')
                    if rr != 2.0:
                        continue
                    
                    status = sig.get('status', 'OPEN')
                    stats['total'] += 1
                    
                    if status == 'OPEN':
                        stats['open'] += 1
                    else:
                        stats['closed'] += 1
                        pnl = sig.get('pnl_usd', 0)
                        stats['pnl'] += pnl
                        
                        if status == 'TP_HIT':
                            stats['tp'] += 1
                        elif status == 'SL_HIT':
                            stats['sl'] += 1
                        elif status == 'TIMEOUT':
                            stats['timeout'] += 1
    except Exception as e:
        print(f"[ERROR] Failed to load baseline: {e}", file=sys.stderr)
    
    return stats

def load_test_1_5():
    """Load 1.5:1 test from fresh daemon-updated file"""
    stats = {'total': 0, 'closed': 0, 'open': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'pnl': 0}
    test_file = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"
    
    if not os.path.exists(test_file):
        return stats
    
    try:
        with open(test_file) as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    rr = sig.get('achieved_rr')
                    if rr != 1.5:
                        continue
                    
                    status = sig.get('status', 'OPEN')
                    stats['total'] += 1
                    
                    if status == 'OPEN':
                        stats['open'] += 1
                    else:
                        stats['closed'] += 1
                        pnl = sig.get('pnl_usd', 0)
                        stats['pnl'] += pnl
                        
                        if status == 'TP_HIT':
                            stats['tp'] += 1
                        elif status == 'SL_HIT':
                            stats['sl'] += 1
                        elif status == 'TIMEOUT':
                            stats['timeout'] += 1
    except Exception as e:
        print(f"[ERROR] Failed to load test: {e}", file=sys.stderr)
    
    return stats

def print_comparison():
    """Print side-by-side comparison"""
    baseline = load_baseline_2_0()
    test = load_test_1_5()
    
    print("\n" + "="*85)
    print("RR VARIANT COMPARISON - DUAL SOURCE")
    print("="*85)
    print("RR | Total | Closed | OPEN | TP | SL | TIMEOUT | WR % | P&L $ | Avg P&L")
    print("-" * 85)
    
    for name, stats in [("2.0:1", baseline), ("1.5:1", test)]:
        wr = (stats['tp'] / (stats['tp'] + stats['sl'])) * 100 if (stats['tp'] + stats['sl']) > 0 else 0
        avg_pnl = stats['pnl'] / stats['closed'] if stats['closed'] > 0 else 0
        print(f"{name:6} | {stats['total']:5} | {stats['closed']:6} | {stats['open']:4} | {stats['tp']:3} | {stats['sl']:3} | {stats['timeout']:7} | {wr:5.1f}% | ${stats['pnl']:+8.2f} | ${avg_pnl:+6.2f}")
    
    # Calculate delta
    baseline_closed = baseline['closed'] if baseline['closed'] > 0 else 1
    test_closed = test['closed'] if test['closed'] > 0 else 1
    baseline_wr = (baseline['tp'] / (baseline['tp'] + baseline['sl'])) * 100 if (baseline['tp'] + baseline['sl']) > 0 else 0
    test_wr = (test['tp'] / (test['tp'] + test['sl'])) * 100 if (test['tp'] + test['sl']) > 0 else 0
    
    print("-" * 85)
    print(f"DELTA  |       |        |      |    |    |         | {test_wr - baseline_wr:+5.1f}% | ${test['pnl'] - baseline['pnl']:+8.2f} |")
    print("="*85 + "\n")

if __name__ == '__main__':
    print_comparison()
