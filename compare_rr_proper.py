#!/usr/bin/env python3
"""
📊 RR VARIANT COMPARISON - PROPER METHODOLOGY
Compares Baseline (pre-enhancement) vs 1.5:1 Test (post-enhancement, fresh)

Using PROPER CUTOFFS and EXCLUSIONS:
- Baseline: All signals fired before 2026-03-07
- 1.5:1 Test: Only signals fired from 2026-03-07 onwards
- Exclusions: STALE_TIMEOUT, REJECTED_NOT_SENT_TELEGRAM (never sent/orphaned)
"""

import json
from datetime import datetime
from collections import defaultdict
import os

workspace = "/Users/geniustarigan/.openclaw/workspace"
signals_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")

FRESH_START = "2026-03-07T00:00:00"

def load_baseline():
    """Load all signals BEFORE 2026-03-07 (baseline/control)"""
    signals = []
    try:
        with open(signals_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    if sig.get('fired_time_utc', '') < FRESH_START:
                        # Exclude stale/rejected
                        if sig.get('status') not in ['STALE_TIMEOUT'] and 'REJECTED' not in sig.get('status', ''):
                            signals.append(sig)
        return signals
    except:
        return signals

def load_fresh():
    """Load all signals FROM 2026-03-07 onwards (test group)"""
    signals = []
    try:
        with open(signals_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    if sig.get('fired_time_utc', '') >= FRESH_START:
                        # Exclude stale/rejected
                        if sig.get('status') not in ['STALE_TIMEOUT'] and 'REJECTED' not in sig.get('status', ''):
                            signals.append(sig)
        return signals
    except:
        return signals

def calculate_metrics(signals, label=""):
    """Calculate comprehensive metrics"""
    tp = sl = timeout = open_pos = 0
    total_pnl = 0.0
    rr_breakdown = defaultdict(lambda: {'tp': 0, 'sl': 0, 'timeout': 0, 'total': 0})
    
    for sig in signals:
        status = sig.get('status')
        rr = sig.get('achieved_rr', 'UNKNOWN')
        pnl = sig.get('pnl_usd', 0)
        
        rr_breakdown[rr]['total'] += 1
        
        if status == 'TP_HIT':
            tp += 1
            rr_breakdown[rr]['tp'] += 1
            total_pnl += float(pnl) if pnl else 0
        elif status == 'SL_HIT':
            sl += 1
            rr_breakdown[rr]['sl'] += 1
            total_pnl += float(pnl) if pnl else 0
        elif status == 'TIMEOUT':
            timeout += 1
            rr_breakdown[rr]['timeout'] += 1
            total_pnl += float(pnl) if pnl else 0
        elif status == 'OPEN':
            open_pos += 1
    
    closed = tp + sl + timeout
    wr = (tp / closed * 100) if closed > 0 else 0
    
    return {
        'total': len(signals),
        'tp': tp,
        'sl': sl,
        'timeout': timeout,
        'open': open_pos,
        'closed': closed,
        'wr': wr,
        'pnl': total_pnl,
        'avg_pnl': total_pnl / len(signals) if len(signals) > 0 else 0,
        'rr_breakdown': rr_breakdown
    }

def main():
    print("\n" + "="*120)
    print("📊 RR VARIANT COMPARISON - CLEAN METHODOLOGY (Proper Cutoffs & Exclusions)")
    print("="*120 + "\n")
    
    baseline = load_baseline()
    fresh = load_fresh()
    
    baseline_metrics = calculate_metrics(baseline, "BASELINE")
    fresh_metrics = calculate_metrics(fresh, "FRESH_1.5:1")
    
    # Print comparison table
    print(f"{'Metric':<30} | {'BASELINE (Pre-2026-03-07)':<35} | {'FRESH 1.5:1 (2026-03-07+)':<35} | {'Delta':<15}")
    print("-"*120)
    
    print(f"{'Total Signals':<30} | {baseline_metrics['total']:<35} | {fresh_metrics['total']:<35} | {fresh_metrics['total'] - baseline_metrics['total']:+<14}")
    print(f"{'TP_HIT':<30} | {baseline_metrics['tp']:<35} | {fresh_metrics['tp']:<35} | {fresh_metrics['tp'] - baseline_metrics['tp']:+<14}")
    print(f"{'SL_HIT':<30} | {baseline_metrics['sl']:<35} | {fresh_metrics['sl']:<35} | {fresh_metrics['sl'] - baseline_metrics['sl']:+<14}")
    print(f"{'TIMEOUT':<30} | {baseline_metrics['timeout']:<35} | {fresh_metrics['timeout']:<35} | {fresh_metrics['timeout'] - baseline_metrics['timeout']:+<14}")
    print(f"{'OPEN':<30} | {baseline_metrics['open']:<35} | {fresh_metrics['open']:<35} | {fresh_metrics['open'] - baseline_metrics['open']:+<14}")
    
    print("-"*120)
    
    print(f"{'Closed Signals':<30} | {baseline_metrics['closed']:<35} | {fresh_metrics['closed']:<35} | {fresh_metrics['closed'] - baseline_metrics['closed']:+<14}")
    print(f"{'WIN RATE %':<30} | {baseline_metrics['wr']:<34.1f}% | {fresh_metrics['wr']:<34.1f}% | {fresh_metrics['wr'] - baseline_metrics['wr']:+<13.1f}%")
    print(f"{'Total P&L':<30} | ${baseline_metrics['pnl']:<33,.2f} | ${fresh_metrics['pnl']:<33,.2f} | ${fresh_metrics['pnl'] - baseline_metrics['pnl']:+<13,.2f}")
    print(f"{'Avg P&L/Signal':<30} | ${baseline_metrics['avg_pnl']:<33.2f} | ${fresh_metrics['avg_pnl']:<33.2f} | ${fresh_metrics['avg_pnl'] - baseline_metrics['avg_pnl']:+<13.2f}")
    
    print("\n" + "="*120)
    print("\n🔍 RR BREAKDOWN - BASELINE (All data before 2026-03-07)")
    print("-"*120)
    print(f"{'RR':<10} | {'Total':<10} | {'TP':<10} | {'SL':<10} | {'TIMEOUT':<10} | {'Closed':<10} | {'WR %':<15}")
    print("-"*120)
    
    for rr in sorted([k for k in baseline_metrics['rr_breakdown'].keys() if isinstance(k, (int, float))]):
        d = baseline_metrics['rr_breakdown'][rr]
        closed = d['tp'] + d['sl'] + d['timeout']
        wr = (d['tp'] / closed * 100) if closed > 0 else 0
        print(f"{rr:<10} | {d['total']:<10} | {d['tp']:<10} | {d['sl']:<10} | {d['timeout']:<10} | {closed:<10} | {wr:<14.1f}%")
    
    print("\n" + "="*120)
    print("\n🔍 RR BREAKDOWN - FRESH 1.5:1 (Data from 2026-03-07 onwards)")
    print("-"*120)
    print(f"{'RR':<10} | {'Total':<10} | {'TP':<10} | {'SL':<10} | {'TIMEOUT':<10} | {'Closed':<10} | {'WR %':<15}")
    print("-"*120)
    
    for rr in sorted([k for k in fresh_metrics['rr_breakdown'].keys() if isinstance(k, (int, float))]):
        d = fresh_metrics['rr_breakdown'][rr]
        closed = d['tp'] + d['sl'] + d['timeout']
        wr = (d['tp'] / closed * 100) if closed > 0 else 0
        print(f"{rr:<10} | {d['total']:<10} | {d['tp']:<10} | {d['sl']:<10} | {d['timeout']:<10} | {closed:<10} | {wr:<14.1f}%")
    
    print("\n" + "="*120)
    print("\n⚠️  KEY FINDINGS")
    print("-"*120)
    print(f"✓ Baseline WR: {baseline_metrics['wr']:.1f}%")
    print(f"✓ Fresh 1.5:1 WR: {fresh_metrics['wr']:.1f}%")
    print(f"→ DELTA: {fresh_metrics['wr'] - baseline_metrics['wr']:+.1f}% (VERDICT: {'IMPROVEMENT ✓' if fresh_metrics['wr'] > baseline_metrics['wr'] else 'DEGRADATION ✗'})")
    print(f"\n✓ Baseline P&L: ${baseline_metrics['pnl']:,.2f}")
    print(f"✓ Fresh 1.5:1 P&L: ${fresh_metrics['pnl']:,.2f}")
    print(f"→ DELTA: ${fresh_metrics['pnl'] - baseline_metrics['pnl']:+,.2f}")
    
    print("\n" + "="*120 + "\n")

if __name__ == '__main__':
    main()
