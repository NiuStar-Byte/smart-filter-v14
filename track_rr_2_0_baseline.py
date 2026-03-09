#!/usr/bin/env python3
"""
📊 RR 2.0:1 BASELINE TRACKER
Uses signals from the Foundation/Baseline period (before filter enhancements)

Historical Data Period (from pec_reporter):
- Pre-enhancement baseline (locked)
- Represents "FOUNDATION BASELINE (IMMUTABLE)" = 853 signals, 25.7% WR

This is the control group for comparing against 1.5:1 variant.
"""

import json
from datetime import datetime
from collections import defaultdict
import os

workspace = "/Users/geniustarigan/.openclaw/workspace"
signals_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")

# Baseline period (immutable foundation baseline)
BASELINE_END = "2026-03-07T00:00:00"

def load_signals():
    """Load RR 2.0 signals (all signals before 2026-03-07, regardless of RR)"""
    signals = []
    
    if not os.path.exists(signals_file):
        return signals
    
    try:
        with open(signals_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    
                    fired = sig.get('fired_time_utc', '')
                    
                    # Filter 1: Only pre-2026-03-07 (baseline period)
                    if fired >= BASELINE_END:
                        continue
                    
                    # Filter 2: Exclude STALE_TIMEOUT
                    if sig.get('status') == 'STALE_TIMEOUT':
                        continue
                    
                    # Filter 3: Exclude REJECTED_NOT_SENT_TELEGRAM
                    if 'REJECTED' in sig.get('status', ''):
                        continue
                    
                    signals.append(sig)
        
        return signals
    except Exception as e:
        print(f"[ERROR] Loading signals: {e}")
        return signals

def analyze_metrics(signals):
    """Calculate metrics for signal group"""
    if not signals:
        return None
    
    tp_hits = 0
    sl_hits = 0
    timeouts = 0
    open_positions = 0
    total_pnl = 0.0
    
    for sig in signals:
        status = sig.get('status')
        pnl = sig.get('pnl_usd')
        
        if status == 'TP_HIT':
            tp_hits += 1
            if pnl:
                total_pnl += float(pnl)
        elif status == 'SL_HIT':
            sl_hits += 1
            if pnl:
                total_pnl += float(pnl)
        elif status == 'TIMEOUT':
            timeouts += 1
            if pnl:
                total_pnl += float(pnl)
        elif status == 'OPEN':
            open_positions += 1
    
    # Win Rate Calculation
    closed = tp_hits + sl_hits + timeouts
    wr_full = (tp_hits / closed * 100) if closed > 0 else 0
    wr_tp_sl = (tp_hits / (tp_hits + sl_hits) * 100) if (tp_hits + sl_hits) > 0 else 0
    
    return {
        'total': len(signals),
        'tp': tp_hits,
        'sl': sl_hits,
        'timeout': timeouts,
        'open': open_positions,
        'closed': closed,
        'wr_full': wr_full,
        'wr_tp_sl': wr_tp_sl,
        'pnl': total_pnl,
        'avg_pnl': total_pnl / len(signals) if len(signals) > 0 else 0
    }

def main():
    print("\n" + "="*100)
    print("📊 RR 2.0:1 BASELINE (Pre-Enhancement, Control Group)")
    print("="*100 + "\n")
    
    signals = load_signals()
    
    if not signals:
        print("[ERROR] No signals loaded!")
        return
    
    metrics = analyze_metrics(signals)
    
    print(f"Data Period: Start → 2026-03-07 (baseline/control)")
    print(f"Total Signals (Baseline): {metrics['total']}")
    print(f"  - TP_HIT: {metrics['tp']}")
    print(f"  - SL_HIT: {metrics['sl']}")
    print(f"  - TIMEOUT: {metrics['timeout']}")
    print(f"  - OPEN: {metrics['open']}")
    print(f"\n")
    
    print(f"Closed Signals: {metrics['closed']} (from {metrics['total']} total)")
    print(f"\n--- WIN RATE CALCULATIONS ---")
    print(f"1. TP+SL+TIMEOUT (full closure): {metrics['tp']} / {metrics['closed']} = {metrics['wr_full']:.1f}%")
    print(f"2. TP+SL only (conservative): {metrics['tp']} / {metrics['tp'] + metrics['sl']} = {metrics['wr_tp_sl']:.1f}%")
    print(f"\n--- PROFIT & LOSS ---")
    print(f"Total P&L: ${metrics['pnl']:,.2f}")
    print(f"Avg P&L per signal: ${metrics['avg_pnl']:.2f}")
    
    # RR breakdown
    print(f"\n--- RR BREAKDOWN ---")
    rr_breakdown = defaultdict(lambda: {'total': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'closed': 0})
    
    for sig in signals:
        rr = sig.get('achieved_rr', 'UNKNOWN')
        status = sig.get('status')
        
        rr_breakdown[rr]['total'] += 1
        if status == 'TP_HIT':
            rr_breakdown[rr]['tp'] += 1
            rr_breakdown[rr]['closed'] += 1
        elif status == 'SL_HIT':
            rr_breakdown[rr]['sl'] += 1
            rr_breakdown[rr]['closed'] += 1
        elif status == 'TIMEOUT':
            rr_breakdown[rr]['timeout'] += 1
            rr_breakdown[rr]['closed'] += 1
    
    for rr in sorted([k for k in rr_breakdown.keys() if isinstance(k, (int, float))]):
        d = rr_breakdown[rr]
        wr = (d['tp'] / d['closed'] * 100) if d['closed'] > 0 else 0
        print(f"  RR {rr}: Total={d['total']} | TP={d['tp']} | SL={d['sl']} | TIMEOUT={d['timeout']} | Closed={d['closed']} | WR={wr:.1f}%")
    
    print("\n" + "="*100 + "\n")

if __name__ == '__main__':
    main()
