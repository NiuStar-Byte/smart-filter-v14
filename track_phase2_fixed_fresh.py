#!/usr/bin/env python3
"""
PHASE 2-FIXED Fresh Start Tracker (2026-03-08 11:20 GMT+7 onwards)

Tracks only signals fired AFTER reset point, ignoring corrupted Mar 3-8 period.
"""

import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os

workspace = "/Users/geniustarigan/.openclaw/workspace"
signals_file = os.path.join(workspace, "SENT_SIGNALS.jsonl")

# Reset point: 2026-03-08 11:20 GMT+7 = 2026-03-08 04:20 UTC
PHASE2_FIXED_FRESH_START = "2026-03-08T04:20:00"

def load_signals():
    """Load Phase 2-FIXED signals (only after fresh start)"""
    signals = []
    
    if not os.path.exists(signals_file):
        print(f"[ERROR] {signals_file} not found")
        return signals
    
    try:
        with open(signals_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    fired = sig.get('fired_time_utc', '')
                    
                    # Only count signals AFTER fresh start
                    if fired >= PHASE2_FIXED_FRESH_START:
                        signals.append(sig)
        
        return signals
    except Exception as e:
        print(f"[ERROR] Loading signals: {e}")
        return signals

def analyze_metrics(signals):
    """Calculate P&L metrics"""
    if not signals:
        return None
    
    # Status breakdown
    by_status = defaultdict(list)
    for sig in signals:
        status = sig.get('status')
        if status:
            by_status[status].append(sig)
    
    # P&L calculation
    wins = 0
    losses = 0
    breaks = 0
    total_pnl = 0.0
    
    for sig in signals:
        pnl = sig.get('pnl_usd')
        if pnl is not None:
            total_pnl += float(pnl)
            if float(pnl) > 0:
                wins += 1
            elif float(pnl) < 0:
                losses += 1
            else:
                breaks += 1
    
    closed = wins + losses
    wr = (wins / closed * 100) if closed > 0 else 0
    avg_pnl = total_pnl / len(signals) if len(signals) > 0 else 0
    
    # By direction
    by_direction = defaultdict(lambda: {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0})
    for sig in signals:
        direction = sig.get('direction')
        if direction in ['LONG', 'SHORT']:
            by_direction[direction]['count'] += 1
            pnl = sig.get('pnl_usd', 0)
            by_direction[direction]['pnl'] += float(pnl) if pnl else 0
            
            if pnl and float(pnl) > 0:
                by_direction[direction]['wins'] += 1
            elif pnl and float(pnl) < 0:
                by_direction[direction]['losses'] += 1
    
    # By regime
    by_regime = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0}))
    for sig in signals:
        regime = sig.get('regime', 'UNKNOWN')
        direction = sig.get('direction')
        if direction in ['LONG', 'SHORT']:
            by_regime[regime][direction]['count'] += 1
            pnl = sig.get('pnl_usd', 0)
            by_regime[regime][direction]['pnl'] += float(pnl) if pnl else 0
            
            if pnl and float(pnl) > 0:
                by_regime[regime][direction]['wins'] += 1
            elif pnl and float(pnl) < 0:
                by_regime[regime][direction]['losses'] += 1
    
    return {
        'total': len(signals),
        'by_status': dict(by_status),
        'wins': wins,
        'losses': losses,
        'breaks': breaks,
        'closed': closed,
        'wr': wr,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'by_direction': dict(by_direction),
        'by_regime': dict(by_regime)
    }

def print_report(metrics):
    """Print formatted report"""
    print("\n" + "="*100)
    print("🚀 PHASE 2-FIXED: FRESH START TRACKER")
    print("="*100)
    
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7)))
    print(f"\nGenerated: {now.strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"Reset Point: 2026-03-08 11:20 GMT+7 (2026-03-08 04:20 UTC)")
    
    if not metrics:
        print("\n⏳ No signals yet. Check back in a few hours.")
        return
    
    print(f"\n{'='*100}")
    print("📊 OVERALL METRICS")
    print(f"{'='*100}")
    
    print(f"\nTotal Signals: {metrics['total']}")
    print(f"  Status Breakdown:")
    for status in sorted(metrics['by_status'].keys()):
        count = len(metrics['by_status'][status])
        print(f"    {status}: {count}")
    
    print(f"\nClosed Trades: {metrics['closed']}")
    print(f"  Wins: {metrics['wins']}")
    print(f"  Losses: {metrics['losses']}")
    print(f"  Breaks (PnL=0): {metrics['breaks']}")
    
    if metrics['closed'] > 0:
        print(f"\n  Win Rate: {metrics['wr']:.1f}%")
        print(f"  Total P&L: ${metrics['total_pnl']:+.2f}")
        print(f"  Avg P&L/Signal: ${metrics['avg_pnl']:+.2f}")
    
    # By Direction
    print(f"\n{'='*100}")
    print("BY DIRECTION")
    print(f"{'='*100}")
    
    for direction in ['LONG', 'SHORT']:
        if direction in metrics['by_direction']:
            d = metrics['by_direction'][direction]
            if d['count'] > 0:
                closed_dir = d['wins'] + d['losses']
                wr_dir = (d['wins'] / closed_dir * 100) if closed_dir > 0 else 0
                print(f"\n{direction}:")
                print(f"  Count: {d['count']}")
                print(f"  Win Rate: {wr_dir:.1f}% ({d['wins']}W {d['losses']}L)")
                print(f"  P&L: ${d['pnl']:+.2f}")
    
    # By Regime
    print(f"\n{'='*100}")
    print("BY REGIME & DIRECTION (KEY METRICS)")
    print(f"{'='*100}")
    
    for regime in sorted(metrics['by_regime'].keys()):
        regime_data = metrics['by_regime'][regime]
        if regime_data:
            print(f"\n{regime}:")
            for direction in ['LONG', 'SHORT']:
                if direction in regime_data:
                    d = regime_data[direction]
                    if d['count'] > 0:
                        closed_dir = d['wins'] + d['losses']
                        wr_dir = (d['wins'] / closed_dir * 100) if closed_dir > 0 else 0
                        print(f"  {direction:5s}: {d['count']:3d} sigs | WR: {wr_dir:5.1f}% | P&L: ${d['pnl']:+8.2f}")
    
    # Comparison to Foundation
    print(f"\n{'='*100}")
    print("COMPARISON TO FOUNDATION BASELINE")
    print(f"{'='*100}")
    
    foundation_wr = 25.7
    print(f"\nFOUNDATION: 25.7% WR (853 signals)")
    print(f"PHASE 2-FIXED: {metrics['wr']:.1f}% WR ({metrics['closed']} closed trades)")
    
    if metrics['wr'] >= foundation_wr:
        diff = metrics['wr'] - foundation_wr
        print(f"\n✅ PHASE 2-FIXED beating Foundation by +{diff:.1f}pp")
    else:
        diff = foundation_wr - metrics['wr']
        print(f"\n⏳ PHASE 2-FIXED behind Foundation by {diff:.1f}pp")
        if metrics['closed'] >= 50:
            print(f"   Need {foundation_wr - metrics['wr']:.1f}pp improvement with {metrics['closed']} closed trades")
        else:
            print(f"   Need more data ({metrics['closed']} closed, target 100)")
    
    print(f"\n{'='*100}\n")

def main():
    signals = load_signals()
    metrics = analyze_metrics(signals)
    print_report(metrics)

if __name__ == '__main__':
    main()
