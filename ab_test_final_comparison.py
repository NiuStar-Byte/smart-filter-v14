#!/usr/bin/env python3
"""
A/B Test Final Comparison: Champion vs Challenger (50 TIMEOUT Signals Each)

Runs when both groups have 50+ TIMEOUT signals.
Compares P&L, Win Rate, and declares winner.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

workspace = "/Users/geniustarigan/.openclaw/workspace"
master_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")

def load_signals():
    """Load all TIMEOUT signals grouped by champion/challenger"""
    champion_timeout = []
    challenger_timeout = []
    
    if not os.path.exists(master_file):
        print(f"[ERROR] {master_file} not found")
        return champion_timeout, challenger_timeout
    
    try:
        with open(master_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    
                    # Skip stale timeouts
                    if sig.get('data_quality_flag') and 'STALE_TIMEOUT' in sig.get('data_quality_flag'):
                        continue
                    
                    # Only TIMEOUT status
                    if sig.get('status') != 'TIMEOUT':
                        continue
                    
                    cc_group = sig.get('champion_challenger_group', 'NONE')
                    
                    if cc_group == 'CHAMPION':
                        champion_timeout.append(sig)
                    elif cc_group == 'CHALLENGER':
                        challenger_timeout.append(sig)
        
        return champion_timeout, challenger_timeout
    
    except Exception as e:
        print(f"[ERROR] Loading signals: {e}")
        return champion_timeout, challenger_timeout

def analyze_group(signals, group_name):
    """Analyze one group's timeout performance"""
    if not signals:
        return None
    
    # Take first 50 if more than 50
    signals = signals[:50]
    
    wins = sum(1 for s in signals if (s.get('pnl_usd') or 0) > 0)
    losses = sum(1 for s in signals if (s.get('pnl_usd') or 0) < 0)
    breaks = sum(1 for s in signals if (s.get('pnl_usd') or 0) == 0)
    
    total_pnl = sum(s.get('pnl_usd', 0) for s in signals)
    avg_pnl = total_pnl / len(signals) if len(signals) > 0 else 0
    
    closed = wins + losses
    wr = (wins / closed * 100) if closed > 0 else 0
    
    # By timeframe
    by_tf = defaultdict(lambda: {'wins': 0, 'losses': 0, 'count': 0, 'pnl': 0})
    for sig in signals:
        tf = sig.get('timeframe', '?')
        pnl = sig.get('pnl_usd', 0)
        by_tf[tf]['count'] += 1
        by_tf[tf]['pnl'] += pnl
        if pnl > 0:
            by_tf[tf]['wins'] += 1
        elif pnl < 0:
            by_tf[tf]['losses'] += 1
    
    return {
        'count': len(signals),
        'wins': wins,
        'losses': losses,
        'breaks': breaks,
        'closed': closed,
        'wr': wr,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'by_tf': by_tf
    }

def main():
    champion_timeout, challenger_timeout = load_signals()
    
    c_count = len(champion_timeout)
    ch_count = len(challenger_timeout)
    
    print("\n" + "="*100)
    print("⚔️  A/B TEST FINAL COMPARISON - TIMEOUT SIGNALS (50 Each)")
    print("="*100)
    
    # Check if ready
    if c_count < 50 or ch_count < 50:
        print(f"\n❌ NOT READY FOR FINAL COMPARISON")
        print(f"\n   Champion TIMEOUT: {c_count} / 50")
        print(f"   Challenger TIMEOUT: {ch_count} / 50")
        print(f"\n   Run: python3 ab_test_cutoff_monitor.py")
        return
    
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7)))
    print(f"\nGenerated: {now.strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"Sample Size: 50 TIMEOUT signals per group (exact match)")
    
    # Analyze
    c_stats = analyze_group(champion_timeout, "CHAMPION")
    ch_stats = analyze_group(challenger_timeout, "CHALLENGER")
    
    # Print comparison
    print(f"\n{'Metric':<30} {'CHAMPION':<25} {'CHALLENGER':<25} {'Winner':<20}")
    print("-"*100)
    
    print(f"{'Total TIMEOUT Signals':<30} {c_stats['count']:<25} {ch_stats['count']:<25}")
    
    print(f"\n{'Wins':<30} {c_stats['wins']:<25} {ch_stats['wins']:<25}", end='')
    if c_stats['wins'] > ch_stats['wins']:
        print(f"🏆 CHAMPION +{c_stats['wins'] - ch_stats['wins']}")
    elif ch_stats['wins'] > c_stats['wins']:
        print(f"🎯 CHALLENGER +{ch_stats['wins'] - c_stats['wins']}")
    else:
        print("TIE")
    
    print(f"{'Losses':<30} {c_stats['losses']:<25} {ch_stats['losses']:<25}", end='')
    if c_stats['losses'] < ch_stats['losses']:
        print(f"🏆 CHAMPION")
    elif ch_stats['losses'] < c_stats['losses']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'Breaks (PnL=0)':<30} {c_stats['breaks']:<25} {ch_stats['breaks']:<25}")
    
    print(f"\n{'Win Rate (Closed) %':<30} {c_stats['wr']:<24.1f}% {ch_stats['wr']:<24.1f}%", end='')
    if c_stats['wr'] > ch_stats['wr']:
        print(f"🏆 CHAMPION +{c_stats['wr'] - ch_stats['wr']:.1f}pp")
    elif ch_stats['wr'] > c_stats['wr']:
        print(f"🎯 CHALLENGER +{ch_stats['wr'] - c_stats['wr']:.1f}pp")
    else:
        print("TIE")
    
    print(f"{'Total P&L $':<30} ${c_stats['total_pnl']:<23.2f} ${ch_stats['total_pnl']:<23.2f}", end='')
    if c_stats['total_pnl'] > ch_stats['total_pnl']:
        print(f"🏆 CHAMPION +${c_stats['total_pnl'] - ch_stats['total_pnl']:.2f}")
    elif ch_stats['total_pnl'] > c_stats['total_pnl']:
        print(f"🎯 CHALLENGER +${ch_stats['total_pnl'] - c_stats['total_pnl']:.2f}")
    else:
        print("TIE")
    
    print(f"{'Avg P&L per Signal $':<30} ${c_stats['avg_pnl']:<23.2f} ${ch_stats['avg_pnl']:<23.2f}", end='')
    if c_stats['avg_pnl'] > ch_stats['avg_pnl']:
        print(f"🏆 CHAMPION +${c_stats['avg_pnl'] - ch_stats['avg_pnl']:.2f}")
    elif ch_stats['avg_pnl'] > c_stats['avg_pnl']:
        print(f"🎯 CHALLENGER +${ch_stats['avg_pnl'] - c_stats['avg_pnl']:.2f}")
    else:
        print("TIE")
    
    # By timeframe
    print(f"\n{'BY TIMEFRAME:':<30}")
    print("-"*100)
    print(f"{'TF':<10} {'Group':<15} {'Count':<8} {'WR%':<10} {'P&L':<15} {'Avg P&L':<15}")
    print("-"*100)
    
    for tf in ['15min', '30min', '1h']:
        c_tf = c_stats['by_tf'].get(tf, {})
        ch_tf = ch_stats['by_tf'].get(tf, {})
        
        if c_tf.get('count', 0) > 0:
            c_wr = (c_tf['wins'] / (c_tf['wins'] + c_tf['losses']) * 100) if (c_tf['wins'] + c_tf['losses']) > 0 else 0
            c_avg = c_tf['pnl'] / c_tf['count']
            print(f"{tf:<10} {'CHAMPION':<15} {c_tf['count']:<8} {c_wr:<9.1f}% ${c_tf['pnl']:<14.2f} ${c_avg:<14.2f}")
        
        if ch_tf.get('count', 0) > 0:
            ch_wr = (ch_tf['wins'] / (ch_tf['wins'] + ch_tf['losses']) * 100) if (ch_tf['wins'] + ch_tf['losses']) > 0 else 0
            ch_avg = ch_tf['pnl'] / ch_tf['count']
            print(f"{'':<10} {'CHALLENGER':<15} {ch_tf['count']:<8} {ch_wr:<9.1f}% ${ch_tf['pnl']:<14.2f} ${ch_avg:<14.2f}")
    
    # Decision
    print(f"\n{'='*100}")
    print("🎯 DECISION")
    print(f"{'='*100}")
    
    # Score the winner
    c_score = 0
    ch_score = 0
    
    if c_stats['wr'] > ch_stats['wr']:
        c_score += 1
    elif ch_stats['wr'] > c_stats['wr']:
        ch_score += 1
    
    if c_stats['total_pnl'] > ch_stats['total_pnl']:
        c_score += 1
    elif ch_stats['total_pnl'] > c_stats['total_pnl']:
        ch_score += 1
    
    if c_stats['avg_pnl'] > ch_stats['avg_pnl']:
        c_score += 1
    elif ch_stats['avg_pnl'] > c_stats['avg_pnl']:
        ch_score += 1
    
    if c_score > ch_score:
        print("\n🏆 CHAMPION WINS")
        print("\n✅ RECOMMENDATION: Keep current Champion strategy active")
        print("   Tier-based Challenger timeout adjustment is not improving performance.")
    elif ch_score > c_score:
        print("\n🎯 CHALLENGER WINS")
        print("\n✅ RECOMMENDATION: Deploy Challenger strategy as new standard")
        print("   Tier-based timeout windows are improving timeout exit performance.")
    else:
        print("\n🤝 RESULTS ARE TIED")
        print("\n⚠️  Cannot declare clear winner. Continue monitoring or collect more data.")
    
    print(f"\n{'='*100}\n")

if __name__ == '__main__':
    main()
