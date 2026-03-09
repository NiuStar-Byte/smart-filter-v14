#!/usr/bin/env python3
"""
A/B Test Staged Review: Champion vs Challenger at 10/20/30/40/50 TIMEOUT Signals

At each stage, shows:
1. TIMEOUT signals only (equal counts as cutoff)
2. Full P&L picture (TP_HIT, SL_HIT, OPEN, TIMEOUT all combined)
3. Head-to-head comparison

Usage:
  python3 ab_test_staged_review.py [stage]
  
  stage options: 10, 20, 30, 40, 50, or 'all' (default: 'all')
  
Examples:
  python3 ab_test_staged_review.py          # Show all stages (10, 20, 30, 40, 50)
  python3 ab_test_staged_review.py 20       # Show only stage 20
  python3 ab_test_staged_review.py all      # Show all stages
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict

workspace = "/Users/geniustarigan/.openclaw/workspace"
master_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")

def load_all_signals():
    """Load all signals grouped by champion/challenger and status"""
    champion_all = {'OPEN': [], 'TP_HIT': [], 'SL_HIT': [], 'TIMEOUT': []}
    challenger_all = {'OPEN': [], 'TP_HIT': [], 'SL_HIT': [], 'TIMEOUT': []}
    
    if not os.path.exists(master_file):
        print(f"[ERROR] {master_file} not found")
        return champion_all, challenger_all
    
    try:
        with open(master_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    
                    # Skip stale timeouts
                    if sig.get('data_quality_flag') and 'STALE_TIMEOUT' in sig.get('data_quality_flag'):
                        continue
                    
                    cc_group = sig.get('champion_challenger_group', 'NONE')
                    status = sig.get('status', 'UNKNOWN')
                    
                    if status not in champion_all:
                        status = 'UNKNOWN'
                    
                    if cc_group == 'CHAMPION':
                        champion_all[status].append(sig)
                    elif cc_group == 'CHALLENGER':
                        challenger_all[status].append(sig)
        
        return champion_all, challenger_all
    
    except Exception as e:
        print(f"[ERROR] Loading signals: {e}")
        return champion_all, challenger_all

def analyze_stage(champ_timeout, chall_timeout, champ_all, chall_all, stage_num):
    """Analyze one stage"""
    
    # Take first N TIMEOUT signals for each group
    c_timeout = champ_timeout[:stage_num]
    ch_timeout = chall_timeout[:stage_num]
    
    # But also include ALL OTHER signals from both groups
    c_all_signals = c_timeout + champ_all['TP_HIT'] + champ_all['SL_HIT'] + champ_all['OPEN']
    ch_all_signals = ch_timeout + chall_all['TP_HIT'] + chall_all['SL_HIT'] + chall_all['OPEN']
    
    def get_metrics(signals, timeout_only_sigs=None):
        """Calculate metrics for a group"""
        wins = sum(1 for s in signals if (s.get('pnl_usd') or 0) > 0)
        losses = sum(1 for s in signals if (s.get('pnl_usd') or 0) < 0)
        breaks = sum(1 for s in signals if (s.get('pnl_usd') or 0) == 0)
        
        total_pnl = sum(s.get('pnl_usd', 0) for s in signals)
        avg_pnl = total_pnl / len(signals) if len(signals) > 0 else 0
        
        closed = wins + losses
        wr = (wins / closed * 100) if closed > 0 else 0
        
        # Count by status
        status_counts = defaultdict(int)
        for s in signals:
            status_counts[s.get('status', 'UNKNOWN')] += 1
        
        return {
            'total': len(signals),
            'wins': wins,
            'losses': losses,
            'breaks': breaks,
            'closed': closed,
            'wr': wr,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'status_counts': dict(status_counts)
        }
    
    c_metrics = get_metrics(c_all_signals, c_timeout)
    ch_metrics = get_metrics(ch_all_signals, ch_timeout)
    
    # Timeout-only metrics
    c_timeout_metrics = get_metrics(c_timeout)
    ch_timeout_metrics = get_metrics(ch_timeout)
    
    return {
        'stage': stage_num,
        'c_timeout': c_timeout,
        'ch_timeout': ch_timeout,
        'c_all': c_all_signals,
        'ch_all': ch_all_signals,
        'c_metrics': c_metrics,
        'ch_metrics': ch_metrics,
        'c_timeout_metrics': c_timeout_metrics,
        'ch_timeout_metrics': ch_timeout_metrics
    }

def print_stage(result):
    """Print one stage"""
    stage = result['stage']
    c_m = result['c_metrics']
    ch_m = result['ch_metrics']
    c_t = result['c_timeout_metrics']
    ch_t = result['ch_timeout_metrics']
    
    print(f"\n{'='*120}")
    print(f"📊 STAGE {stage} - REVIEW POINT (TIMEOUT: {stage}/{stage} signals)")
    print(f"{'='*120}")
    
    print(f"\n{'SECTION 1: TIMEOUT SIGNALS ONLY (Equal Count = {stage} each)':<120}")
    print("-"*120)
    print(f"{'Metric':<35} {'CHAMPION':<35} {'CHALLENGER':<35} {'Winner':<15}")
    print("-"*120)
    
    print(f"{'Total TIMEOUT':<35} {c_t['total']:<35} {ch_t['total']:<35}")
    print(f"{'  Wins':<35} {c_t['wins']:<35} {ch_t['wins']:<35}", end='')
    if c_t['wins'] > ch_t['wins']:
        print(f"🏆 CHAMPION")
    elif ch_t['wins'] > c_t['wins']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'  Losses':<35} {c_t['losses']:<35} {ch_t['losses']:<35}", end='')
    if c_t['losses'] < ch_t['losses']:
        print(f"🏆 CHAMPION")
    elif ch_t['losses'] < c_t['losses']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'  Win Rate %':<35} {c_t['wr']:<34.1f}% {ch_t['wr']:<34.1f}%", end='')
    if c_t['wr'] > ch_t['wr']:
        print(f"🏆 CHAMPION")
    elif ch_t['wr'] > c_t['wr']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'  Total P&L':<35} ${c_t['total_pnl']:<33.2f} ${ch_t['total_pnl']:<33.2f}", end='')
    if c_t['total_pnl'] > ch_t['total_pnl']:
        print(f"🏆 CHAMPION")
    elif ch_t['total_pnl'] > c_t['total_pnl']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'  Avg P&L/Signal':<35} ${c_t['avg_pnl']:<33.2f} ${ch_t['avg_pnl']:<33.2f}", end='')
    if c_t['avg_pnl'] > ch_t['avg_pnl']:
        print(f"🏆 CHAMPION")
    elif ch_t['avg_pnl'] > c_t['avg_pnl']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"\n{'SECTION 2: FULL P&L PICTURE (All Exit Types Combined)':<120}")
    print("-"*120)
    print(f"{'Metric':<35} {'CHAMPION':<35} {'CHALLENGER':<35} {'Winner':<15}")
    print("-"*120)
    
    print(f"{'Total Signals':<35} {c_m['total']:<35} {ch_m['total']:<35}")
    
    print(f"\n{'Exit Type Breakdown':<35}")
    for status in ['TIMEOUT', 'TP_HIT', 'SL_HIT', 'OPEN']:
        c_count = c_m['status_counts'].get(status, 0)
        ch_count = ch_m['status_counts'].get(status, 0)
        print(f"  {status:<33} {c_count:<35} {ch_count:<35}")
    
    print(f"\n{'Wins':<35} {c_m['wins']:<35} {ch_m['wins']:<35}", end='')
    if c_m['wins'] > ch_m['wins']:
        print(f"🏆 CHAMPION")
    elif ch_m['wins'] > c_m['wins']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'Losses':<35} {c_m['losses']:<35} {ch_m['losses']:<35}", end='')
    if c_m['losses'] < ch_m['losses']:
        print(f"🏆 CHAMPION")
    elif ch_m['losses'] < c_m['losses']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'Win Rate %':<35} {c_m['wr']:<34.1f}% {ch_m['wr']:<34.1f}%", end='')
    if c_m['wr'] > ch_m['wr']:
        print(f"🏆 CHAMPION")
    elif ch_m['wr'] > c_m['wr']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'Total P&L':<35} ${c_m['total_pnl']:<33.2f} ${ch_m['total_pnl']:<33.2f}", end='')
    if c_m['total_pnl'] > ch_m['total_pnl']:
        print(f"🏆 CHAMPION")
    elif ch_m['total_pnl'] > c_m['total_pnl']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'Avg P&L/Signal':<35} ${c_m['avg_pnl']:<33.2f} ${ch_m['avg_pnl']:<33.2f}", end='')
    if c_m['avg_pnl'] > ch_m['avg_pnl']:
        print(f"🏆 CHAMPION")
    elif ch_m['avg_pnl'] > c_m['avg_pnl']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"\n{'='*120}\n")

def main():
    champion_all, challenger_all = load_all_signals()
    
    c_timeout_all = champion_all['TIMEOUT']
    ch_timeout_all = challenger_all['TIMEOUT']
    
    # Determine which stages to show
    if len(sys.argv) > 1 and sys.argv[1] != 'all':
        try:
            stages = [int(sys.argv[1])]
        except ValueError:
            stages = [10, 20, 30, 40, 50]
    else:
        stages = [10, 20, 30, 40, 50]
    
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7)))
    
    print(f"\n⚔️  A/B TEST STAGED REVIEW - Champion vs Challenger")
    print(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"\nCurrent TIMEOUT Collection:")
    print(f"  Champion: {len(c_timeout_all)} TIMEOUT signals")
    print(f"  Challenger: {len(ch_timeout_all)} TIMEOUT signals")
    
    for stage in stages:
        if len(c_timeout_all) >= stage and len(ch_timeout_all) >= stage:
            result = analyze_stage(c_timeout_all, ch_timeout_all, champion_all, challenger_all, stage)
            print_stage(result)
        else:
            print(f"\n⏳ Stage {stage}: Not enough data yet")
            print(f"   Need {stage} TIMEOUT from each group")
            print(f"   Champion: {min(len(c_timeout_all), stage)} / {stage}")
            print(f"   Challenger: {min(len(ch_timeout_all), stage)} / {stage}")

if __name__ == '__main__':
    main()
