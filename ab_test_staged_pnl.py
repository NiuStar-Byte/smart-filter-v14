#!/usr/bin/env python3
"""
A/B TEST STAGED REVIEW - Full P&L Picture (TIMEOUT + All Exits)

Shows both:
1. TIMEOUT signals only (equal count cutoff)
2. Full P&L picture (all exit types combined)

Usage:
  python3 ab_test_staged_pnl.py              # Auto-refresh every 5 seconds
  python3 ab_test_staged_pnl.py 20           # Show stage 20 only
  python3 ab_test_staged_pnl.py --once       # Single run (no auto-refresh)
"""

import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import sys
import os
import subprocess
import time

workspace = "/Users/geniustarigan/.openclaw/workspace"
master_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")

def clear_screen():
    """Clear terminal"""
    subprocess.call('clear' if os.name == 'posix' else 'cls', shell=True)

def load_all_signals():
    """Load all signals by champion/challenger and status"""
    champion_all = {'OPEN': [], 'TP_HIT': [], 'SL_HIT': [], 'TIMEOUT': []}
    challenger_all = {'OPEN': [], 'TP_HIT': [], 'SL_HIT': [], 'TIMEOUT': []}
    
    if not os.path.exists(master_file):
        return champion_all, challenger_all
    
    try:
        with open(master_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    
                    # Skip stale
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
    except:
        return champion_all, challenger_all

def analyze_signals(sigs):
    """Analyze one group"""
    wins = sum(1 for s in sigs if (s.get('pnl_usd') or 0) > 0)
    losses = sum(1 for s in sigs if (s.get('pnl_usd') or 0) < 0)
    
    total_pnl = 0.0
    for s in sigs:
        pnl = s.get('pnl_usd')
        if pnl is not None:
            total_pnl += float(pnl)
    
    closed = wins + losses
    wr = (wins / closed * 100) if closed > 0 else 0
    avg = total_pnl / len(sigs) if len(sigs) > 0 else 0
    
    return {
        'count': len(sigs),
        'wins': wins,
        'losses': losses,
        'closed': closed,
        'wr': wr,
        'total_pnl': total_pnl,
        'avg_pnl': avg
    }

def print_stage(stage_num, champion_all, challenger_all):
    """Print one stage"""
    # Get TIMEOUT signals up to stage number
    c_timeout = champion_all['TIMEOUT'][:stage_num]
    ch_timeout = challenger_all['TIMEOUT'][:stage_num]
    
    # Check if ready
    if len(c_timeout) < stage_num or len(ch_timeout) < stage_num:
        print(f"\n⏳ Stage {stage_num}: Not enough data yet")
        print(f"   Champion TIMEOUT: {len(c_timeout)} / {stage_num}")
        print(f"   Challenger TIMEOUT: {len(ch_timeout)} / {stage_num}")
        return False
    
    # Get all signals
    c_all = c_timeout + champion_all['TP_HIT'] + champion_all['SL_HIT'] + champion_all['OPEN']
    ch_all = ch_timeout + challenger_all['TP_HIT'] + challenger_all['SL_HIT'] + challenger_all['OPEN']
    
    # Analyze
    c_t = analyze_signals(c_timeout)
    ch_t = analyze_signals(ch_timeout)
    c_m = analyze_signals(c_all)
    ch_m = analyze_signals(ch_all)
    
    print("\n" + "="*120)
    print(f"📊 STAGE {stage_num} - FULL P&L PICTURE (TIMEOUT: {stage_num}/{stage_num} signals)")
    print("="*120)
    
    # SECTION 1: TIMEOUT ONLY
    print(f"\n{'SECTION 1: TIMEOUT SIGNALS ONLY':<120}")
    print("-"*120)
    print(f"{'Metric':<35} {'CHAMPION':<35} {'CHALLENGER':<35} {'Winner':<15}")
    print("-"*120)
    
    print(f"{'Total TIMEOUT':<35} {c_t['count']:<35} {ch_t['count']:<35}")
    
    print(f"{'Wins':<35} {c_t['wins']:<35} {ch_t['wins']:<35}", end='')
    if c_t['wins'] > ch_t['wins']:
        print(f"🏆 CHAMPION")
    elif ch_t['wins'] > c_t['wins']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'Losses':<35} {c_t['losses']:<35} {ch_t['losses']:<35}", end='')
    if c_t['losses'] < ch_t['losses']:
        print(f"🏆 CHAMPION")
    elif ch_t['losses'] < c_t['losses']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'Win Rate %':<35} {c_t['wr']:<34.1f}% {ch_t['wr']:<34.1f}%", end='')
    if c_t['wr'] > ch_t['wr']:
        print(f"🏆 CHAMPION")
    elif ch_t['wr'] > c_t['wr']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'Total P&L':<35} ${c_t['total_pnl']:<33.2f} ${ch_t['total_pnl']:<33.2f}", end='')
    if c_t['total_pnl'] > ch_t['total_pnl']:
        print(f"🏆 CHAMPION")
    elif ch_t['total_pnl'] > c_t['total_pnl']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    print(f"{'Avg P&L/Signal':<35} ${c_t['avg_pnl']:<33.2f} ${ch_t['avg_pnl']:<33.2f}", end='')
    if c_t['avg_pnl'] > ch_t['avg_pnl']:
        print(f"🏆 CHAMPION")
    elif ch_t['avg_pnl'] > c_t['avg_pnl']:
        print(f"🎯 CHALLENGER")
    else:
        print("TIE")
    
    # SECTION 2: FULL PICTURE
    print(f"\n{'SECTION 2: FULL P&L PICTURE (All Exit Types Combined)':<120}")
    print("-"*120)
    print(f"{'Metric':<35} {'CHAMPION':<35} {'CHALLENGER':<35} {'Winner':<15}")
    print("-"*120)
    
    c_tp = len([s for s in champion_all['TP_HIT'] if s in c_all])
    c_sl = len([s for s in champion_all['SL_HIT'] if s in c_all])
    c_open = len([s for s in champion_all['OPEN'] if s in c_all])
    
    ch_tp = len([s for s in challenger_all['TP_HIT'] if s in ch_all])
    ch_sl = len([s for s in challenger_all['SL_HIT'] if s in ch_all])
    ch_open = len([s for s in challenger_all['OPEN'] if s in ch_all])
    
    print(f"{'Total Signals':<35} {c_m['count']:<35} {ch_m['count']:<35}")
    print(f"\n{'Exit Type Breakdown':<35}")
    print(f"  {'TIMEOUT':<33} {c_t['count']:<35} {ch_t['count']:<35}")
    print(f"  {'TP_HIT':<33} {c_tp:<35} {ch_tp:<35}")
    print(f"  {'SL_HIT':<33} {c_sl:<35} {ch_sl:<35}")
    print(f"  {'OPEN':<33} {c_open:<35} {ch_open:<35}")
    
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
    return True

def main():
    champion_all, challenger_all = load_all_signals()
    
    # Determine stage
    if len(sys.argv) > 1:
        if sys.argv[1] == '--once':
            stages = [10, 20, 30, 40, 50]
        else:
            try:
                stages = [int(sys.argv[1])]
            except:
                stages = [10, 20, 30, 40, 50]
    else:
        stages = [10, 20, 30, 40, 50]
    
    # Show available stages
    c_timeout_count = len(champion_all['TIMEOUT'])
    ch_timeout_count = len(challenger_all['TIMEOUT'])
    
    print(f"\n⚔️  A/B TEST - STAGED REVIEW (Full P&L Picture)")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"\nCurrent TIMEOUT Collection:")
    print(f"  Champion: {c_timeout_count} signals")
    print(f"  Challenger: {ch_timeout_count} signals")
    
    for stage in stages:
        print_stage(stage, champion_all, challenger_all)

if __name__ == '__main__':
    if '--once' in sys.argv:
        main()
    else:
        try:
            while True:
                clear_screen()
                main()
                time.sleep(5)
        except KeyboardInterrupt:
            clear_screen()
            print("✅ Stopped\n")
