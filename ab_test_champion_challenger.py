#!/usr/bin/env python3
"""
CHAMPION vs CHALLENGER: A/B Test Reporter

Champion: Current strategy (all signals, no tier distinction)
  TF15min: 3h 45m | TF30min: 5h | TF1h: 5h

Challenger: Tier-based timeout windows (SILENT test)
  TIER-1 (60%+ WR): TF15min: 4h | TF30min: 5h | TF1h: 6h
  TIER-2 (40-60%): TF15min: 3h | TF30min: 4h | TF1h: 5h
  TIER-3 (<40%): TF15min: 2h | TF30min: 3h | TF1h: 4h

Collects TIMEOUT signals and compares performance by tier.
"""

import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os

workspace = "/Users/geniustarigan/.openclaw/workspace"
master_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")

def load_timeout_signals():
    """Load TIMEOUT signals with Champion/Challenger tags"""
    champion = {'15min': [], '30min': [], '1h': []}
    challenger = {'15min': [], '30min': [], '1h': []}
    
    if not os.path.exists(master_file):
        print(f"[ERROR] {master_file} not found")
        return champion, challenger
    
    try:
        with open(master_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    
                    # Only TIMEOUT (not STALE_TIMEOUT)
                    if sig.get('status') != 'TIMEOUT':
                        continue
                    
                    # Skip stale
                    if sig.get('data_quality_flag') and 'STALE_TIMEOUT' in sig.get('data_quality_flag'):
                        continue
                    
                    tf = sig.get('timeframe', 'UNKNOWN')
                    cc_group = sig.get('champion_challenger_group', 'UNKNOWN')
                    
                    if cc_group == 'CHAMPION':
                        champion[tf].append(sig)
                    elif cc_group == 'CHALLENGER':
                        challenger[tf].append(sig)
        
        return champion, challenger
    
    except Exception as e:
        print(f"[ERROR] Loading signals: {e}")
        return champion, challenger

def analyze_group(signals_list, group_name, timeframe):
    """Analyze one group of signals"""
    if not signals_list:
        return None
    
    stats = {
        'count': len(signals_list),
        'wins': 0,
        'losses': 0,
        'total_pnl': 0.0,
        'signals': signals_list
    }
    
    for sig in signals_list:
        pnl = sig.get('pnl_usd', 0)
        stats['total_pnl'] += float(pnl)
        
        if pnl > 0:
            stats['wins'] += 1
        elif pnl < 0:
            stats['losses'] += 1
    
    total_closed = stats['wins'] + stats['losses']
    if total_closed > 0:
        stats['wr_pct'] = (stats['wins'] / total_closed) * 100
        stats['avg_pnl'] = stats['total_pnl'] / stats['count']
    else:
        stats['wr_pct'] = 0
        stats['avg_pnl'] = 0
    
    return stats

def print_tier_analysis(champion_by_tier, challenger_by_tier):
    """Print results by tier"""
    print("\n" + "="*180)
    print("🏆 CHAMPION vs 🎯 CHALLENGER - TIER-BY-TIER ANALYSIS")
    print("="*180)
    
    for tier in ['TIER-1', 'TIER-2', 'TIER-3']:
        print(f"\n{tier.upper()} TIMEOUT SIGNALS:")
        print("-"*180)
        print(f"{'TF':<8} | {'Group':<12} | {'Count':<6} | {'Wins':<6} | {'Loss':<6} | {'WR%':<8} | {'Total P&L':<12} | {'Avg P&L':<12} | {'Advantage':<12}")
        print("-"*180)
        
        for tf in ['15min', '30min', '1h']:
            champ_signals = [s for s in champion_by_tier.get(tier, []) if s.get('timeframe') == tf]
            chall_signals = [s for s in challenger_by_tier.get(tier, []) if s.get('timeframe') == tf]
            
            champ_stats = analyze_group(champ_signals, 'CHAMPION', tf)
            chall_stats = analyze_group(chall_signals, 'CHALLENGER', tf)
            
            if champ_stats:
                print(f"{tf:<8} | {'CHAMPION':<12} | {champ_stats['count']:<6} | {champ_stats['wins']:<6} | {champ_stats['losses']:<6} | {champ_stats['wr_pct']:>6.1f}% | ${champ_stats['total_pnl']:>+10.2f} | ${champ_stats['avg_pnl']:>+10.2f} |")
            
            if chall_stats:
                # Calculate advantage
                wr_diff = chall_stats['wr_pct'] - (champ_stats['wr_pct'] if champ_stats else 0)
                pnl_diff = chall_stats['total_pnl'] - (champ_stats['total_pnl'] if champ_stats else 0)
                
                advantage = ""
                if wr_diff > 5:
                    advantage = f"✓ +{wr_diff:.1f}%"
                elif wr_diff < -5:
                    advantage = f"✗ {wr_diff:.1f}%"
                else:
                    advantage = f"~ {wr_diff:+.1f}%"
                
                print(f"{tf:<8} | {'CHALLENGER':<12} | {chall_stats['count']:<6} | {chall_stats['wins']:<6} | {chall_stats['losses']:<6} | {chall_stats['wr_pct']:>6.1f}% | ${chall_stats['total_pnl']:>+10.2f} | ${chall_stats['avg_pnl']:>+10.2f} | {advantage:<12}")
                print()

def print_overall_summary(champion, challenger):
    """Print overall summary"""
    print("\n" + "="*180)
    print("📊 OVERALL SUMMARY")
    print("="*180)
    
    for tf in ['15min', '30min', '1h']:
        print(f"\n{tf.upper()}:")
        print("-"*120)
        
        champ_signals = champion[tf]
        chall_signals = challenger[tf]
        
        champ_stats = analyze_group(champ_signals, 'CHAMPION', tf)
        chall_stats = analyze_group(chall_signals, 'CHALLENGER', tf)
        
        if champ_stats and chall_stats:
            wr_diff = chall_stats['wr_pct'] - champ_stats['wr_pct']
            pnl_diff = chall_stats['total_pnl'] - champ_stats['total_pnl']
            
            winner = "🎯 CHALLENGER" if pnl_diff > 0 else "🏆 CHAMPION"
            
            print(f"Champion: {champ_stats['count']} signals, WR={champ_stats['wr_pct']:.1f}%, P&L=${champ_stats['total_pnl']:+.2f} (Avg: ${champ_stats['avg_pnl']:+.2f}/signal)")
            print(f"Challenger: {chall_stats['count']} signals, WR={chall_stats['wr_pct']:.1f}%, P&L=${chall_stats['total_pnl']:+.2f} (Avg: ${chall_stats['avg_pnl']:+.2f}/signal)")
            print(f"\nResult: {winner}")
            print(f"  WR Improvement: {wr_diff:+.1f}%")
            print(f"  P&L Improvement: ${pnl_diff:+.2f}")
            print(f"  Per-Signal Improvement: ${(pnl_diff / chall_stats['count']):+.2f}")

def main():
    print("\n" + "="*180)
    print("🏆 CHAMPION vs 🎯 CHALLENGER - A/B TEST REPORT")
    print("="*180)
    print(f"Report Generated: {datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"Test Status: SILENT (Traders unaware of variation)")
    
    champion, challenger = load_timeout_signals()
    
    # Count total
    total_champ = sum(len(v) for v in champion.values())
    total_chall = sum(len(v) for v in challenger.values())
    
    print(f"Test Progress: {total_champ} CHAMPION signals | {total_chall} CHALLENGER signals")
    print(f"Target: 200+ per group per timeframe for statistical significance")
    
    if total_champ < 50 or total_chall < 50:
        print("\n⏳ Not enough data yet. Keep running test (need 200+ per group)")
        return
    
    # Analyze by tier
    champion_by_tier = defaultdict(list)
    challenger_by_tier = defaultdict(list)
    
    for tf, signals in champion.items():
        for sig in signals:
            tier = sig.get('tier', 'TIER-3')
            champion_by_tier[tier].append(sig)
    
    for tf, signals in challenger.items():
        for sig in signals:
            tier = sig.get('tier', 'TIER-3')
            challenger_by_tier[tier].append(sig)
    
    # Print analyses
    print_tier_analysis(champion_by_tier, challenger_by_tier)
    print_overall_summary(champion, challenger)
    
    # Recommendation
    print("\n" + "="*180)
    print("🎯 RECOMMENDATION")
    print("="*180)
    
    total_champ_pnl = sum(sum(s.get('pnl_usd', 0) for s in signals) for signals in champion.values())
    total_chall_pnl = sum(sum(s.get('pnl_usd', 0) for s in signals) for signals in challenger.values())
    
    if total_chall_pnl > total_champ_pnl:
        improvement = total_chall_pnl - total_champ_pnl
        pct_gain = (improvement / abs(total_champ_pnl) * 100) if total_champ_pnl != 0 else 0
        print(f"\n✅ CHALLENGER WINNING: +${improvement:.2f} ({pct_gain:+.1f}% improvement)")
        print(f"Recommendation: Deploy CHALLENGER as new standard after reaching {max(200, total_chall)} signals")
    else:
        print(f"\n⏳ Test Still Running")
        print(f"Continue monitoring. Need {max(200, total_champ + total_chall)} total signals for conclusive result")
    
    print("\n" + "="*180)

if __name__ == '__main__':
    main()
