#!/usr/bin/env python3
"""
A/B Test Cutoff Monitor: TIMEOUT Signal Count Tracking

Monitors progress toward 50 CHAMPION vs 50 CHALLENGER TIMEOUT signals cutoff.
When both reach 50, final comparison will be run.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

workspace = "/Users/geniustarigan/.openclaw/workspace"
master_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")

def load_signals():
    """Load all signals grouped by champion/challenger and status"""
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

def get_time_format(dt_iso):
    """Parse ISO timestamp and return formatted string"""
    try:
        dt = datetime.fromisoformat(dt_iso.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return dt_iso[:19]

def main():
    champion_timeout, challenger_timeout = load_signals()
    
    c_count = len(champion_timeout)
    ch_count = len(challenger_timeout)
    
    cutoff_target = 50
    c_remaining = max(0, cutoff_target - c_count)
    ch_remaining = max(0, cutoff_target - ch_count)
    
    print("\n" + "="*100)
    print("⚔️  A/B TEST CUTOFF MONITOR - TIMEOUT SIGNALS")
    print("="*100)
    
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7)))
    print(f"\nGenerated: {now.strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    
    print(f"\nCutoff Target: {cutoff_target} TIMEOUT signals per group")
    
    print(f"\n{'Group':<15} {'Count':<10} {'Remaining':<12} {'% Complete':<15} {'Status':<20}")
    print("-"*80)
    
    c_pct = (c_count / cutoff_target * 100) if cutoff_target > 0 else 0
    ch_pct = (ch_count / cutoff_target * 100) if cutoff_target > 0 else 0
    
    c_status = "✅ REACHED CUTOFF" if c_count >= cutoff_target else "🔄 Accumulating"
    ch_status = "✅ REACHED CUTOFF" if ch_count >= cutoff_target else "🔄 Accumulating"
    
    print(f"{'CHAMPION':<15} {c_count:<10} {c_remaining:<12} {c_pct:<14.1f}% {c_status:<20}")
    print(f"{'CHALLENGER':<15} {ch_count:<10} {ch_remaining:<12} {ch_pct:<14.1f}% {ch_status:<20}")
    
    print("\n" + "="*100)
    
    if c_count >= cutoff_target and ch_count >= cutoff_target:
        print("\n🎉 CUTOFF REACHED - Both groups have 50+ TIMEOUT signals!")
        print("\nREADY FOR FINAL COMPARISON:")
        print("  Run: python3 ab_test_final_comparison.py")
        print("\nThis will compare P&L, Win Rate, and determine the winner.")
        return
    
    # Estimate ETA
    if c_count > 0 or ch_count > 0:
        total_so_far = c_count + ch_count
        avg_per_group = total_so_far / 2 if total_so_far > 0 else 1
        
        print(f"\n📊 PROGRESS:")
        print(f"  Total TIMEOUT signals collected: {total_so_far} / 100")
        print(f"  Average per group: {avg_per_group:.1f}")
        
        # Try to estimate daily rate
        # Get oldest and newest TIMEOUT signals
        if champion_timeout or challenger_timeout:
            all_timeout = champion_timeout + challenger_timeout
            oldest = min(all_timeout, key=lambda s: s.get('fired_time_utc', ''))
            newest = max(all_timeout, key=lambda s: s.get('fired_time_utc', ''))
            
            oldest_time = oldest.get('fired_time_utc', '')
            newest_time = newest.get('fired_time_utc', '')
            
            if oldest_time and newest_time:
                try:
                    t_oldest = datetime.fromisoformat(oldest_time.replace('Z', '+00:00'))
                    t_newest = datetime.fromisoformat(newest_time.replace('Z', '+00:00'))
                    
                    span_days = (t_newest - t_oldest).total_seconds() / 86400
                    if span_days > 0:
                        daily_rate = total_so_far / span_days
                        
                        remaining_signals = max(0, cutoff_target - c_count) + max(0, cutoff_target - ch_count)
                        eta_days = remaining_signals / daily_rate if daily_rate > 0 else 0
                        
                        print(f"  Collection period: {span_days:.1f} days")
                        print(f"  Average daily rate: {daily_rate:.1f} TIMEOUT signals/day")
                        print(f"\n⏳ ESTIMATED TIME TO CUTOFF: {eta_days:.1f} days")
                        
                        if eta_days > 0:
                            eta_date = now + timedelta(days=eta_days)
                            print(f"   Target date: {eta_date.strftime('%Y-%m-%d %H:%M GMT+7')}")
                except:
                    pass
    
    print("\n" + "="*100)

if __name__ == '__main__':
    main()
