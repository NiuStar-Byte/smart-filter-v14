#!/usr/bin/env python3
"""
TIER ASSIGNMENT DIAGNOSTIC
==========================
Shows:
1. What pattern in SIGNAL_TIERS.json is assigning tier
2. What signals are being fired with that tier
3. What their actual performance is
4. Do they meet threshold criteria?
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
SIGNALS_FILE = os.path.join(WORKSPACE, "SIGNALS_MASTER.jsonl")
TIERS_FILE = os.path.join(WORKSPACE, "SIGNAL_TIERS.json")

TIER_THRESHOLDS = {
    1: {"wr": 0.60, "pnl": 5.50, "trades": 60},
    2: {"wr": 0.50, "pnl": 3.50, "trades": 50},
    3: {"wr": 0.40, "pnl": 2.00, "trades": 40},
}


def load_tier_patterns():
    """Load tier patterns from SIGNAL_TIERS.json"""
    if not os.path.exists(TIERS_FILE):
        return {}, {}
    
    try:
        with open(TIERS_FILE, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list) and data:
            latest = data[-1]
            
            tier_patterns = {
                1: latest.get('tier1', []),
                2: latest.get('tier2', []),
                3: latest.get('tier3', []),
            }
            
            # Build reverse map: pattern → tier
            pattern_to_tier = {}
            for tier_num, patterns in tier_patterns.items():
                for pattern in patterns:
                    pattern_to_tier[pattern] = tier_num
            
            return tier_patterns, pattern_to_tier
    except Exception as e:
        print(f"Error loading patterns: {e}")
    
    return {}, {}


def load_signals():
    signals = []
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        signals.append(json.loads(line))
                    except:
                        pass
    return signals


def build_full_combo_key(signal):
    """Build 6D combo key from signal"""
    tf = signal.get('timeframe', '') or ''
    direction = signal.get('direction', '') or ''
    route = signal.get('route', '') or ''
    regime = signal.get('regime', '') or ''
    symbol_group = signal.get('symbol_group', '') or ''
    
    parts = [tf, direction, route, regime, symbol_group]
    key = '|'.join(filter(None, parts))
    return key if key else None


def generate_diagnostic():
    print("\n" + "="*140)
    print("🔍 TIER ASSIGNMENT DIAGNOSTIC")
    print("="*140)
    
    tier_patterns, pattern_to_tier = load_tier_patterns()
    signals = load_signals()
    
    # Group by tier assignment
    tier_assignments = {1: [], 2: [], 3: []}
    
    for signal in signals:
        tier = signal.get('tier')
        
        if tier is None or 'X' in str(tier).upper():
            continue
        
        # Extract tier number
        tier_num = None
        if '1' in str(tier):
            tier_num = 1
        elif '2' in str(tier):
            tier_num = 2
        elif '3' in str(tier):
            tier_num = 3
        else:
            continue
        
        tier_assignments[tier_num].append(signal)
    
    # Analyze each tier
    for tier_num in [2, 1, 3]:  # Start with Tier-2 as you mentioned
        signals_with_tier = tier_assignments[tier_num]
        
        if not signals_with_tier:
            continue
        
        print(f"\n{'='*140}")
        print(f"🥇 TIER-{tier_num} ANALYSIS ({len(signals_with_tier)} signals assigned)")
        print(f"{'='*140}")
        
        # Group by 6D combo key
        combos_by_key = defaultdict(lambda: {'total': 0, 'open': 0, 'closed': 0, 'tp': 0, 'sl': 0, 'pnl': 0.0})
        
        for signal in signals_with_tier:
            combo_key = build_full_combo_key(signal)
            if not combo_key:
                combo_key = f"INCOMPLETE: {signal.get('timeframe')}|{signal.get('route')}"
            
            status = signal.get('status', 'OPEN')
            pnl = signal.get('pnl_usd', 0.0) or 0.0
            
            combos_by_key[combo_key]['total'] += 1
            
            if status == 'OPEN':
                combos_by_key[combo_key]['open'] += 1
            else:
                combos_by_key[combo_key]['closed'] += 1
                if status == 'TP_HIT':
                    combos_by_key[combo_key]['tp'] += 1
                    combos_by_key[combo_key]['pnl'] += pnl
                elif status == 'SL_HIT':
                    combos_by_key[combo_key]['pnl'] += pnl
        
        # Show combos with results
        combos_with_results = [(k, v) for k, v in combos_by_key.items() if v['closed'] > 0]
        
        if combos_with_results:
            print(f"\nCOMBOS WITH CLOSED RESULTS:")
            print(f"{'Combo (Full 6D)':<80} {'Fired':>8} {'Open':>8} {'Closed':>8} {'WR':>8} {'Avg P&L':>12} {'Qualifies?':>15}")
            print("─"*140)
            
            for combo_key, stats in sorted(combos_with_results, key=lambda x: -x[1]['closed']):
                if stats['closed'] > 0:
                    wr = (stats['tp'] / stats['closed']) * 100
                    avg_pnl = stats['pnl'] / stats['closed']
                    
                    # Check threshold
                    threshold = TIER_THRESHOLDS[tier_num]
                    qualifies = (wr >= threshold['wr']*100 and 
                                avg_pnl >= threshold['pnl'] and 
                                stats['closed'] >= threshold['trades'])
                    
                    qual_str = "✅ YES" if qualifies else f"❌ NO (WR:{wr:.1f}%, Avg:${avg_pnl:.2f}, Trades:{stats['closed']})"
                    
                    print(f"{combo_key:<80} {stats['total']:>8} {stats['open']:>8} {stats['closed']:>8} {wr:>7.1f}% ${avg_pnl:>11.2f} {qual_str:>15}")
        else:
            print(f"\nNo closed results yet for Tier-{tier_num} combos (all signals still OPEN)")
            
            # Show incomplete combos
            incomplete_combos = [(k, v) for k, v in combos_by_key.items() if v['total'] > 0]
            if incomplete_combos:
                print(f"\nOPEN SIGNALS BY COMBO:")
                print(f"{'Combo Key':<80} {'Fired':>8} {'Open':>8}")
                print("─"*140)
                for combo_key, stats in sorted(incomplete_combos, key=lambda x: -x[1]['total']):
                    print(f"{combo_key:<80} {stats['total']:>8} {stats['open']:>8}")
        
        # Show threshold requirements
        threshold = TIER_THRESHOLDS[tier_num]
        print(f"\n📋 TIER-{tier_num} THRESHOLD REQUIREMENTS:")
        print(f"  • Win Rate: ≥ {threshold['wr']*100:.0f}%")
        print(f"  • Avg P&L: ≥ ${threshold['pnl']:.2f}")
        print(f"  • Min Trades: ≥ {threshold['trades']}")


if __name__ == "__main__":
    generate_diagnostic()
