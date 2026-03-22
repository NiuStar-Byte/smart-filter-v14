#!/usr/bin/env python3
"""
Filter Effectiveness Analyzer - Detailed
Analyzes per-filter correlation with win/loss outcomes on closed signals
Compares effectiveness baseline (all signals) vs per-filter performance

Usage:
  python3 filter_effectiveness_analyzer_detailed.py

Output:
  - Per-filter win rates and effectiveness scores
  - Baseline WR for reference
  - Ranking by effectiveness (highest → lowest)
"""

import json
from collections import defaultdict
from datetime import datetime
import sys

SIGNALS_MASTER = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"

def load_signals():
    """Load all signals from SIGNALS_MASTER.jsonl"""
    signals = []
    try:
        with open(SIGNALS_MASTER, 'r') as f:
            for line in f:
                try:
                    signal = json.loads(line.strip())
                    signals.append(signal)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"ERROR: Cannot find {SIGNALS_MASTER}")
        exit(1)
    return signals

def calculate_effectiveness(signals):
    """Calculate per-filter effectiveness on closed signals"""
    
    # Filter stats: {filter_name: {'passed': count, 'won': count, 'lost': count}}
    filter_stats = defaultdict(lambda: {'passed': 0, 'won': 0, 'lost': 0})
    
    total = 0
    instrumented = 0
    closed = 0
    
    for signal in signals:
        total += 1
        
        # Must have instrumentation
        if 'passed_filters' not in signal:
            continue
        
        instrumented += 1
        status = signal.get('status', 'UNKNOWN')
        
        # Only analyze CLOSED signals
        if status not in ['TP_HIT', 'SL_HIT']:
            continue
        
        closed += 1
        is_win = (status == 'TP_HIT')
        
        # Track each passed filter
        for filt in signal.get('passed_filters', []):
            filter_stats[filt]['passed'] += 1
            if is_win:
                filter_stats[filt]['won'] += 1
            else:
                filter_stats[filt]['lost'] += 1
    
    return filter_stats, total, instrumented, closed

def get_current_weights():
    """Load current filter weights from smart_filter.py"""
    weights = {}
    try:
        smart_filter_path = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/smart_filter.py"
        with open(smart_filter_path, 'r') as f:
            content = f.read()
            
        # Extract filter_weights_long dictionary
        import re
        match = re.search(r'self\.filter_weights_long = \{(.*?)\}', content, re.DOTALL)
        if match:
            # Parse each "Filter": value line
            lines = match.group(1).split('\n')
            for line in lines:
                if ':' in line:
                    parts = line.split(':')
                    filter_name = parts[0].strip().strip('"').strip("'")
                    try:
                        weight_str = parts[1].split('#')[0].strip().rstrip(',')
                        weight = float(weight_str)
                        weights[filter_name] = weight
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        print(f"Warning: Could not load weights: {e}")
    
    return weights

def main():
    print("=" * 100)
    print("FILTER EFFECTIVENESS ANALYZER - DETAILED")
    print("=" * 100)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"Source: {SIGNALS_MASTER}")
    print("")
    
    # Load signals and analyze
    signals = load_signals()
    filter_stats, total, instrumented, closed = calculate_effectiveness(signals)
    
    print(f"Dataset Summary:")
    print(f"  Total signals in MASTER: {total}")
    print(f"  Instrumented (with passed_filters): {instrumented}")
    print(f"  Closed (TP_HIT or SL_HIT): {closed}")
    print("")
    
    if closed == 0:
        print("No closed signals yet. Cannot calculate effectiveness.")
        return
    
    # Get current weights
    weights = get_current_weights()
    
    # Calculate baseline WR
    baseline_wr = closed / max(1, instrumented)
    
    print(f"Baseline WR (all closed signals): {baseline_wr * 100:.1f}%")
    print("")
    
    # All 20 filters (from smart_filter.py)
    all_filters = [
        "MACD", "Volume Spike", "Fractal Zone", "TREND", "Momentum",
        "ATR Momentum Burst", "MTF Volume Agreement", "HH/LL Trend", "Volatility Model",
        "Liquidity Awareness", "Volatility Squeeze", "Candle Confirmation",
        "VWAP Divergence", "Spread Filter", "Chop Zone", "Liquidity Pool",
        "Support/Resistance", "Smart Money Bias", "Absorption", "Wick Dominance"
    ]
    
    # Build effectiveness ranking for ALL filters
    effectiveness = []
    
    for filter_name in all_filters:
        stats = filter_stats.get(filter_name, {'passed': 0, 'won': 0, 'lost': 0})
        current_weight = weights.get(filter_name, 'N/A')
        
        if stats['passed'] == 0:
            effectiveness.append({
                'name': filter_name,
                'passed': 0,
                'won': 0,
                'lost': 0,
                'wr': None,
                'effectiveness': None,
                'weight': current_weight
            })
        else:
            filter_wr = stats['won'] / max(1, stats['passed'])
            effectiveness_pp = (filter_wr - baseline_wr) * 100
            
            effectiveness.append({
                'name': filter_name,
                'passed': stats['passed'],
                'won': stats['won'],
                'lost': stats['lost'],
                'wr': filter_wr,
                'effectiveness': effectiveness_pp,
                'weight': current_weight
            })
    
    # Sort by effectiveness (descending, never-passed last)
    effectiveness.sort(key=lambda x: (x['effectiveness'] is None, x['effectiveness'] if x['effectiveness'] is not None else -999), reverse=True)
    
    print("=" * 125)
    print("PER-FILTER EFFECTIVENESS RANKING - ALL 20 FILTERS (sorted by effectiveness)")
    print("=" * 125)
    print(f"Dataset: {closed} closed signals | {instrumented} instrumented | Baseline WR: {baseline_wr*100:.1f}%")
    print("=" * 125)
    print("")
    
    print(f"{'Rank':<5} {'Filter Name':<30} {'Passed':<8} {'Wins':<8} {'WR':<10} {'FA':<8} {'Effectiveness':<16} {'Weight':<8} {'Status':<10}")
    print("-" * 145)
    sys.stdout.flush()
    
    for idx, item in enumerate(effectiveness, 1):
        if item['passed'] == 0:
            wr_str = "N/A"
            eff_str = "N/A"
            status = "○ (0 pass)"
            fa_str = "0.00%"
        else:
            wr_str = f"{item['wr']*100:5.1f}%"
            eff_str = f"{item['effectiveness']:+6.1f}pp"
            fa_pct = (item['passed'] / max(1, closed)) * 100
            fa_str = f"{fa_pct:5.2f}%"
            if item['wr'] >= 0.70:
                status = "⭐ High"
            elif item['wr'] >= 0.50:
                status = "✓ Mid"
            else:
                status = "· Low"
        weight_str = f"{item['weight']}" if isinstance(item['weight'], (int, float)) else str(item['weight'])
        print(f"{idx:<5} {item['name']:<30} {item['passed']:<8} {item['won']:<8} {wr_str:<10} {fa_str:<8} {eff_str:<16} {weight_str:<8} {status:<10}")
    
    sys.stdout.flush()
    
    print("")
    print("=" * 125)
    print("CATEGORY BREAKDOWN BY PERFORMANCE TIER")
    print("=" * 125)
    
    # High performers (70%+)
    high_performers = [e for e in effectiveness if e['wr'] is not None and e['wr'] >= 0.70]
    if high_performers:
        print(f"\n⭐ HIGH PERFORMERS (70%+ WR): {len(high_performers)} filter(s)")
        for item in high_performers:
            fa_pct = (item['passed'] / max(1, closed)) * 100
            print(f"   • {item['name']:30s} | {item['passed']:3d} passed | {fa_pct:5.2f}% FA | {item['wr']*100:5.1f}% WR | {item['effectiveness']:+6.1f}pp | Weight: {item['weight']}")
    else:
        print(f"\n⭐ HIGH PERFORMERS (70%+ WR): None yet")
    
    # Mid performers (50-70%)
    mid_performers = [e for e in effectiveness if e['wr'] is not None and 0.50 <= e['wr'] < 0.70]
    if mid_performers:
        print(f"\n✓ MID PERFORMERS (50-70% WR): {len(mid_performers)} filter(s)")
        for item in mid_performers:
            fa_pct = (item['passed'] / max(1, closed)) * 100
            print(f"   • {item['name']:30s} | {item['passed']:3d} passed | {fa_pct:5.2f}% FA | {item['wr']*100:5.1f}% WR | {item['effectiveness']:+6.1f}pp | Weight: {item['weight']}")
    else:
        print(f"\n✓ MID PERFORMERS (50-70% WR): None yet")
    
    # Low performers (<50%)
    low_performers = [e for e in effectiveness if e['wr'] is not None and e['wr'] < 0.50]
    if low_performers:
        print(f"\n· LOW PERFORMERS (<50% WR): {len(low_performers)} filter(s)")
        for item in low_performers:
            fa_pct = (item['passed'] / max(1, closed)) * 100
            print(f"   • {item['name']:30s} | {item['passed']:3d} passed | {fa_pct:5.2f}% FA | {item['wr']*100:5.1f}% WR | {item['effectiveness']:+6.1f}pp | Weight: {item['weight']}")
    
    # Not yet triggered (0 passes)
    never_passed = [e for e in effectiveness if e['passed'] == 0]
    if never_passed:
        print(f"\n○ NOT YET TRIGGERED (0 passes): {len(never_passed)} filter(s)")
        for item in never_passed:
            print(f"   • {item['name']:30s} | 0 passed | 0.00% FA | Waiting for market conditions | Weight: {item['weight']}")
        print(f"   Note: These filters may activate as market conditions change. Monitor over time.")
    
    print("")
    print("=" * 100)
    print("METRIC DEFINITIONS")
    print("=" * 100)
    print("• WR (Win Rate) = Wins / Passed - Of signals where filter passed, how many won")
    print("• FA (Filter Availability) = Passed / Total Closed - In what % of signals does filter appear")
    print("• Effectiveness = Filter WR - Baseline WR - Shows correlation advantage vs baseline")
    print("• Example: VWAP Divergence | 1 passed | 2.22% FA | 100.0% WR | +64.6pp")
    print("   → Out of 45 closed signals, only 1 had VWAP pass (2.22% availability)")
    print("   → But that 1 signal won (100% WR)")
    print("   → +64.6pp = 100% WR vs 35.4% baseline")
    print("")
    print("• Sample size matters: <10 samples = high variance, unreliable")
    print("• Mixed-filter problem: Each signal has ~12 passed + ~8 failed filters")
    print("• Cannot isolate individual filter causation (correlation ≠ causation)")
    print("• Use this data to identify patterns, then validate in backtest")
    print("")

if __name__ == '__main__':
    main()
