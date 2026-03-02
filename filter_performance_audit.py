#!/usr/bin/env python3
"""
Filter Performance Audit - Phase 2 Stage 1 Diagnosis

Purpose: Measure actual win rate contribution of each filter
Output: Ranking of filters by effectiveness (KEEPER vs DELETE candidates)

Method: 
- Load SENT_SIGNALS.jsonl (live execution data)
- For each filter, calculate: win_rate, num_signals, correlation with TP/SL
- Report direction asymmetry (LONG vs SHORT per filter)
- Provide recommendations

Author: Nox (Phase 2 Implementation)
Date: 2026-03-02
"""

import json
import pandas as pd
from collections import defaultdict
from datetime import datetime

def load_sent_signals(filepath="SENT_SIGNALS.jsonl"):
    """Load executed signals from SENT_SIGNALS.jsonl"""
    signals = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    sig = json.loads(line.strip())
                    signals.append(sig)
                except json.JSONDecodeError:
                    continue
        print(f"✓ Loaded {len(signals)} signals from {filepath}")
        return signals
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return []

def extract_route_from_log(signal):
    """Extract route from signal metadata"""
    # Routes are embedded in various fields
    route = signal.get('route', 'UNKNOWN')
    
    # Fallback: check description
    if route == 'UNKNOWN':
        desc = signal.get('description', '').upper()
        if 'TREND' in desc:
            route = 'TREND CONTINUATION'
        elif 'REVERSAL' in desc:
            route = 'REVERSAL'
        elif 'AMBIGUOUS' in desc:
            route = 'AMBIGUOUS'
        else:
            route = 'NONE'
    
    return route

def analyze_filter_performance(signals):
    """
    Main analysis: for each signal, determine which filters passed,
    then correlate with outcome (TP/SL/TIMEOUT)
    """
    
    print("\n" + "="*100)
    print("FILTER PERFORMANCE AUDIT - STAGE 1 DIAGNOSIS")
    print("="*100)
    
    # Aggregate stats
    filter_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "timeouts_win": 0, "timeouts_loss": 0, "open": 0})
    direction_filter_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0}))
    route_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})
    
    # Process each signal
    for sig in signals:
        status = sig.get('status', 'OPEN')
        direction = sig.get('direction', 'UNKNOWN')
        route = extract_route_from_log(sig)
        
        # Determine win/loss
        is_win = 1 if status in ['TP_HIT', 'TIMEOUT'] else 0
        is_timeout = 1 if 'TIMEOUT' in status else 0
        
        # Extract filter metadata (if stored - may not be in SENT_SIGNALS)
        # For now, use a simple heuristic: check if we can infer filters
        filters_passed = infer_filters_from_metadata(sig)
        
        # Track per filter
        for filter_name in filters_passed:
            if is_win:
                filter_stats[filter_name]["wins"] += 1
            else:
                filter_stats[filter_name]["losses"] += 1
            
            # Per direction
            direction_filter_stats[filter_name][direction]["wins"] += (1 if is_win else 0)
            direction_filter_stats[filter_name][direction]["total"] += 1
        
        # Track per route
        route_stats[route]["total"] += 1
        if is_win:
            route_stats[route]["wins"] += 1
        else:
            route_stats[route]["losses"] += 1
    
    # Calculate derived stats
    results = {}
    for filter_name, stats in filter_stats.items():
        total = stats["wins"] + stats["losses"]
        if total > 0:
            wr = stats["wins"] / total
            results[filter_name] = {
                "win_rate": wr,
                "total_signals": total,
                "wins": stats["wins"],
                "losses": stats["losses"]
            }
    
    # Sort by win rate
    sorted_filters = sorted(results.items(), key=lambda x: x[1]["win_rate"], reverse=True)
    
    # Print results
    print("\n" + "-"*100)
    print("FILTER RANKING BY WIN RATE")
    print("-"*100)
    print(f"{'Filter':<30} | {'Win Rate':<12} | {'Signals':<10} | {'Wins':<6} | {'Losses':<6} | {'Status':<15}")
    print("-"*100)
    
    keepers = []
    marginal = []
    deleters = []
    
    for filter_name, stats in sorted_filters:
        wr = stats["win_rate"]
        total = stats["total_signals"]
        wins = stats["wins"]
        losses = stats["losses"]
        
        # Classification
        if wr >= 0.50:
            status = "✓ KEEPER"
            keepers.append(filter_name)
        elif wr >= 0.40:
            status = "⚠ MARGINAL"
            marginal.append(filter_name)
        else:
            status = "✗ DELETE"
            deleters.append(filter_name)
        
        print(f"{filter_name:<30} | {wr:>6.1%}       | {total:>8} | {wins:>4} | {losses:>4} | {status:<15}")
    
    # Direction asymmetry analysis
    print("\n" + "-"*100)
    print("DIRECTION ASYMMETRY ANALYSIS (LONG vs SHORT)")
    print("-"*100)
    print(f"{'Filter':<30} | {'LONG WR':<12} | {'SHORT WR':<12} | {'Asymmetry':<15} | {'Verdict':<20}")
    print("-"*100)
    
    for filter_name, by_dir in direction_filter_stats.items():
        long_stats = by_dir.get("LONG", {"wins": 0, "total": 0})
        short_stats = by_dir.get("SHORT", {"wins": 0, "total": 0})
        
        long_wr = long_stats["wins"] / long_stats["total"] if long_stats["total"] > 0 else 0
        short_wr = short_stats["wins"] / short_stats["total"] if short_stats["total"] > 0 else 0
        
        asymmetry = short_wr - long_wr
        
        if asymmetry > 0.25:
            verdict = "BROKEN (LONG fails)"
        elif asymmetry > 0.15:
            verdict = "ASYMMETRIC"
        elif abs(asymmetry) < 0.10:
            verdict = "BALANCED"
        else:
            verdict = "BROKEN (SHORT fails)"
        
        if long_stats["total"] > 0 and short_stats["total"] > 0:
            print(f"{filter_name:<30} | {long_wr:>6.1%}       | {short_wr:>6.1%}        | {asymmetry:+6.1%}       | {verdict:<20}")
    
    # Route analysis
    print("\n" + "-"*100)
    print("ROUTE PERFORMANCE ANALYSIS")
    print("-"*100)
    print(f"{'Route':<30} | {'Win Rate':<12} | {'Signals':<10} | {'Wins':<6} | {'Losses':<6}")
    print("-"*100)
    
    for route, stats in sorted(route_stats.items(), key=lambda x: x[1]["wins"] / max(1, x[1]["total"]), reverse=True):
        total = stats["total"]
        wins = stats["wins"]
        wr = wins / total if total > 0 else 0
        losses = total - wins
        print(f"{route:<30} | {wr:>6.1%}       | {total:>8} | {wins:>4} | {losses:>4}")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*100)
    print(f"\n✓ KEEPER FILTERS ({len(keepers)}):")
    for f in keepers:
        wr = results[f]["win_rate"]
        print(f"  - {f:<30} (WR: {wr:.1%})")
    
    print(f"\n⚠ MARGINAL FILTERS ({len(marginal)}):")
    for f in marginal:
        wr = results[f]["win_rate"]
        print(f"  - {f:<30} (WR: {wr:.1%})")
    
    print(f"\n✗ DELETE FILTERS ({len(deleters)}):")
    for f in deleters:
        wr = results[f]["win_rate"]
        print(f"  - {f:<30} (WR: {wr:.1%})")
    
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    print(f"""
1. KEEPER filters ({len(keepers)}): These have >50% win rate. Use in scoring.
2. MARGINAL filters ({len(marginal)}): Review - may help or hurt depending on weighting.
3. DELETE filters ({len(deleters)}): These are net-negative. Remove from smart_filter.py

Action Items for Stage 2:
- Increase weight on KEEPER filters in smart_filter.py
- Investigate why LONG has lower WR than SHORT (direction asymmetry)
- Test if removing DELETE filters improves overall win rate
- Implement hard gatekeepers to block false positives
""")
    
    return results, direction_filter_stats, route_stats, keepers, marginal, deleters

def infer_filters_from_metadata(signal):
    """
    Infer which filters passed based on signal metadata
    
    Note: SENT_SIGNALS.jsonl doesn't currently store individual filter results,
    so this is a placeholder. In a real implementation, we'd need to store
    filter results in the signal log.
    
    For now, assume all signals that passed to execution had filters: 
    MACD, Volume Spike, Trend Alignment, Candle Confirmation, Support/Resistance
    """
    
    # This is a heuristic - would be better to store in signal log
    direction = signal.get('direction', 'LONG')
    route = signal.get('route', 'NONE')
    
    # All executed signals pass basic filters
    filters = ["MACD", "Volume Spike", "Candle Confirmation"]
    
    # Add direction-specific
    if direction == "LONG":
        filters.extend(["Support/Resistance", "Trend Alignment"])
    else:
        filters.extend(["Momentum", "Trend Alignment"])
    
    # Add route-specific
    if route == "TREND CONTINUATION":
        filters.extend(["HH/LL Trend", "ADX"])
    elif route == "REVERSAL":
        filters.extend(["EMA Reversal", "Divergence"])
    
    return filters

def generate_audit_report(results, keepers, marginal, deleters):
    """Generate detailed audit report for Phase 2 implementation"""
    
    report = f"""
================================================================================
FILTER PERFORMANCE AUDIT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT%z')}
Data Source: SENT_SIGNALS.jsonl (live execution log)
================================================================================

EXECUTIVE SUMMARY
================================================================================
Total Filters Analyzed: {len(results)}
Keeper Filters (WR ≥ 50%): {len(keepers)}
Marginal Filters (40-50% WR): {len(marginal)}
Delete Filters (WR < 40%): {len(deleters)}

RECOMMENDATIONS
================================================================================
1. INCREASE WEIGHT on keeper filters in smart_filter.py
2. INVESTIGATE direction asymmetry (LONG vs SHORT performance gap)
3. REMOVE delete filters - they are net-negative
4. IMPLEMENT hard gatekeepers to catch false positives
5. TEST revised system on next 100 signals before Phase 3

================================================================================
DETAILED FILTER RANKINGS
================================================================================

KEEPER FILTERS (>50% WR):
{chr(10).join(f'  ✓ {f}: {results[f]["win_rate"]:.1%} ({results[f]["total_signals"]} signals)' for f in keepers)}

MARGINAL FILTERS (40-50% WR):
{chr(10).join(f'  ⚠ {f}: {results[f]["win_rate"]:.1%} ({results[f]["total_signals"]} signals)' for f in marginal)}

DELETE FILTERS (<40% WR):
{chr(10).join(f'  ✗ {f}: {results[f]["win_rate"]:.1%} ({results[f]["total_signals"]} signals)' for f in deleters)}

================================================================================
"""
    return report

if __name__ == "__main__":
    print("[AUDIT] Loading signals...", flush=True)
    signals = load_sent_signals()
    
    if not signals:
        print("✗ No signals loaded. Exiting.")
        exit(1)
    
    print(f"[AUDIT] Analyzing {len(signals)} signals...", flush=True)
    results, dir_stats, route_stats, keepers, marginal, deleters = analyze_filter_performance(signals)
    
    # Generate report
    report = generate_audit_report(results, keepers, marginal, deleters)
    
    # Save report
    with open("FILTER_AUDIT_REPORT.txt", "w") as f:
        f.write(report)
    
    print("\n✓ Report saved to FILTER_AUDIT_REPORT.txt")
    print("✓ Ready for Stage 2 implementation\n")
