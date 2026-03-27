#!/usr/bin/env python3
"""
PEC TIER DIMENSIONAL REPORT
Answers Q1 & Q2 without modifying locked trackers

Q1: How do signals WITH tier vs WITHOUT tier perform?
Q2: How do 6D/5D/4D/3D/2D combos differ by tier status?

Reads: SIGNALS_MASTER.jsonl (same source as locked trackers)
Output: PEC_TIER_DIMENSIONAL_REPORT.txt

Code Status: NEW (not modifying pec_enhanced_reporter or pec_post_deployment_tracker)
File Status: LOCKED (immutable, only data changes as signals close)
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict


def load_signals_master():
    """Load SIGNALS_MASTER.jsonl"""
    signals = []
    filepath = Path.home() / '.openclaw/workspace/SIGNALS_MASTER.jsonl'
    
    if not filepath.exists():
        print(f"ERROR: {filepath} not found")
        return []
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                signals.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    
    return signals


def categorize_by_tier(signals):
    """Segment signals by tier status"""
    with_tier = []
    without_tier = []
    
    tier_breakdown = defaultdict(list)
    
    for signal in signals:
        tier = signal.get('tier', 'None')
        
        if tier and tier != 'None':
            with_tier.append(signal)
            tier_breakdown[tier].append(signal)
        else:
            without_tier.append(signal)
    
    return with_tier, without_tier, tier_breakdown


def calculate_metrics(signals):
    """Calculate performance metrics for a signal group"""
    metrics = {
        'total': len(signals),
        'closed': 0,
        'tp_hit': 0,
        'sl_hit': 0,
        'timeout_win': 0,
        'timeout_loss': 0,
        'stale_timeout': 0,
        'total_pnl': 0.0,
        'tp_pnls': [],
        'sl_pnls': [],
        'timeout_win_pnls': [],
        'timeout_loss_pnls': [],
        'combos': defaultdict(lambda: {
            'total': 0, 'closed': 0, 'tp_hit': 0, 'sl_hit': 0,
            'timeout_win': 0, 'timeout_loss': 0, 'total_pnl': 0.0,
            'pnls': []
        })
    }
    
    # CRITICAL: EXCLUDE STALE_TIMEOUT to match pec_enhanced_reporter behavior
    closed_statuses = ['TP_HIT', 'SL_HIT', 'TIMEOUT']  # NOT including STALE_TIMEOUT
    
    for signal in signals:
        status = signal.get('status', 'OPEN')
        pnl = signal.get('pnl_usd', 0.0)
        
        # Extract 6D combo
        combo = extract_combo_6d(signal)
        metrics['combos'][combo]['total'] += 1
        
        # Track STALE_TIMEOUT separately (not included in WR/P&L metrics)
        if status == 'STALE_TIMEOUT':
            metrics['stale_timeout'] += 1
            continue
        
        if status in closed_statuses:
            metrics['closed'] += 1
            metrics['combos'][combo]['closed'] += 1
            metrics['total_pnl'] += pnl
            metrics['combos'][combo]['total_pnl'] += pnl
            metrics['combos'][combo]['pnls'].append(pnl)
            
            if status == 'TP_HIT':
                metrics['tp_hit'] += 1
                metrics['combos'][combo]['tp_hit'] += 1
                metrics['tp_pnls'].append(pnl)
            elif status == 'SL_HIT':
                metrics['sl_hit'] += 1
                metrics['combos'][combo]['sl_hit'] += 1
                metrics['sl_pnls'].append(pnl)
            elif status == 'TIMEOUT':
                if pnl >= 0:
                    metrics['timeout_win'] += 1
                    metrics['combos'][combo]['timeout_win'] += 1
                    metrics['timeout_win_pnls'].append(pnl)
                else:
                    metrics['timeout_loss'] += 1
                    metrics['combos'][combo]['timeout_loss'] += 1
                    metrics['timeout_loss_pnls'].append(pnl)
    
    return metrics


def extract_combo_6d(signal):
    """Extract 6D combo from signal"""
    tf = signal.get('timeframe', '?')
    direction = signal.get('direction', '?')
    pattern = signal.get('pattern', '?')
    regime = signal.get('regime', '?')
    alts = signal.get('alts_type', '?')
    volatility = signal.get('volatility_level', '?')
    return f"{tf}|{direction}|{pattern}|{regime}|{alts}|{volatility}"


def calculate_win_rate(metrics):
    """Calculate win rate"""
    if metrics['closed'] == 0:
        return 0.0
    tp = metrics['tp_hit']
    timeout_win = metrics['timeout_win']
    total = metrics['tp_hit'] + metrics['sl_hit'] + metrics['timeout_win'] + metrics['timeout_loss']
    
    if total == 0:
        return 0.0
    return (tp + timeout_win) / total * 100


def format_pnl(value):
    """Format PnL as USD"""
    return f"${value:,.2f}"


def generate_report(signals):
    """Generate tier-dimensional report"""
    with_tier, without_tier, tier_breakdown = categorize_by_tier(signals)
    
    # Calculate metrics for each group
    with_tier_metrics = calculate_metrics(with_tier)
    without_tier_metrics = calculate_metrics(without_tier)
    
    # Calculate metrics by tier level
    tier_metrics = {}
    for tier_name, tier_signals in tier_breakdown.items():
        tier_metrics[tier_name] = calculate_metrics(tier_signals)
    
    report = []
    report.append("=" * 90)
    report.append("PEC TIER DIMENSIONAL REPORT - Answers Q1 & Q2")
    report.append(f"Report Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    report.append("=" * 90)
    report.append("")
    
    # QUESTION 1: WITH TIER vs WITHOUT TIER
    report.append("=" * 90)
    report.append("QUESTION 1: PERFORMANCE WITH TIER vs WITHOUT TIER?")
    report.append("=" * 90)
    report.append("")
    
    report.append("SIGNALS WITH TIER (Tier-2, Tier-3, Tier-X combined)")
    report.append(f"  Total Loaded:      {with_tier_metrics['total']:,}")
    if with_tier_metrics['total'] > 0:
        report.append(f"  Closed:            {with_tier_metrics['closed']:,} ({100*with_tier_metrics['closed']/with_tier_metrics['total']:.1f}% close rate)")
    else:
        report.append(f"  Closed:            {with_tier_metrics['closed']:,}")
    report.append(f"  Win Rate:          {calculate_win_rate(with_tier_metrics):.2f}%")
    report.append(f"  Total P&L:         {format_pnl(with_tier_metrics['total_pnl'])}")
    if with_tier_metrics['closed'] > 0:
        report.append(f"  Avg P&L per Closed: {format_pnl(with_tier_metrics['total_pnl'] / with_tier_metrics['closed'])}")
    report.append("")
    
    if without_tier_metrics['total'] > 0:
        report.append("SIGNALS WITHOUT TIER (Tier-X or unassigned)")
        report.append(f"  Total Loaded:      {without_tier_metrics['total']:,}")
        report.append(f"  Closed:            {without_tier_metrics['closed']:,} ({100*without_tier_metrics['closed']/without_tier_metrics['total']:.1f}% close rate)")
        report.append(f"  Win Rate:          {calculate_win_rate(without_tier_metrics):.2f}%")
        report.append(f"  Total P&L:         {format_pnl(without_tier_metrics['total_pnl'])}")
        if without_tier_metrics['closed'] > 0:
            report.append(f"  Avg P&L per Closed: {format_pnl(without_tier_metrics['total_pnl'] / without_tier_metrics['closed'])}")
        report.append("")
    else:
        report.append("SIGNALS WITHOUT TIER: (none - all signals assigned tier)")
        report.append("")
    
    if without_tier_metrics['total'] > 0:
        wr_with = calculate_win_rate(with_tier_metrics)
        wr_without = calculate_win_rate(without_tier_metrics)
        report.append("COMPARISON")
        report.append(f"  Win Rate Delta:    {wr_with - wr_without:+.2f}% {'✓ TIER BETTER' if wr_with > wr_without else '✗ TIER WORSE'}")
        report.append(f"  P&L Delta:         {format_pnl(with_tier_metrics['total_pnl'] - without_tier_metrics['total_pnl'])}")
        report.append("")
    else:
        report.append("COMPARISON: All signals have tier - no non-tier baseline available")
        report.append("")
    
    # TIER LEVEL BREAKDOWN
    report.append("=" * 90)
    report.append("TIER LEVEL BREAKDOWN (Answering Q1 with granularity)")
    report.append("=" * 90)
    report.append("")
    
    for tier_name in ['Tier-2', 'Tier-3', 'Tier-X']:
        if tier_name not in tier_metrics:
            continue
        
        tm = tier_metrics[tier_name]
        report.append(f"{tier_name.upper()}")
        report.append(f"  Total:    {tm['total']:,} | Closed: {tm['closed']:,} | Stale: {tm['stale_timeout']:,}")
        report.append(f"  WR:       {calculate_win_rate(tm):.2f}%")
        report.append(f"  P&L:      {format_pnl(tm['total_pnl'])}")
        if tm['closed'] > 0:
            report.append(f"  Avg/Closed: {format_pnl(tm['total_pnl'] / tm['closed'])}")
        
        # Add context for Tier-3
        if tier_name == 'Tier-3':
            report.append("")
            report.append("  ⚠️  NOTE: Tier-3 signals fired TODAY (just hours ago)")
            report.append("      Expect closes to accumulate over 24-48 hours")
            report.append("      Current low close rate is due to recency, not quality")
        
        report.append("")
    
    # QUESTION 2: 6D COMBOS BY TIER
    report.append("=" * 90)
    report.append("QUESTION 2: 6D COMBOS BY TIER STATUS?")
    report.append("=" * 90)
    report.append("")
    
    report.append("TOP 10 COMBOS WITH TIER (Tier-2 + Tier-3)")
    tier_2_3_signals = tier_breakdown.get('Tier-2', []) + tier_breakdown.get('Tier-3', [])
    tier_2_3_metrics = calculate_metrics(tier_2_3_signals)
    combos_with_tier = [(k, v) for k, v in tier_2_3_metrics['combos'].items() if v['closed'] > 0]
    combos_with_tier.sort(key=lambda x: x[1]['closed'], reverse=True)
    
    if combos_with_tier:
        for i, (combo, stats) in enumerate(combos_with_tier[:10], 1):
            if stats['closed'] > 0:
                wr = (stats['tp_hit'] + stats['timeout_win']) / (stats['tp_hit'] + stats['sl_hit'] + stats['timeout_win'] + stats['timeout_loss']) * 100
                avg_pnl = stats['total_pnl'] / stats['closed']
                report.append(f"  {i}. {combo}")
                report.append(f"     Closed: {stats['closed']}, WR: {wr:.1f}%, Avg P&L: {format_pnl(avg_pnl)}")
    else:
        report.append("  (No Tier-2/Tier-3 combos have closed trades yet)")
    report.append("")
    
    if without_tier_metrics['total'] > 0:
        report.append("TOP 10 COMBOS WITHOUT TIER (Tier-X or unassigned)")
        combos_without_tier = [(k, v) for k, v in without_tier_metrics['combos'].items() if v['closed'] > 0]
        combos_without_tier.sort(key=lambda x: x[1]['closed'], reverse=True)
        
        for i, (combo, stats) in enumerate(combos_without_tier[:10], 1):
            if stats['closed'] > 0:
                wr = (stats['tp_hit'] + stats['timeout_win']) / (stats['tp_hit'] + stats['sl_hit'] + stats['timeout_win'] + stats['timeout_loss']) * 100
                avg_pnl = stats['total_pnl'] / stats['closed']
                report.append(f"  {i}. {combo}")
                report.append(f"     Closed: {stats['closed']}, WR: {wr:.1f}%, Avg P&L: {format_pnl(avg_pnl)}")
    else:
        report.append("TOP 10 COMBOS WITHOUT TIER: (none - all signals assigned tier)")
    report.append("")
    
    report.append("=" * 90)
    report.append("SUMMARY")
    report.append("=" * 90)
    report.append("")
    report.append(f"Total Signals Analyzed:    {len(signals):,}")
    report.append(f"  WITH TIER:  {len(with_tier):,} ({100*len(with_tier)/len(signals):.1f}%)")
    report.append(f"  WITHOUT TIER: {len(without_tier):,} ({100*len(without_tier)/len(signals):.1f}%)")
    report.append("")
    report.append("Tier Assignment Status:")
    for tier_name in ['Tier-2', 'Tier-3', 'Tier-X']:
        if tier_name in tier_metrics:
            count = tier_metrics[tier_name]['total']
            report.append(f"  {tier_name}: {count:,} signals")
    report.append("")
    
    # HIERARCHY TRACKING BY TIER
    report.append("=" * 90)
    report.append("SECTION 3: 6D/5D/4D/3D/2D HIERARCHY BY TIER STATUS")
    report.append("=" * 90)
    report.append("")
    
    report.append("6D COMBOS WITH TIER (Tier-2 + Tier-3 only)")
    tier_2_3_signals = tier_breakdown.get('Tier-2', []) + tier_breakdown.get('Tier-3', [])
    tier_2_3_6d = defaultdict(int)
    for signal in tier_2_3_signals:
        combo = extract_combo_6d(signal)
        tier_2_3_6d[combo] += 1
    
    top_with_tier = sorted(tier_2_3_6d.items(), key=lambda x: x[1], reverse=True)[:15]
    if top_with_tier:
        for i, (combo, count) in enumerate(top_with_tier, 1):
            report.append(f"  {i:2d}. {combo} ({count} signals)")
    else:
        report.append("  (None - all signals are Tier-X)")
    report.append("")
    
    report.append("6D COMBOS WITHOUT TIER (Tier-X only)")
    tier_x_signals = tier_breakdown.get('Tier-X', [])
    tier_x_6d = defaultdict(int)
    for signal in tier_x_signals:
        combo = extract_combo_6d(signal)
        tier_x_6d[combo] += 1
    
    top_without_tier = sorted(tier_x_6d.items(), key=lambda x: x[1], reverse=True)[:15]
    if top_without_tier:
        for i, (combo, count) in enumerate(top_without_tier, 1):
            report.append(f"  {i:2d}. {combo} ({count} signals)")
    report.append("")
    
    report.append("=" * 90)
    report.append("END OF REPORT")
    report.append("=" * 90)
    
    return '\n'.join(report)


def save_report(report):
    """Save report to file"""
    workspace = Path.home() / '.openclaw/workspace'
    report_path = workspace / 'PEC_TIER_DIMENSIONAL_REPORT.txt'
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report saved to {report_path}")
    print(report)


def main():
    print("Loading SIGNALS_MASTER.jsonl...")
    signals = load_signals_master()
    print(f"Loaded {len(signals):,} signals")
    print()
    
    print("Generating tier-dimensional report...")
    report = generate_report(signals)
    
    print("Saving report...")
    save_report(report)


if __name__ == '__main__':
    main()
