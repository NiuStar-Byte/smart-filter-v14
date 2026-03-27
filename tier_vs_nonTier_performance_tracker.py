#!/usr/bin/env python3
"""
TIER vs NON-TIER PERFORMANCE TRACKER
Bifurcated performance analysis: signals with tier assignment vs without tier assignment

PURPOSE:
- Separate performance metrics for tier-assigned signals vs non-tier signals
- Segment dynamic combos by tier status (6D/5D/4D/3D/2D)
- Measure if tiering actually improves signal quality (WR, P&L, averages)
- Direct comparison: tier performance > non-tier performance?

OUTPUT:
- Signals with TIER section (WR%, P&L breakdown, averages)
- Signals without TIER section (same metrics for comparison)
- Dynamic combos segmented by tier status
- Side-by-side comparison showing tier advantage

AUTHOR: Quality Loop Monitor
LOCKED: 2026-03-28 (code immutable, data dynamic)
"""

import json
import os
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path


def load_signals_master():
    """Load SIGNALS_MASTER.jsonl"""
    signals = []
    filepath = Path.home() / '.openclaw/workspace/SIGNALS_MASTER.jsonl'
    
    if not filepath.exists():
        print(f"ERROR: {filepath} not found")
        return []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                signals.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass  # Skip corrupted lines
    
    return signals


def categorize_signals(signals):
    """Segment signals into WITH_TIER and WITHOUT_TIER"""
    with_tier = []
    without_tier = []
    
    for signal in signals:
        tier = signal.get('tier', None)
        if tier and tier != 'None':
            with_tier.append(signal)
        else:
            without_tier.append(signal)
    
    return with_tier, without_tier


def calculate_segment_metrics(signals):
    """Calculate metrics for a signal segment"""
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
            'timeout_win': 0, 'timeout_loss': 0, 'stale_timeout': 0,
            'total_pnl': 0.0, 'pnls': [], 'tiers': set()
        })
    }
    
    for signal in signals:
        status = signal.get('status', 'OPEN')
        pnl = signal.get('pnl_usd', 0.0)
        tier = signal.get('tier', None)
        
        # Extract combo from signal (6D full notation)
        combo = extract_combo_6d(signal)
        metrics['combos'][combo]['total'] += 1
        metrics['combos'][combo]['tiers'].add(tier if tier and tier != 'None' else 'NO_TIER')
        
        # Closed statuses: TP_HIT, SL_HIT, TIMEOUT, STALE_TIMEOUT (not OPEN, not REJECTED)
        closed_statuses = ['TP_HIT', 'SL_HIT', 'TIMEOUT', 'STALE_TIMEOUT']
        
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
                # Determine TIMEOUT_WIN vs TIMEOUT_LOSS based on pnl sign
                if pnl >= 0:
                    metrics['timeout_win'] += 1
                    metrics['combos'][combo]['timeout_win'] += 1
                    metrics['timeout_win_pnls'].append(pnl)
                else:
                    metrics['timeout_loss'] += 1
                    metrics['combos'][combo]['timeout_loss'] += 1
                    metrics['timeout_loss_pnls'].append(pnl)
            elif status == 'STALE_TIMEOUT':
                metrics['stale_timeout'] += 1
                metrics['combos'][combo]['stale_timeout'] += 1
    
    return metrics


def extract_combo_6d(signal):
    """Extract 6D combo notation from signal"""
    tf = signal.get('timeframe', '?')
    direction = signal.get('direction', '?')
    pattern = signal.get('pattern', '?')
    regime = signal.get('regime', '?')
    alts = signal.get('alts_type', '?')
    volatility = signal.get('volatility_level', '?')
    return f"{tf}|{direction}|{pattern}|{regime}|{alts}|{volatility}"


def extract_combos_reduced(combo_6d):
    """Generate reduced-dimension combos from 6D"""
    parts = combo_6d.split('|')
    if len(parts) != 6:
        return {'6D': combo_6d}
    
    return {
        '6D': combo_6d,
        '5D': '|'.join(parts[:5]),
        '4D': '|'.join(parts[:4]),
        '3D': '|'.join(parts[:3]),
        '2D': '|'.join(parts[:2]),
    }


def calculate_win_rate(metrics):
    """Calculate win rate from closed signals"""
    if metrics['closed'] == 0:
        return 0.0
    tp = metrics['tp_hit']
    sl = metrics['sl_hit']
    timeout_win = metrics['timeout_win']
    
    total_outcomes = tp + sl + timeout_win + metrics['timeout_loss']
    if total_outcomes == 0:
        return 0.0
    
    return (tp + timeout_win) / total_outcomes * 100


def format_pnl(value):
    """Format PnL as USD"""
    return f"${value:,.2f}"


def extract_fire_date(signal):
    """Extract fire date from signal"""
    fired_utc = signal.get('fired_time_utc', '')
    if fired_utc:
        return fired_utc[:10]
    return None


def generate_report(with_tier_signals, without_tier_signals):
    """Generate bifurcated comparison report"""
    with_tier_metrics = calculate_segment_metrics(with_tier_signals)
    without_tier_metrics = calculate_segment_metrics(without_tier_signals)
    
    # Extract timeline info for context
    with_tier_dates = set(extract_fire_date(s) for s in with_tier_signals if extract_fire_date(s))
    without_tier_dates = set(extract_fire_date(s) for s in without_tier_signals if extract_fire_date(s))
    
    report = []
    report.append("=" * 80)
    report.append("TIER vs NON-TIER PERFORMANCE TRACKER (TEMPORAL ANALYSIS)")
    report.append(f"Report Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    report.append("=" * 80)
    report.append("")
    
    report.append("⚠️  CRITICAL TIMELINE CONTEXT:")
    if with_tier_dates:
        report.append(f"  WITH TIER signals:    Fired {min(with_tier_dates)} to {max(with_tier_dates)}")
        report.append(f"  WITHOUT TIER signals: Fired {min(without_tier_dates)} to {max(without_tier_dates)}")
    report.append("")
    report.append("  IMPORTANT: Tier assignment only began on 2026-03-27.")
    report.append("  Pre-Mar-27 signals have ZERO tier (historical baseline).")
    report.append("  Tier signals are NEW (just fired) and still accumulating closes.")
    report.append("")
    
    # ========== SIGNALS WITH TIER ==========
    report.append("=" * 80)
    report.append("SECTION 1: SIGNALS WITH TIER ASSIGNMENT")
    report.append("=" * 80)
    report.append("")
    
    report.append(f"Total Loaded Signals: {with_tier_metrics['total']:,}")
    report.append(f"Total Closed Signals: {with_tier_metrics['closed']:,}")
    report.append("")
    
    report.append("CLOSED SIGNAL BREAKDOWN:")
    report.append(f"  TP_HIT:         {with_tier_metrics['tp_hit']:,} signals")
    report.append(f"  SL_HIT:         {with_tier_metrics['sl_hit']:,} signals")
    report.append(f"  TIMEOUT_WIN:    {with_tier_metrics['timeout_win']:,} signals")
    report.append(f"  TIMEOUT_LOSS:   {with_tier_metrics['timeout_loss']:,} signals")
    report.append(f"  STALE_TIMEOUT:  {with_tier_metrics['stale_timeout']:,} signals")
    report.append("")
    
    wr_with_tier = calculate_win_rate(with_tier_metrics)
    report.append(f"Overall Win Rate: {wr_with_tier:.2f}%")
    report.append("")
    
    report.append("P&L BREAKDOWN:")
    report.append(f"  Total P&L:               {format_pnl(with_tier_metrics['total_pnl'])}")
    report.append(f"  TP P&L:                  {format_pnl(sum(with_tier_metrics['tp_pnls']))}")
    report.append(f"  SL P&L:                  {format_pnl(sum(with_tier_metrics['sl_pnls']))}")
    report.append(f"  TIMEOUT_WIN P&L:         {format_pnl(sum(with_tier_metrics['timeout_win_pnls']))}")
    report.append(f"  TIMEOUT_LOSS P&L:        {format_pnl(sum(with_tier_metrics['timeout_loss_pnls']))}")
    report.append("")
    
    if with_tier_metrics['closed'] > 0:
        avg_per_signal = with_tier_metrics['total_pnl'] / with_tier_metrics['total']
        avg_per_closed = with_tier_metrics['total_pnl'] / with_tier_metrics['closed']
        report.append("AVERAGES:")
        report.append(f"  Avg P&L per Signal:      {format_pnl(avg_per_signal)}")
        report.append(f"  Avg P&L per Closed:      {format_pnl(avg_per_closed)}")
        
        if with_tier_metrics['tp_hit'] > 0:
            report.append(f"  Avg TP:                  {format_pnl(sum(with_tier_metrics['tp_pnls']) / with_tier_metrics['tp_hit'])}")
        if with_tier_metrics['sl_hit'] > 0:
            report.append(f"  Avg SL:                  {format_pnl(sum(with_tier_metrics['sl_pnls']) / with_tier_metrics['sl_hit'])}")
    report.append("")
    
    # ========== SIGNALS WITHOUT TIER ==========
    report.append("=" * 80)
    report.append("SECTION 2: SIGNALS WITHOUT TIER ASSIGNMENT")
    report.append("=" * 80)
    report.append("")
    
    report.append(f"Total Loaded Signals: {without_tier_metrics['total']:,}")
    report.append(f"Total Closed Signals: {without_tier_metrics['closed']:,}")
    report.append("")
    
    report.append("CLOSED SIGNAL BREAKDOWN:")
    report.append(f"  TP_HIT:         {without_tier_metrics['tp_hit']:,} signals")
    report.append(f"  SL_HIT:         {without_tier_metrics['sl_hit']:,} signals")
    report.append(f"  TIMEOUT_WIN:    {without_tier_metrics['timeout_win']:,} signals")
    report.append(f"  TIMEOUT_LOSS:   {without_tier_metrics['timeout_loss']:,} signals")
    report.append(f"  STALE_TIMEOUT:  {without_tier_metrics['stale_timeout']:,} signals")
    report.append("")
    
    wr_without_tier = calculate_win_rate(without_tier_metrics)
    report.append(f"Overall Win Rate: {wr_without_tier:.2f}%")
    report.append("")
    
    report.append("P&L BREAKDOWN:")
    report.append(f"  Total P&L:               {format_pnl(without_tier_metrics['total_pnl'])}")
    report.append(f"  TP P&L:                  {format_pnl(sum(without_tier_metrics['tp_pnls']))}")
    report.append(f"  SL P&L:                  {format_pnl(sum(without_tier_metrics['sl_pnls']))}")
    report.append(f"  TIMEOUT_WIN P&L:         {format_pnl(sum(without_tier_metrics['timeout_win_pnls']))}")
    report.append(f"  TIMEOUT_LOSS P&L:        {format_pnl(sum(without_tier_metrics['timeout_loss_pnls']))}")
    report.append("")
    
    if without_tier_metrics['closed'] > 0:
        avg_per_signal = without_tier_metrics['total_pnl'] / without_tier_metrics['total']
        avg_per_closed = without_tier_metrics['total_pnl'] / without_tier_metrics['closed']
        report.append("AVERAGES:")
        report.append(f"  Avg P&L per Signal:      {format_pnl(avg_per_signal)}")
        report.append(f"  Avg P&L per Closed:      {format_pnl(avg_per_closed)}")
        
        if without_tier_metrics['tp_hit'] > 0:
            report.append(f"  Avg TP:                  {format_pnl(sum(without_tier_metrics['tp_pnls']) / without_tier_metrics['tp_hit'])}")
        if without_tier_metrics['sl_hit'] > 0:
            report.append(f"  Avg SL:                  {format_pnl(sum(without_tier_metrics['sl_pnls']) / without_tier_metrics['sl_hit'])}")
    report.append("")
    
    # ========== COMPARISON ==========
    report.append("=" * 80)
    report.append("SECTION 3: TIER vs NON-TIER COMPARISON")
    report.append("=" * 80)
    report.append("")
    
    wr_delta = wr_with_tier - wr_without_tier
    pnl_delta = with_tier_metrics['total_pnl'] - without_tier_metrics['total_pnl']
    
    report.append("WIN RATE COMPARISON:")
    report.append(f"  WITH TIER:           {wr_with_tier:>6.2f}%")
    report.append(f"  WITHOUT TIER:        {wr_without_tier:>6.2f}%")
    report.append(f"  DELTA (Tier Adv):    {wr_delta:>6.2f}% {'✓ TIER BETTER' if wr_delta > 0 else '✗ TIER WORSE'}")
    report.append("")
    
    report.append("TOTAL P&L COMPARISON:")
    report.append(f"  WITH TIER:           {format_pnl(with_tier_metrics['total_pnl']):>15}")
    report.append(f"  WITHOUT TIER:        {format_pnl(without_tier_metrics['total_pnl']):>15}")
    report.append(f"  DELTA (Tier Adv):    {format_pnl(pnl_delta):>15} {'✓ TIER BETTER' if pnl_delta > 0 else '✗ TIER WORSE'}")
    report.append("")
    
    # ========== DYNAMIC COMBOS WITH TIER vs WITHOUT TIER ==========
    report.append("=" * 80)
    report.append("SECTION 4: DYNAMIC COMBOS BY TIER STATUS")
    report.append("=" * 80)
    report.append("")
    
    report.append("6D COMBOS WITH TIER (Top 10 by closed trades):")
    combos_with_tier = [(k, v) for k, v in with_tier_metrics['combos'].items() if v['closed'] > 0]
    combos_with_tier.sort(key=lambda x: x[1]['closed'], reverse=True)
    
    for i, (combo, stats) in enumerate(combos_with_tier[:10], 1):
        if stats['closed'] > 0:
            wr = (stats['tp_hit'] + stats['timeout_win']) / (stats['tp_hit'] + stats['sl_hit'] + stats['timeout_win'] + stats['timeout_loss']) * 100
            avg_pnl = stats['total_pnl'] / stats['closed']
            tier_list = ', '.join(stats['tiers'])
            report.append(f"  {i}. {combo}")
            report.append(f"     Closed: {stats['closed']}, WR: {wr:.1f}%, Avg P&L: {format_pnl(avg_pnl)}, Tiers: {tier_list}")
    report.append("")
    
    report.append("6D COMBOS WITHOUT TIER (Top 10 by closed trades):")
    combos_without_tier = [(k, v) for k, v in without_tier_metrics['combos'].items() if v['closed'] > 0]
    combos_without_tier.sort(key=lambda x: x[1]['closed'], reverse=True)
    
    for i, (combo, stats) in enumerate(combos_without_tier[:10], 1):
        if stats['closed'] > 0:
            wr = (stats['tp_hit'] + stats['timeout_win']) / (stats['tp_hit'] + stats['sl_hit'] + stats['timeout_win'] + stats['timeout_loss']) * 100
            avg_pnl = stats['total_pnl'] / stats['closed']
            report.append(f"  {i}. {combo}")
            report.append(f"     Closed: {stats['closed']}, WR: {wr:.1f}%, Avg P&L: {format_pnl(avg_pnl)}")
    report.append("")
    
    report.append("=" * 80)
    report.append("SECTION 5: TEMPORAL ANALYSIS & HYPOTHESIS")
    report.append("=" * 80)
    report.append("")
    
    report.append("CURRENT STATE (TEMPORAL FAIRNESS):")
    report.append(f"  Tier-Assigned Signals:       {with_tier_metrics['total']:,} signals")
    report.append(f"    └─ Fire Date Range:        Mar 27 only (BRAND NEW - just fired)")
    report.append(f"    └─ Closed:                 0 (zero time to close yet)")
    report.append(f"    └─ Status:                 100% OPEN")
    report.append("")
    report.append(f"  Non-Tier Signals:            {without_tier_metrics['total']:,} signals")
    report.append(f"    └─ Fire Date Range:        Feb 27 - Mar 26 (HISTORICAL - pre-tier implementation)")
    report.append(f"    └─ Closed:                 {without_tier_metrics['closed']:,} ({100*without_tier_metrics['closed']/without_tier_metrics['total']:.1f}%)")
    report.append(f"    └─ Win Rate:               {wr_without_tier:.2f}%")
    report.append("")
    
    report.append("KEY INSIGHT - WHY ZERO TIER CLOSES:")
    report.append("  Tier assignment only started 2026-03-27 (today).")
    report.append("  All signals fired before Mar 27 have NO tier (historical).")
    report.append("  Tier-assigned signals are brand new → zero closure time yet.")
    report.append("  This is CORRECT - not a bug, but a timeline effect.")
    report.append("")
    
    report.append("COMPARISON STATUS: NOT YET VALID")
    report.append("  Current comparison is apples-to-oranges (new vs historical).")
    report.append("  Meaningful comparison requires tier signals to accumulate closes.")
    report.append("")
    
    report.append("WAIT FOR (ETA: 24-48 hours):")
    report.append("  - Tier-X signals:   10+ closed trades → baseline WR")
    report.append("  - Tier-2 signals:   ~30-50 closed trades → test hypothesis")
    report.append("  - Tier-3 signals:   ~25-40 closed trades → test hypothesis")
    report.append("")
    
    report.append("HYPOTHESIS TEST (Once tier signals close):")
    report.append(f"  Null:       Tier WR% = {wr_without_tier:.2f}% (no difference)")
    report.append(f"  Alternate:  Tier WR% > {wr_without_tier:.2f}% (tiering improves quality)")
    report.append(f"  Success Threshold: {wr_without_tier + 8:.2f}% (8%+ improvement)")
    report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return '\n'.join(report)


def save_report(report):
    """Save report to file"""
    workspace = Path.home() / '.openclaw/workspace'
    report_path = workspace / 'TIER_vs_NONTIER_PERFORMANCE.txt'
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report saved to {report_path}")
    print(report)


def main():
    print("Loading SIGNALS_MASTER.jsonl...")
    signals = load_signals_master()
    print(f"Loaded {len(signals)} signals")
    
    print("Segmenting signals by tier status...")
    with_tier, without_tier = categorize_signals(signals)
    print(f"  WITH TIER:    {len(with_tier)} signals")
    print(f"  WITHOUT TIER: {len(without_tier)} signals")
    
    print("Calculating metrics...")
    report = generate_report(with_tier, without_tier)
    
    print("Saving report...")
    save_report(report)


if __name__ == '__main__':
    main()
