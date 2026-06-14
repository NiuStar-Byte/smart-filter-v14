#!/usr/bin/env python3
"""
ASTERDEX Performance Analytics - Generate trading metrics and reports
Calculates WR, P&L by Tier, Symbol, Timeframe, MTF alignment, Route
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

# File locations
CORRELATED_FILE = Path(__file__).parent / "ASTERDEX_PERFORMANCE_CORRELATED.jsonl"


def load_correlated():
    """Load all correlated trades."""
    trades = []
    if CORRELATED_FILE.exists():
        with open(CORRELATED_FILE, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    trades.append(json.loads(line))
                except:
                    pass
    return trades


def determine_win(trade):
    """Determine if trade was a win (positive P&L)."""
    pnl = trade.get('realized_pnl_usd', 0)
    return pnl > 0


def calculate_metrics_for_trades(trades):
    """
    Calculate WR and P&L metrics for a list of trades.
    Returns: {wr, count, total_pnl, avg_pnl, min_pnl, max_pnl, stdev_pnl}
    """
    if not trades:
        return None
    
    count = len(trades)
    wins = sum(1 for t in trades if determine_win(t))
    wr = (wins / count * 100) if count > 0 else 0
    
    pnls = [t.get('realized_pnl_usd', 0) for t in trades]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / count if count > 0 else 0
    min_pnl = min(pnls) if pnls else 0
    max_pnl = max(pnls) if pnls else 0
    
    try:
        stdev_pnl = stdev(pnls) if count > 1 else 0
    except:
        stdev_pnl = 0
    
    return {
        'wr': wr,
        'count': count,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'min_pnl': min_pnl,
        'max_pnl': max_pnl,
        'stdev_pnl': stdev_pnl,
    }


def analyze_performance():
    """
    Analyze all correlated trades and generate metrics by dimension.
    """
    trades = load_correlated()
    
    if not trades:
        print("[WARN] No correlated trades yet. Nothing to analyze.")
        return {}
    
    print(f"\n[{datetime.now().isoformat()}] Analyzing {len(trades)} trades...")
    
    analysis = {
        'total_trades': len(trades),
        'last_updated': datetime.now().isoformat(),
        'by_tier': {},
        'by_symbol': {},
        'by_timeframe': {},
        'by_mtf_band': {},
        'by_route': {},
        'by_combination': {},  # tier + symbol + timeframe
    }
    
    # Group by tier
    tier_groups = defaultdict(list)
    for trade in trades:
        tier = trade.get('tier', 'unknown')
        tier_groups[tier].append(trade)
    
    for tier, tier_trades in tier_groups.items():
        metrics = calculate_metrics_for_trades(tier_trades)
        if metrics:
            analysis['by_tier'][f'Tier-{tier}'] = metrics
    
    # Group by symbol
    symbol_groups = defaultdict(list)
    for trade in trades:
        symbol = trade.get('symbol', 'unknown')
        symbol_groups[symbol].append(trade)
    
    for symbol, symbol_trades in symbol_groups.items():
        metrics = calculate_metrics_for_trades(symbol_trades)
        if metrics:
            analysis['by_symbol'][symbol] = metrics
    
    # Group by timeframe
    tf_groups = defaultdict(list)
    for trade in trades:
        tf = trade.get('timeframe', 'unknown')
        tf_groups[tf].append(trade)
    
    for tf, tf_trades in tf_groups.items():
        metrics = calculate_metrics_for_trades(tf_trades)
        if metrics:
            analysis['by_timeframe'][tf] = metrics
    
    # Group by MTF alignment
    mtf_groups = defaultdict(list)
    for trade in trades:
        mtf = trade.get('mtf_alignment_band', 'unknown')
        mtf_groups[mtf].append(trade)
    
    for mtf, mtf_trades in mtf_groups.items():
        metrics = calculate_metrics_for_trades(mtf_trades)
        if metrics:
            analysis['by_mtf_band'][mtf] = metrics
    
    # Group by route
    route_groups = defaultdict(list)
    for trade in trades:
        route = trade.get('route', 'unknown')
        route_groups[route].append(trade)
    
    for route, route_trades in route_groups.items():
        metrics = calculate_metrics_for_trades(route_trades)
        if metrics:
            analysis['by_route'][route] = metrics
    
    return analysis


def format_report(analysis):
    """Format analysis into readable report."""
    if not analysis:
        return "No trading data available yet."
    
    lines = []
    lines.append("=" * 60)
    lines.append("ASTERDEX PERFORMANCE REPORT")
    lines.append(f"Generated: {analysis['last_updated']}")
    lines.append("=" * 60)
    lines.append("")
    
    lines.append(f"Total Trades Analyzed: {analysis['total_trades']}")
    lines.append("")
    
    # By tier
    if analysis['by_tier']:
        lines.append("Performance by TIER:")
        lines.append("-" * 60)
        for tier, metrics in sorted(analysis['by_tier'].items()):
            lines.append(
                f"{tier:12} | WR: {metrics['wr']:5.1f}% | "
                f"Trades: {metrics['count']:3} | "
                f"Avg P&L: ${metrics['avg_pnl']:+8.2f} | "
                f"Total: ${metrics['total_pnl']:+8.2f}"
            )
        lines.append("")
    
    # By symbol
    if analysis['by_symbol']:
        lines.append("Performance by SYMBOL:")
        lines.append("-" * 60)
        # Sort by total P&L
        for symbol, metrics in sorted(
            analysis['by_symbol'].items(),
            key=lambda x: x[1]['total_pnl'],
            reverse=True
        ):
            lines.append(
                f"{symbol:12} | WR: {metrics['wr']:5.1f}% | "
                f"Trades: {metrics['count']:3} | "
                f"Avg P&L: ${metrics['avg_pnl']:+8.2f} | "
                f"Total: ${metrics['total_pnl']:+8.2f}"
            )
        lines.append("")
    
    # By timeframe
    if analysis['by_timeframe']:
        lines.append("Performance by TIMEFRAME:")
        lines.append("-" * 60)
        for tf, metrics in sorted(analysis['by_timeframe'].items()):
            lines.append(
                f"{tf:12} | WR: {metrics['wr']:5.1f}% | "
                f"Trades: {metrics['count']:3} | "
                f"Avg P&L: ${metrics['avg_pnl']:+8.2f}"
            )
        lines.append("")
    
    # By MTF alignment
    if analysis['by_mtf_band']:
        lines.append("Performance by MTF ALIGNMENT:")
        lines.append("-" * 60)
        for mtf, metrics in sorted(analysis['by_mtf_band'].items()):
            lines.append(
                f"{mtf:12} | WR: {metrics['wr']:5.1f}% | "
                f"Trades: {metrics['count']:3} | "
                f"Avg P&L: ${metrics['avg_pnl']:+8.2f} | "
                f"Total: ${metrics['total_pnl']:+8.2f}"
            )
        lines.append("")
    
    # By route
    if analysis['by_route']:
        lines.append("Performance by ROUTE:")
        lines.append("-" * 60)
        for route, metrics in sorted(analysis['by_route'].items()):
            lines.append(
                f"{route:25} | WR: {metrics['wr']:5.1f}% | "
                f"Trades: {metrics['count']:3} | "
                f"Avg P&L: ${metrics['avg_pnl']:+8.2f}"
            )
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_analysis(analysis):
    """Save analysis to JSON file."""
    output_file = Path(__file__).parent / "ASTERDEX_PERFORMANCE_ANALYSIS.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"[INFO] Analysis saved to {output_file}")


def main():
    """Main entry point."""
    analysis = analyze_performance()
    
    if analysis:
        report = format_report(analysis)
        print(report)
        save_analysis(analysis)
    else:
        print("[WARN] No data to analyze")


if __name__ == '__main__':
    main()
