#!/usr/bin/env python3
"""
UNIVERSAL SIGNAL LOADER
Standard interface for ALL trackers to load signals with closures applied
Built: May 19 2026, 11:56 GMT+7

CRITICAL: This is the CANONICAL way to load signals. All trackers MUST use this.

Architecture:
  - Reads SIGNALS_MASTER.jsonl (fire events)
  - Reads SIGNALS_CLOSURES.jsonl (closure events from pec_executor_safe)
  - Merges closures into signal view (status=OPEN/TP_HIT/SL_HIT/TIMEOUT/etc)
  - Returns unified list (as if closures were always in SIGNALS_MASTER)

Benefits:
  ✅ All trackers see closures immediately
  ✅ No rewriting SIGNALS_MASTER (safe)
  ✅ No data loss (append-only closures file)
  ✅ RAM-efficient (streaming)
  ✅ Consistent across all trackers

Usage in ANY tracker:
  from signal_loader import load_all_signals
  signals = load_all_signals()  # Gets all signals with closures applied
"""

import json
import os
import sys
from typing import List, Dict, Any


def load_all_signals(workspace_root=None) -> List[Dict[str, Any]]:
    """
    Load ALL signals with closures applied (CANONICAL FUNCTION FOR ALL TRACKERS)
    
    Args:
        workspace_root: Path to smart-filter-v14-main (auto-detected if None)
        
    Returns:
        list: Signal dicts with closure data merged
        
    Usage:
        from signal_loader import load_all_signals
        signals = load_all_signals()
    """
    if workspace_root is None:
        workspace_root = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Try closure reconciliation first (preferred method)
        from closure_reconciliation import get_reconciliation_reader
        reader = get_reconciliation_reader()
        return reader.get_all_signals_with_closures()
    except Exception as e:
        # Fallback: Load from SIGNALS_MASTER only (for safety)
        print(f"[WARN] Closure reconciliation failed ({str(e)[:40]}), using SIGNALS_MASTER only", flush=True)
        return _load_signals_fallback(workspace_root)


def load_signals_filtered(workspace_root=None, status_filter=None) -> List[Dict[str, Any]]:
    """
    Load signals filtered by status (e.g., only OPEN, or only TP_HIT)
    
    Args:
        workspace_root: Path to smart-filter-v14-main (auto-detected if None)
        status_filter: List of statuses to include, e.g., ['TP_HIT', 'SL_HIT'] or None for all
        
    Returns:
        list: Filtered signal dicts
        
    Usage:
        from signal_loader import load_signals_filtered
        
        # Get only closed signals
        closed = load_signals_filtered(status_filter=['TP_HIT', 'SL_HIT', 'TIMEOUT'])
        
        # Get only open signals  
        open_sigs = load_signals_filtered(status_filter=['OPEN'])
    """
    signals = load_all_signals(workspace_root)
    
    if status_filter is None:
        return signals
    
    return [s for s in signals if s.get('status') in status_filter]


def get_signal_metrics(signals=None, workspace_root=None) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics (WR, P&L, counts) from signals
    
    Args:
        signals: List of signals (loads if None)
        workspace_root: Path to smart-filter-v14-main (auto-detected if None)
        
    Returns:
        dict: {total, tp_count, sl_count, open_count, timeout_count, wr_pct, total_pnl, ...}
        
    Usage:
        from signal_loader import get_signal_metrics
        metrics = get_signal_metrics()
        print(f"WR: {metrics['wr_pct']:.2f}% | P&L: ${metrics['total_pnl']:,.2f}")
    """
    if signals is None:
        signals = load_all_signals(workspace_root)
    
    metrics = {
        'total': len(signals),
        'tp_count': 0,
        'sl_count': 0,
        'timeout_count': 0,
        'open_count': 0,
        'other_count': 0,
        'tp_pnl': 0.0,
        'sl_pnl': 0.0,
        'timeout_pnl': 0.0,
        'total_pnl': 0.0,
        'wr_pct': 0.0,
        'avg_profit_per_tp': 0.0,
        'avg_loss_per_sl': 0.0,
    }
    
    for sig in signals:
        status = sig.get('status', 'UNKNOWN')
        pnl = float(sig.get('pnl_usd', 0)) if sig.get('pnl_usd') else 0
        
        if status == 'TP_HIT':
            metrics['tp_count'] += 1
            metrics['tp_pnl'] += pnl
        elif status == 'SL_HIT':
            metrics['sl_count'] += 1
            metrics['sl_pnl'] += pnl
        elif status in ['TIMEOUT', 'STALE_TIMEOUT']:
            metrics['timeout_count'] += 1
            metrics['timeout_pnl'] += pnl
        elif status == 'OPEN':
            metrics['open_count'] += 1
        else:
            metrics['other_count'] += 1
    
    # Calculate derived metrics
    metrics['total_pnl'] = metrics['tp_pnl'] + metrics['sl_pnl'] + metrics['timeout_pnl']
    
    closed_count = metrics['tp_count'] + metrics['sl_count']
    if closed_count > 0:
        metrics['wr_pct'] = (metrics['tp_count'] / closed_count) * 100
        metrics['avg_profit_per_tp'] = metrics['tp_pnl'] / metrics['tp_count']
        metrics['avg_loss_per_sl'] = metrics['sl_pnl'] / metrics['sl_count'] if metrics['sl_count'] > 0 else 0
    
    return metrics


def _load_signals_fallback(workspace_root) -> List[Dict[str, Any]]:
    """Fallback: Load from SIGNALS_MASTER.jsonl only (no closures)"""
    signals_master = os.path.join(workspace_root, 'SIGNALS_MASTER.jsonl')
    signals = []
    
    try:
        with open(signals_master, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        signals.append(json.loads(line))
                    except:
                        pass
        print(f"[FALLBACK] Loaded {len(signals)} signals from SIGNALS_MASTER.jsonl (no closures)", flush=True)
        return signals
    except Exception as e:
        print(f"[ERROR] Could not load signals: {str(e)}", flush=True)
        return []


# Quick test
if __name__ == '__main__':
    print("Testing signal_loader...")
    signals = load_all_signals()
    metrics = get_signal_metrics(signals)
    print(f"✅ Loaded {metrics['total']} signals")
    print(f"   OPEN: {metrics['open_count']} | TP: {metrics['tp_count']} | SL: {metrics['sl_count']} | TIMEOUT: {metrics['timeout_count']}")
    print(f"   WR: {metrics['wr_pct']:.2f}% | P&L: ${metrics['total_pnl']:,.2f}")
