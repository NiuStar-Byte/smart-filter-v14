#!/usr/bin/env python3
"""
CLOSURE RECONCILIATION READER
Merges SIGNALS_MASTER.jsonl + SIGNALS_CLOSURES.jsonl for unified signal view
Built: May 19 2026, 11:49 GMT+7

Architecture:
- SIGNALS_MASTER: Fire events (read-only, never modified)
- SIGNALS_CLOSURES: Closure events (append-only from pec_executor_safe)
- This module: Reads both, applies closures, returns merged view

Safe: No writes, no atomic writer, no data loss, RAM-efficient (streaming)
"""

import json
import os
from collections import defaultdict
from pathlib import Path

class ClosureReconciliationReader:
    """
    Safely merges closure data from two files into unified signal view
    
    Usage:
        reader = ClosureReconciliationReader()
        signals = reader.get_all_signals_with_closures()  # Full merged view
        metrics = reader.calculate_metrics()  # WR, P&L, counts
    """
    
    def __init__(self, workspace_root=None):
        """
        Initialize reconciliation reader
        
        Args:
            workspace_root: Path to workspace root (auto-detected if None)
        """
        if workspace_root is None:
            workspace_root = os.path.dirname(os.path.abspath(__file__))
        
        self.workspace = os.path.dirname(workspace_root)
        self.signals_master = os.path.join(workspace_root, 'SIGNALS_MASTER.jsonl')
        self.signals_closures = os.path.join(self.workspace, 'SIGNALS_CLOSURES.jsonl')
        
        # Cache: closure events by UUID (build once, reuse)
        self._closure_cache = None
        self._master_cache = None
        
        print(f"[RECONCILIATION] ✅ Initialized closure reconciliation reader", flush=True)
    
    def _load_closures(self):
        """
        Load all closure events from SIGNALS_CLOSURES.jsonl
        
        Returns:
            dict: {signal_uuid: closure_event}
        """
        if self._closure_cache is not None:
            return self._closure_cache
        
        closures_by_uuid = {}
        
        if not os.path.exists(self.signals_closures):
            print(f"[RECONCILIATION] ℹ️  No SIGNALS_CLOSURES.jsonl yet (pec_executor hasn't run)", flush=True)
            self._closure_cache = {}
            return closures_by_uuid
        
        try:
            with open(self.signals_closures, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        closure = json.loads(line)
                        uuid = closure.get('signal_uuid')
                        if uuid:
                            # File is now deduplicated (single closure per UUID)
                            # Store all closure events
                            if uuid not in closures_by_uuid:
                                closures_by_uuid[uuid] = closure
                    except:
                        pass
            
            print(f"[RECONCILIATION] ✅ Loaded {len(closures_by_uuid)} closure events (file deduplicated)", flush=True)
            self._closure_cache = closures_by_uuid
            return closures_by_uuid
        
        except Exception as e:
            print(f"[RECONCILIATION] ⚠️  Error loading closures: {str(e)[:60]}", flush=True)
            self._closure_cache = {}
            return {}
    
    def _load_master_signals(self):
        """
        Load all signals from SIGNALS_MASTER.jsonl
        
        Returns:
            dict: {signal_uuid: signal}
        """
        if self._master_cache is not None:
            return self._master_cache
        
        signals_by_uuid = {}
        
        try:
            with open(self.signals_master, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        signal = json.loads(line)
                        uuid = signal.get('signal_uuid')
                        if uuid:
                            signals_by_uuid[uuid] = signal
                    except:
                        pass
            
            print(f"[RECONCILIATION] ✅ Loaded {len(signals_by_uuid)} master signals", flush=True)
            self._master_cache = signals_by_uuid
            return signals_by_uuid
        
        except Exception as e:
            print(f"[RECONCILIATION] ⚠️  Error loading master: {str(e)[:60]}", flush=True)
            self._master_cache = {}
            return {}
    
    def get_all_signals_with_closures(self):
        """
        Get unified view: SIGNALS_MASTER + closures applied
        
        Returns:
            list: Signal dicts with closure data merged
        """
        master_signals = self._load_master_signals()
        closures = self._load_closures()
        
        merged_signals = []
        
        for uuid, signal in master_signals.items():
            # Start with original signal
            merged = signal.copy()
            
            # Apply closure if available (overrides original status)
            if uuid in closures:
                closure = closures[uuid]
                merged['status'] = closure.get('status', signal.get('status'))
                merged['actual_exit_price'] = closure.get('actual_exit_price')
                merged['closed_at'] = closure.get('closed_at')
                merged['close_reason'] = closure.get('close_reason')
                merged['was_closed_by_pec'] = True  # Mark for audit
                
                # Calculate P&L from closure data (direction-aware)
                entry = float(merged.get('entry_price', 0)) if merged.get('entry_price') else 0
                exit_price = closure.get('actual_exit_price')
                if entry and exit_price:
                    exit_price = float(exit_price)
                    quantity = float(merged.get('quantity', 1)) if merged.get('quantity') else 1
                    direction = merged.get('signal_type') or merged.get('direction', 'LONG').upper()
                    
                    # Direction-aware P&L calculation
                    if direction == 'SHORT':
                        merged['pnl_usd'] = (entry - exit_price) * quantity  # SHORT: profit when exit < entry
                        merged['pnl_pct'] = ((entry - exit_price) / entry * 100) if entry != 0 else 0
                    else:  # LONG
                        merged['pnl_usd'] = (exit_price - entry) * quantity  # LONG: profit when exit > entry
                        merged['pnl_pct'] = ((exit_price - entry) / entry * 100) if entry != 0 else 0
            else:
                merged['was_closed_by_pec'] = False
            
            merged_signals.append(merged)
        
        return merged_signals
    
    def get_signal_status(self, signal_uuid):
        """
        Get current status of a single signal (with closures applied)
        
        Args:
            signal_uuid: UUID to look up
            
        Returns:
            dict: Signal with closure data merged, or None if not found
        """
        master_signals = self._load_master_signals()
        closures = self._load_closures()
        
        if signal_uuid not in master_signals:
            return None
        
        signal = master_signals[signal_uuid].copy()
        
        if signal_uuid in closures:
            closure = closures[signal_uuid]
            signal['status'] = closure.get('status', signal.get('status'))
            signal['actual_exit_price'] = closure.get('actual_exit_price')
            signal['closed_at'] = closure.get('closed_at')
            signal['close_reason'] = closure.get('close_reason')
            signal['was_closed_by_pec'] = True
        else:
            signal['was_closed_by_pec'] = False
        
        return signal
    
    def calculate_metrics(self):
        """
        Calculate WR, P&L, and counts with closures applied
        
        Returns:
            dict: Comprehensive metrics
        """
        signals = self.get_all_signals_with_closures()
        
        metrics = {
            'total_signals': len(signals),
            'counts': defaultdict(int),
            'tp_count': 0,
            'sl_count': 0,
            'timeout_count': 0,
            'open_count': 0,
            'tp_total_pnl': 0.0,
            'sl_total_pnl': 0.0,
            'timeout_total_pnl': 0.0,
        }
        
        for sig in signals:
            status = sig.get('status', 'UNKNOWN')
            metrics['counts'][status] += 1
            
            if status == 'TP_HIT':
                metrics['tp_count'] += 1
                pnl = sig.get('pnl_usd', 0)
                metrics['tp_total_pnl'] += float(pnl) if pnl else 0
            elif status == 'SL_HIT':
                metrics['sl_count'] += 1
                pnl = sig.get('pnl_usd', 0)
                metrics['sl_total_pnl'] += float(pnl) if pnl else 0
            elif status in ['TIMEOUT', 'STALE_TIMEOUT']:
                metrics['timeout_count'] += 1
                pnl = sig.get('pnl_usd', 0)
                metrics['timeout_total_pnl'] += float(pnl) if pnl else 0
            elif status == 'OPEN':
                metrics['open_count'] += 1
        
        # Calculate win rate
        total_closures = metrics['tp_count'] + metrics['sl_count']
        metrics['win_rate_pct'] = (
            (metrics['tp_count'] / total_closures * 100)
            if total_closures > 0 else 0
        )
        
        # Calculate total P&L
        metrics['total_pnl'] = (
            metrics['tp_total_pnl'] + 
            metrics['sl_total_pnl'] + 
            metrics['timeout_total_pnl']
        )
        
        # Calculate metrics per category
        metrics['avg_pnl_per_tp'] = (
            metrics['tp_total_pnl'] / metrics['tp_count']
            if metrics['tp_count'] > 0 else 0
        )
        metrics['avg_pnl_per_sl'] = (
            metrics['sl_total_pnl'] / metrics['sl_count']
            if metrics['sl_count'] > 0 else 0
        )
        metrics['avg_pnl_per_timeout'] = (
            metrics['timeout_total_pnl'] / metrics['timeout_count']
            if metrics['timeout_count'] > 0 else 0
        )
        
        return metrics
    
    def get_summary(self):
        """
        Get human-readable summary of current system state
        
        Returns:
            str: Formatted summary
        """
        metrics = self.calculate_metrics()
        
        summary = f"""
╔════════════════════════════════════════════════════════════════╗
║          CLOSURE RECONCILIATION SUMMARY                        ║
╚════════════════════════════════════════════════════════════════╝

Total Signals: {metrics['total_signals']:,}
├─ OPEN: {metrics['open_count']:,}
├─ TP_HIT: {metrics['tp_count']:,}
├─ SL_HIT: {metrics['sl_count']:,}
└─ TIMEOUT: {metrics['timeout_count']:,}

Win Rate: {metrics['win_rate_pct']:.2f}% ({metrics['tp_count']} wins / {metrics['tp_count'] + metrics['sl_count']} closed)

P&L Summary:
├─ Total P&L: ${metrics['total_pnl']:,.2f}
├─ TP Profits: ${metrics['tp_total_pnl']:,.2f} (avg: ${metrics['avg_pnl_per_tp']:,.2f}/signal)
├─ SL Losses: ${metrics['sl_total_pnl']:,.2f} (avg: ${metrics['avg_pnl_per_sl']:,.2f}/signal)
└─ TIMEOUT: ${metrics['timeout_total_pnl']:,.2f} (avg: ${metrics['avg_pnl_per_timeout']:,.2f}/signal)

Data Sources:
├─ SIGNALS_MASTER.jsonl: {metrics['total_signals']:,} fire events
└─ SIGNALS_CLOSURES.jsonl: {sum(1 for m in metrics['counts'] if m in ['TP_HIT', 'SL_HIT', 'TIMEOUT'])} closure events
"""
        return summary.strip()


# Public interface for trackers
def get_reconciliation_reader():
    """Factory function to get closure reconciliation reader"""
    return ClosureReconciliationReader()
