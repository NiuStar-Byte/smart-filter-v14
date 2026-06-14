#!/usr/bin/env python3
"""
ASTERDEX Comprehensive Order Matcher - Match all 28+ positions with full order history
Uses ORDER ID matching (entry_order_id → exit_order_id already extracted)
"""

import json
from datetime import datetime
from pathlib import Path

ENTRIES_FILE = Path(__file__).parent / "ASTERDEX_COMPREHENSIVE_ENTRIES.jsonl"
ORDERS_FILE = Path(__file__).parent / "ASTERDEX_TRADES_COMPREHENSIVE.jsonl"
CORRELATED_FILE = Path(__file__).parent / "ASTERDEX_PERFORMANCE_CORRELATED.jsonl"


def load_entries():
    """Load comprehensive entries."""
    entries = []
    if ENTRIES_FILE.exists():
        with open(ENTRIES_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except:
                        pass
    return entries


def load_orders():
    """Load comprehensive orders."""
    orders = []
    if ORDERS_FILE.exists():
        with open(ORDERS_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        orders.append(json.loads(line))
                    except:
                        pass
    return orders


def load_correlated():
    """Load already correlated positions."""
    correlated = set()
    if CORRELATED_FILE.exists():
        with open(CORRELATED_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        correlated.add(entry.get('signal_uuid'))
                    except:
                        pass
    return correlated


def build_order_index(orders):
    """Build index by orderId."""
    index = {}
    for order in orders:
        order_id = order.get('orderId')
        if order_id:
            index[order_id] = order
    return index


def match_comprehensive():
    """Match all comprehensive entries with orders."""
    print(f"\n[{datetime.now().isoformat()}] Starting comprehensive ORDER ID matching...")
    
    entries = load_entries()
    orders = load_orders()
    already_correlated = load_correlated()
    
    print(f"[INFO] Loaded {len(entries)} comprehensive entries")
    print(f"[INFO] Loaded {len(orders)} comprehensive orders")
    print(f"[INFO] Already correlated: {len(already_correlated)}")
    
    order_index = build_order_index(orders)
    print(f"[INFO] Built order index with {len(order_index)} unique order IDs")
    
    matches = 0
    no_entry = 0
    no_exit = 0
    
    with open(CORRELATED_FILE, 'a') as f:
        for entry in entries:
            uuid = entry.get('signal_uuid')
            
            if uuid in already_correlated:
                continue
            
            # Get entry order ID
            entry_oid = entry.get('entry_order_id')
            if not entry_oid or entry_oid not in order_index:
                no_entry += 1
                continue
            
            entry_order = order_index[entry_oid]
            
            # Get exit order ID
            exit_oid = entry.get('tp_order_id') or entry.get('exit_order_id')
            if not exit_oid or exit_oid not in order_index:
                no_exit += 1
                continue
            
            exit_order = order_index[exit_oid]
            
            # Calculate P&L
            entry_cost = float(entry_order.get('cumQuote', 0))
            exit_revenue = float(exit_order.get('cumQuote', 0))
            commission = (entry_cost + exit_revenue) * 0.0002
            
            side = entry.get('side')
            if side == 'LONG':
                pnl = exit_revenue - entry_cost - commission
            else:
                pnl = entry_cost - exit_revenue - commission
            
            # Build record
            correlated = {
                'signal_uuid': uuid,
                'symbol': entry.get('symbol'),
                'tier': entry.get('tier'),
                'mtf_alignment_band': entry.get('mtf_alignment_band'),
                'route': entry.get('route'),
                'timeframe': entry.get('timeframe'),
                'confidence_level': entry.get('confidence_level'),
                'side': side,
                'entry_price': float(entry_order.get('avgPrice', 0)),
                'exit_price': float(exit_order.get('avgPrice', 0)),
                'quantity': entry.get('quantity'),
                'posted_timestamp': entry.get('posted_timestamp'),
                'exit_timestamp': exit_order.get('time'),
                'realized_pnl_usd': pnl,
                'exit_type': 'CLOSED',
                'win': pnl > 0,
                'entry_order_id': entry_oid,
                'exit_order_id': exit_oid,
                'match_method': 'order_id_comprehensive',
            }
            
            f.write(json.dumps(correlated) + '\n')
            matches += 1
    
    print(f"[INFO] Matched positions: {matches}")
    print(f"[INFO] No entry order: {no_entry}")
    print(f"[INFO] No exit order: {no_exit}")
    
    return matches


if __name__ == '__main__':
    match_comprehensive()
