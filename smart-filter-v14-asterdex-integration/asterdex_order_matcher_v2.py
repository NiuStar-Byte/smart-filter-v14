#!/usr/bin/env python3
"""
ASTERDEX Order ID Matcher v2 - Match posted entries with actual filled orders
Works with /fapi/v3/allOrders endpoint data

Strategy:
- Posted entries have ORDER IDs (entry_order_id, tp_order_id, sl_order_id)
- Filled orders have ORDER IDs in the API response
- Match by ORDER ID directly → lookup trade data
- Calculate P&L from cumQuote (cost for entry, revenue for exit)
"""

import json
from datetime import datetime
from pathlib import Path

POSTED_ENTRIES_FILE = Path(__file__).parent / "ASTERDEX_POSTED_ENTRIES.jsonl"
ORDERS_FILE = Path(__file__).parent / "ASTERDEX_TRADES.jsonl"  # Actually /allOrders response
CORRELATED_FILE = Path(__file__).parent / "ASTERDEX_PERFORMANCE_CORRELATED.jsonl"


def load_posted_entries():
    """Load all posted entries."""
    entries = []
    if POSTED_ENTRIES_FILE.exists():
        with open(POSTED_ENTRIES_FILE, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                try:
                    entries.append(json.loads(line))
                except:
                    pass
    return entries


def load_orders():
    """Load all filled orders from /allOrders endpoint."""
    orders = []
    if ORDERS_FILE.exists():
        with open(ORDERS_FILE, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
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
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    correlated.add(entry.get('signal_uuid'))
                except:
                    pass
    return correlated


def build_order_index(orders):
    """
    Build index of orders by ORDER ID for fast lookup.
    
    Returns:
        Dict: {orderId: order}
    """
    index = {}
    for order in orders:
        order_id = order.get('orderId')
        if order_id:
            index[order_id] = order
    return index


def match_positions_by_order_id(start_date='2026-06-07T00:00:01+07:00'):
    """
    Match all posted entries to their actual filled orders using ORDER IDs.
    
    Logic:
    - Entry posted with entry_order_id → lookup in order history
    - If found, calculate P&L from entry and exit orders cumQuote
    - Exit = either tp_order_id or sl_order_id hit (opposite side, reduceOnly=true)
    - Only match entries posted from start_date onwards (clean baseline tracking)
    """
    print(f"\n[{datetime.now().isoformat()}] Starting ORDER ID-based position matching...")
    print(f"[INFO] Filtering entries from {start_date} onwards")
    
    entries = load_posted_entries()
    orders = load_orders()
    already_correlated = load_correlated()
    
    # Parse start date
    try:
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    except:
        start_dt = datetime(2026, 6, 7, 0, 0, 1)
    
    # Filter entries to only those posted from start_date onwards
    filtered_entries = []
    for entry in entries:
        try:
            entry_dt = datetime.fromisoformat(entry.get('posted_timestamp', '').replace('Z', '+00:00'))
            if entry_dt >= start_dt:
                filtered_entries.append(entry)
        except:
            pass
    
    entries = filtered_entries
    
    print(f"[INFO] Loaded {len(entries)} posted entries (from {start_date} onwards)")
    print(f"[INFO] Loaded {len(orders)} filled orders")
    print(f"[INFO] Already correlated: {len(already_correlated)}")
    
    if not orders:
        print("[WARN] No orders available. Skipping matching.")
        return 0
    
    # Build order index for fast lookup
    order_index = build_order_index(orders)
    print(f"[INFO] Built order index with {len(order_index)} unique order IDs")
    
    matches_found = 0
    no_entry_order = 0
    no_exit_order = 0
    
    with open(CORRELATED_FILE, 'a') as f:
        for entry in entries:
            signal_uuid = entry.get('signal_uuid')
            
            # Skip already matched
            if signal_uuid in already_correlated:
                continue
            
            # Find entry order using ORDER ID
            entry_order_id = entry.get('entry_order_id')
            if not entry_order_id or entry_order_id not in order_index:
                no_entry_order += 1
                continue
            
            entry_order = order_index[entry_order_id]
            
            # Find exit order by matching rules:
            # - Opposite side (LONG entry = SELL exit, SHORT entry = BUY exit)
            # - Same symbol
            # - reduceOnly = true (closing order)
            # - Timestamp > entry timestamp
            exit_order = None
            exit_type = None
            
            entry_side = entry.get('side')
            exit_side = 'SELL' if entry_side == 'LONG' else 'BUY'
            entry_symbol = entry.get('symbol')
            entry_qty = float(entry.get('quantity', 0))
            entry_time = entry_order.get('time', 0)
            
            # Search all orders for matching exit
            for order in orders:
                # Must be opposite side, same symbol, closing order
                if (order.get('side') == exit_side and 
                    order.get('symbol') == entry_symbol and
                    order.get('reduceOnly') == True and
                    order.get('time', 0) > entry_time):
                    
                    # Check if quantity roughly matches (within 10%)
                    order_qty = float(order.get('executedQty', 0))
                    qty_match = abs(order_qty - entry_qty) / entry_qty if entry_qty else 0
                    
                    if qty_match <= 0.1:  # Within 10% quantity tolerance
                        exit_order = order
                        # Determine if TP or SL (we don't have exact target, so just mark as UNKNOWN)
                        exit_type = "EXIT"
                        break
            
            if not exit_order:
                no_exit_order += 1
                continue
            
            # Calculate P&L from cumQuote values
            # cumQuote = total cost (for BUY) or revenue (for SELL)
            entry_cost = float(entry_order.get('cumQuote', 0))
            exit_revenue = float(exit_order.get('cumQuote', 0))
            # Estimate total commission as 0.02% of volume (entry + exit)
            total_commission = (entry_cost + exit_revenue) * 0.0002
            
            side = entry.get('side')
            if side == 'LONG':
                # Entry: BUY (cost money), Exit: SELL (receive money)
                pnl = exit_revenue - entry_cost - total_commission
            else:  # SHORT
                # Entry: SELL (receive money), Exit: BUY (cost money)
                pnl = entry_cost - exit_revenue - total_commission
            
            # Build correlated record
            correlated = {
                'signal_uuid': signal_uuid,
                'symbol': entry.get('symbol'),
                'tier': entry.get('tier'),
                'mtf_alignment_band': entry.get('mtf_alignment_band'),
                'route': entry.get('route'),
                'timeframe': entry.get('timeframe'),
                'confidence_level': entry.get('confidence_level'),
                'side': side,
                'entry_price': entry.get('entry_price'),
                'exit_price': float(exit_order.get('avgPrice', 0)),
                'quantity': entry.get('quantity'),
                'posted_timestamp': entry.get('posted_timestamp'),
                'exit_timestamp': exit_order.get('time'),
                'realized_pnl_usd': pnl,
                'exit_type': exit_type,
                'win': pnl > 0,
                # Order IDs - proof of matching
                'entry_order_id': entry_order_id,
                'exit_order_id': exit_order.get('orderId'),
                'match_method': 'order_id_with_exit_inference',
            }
            
            f.write(json.dumps(correlated) + '\n')
            matches_found += 1
    
    print(f"[INFO] Matched positions: {matches_found}")
    print(f"[INFO] No entry order found: {no_entry_order}")
    print(f"[INFO] Entries without exit (still open): {no_exit_order}")
    print(f"[OK] Order ID matching complete")
    
    return matches_found


if __name__ == '__main__':
    match_positions_by_order_id()
