#!/usr/bin/env python3
"""
ASTERDEX Order ID Matcher - Match positions using ORDER ID
This is the correct approach - ORDER ID is the unique identifier from Asterdex
(just like signal_uuid is the unique identifier for signals).

Every posted entry generates ORDER IDs for:
- Entry order (LIMIT BUY or SELL)
- TP order (TAKE_PROFIT_MARKET)
- SL order (STOP_LOSS_MARKET)

These ORDER IDs directly appear in trade history, making matching trivial.
"""

import json
from datetime import datetime
from pathlib import Path

# File locations
POSTED_ENTRIES_FILE = Path(__file__).parent / "ASTERDEX_POSTED_ENTRIES.jsonl"
TRADES_FILE = Path(__file__).parent / "ASTERDEX_TRADES.jsonl"
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


def load_trades():
    """Load all trades from Asterdex history."""
    trades = []
    if TRADES_FILE.exists():
        with open(TRADES_FILE, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    trades.append(json.loads(line))
                except:
                    pass
    return trades


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


def build_trade_index(trades):
    """
    Build index of trades by ORDER ID for fast lookup.
    
    Returns:
        Dict: {order_id: trade}
    """
    index = {}
    for trade in trades:
        order_id = trade.get('orderId')  # ← Correct field name from Asterdex API
        if order_id:
            index[order_id] = trade
    return index


def match_positions_by_order_id():
    """
    Match all posted entries to their trades using ORDER IDs.
    
    This is the correct approach:
    - Entry order → Entry trade (ORDER ID match)
    - TP order → Exit trade at TP level (ORDER ID match)
    - SL order → Exit trade at SL level (ORDER ID match)
    
    No guessing, no time windows, no price matching needed.
    """
    print(f"\n[{datetime.now().isoformat()}] Starting ORDER ID-based position matching...")
    
    entries = load_posted_entries()
    trades = load_trades()
    already_correlated = load_correlated()
    
    print(f"[INFO] Loaded {len(entries)} posted entries")
    print(f"[INFO] Loaded {len(trades)} trades from history")
    print(f"[INFO] Already correlated: {len(already_correlated)}")
    
    if not trades:
        print("[WARN] No trades available. Skipping matching.")
        return 0
    
    # Build trade index for fast lookup
    trade_index = build_trade_index(trades)
    print(f"[INFO] Built trade index with {len(trade_index)} unique order IDs")
    
    matches_found = 0
    no_entry_order = 0
    no_exit_order = 0
    
    with open(CORRELATED_FILE, 'a') as f:
        for entry in entries:
            signal_uuid = entry.get('signal_uuid')
            
            # Skip already matched
            if signal_uuid in already_correlated:
                continue
            
            # Find entry trade using ORDER ID
            entry_order_id = entry.get('entry_order_id')
            if not entry_order_id or entry_order_id not in trade_index:
                no_entry_order += 1
                continue
            
            entry_trade = trade_index[entry_order_id]
            
            # Find exit trade using TP or SL ORDER ID (whichever exists)
            exit_trade = None
            exit_type = None
            
            tp_order_id = entry.get('tp_order_id')
            sl_order_id = entry.get('sl_order_id')
            
            if tp_order_id and tp_order_id in trade_index:
                exit_trade = trade_index[tp_order_id]
                exit_type = "TP_HIT"
            elif sl_order_id and sl_order_id in trade_index:
                exit_trade = trade_index[sl_order_id]
                exit_type = "SL_HIT"
            
            if not exit_trade:
                no_exit_order += 1
                continue
            
            # Calculate P&L
            entry_price = entry.get('entry_price', 0)
            exit_price = float(exit_trade.get('executedPrice', 0))  # Asterdex API uses 'executedPrice'
            qty = entry.get('quantity', 0)
            side = entry.get('side')
            
            if side == 'LONG':
                pnl = (exit_price - entry_price) * qty
            else:  # SHORT
                pnl = (entry_price - exit_price) * qty
            
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
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': qty,
                'posted_timestamp': entry.get('posted_timestamp'),
                'exit_timestamp': exit_trade.get('time'),
                'realized_pnl_usd': pnl,
                'exit_type': exit_type,
                'win': pnl > 0,
                # Trade IDs - proof of exact matching
                'entry_order_id': entry_order_id,
                'exit_trade_id': tp_order_id if exit_type == "TP_HIT" else sl_order_id,
                'match_method': 'trade_id_exact',
            }
            
            f.write(json.dumps(correlated) + '\n')
            matches_found += 1
    
    print(f"[INFO] Position matches found: {matches_found}")
    print(f"[INFO] Entries without order: {no_entry_order}")
    print(f"[INFO] Entries without exit (still open): {no_exit_order}")
    print(f"[OK] Order ID matching complete")
    
    return matches_found


if __name__ == '__main__':
    match_positions_by_order_id()
