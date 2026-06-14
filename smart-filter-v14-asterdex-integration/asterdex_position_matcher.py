#!/usr/bin/env python3
"""
ASTERDEX Position Matcher - Match complete positions (entry → exit)
Instead of matching individual trades, match position lifecycles:
- Entry trade (BUY or SELL at entry price)
- Exit trade (opposite side at TP/SL price)

This is the correct approach for tracking trading signals.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

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


def find_position_exit(entry, trades, time_window_hours=48):
    """
    Find the exit trade for a posted entry position.
    
    A position is closed when:
    - Same symbol, opposite side (LONG/BUY → SHORT/SELL, or vice versa)
    - Exit price at or beyond TP/SL levels
    - Within time window (default 48 hours)
    
    Args:
        entry: Posted entry with symbol, side, entry_price, tp_price, sl_price
        trades: List of all trades
        time_window_hours: Hours to look for exit
    
    Returns:
        (exit_trade, confidence) or (None, 0)
    """
    symbol = entry.get('symbol')
    side = entry.get('side')  # LONG or SHORT
    entry_price = entry.get('entry_price', 0)
    tp_price = entry.get('tp_price') or entry.get('tp', 0)
    sl_price = entry.get('sl_price') or entry.get('sl', 0)
    posted_time = entry.get('posted_timestamp', '2026-06-08T00:00:00Z')
    
    if not symbol or not entry_price:
        return None, 0
    
    try:
        posted_dt = datetime.fromisoformat(posted_time.replace('Z', '+00:00'))
    except:
        return None, 0
    
    # Opposite side for exit
    exit_side = 'SELL' if side == 'LONG' else 'BUY'
    
    candidates = []
    
    for trade in trades:
        # Filter: same symbol
        if trade.get('symbol') != symbol:
            continue
        
        # Filter: opposite side (exit)
        if trade.get('side') != exit_side:
            continue
        
        # Filter: time window
        try:
            trade_dt = datetime.fromtimestamp(trade.get('time', 0) / 1000)
            if not (posted_dt <= trade_dt <= posted_dt + timedelta(hours=time_window_hours)):
                continue
        except:
            continue
        
        exit_price = trade.get('executedPrice', 0)
        
        # Check if exit price is at TP or SL level
        # For LONG: TP should be above entry, SL below
        # For SHORT: TP should be below entry, SL above
        
        if side == 'LONG':
            # TP hit: exit_price >= tp_price
            # SL hit: exit_price <= sl_price
            if (tp_price > 0 and exit_price >= tp_price * 0.99) or \
               (sl_price > 0 and exit_price <= sl_price * 1.01):
                is_tp = exit_price >= tp_price * 0.99 if tp_price > 0 else False
                is_sl = exit_price <= sl_price * 1.01 if sl_price > 0 else False
                
                # Calculate confidence
                time_diff = (trade_dt - posted_dt).total_seconds()
                
                candidates.append({
                    'trade': trade,
                    'is_tp': is_tp,
                    'is_sl': is_sl,
                    'exit_price': exit_price,
                    'time_diff': time_diff,
                    'confidence': 0.95 if is_tp or is_sl else 0.5
                })
        
        elif side == 'SHORT':
            # TP hit: exit_price <= tp_price
            # SL hit: exit_price >= sl_price
            if (tp_price > 0 and exit_price <= tp_price * 1.01) or \
               (sl_price > 0 and exit_price >= sl_price * 0.99):
                is_tp = exit_price <= tp_price * 1.01 if tp_price > 0 else False
                is_sl = exit_price >= sl_price * 0.99 if sl_price > 0 else False
                
                time_diff = (trade_dt - posted_dt).total_seconds()
                
                candidates.append({
                    'trade': trade,
                    'is_tp': is_tp,
                    'is_sl': is_sl,
                    'exit_price': exit_price,
                    'time_diff': time_diff,
                    'confidence': 0.95 if is_tp or is_sl else 0.5
                })
    
    if not candidates:
        return None, 0
    
    # Return best match (highest confidence, shortest time)
    best = max(candidates, key=lambda x: (x['confidence'], -x['time_diff']))
    return best['trade'], best['confidence']


def match_positions():
    """
    Match all posted entries to their exit trades.
    """
    print(f"\n[{datetime.now().isoformat()}] Starting position matching...")
    
    entries = load_posted_entries()
    trades = load_trades()
    already_correlated = load_correlated()
    
    print(f"[INFO] Loaded {len(entries)} posted entries")
    print(f"[INFO] Loaded {len(trades)} trades from history")
    print(f"[INFO] Already correlated: {len(already_correlated)}")
    
    if not trades:
        print("[WARN] No trades available. Skipping matching.")
        return 0
    
    matches_found = 0
    
    with open(CORRELATED_FILE, 'a') as f:
        for entry in entries:
            signal_uuid = entry.get('signal_uuid')
            
            # Skip already matched
            if signal_uuid in already_correlated:
                continue
            
            # Find exit trade
            exit_trade, confidence = find_position_exit(entry, trades)
            
            if exit_trade and confidence > 0.5:
                # Calculate P&L
                entry_price = entry.get('entry_price', 0)
                exit_price = exit_trade.get('executedPrice', 0)
                qty = entry.get('quantity', 0)
                side = entry.get('side')
                
                if side == 'LONG':
                    pnl = (exit_price - entry_price) * qty
                else:  # SHORT
                    pnl = (entry_price - exit_price) * qty
                
                # Determine if TP or SL
                tp_price = entry.get('tp_price') or entry.get('tp', 0)
                sl_price = entry.get('sl_price') or entry.get('sl', 0)
                
                is_tp = False
                if side == 'LONG':
                    is_tp = exit_price >= tp_price * 0.99 if tp_price > 0 else False
                else:
                    is_tp = exit_price <= tp_price * 1.01 if tp_price > 0 else False
                
                exit_type = "TP_HIT" if is_tp else "SL_HIT"
                
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
                    'match_confidence': confidence,
                }
                
                f.write(json.dumps(correlated) + '\n')
                matches_found += 1
    
    print(f"[INFO] New position matches found: {matches_found}")
    print(f"[OK] Position matching complete")
    
    return matches_found


if __name__ == '__main__':
    from datetime import timedelta
    match_positions()
