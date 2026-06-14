#!/usr/bin/env python3
"""
ASTERDEX Simple Position Matcher - Match entries to trades by characteristics
Uses symbol + side + price + time (practical approach)

Instead of trying to match ORDER IDs (which may not correlate to trade IDs),
match by the actual trade characteristics: symbol, side, entry price, exit price.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

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
    """Load all trades."""
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


def find_entry_trade(entry, trades, time_tolerance_min=5):
    """
    Find the entry trade (LIMIT order fill) for a posted entry.
    
    Match by:
    - Symbol (exact)
    - Side (exact: BUY for LONG, SELL for SHORT)
    - Entry price (within 0.5%)
    - Time (within time_tolerance_min)
    """
    symbol = entry.get('symbol', '')
    side = entry.get('side')
    entry_price = entry.get('entry_price', 0)
    posted_time_str = entry.get('posted_timestamp', '')
    
    if not symbol or not entry_price:
        return None, 0
    
    # Convert side: LONG → BUY, SHORT → SELL
    expected_side = 'BUY' if side == 'LONG' else 'SELL'
    
    # Parse posted time (convert to naive UTC for comparison)
    try:
        posted_dt = datetime.fromisoformat(posted_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
    except:
        return None, 0
    
    time_min = posted_dt - timedelta(minutes=time_tolerance_min)
    time_max = posted_dt + timedelta(minutes=time_tolerance_min)
    
    candidates = []
    
    for trade in trades:
        # Match symbol (format: BTCUSDT from API, but we stored as BTC-USDT)
        trade_symbol = trade.get('symbol', '')
        if trade_symbol != symbol and trade_symbol.replace('-', '') != symbol.replace('-', ''):
            continue
        
        # Match side
        if trade.get('side') != expected_side:
            continue
        
        # Match price (within 0.5%)
        trade_price = float(trade.get('price', 0))
        price_diff_pct = abs(trade_price - entry_price) / entry_price if entry_price else 1
        if price_diff_pct > 0.005:  # 0.5% tolerance
            continue
        
        # Match time
        try:
            trade_time = datetime.fromtimestamp(trade.get('time', 0) / 1000)
            if not (time_min <= trade_time <= time_max):
                continue
        except:
            continue
        
        # Confidence based on price match
        price_confidence = 1.0 - (price_diff_pct / 0.005)
        candidates.append((trade, price_confidence))
    
    if not candidates:
        return None, 0
    
    # Return best match (highest confidence)
    return max(candidates, key=lambda x: x[1])


def find_exit_trade(entry, entry_trade, trades, time_window_hours=48):
    """
    Find the exit trade (TP or SL hit) for a position.
    
    Match by:
    - Symbol (exact)
    - Side (opposite of entry)
    - Exit price near TP/SL
    - Time (within time_window_hours after entry)
    """
    if not entry_trade:
        return None, None, 0
    
    symbol = entry.get('symbol', '')
    side = entry.get('side')
    tp_price = entry.get('tp_price') or entry.get('tp', 0)
    sl_price = entry.get('sl_price') or entry.get('sl', 0)
    entry_time_str = entry.get('posted_timestamp', '')
    
    if not symbol or (not tp_price and not sl_price):
        return None, None, 0
    
    # Opposite side for exit
    exit_side = 'SELL' if side == 'LONG' else 'BUY'
    
    # Parse entry time (convert to naive UTC for comparison)
    try:
        entry_dt = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
    except:
        return None, None, 0
    
    time_max = entry_dt + timedelta(hours=time_window_hours)
    
    candidates = []
    
    for trade in trades:
        # Match symbol
        trade_symbol = trade.get('symbol', '')
        if trade_symbol != symbol and trade_symbol.replace('-', '') != symbol.replace('-', ''):
            continue
        
        # Match side (exit)
        if trade.get('side') != exit_side:
            continue
        
        # Match time window
        try:
            trade_time = datetime.fromtimestamp(trade.get('time', 0) / 1000)
            if not (entry_dt <= trade_time <= time_max):
                continue
        except:
            continue
        
        exit_price = float(trade.get('price', 0))
        is_tp = False
        is_sl = False
        
        # Check if price is at TP or SL level (within 1%)
        if side == 'LONG':
            if tp_price and abs(exit_price - tp_price) / tp_price <= 0.01:
                is_tp = True
            if sl_price and abs(exit_price - sl_price) / sl_price <= 0.01:
                is_sl = True
        else:  # SHORT
            if tp_price and abs(exit_price - tp_price) / tp_price <= 0.01:
                is_tp = True
            if sl_price and abs(exit_price - sl_price) / sl_price <= 0.01:
                is_sl = True
        
        if is_tp or is_sl:
            exit_type = "TP_HIT" if is_tp else "SL_HIT"
            time_diff = (trade_time - entry_dt).total_seconds()
            confidence = 0.95 if is_tp or is_sl else 0.5
            candidates.append((trade, exit_type, confidence))
    
    if not candidates:
        return None, None, 0
    
    # Return best match (highest confidence)
    best = max(candidates, key=lambda x: x[2])
    return best[0], best[1], best[2]


def match_positions():
    """Match all posted entries to their trades."""
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
    no_entry_trade = 0
    no_exit_trade = 0
    
    with open(CORRELATED_FILE, 'a') as f:
        for entry in entries:
            signal_uuid = entry.get('signal_uuid')
            
            # Skip already matched
            if signal_uuid in already_correlated:
                continue
            
            # Find entry trade
            entry_trade, entry_confidence = find_entry_trade(entry, trades)
            
            if not entry_trade or entry_confidence < 0.7:
                no_entry_trade += 1
                continue
            
            # Find exit trade
            exit_trade, exit_type, exit_confidence = find_exit_trade(entry, entry_trade, trades)
            
            if not exit_trade:
                no_exit_trade += 1
                continue
            
            # Calculate P&L
            entry_price = entry.get('entry_price', 0)
            exit_price = float(exit_trade.get('price', 0))
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
                'match_confidence': min(entry_confidence, exit_confidence),
                'match_method': 'symbol_side_price_time',
            }
            
            f.write(json.dumps(correlated) + '\n')
            matches_found += 1
    
    print(f"[INFO] Matched positions: {matches_found}")
    print(f"[INFO] No entry trade found: {no_entry_trade}")
    print(f"[INFO] Entries without exit (still open): {no_exit_trade}")
    print(f"[OK] Position matching complete")
    
    return matches_found


if __name__ == '__main__':
    match_positions()
