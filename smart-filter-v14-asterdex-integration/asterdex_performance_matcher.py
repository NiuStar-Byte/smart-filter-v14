#!/usr/bin/env python3
"""
ASTERDEX Performance Matcher - Correlate Posted Entries ↔ Executed Trades
Matches entries without using UUID (not available in Asterdex).
Matching strategy: symbol + side + time window + price proximity.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def load_cached_trades():
    """Load all cached trades."""
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
    """Load already correlated entries."""
    correlated = set()
    if CORRELATED_FILE.exists():
        with open(CORRELATED_FILE, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    correlated.add(entry['signal_uuid'])
                except:
                    pass
    return correlated


def parse_timestamp(ts_str):
    """Parse ISO 8601 timestamp to datetime."""
    try:
        if isinstance(ts_str, (int, float)):
            return datetime.fromtimestamp(ts_str / 1000)
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except:
        return None


def match_entry_to_trade(entry, trades, time_window_min=10, price_tolerance=0.02):
    """
    Find best matching trade for a posted entry.
    
    Matching criteria:
    1. Symbol (exact)
    2. Side (SHORT → SELL, LONG → BUY)
    3. Time (±time_window_min)
    4. Price (±price_tolerance %)
    
    Returns: (matched_trade, confidence_score) or (None, 0)
    """
    
    symbol = entry['symbol']
    side = entry['side']
    posted_price = entry['entry_price']
    posted_time = parse_timestamp(entry['posted_timestamp'])
    
    if not posted_time:
        return None, 0
    
    # Convert side: LONG→BUY, SHORT→SELL for Asterdex
    expected_side = 'BUY' if side == 'LONG' else 'SELL'
    
    # Time window
    time_min = posted_time - timedelta(minutes=time_window_min)
    time_max = posted_time + timedelta(minutes=time_window_min)
    
    candidates = []
    
    for trade in trades:
        # Filter by symbol
        if trade['symbol'] != symbol:
            continue
        
        # Filter by side
        if trade['side'] != expected_side:
            continue
        
        # Filter by time
        trade_time = parse_timestamp(trade['time'])
        if not trade_time or not (time_min <= trade_time <= time_max):
            continue
        
        # Filter by price proximity
        executed_price = trade['executedPrice']
        price_diff_pct = abs(executed_price - posted_price) / posted_price if posted_price else 1
        
        if price_diff_pct > price_tolerance:
            continue
        
        # Calculate match confidence
        time_diff_sec = abs((trade_time - posted_time).total_seconds())
        time_score = 1.0 - (time_diff_sec / 600)  # 600 sec = 10 min window
        price_score = 1.0 - (price_diff_pct / price_tolerance)
        confidence = (time_score + price_score) / 2
        
        candidates.append((trade, confidence))
    
    if not candidates:
        return None, 0
    
    # Return best match (highest confidence)
    return max(candidates, key=lambda x: x[1])


def correlate_entries_to_trades():
    """
    Main correlation logic.
    Match each unmatched posted entry to executed trades.
    """
    
    print(f"\n[{datetime.now().isoformat()}] Starting correlation...")
    
    posted_entries = load_posted_entries()
    trades = load_cached_trades()
    already_correlated = load_correlated()
    
    print(f"[INFO] Loaded {len(posted_entries)} posted entries")
    print(f"[INFO] Loaded {len(trades)} cached trades")
    print(f"[INFO] Already correlated: {len(already_correlated)}")
    
    if not trades:
        print("[WARN] No trades available yet. Skipping correlation.")
        return
    
    new_correlations = 0
    matches_found = 0
    no_matches = 0
    
    with open(CORRELATED_FILE, 'a') as f:
        for entry in posted_entries:
            signal_uuid = entry['signal_uuid']
            
            # Skip already correlated
            if signal_uuid in already_correlated:
                continue
            
            # Try to match
            matched_trade, confidence = match_entry_to_trade(entry, trades)
            
            if matched_trade and confidence > 0.5:
                # Build correlated record
                correlated = {
                    'signal_uuid': signal_uuid,
                    'symbol': entry['symbol'],
                    'tier': entry.get('tier'),
                    'mtf_alignment_band': entry.get('mtf_alignment_band'),
                    'route': entry.get('route'),
                    'timeframe': entry.get('timeframe'),
                    'confidence_level': entry.get('confidence_level'),
                    'side': entry['side'],
                    'posted_price': entry['entry_price'],
                    'executed_price': matched_trade['executedPrice'],
                    'quantity': entry['quantity'],
                    'posted_timestamp': entry['posted_timestamp'],
                    'executed_timestamp': matched_trade['time'],
                    'realized_pnl_usd': matched_trade.get('cumQuote', 0),  # Placeholder
                    'match_confidence': confidence,
                    'match_method': 'symbol_side_time_price',
                    'order_id': matched_trade.get('orderId'),
                    'status': 'MATCHED',
                }
                
                f.write(json.dumps(correlated) + '\n')
                matches_found += 1
                new_correlations += 1
            else:
                no_matches += 1
    
    print(f"[INFO] New matches: {matches_found}")
    print(f"[INFO] No matches found: {no_matches}")
    print(f"[OK] Correlation complete")
    
    return new_correlations


def main():
    """Main entry point."""
    correlate_entries_to_trades()


if __name__ == '__main__':
    main()
