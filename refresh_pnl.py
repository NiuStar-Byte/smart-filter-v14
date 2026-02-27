#!/usr/bin/env python3
"""
Refresh P&L calculations for all closed signals with FIXED position size (1.0)
Recalculates pnl_usd for all TP_HIT/SL_HIT/TIMEOUT signals
"""

import json
import os
from datetime import datetime

def refresh_pnl(signals_path: str = "SENT_SIGNALS.jsonl"):
    """Recalculate P&L with fixed position size ($100 @ 10x leverage = $1,000 notional) for all closed trades"""
    
    NOTIONAL_POSITION = 1000.0  # $100 position × 10x leverage
    
    if not os.path.exists(signals_path):
        print(f"❌ File not found: {signals_path}")
        return
    
    updated_count = 0
    stats_before = {'tp_hit': 0, 'sl_hit': 0, 'total_pnl_before': 0.0}
    stats_after = {'tp_hit': 0, 'sl_hit': 0, 'total_pnl_after': 0.0}
    
    # Read all records
    records = []
    try:
        with open(signals_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records.append(record)
    except Exception as e:
        print(f"❌ Failed to read {signals_path}: {e}")
        return
    
    print(f"📋 Processing {len(records)} signals...\n")
    
    # Recalculate P&L for closed signals
    for record in records:
        status = record.get('status', 'OPEN')
        
        # Only process closed trades
        if status not in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
            continue
        
        entry_price = record.get('entry_price')
        actual_exit_price = record.get('actual_exit_price')
        signal_type = record.get('signal_type')
        
        if not all([entry_price, actual_exit_price, signal_type]):
            continue
        
        # Track before
        pnl_before = record.get('pnl_usd', 0)
        if status == 'TP_HIT':
            stats_before['tp_hit'] += 1
        elif status == 'SL_HIT':
            stats_before['sl_hit'] += 1
        stats_before['total_pnl_before'] += pnl_before if pnl_before else 0
        
        # Recalculate with fixed position size ($1000 @ 10x = $10,000 notional)
        # Formula: (price_move / entry_price) * notional_position
        if signal_type == 'LONG':
            pnl_usd = ((actual_exit_price - entry_price) / entry_price) * NOTIONAL_POSITION
        else:  # SHORT
            pnl_usd = ((entry_price - actual_exit_price) / entry_price) * NOTIONAL_POSITION
        
        # Update record
        record['pnl_usd'] = round(pnl_usd, 4)
        updated_count += 1
        
        # Track after
        if status == 'TP_HIT':
            stats_after['tp_hit'] += 1
        elif status == 'SL_HIT':
            stats_after['sl_hit'] += 1
        stats_after['total_pnl_after'] += record['pnl_usd']
    
    # Write back updated records
    try:
        with open(signals_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        print(f"✅ Updated {updated_count} closed signals with correct P&L (leverage-adjusted)")
    except Exception as e:
        print(f"❌ Failed to write {signals_path}: {e}")
        return
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 P&L REFRESH SUMMARY (NOTIONAL = $1000: $100 position × 10x leverage)")
    print(f"{'='*70}")
    print(f"\nBEFORE REFRESH:")
    print(f"  TP Hits: {stats_before['tp_hit']}")
    print(f"  SL Hits: {stats_before['sl_hit']}")
    print(f"  Total P&L: ${stats_before['total_pnl_before']:+.4f}")
    print(f"\nAFTER REFRESH:")
    print(f"  TP Hits: {stats_after['tp_hit']}")
    print(f"  SL Hits: {stats_after['sl_hit']}")
    print(f"  Total P&L: ${stats_after['total_pnl_after']:+.4f}")
    print(f"\nWin Rate: {stats_after['tp_hit']}/{stats_after['tp_hit'] + stats_after['sl_hit']} = {100*stats_after['tp_hit']/(stats_after['tp_hit']+stats_after['sl_hit']):.1f}%")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    refresh_pnl()
