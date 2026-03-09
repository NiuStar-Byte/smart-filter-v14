#!/usr/bin/env python3
"""
📊 RR 1.5:1 TRACKER - PROPER CUTOFF
Uses ONLY signals fired from 2026-03-07 onwards (matching pec_reporter SUMMARY)

Fresh Data Period:
- 2026-03-07: 14 signals
- 2026-03-08: 172 signals
- 2026-03-09: 39 signals (accumulating)
- Total: 225 signals (clean, no corrupted Mar 3-6 data, no STALE_TIMEOUT)

Excludes:
- Pre-2026-03-07 (contains corrupted data)
- STALE_TIMEOUT (orphaned signals with no valid exit data)
- REJECTED_NOT_SENT_TELEGRAM (never sent to market)
"""

import json
from datetime import datetime
from collections import defaultdict
import os
import sys

workspace = "/Users/geniustarigan/.openclaw/workspace"
signals_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")

# Fresh data cutoff (start from 2026-03-07)
FRESH_START = "2026-03-07T00:00:00"

def load_signals():
    """Load 1.5:1 RR signals fired from 2026-03-07 onwards"""
    signals = []
    
    if not os.path.exists(signals_file):
        return signals
    
    try:
        with open(signals_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    
                    # Filter 1
                    if sig.get('achieved_rr') != 1.5:
                        continue
                    
                    fired = sig.get('fired_time_utc', '')
                    
                    # Filter 2: Only from 2026-03-07 onwards
                    if fired < FRESH_START:
                        continue
                    
                    # Filter 3: Exclude STALE_TIMEOUT (orphaned/rejected)
                    if sig.get('status') == 'STALE_TIMEOUT':
                        continue
                    
                    # Filter 4: Exclude REJECTED_NOT_SENT_TELEGRAM
                    if 'REJECTED' in sig.get('status', ''):
                        continue
                    
                    signals.append(sig)
        
        return signals
    except Exception as e:
        print(f"[ERROR] Loading signals: {e}")
        return signals

def analyze_metrics(signals):
    """Calculate metrics for signal group"""
    if not signals:
        return None
    
    tp_hits = 0
    sl_hits = 0
    timeouts = 0
    open_positions = 0
    total_pnl = 0.0
    
    for sig in signals:
        status = sig.get('status')
        pnl = sig.get('pnl_usd')
        
        if status == 'TP_HIT':
            tp_hits += 1
            if pnl:
                total_pnl += float(pnl)
        elif status == 'SL_HIT':
            sl_hits += 1
            if pnl:
                total_pnl += float(pnl)
        elif status == 'TIMEOUT':
            timeouts += 1
            if pnl:
                total_pnl += float(pnl)
        elif status == 'OPEN':
            open_positions += 1
    
    # Win Rate Calculation
    closed = tp_hits + sl_hits + timeouts
    wr_tp_sl_timeout = (tp_hits / closed * 100) if closed > 0 else 0
    wr_tp_sl_only = (tp_hits / (tp_hits + sl_hits) * 100) if (tp_hits + sl_hits) > 0 else 0
    
    return {
        'total': len(signals),
        'tp': tp_hits,
        'sl': sl_hits,
        'timeout': timeouts,
        'open': open_positions,
        'closed': closed,
        'wr_full': wr_tp_sl_timeout,
        'wr_tp_sl': wr_tp_sl_only,
        'pnl': total_pnl,
        'avg_pnl': total_pnl / len(signals) if len(signals) > 0 else 0
    }

def main():
    print("\n" + "="*100)
    print("📊 RR 1.5:1 TRACKER - PROPER CUTOFF (Fresh Data from 2026-03-07)")
    print("="*100 + "\n")
    
    signals = load_signals()
    
    if not signals:
        print("[ERROR] No signals loaded!")
        return
    
    metrics = analyze_metrics(signals)
    
    print(f"Data Period: 2026-03-07 → 2026-03-09")
    print(f"Total Signals (Fresh): {metrics['total']}")
    print(f"  - TP_HIT: {metrics['tp']}")
    print(f"  - SL_HIT: {metrics['sl']}")
    print(f"  - TIMEOUT: {metrics['timeout']}")
    print(f"  - OPEN: {metrics['open']}")
    print(f"\n")
    
    print(f"Closed Signals: {metrics['closed']} (from {metrics['total']} total)")
    print(f"\n--- WIN RATE CALCULATIONS ---")
    print(f"1. TP+SL+TIMEOUT (full closure): {metrics['tp']} / {metrics['closed']} = {metrics['wr_full']:.1f}%")
    print(f"2. TP+SL only (conservative): {metrics['tp']} / {metrics['tp'] + metrics['sl']} = {metrics['wr_tp_sl']:.1f}%")
    print(f"\n--- PROFIT & LOSS ---")
    print(f"Total P&L: ${metrics['pnl']:,.2f}")
    print(f"Avg P&L per signal: ${metrics['avg_pnl']:.2f}")
    
    # Daily breakdown
    print(f"\n--- DAILY BREAKDOWN ---")
    daily = defaultdict(lambda: {'total': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'open': 0})
    
    for sig in signals:
        date = sig.get('fired_time_utc', '')[:10]  # YYYY-MM-DD
        status = sig.get('status')
        
        daily[date]['total'] += 1
        if status == 'TP_HIT':
            daily[date]['tp'] += 1
        elif status == 'SL_HIT':
            daily[date]['sl'] += 1
        elif status == 'TIMEOUT':
            daily[date]['timeout'] += 1
        elif status == 'OPEN':
            daily[date]['open'] += 1
    
    for date in sorted(daily.keys()):
        d = daily[date]
        closed = d['tp'] + d['sl'] + d['timeout']
        wr = (d['tp'] / closed * 100) if closed > 0 else 0
        print(f"  {date}: Total={d['total']} | TP={d['tp']} | SL={d['sl']} | TIMEOUT={d['timeout']} | OPEN={d['open']} | WR={wr:.1f}%")
    
    print("\n" + "="*100 + "\n")

if __name__ == '__main__':
    main()
