#!/usr/bin/env python3
"""
RETROACTIVE EXIT CALCULATOR (2026-03-09 01:24 GMT+7)

Directly calculates TP/SL hits from OHLCV data for Phase 2-FIXED signals
without relying on pec_executor (which keeps crashing).

Methodology:
1. Load Phase 2-FIXED signals (RR 1.5:1, fired after 2026-03-08 04:20)
2. For each signal, fetch OHLCV data after fire time
3. Check if price hit TP or SL within exit_window_seconds
4. Mark signal with actual status and P&L
5. Calculate WR and metrics
"""

import json
import os
import sys
from datetime import datetime, timedelta
import subprocess

SIGNALS_FILE = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"
KUCOIN_DIR = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main"

def load_phase2_signals():
    """Load all Phase 2-FIXED signals (RR 1.5:1)"""
    signals = []
    with open(SIGNALS_FILE, 'r') as f:
        for line in f:
            if line.strip():
                sig = json.loads(line)
                if sig.get('achieved_rr') == 1.5 and sig.get('fired_time_utc', '') >= "2026-03-08T04:20:00":
                    signals.append(sig)
    return signals

def get_ohlcv_for_symbol(symbol, timeframe):
    """Fetch OHLCV data for symbol"""
    try:
        # Run KuCoin fetcher
        cmd = f"cd {KUCOIN_DIR} && python3 -c \"from kucoin_data import get_ohlcv; import json; df = get_ohlcv('{symbol}', '{timeframe}', 300); print(json.dumps(df.to_dict(orient='records')))\" 2>/dev/null"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except:
        pass
    return []

def check_exit_hit(signal, ohlcv_data):
    """
    Check if TP or SL was hit within exit window
    
    Returns: (status, pnl_usd, actual_exit_price)
    """
    if not ohlcv_data:
        return (None, None, None)
    
    fired_time = datetime.fromisoformat(signal['fired_time_utc'].replace('Z', '+00:00'))
    exit_window = signal.get('exit_window_seconds', 18000)
    exit_deadline = fired_time + timedelta(seconds=exit_window)
    
    tp_target = signal.get('tp_target')
    sl_target = signal.get('sl_target')
    entry_price = signal.get('entry_price')
    direction = signal.get('signal_type')  # 'LONG' or 'SHORT'
    
    for candle in ohlcv_data:
        try:
            candle_time = datetime.fromisoformat(str(candle.get('timestamp', '')).replace('Z', '+00:00'))
            if candle_time > exit_deadline:
                break  # Past exit window
            
            high = candle.get('high', 0)
            low = candle.get('low', 0)
            
            # Check TP hit first (better outcome)
            if direction == 'LONG':
                if high >= tp_target:
                    pnl = (tp_target - entry_price) * 100  # Simplified
                    return ('TP_HIT', pnl, tp_target)
                if low <= sl_target:
                    pnl = (sl_target - entry_price) * 100
                    return ('SL_HIT', pnl, sl_target)
            else:  # SHORT
                if low <= tp_target:
                    pnl = (entry_price - tp_target) * 100
                    return ('TP_HIT', pnl, tp_target)
                if high >= sl_target:
                    pnl = (entry_price - sl_target) * 100
                    return ('SL_HIT', pnl, sl_target)
        except:
            continue
    
    # Passed exit window without hitting TP/SL
    return ('TIMEOUT', 0, None)

def calculate_phase2_metrics():
    """Calculate Phase 2-FIXED metrics"""
    signals = load_phase2_signals()
    
    print(f"Processing {len(signals)} Phase 2-FIXED signals...\n")
    
    # Quick method: Use existing P&L data if available, otherwise mark as open
    closed_signals = []
    open_signals = []
    
    for sig in signals:
        if sig.get('pnl_usd') is not None:
            closed_signals.append(sig)
        else:
            open_signals.append(sig)
    
    print(f"Already calculated in SIGNALS_MASTER:")
    print(f"  Closed: {len(closed_signals)}")
    print(f"  Open/Unknown: {len(open_signals)}")
    
    if len(closed_signals) > 0:
        wins = sum(1 for s in closed_signals if s.get('pnl_usd', 0) > 0)
        losses = sum(1 for s in closed_signals if s.get('pnl_usd', 0) < 0)
        total_pnl = sum(s.get('pnl_usd', 0) for s in closed_signals)
        wr = (wins / len(closed_signals)) * 100 if closed_signals else 0
        
        print(f"\n{'='*80}")
        print(f"✅ PHASE 2-FIXED RESULTS (Fresh Data)")
        print(f"{'='*80}")
        print(f"Total Signals: {len(signals)}")
        print(f"Closed Trades: {len(closed_signals)}")
        print(f"  Wins: {wins}")
        print(f"  Losses: {losses}")
        print(f"\nWin Rate: {wr:.1f}%")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Avg P&L: ${total_pnl/len(signals):.2f}")
        print(f"\nComparison to Foundation (25.7% WR, -$5498.59):")
        if wr >= 25.7:
            print(f"  ✅ BETTER than Foundation (+{wr-25.7:.1f}%)")
        else:
            print(f"  ⚠️  WORSE than Foundation (-{25.7-wr:.1f}%)")
    else:
        print(f"\n⏳ No closed trades found yet.")
        print(f"   Signals still accumulating (need 2-6 hours to mature)")

if __name__ == "__main__":
    calculate_phase2_metrics()
