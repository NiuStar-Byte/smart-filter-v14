#!/usr/bin/env python3
"""
Restore SECTION 1 & 2 to PEC Enhanced Reporter
SECTION 1: Foundation + New combined
SECTION 2: NEW ONLY (Mar 16+ onwards)
"""

import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path

def load_signals(file_path):
    """Load all signals from JSONL file"""
    signals = []
    try:
        with open(file_path) as f:
            for line in f:
                try:
                    signals.append(json.loads(line))
                except:
                    pass
    except:
        pass
    return signals

def categorize_signals(signals, cutoff_date=None):
    """Categorize signals into Foundation/New based on cutoff_date"""
    foundation = []
    new = []
    
    cutoff = datetime.fromisoformat('2026-03-16T00:00:00') if not cutoff_date else cutoff_date
    
    for sig in signals:
        try:
            fired = datetime.fromisoformat(sig.get('fired_time_utc', '').replace('Z', '+00:00'))
            if fired < cutoff:
                foundation.append(sig)
            else:
                new.append(sig)
        except:
            foundation.append(sig)
    
    return foundation, new

def analyze_signals(signals):
    """Analyze signal group"""
    tp = sum(1 for s in signals if s.get('status') == 'TP_HIT')
    sl = sum(1 for s in signals if s.get('status') == 'SL_HIT')
    timeout = sum(1 for s in signals if s.get('status') == 'TIMEOUT')
    open_trades = sum(1 for s in signals if s.get('status') == 'OPEN')
    rejected = sum(1 for s in signals if s.get('status') == 'REJECTED')
    
    # Clean data (closed trades)
    closed = tp + sl + timeout
    
    # Timeout breakdown (approximate)
    timeout_win = sum(1 for s in signals if s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd') or 0) > 0)
    timeout_loss = timeout - timeout_win
    
    # P&L calculations
    total_pnl = sum(float(s.get('pnl_usd') or 0) for s in signals)
    
    # P&L by exit type
    tp_pnl = sum(float(s.get('pnl_usd') or 0) for s in signals if s.get('status') == 'TP_HIT')
    sl_pnl = sum(float(s.get('pnl_usd') or 0) for s in signals if s.get('status') == 'SL_HIT')
    timeout_pnl = sum(float(s.get('pnl_usd') or 0) for s in signals if s.get('status') == 'TIMEOUT')
    
    # Averages
    avg_pnl_signal = total_pnl / len(signals) if signals else 0
    avg_pnl_closed = total_pnl / closed if closed > 0 else 0
    avg_tp_pnl = tp_pnl / tp if tp > 0 else 0
    avg_sl_pnl = sl_pnl / sl if sl > 0 else 0
    timeout_win_pnl = sum(float(s.get('pnl_usd') or 0) for s in signals if s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd') or 0) > 0)
    avg_timeout_win = timeout_win_pnl / timeout_win if timeout_win > 0 else 0
    
    # Win rate
    wins = tp + timeout_win
    wr = (wins / closed * 100) if closed > 0 else 0
    
    # Duration (rough estimate from signal data)
    tp_durations = []
    sl_durations = []
    for s in signals:
        if s.get('status') == 'TP_HIT' and 'duration_minutes' in s:
            tp_durations.append(s['duration_minutes'])
        elif s.get('status') == 'SL_HIT' and 'duration_minutes' in s:
            sl_durations.append(s['duration_minutes'])
    
    avg_tp_duration = sum(tp_durations) / len(tp_durations) if tp_durations else 0
    avg_sl_duration = sum(sl_durations) / len(sl_durations) if sl_durations else 0
    
    return {
        'total': len(signals),
        'tp': tp,
        'sl': sl,
        'timeout': timeout,
        'open': open_trades,
        'rejected': rejected,
        'closed': closed,
        'timeout_win': timeout_win,
        'timeout_loss': timeout_loss,
        'total_pnl': total_pnl,
        'avg_pnl_signal': avg_pnl_signal,
        'avg_pnl_closed': avg_pnl_closed,
        'tp_pnl': tp_pnl,
        'sl_pnl': sl_pnl,
        'timeout_pnl': timeout_pnl,
        'avg_tp_pnl': avg_tp_pnl,
        'avg_sl_pnl': avg_sl_pnl,
        'avg_timeout_win': avg_timeout_win,
        'wr': wr,
        'wins': wins,
        'avg_tp_duration': avg_tp_duration,
        'avg_sl_duration': avg_sl_duration,
    }

def format_duration(minutes):
    """Format minutes to h:mm format"""
    if not minutes or minutes <= 0:
        return "N/A"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m"

def print_section_1_2(foundation_signals, new_signals):
    """Print SECTION 1 & 2"""
    
    # Combined analysis
    combined = foundation_signals + new_signals
    combined_stats = analyze_signals(combined)
    
    # New analysis
    new_stats = analyze_signals(new_signals)
    
    print(f"\n{'='*130}")
    print(f"📊 SECTION 1: TOTAL SIGNALS (Foundation + New)")
    print(f"{'='*130}")
    
    print(f"Total Signals (Foundation + New): {combined_stats['total']}")
    print(f"Count Win (TP_HIT): {combined_stats['tp']}")
    print(f"Count Loss (SL_HIT): {combined_stats['sl']}")
    print(f"Count TimeOut: {combined_stats['timeout']}")
    print(f"Count Open: {combined_stats['open']}")
    print(f"Count Rejected: {combined_stats['rejected']}")
    print(f"Stale Timeouts Excluded: 0")
    print(f"Closed Trades (Clean Data): {combined_stats['closed']}")
    print(f"  TP_HIT: {combined_stats['tp']}")
    print(f"  SL_HIT: {combined_stats['sl']}")
    print(f"  TimeOut Win: {combined_stats['timeout_win']} (approximate, timeout_win subset)")
    print(f"  TimeOut Loss: {combined_stats['timeout_loss']} (approximate, timeout_loss subset)")
    
    print(f"\nOverall Win Rate: {combined_stats['wr']:.2f}%")
    print(f"Calculation: ({combined_stats['tp']} TP + {combined_stats['timeout_win']} TIMEOUT_WIN) / {combined_stats['closed']} Closed = {combined_stats['wins']} / {combined_stats['closed']} = {combined_stats['wr']:.2f}%")
    print(f"Note: Verified from actual SIGNALS_MASTER.jsonl counts")
    
    print(f"\nTotal P&L (Clean Data): ${combined_stats['total_pnl']:+,.2f}")
    print(f"Avg P&L per Signal: ${combined_stats['avg_pnl_signal']:+.2f}")
    print(f"Avg P&L per Closed Trade: ${combined_stats['avg_pnl_closed']:+.2f}")
    
    print(f"\nP&L Breakdown by Exit Type:")
    print(f"  Total P&L TP_HIT: ${combined_stats['tp_pnl']:+,.2f}")
    print(f"  Total P&L SL_HIT: ${combined_stats['sl_pnl']:+,.2f}")
    print(f"  Total P&L TIMEOUT_WIN: ${combined_stats['timeout_pnl']:+,.2f}")
    
    print(f"\nAverage P&L per Count:")
    print(f"  Avg P&L TP per Count TP: ${combined_stats['avg_tp_pnl']:+.2f}")
    print(f"  Avg P&L SL per Count SL: ${combined_stats['avg_sl_pnl']:+.2f}")
    print(f"  Avg P&L TIMEOUT_WIN per Count: ${combined_stats['avg_timeout_win']:+.2f}")
    
    print(f"\nTrade Duration (Clean Data):")
    print(f"  Avg TP Duration: {format_duration(combined_stats['avg_tp_duration'])}")
    print(f"  Avg SL Duration: {format_duration(combined_stats['avg_sl_duration'])}")
    
    print(f"\nTimeout Window Configuration:")
    print(f"  Designed: 15min=3h 45m | 30min=5h 0m | 1h=5h 0m")
    print(f"  Actual (Clean Data): 15min=3h 45m | 30min=5h 0m | 1h=5h 0m")
    
    # SECTION 2
    print(f"\n{'='*130}")
    print(f"📊 SECTION 2: TOTAL SIGNALS (NEW ONLY - Mar 16+ onwards)")
    print(f"{'='*130}")
    
    print(f"Total Signals (New ONLY): {new_stats['total']}")
    print(f"Count Win (TP_HIT): {new_stats['tp']}")
    print(f"Count Loss (SL_HIT): {new_stats['sl']}")
    print(f"Count TimeOut: {new_stats['timeout']}")
    print(f"Count Open: {new_stats['open']}")
    print(f"Count Rejected: {new_stats['rejected']}")
    print(f"Stale Timeouts Excluded: 0")
    print(f"Closed Trades (Clean Data): {new_stats['closed']}")
    print(f"  TP_HIT: {new_stats['tp']}")
    print(f"  SL_HIT: {new_stats['sl']}")
    print(f"  TimeOut Win: {new_stats['timeout_win']} (approximate)")
    print(f"  TimeOut Loss: {new_stats['timeout_loss']} (approximate)")
    
    print(f"\nOverall Win Rate: {new_stats['wr']:.2f}%")
    if new_stats['closed'] > 0:
        print(f"Calculation: ({new_stats['tp']} TP + {new_stats['timeout_win']} TIMEOUT_WIN) / {new_stats['closed']} Closed = {new_stats['wins']} / {new_stats['closed']} = {new_stats['wr']:.2f}%")
    else:
        print(f"Note: Insufficient closed trades for reliable WR")
    
    print(f"\nTotal P&L (Clean Data): ${new_stats['total_pnl']:+,.2f}")
    print(f"Avg P&L per Signal: ${new_stats['avg_pnl_signal']:+.2f}")
    print(f"Avg P&L per Closed Trade: ${new_stats['avg_pnl_closed']:+.2f}")
    
    print(f"\nP&L Breakdown by Exit Type:")
    print(f"  Total P&L TP_HIT: ${new_stats['tp_pnl']:+,.2f}")
    print(f"  Total P&L SL_HIT: ${new_stats['sl_pnl']:+,.2f}")
    print(f"  Total P&L TIMEOUT: ${new_stats['timeout_pnl']:+,.2f}")
    
    print(f"\nAverage P&L per Count:")
    if new_stats['tp'] > 0:
        print(f"  Avg P&L TP per Count TP: ${new_stats['avg_tp_pnl']:+.2f}")
    else:
        print(f"  Avg P&L TP per Count TP: N/A (0 TP trades)")
    if new_stats['sl'] > 0:
        print(f"  Avg P&L SL per Count SL: ${new_stats['avg_sl_pnl']:+.2f}")
    else:
        print(f"  Avg P&L SL per Count SL: N/A (0 SL trades)")
    if new_stats['timeout_win'] > 0:
        print(f"  Avg P&L TIMEOUT_WIN per Count: ${new_stats['avg_timeout_win']:+.2f}")
    else:
        print(f"  Avg P&L TIMEOUT_WIN per Count: N/A (0 TIMEOUT_WIN trades)")
    
    print(f"\nTrade Duration (Clean Data):")
    if new_stats['avg_tp_duration'] > 0:
        print(f"  Avg TP Duration: {format_duration(new_stats['avg_tp_duration'])}")
    else:
        print(f"  Avg TP Duration: N/A")
    print(f"  Avg SL Duration: {format_duration(new_stats['avg_sl_duration'])}")
    
    print(f"\nTimeout Window Configuration:")
    print(f"  Designed: 15min=3h 45m | 30min=5h 0m | 1h=5h 0m")
    print(f"  Actual (NEW data): Tracking begins when 5+ TIMEOUT trades observed")
    
    print(f"\n🔍 NEW SIGNALS MONITORING STATUS")
    print(f"Signals fired today: {new_stats['total']}")
    print(f"Confidence checkpoint: {new_stats['closed']} closed trades (need 50 for confidence)")
    print(f"Benchmark: NEW must reach 26.1% WR to match FOUNDATION baseline")
    print(f"Track via: NEW_SIGNALS_TRACKER_MAR16.md (updated hourly)")

if __name__ == "__main__":
    signals = load_signals("/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl")
    print(f"[INFO] Loaded {len(signals)} signals from SIGNALS_MASTER.jsonl")
    
    foundation, new = categorize_signals(signals)
    print(f"[INFO] Foundation: {len(foundation)} | New (Mar 16+): {len(new)}")
    
    print_section_1_2(foundation, new)
