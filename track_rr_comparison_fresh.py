#!/usr/bin/env python3
"""
📊 RR VARIANT COMPARISON (FRESH START)
Compares 2.0:1 (PROD) vs 1.5:1 (NEW) - excluding corrupted Mar 3-8 signals

2.0:1 RR (PROD):  Feb 27 15:55 UTC → Mar 4 17:51 UTC [FIXED: locked baseline]
1.5:1 RR (NEW):   Mar 4 17:51 UTC → NOW [DYNAMIC: clean signals only]

Deployment: 2026-03-04 20:32 GMT+7 (commit e3f110b)
Fresh Start: 2026-03-08 04:20 UTC (corrupted signals discarded)
"""

import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os
import sys
import subprocess
import time

workspace = "/Users/geniustarigan/.openclaw/workspace"
signals_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")

# 2.0:1 RR cutoff (locked period)
RR_2_0_START = "2026-02-27T15:55:00"
RR_2_0_END = "2026-03-04T17:51:00"

# 1.5:1 RR cutoff (fresh start, excluding corrupted Mar 3-8)
RR_1_5_START = "2026-03-04T17:51:00"
RR_1_5_FRESH = "2026-03-08T04:20:00"  # Fresh start, discard before this

def clear_screen():
    """Clear terminal"""
    subprocess.call('clear' if os.name == 'posix' else 'cls', shell=True)

def load_signals():
    """Load signals separated by RR variant"""
    signals_2_0 = []
    signals_1_5_all = []
    signals_1_5_fresh = []
    
    if not os.path.exists(signals_file):
        return signals_2_0, signals_1_5_all, signals_1_5_fresh
    
    try:
        with open(signals_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    
                    # Skip stale
                    if sig.get('data_quality_flag') and 'STALE_TIMEOUT' in sig.get('data_quality_flag'):
                        continue
                    
                    # Count all trades (including OPEN)
                    status = sig.get('status')
                    
                    fired = sig.get('fired_time_utc', '')
                    rr = sig.get('achieved_rr')
                    
                    # 2.0:1 RR (locked period)
                    if rr == 2.0 and RR_2_0_START <= fired <= RR_2_0_END:
                        signals_2_0.append(sig)
                    
                    # 1.5:1 RR (all accumulated)
                    elif rr == 1.5 and fired >= RR_1_5_START:
                        signals_1_5_all.append(sig)
                        
                        # 1.5:1 RR (fresh start, excluding corrupted)
                        if fired >= RR_1_5_FRESH:
                            signals_1_5_fresh.append(sig)
        
        return signals_2_0, signals_1_5_all, signals_1_5_fresh
    except:
        return signals_2_0, signals_1_5_all, signals_1_5_fresh

def analyze_metrics(signals):
    """Calculate metrics for a signal group"""
    if not signals:
        return None
    
    wins = 0
    losses = 0
    opens = 0
    none_status = 0
    tp_hits = 0
    sl_hits = 0
    timeouts = 0
    total_pnl = 0.0
    
    tp_durations = []
    sl_durations = []
    
    for sig in signals:
        status = sig.get('status')
        pnl = sig.get('pnl_usd')
        
        if status == 'OPEN':
            opens += 1
        elif status is None:
            none_status += 1
        elif pnl is not None:
            total_pnl += float(pnl)
            if float(pnl) > 0:
                wins += 1
            elif float(pnl) < 0:
                losses += 1
        
        if status == 'TP_HIT':
            tp_hits += 1
            dur = sig.get('duration_seconds')
            if dur:
                tp_durations.append(dur)
        elif status == 'SL_HIT':
            sl_hits += 1
            dur = sig.get('duration_seconds')
            if dur:
                sl_durations.append(dur)
        elif status == 'TIMEOUT':
            timeouts += 1
    
    closed = wins + losses
    wr = (wins / closed * 100) if closed > 0 else 0
    avg_pnl = total_pnl / len(signals) if len(signals) > 0 else 0
    
    avg_tp_dur = sum(tp_durations) / len(tp_durations) if tp_durations else 0
    avg_sl_dur = sum(sl_durations) / len(sl_durations) if sl_durations else 0
    
    return {
        'total': len(signals),
        'closed': closed,
        'wins': wins,
        'losses': losses,
        'opens': opens,
        'none_status': none_status,
        'wr': wr,
        'tp_hits': tp_hits,
        'sl_hits': sl_hits,
        'timeouts': timeouts,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_tp_duration': avg_tp_dur,
        'avg_sl_duration': avg_sl_dur
    }

def format_duration(seconds):
    """Convert seconds to human readable format"""
    if seconds == 0:
        return "0h 0m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

def print_report(signals_2_0, signals_1_5_all, signals_1_5_fresh):
    """Print formatted report"""
    print("\n" + "="*100)
    print("📊 RR VARIANT COMPARISON (FRESH START)")
    print("="*100)
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')
    print(f"\nGenerated: {now}")
    
    print(f"\n⏱️  CUTOFF TIME BETWEEN RR VARIANTS:")
    print(f"   • 2.0:1 RR (PROD):  Feb 27 15:55 UTC → Mar 4 17:51 UTC [FIXED: locked baseline]")
    print(f"   • 1.5:1 RR (NEW):   Mar 4 17:51 UTC → Mar 8 04:20 UTC [CORRUPTED: discarded]")
    print(f"   • 1.5:1 RR (FRESH): Mar 8 04:20 UTC → NOW [DYNAMIC: clean signals only]")
    print(f"   • Deployment: 2026-03-04 20:32 GMT+7 (commit e3f110b)")
    
    m_2_0 = analyze_metrics(signals_2_0)
    m_1_5_fresh = analyze_metrics(signals_1_5_fresh)
    
    print(f"\n{'='*100}")
    print("COMPARISON: 2.0:1 (PROD) vs 1.5:1 (NEW, FRESH START)")
    print(f"{'='*100}")
    
    print(f"\n{'RR':<8} | {'Total':<8} | {'Closed':<8} | {'OPEN':<8} | {'TP':<6} | {'SL':<6} | {'TIMEOUT':<8} | {'WR %':<8} | {'P&L $':<12} | {'Avg P&L':<10}")
    print("-"*120)
    
    if m_2_0:
        print(f"{'2.0:1':<8} | {m_2_0['total']:<8} | {m_2_0['closed']:<8} | {m_2_0['opens']:<8} | {m_2_0['tp_hits']:<6} | {m_2_0['sl_hits']:<6} | {m_2_0['timeouts']:<8} | {m_2_0['wr']:<7.2f}% | ${m_2_0['total_pnl']:<10.2f} | ${m_2_0['avg_pnl']:<8.2f}")
    
    if m_1_5_fresh:
        print(f"{'1.5:1':<8} | {m_1_5_fresh['total']:<8} | {m_1_5_fresh['closed']:<8} | {m_1_5_fresh['opens']:<8} | {m_1_5_fresh['tp_hits']:<6} | {m_1_5_fresh['sl_hits']:<6} | {m_1_5_fresh['timeouts']:<8} | {m_1_5_fresh['wr']:<7.2f}% | ${m_1_5_fresh['total_pnl']:<10.2f} | ${m_1_5_fresh['avg_pnl']:<8.2f}")
    
    # Impact analysis
    if m_2_0 and m_1_5_fresh:
        print(f"\n{'='*100}")
        print("📈 IMPACT ANALYSIS - 2.0:1 (BASELINE) vs 1.5:1 (NEW)")
        print(f"{'='*100}")
        
        wr_change = m_1_5_fresh['wr'] - m_2_0['wr']
        pnl_change = m_1_5_fresh['total_pnl'] - m_2_0['total_pnl']
        tp_dur_change = m_1_5_fresh['avg_tp_duration'] - m_2_0['avg_tp_duration']
        sl_dur_change = m_1_5_fresh['avg_sl_duration'] - m_2_0['avg_sl_duration']
        
        print(f"\nWin Rate Change:    {m_2_0['wr']:.2f}% → {m_1_5_fresh['wr']:.2f}% ({wr_change:+.2f}pp)", end='')
        if wr_change > 5:
            print(" ✅ IMPROVED")
        elif wr_change < -5:
            print(" ⚠️  DECLINED")
        else:
            print(" → No major change")
        
        print(f"TP Avg Duration:    {format_duration(m_2_0['avg_tp_duration'])} → {format_duration(m_1_5_fresh['avg_tp_duration'])} ({tp_dur_change:+.0f}s)")
        print(f"SL Avg Duration:    {format_duration(m_2_0['avg_sl_duration'])} → {format_duration(m_1_5_fresh['avg_sl_duration'])} ({sl_dur_change:+.0f}s)")
        print(f"Total P&L Change:   ${m_2_0['total_pnl']:+.2f} → ${m_1_5_fresh['total_pnl']:+.2f} ({pnl_change:+.2f})")
        
        print(f"\nSignal Volume:      {m_2_0['total']} (FIXED) vs {m_1_5_fresh['total']} (accumulating)")
        print(f"Closed Trades:      {m_2_0['closed']} vs {m_1_5_fresh['closed']}")
        print(f"OPEN Trades:        {m_2_0['opens']} vs {m_1_5_fresh['opens']} (still accumulating)")
        print(f"Unprocessed:        {m_1_5_fresh['none_status']} signals (raw, not yet tracked)")
        print(f"Success Metric:     Need {max(100, m_1_5_fresh['closed'] * 2)} closed trades for confidence")
    
    print(f"\n{'='*100}\n")

def main():
    signals_2_0, signals_1_5_all, signals_1_5_fresh = load_signals()
    print_report(signals_2_0, signals_1_5_all, signals_1_5_fresh)

if __name__ == '__main__':
    if '--once' in sys.argv:
        main()
    else:
        try:
            while True:
                clear_screen()
                main()
                time.sleep(5)
        except KeyboardInterrupt:
            clear_screen()
            print("✅ Stopped\n")
