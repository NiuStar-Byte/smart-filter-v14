#!/usr/bin/env python3
"""
A/B TEST COMPARISON SCRIPT - Phase 1 (A) vs Phase 2-FIXED (B)
Uses EXACT same calculation method as pec_enhanced_reporter.py

⚠️ CRITICAL CUTOFF: Uses 13:16 UTC (20:16 GMT+7) - AFTER critical fixes applied
- Excludes signals from broken period (10:36-13:16 UTC)
- Includes ONLY signals from fixed code

✅ PHASE 1 (A): All signals BEFORE 2026-03-03 13:16 UTC (critical fixes deployment)
✅ PHASE 2-FIXED (B): All signals AFTER 2026-03-03 13:16 UTC (FRESH signals from fixed code)

Timeline:
  10:36 UTC: Phase 2-FIXED deployed (gates broken)
  13:16 UTC: Critical fixes applied (momentum + threshold logic)
  → Only count signals from 13:16 UTC onwards (FRESH data)

Usage:
  python3 COMPARE_AB_TEST.py --once           (single run)
  python3 COMPARE_AB_TEST.py                  (live refresh every 5s)
"""

import json
from datetime import datetime, timezone, timedelta
import os
import sys
import time

PHASE1_CUTOFF = datetime(2026, 3, 3, 13, 16, 0, tzinfo=timezone.utc)  # Mar 03 20:16 GMT+7 = 13:16 UTC - CRITICAL FIXES APPLIED (Fresh signals only)
SIGNALS_FILE = "SENT_SIGNALS.jsonl"

def load_and_split_signals():
    """Load all signals and split into Phase 1 & Phase 2"""
    phase1 = []
    phase2 = []
    
    try:
        with open(SIGNALS_FILE, 'r') as f:
            for line in f:
                try:
                    sig = json.loads(line.strip())
                    if not sig:
                        continue
                    
                    fired_str = sig.get('fired_at') or sig.get('fired_time_utc') or sig.get('sent_time_utc', '')
                    if not fired_str:
                        continue
                    
                    # Parse the datetime and ensure it's UTC-aware
                    fired = datetime.fromisoformat(fired_str.replace('Z', '+00:00'))
                    if fired.tzinfo is None:
                        fired = fired.replace(tzinfo=timezone.utc)
                    
                    if fired < PHASE1_CUTOFF:
                        phase1.append(sig)
                    else:
                        phase2.append(sig)
                except:
                    continue
    except FileNotFoundError:
        print(f"❌ File not found: {SIGNALS_FILE}")
        return [], []
    
    return phase1, phase2

def format_duration(minutes):
    """Convert minutes to human readable format (e.g., 1h 59m)"""
    if not minutes or minutes == 0 or minutes < 0:
        return "N/A"
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h == 0 and m == 0:
        return "N/A"
    return f"{h}h {m}m" if h > 0 else f"{m}m"

def calculate_pnl_usd(entry_price, exit_price, signal_type, notional=1000.0):
    """Calculate P&L in USD using $1000 notional position"""
    if not entry_price or not exit_price:
        return None
    
    try:
        entry = float(entry_price)
        exit = float(exit_price)
        
        if signal_type == 'LONG':
            pct_change = (exit - entry) / entry
        else:  # SHORT
            pct_change = (entry - exit) / entry
        
        return notional * pct_change
    except:
        return None

def calculate_detailed_metrics(signals, phase_name=""):
    """Calculate comprehensive metrics matching pec_enhanced_reporter format"""
    if not signals:
        return None
    
    total_signals = len(signals)
    
    # Count by status
    total_tp = sum(1 for s in signals if s.get('status') == 'TP_HIT')
    total_sl = sum(1 for s in signals if s.get('status') == 'SL_HIT')
    total_timeout = sum(1 for s in signals if s.get('status') == 'TIMEOUT')
    total_open = sum(1 for s in signals if s.get('status') == 'OPEN')
    
    # Count stale timeouts (marked with data_quality_flag)
    stale_timeout_count = sum(1 for s in signals if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'))
    
    # Split timeouts into wins and losses (excluding stale)
    timeout_wins = 0
    timeout_losses = 0
    for s in signals:
        if s.get('status') == 'TIMEOUT' and not (s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag')):
            pnl = calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), s.get('signal_type'))
            if pnl and pnl > 0:
                timeout_wins += 1
            elif pnl:
                timeout_losses += 1
    
    # Closed trades = TP + SL + TIMEOUT_WIN + TIMEOUT_LOSS (excludes stale)
    closed_signals = total_tp + total_sl + timeout_wins + timeout_losses
    
    # Win rate
    win_count = total_tp + timeout_wins
    overall_wr = (win_count / closed_signals * 100) if closed_signals > 0 else 0
    
    # Direction breakdown (clean data only)
    long_tp = 0
    long_sl = 0
    long_timeout_wins = 0
    long_timeout_losses = 0
    short_tp = 0
    short_sl = 0
    short_timeout_wins = 0
    short_timeout_losses = 0
    
    for s in signals:
        if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'):
            continue  # Skip stale
        
        direction = s.get('signal_type')
        status = s.get('status')
        
        if status == 'TP_HIT':
            if direction == 'LONG':
                long_tp += 1
            else:
                short_tp += 1
        elif status == 'SL_HIT':
            if direction == 'LONG':
                long_sl += 1
            else:
                short_sl += 1
        elif status == 'TIMEOUT':
            pnl = calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
            if pnl and pnl > 0:
                if direction == 'LONG':
                    long_timeout_wins += 1
                else:
                    short_timeout_wins += 1
            elif pnl:
                if direction == 'LONG':
                    long_timeout_losses += 1
                else:
                    short_timeout_losses += 1
    
    long_closed = long_tp + long_sl + long_timeout_wins + long_timeout_losses
    short_closed = short_tp + short_sl + short_timeout_wins + short_timeout_losses
    
    long_wr = ((long_tp + long_timeout_wins) / long_closed * 100) if long_closed > 0 else 0
    short_wr = ((short_tp + short_timeout_wins) / short_closed * 100) if short_closed > 0 else 0
    
    # P&L (exclude stale timeouts)
    total_pnl = 0.0
    for s in signals:
        if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'):
            continue  # Skip stale timeouts from P&L
        if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
            pnl = calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), s.get('signal_type'))
            if pnl is not None:
                total_pnl += pnl
    
    # Duration metrics (only TP_HIT and SL_HIT, excluding stale)
    tp_durations = []
    sl_durations = []
    
    for s in signals:
        if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'):
            continue
        
        try:
            closed_at = s.get('closed_at')
            fired_str = s.get('fired_time_utc')
            
            if closed_at and fired_str:
                closed = datetime.fromisoformat(closed_at.replace('Z', '+00:00'))
                fired = datetime.fromisoformat(fired_str.replace('Z', '+00:00'))
                duration_min = (closed - fired).total_seconds() / 60
                
                if s.get('status') == 'TP_HIT':
                    tp_durations.append(duration_min)
                elif s.get('status') == 'SL_HIT':
                    sl_durations.append(duration_min)
        except:
            pass
    
    avg_tp_duration = sum(tp_durations) / len(tp_durations) if tp_durations else 0
    avg_sl_duration = sum(sl_durations) / len(sl_durations) if sl_durations else 0
    
    return {
        "total_signals": total_signals,
        "count_win_tp": total_tp,
        "count_loss_sl": total_sl,
        "count_timeout": timeout_wins + timeout_losses,  # CLEAN COUNT: win + loss only
        "count_open": total_open,
        "stale_timeouts_excluded": stale_timeout_count,
        "closed_trades": closed_signals,
        "timeout_wins": timeout_wins,
        "timeout_losses": timeout_losses,
        "overall_wr": overall_wr,
        "long_trades": long_closed,
        "long_tp": long_tp,
        "long_timeout_wins": long_timeout_wins,
        "long_wr": long_wr,
        "short_trades": short_closed,
        "short_tp": short_tp,
        "short_timeout_wins": short_timeout_wins,
        "short_wr": short_wr,
        "total_pnl": total_pnl,
        "avg_tp_duration_min": avg_tp_duration,
        "avg_sl_duration_min": avg_sl_duration,
    }

def print_comparison():
    """Print formatted A/B test comparison"""
    print("\n" + "="*200)
    print("🧪 A/B TEST COMPARISON - PHASE 1 (BASELINE) vs PHASE 2-FIXED (CHALLENGER w/ Critical Fixes)")
    print("="*200)
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"Split Point: 2026-03-03 20:16 GMT+7 (13:16 UTC) - CRITICAL FIXES APPLIED (Fresh signals only)")
    print(f"Data Source: {SIGNALS_FILE} (All signals = Phase 1 + Phase 2 = TOTAL)")
    print("="*200)
    print()
    
    phase1, phase2 = load_and_split_signals()
    m1 = calculate_detailed_metrics(phase1, "PHASE 1")
    m2 = calculate_detailed_metrics(phase2, "PHASE 2") if phase2 else None
    
    if not m1:
        print("❌ No Phase 1 data found")
        return
    
    has_phase2_closed = m2 and m2["closed_trades"] > 0
    
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # TABLE HEADER
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    print("METRIC                               │   PHASE 1 (A)     │ PHASE 2-FIXED (B) │   TOTAL (A+B) │   DELTA       │ STATUS")
    print("═"*200)
    
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # SIGNAL COUNTS SECTION
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    # Total Signals
    p2_signals = m2["total_signals"] if m2 else 0
    total_signals = m1["total_signals"] + p2_signals
    delta_signals = p2_signals - m1["total_signals"]
    print(f"Total Signals Fired                  │  {m1['total_signals']:6d}      │  {p2_signals:6d}       │  {total_signals:6d}       │  {delta_signals:+6d}     │ 📊")
    print("─"*200)
    
    # Count Win (TP)
    p2_tp = m2["count_win_tp"] if m2 else 0
    total_tp = m1["count_win_tp"] + p2_tp
    delta_tp = p2_tp - m1["count_win_tp"]
    print(f"  ├─ Count Win (TP Hit)              │  {m1['count_win_tp']:6d}      │  {p2_tp:6d}       │  {total_tp:6d}       │  {delta_tp:+6d}     │")
    
    # Count Loss (SL)
    p2_sl = m2["count_loss_sl"] if m2 else 0
    total_sl = m1["count_loss_sl"] + p2_sl
    delta_sl = p2_sl - m1["count_loss_sl"]
    print(f"  ├─ Count Loss (SL Hit)             │  {m1['count_loss_sl']:6d}      │  {p2_sl:6d}       │  {total_sl:6d}       │  {delta_sl:+6d}     │")
    
    # Count Timeout (CLEAN)
    p2_timeout = m2["count_timeout"] if m2 else 0
    total_timeout = m1["count_timeout"] + p2_timeout
    delta_timeout = p2_timeout - m1["count_timeout"]
    print(f"  ├─ Count Timeout (Clean)           │  {m1['count_timeout']:6d}      │  {p2_timeout:6d}       │  {total_timeout:6d}       │  {delta_timeout:+6d}     │")
    
    # Count Open
    p2_open = m2["count_open"] if m2 else 0
    total_open = m1["count_open"] + p2_open
    delta_open = p2_open - m1["count_open"]
    print(f"  ├─ Count Open                      │  {m1['count_open']:6d}      │  {p2_open:6d}       │  {total_open:6d}       │  {delta_open:+6d}     │")
    
    # Stale Timeouts Excluded
    p2_stale = m2["stale_timeouts_excluded"] if m2 else 0
    total_stale = m1["stale_timeouts_excluded"] + p2_stale
    delta_stale = p2_stale - m1["stale_timeouts_excluded"]
    print(f"  └─ Stale Timeouts Excluded         │  {m1['stale_timeouts_excluded']:6d}      │  {p2_stale:6d}       │  {total_stale:6d}       │  {delta_stale:+6d}     │")
    
    print("═"*200)
    
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # CLOSED TRADES & BREAKDOWN SECTION
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    # Closed Trades (Clean)
    p2_closed = m2["closed_trades"] if m2 else 0
    total_closed = m1["closed_trades"] + p2_closed
    delta_closed = p2_closed - m1["closed_trades"]
    print(f"Closed Trades (Clean Data)           │  {m1['closed_trades']:6d}      │  {p2_closed:6d}       │  {total_closed:6d}       │  {delta_closed:+6d}     │ ✓")
    
    # Timeout Win
    p2_tw = m2["timeout_wins"] if m2 else 0
    total_tw = m1["timeout_wins"] + p2_tw
    delta_tw = p2_tw - m1["timeout_wins"]
    print(f"  ├─ TimeOut Win                     │  {m1['timeout_wins']:6d}      │  {p2_tw:6d}       │  {total_tw:6d}       │  {delta_tw:+6d}     │")
    
    # Timeout Loss
    p2_tl = m2["timeout_losses"] if m2 else 0
    total_tl = m1["timeout_losses"] + p2_tl
    delta_tl = p2_tl - m1["timeout_losses"]
    print(f"  └─ TimeOut Loss                    │  {m1['timeout_losses']:6d}      │  {p2_tl:6d}       │  {total_tl:6d}       │  {delta_tl:+6d}     │")
    
    print("═"*200)
    
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # WIN RATE METRICS SECTION
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    # Win Rate - TOTAL uses exact formula: (Total TP + Total Timeout Wins) / Total Closed
    if has_phase2_closed:
        total_win_count = m1["count_win_tp"] + m2["count_win_tp"] + m1["timeout_wins"] + m2["timeout_wins"]
        total_closed_all = m1["closed_trades"] + m2["closed_trades"]
        total_wr = (total_win_count / total_closed_all * 100) if total_closed_all > 0 else 0
        delta_wr = (m2["overall_wr"] - m1["overall_wr"])
        status_wr = "✅ PASS" if delta_wr >= 4.0 else ("⚠️  WATCH" if delta_wr >= -1.0 else "❌ FAIL")
        print(f"Overall Win Rate                     │  {m1['overall_wr']:6.2f}%    │  {m2['overall_wr']:6.2f}%    │  {total_wr:6.2f}%       │  {delta_wr:+6.2f}%   │ {status_wr}")
    else:
        print(f"Overall Win Rate                     │  {m1['overall_wr']:6.2f}%    │   {'N/A':6s}   │   {'N/A':6s}     │   {'N/A':6s}  │ ⏳")
    
    print("─"*200)
    
    # LONG WR - TOTAL uses exact formula: (LONG TP + LONG Timeout Wins) / Total LONG Closed
    if has_phase2_closed:
        long_total_trades = m1["long_trades"] + m2["long_trades"]
        long_total_wins = (m1["long_tp"] + m2["long_tp"] + m1["long_timeout_wins"] + m2["long_timeout_wins"])
        long_total_wr = (long_total_wins / long_total_trades * 100) if long_total_trades > 0 else 0
        delta_long = (m2["long_wr"] - m1["long_wr"])
        status_long = "✅ PASS" if delta_long >= 7.0 else ("⚠️  WATCH" if delta_long >= 0 else "❌ FAIL")
        print(f"  ├─ LONG WR (MAIN FIX)              │  {m1['long_wr']:6.2f}% ({m1['long_trades']:3d})│  {m2['long_wr']:6.2f}% ({m2['long_trades']:3d}) │  {long_total_wr:6.2f}% ({long_total_trades:3d}) │  {delta_long:+6.2f}%   │ {status_long}")
    else:
        print(f"  ├─ LONG WR (MAIN FIX)              │  {m1['long_wr']:6.2f}% ({m1['long_trades']:3d})│   {'N/A':6s}   │   {'N/A':6s}     │   {'N/A':6s}  │ ⏳")
    
    # SHORT WR - TOTAL uses exact formula: (SHORT TP + SHORT Timeout Wins) / Total SHORT Closed
    if has_phase2_closed:
        short_total_trades = m1["short_trades"] + m2["short_trades"]
        short_total_wins = (m1["short_tp"] + m2["short_tp"] + m1["short_timeout_wins"] + m2["short_timeout_wins"])
        short_total_wr = (short_total_wins / short_total_trades * 100) if short_total_trades > 0 else 0
        delta_short = (m2["short_wr"] - m1["short_wr"])
        status_short = "✅ PASS" if delta_short >= 0 else "⚠️  WATCH"
        print(f"  └─ SHORT WR (MAINTAIN)             │  {m1['short_wr']:6.2f}% ({m1['short_trades']:3d})│  {m2['short_wr']:6.2f}% ({m2['short_trades']:3d}) │  {short_total_wr:6.2f}% ({short_total_trades:3d}) │  {delta_short:+6.2f}%   │ {status_short}")
    else:
        print(f"  └─ SHORT WR (MAINTAIN)             │  {m1['short_wr']:6.2f}% ({m1['short_trades']:3d})│   {'N/A':6s}   │   {'N/A':6s}     │   {'N/A':6s}  │ ⏳")
    
    print("═"*200)
    
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # P&L & DURATION SECTION
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    # P&L
    if has_phase2_closed:
        total_pnl = m1["total_pnl"] + m2["total_pnl"]
        delta_pnl = (m2["total_pnl"] - m1["total_pnl"])
        status_pnl = "✅ PASS" if delta_pnl > -500 else ("⚠️  WATCH" if delta_pnl > -1000 else "❌ FAIL")
        print(f"Total P&L (Clean Data)               │  ${m1['total_pnl']:9.2f}  │  ${m2['total_pnl']:9.2f}  │  ${total_pnl:9.2f}     │  ${delta_pnl:+9.2f}  │ {status_pnl}")
    else:
        print(f"Total P&L (Clean Data)               │  ${m1['total_pnl']:9.2f}  │   {'N/A':9s}  │   {'N/A':9s}    │   {'N/A':9s} │ ⏳")
    
    print("─"*200)
    
    # Duration metrics - TOTAL uses weighted average
    tp_dur_1 = format_duration(m1["avg_tp_duration_min"]) if m1["count_win_tp"] > 0 else "N/A"
    tp_dur_2 = format_duration(m2["avg_tp_duration_min"]) if (m2 and m2["count_win_tp"] > 0) else "N/A"
    
    if total_tp > 0:
        total_tp_avg_min = (m1["avg_tp_duration_min"] * m1["count_win_tp"] + m2["avg_tp_duration_min"] * m2["count_win_tp"]) / total_tp
        total_tp_dur = format_duration(total_tp_avg_min)
    else:
        total_tp_dur = "N/A"
    print(f"  ├─ Avg TP Duration (Clean)         │  {tp_dur_1:10s}  │  {tp_dur_2:10s}  │  {total_tp_dur:10s}   │            │")
    
    sl_dur_1 = format_duration(m1["avg_sl_duration_min"]) if m1["count_loss_sl"] > 0 else "N/A"
    sl_dur_2 = format_duration(m2["avg_sl_duration_min"]) if (m2 and m2["count_loss_sl"] > 0) else "N/A"
    
    if total_sl > 0:
        total_sl_avg_min = (m1["avg_sl_duration_min"] * m1["count_loss_sl"] + m2["avg_sl_duration_min"] * m2["count_loss_sl"]) / total_sl
        total_sl_dur = format_duration(total_sl_avg_min)
    else:
        total_sl_dur = "N/A"
    print(f"  └─ Avg SL Duration (Clean)         │  {sl_dur_1:10s}  │  {sl_dur_2:10s}  │  {total_sl_dur:10s}   │            │")
    
    print("═"*200)
    
    # Summary
    print()
    print("📊 SUMMARY")
    print("─"*200)
    
    if not has_phase2_closed:
        print(f"⏳ Phase 2 (B) is collecting data...")
        print(f"   Phase 1 (A) LOCKED: {m1['overall_wr']:.2f}% WR on {m1['closed_trades']} closed trades | P&L: ${m1['total_pnl']:.2f}")
        if m2:
            print(f"   Phase 2 (B) PROGRESS: {m2['total_signals']} signals fired, {m2['closed_trades']} closed trades | P&L: ${m2['total_pnl']:.2f}")
        print(f"   Rerun this script to track Phase 2 metrics as trades close.")
    else:
        print(f"Phase 2-FIXED (B) has {m2['closed_trades']} closed trades.")
        wr_delta = (m2["overall_wr"] - m1["overall_wr"])
        if wr_delta >= 4.0:
            print(f"✅ Overall WR improved by {wr_delta:.2f}% - ON TRACK for Phase 2 success")
        elif wr_delta >= 0:
            print(f"⚠️  Overall WR showed minor improvement ({wr_delta:.2f}%) - Watch for more data")
        else:
            print(f"❌ Overall WR declined by {abs(wr_delta):.2f}% - Check kill switch threshold")
    
    print()
    print("🎯 NEXT STEPS")
    print("─"*200)
    print("Run daily:  python3 COMPARE_AB_TEST.py --once")
    print("Live watch: ./watch_ab_test.sh  (refreshes every 5 seconds)")
    print("Full report: python3 pec_enhanced_reporter.py")
    print("═"*200 + "\n")

if __name__ == "__main__":
    once_only = "--once" in sys.argv
    
    try:
        while True:
            if not once_only:
                os.system('clear')
            print_comparison()
            if once_only:
                break
            import time
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n✅ A/B Test Monitor stopped")
        exit(0)
