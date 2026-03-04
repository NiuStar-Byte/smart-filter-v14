#!/usr/bin/env python3
"""
PHASE 2-FIXED Performance Tracker (AUTO-REFRESH EVERY 5 SECONDS)

✅ LIVE MONITORING DASHBOARD - Updates every 5 seconds automatically
  
Monitors SHORT signal recovery in real-time:
  1. Compares Phase 2 original vs Phase 2-FIXED
  2. Tracks SHORT WR by regime (BULL, BEAR, RANGE)
  3. Monitors for regressions in LONG signals
  4. Generates daily reports with [PHASE2-FIXED] tag analysis
  5. AUTO-REFRESHES every 5 seconds (no manual re-run needed)

Usage:
  python3 track_phase2_fixed.py              # Full report (auto-refresh every 5s)
  python3 track_phase2_fixed.py --watch      # Live tail of relevant logs
  python3 track_phase2_fixed.py --short      # SHORT signals only
  python3 track_phase2_fixed.py --bear       # BEAR regime only
  python3 track_phase2_fixed.py --once       # Single run (no auto-refresh)
"""

import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import sys
import subprocess
import time
import os

SIGNALS_FILE = "SENT_SIGNALS.jsonl"

# Phase 2-FIXED deployment time (2026-03-03 17:36 GMT+7 = 2026-03-03 10:36 UTC)
PHASE2_FIXED_START = datetime(2026, 3, 3, 10, 36, 0, tzinfo=timezone.utc)  # Deployment time UTC

def parse_utc(time_str):
    """Parse UTC timestamp."""
    if isinstance(time_str, str):
        dt = datetime.fromisoformat(time_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None

def load_signals():
    """Load all signals from JSONL."""
    signals = []
    try:
        with open(SIGNALS_FILE, 'r') as f:
            for line in f:
                try:
                    signals.append(json.loads(line.strip()))
                except:
                    pass
    except FileNotFoundError:
        print(f"❌ {SIGNALS_FILE} not found")
        return []
    return signals

def analyze_signals(signals, phase_fixed_only=False):
    """Analyze signals by regime, direction, status."""
    if phase_fixed_only:
        # Only signals after Phase 2-FIXED deployment
        signals = [s for s in signals if parse_utc(s.get('fired_time_utc')) and 
                   parse_utc(s.get('fired_time_utc')) >= PHASE2_FIXED_START]
    
    results = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'open': 0, 'pnl': 0}))
    
    for sig in signals:
        if sig.get('status') not in ['TP_HIT', 'SL_HIT', 'TIMEOUT', 'OPEN']:
            continue
        
        regime = sig.get('regime', 'UNKNOWN')
        direction = sig.get('signal_type', 'UNKNOWN')
        status = sig.get('status', 'UNKNOWN')
        
        data = results[regime][direction]
        data['total'] += 1
        
        if status == 'TP_HIT':
            data['tp'] += 1
        elif status == 'SL_HIT':
            data['sl'] += 1
        elif status == 'TIMEOUT':
            data['timeout'] += 1
        elif status == 'OPEN':
            data['open'] += 1
        
        if sig.get('pnl_usd') is not None:
            data['pnl'] += sig.get('pnl_usd', 0)
    
    return results

def print_report(results, phase_label="", auto_refresh=False):
    """Print formatted report."""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    refresh_indicator = "🔄 AUTO-REFRESH (every 5 seconds)" if auto_refresh else "📊 SINGLE REPORT"
    
    print("\n" + "=" * 100)
    print(f"📊 PHASE 2-FIXED PERFORMANCE TRACK {phase_label}")
    print(f"   {refresh_indicator}")
    print(f"   Last updated: {timestamp}")
    print("=" * 100)
    print()
    
    for regime in sorted(results.keys()):
        print(f"\n🌊 {regime} REGIME")
        print("─" * 100)
        
        for direction in ['LONG', 'SHORT']:
            if direction not in results[regime]:
                continue
            
            data = results[regime][direction]
            total = data['total']
            closed = data['tp'] + data['sl'] + data['timeout']
            wr = (data['tp'] / closed * 100) if closed > 0 else 0
            pnl = data['pnl']
            
            status = "🚨" if wr < 15 and direction == "SHORT" else "✅" if wr > 20 else "⚠️"
            
            print(f"  {direction:<8} | Total: {total:<4} | Closed: {closed:<4} | "
                  f"TP: {data['tp']:<3} SL: {data['sl']:<3} TO: {data['timeout']:<3} | "
                  f"WR: {wr:>6.1f}% {status} | P&L: ${pnl:>+10.2f}")
    
    print("\n" + "=" * 100)

def generate_daily_report(auto_refresh=True):
    """Generate daily performance summary (with optional auto-refresh every 5 seconds)."""
    cycle = 1
    while True:
        # Clear screen for auto-refresh (not on first run if single report)
        if auto_refresh and cycle > 1:
            os.system('clear')
        
        signals = load_signals()
        results = analyze_signals(signals, phase_fixed_only=True)
        
        print_report(results, "(PHASE 2-FIXED ONLY)", auto_refresh=auto_refresh)
        
        # Summary section
        total_signals = sum(sum(v.get('total', 0) for v in regime_dict.values()) for regime_dict in results.values())
        print(f"\n📈 SUMMARY")
        print(f"  Total signals (Phase 2-FIXED): {total_signals}")
        print(f"  Deployment time: {PHASE2_FIXED_START.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Elapsed: {(datetime.now(timezone.utc) - PHASE2_FIXED_START).total_seconds() / 3600:.1f} hours")
        
        # Key metrics
        bear_short = results.get('BEAR', {}).get('SHORT', {})
        if bear_short and bear_short.get('total', 0) > 0:
            closed = bear_short['tp'] + bear_short['sl'] + bear_short['timeout']
            wr = (bear_short['tp'] / closed * 100) if closed > 0 else 0
            print(f"\n🎯 KEY METRIC: BEAR SHORT WR")
            print(f"   Current: {wr:.1f}% (target: 25%+)")
            print(f"   Signals: {bear_short['total']} | Closed: {closed}")
            if wr > 15:
                print(f"   ✅ SHORT RECOVERY PROGRESSING")
            else:
                print(f"   ⚠️  Need more data or adjustment")
        else:
            print(f"\n🎯 KEY METRIC: BEAR SHORT WR")
            print(f"   ⏳ No BEAR SHORT signals yet (still collecting data)")
            print(f"   Expected: Signals should start appearing within 1-2 hours")
        
        # Auto-refresh logic
        if auto_refresh:
            print(f"\n" + "=" * 100)
            print(f"⏱️  Next update in 5 seconds... (press Ctrl+C to stop)")
            print(f"=" * 100)
            try:
                time.sleep(5)
                cycle += 1
            except KeyboardInterrupt:
                print(f"\n\n👋 Stopped auto-refresh monitoring")
                break
        else:
            break

def watch_live_logs():
    """Watch live daemon logs for Phase 2-FIXED tags."""
    print("📺 Watching main_daemon.log for [PHASE2-FIXED] tags...")
    print("(Ctrl+C to stop)\n")
    
    try:
        cmd = "tail -f main_daemon.log | grep 'PHASE2-FIXED'"
        subprocess.run(cmd, shell=True, cwd="/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main")
    except KeyboardInterrupt:
        print("\n\n👋 Stopped watching")

def short_signals_only():
    """Show only SHORT signals."""
    signals = load_signals()
    results = analyze_signals(signals, phase_fixed_only=True)
    
    print("\n" + "=" * 100)
    print("🎯 SHORT SIGNALS ONLY (Phase 2-FIXED)")
    print("=" * 100)
    
    for regime in ['BULL', 'BEAR', 'RANGE']:
        if regime not in results or 'SHORT' not in results[regime]:
            continue
        
        data = results[regime]['SHORT']
        closed = data['tp'] + data['sl'] + data['timeout']
        wr = (data['tp'] / closed * 100) if closed > 0 else 0
        
        print(f"\n{regime}:")
        print(f"  Total: {data['total']} | Closed: {closed} | TP: {data['tp']} | SL: {data['sl']} | TO: {data['timeout']}")
        print(f"  WR: {wr:.1f}% | P&L: ${data['pnl']:+.2f}")
        print(f"  Avg/trade: ${data['pnl']/closed:+.2f}" if closed > 0 else "")

def bear_regime_only():
    """Show only BEAR regime signals."""
    signals = load_signals()
    results = analyze_signals(signals, phase_fixed_only=True)
    
    print("\n" + "=" * 100)
    print("🌊 BEAR REGIME ONLY (Phase 2-FIXED)")
    print("=" * 100)
    
    if 'BEAR' not in results:
        print("  No BEAR signals yet")
        return
    
    for direction in ['LONG', 'SHORT']:
        if direction not in results['BEAR']:
            continue
        
        data = results['BEAR'][direction]
        closed = data['tp'] + data['sl'] + data['timeout']
        wr = (data['tp'] / closed * 100) if closed > 0 else 0
        
        icon = "🚨" if direction == "SHORT" and wr < 20 else "✅"
        
        print(f"\n{direction} {icon}:")
        print(f"  Total: {data['total']} | Closed: {closed} | TP: {data['tp']} | SL: {data['sl']} | TO: {data['timeout']}")
        print(f"  WR: {wr:.1f}% | P&L: ${data['pnl']:+.2f}")
        print(f"  Avg/trade: ${data['pnl']/closed:+.2f}" if closed > 0 else "")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--watch":
            watch_live_logs()
        elif arg == "--short":
            short_signals_only()
        elif arg == "--bear":
            bear_regime_only()
        elif arg == "--once":
            generate_daily_report(auto_refresh=False)
        else:
            print(f"Unknown option: {arg}")
            print("\n" + "=" * 100)
            print("📊 PHASE 2-FIXED Performance Tracker - USAGE")
            print("=" * 100)
            print("\n✅ DEFAULT (AUTO-REFRESH every 5 seconds):")
            print("  python3 track_phase2_fixed.py              # Full report (auto-refresh)")
            print("\n📋 OTHER OPTIONS:")
            print("  python3 track_phase2_fixed.py --once       # Single report (no auto-refresh)")
            print("  python3 track_phase2_fixed.py --watch      # Live tail daemon logs")
            print("  python3 track_phase2_fixed.py --short      # SHORT signals only (auto-refresh)")
            print("  python3 track_phase2_fixed.py --bear       # BEAR regime only (auto-refresh)")
            print("\n💡 TIPS:")
            print("  • Default command updates every 5 seconds - just run and watch!")
            print("  • Press Ctrl+C to stop at any time")
            print("  • Use --once for single snapshot without auto-refresh")
            print("=" * 100 + "\n")
    else:
        # Default: auto-refresh every 5 seconds
        generate_daily_report(auto_refresh=True)
