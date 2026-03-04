#!/usr/bin/env python3
"""
🔒 A/B TEST - Foundation Baseline vs Phase 2-FIXED
One True Baseline. No conflicting numbers.
"""

import json
from datetime import datetime, timezone
import os, sys, time
import subprocess

# 🔒 FOUNDATION - THE ONLY BASELINE (LOCKED 2026-03-04 01:10 GMT+7)
FOUNDATION = {
    "total_signals": 853,
    "closed_trades": 830,
    "win_rate": 25.7,
    "long_wr": 29.6,
    "short_wr": 46.2,
    "pnl": -5498.59
}

SIGNALS_FILE = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"  # Daemon writes to workspace root
PHASE2_CUTOFF = datetime(2026, 3, 3, 13, 16, 0, tzinfo=timezone.utc)

def clear_screen():
    """Clear terminal screen (cross-platform)"""
    subprocess.call('clear' if os.name == 'posix' else 'cls', shell=True)

def get_phase2_signals():
    """Load Phase 2-FIXED signals (after 13:16 UTC cutoff)"""
    phase2 = []
    try:
        with open(SIGNALS_FILE, 'r') as f:
            for line in f:
                try:
                    sig = json.loads(line.strip())
                    fired_str = sig.get('fired_time_utc', '')
                    if not fired_str:
                        continue
                    fired = datetime.fromisoformat(fired_str.split('+')[0]).replace(tzinfo=timezone.utc)
                    if fired >= PHASE2_CUTOFF:
                        phase2.append(sig)
                except:
                    continue
    except:
        pass
    return phase2

def main():
    phase2_sigs = get_phase2_signals()
    p2_closed = len([s for s in phase2_sigs if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']])
    p2_wins = len([s for s in phase2_sigs if s.get('status') == 'TP_HIT'])
    p2_wr = (p2_wins / p2_closed * 100) if p2_closed > 0 else 0
    p2_longs = len([s for s in phase2_sigs if s.get('signal_type') == 'LONG' and s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']])
    p2_long_wins = len([s for s in phase2_sigs if s.get('signal_type') == 'LONG' and s.get('status') == 'TP_HIT'])
    p2_long_wr = (p2_long_wins / p2_longs * 100) if p2_longs > 0 else 0
    
    p2_shorts = len([s for s in phase2_sigs if s.get('signal_type') == 'SHORT' and s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']])
    p2_short_wins = len([s for s in phase2_sigs if s.get('signal_type') == 'SHORT' and s.get('status') == 'TP_HIT'])
    p2_short_wr = (p2_short_wins / p2_shorts * 100) if p2_shorts > 0 else 0
    
    print("\n" + "="*100)
    print("🧪 A/B TEST - FOUNDATION vs PHASE 2-FIXED")
    print("="*100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"Foundation (Locked): {FOUNDATION['total_signals']} signals | {FOUNDATION['win_rate']}% WR | ${FOUNDATION['pnl']}")
    print("="*100)
    print()
    
    # Comparison table
    print(f"METRIC              │  FOUNDATION (A)     │ PHASE 2-FIXED (B)  │ STATUS")
    print("─"*100)
    print(f"Total Signals       │  {FOUNDATION['total_signals']:>6d}          │ {len(phase2_sigs):>6d}         │ {'📊' if len(phase2_sigs) > 0 else '⏳'}")
    print(f"Closed Trades       │  {FOUNDATION['closed_trades']:>6d}          │ {p2_closed:>6d}         │ {'✅' if p2_closed > 0 else '⏳'}")
    print("─"*100)
    print(f"Win Rate            │  {FOUNDATION['win_rate']:>6.1f}%         │ {p2_wr:>6.1f}%        │ {'✅ APPROVE' if p2_wr >= FOUNDATION['win_rate'] else '⏳ COLLECTING'}")
    print(f"LONG WR             │  {FOUNDATION['long_wr']:>6.1f}%         │ {p2_long_wr:>6.1f}%        │ {'📈' if p2_longs > 10 else '⏳' if p2_longs > 0 else '⏸️'}")
    print(f"SHORT WR            │  {FOUNDATION['short_wr']:>6.1f}%         │ {p2_short_wr:>6.1f}%        │ {'📈' if p2_shorts > 10 else '⏳' if p2_shorts > 0 else '⏸️'}")
    print("─"*100)
    print(f"Total P&L           │  ${FOUNDATION['pnl']:>9.2f}      │  TBD         │ TBD")
    print()
    print("="*100)
    print(f"📌 Success Criterion: Phase 2-FIXED WR ≥ {FOUNDATION['win_rate']}%")
    print(f"📌 Decision Date: Mar 10, 2026 14:30 GMT+7 (Day 7)")
    print(f"📌 Current Progress: {len(phase2_sigs)} signals collected | {p2_closed} closed")
    print(f"   → LONG: {p2_longs} closed ({p2_long_wins} TP) | SHORT: {p2_shorts} closed ({p2_short_wins} TP)")
    print("="*100 + "\n")

if __name__ == "__main__":
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
