#!/usr/bin/env python3
"""
🔒 A/B TEST COMPARISON - Phase 1 (LOCKED BASELINE) vs Phase 2-FIXED
Uses OFFICIAL LOCKED baseline values (immune to file overwrites)
"""

import json
from datetime import datetime, timezone
import os, sys, time

PHASE1_CUTOFF = datetime(2026, 3, 3, 13, 16, 0, tzinfo=timezone.utc)
SIGNALS_FILE = "SENT_SIGNALS.jsonl"

# 🔒 PHASE 1 BASELINE - OFFICIALLY LOCKED (DO NOT CHANGE)
PHASE1_LOCKED = {
    "total_signals": 1205,
    "closed_trades": 1052,
    "win_rate": 29.66,
    "pnl": -5727.12,
    "long_wr": 27.69,
    "short_wr": 43.51
}

def get_phase2_signals():
    """Load ONLY Phase 2-FIXED signals (after 13:16 UTC)"""
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
                    if fired >= PHASE1_CUTOFF:
                        phase2.append(sig)
                except:
                    continue
    except:
        pass
    return phase2

def main():
    print("\n" + "="*120)
    print("🧪 A/B TEST - PHASE 1 (LOCKED BASELINE) vs PHASE 2-FIXED (FRESH)")
    print("="*120)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"Baseline Locked: {PHASE1_LOCKED['total_signals']} signals | {PHASE1_LOCKED['win_rate']}% WR | {PHASE1_LOCKED['pnl']}$ P&L")
    print("="*120)
    print()
    
    # Get Phase 2-FIXED metrics
    phase2_sigs = get_phase2_signals()
    p2_closed = len([s for s in phase2_sigs if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']])
    p2_wins = len([s for s in phase2_sigs if s.get('status') in ['TP_HIT']])
    p2_wr = (p2_wins / p2_closed * 100) if p2_closed > 0 else 0
    
    # Build comparison table
    print(f"METRIC              │  PHASE 1 (A)        │ PHASE 2-FIXED (B)  │ DELTA                │ STATUS")
    print("─"*120)
    
    print(f"Total Signals       │ {PHASE1_LOCKED['total_signals']:6d}        │ {len(phase2_sigs):6d}         │ {len(phase2_sigs) - PHASE1_LOCKED['total_signals']:+6d}             │ 📊")
    print(f"Closed Trades       │ {PHASE1_LOCKED['closed_trades']:6d}        │ {p2_closed:6d}         │ {p2_closed - PHASE1_LOCKED['closed_trades']:+6d}             │ ✓")
    print("─"*120)
    
    print(f"Win Rate            │ {PHASE1_LOCKED['win_rate']:6.2f}%        │ {p2_wr:6.2f}%        │ {p2_wr - PHASE1_LOCKED['win_rate']:+6.2f}%            │ {'✅ ON TRACK' if p2_wr >= PHASE1_LOCKED['win_rate'] else '⏳ COLLECTING'}")
    print(f"  LONG WR           │ {PHASE1_LOCKED['long_wr']:6.2f}%        │  TBD                │  TBD                  │")
    print(f"  SHORT WR          │ {PHASE1_LOCKED['short_wr']:6.2f}%        │  TBD                │  TBD                  │")
    print("─"*120)
    
    print(f"Total P&L (USD)     │ ${PHASE1_LOCKED['pnl']:9.2f}    │ ${0:9.2f}    │ ${0 - PHASE1_LOCKED['pnl']:+9.2f}    │")
    print()
    print("="*120)
    print("📊 DECISION FRAMEWORK (Mar 10)")
    print("="*120)
    print(f"Phase 1 LOCKED Baseline: {PHASE1_LOCKED['total_signals']} signals | {PHASE1_LOCKED['win_rate']}% WR")
    print(f"Phase 2-FIXED Status: {len(phase2_sigs)} signals collected | Closed: {p2_closed}")
    print(f"Decision Criteria: Phase 2-FIXED WR must be > {PHASE1_LOCKED['win_rate']}%")
    print("="*120 + "\n")

if __name__ == "__main__":
    if '--once' in sys.argv:
        main()
    else:
        try:
            while True:
                main()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n✅ Stopped")
