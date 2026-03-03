#!/usr/bin/env python3
"""
🔒 PHASE 3 TRACKER - HISTORICAL REFERENCE ONLY

⚠️ STATUS: PHASE 3 WAS REVERTED (2026-03-03 14:50 GMT+7)
   Phase 3 (route optimization) was found to collapse SHORT signals
   Now running Phase 2-FIXED + Phase 3B instead

🔒 FOUNDATION BASELINE - LOCKED AT 853 SIGNALS (2026-03-04 01:10 GMT+7):
  - All signals in current dataset analyzed by pec_enhanced_reporter.py
  - 25.7% WR (locked, never changes)
  - Used across ALL comparisons (COMPARE_AB_TEST_LOCKED, Phase 3, Phase 4A)

⚠️ PHASE 3 WINDOW (Historical - Empty):
  - Time window: Mar 2 14:30 UTC to Mar 3 13:16 UTC
  - Status: REVERTED - not actively running
  - Signals in this window: 0 (we're now collecting Phase 2-FIXED from 13:16 UTC onwards)
  - This tracker shows historical data only (for reference/analysis)

Current Active Phases:
  ✅ Phase 2-FIXED - Direction-aware gates (live from Mar 3 13:16 UTC)
  ✅ Phase 3B - Reversal quality gates (parallel with Phase 2-FIXED)
  ✅ Phase 4A - Multi-TF alignment filter (independent)

Usage:
  python3 PHASE3_TRACKER.py --once
"""

import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta

PHASE1_CUTOFF = datetime(2026, 3, 3, 13, 16, 0, tzinfo=timezone.utc)  # Mar 03 20:16 GMT+7 = 13:16 UTC (CRITICAL FIXES) - LOCKED BASELINE
PHASE3_START = datetime(2026, 3, 2, 14, 30, 0, tzinfo=timezone.utc)  # Mar 2 21:30 GMT+7 = 14:30 UTC
PHASE3_END = datetime(2026, 3, 3, 13, 16, 0, tzinfo=timezone.utc)    # Mar 03 20:16 GMT+7 = 13:16 UTC (when Phase 2-FIXED critical fixes deployed)
SIGNALS_FILE = "SENT_SIGNALS.jsonl"

# 🔒 FOUNDATION BASELINE LOCKED (2026-03-04 01:10 GMT+7)
# NO MORE CONFLICTING NUMBERS - THIS IS THE ONLY BASELINE USED
PHASE1_LOCKED = {
    "total_signals": 853,
    "closed_trades": 830,
    "win_rate": 25.7,
    "pnl": -5498.59,
    "long_wr": 29.6,
    "short_wr": 46.2
}

class Phase3Tracker:
    def __init__(self):
        self.phase1_signals = []
        self.phase3_signals = []
        self.load_signals()
    
    def load_signals(self):
        """Split signals into Phase 1 vs Phase 3"""
        try:
            with open(SIGNALS_FILE, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line.strip())
                        if not sig:
                            continue
                        
                        # Skip OPEN signals
                        if sig.get('status') == 'OPEN':
                            continue
                        
                        fired_str = sig.get('fired_time_utc') or sig.get('fired_at', '')
                        if not fired_str:
                            continue
                        
                        # Parse as naive datetime and make UTC
                        fired = datetime.fromisoformat(fired_str.split('+')[0]).replace(tzinfo=timezone.utc)
                        
                        # PHASE 1: Everything BEFORE 13:16 UTC Mar 3 (LOCKED FOUNDATION - 853 signals)
                        if fired < PHASE1_CUTOFF:
                            self.phase1_signals.append(sig)
                        # PHASE 3: From 14:30 UTC Mar 2 TO 13:16 UTC Mar 3 (Route-optimized period)
                        elif PHASE3_START <= fired < PHASE3_END:
                            self.phase3_signals.append(sig)
                    except Exception as e:
                        continue
        except FileNotFoundError:
            print(f"❌ {SIGNALS_FILE} not found")
    
    def calculate_metrics(self, signals, phase_name=""):
        """Calculate comprehensive metrics (excluding stale timeouts - matches COMPARE_AB_TEST)"""
        if not signals:
            return None
        
        # Filter: closed trades + clean data (no stale timeouts)
        closed = []
        for s in signals:
            if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                # Exclude stale timeouts (marked with data_quality_flag)
                flag = s.get('data_quality_flag', '')
                if flag and 'STALE_TIMEOUT' in flag:
                    continue  # Skip stale timeout
                closed.append(s)
        
        if not closed:
            return None
        
        # Win rate calculation
        tp_hits = len([s for s in closed if s.get('status') == 'TP_HIT'])
        sl_hits = len([s for s in closed if s.get('status') == 'SL_HIT'])
        timeout_trades = [s for s in closed if s.get('status') == 'TIMEOUT']
        timeout_wins = len([s for s in timeout_trades if float(s.get('pnl_usd', 0)) > 0])
        
        win_rate = ((tp_hits + timeout_wins) / len(closed) * 100) if closed else 0
        
        # P&L
        total_pnl = sum(float(s.get('pnl_usd', 0) or 0) for s in closed)
        
        # Route breakdown
        routes = defaultdict(lambda: {'wins': 0, 'closed': 0, 'pnl': 0.0})
        for s in closed:
            route = s.get('route', 'UNKNOWN')
            routes[route]['closed'] += 1
            routes[route]['pnl'] += float(s.get('pnl_usd', 0) or 0)
            
            if s.get('status') == 'TP_HIT':
                routes[route]['wins'] += 1
            elif s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd', 0)) > 0:
                routes[route]['wins'] += 1
        
        # Direction breakdown
        long_wr = self._calculate_direction_wr(closed, 'LONG')
        short_wr = self._calculate_direction_wr(closed, 'SHORT')
        
        return {
            'total_signals': len(signals),
            'closed_trades': len(closed),
            'win_rate': win_rate,
            'long_wr': long_wr,
            'short_wr': short_wr,
            'total_pnl': total_pnl,
            'routes': dict(routes),
            'tp_hits': tp_hits,
            'sl_hits': sl_hits,
            'timeouts': len(timeout_trades),
            'timeout_wins': timeout_wins,
        }
    
    def _calculate_direction_wr(self, closed, direction):
        """Calculate WR for specific direction"""
        dir_trades = [s for s in closed if s.get('signal_type') == direction]
        if not dir_trades:
            return 0
        
        wins = len([s for s in dir_trades if s.get('status') == 'TP_HIT'])
        timeout_wins = len([s for s in dir_trades if s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd', 0)) > 0])
        
        return ((wins + timeout_wins) / len(dir_trades) * 100) if dir_trades else 0
    
    def _print_route_summary(self, routes, phase_name):
        """Print route breakdown"""
        if not routes:
            print(f"No routes found in {phase_name}")
            return
        
        print(f"Route                  | Closed | Wins | WR    | P&L        | Status")
        print("-" * 75)
        
        for route, stats in sorted(routes.items(), key=lambda x: x[1]['closed'], reverse=True):
            closed = stats['closed']
            wins = stats['wins']
            wr = (wins / closed * 100) if closed > 0 else 0
            pnl = stats['pnl']
            
            if wr > 40:
                status = "✅ Good"
            elif wr > 30:
                status = "⚠️ Marginal"
            else:
                status = "❌ Poor"
            
            print(f"{route:<21} | {closed:6d} | {wins:4d} | {wr:5.1f}% | ${pnl:9.2f} | {status}")
    
    def print_report(self):
        """Generate Phase 3 tracking report"""
        print("\n" + "="*180)
        print("📊 PHASE 3: UNIFIED ROUTE OPTIMIZATION - TRACKING REPORT")
        print("="*180)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
        print(f"🔒 FOUNDATION BASELINE (LOCKED): 853 signals @ 25.7% WR (2026-03-04 01:10 GMT+7)")
        print(f"⚠️ PHASE 3 WINDOW: Mar 2 21:30 - Mar 3 20:16 GMT+7 (REVERTED - Historical reference)")
        print("="*180)
        print()
        
        # PHASE 1: Use LOCKED baseline (never changes) + calculate route breakdown from phase1_signals
        m1_calculated = self.calculate_metrics(self.phase1_signals, "PHASE 1")
        m1 = {
            'total_signals': PHASE1_LOCKED['total_signals'],
            'closed_trades': PHASE1_LOCKED['closed_trades'],
            'win_rate': PHASE1_LOCKED['win_rate'],
            'long_wr': PHASE1_LOCKED['long_wr'],
            'short_wr': PHASE1_LOCKED['short_wr'],
            'total_pnl': PHASE1_LOCKED['pnl'],
            'routes': m1_calculated['routes'] if m1_calculated else {}  # Route breakdown from loaded signals
        }
        
        m3 = self.calculate_metrics(self.phase3_signals, "PHASE 3")
        
        if not m1:
            print("❌ Phase 1 baseline not available")
            return
        
        print("METRIC                          │   FOUNDATION (Locked)  │   PHASE 3 (Reverted)  │   DELTA      │ STATUS")
        print("─"*180)
        
        # Overall metrics
        p3_signals = m3['total_signals'] if m3 else 0
        p3_closed = m3['closed_trades'] if m3 else 0
        delta_signals = (m3['total_signals'] - m1['total_signals']) if m3 else 0
        delta_closed = (m3['closed_trades'] - m1['closed_trades']) if m3 else 0
        
        print(f"Total Signals                   │  {m1['total_signals']:6d}          │  {p3_signals:6d}          │  {delta_signals:+6d}    │ 📊")
        print(f"Closed Trades                   │  {m1['closed_trades']:6d}          │  {p3_closed:6d}          │  {delta_closed:+6d}    │ ✓")
        print("─"*180)
        
        # Win rate
        if m3 and m3['closed_trades'] > 0:
            delta_wr = m3['win_rate'] - m1['win_rate']
            if delta_wr > 2:
                status = "✅ IMPROVING"
            elif delta_wr > 0:
                status = "⚠️ MARGINAL"
            else:
                status = "❌ DECLINING"
            print(f"Overall Win Rate                │  {m1['win_rate']:6.2f}%          │  {m3['win_rate']:6.2f}%          │  {delta_wr:+6.2f}%  │ {status}")
        else:
            print(f"Overall Win Rate                │  {m1['win_rate']:6.2f}%          │  {'Collecting':6s}       │  {'N/A':6s}  │ ⏳")
        
        # LONG/SHORT
        if m3 and m3['closed_trades'] > 0:
            delta_long = m3['long_wr'] - m1['long_wr']
            delta_short = m3['short_wr'] - m1['short_wr']
            print(f"  ├─ LONG WR                    │  {m1['long_wr']:6.2f}%          │  {m3['long_wr']:6.2f}%          │  {delta_long:+6.2f}%  │ {'✅' if delta_long > 0 else '⚠️'}")
            print(f"  └─ SHORT WR                   │  {m1['short_wr']:6.2f}%          │  {m3['short_wr']:6.2f}%          │  {delta_short:+6.2f}%  │ {'✅' if delta_short > 0 else '⚠️'}")
        
        print("─"*180)
        
        # P&L
        if m3 and m3['closed_trades'] > 0:
            delta_pnl = m3['total_pnl'] - m1['total_pnl']
            if delta_pnl > 500:
                status = "✅ PROFITABLE"
            elif delta_pnl > 0:
                status = "⚠️ IMPROVING"
            else:
                status = "❌ WORSE"
            print(f"Total P&L (Clean Data)          │  ${m1['total_pnl']:9.2f}     │  ${m3['total_pnl']:9.2f}     │  ${delta_pnl:+9.2f}  │ {status}")
        else:
            print(f"Total P&L (Clean Data)          │  ${m1['total_pnl']:9.2f}     │  {'Collecting':9s}    │  {'N/A':9s} │ ⏳")
        
        print()
        print("="*180)
        print("🛣️ ROUTE BREAKDOWN (FOUNDATION - 853 Signals Locked)")
        print("="*180)
        print()
        
        self._print_route_summary(m1['routes'], "PHASE 1")
        
        if m3 and m3['closed_trades'] > 0:
            print()
            print("="*180)
            print("🛣️ ROUTE BREAKDOWN (Phase 3 - Reverted Window / Empty)")
            print("="*180)
            print()
            
            self._print_route_summary(m3['routes'], "PHASE 3")
        
        print()
        print("="*180)
        print("📋 PHASE 3 DECISION FRAMEWORK")
        print("="*180)
        print()
        
        if m3 and m3['closed_trades'] >= 20:
            if m3['win_rate'] > 35:
                print("✅ PHASE 3 SUCCESS: WR > 35% - Route filtering is working!")
                print("   Impact: Route optimization achieved +{:.1f}% WR improvement".format(m3['win_rate'] - m1['win_rate']))
                print("   Next: Finalize Phase 3, prepare for next optimization phase")
            elif m3['win_rate'] > 30:
                print("⚠️ PHASE 3 MARGINAL: 30% < WR < 35% - Monitor for more data")
                print("   Impact: Route optimization showing {:.1f}% WR change".format(m3['win_rate'] - m1['win_rate']))
                print("   Next: Collect more trades, assess trend (improving or declining?)")
            else:
                print("❌ PHASE 3 REGRESSION: WR < 30% - Route filtering hurt performance")
                print("   Impact: Route optimization declined {:.1f}%".format(m1['win_rate'] - m3['win_rate']))
                print("   Next: Rollback and revise strategy")
        else:
            trades_needed = 20 - (m3['closed_trades'] if m3 else 0)
            print("⏳ PHASE 3 IN PROGRESS")
            print(f"   Closed trades: {m3['closed_trades'] if m3 else 0}/20 (need {trades_needed} more)")
            print("   Continue monitoring daily")
        
        print()
        print("="*180)

if __name__ == "__main__":
    import sys
    import time
    
    # Check for --once flag (single run)
    once_mode = '--once' in sys.argv
    
    if once_mode:
        tracker = Phase3Tracker()
        tracker.print_report()
    else:
        # Live mode: refresh every 5 seconds
        try:
            while True:
                # Clear screen
                import os
                os.system('clear' if os.name != 'nt' else 'cls')
                
                # Run tracker
                tracker = Phase3Tracker()
                tracker.print_report()
                
                # Print refresh info
                print("\n[Refreshing in 5 seconds... Press Ctrl+C to stop]")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n[Tracker stopped]")
            sys.exit(0)
