#!/usr/bin/env python3
"""
📊 RR VARIANT COMPARISON - Live Tracker
Compares 2.0:1 (PROD) vs 1.5:1 (Phase 2-FIXED)

⚠️ NOTE: 3.0:1 RR signals (105 from Feb 27-28 pre-optimization era) excluded from this analysis
"""

import json
from datetime import datetime, timezone
from collections import defaultdict

SIGNALS_FILE = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"

class RRTracker:
    def __init__(self):
        self.signals_2_0 = []
        self.signals_1_5 = []
        self.load_signals()
    
    def load_signals(self):
        """Load signals, separate by RR variant (exclude 3.0)"""
        try:
            with open(SIGNALS_FILE, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line.strip())
                        if not sig:
                            continue
                        
                        # Skip 3.0 RR (legacy, pre-optimization)
                        rr = sig.get('achieved_rr')
                        if rr == 3.0:
                            continue
                        
                        # Skip OPEN signals
                        if sig.get('status') == 'OPEN':
                            continue
                        
                        # Skip stale timeouts
                        flag = sig.get('data_quality_flag', '')
                        if flag and 'STALE_TIMEOUT' in flag:
                            continue
                        
                        if rr == 2.0:
                            self.signals_2_0.append(sig)
                        elif rr == 1.5:
                            self.signals_1_5.append(sig)
                    except:
                        continue
        except FileNotFoundError:
            print(f"❌ {SIGNALS_FILE} not found")
    
    def calculate_metrics(self, signals):
        """Calculate RR variant metrics"""
        if not signals:
            return None
        
        tp_hits = len([s for s in signals if s.get('status') == 'TP_HIT'])
        sl_hits = len([s for s in signals if s.get('status') == 'SL_HIT'])
        timeout_trades = [s for s in signals if s.get('status') == 'TIMEOUT']
        timeout_wins = len([s for s in timeout_trades if float(s.get('pnl_usd', 0)) > 0])
        open_count = len([s for s in signals if s.get('status') == 'OPEN'])
        
        closed = len(signals) - open_count
        wins = tp_hits + timeout_wins
        wr = (wins / closed * 100) if closed > 0 else 0
        
        # P&L
        total_pnl = sum(float(s.get('pnl_usd', 0) or 0) for s in signals)
        
        # Duration
        tp_durations = []
        sl_durations = []
        
        for s in signals:
            if s.get('status') == 'TP_HIT':
                fired = datetime.fromisoformat(s.get('fired_time_utc', '').split('+')[0]).replace(tzinfo=timezone.utc)
                closed_at = datetime.fromisoformat(s.get('closed_at', '').split('+')[0]).replace(tzinfo=timezone.utc) if s.get('closed_at') else None
                if closed_at:
                    duration = (closed_at - fired).total_seconds() / 3600  # hours
                    tp_durations.append(duration)
            
            elif s.get('status') == 'SL_HIT':
                fired = datetime.fromisoformat(s.get('fired_time_utc', '').split('+')[0]).replace(tzinfo=timezone.utc)
                closed_at = datetime.fromisoformat(s.get('closed_at', '').split('+')[0]).replace(tzinfo=timezone.utc) if s.get('closed_at') else None
                if closed_at:
                    duration = (closed_at - fired).total_seconds() / 3600  # hours
                    sl_durations.append(duration)
        
        avg_tp_dur = sum(tp_durations) / len(tp_durations) if tp_durations else 0
        avg_sl_dur = sum(sl_durations) / len(sl_durations) if sl_durations else 0
        
        return {
            'total_signals': len(signals),
            'closed': closed,
            'open': open_count,
            'tp_hits': tp_hits,
            'sl_hits': sl_hits,
            'timeouts': len(timeout_trades),
            'timeout_wins': timeout_wins,
            'wr': wr,
            'total_pnl': total_pnl,
            'avg_tp_dur': avg_tp_dur,
            'avg_sl_dur': avg_sl_dur,
        }
    
    def format_duration(self, hours):
        """Format hours as h:mm"""
        if hours == 0:
            return "0h 0m"
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h}h {m:02d}m"
    
    def print_report(self):
        """Generate RR comparison report"""
        m2_0 = self.calculate_metrics(self.signals_2_0)
        m1_5 = self.calculate_metrics(self.signals_1_5)
        
        print("\n" + "="*110)
        print("📊 RR VARIANT COMPARISON (2.0:1 PROD vs 1.5:1 Phase 2-FIXED)")
        print("="*110)
        print()
        print("⏱️ CUTOFF TIME BETWEEN RR VARIANTS:")
        print("   • 2.0:1 RR (PROD):      Feb 27 15:55 UTC → Mar 4 17:51 UTC (895 signals)")
        print("   • 1.5:1 RR (Phase 2):   Mar 4 17:51 UTC → Mar 5 04:29 UTC (97 signals)")
        print("   • Deployment:           2026-03-04 20:32 GMT+7 (commit e3f110b)")
        print()
        print("⚠️  3.0:1 RR (Legacy):      105 signals from Feb 27-28 (pre-optimization era) - EXCLUDED")
        print()
        print("="*110)
        print()
        
        if not m2_0 or not m1_5:
            print("❌ Insufficient data for comparison")
            return
        
        # Header row
        print(f"{'RR':<8} | {'Total':>6} | {'TP':>4} | {'SL':>4} | {'TIMEOUT':>7} | {'OPEN':>4} | {'WR %':>6} | {'P&L $':>10} | {'TP Dur':>9} | {'SL Dur':>9}")
        print("─"*110)
        
        # 2.0 RR row
        print(f"{'2.0:1':<8} | {m2_0['total_signals']:>6d} | {m2_0['tp_hits']:>4d} | {m2_0['sl_hits']:>4d} | {m2_0['timeouts']:>7d} | {m2_0['open']:>4d} | {m2_0['wr']:>6.2f} | ${m2_0['total_pnl']:>9.0f} | {self.format_duration(m2_0['avg_tp_dur']):>9} | {self.format_duration(m2_0['avg_sl_dur']):>9}")
        
        # 1.5 RR row
        print(f"{'1.5:1':<8} | {m1_5['total_signals']:>6d} | {m1_5['tp_hits']:>4d} | {m1_5['sl_hits']:>4d} | {m1_5['timeouts']:>7d} | {m1_5['open']:>4d} | {m1_5['wr']:>6.2f} | ${m1_5['total_pnl']:>9.0f} | {self.format_duration(m1_5['avg_tp_dur']):>9} | {self.format_duration(m1_5['avg_sl_dur']):>9}")
        
        print()
        print("="*110)
        print("📊 IMPACT ANALYSIS")
        print("="*110)
        print()
        
        delta_wr = m1_5['wr'] - m2_0['wr']
        delta_pnl = m1_5['total_pnl'] - m2_0['total_pnl']
        delta_tp_dur = m1_5['avg_tp_dur'] - m2_0['avg_tp_dur']
        delta_sl_dur = m1_5['avg_sl_dur'] - m2_0['avg_sl_dur']
        
        print(f"Comparing 2.0:1 (PROD) vs 1.5:1 (Phase 2):")
        print()
        print(f"  WR Change:        {m2_0['wr']:.2f}% → {m1_5['wr']:.2f}% ({delta_wr:+.2f}pp) {'✅ IMPROVING' if delta_wr > 0 else '⚠️ DECLINING'}")
        print(f"  TP Duration:      {self.format_duration(m2_0['avg_tp_dur'])} → {self.format_duration(m1_5['avg_tp_dur'])} ({delta_tp_dur:+.1f}h)")
        print(f"  SL Duration:      {self.format_duration(m2_0['avg_sl_dur'])} → {self.format_duration(m1_5['avg_sl_dur'])} ({delta_sl_dur:+.1f}h)")
        print(f"  P&L Change:       ${m2_0['total_pnl']:,.0f} → ${m1_5['total_pnl']:,.0f} ({delta_pnl:+,.0f})")
        print(f"  Signal Volume:    {m2_0['total_signals']} → {m1_5['total_signals']} ({m1_5['total_signals']-m2_0['total_signals']:+d} signals)")
        
        print()
        print("="*110)
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
        print("="*110 + "\n")

if __name__ == "__main__":
    import sys
    import time
    import subprocess
    import os
    
    def clear_screen():
        subprocess.call('clear' if os.name == 'posix' else 'cls', shell=True)
    
    if '--once' in sys.argv:
        tracker = RRTracker()
        tracker.print_report()
    else:
        # Live watch mode
        try:
            while True:
                clear_screen()
                tracker = RRTracker()
                tracker.print_report()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n✓ Stopped")
