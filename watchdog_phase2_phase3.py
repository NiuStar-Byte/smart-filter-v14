#!/usr/bin/env python3
"""
ENHANCED WATCHDOG - Phase 2 + Phase 3 Health Monitor

Monitors:
1. Process health (is daemon running?)
2. Phase 2 metrics (HARD-GATES firing rate, gate effectiveness)
3. Phase 3 metrics (route optimization, direction enforcement)
4. Performance thresholds (alert if WR degrades)

Runs every 5 minutes and reports Phase 2/3 specific metrics.
"""

import subprocess
import time
import os
import sys
import json
import re
from datetime import datetime, timedelta
from collections import defaultdict

# Configuration
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(WORK_DIR, "watchdog_phase2_phase3.log")
MAIN_LOG = os.path.join(WORK_DIR, "main_daemon.log")
SIGNALS_FILE = os.path.join(WORK_DIR, "SENT_SIGNALS.jsonl")
CHECK_INTERVAL = 300  # 5 minutes

# Phase thresholds
PHASE2_MIN_WR = 0.35  # Should be improving toward 35%+
PHASE3_MIN_WR = 0.36  # Should be improving toward 36%+
GATE_PASS_RATE_MIN = 0.4  # At least 40% of signals should pass gates

class PhaseWatchdog:
    def __init__(self):
        self.phase2_cutoff = datetime(2026, 3, 2, 11, 4, 0)  # Mar 2 18:04 GMT+7 = 11:04 UTC
        self.phase3_start = datetime(2026, 3, 2, 14, 30, 0)   # Mar 2 21:30 GMT+7 = 14:30 UTC
        self.last_check_line = 0
        
    def log(self, msg, level="INFO"):
        """Log with timestamp and level."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S GMT+7")
        prefix = f"[{ts}]"
        
        if level == "ERROR":
            prefix += " ❌"
        elif level == "WARN":
            prefix += " ⚠️"
        elif level == "OK":
            prefix += " ✅"
        elif level == "INFO":
            prefix += " ℹ️"
        
        line = f"{prefix} {msg}"
        print(line)
        
        try:
            with open(LOG_FILE, "a") as f:
                f.write(line + "\n")
        except:
            pass
    
    def check_process_health(self):
        """Check if main.py and pec_executor.py are running."""
        try:
            # Check main.py
            main_running = subprocess.run(
                ["pgrep", "-f", "python3 main.py"],
                capture_output=True,
                timeout=5
            ).returncode == 0
            
            # Check pec_executor.py
            pec_running = subprocess.run(
                ["pgrep", "-f", "python3 pec_executor.py"],
                capture_output=True,
                timeout=5
            ).returncode == 0
            
            return {
                'main.py': main_running,
                'pec_executor.py': pec_running,
                'both_running': main_running and pec_running
            }
        except Exception as e:
            self.log(f"Process check failed: {e}", "ERROR")
            return {'main.py': False, 'pec_executor.py': False, 'both_running': False}
    
    def parse_recent_logs(self, lines=500):
        """Parse recent log lines for Phase 2 + Phase 3 tags."""
        try:
            if not os.path.exists(MAIN_LOG):
                return None
            
            with open(MAIN_LOG, 'r') as f:
                all_lines = f.readlines()
            
            # Get last N lines
            recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
            log_text = ''.join(recent)
            
            return {
                'phase2_gates': self._count_pattern(log_text, r'\[HARD-GATES\]'),
                'phase2_scores': self._count_pattern(log_text, r'\[SCORE-ADJUSTED\]'),
                'phase3_trend': self._count_pattern(log_text, r'\[PHASE3-TREND\]'),
                'phase3_reversal': self._count_pattern(log_text, r'\[PHASE3-REVERSAL\]'),
                'phase3_filtered': self._count_pattern(log_text, r'\[PHASE3-FILTERED\]'),
                'errors': self._count_pattern(log_text, r'\[ERROR\]'),
            }
        except Exception as e:
            self.log(f"Log parsing failed: {e}", "ERROR")
            return None
    
    def _count_pattern(self, text, pattern):
        """Count occurrences of regex pattern."""
        return len(re.findall(pattern, text))
    
    def check_phase2_metrics(self):
        """Analyze Phase 2 performance from signals file."""
        try:
            phase2_signals = []
            
            with open(SIGNALS_FILE, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line)
                        if not sig or sig.get('status') == 'OPEN':
                            continue
                        
                        fired_str = sig.get('fired_time_utc', '')
                        if not fired_str:
                            continue
                        
                        fired = datetime.fromisoformat(fired_str.split('+')[0])
                        
                        if self.phase2_cutoff <= fired < self.phase3_start:
                            phase2_signals.append(sig)
                    except:
                        pass
            
            if not phase2_signals:
                return None
            
            # Calculate metrics
            closed = [s for s in phase2_signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']]
            wins = len([s for s in closed if s.get('status') == 'TP_HIT'])
            timeout_wins = len([s for s in closed if s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd', 0)) > 0])
            
            wr = ((wins + timeout_wins) / len(closed) * 100) if closed else 0
            pnl = sum(float(s.get('pnl_usd', 0)) for s in closed)
            
            # LONG/SHORT breakdown
            long_trades = [s for s in closed if s.get('signal_type') == 'LONG']
            long_wins = sum(1 for s in long_trades if s.get('status') == 'TP_HIT')
            long_timeout_wins = sum(1 for s in long_trades if s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd', 0)) > 0)
            long_wr = ((long_wins + long_timeout_wins) / len(long_trades) * 100) if long_trades else 0
            
            return {
                'signals': len(phase2_signals),
                'closed_trades': len(closed),
                'wr': wr,
                'long_wr': long_wr,
                'pnl': pnl,
                'status': 'OK' if wr >= PHASE2_MIN_WR else 'WATCH'
            }
        except Exception as e:
            self.log(f"Phase 2 metrics failed: {e}", "ERROR")
            return None
    
    def check_phase3_metrics(self):
        """Analyze Phase 3 performance from signals file."""
        try:
            phase3_signals = []
            
            with open(SIGNALS_FILE, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line)
                        if not sig or sig.get('status') == 'OPEN':
                            continue
                        
                        # Skip stale timeouts
                        if sig.get('data_quality_flag') and 'STALE_TIMEOUT' in sig['data_quality_flag']:
                            continue
                        
                        fired_str = sig.get('fired_time_utc', '')
                        if not fired_str:
                            continue
                        
                        fired = datetime.fromisoformat(fired_str.split('+')[0])
                        
                        if fired >= self.phase3_start:
                            phase3_signals.append(sig)
                    except:
                        pass
            
            if not phase3_signals:
                return None
            
            # Calculate metrics
            closed = [s for s in phase3_signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']]
            wins = len([s for s in closed if s.get('status') == 'TP_HIT'])
            timeout_wins = len([s for s in closed if s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd', 0)) > 0])
            
            wr = ((wins + timeout_wins) / len(closed) * 100) if closed else 0
            pnl = sum(float(s.get('pnl_usd', 0)) for s in closed)
            
            # Route breakdown
            routes = defaultdict(lambda: {'closed': 0, 'wins': 0})
            for s in closed:
                route = s.get('route', 'UNKNOWN')
                routes[route]['closed'] += 1
                if s.get('status') == 'TP_HIT' or (s.get('status') == 'TIMEOUT' and float(s.get('pnl_usd', 0)) > 0):
                    routes[route]['wins'] += 1
            
            return {
                'signals': len(phase3_signals),
                'closed_trades': len(closed),
                'wr': wr,
                'pnl': pnl,
                'routes': dict(routes),
                'status': 'OK' if wr >= PHASE3_MIN_WR else 'WATCH'
            }
        except Exception as e:
            self.log(f"Phase 3 metrics failed: {e}", "ERROR")
            return None
    
    def report(self):
        """Generate comprehensive health report."""
        print("\n" + "="*100)
        print("🔍 PHASE 2 + PHASE 3 HEALTH REPORT")
        print("="*100)
        
        # 1. Process Health
        print("\n[1] PROCESS HEALTH")
        proc_health = self.check_process_health()
        if proc_health['both_running']:
            self.log("Both main.py and pec_executor.py running", "OK")
        else:
            if not proc_health['main.py']:
                self.log("main.py NOT RUNNING", "ERROR")
            if not proc_health['pec_executor.py']:
                self.log("pec_executor.py NOT RUNNING", "ERROR")
        
        # 2. Log Activity
        print("\n[2] RECENT LOG ACTIVITY (last 500 lines)")
        logs = self.parse_recent_logs()
        if logs:
            print(f"  Phase 2 Gates Fired:       {logs['phase2_gates']} times")
            print(f"  Phase 2 Score Adjustments: {logs['phase2_scores']} times")
            print(f"  Phase 3 TREND Signals:     {logs['phase3_trend']} times")
            print(f"  Phase 3 REVERSAL Signals:  {logs['phase3_reversal']} times")
            print(f"  Phase 3 Filtered:          {logs['phase3_filtered']} times")
            print(f"  Errors:                    {logs['errors']} times")
        
        # 3. Phase 2 Metrics
        print("\n[3] PHASE 2 METRICS (Mar 2 18:04 - 21:30 GMT+7)")
        p2_metrics = self.check_phase2_metrics()
        if p2_metrics:
            print(f"  Total Signals:   {p2_metrics['signals']}")
            print(f"  Closed Trades:   {p2_metrics['closed_trades']}")
            print(f"  Overall WR:      {p2_metrics['wr']:.2f}% (target: 35%+)")
            print(f"  LONG WR:         {p2_metrics['long_wr']:.2f}%")
            print(f"  Total P&L:       ${p2_metrics['pnl']:.2f}")
            print(f"  Status:          {'✅ ON TRACK' if p2_metrics['status'] == 'OK' else '⚠️ WATCH'}")
        else:
            print(f"  No Phase 2 data yet (still collecting)")
        
        # 4. Phase 3 Metrics
        print("\n[4] PHASE 3 METRICS (Mar 2 21:30+ GMT+7)")
        p3_metrics = self.check_phase3_metrics()
        if p3_metrics and p3_metrics['closed_trades'] > 0:
            print(f"  Total Signals:   {p3_metrics['signals']}")
            print(f"  Closed Trades:   {p3_metrics['closed_trades']}/20 (target for decision)")
            print(f"  Overall WR:      {p3_metrics['wr']:.2f}% (target: 36%+)")
            print(f"  Total P&L:       ${p3_metrics['pnl']:.2f}")
            print(f"  Status:          {'✅ ON TRACK' if p3_metrics['status'] == 'OK' else '⚠️ WATCH'}")
            
            if p3_metrics['routes']:
                print(f"\n  Route Breakdown:")
                for route, stats in p3_metrics['routes'].items():
                    if stats['closed'] > 0:
                        wr = (stats['wins'] / stats['closed'] * 100)
                        print(f"    {route}: {wr:.1f}% WR ({stats['closed']} trades)")
        else:
            print(f"  Minimal Phase 3 data (still collecting)")
        
        print("\n" + "="*100 + "\n")
    
    def run(self):
        """Main watchdog loop."""
        self.log("🦇 Phase 2 + Phase 3 Watchdog started")
        
        while True:
            try:
                self.report()
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                self.log("Watchdog stopped by user", "INFO")
                sys.exit(0)
            except Exception as e:
                self.log(f"Watchdog error: {e}", "ERROR")
                time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    watchdog = PhaseWatchdog()
    watchdog.run()
