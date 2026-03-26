#!/usr/bin/env python3
"""
POST-DEPLOYMENT TRACKER - 3-Factor + 4-Factor Normalization (2026-03-26 08:54 GMT+7)

Purpose: Track signal performance AFTER 3-factor and 4-factor normalization deployment
Source: SIGNALS_MASTER.jsonl (same as pec_enhanced_reporter.py)
Cut-off: 2026-03-25T17:54:00Z (00:54 GMT+7 2026-03-26 deployment timestamp - onwards only)

Code Changes Applied:
- 3-Factor: smart_filter.py - volatility-based thresholds, S/R proximity, reversal frequency
- 4-Factor: main.py - dynamic MIN_SCORE (8 for 2h/4h instead of global 10)

Protocol: Each code deployment creates NEW tracker with same source, different cut-off.
This prevents silent drift by isolating code changes from market conditions.

Pre-Deployment Baseline: pec_enhanced_reporter.py (locked, <= 2026-03-26T08:54:00Z)
Post-Deployment Baseline: this tracker (new, >= 2026-03-26T08:54:00Z)
"""

import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os

# DEPLOYMENT CUT-OFF TIMESTAMP (when daemon restarted with 3-factor + 4-factor)
# 00:54 GMT+7 (2026-03-26) = 17:54 UTC (2026-03-25)
DEPLOYMENT_CUTOFF_UTC = datetime.fromisoformat('2026-03-25T17:54:00+00:00')

class PostDeploymentTracker:
    def __init__(self, signals_file=None):
        if signals_file is None:
            workspace = "/Users/geniustarigan/.openclaw/workspace"
            signals_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")
        
        self.signals_file = signals_file
        self.signals = []
        self.load_signals()
    
    def load_signals(self):
        """Load signals from SIGNALS_MASTER.jsonl, filter by deployment cut-off"""
        if not os.path.exists(self.signals_file):
            print(f"[WARN] {self.signals_file} not found")
            return
        
        try:
            with open(self.signals_file, 'r') as f:
                count = 0
                filtered = 0
                for line in f:
                    try:
                        signal = json.loads(line.strip())
                        count += 1
                        
                        # Filter: only signals fired >= deployment cut-off
                        fired_str = signal.get('fired_time_utc', '')
                        if not fired_str:
                            continue
                        
                        # Parse UTC time
                        try:
                            fired_dt = datetime.fromisoformat(fired_str.replace('Z', '+00:00'))
                            if fired_dt.tzinfo is None:
                                fired_dt = fired_dt.replace(tzinfo=timezone.utc)
                        except:
                            continue
                        
                        # Include only if >= deployment cut-off
                        if fired_dt >= DEPLOYMENT_CUTOFF_UTC:
                            # Normalize field names
                            if 'direction' in signal and 'signal_type' not in signal:
                                signal['signal_type'] = signal['direction']
                            self.signals.append(signal)
                            filtered += 1
                    except:
                        pass
                
                print(f"[INFO] Loaded {filtered} post-deployment signals (from {count} total in {os.path.basename(self.signals_file)})", flush=True)
                print(f"[INFO] Cut-off: 2026-03-26T08:54:00Z onwards", flush=True)
        except Exception as e:
            print(f"[WARN] Error loading signals: {e}")
    
    def generate_summary(self):
        """Generate quick summary of post-deployment signals"""
        report = []
        
        report.append("")
        report.append("=" * 140)
        report.append("📊 POST-DEPLOYMENT TRACKER (3-Factor + 4-Factor Normalization)")
        report.append("=" * 140)
        report.append("")
        report.append(f"Deployment Cut-off: 2026-03-25T17:54:00Z / 00:54 GMT+7 (2026-03-26) — onwards")
        report.append(f"Report Generated: {datetime.now(timezone(timedelta(hours=7))).strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
        report.append("")
        
        if not self.signals:
            report.append("⏳ No post-deployment signals yet (still accumulating)")
            report.append("")
            return "\n".join(report)
        
        # Signal counts by status
        tp = sum(1 for s in self.signals if s.get('status') == 'TP_HIT')
        sl = sum(1 for s in self.signals if s.get('status') == 'SL_HIT')
        timeout = sum(1 for s in self.signals if s.get('status') == 'TIMEOUT')
        open_trades = sum(1 for s in self.signals if s.get('status') == 'OPEN')
        rejected = sum(1 for s in self.signals if s.get('status') == 'REJECTED_NOT_SENT_TELEGRAM')
        stale = sum(1 for s in self.signals if s.get('status') == 'STALE_TIMEOUT')
        
        total = len(self.signals)
        
        report.append(f"Total Post-Deployment Signals: {total}")
        report.append(f"  TP_HIT: {tp}")
        report.append(f"  SL_HIT: {sl}")
        report.append(f"  TIMEOUT: {timeout}")
        report.append(f"  OPEN: {open_trades}")
        report.append(f"  REJECTED: {rejected}")
        report.append(f"  STALE: {stale}")
        report.append("")
        
        # Closed trades
        closed = tp + sl + timeout
        if closed > 0:
            # Separate timeout into wins/losses
            timeout_wins = 0
            for s in self.signals:
                if s.get('status') == 'TIMEOUT' and s.get('actual_exit_price'):
                    entry = float(s.get('entry_price', 0))
                    exit_p = float(s.get('actual_exit_price', 0))
                    direction = s.get('signal_type', 'LONG')
                    
                    if entry > 0 and exit_p > 0:
                        if direction.upper() == 'LONG':
                            pnl = ((exit_p - entry) / entry) * 1000
                        else:
                            pnl = ((entry - exit_p) / exit_p) * 1000
                        
                        if pnl > 0:
                            timeout_wins += 1
            
            wins = tp + timeout_wins
            wr = (wins / closed * 100) if closed > 0 else 0
            
            report.append(f"Closed Trades: {closed}")
            report.append(f"  TP: {tp} | SL: {sl} | TIMEOUT: {timeout} (W:{timeout_wins} L:{timeout - timeout_wins})")
            report.append(f"  Win Rate: {wr:.2f}%")
            report.append("")
        
        # By timeframe
        report.append("By Timeframe:")
        tf_stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'open': 0})
        for s in self.signals:
            tf = s.get('timeframe', 'N/A')
            status = s.get('status', 'OPEN')
            tf_stats[tf]['count'] += 1
            
            if status == 'TP_HIT':
                tf_stats[tf]['tp'] += 1
            elif status == 'SL_HIT':
                tf_stats[tf]['sl'] += 1
            elif status == 'TIMEOUT':
                tf_stats[tf]['timeout'] += 1
            elif status == 'OPEN':
                tf_stats[tf]['open'] += 1
        
        for tf in sorted(tf_stats.keys()):
            stats = tf_stats[tf]
            report.append(f"  {tf}: {stats['count']} (TP:{stats['tp']} SL:{stats['sl']} TIMEOUT:{stats['timeout']} OPEN:{stats['open']})")
        
        report.append("")
        report.append("=" * 140)
        
        return "\n".join(report)

if __name__ == "__main__":
    tracker = PostDeploymentTracker()
    report = tracker.generate_summary()
    print(report)
    
    # Save to file
    with open("PEC_POST_DEPLOYMENT_TRACKER.txt", "w") as f:
        f.write(report)
    
    print("\n✅ Report saved to PEC_POST_DEPLOYMENT_TRACKER.txt")
