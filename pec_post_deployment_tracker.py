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
    
    def _format_duration_hm(self, total_seconds):
        """Format duration as Xh Ym"""
        if total_seconds <= 0:
            return "N/A"
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def _calculate_pnl_usd(self, entry_price, exit_price, direction):
        """Calculate P&L USD using notional position of $1000"""
        try:
            if not entry_price or entry_price == 0 or not exit_price or exit_price == 0:
                return None
            
            entry = float(entry_price)
            exit_val = float(exit_price)
            notional_position = 1000.0
            
            dir_up = str(direction).strip().upper() == "LONG"
            dir_down = str(direction).strip().upper() == "SHORT"
            
            if dir_up:
                pnl_usd = ((exit_val - entry) / entry) * notional_position
            elif dir_down:
                pnl_usd = ((entry - exit_val) / exit_val) * notional_position
            else:
                return None
            
            return round(pnl_usd, 2)
        except:
            return None
    
    def _calculate_avg_duration_by_status(self, signals_list, status):
        """Calculate average duration for signals with specific status"""
        try:
            matching_signals = []
            for s in signals_list:
                if s.get('status') == status and s.get('fired_time_utc') and s.get('closed_at'):
                    try:
                        fired = datetime.fromisoformat(s.get('fired_time_utc').replace('Z', '+00:00'))
                        closed = datetime.fromisoformat(s.get('closed_at').replace('Z', '+00:00'))
                        delta = closed - fired
                        total_seconds = int(delta.total_seconds())
                        if total_seconds > 0:
                            matching_signals.append(total_seconds)
                    except:
                        pass
            
            if not matching_signals:
                return None
            
            avg_seconds = sum(matching_signals) / len(matching_signals)
            return int(avg_seconds)
        except:
            return None

    def generate_summary(self):
        """Generate detailed summary of post-deployment signals"""
        report = []
        
        report.append("")
        report.append("=" * 200)
        report.append("📊 POST-DEPLOYMENT TRACKER (3-Factor + 4-Factor Normalization)")
        report.append("=" * 200)
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
        
        # BY TIMEFRAME (Enhanced table)
        report.append("🕐 BY TIMEFRAME")
        report.append("─" * 200)
        report.append(f"{'TF':<8} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<10} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<12} | {'Avg TP Dur':<12} | {'Avg SL Dur':<12}")
        report.append("─" * 200)
        
        # Build timeframe statistics
        tf_stats = defaultdict(lambda: {
            'count': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'timeout_win': 0, 'timeout_loss': 0, 'open': 0, 'pnl': 0.0
        })
        
        for s in self.signals:
            tf = s.get('timeframe', 'N/A')
            status = s.get('status', 'OPEN')
            direction = s.get('signal_type', 'LONG')
            
            tf_stats[tf]['count'] += 1
            
            if status == 'TP_HIT':
                tf_stats[tf]['tp'] += 1
            elif status == 'SL_HIT':
                tf_stats[tf]['sl'] += 1
            elif status == 'TIMEOUT':
                tf_stats[tf]['timeout'] += 1
                # Separate into win/loss
                if s.get('actual_exit_price'):
                    pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                    if pnl_calc and pnl_calc > 0:
                        tf_stats[tf]['timeout_win'] += 1
                    elif pnl_calc and pnl_calc < 0:
                        tf_stats[tf]['timeout_loss'] += 1
            elif status == 'OPEN':
                tf_stats[tf]['open'] += 1
            
            # Calculate P&L
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                if pnl_calc:
                    tf_stats[tf]['pnl'] += pnl_calc
        
        # Print by timeframe in order
        for tf in sorted(tf_stats.keys()):
            stats = tf_stats[tf]
            closed = stats['tp'] + stats['sl'] + stats['timeout']
            open_count = stats['open']
            timeout_str = f"{stats['timeout_win']}W/{stats['timeout_loss']}L" if stats['timeout'] > 0 else "-"
            
            if closed > 0:
                wins = stats['tp'] + stats['timeout_win']
                wr = (wins / closed * 100) if closed > 0 else 0
            else:
                wr = 0
            
            pnl_str = f"${stats['pnl']:+.2f}"
            
            # Calculate average durations for this timeframe
            tf_signals = [s for s in self.signals if s.get('timeframe') == tf]
            avg_tp_dur = self._calculate_avg_duration_by_status(tf_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(tf_signals, 'SL_HIT')
            
            avg_tp_dur_str = self._format_duration_hm(avg_tp_dur) if avg_tp_dur else "N/A"
            avg_sl_dur_str = self._format_duration_hm(avg_sl_dur) if avg_sl_dur else "N/A"
            
            report.append(f"{tf:<8} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {timeout_str:<10} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | {pnl_str:>10} | {avg_tp_dur_str:<12} | {avg_sl_dur_str:<12}")
        
        report.append("─" * 200)
        report.append("")
        report.append("=" * 200)
        
        return "\n".join(report)

if __name__ == "__main__":
    tracker = PostDeploymentTracker()
    report = tracker.generate_summary()
    print(report)
    
    # Save to file
    with open("PEC_POST_DEPLOYMENT_TRACKER.txt", "w") as f:
        f.write(report)
    
    print("\n✅ Report saved to PEC_POST_DEPLOYMENT_TRACKER.txt")
