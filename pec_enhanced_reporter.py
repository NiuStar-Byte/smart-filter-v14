#!/usr/bin/env python3
"""
PEC Enhanced Reporter - Multi-Dimensional Signal Tracking
Breaks down PEC results by: TimeFrame, Direction, Confidence, Regime, Route
Enhanced Features:
- Position size & leverage header
- Extended columns: Route, Exit Price, Exit Time, Duration
- 5D aggregates: TF, Direction, Route, Regime, Confidence
"""

import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os

class PECEnhancedReporter:
    def __init__(self, sent_signals_file="SENT_SIGNALS.jsonl"):
        self.sent_signals_file = sent_signals_file
        self.signals = []
        self.load_signals()
    
    def load_signals(self):
        """Load all signals from SENT_SIGNALS.jsonl"""
        if not os.path.exists(self.sent_signals_file):
            print(f"[WARN] {self.sent_signals_file} not found")
            return
        
        try:
            with open(self.sent_signals_file, 'r') as f:
                for line in f:
                    try:
                        signal = json.loads(line.strip())
                        self.signals.append(signal)
                    except:
                        pass
        except Exception as e:
            print(f"[WARN] Error loading signals: {e}")
    
    def get_gmt7_time(self, utc_time_str):
        """Convert UTC time string to GMT+7 (Bangkok) time"""
        if not utc_time_str:
            return "N/A"
        try:
            # Parse ISO format UTC time
            if 'T' in utc_time_str:
                dt = datetime.fromisoformat(utc_time_str.replace('Z', '+00:00'))
            else:
                return utc_time_str[:19]
            
            # Convert to GMT+7
            gmt7 = dt.astimezone(timezone(timedelta(hours=7)))
            return gmt7.strftime('%H:%M:%S')
        except:
            return str(utc_time_str)[:19]
    
    def _calculate_pnl_usd(self, entry_price: float, exit_price: float, direction: str) -> float:
        """Recalculate P&L USD using notional position of $1,000 ($100 × 10x leverage)"""
        try:
            if not entry_price or entry_price == 0:
                return None
            
            entry = float(entry_price)
            exit_val = float(exit_price)
            notional_position = 1000.0  # $100 × 10x leverage
            
            dir_up = str(direction).strip().upper() == "LONG"
            dir_down = str(direction).strip().upper() == "SHORT"
            
            if dir_up:
                pnl_usd = ((exit_val - entry) / entry) * notional_position
            elif dir_down:
                pnl_usd = ((entry - exit_val) / entry) * notional_position
            else:
                return None
            
            return round(pnl_usd, 4)
        except:
            return None
    
    def generate_report(self):
        """Generate comprehensive report"""
        report = []
        
        # Header (at top before everything)
        report.append("")
        report.append("=" * 200)
        report.append("📊 PEC ENHANCED REPORTER - SIGNAL PERFORMANCE ANALYSIS")
        report.append("=" * 200)
        report.append("")
        
        # Main header for detailed signal list
        report.append("📋 DETAILED SIGNAL LIST: FIX POSITION SIZE $100, LEVERAGE 10x")
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
        report.append(f"Total Signals Loaded: {len(self.signals)}")
        report.append("")
        
        # Column headers
        report.append("─" * 200)
        report.append(f"{'Symbol':<12} {'TF':<8} {'Dir':<5} {'Route':<18} {'Regime':<6} {'Conf':<6} "
                     f"{'Status':<10} {'Entry':<12} {'Exit':<12} {'PnL':<10} "
                     f"{'Fired Time':<12} {'Exit Time/TimeOut':<18} {'Duration':<12}")
        report.append("─" * 200)
        
        for signal in sorted(self.signals, key=lambda s: s.get('fired_time_utc', ''), reverse=True):
            symbol = signal.get('symbol', 'N/A')[:11]
            tf = signal.get('timeframe', 'N/A')[:7]
            direction = signal.get('signal_type', 'N/A')[:4]
            route = signal.get('route', 'N/A')[:17]  # FIXED: lowercase 'route'
            regime = signal.get('regime', 'N/A')[:5]
            confidence = f"{signal.get('confidence', 0):.0f}%"[:5]
            status = signal.get('status', 'OPEN')[:9]
            entry = f"{signal.get('entry_price', 0):.6f}"[:11]
            
            # Exit price
            exit_price = signal.get('actual_exit_price')
            if exit_price:
                exit_str = f"{exit_price:.6f}"[:11]
            else:
                exit_str = "N/A"
            
            # P&L - RECALCULATE using notional position of $1,000
            if exit_price and status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(signal.get('entry_price'), exit_price, direction)
                if pnl_calc is not None:
                    pnl_str = f"${pnl_calc:+.4f}"
                else:
                    pnl_str = "ERROR"
            else:
                pnl_str = "OPEN"
            pnl_str = pnl_str[:9]
            
            # Times and Duration
            fired_time = self.get_gmt7_time(signal.get('fired_time_utc'))
            
            # Exit Time/TimeOut column - show actual close time
            if signal.get('closed_at'):
                # Trade is closed - show actual close time
                exit_time = self.get_gmt7_time(signal.get('closed_at'))
            else:
                # Trade is still open - show dash
                exit_time = "-"
            
            # Duration column
            if status == 'TIMEOUT':
                # For TIMEOUT: show expected timeout window
                tf_val = signal.get('timeframe', '')
                timeout_mins = {'15min': 225, '30min': 300, '1h': 300}.get(tf_val, 300)
                hours = timeout_mins // 60
                mins = timeout_mins % 60
                duration = f"~{hours}h {mins}m"
            elif signal.get('closed_at'):
                # For TP_HIT/SL_HIT: calculate actual duration
                try:
                    fired = datetime.fromisoformat(signal.get('fired_time_utc').replace('Z', '+00:00'))
                    closed = datetime.fromisoformat(signal.get('closed_at').replace('Z', '+00:00'))
                    delta = closed - fired
                    mins = int(delta.total_seconds() // 60)
                    duration = f"{mins}m"
                except:
                    duration = "-"
            else:
                # Trade is still open - show dash
                duration = "-"
            
            report.append(f"{symbol:<12} {tf:<8} {direction:<5} {route:<18} {regime:<6} {confidence:<6} "
                         f"{status:<10} {entry:<12} {exit_str:<12} {pnl_str:<10} "
                         f"{fired_time:<12} {exit_time:<18} {duration:<12}")
        
        # Aggregates Section
        report.append("")
        report.append("=" * 200)
        report.append("📊 AGGREGATES - DIMENSIONAL BREAKDOWN")
        report.append("=" * 200)
        report.append("")
        
        # By TimeFrame
        report.append("🕐 BY TIMEFRAME")
        report.append("─" * 120)
        tf_stats = self._aggregate_by('timeframe')
        for key, stats in sorted(tf_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout']
            wr = (stats['tp'] / closed * 100) if closed > 0 else 0
            report.append(f"{key:<12} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {stats['timeout']:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Direction
        report.append("📈 BY DIRECTION")
        report.append("─" * 120)
        dir_stats = self._aggregate_by('signal_type')
        for key, stats in sorted(dir_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout']
            wr = (stats['tp'] / closed * 100) if closed > 0 else 0
            report.append(f"{key:<12} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {stats['timeout']:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Route
        report.append("🛣️  BY ROUTE")
        report.append("─" * 120)
        route_stats = self._aggregate_by('route')  # lowercase in JSON
        for key, stats in sorted(route_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout']
            wr = (stats['tp'] / closed * 100) if closed > 0 else 0
            report.append(f"{key:<18} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {stats['timeout']:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Regime
        report.append("🌊 BY REGIME")
        report.append("─" * 120)
        regime_stats = self._aggregate_by('regime')
        for key, stats in sorted(regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout']
            wr = (stats['tp'] / closed * 100) if closed > 0 else 0
            report.append(f"{key:<12} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {stats['timeout']:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Confidence
        report.append("💡 BY CONFIDENCE LEVEL")
        report.append("─" * 120)
        
        # Bin confidence levels
        high_conf = [s for s in self.signals if s.get('confidence', 0) >= 76]
        mid_conf = [s for s in self.signals if 51 <= s.get('confidence', 0) < 76]
        low_conf = [s for s in self.signals if s.get('confidence', 0) <= 50]
        
        for conf_level, signals_list, label in [
            ('HIGH', high_conf, 'HIGH (≥76%)'),
            ('MID', mid_conf, 'MID (51-75%)'),
            ('LOW', low_conf, 'LOW (≤50%)')
        ]:
            tp = sum(1 for s in signals_list if s.get('status') == 'TP_HIT')
            sl = sum(1 for s in signals_list if s.get('status') == 'SL_HIT')
            timeout = sum(1 for s in signals_list if s.get('status') == 'TIMEOUT')
            closed = tp + sl + timeout
            total = len(signals_list)
            wr = (tp / closed * 100) if closed > 0 else 0
            
            # RECALCULATE P&L using notional position of $1,000
            pnl = 0.0
            for s in signals_list:
                if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                    pnl_calc = self._calculate_pnl_usd(
                        s.get('entry_price'),
                        s.get('actual_exit_price'),
                        s.get('signal_type')
                    )
                    if pnl_calc is not None:
                        pnl += pnl_calc
            
            report.append(f"{label:<18} | Total: {total:<4} | TP: {tp:<3} | SL: {sl:<3} | "
                         f"TIMEOUT: {timeout:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${pnl:+.2f}")
        report.append("")
        
        # Summary
        report.append("=" * 200)
        report.append("📊 SUMMARY")
        report.append("=" * 200)
        total_signals = len(self.signals)
        closed_signals = sum(1 for s in self.signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT'])
        total_tp = sum(1 for s in self.signals if s.get('status') == 'TP_HIT')
        total_sl = sum(1 for s in self.signals if s.get('status') == 'SL_HIT')
        total_timeout = sum(1 for s in self.signals if s.get('status') == 'TIMEOUT')
        overall_wr = (total_tp / closed_signals * 100) if closed_signals > 0 else 0
        
        # RECALCULATE total P&L using notional position of $1,000
        total_pnl = 0.0
        for s in self.signals:
            if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(
                    s.get('entry_price'),
                    s.get('actual_exit_price'),
                    s.get('signal_type')
                )
                if pnl_calc is not None:
                    total_pnl += pnl_calc
        
        report.append(f"Total Signals: {total_signals}")
        report.append(f"Closed Trades: {closed_signals} (TP: {total_tp}, SL: {total_sl}, TIMEOUT: {total_timeout})")
        report.append(f"Overall Win Rate: {overall_wr:.1f}%")
        report.append(f"Total P&L: ${total_pnl:+.2f}")
        report.append("")
        
        return "\n".join(report)
    
    def _aggregate_by(self, dimension):
        """Aggregate statistics by dimension (all lowercase field names)"""
        stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'pnl': 0.0})
        
        for signal in self.signals:
            # Get the key value for this dimension
            key = signal.get(dimension, 'N/A')
            
            stats[key]['count'] += 1
            
            status = signal.get('status', 'OPEN')
            if status == 'TP_HIT':
                stats[key]['tp'] += 1
            elif status == 'SL_HIT':
                stats[key]['sl'] += 1
            elif status == 'TIMEOUT':
                stats[key]['timeout'] += 1
            
            # RECALCULATE P&L using notional position of $1,000
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(
                    signal.get('entry_price'),
                    signal.get('actual_exit_price'),
                    signal.get('signal_type')
                )
                if pnl_calc is not None:
                    stats[key]['pnl'] += pnl_calc
        
        return stats

if __name__ == "__main__":
    reporter = PECEnhancedReporter()
    report = reporter.generate_report()
    print(report)
    
    # Also save to file
    with open("PEC_ENHANCED_REPORT.txt", "w") as f:
        f.write(report)
    
    print("\n✅ Report saved to PEC_ENHANCED_REPORT.txt")
