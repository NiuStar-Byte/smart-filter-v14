#!/usr/bin/env python3
"""
PEC Enhanced Reporter - Multi-Dimensional Signal Tracking
Breaks down PEC results by: TimeFrame, Direction, Confidence, Regime, Route
Enhanced Features:
- Position size & leverage header
- Extended columns: Route, Exit Price, Exit Time, Duration
- 5D aggregates: TF, Direction, Route, Regime, Confidence
- Dynamic tier configuration from tier_config.py
"""

import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os
from tier_config import TIER_THRESHOLDS

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
                # If datetime is naive (no timezone), mark as UTC explicitly
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            else:
                return utc_time_str[:19]
            
            # Convert to GMT+7
            gmt7 = dt.astimezone(timezone(timedelta(hours=7)))
            return gmt7.strftime('%H:%M:%S')
        except:
            return str(utc_time_str)[:19]
    
    def _calculate_pnl_usd(self, entry_price: float, exit_price: float, direction: str) -> float:
        """Calculate P&L USD using notional position of $1,000 ($100 × 10x leverage)"""
        try:
            if not entry_price or entry_price == 0 or not exit_price or exit_price == 0:
                return None
            
            entry = float(entry_price)
            exit_val = float(exit_price)
            notional_position = 1000.0  # $100 entry × 10x leverage = $1,000 notional
            
            dir_up = str(direction).strip().upper() == "LONG"
            dir_down = str(direction).strip().upper() == "SHORT"
            
            if dir_up:
                # LONG: use entry as denominator
                pnl_usd = ((exit_val - entry) / entry) * notional_position
            elif dir_down:
                # SHORT: use exit as denominator (notional basis at exit)
                pnl_usd = ((entry - exit_val) / exit_val) * notional_position
            else:
                return None
            
            return round(pnl_usd, 4)
        except:
            return None
    
    def _format_duration_hms(self, total_seconds: int) -> str:
        """Format duration as HH:MM:SS with zero-padding"""
        try:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except:
            return "-"
    
    def _format_duration_hm(self, total_seconds: int) -> str:
        """Format duration as Xh Ym (human readable)"""
        try:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except:
            return "-"
    
    def _calculate_avg_duration_by_status(self, signals_list, status: str) -> str:
        """Calculate average duration for signals with specific status (TP_HIT or SL_HIT)"""
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
                return "N/A"
            
            avg_seconds = sum(matching_signals) / len(matching_signals)
            return self._format_duration_hm(int(avg_seconds))
        except:
            return "N/A"
    
    def _calculate_avg_timeout_by_timeframe(self, timeframe: str) -> str:
        """Calculate average duration for TIMEOUT trades by specific timeframe"""
        try:
            matching_signals = []
            for s in self.signals:
                if s.get('status') == 'TIMEOUT' and s.get('timeframe') == timeframe and s.get('fired_time_utc') and s.get('closed_at'):
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
                return "N/A"
            
            avg_seconds = sum(matching_signals) / len(matching_signals)
            return self._format_duration_hm(int(avg_seconds))
        except:
            return "N/A"
    
    def _generate_detailed_signal_list(self):
        """Generate detailed signal list section"""
        detail_lines = []
        
        # Main header for detailed signal list
        detail_lines.append("")
        detail_lines.append("=" * 200)
        detail_lines.append("📋 DETAILED SIGNAL LIST: FIX POSITION SIZE $100, LEVERAGE 10x")
        detail_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
        detail_lines.append(f"Total Signals Loaded: {len(self.signals)}")
        detail_lines.append("")
        
        # Column headers
        detail_lines.append("─" * 220)
        detail_lines.append(f"{'Symbol':<12} {'TF':<8} {'Dir':<6} {'Route':<20} {'Regime':<6} {'Conf':<6} "
                           f"{'Status':<10} {'Entry':<12} {'Exit':<12} {'PnL':<10} "
                           f"{'Fired Time':<12} {'Exit Time/TimeOut':<18} {'Duration':<12}")
        detail_lines.append("─" * 220)
        
        for signal in sorted(self.signals, key=lambda s: s.get('fired_time_utc', ''), reverse=True):
            symbol = signal.get('symbol', 'N/A')[:11]
            tf = signal.get('timeframe', 'N/A')[:7]
            direction = signal.get('signal_type', 'N/A')[:5]
            route = signal.get('route', 'N/A')[:19]  # FIXED: lowercase 'route'
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
                # For TIMEOUT: show expected timeout window in HH:MM:SS format
                tf_val = signal.get('timeframe', '')
                timeout_secs = {'15min': 225*60, '30min': 300*60, '1h': 300*60}.get(tf_val, 300*60)
                duration = self._format_duration_hms(timeout_secs)
            elif signal.get('closed_at'):
                # For TP_HIT/SL_HIT: calculate actual duration in HH:MM:SS format
                try:
                    fired = datetime.fromisoformat(signal.get('fired_time_utc').replace('Z', '+00:00'))
                    closed = datetime.fromisoformat(signal.get('closed_at').replace('Z', '+00:00'))
                    delta = closed - fired
                    total_seconds = int(delta.total_seconds())
                    duration = self._format_duration_hms(total_seconds)
                except:
                    duration = "-"
            else:
                # Trade is still open - show dash
                duration = "-"
            
            detail_lines.append(f"{symbol:<12} {tf:<8} {direction:<6} {route:<20} {regime:<6} {confidence:<6} "
                              f"{status:<10} {entry:<12} {exit_str:<12} {pnl_str:<10} "
                              f"{fired_time:<12} {exit_time:<18} {duration:<12}")
        
        return detail_lines
    
    def generate_report(self):
        """Generate comprehensive report"""
        report = []
        
        # Header (at top before everything)
        report.append("")
        report.append("=" * 200)
        report.append("📊 PEC ENHANCED REPORTER - SIGNAL PERFORMANCE ANALYSIS")
        report.append("=" * 200)
        report.append("")
        
        # Aggregates Section
        report.append("=" * 200)
        report.append("📊 AGGREGATES - DIMENSIONAL BREAKDOWN")
        report.append("=" * 200)
        report.append("")
        
        # By TimeFrame
        report.append("🕐 BY TIMEFRAME")
        report.append("─" * 165)
        report.append(f"{'TimeFrame':<12} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 165)
        tf_stats = self._aggregate_by('timeframe')
        for key, stats in sorted(tf_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            tf_signals = [s for s in self.signals if s.get('timeframe') == key]
            avg_tp_dur = self._calculate_avg_duration_by_status(tf_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(tf_signals, 'SL_HIT')
            
            report.append(f"{key:<12} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By Direction
        report.append("📈 BY DIRECTION")
        report.append("─" * 165)
        report.append(f"{'Direction':<12} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 165)
        dir_stats = self._aggregate_by('signal_type')
        for key, stats in sorted(dir_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            dir_signals = [s for s in self.signals if s.get('signal_type') == key]
            avg_tp_dur = self._calculate_avg_duration_by_status(dir_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(dir_signals, 'SL_HIT')
            
            report.append(f"{key:<12} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By Route
        report.append("🛣️  BY ROUTE")
        report.append("─" * 175)
        report.append(f"{'Route':<20} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 175)
        route_stats = self._aggregate_by('route')  # lowercase in JSON
        for key, stats in sorted(route_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            route_signals = [s for s in self.signals if s.get('route') == key]
            avg_tp_dur = self._calculate_avg_duration_by_status(route_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(route_signals, 'SL_HIT')
            
            report.append(f"{key:<20} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By Regime
        report.append("🌊 BY REGIME")
        report.append("─" * 165)
        report.append(f"{'Regime':<12} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 165)
        regime_stats = self._aggregate_by('regime')
        for key, stats in sorted(regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            regime_signals = [s for s in self.signals if s.get('regime') == key]
            avg_tp_dur = self._calculate_avg_duration_by_status(regime_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(regime_signals, 'SL_HIT')
            
            report.append(f"{key:<12} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By Confidence
        report.append("💡 BY CONFIDENCE LEVEL")
        report.append("─" * 175)
        report.append(f"{'Confidence':<20} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 175)
        
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
            timeout_count = sum(1 for s in signals_list if s.get('status') == 'TIMEOUT')
            closed = tp + sl + timeout_count
            total = len(signals_list)
            open_count = total - closed
            
            # Separate TIMEOUT into wins/losses for accurate WR
            timeout_wins = 0
            for s in signals_list:
                if s.get('status') == 'TIMEOUT' and s.get('actual_exit_price'):
                    pnl_calc = self._calculate_pnl_usd(
                        s.get('entry_price'),
                        s.get('actual_exit_price'),
                        s.get('signal_type')
                    )
                    if pnl_calc is not None and pnl_calc > 0:
                        timeout_wins += 1
            
            wr = ((tp + timeout_wins) / closed * 100) if closed > 0 else 0
            
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
            
            # Add duration metrics
            avg_tp_dur = self._calculate_avg_duration_by_status(signals_list, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(signals_list, 'SL_HIT')
            
            report.append(f"{label:<20} | {total:<6} | {tp:<4} | {sl:<4} | {timeout_count:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${pnl:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # === MULTI-DIMENSIONAL AGGREGATES ===
        report.append("=" * 200)
        report.append("📊 MULTI-DIMENSIONAL AGGREGATES")
        report.append("=" * 200)
        report.append("")
        
        # By TimeFrame x Direction
        report.append("🕐📈 BY TIMEFRAME x DIRECTION")
        report.append("─" * 185)
        report.append(f"{'TF':<8} | {'Dir':<6} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 185)
        tf_dir_stats = self._aggregate_by_dimensions(['timeframe', 'signal_type'])
        for key, stats in sorted(tf_dir_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            combo_signals = [s for s in self.signals if s.get('timeframe') == key[0] and s.get('signal_type') == key[1]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<8} | {key[1]:<6} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By TimeFrame x Regime
        report.append("🕐🌊 BY TIMEFRAME x REGIME")
        report.append("─" * 185)
        report.append(f"{'TF':<8} | {'Regime':<8} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 185)
        tf_regime_stats = self._aggregate_by_dimensions(['timeframe', 'regime'])
        for key, stats in sorted(tf_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            combo_signals = [s for s in self.signals if s.get('timeframe') == key[0] and s.get('regime') == key[1]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<8} | {key[1]:<8} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By Direction x Regime
        report.append("📈🌊 BY DIRECTION x REGIME")
        report.append("─" * 185)
        report.append(f"{'Dir':<6} | {'Regime':<8} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 185)
        dir_regime_stats = self._aggregate_by_dimensions(['signal_type', 'regime'])
        for key, stats in sorted(dir_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            combo_signals = [s for s in self.signals if s.get('signal_type') == key[0] and s.get('regime') == key[1]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<6} | {key[1]:<8} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By Direction x Route
        report.append("📈🛣️  BY DIRECTION x ROUTE")
        report.append("─" * 205)
        report.append(f"{'Dir':<6} | {'Route':<20} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 205)
        dir_route_stats = self._aggregate_by_dimensions(['signal_type', 'route'])
        for key, stats in sorted(dir_route_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            combo_signals = [s for s in self.signals if s.get('signal_type') == key[0] and s.get('route') == key[1]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<6} | {key[1]:<20} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By Route x Regime
        report.append("🛣️ 🌊 BY ROUTE x REGIME")
        report.append("─" * 205)
        report.append(f"{'Route':<20} | {'Regime':<8} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 205)
        route_regime_stats = self._aggregate_by_dimensions(['route', 'regime'])
        for key, stats in sorted(route_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            combo_signals = [s for s in self.signals if s.get('route') == key[0] and s.get('regime') == key[1]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<20} | {key[1]:<8} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By TimeFrame x Direction x Route
        report.append("🕐📈🛣️  BY TIMEFRAME x DIRECTION x ROUTE")
        report.append("─" * 225)
        report.append(f"{'TF':<8} | {'Dir':<6} | {'Route':<20} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 225)
        tf_dir_route_stats = self._aggregate_by_dimensions(['timeframe', 'signal_type', 'route'])
        for key, stats in sorted(tf_dir_route_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            combo_signals = [s for s in self.signals if s.get('timeframe') == key[0] and s.get('signal_type') == key[1] and s.get('route') == key[2]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<8} | {key[1]:<6} | {key[2]:<20} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By TimeFrame x Direction x Regime
        report.append("🕐📈🌊 BY TIMEFRAME x DIRECTION x REGIME")
        report.append("─" * 205)
        report.append(f"{'TF':<8} | {'Dir':<6} | {'Regime':<8} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 205)
        tf_dir_regime_stats = self._aggregate_by_dimensions(['timeframe', 'signal_type', 'regime'])
        for key, stats in sorted(tf_dir_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            combo_signals = [s for s in self.signals if s.get('timeframe') == key[0] and s.get('signal_type') == key[1] and s.get('regime') == key[2]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<8} | {key[1]:<6} | {key[2]:<8} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By Direction x Route x Regime
        report.append("📈🛣️ 🌊 BY DIRECTION x ROUTE x REGIME")
        report.append("─" * 225)
        report.append(f"{'Dir':<6} | {'Route':<20} | {'Regime':<8} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 225)
        dir_route_regime_stats = self._aggregate_by_dimensions(['signal_type', 'route', 'regime'])
        for key, stats in sorted(dir_route_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            combo_signals = [s for s in self.signals if s.get('signal_type') == key[0] and s.get('route') == key[1] and s.get('regime') == key[2]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<6} | {key[1]:<20} | {key[2]:<8} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By TimeFrame x Direction x Confidence
        report.append("🕐📈💡 BY TIMEFRAME x DIRECTION x CONFIDENCE")
        report.append("─" * 225)
        report.append(f"{'TF':<8} | {'Dir':<6} | {'Confidence':<12} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 225)
        tf_dir_conf_stats = self._aggregate_by_dimensions(['timeframe', 'signal_type', 'confidence_level'])
        for key, stats in sorted(tf_dir_conf_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            # key[2] is confidence_level (HIGH/MID/LOW), need to filter by actual confidence ranges
            conf_mapping = {'HIGH': (76, 100), 'MID': (51, 75), 'LOW': (0, 50)}
            conf_range = conf_mapping.get(key[2], (0, 100))
            combo_signals = [s for s in self.signals if s.get('timeframe') == key[0] and s.get('signal_type') == key[1] 
                           and conf_range[0] <= s.get('confidence', 0) <= conf_range[1]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<8} | {key[1]:<6} | {key[2]:<12} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By TimeFrame x Route x Regime
        report.append("🕐🛣️ 🌊 BY TIMEFRAME x ROUTE x REGIME")
        report.append("─" * 225)
        report.append(f"{'TF':<8} | {'Route':<20} | {'Regime':<8} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 225)
        tf_route_regime_stats = self._aggregate_by_dimensions(['timeframe', 'route', 'regime'])
        for key, stats in sorted(tf_route_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Add duration metrics
            combo_signals = [s for s in self.signals if s.get('timeframe') == key[0] and s.get('route') == key[1] and s.get('regime') == key[2]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<8} | {key[1]:<20} | {key[2]:<8} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # Add Detailed Signal List section (before summary)
        report.extend(self._generate_detailed_signal_list())
        
        # Summary
        report.append("")
        report.append("=" * 200)
        report.append("📊 SUMMARY")
        report.append("=" * 200)
        
        # Count signals by status
        total_signals = len(self.signals)
        total_tp = sum(1 for s in self.signals if s.get('status') == 'TP_HIT')
        total_sl = sum(1 for s in self.signals if s.get('status') == 'SL_HIT')
        
        # Separate TIMEOUT into wins and losses based on P&L
        timeout_wins = 0
        timeout_losses = 0
        for s in self.signals:
            if s.get('status') == 'TIMEOUT' and s.get('actual_exit_price'):
                pnl_calc = self._calculate_pnl_usd(
                    s.get('entry_price'),
                    s.get('actual_exit_price'),
                    s.get('signal_type')
                )
                if pnl_calc is not None:
                    if pnl_calc > 0:
                        timeout_wins += 1
                    elif pnl_calc < 0:
                        timeout_losses += 1
        
        total_timeout = timeout_wins + timeout_losses
        total_open = sum(1 for s in self.signals if s.get('status') == 'OPEN')
        
        # Closed trades = TP + SL + TIMEOUT_WIN + TIMEOUT_LOSS
        closed_signals = total_tp + total_sl + timeout_wins + timeout_losses
        
        # Win Rate = (TP + TIMEOUT_WIN) / Closed Trades
        win_count = total_tp + timeout_wins
        overall_wr = (win_count / closed_signals * 100) if closed_signals > 0 else 0
        
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
        
        # Calculate average durations for summary
        avg_tp_duration_summary = self._calculate_avg_duration_by_status(self.signals, 'TP_HIT')
        avg_sl_duration_summary = self._calculate_avg_duration_by_status(self.signals, 'SL_HIT')
        
        # Calculate average TIMEOUT durations by timeframe
        avg_timeout_15min = self._calculate_avg_timeout_by_timeframe('15min')
        avg_timeout_30min = self._calculate_avg_timeout_by_timeframe('30min')
        avg_timeout_1h = self._calculate_avg_timeout_by_timeframe('1h')
        
        # Display Summary
        report.append(f"Total Signals: {total_signals} (Count Win = {total_tp}; Count Loss = {total_sl}; Count TimeOut = {total_timeout}; Count Open = {total_open})")
        report.append(f"Closed Trades: {closed_signals} (TP: {total_tp}, SL: {total_sl}; TimeOut Win = {timeout_wins}; TimeOut Loss = {timeout_losses})")
        report.append(f"Overall Win Rate: {overall_wr:.2f}% >> [ (count TP + Count TimeOut Win) / (Closed Trades) ] = [ ({total_tp}+{timeout_wins}) / {closed_signals} ]")
        report.append(f"Total P&L: ${total_pnl:+.2f}")
        report.append(f"Avg TP Duration: {avg_tp_duration_summary} | Avg SL Duration: {avg_sl_duration_summary}")
        report.append(f"Avg TIMEOUT Duration: 15min={avg_timeout_15min} | 30min={avg_timeout_30min} | 1h={avg_timeout_1h}")
        report.append("")
        
        # === NEW: HIERARCHY RANKING SECTION ===
        report.append("=" * 200)
        report.append("🎯 HIERARCHY RANKING - 2D / 3D / 4D PERFORMANCE TRACKING")
        report.append("=" * 200)
        report.append("")
        
        # Get hierarchy rankings
        hierarchy_section = self._generate_hierarchy_ranking()
        report.append(hierarchy_section)
        
        return "\n".join(report)
    
    def _aggregate_by(self, dimension):
        """Aggregate statistics by dimension (all lowercase field names)"""
        stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout_win': 0, 'timeout_loss': 0, 'pnl': 0.0})
        
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
                # Separate TIMEOUT into wins and losses based on P&L
                if signal.get('actual_exit_price'):
                    pnl_calc = self._calculate_pnl_usd(
                        signal.get('entry_price'),
                        signal.get('actual_exit_price'),
                        signal.get('signal_type')
                    )
                    if pnl_calc is not None:
                        if pnl_calc > 0:
                            stats[key]['timeout_win'] += 1
                        elif pnl_calc < 0:
                            stats[key]['timeout_loss'] += 1
            
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
    
    def _aggregate_by_dimensions(self, dimensions):
        """Aggregate statistics by multiple dimensions (tuple of field names)"""
        stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout_win': 0, 'timeout_loss': 0, 'pnl': 0.0})
        
        for signal in self.signals:
            # Build tuple key from multiple dimensions
            key_parts = []
            for dim in dimensions:
                if dim == 'confidence_level':
                    # Convert confidence to level
                    conf = signal.get('confidence', 0)
                    if conf >= 76:
                        key_parts.append('HIGH')
                    elif 51 <= conf < 76:
                        key_parts.append('MID')
                    else:
                        key_parts.append('LOW')
                else:
                    key_parts.append(str(signal.get(dim, 'N/A')))
            
            key = tuple(key_parts)
            
            stats[key]['count'] += 1
            
            status = signal.get('status', 'OPEN')
            if status == 'TP_HIT':
                stats[key]['tp'] += 1
            elif status == 'SL_HIT':
                stats[key]['sl'] += 1
            elif status == 'TIMEOUT':
                # Separate TIMEOUT into wins and losses based on P&L
                if signal.get('actual_exit_price'):
                    pnl_calc = self._calculate_pnl_usd(
                        signal.get('entry_price'),
                        signal.get('actual_exit_price'),
                        signal.get('signal_type')
                    )
                    if pnl_calc is not None:
                        if pnl_calc > 0:
                            stats[key]['timeout_win'] += 1
                        elif pnl_calc < 0:
                            stats[key]['timeout_loss'] += 1
            
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
    
    def generate_signal_tiers(self):
        """
        Generate SIGNAL_TIERS.json based on PEC aggregates.
        Evaluates all dimension combos against configurable tier criteria.
        
        Criteria loaded from tier_config.py (TIER_THRESHOLDS):
        - min_trades: Minimum closed trades required
        - tier1_wr: Win rate threshold for Tier-1
        - tier1_pnl: Avg P&L per trade for Tier-1
        - tier2_wr_min: Min win rate for Tier-2
        - tier2_pnl: Avg P&L per trade for Tier-2
        - tier3_min_pnl: Min P&L for Tier-3
        """
        tiers = {'tier1': [], 'tier2': [], 'tier3': [], 'tierx': []}
        
        # Get dynamic thresholds from config
        th = TIER_THRESHOLDS
        min_trades = th.get("min_trades", 25)
        tier1_wr = th.get("tier1_wr", 0.60)
        tier1_pnl = th.get("tier1_pnl", 5.0)
        tier2_wr_min = th.get("tier2_wr_min", 0.40)
        tier2_pnl = th.get("tier2_pnl", 2.0)
        tier3_min_pnl = th.get("tier3_min_pnl", 0.00)
        
        print(f"[TIER-CONFIG] min_trades={min_trades}, tier1_wr={tier1_wr}, tier1_pnl={tier1_pnl}, tier2_wr_min={tier2_wr_min}, tier2_pnl={tier2_pnl}", flush=True)
        
        # Evaluate all 2-dimensional combos
        combos_2d = [
            (['timeframe', 'signal_type'], 'TF_DIR'),
            (['timeframe', 'regime'], 'TF_REGIME'),
            (['signal_type', 'regime'], 'DIR_REGIME'),
            (['signal_type', 'route'], 'DIR_ROUTE'),
            (['route', 'regime'], 'ROUTE_REGIME'),
        ]
        
        for dimensions, label in combos_2d:
            stats = self._aggregate_by_dimensions(dimensions)
            for key, stat in stats.items():
                combo_name = f"{label}_{key[0]}_{key[1]}"
                closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
                
                if closed < min_trades:
                    tiers['tierx'].append(combo_name)
                else:
                    win_count = stat['tp'] + stat['timeout_win']
                    wr = (win_count / closed) if closed > 0 else 0
                    avg_pnl_per_trade = stat['pnl'] / closed if closed > 0 else 0
                    
                    if wr >= tier1_wr and avg_pnl_per_trade >= tier1_pnl:
                        tiers['tier1'].append(combo_name)
                    elif tier2_wr_min <= wr < tier1_wr and avg_pnl_per_trade >= tier2_pnl:
                        tiers['tier2'].append(combo_name)
                    elif avg_pnl_per_trade >= tier3_min_pnl:
                        tiers['tier3'].append(combo_name)
                    else:
                        tiers['tierx'].append(combo_name)
        
        # Evaluate all 3-dimensional combos
        combos_3d = [
            (['timeframe', 'signal_type', 'route'], 'TF_DIR_ROUTE'),
            (['timeframe', 'signal_type', 'regime'], 'TF_DIR_REGIME'),
            (['signal_type', 'route', 'regime'], 'DIR_ROUTE_REGIME'),
            (['timeframe', 'route', 'regime'], 'TF_ROUTE_REGIME'),
        ]
        
        for dimensions, label in combos_3d:
            stats = self._aggregate_by_dimensions(dimensions)
            for key, stat in stats.items():
                combo_name = f"{label}_{key[0]}_{key[1]}_{key[2]}"
                closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
                
                if closed < min_trades:
                    tiers['tierx'].append(combo_name)
                else:
                    win_count = stat['tp'] + stat['timeout_win']
                    wr = (win_count / closed) if closed > 0 else 0
                    avg_pnl_per_trade = stat['pnl'] / closed if closed > 0 else 0
                    
                    if wr >= tier1_wr and avg_pnl_per_trade >= tier1_pnl:
                        tiers['tier1'].append(combo_name)
                    elif tier2_wr_min <= wr < tier1_wr and avg_pnl_per_trade >= tier2_pnl:
                        tiers['tier2'].append(combo_name)
                    elif avg_pnl_per_trade >= tier3_min_pnl:
                        tiers['tier3'].append(combo_name)
                    else:
                        tiers['tierx'].append(combo_name)
        
        # Add timestamp and config version
        tiers['generated_at'] = datetime.now(timezone(timedelta(hours=7))).strftime('%Y-%m-%d %H:%M:%S GMT+7')
        tiers['config_version'] = 'A (LOOSE - DEMO)' if min_trades == 5 else 'B (AGREED)' if min_trades == 25 else 'C (STRICT)'
        
        return tiers
    
    def _generate_hierarchy_ranking(self):
        """Generate 2D, 3D, 4D hierarchy ranking for decision-making"""
        output = []
        
        # Get 2D aggregates
        combos_2d = [
            (['timeframe', 'signal_type'], 'TF_DIR'),
            (['timeframe', 'regime'], 'TF_REGIME'),
            (['signal_type', 'regime'], 'DIR_REGIME'),
            (['signal_type', 'route'], 'DIR_ROUTE'),
            (['route', 'regime'], 'ROUTE_REGIME'),
        ]
        
        output.append("📊 2-DIMENSIONAL COMBOS (Top Performers)")
        output.append("─" * 140)
        
        for dimensions, label in combos_2d:
            stats = self._aggregate_by_dimensions(dimensions)
            min_trades = TIER_THRESHOLDS.get("min_trades", 25)
            
            # Filter & rank by WR then P&L
            valid_combos = []
            for key, stat in stats.items():
                closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
                if closed >= min_trades:
                    win_count = stat['tp'] + stat['timeout_win']
                    wr = (win_count / closed) if closed > 0 else 0
                    avg_pnl = stat['pnl'] / closed if closed > 0 else 0
                    valid_combos.append({
                        'name': f"{label}_{key[0]}_{key[1]}",
                        'wr': wr,
                        'pnl': stat['pnl'],
                        'avg_pnl': avg_pnl,
                        'closed': closed,
                        'tp': stat['tp'],
                        'sl': stat['sl']
                    })
            
            # Sort by WR descending, then by P&L
            valid_combos.sort(key=lambda x: (x['wr'], x['pnl']), reverse=True)
            
            if valid_combos:
                output.append(f"\n  {label}:")
                for combo in valid_combos[:3]:  # Top 3 per category
                    output.append(f"    ✓ {combo['name']:40} | WR: {combo['wr']*100:5.1f}% | P&L: ${combo['pnl']:+8.2f} | Avg: ${combo['avg_pnl']:+.2f} | Closed: {combo['closed']}")
        
        output.append("")
        output.append("📊 3-DIMENSIONAL COMBOS (Top Performers)")
        output.append("─" * 140)
        
        # Get 3D aggregates
        combos_3d = [
            (['timeframe', 'signal_type', 'route'], 'TF_DIR_ROUTE'),
            (['timeframe', 'signal_type', 'regime'], 'TF_DIR_REGIME'),
            (['signal_type', 'route', 'regime'], 'DIR_ROUTE_REGIME'),
            (['timeframe', 'route', 'regime'], 'TF_ROUTE_REGIME'),
        ]
        
        all_3d = []
        for dimensions, label in combos_3d:
            stats = self._aggregate_by_dimensions(dimensions)
            min_trades = TIER_THRESHOLDS.get("min_trades", 25)
            
            for key, stat in stats.items():
                closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
                if closed >= min_trades:
                    win_count = stat['tp'] + stat['timeout_win']
                    wr = (win_count / closed) if closed > 0 else 0
                    avg_pnl = stat['pnl'] / closed if closed > 0 else 0
                    all_3d.append({
                        'name': f"{label}_{key[0]}_{key[1]}_{key[2]}",
                        'wr': wr,
                        'pnl': stat['pnl'],
                        'avg_pnl': avg_pnl,
                        'closed': closed,
                        'tp': stat['tp'],
                        'sl': stat['sl']
                    })
        
        all_3d.sort(key=lambda x: (x['wr'], x['pnl']), reverse=True)
        
        if all_3d:
            output.append("\n  Top 5 3D Combos by WR:")
            for combo in all_3d[:5]:
                output.append(f"    ✓ {combo['name']:55} | WR: {combo['wr']*100:5.1f}% | P&L: ${combo['pnl']:+8.2f} | Avg: ${combo['avg_pnl']:+.2f} | Closed: {combo['closed']}")
        else:
            output.append("\n  No 3D combos meet minimum trade threshold yet.")
        
        output.append("")
        output.append("📊 4-DIMENSIONAL COMBOS (Top Performers)")
        output.append("─" * 140)
        
        # Get 4D aggregate (TF x Direction x Route x Regime)
        stats_4d = self._aggregate_by_dimensions(['timeframe', 'signal_type', 'route', 'regime'])
        min_trades = TIER_THRESHOLDS.get("min_trades", 25)
        
        all_4d = []
        for key, stat in stats_4d.items():
            closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
            if closed >= min_trades:
                win_count = stat['tp'] + stat['timeout_win']
                wr = (win_count / closed) if closed > 0 else 0
                avg_pnl = stat['pnl'] / closed if closed > 0 else 0
                all_4d.append({
                    'name': f"TF_DIR_ROUTE_REGIME_{key[0]}_{key[1]}_{key[2]}_{key[3]}",
                    'wr': wr,
                    'pnl': stat['pnl'],
                    'avg_pnl': avg_pnl,
                    'closed': closed,
                    'tp': stat['tp'],
                    'sl': stat['sl']
                })
        
        all_4d.sort(key=lambda x: (x['wr'], x['pnl']), reverse=True)
        
        if all_4d:
            output.append("\n  Top 5 4D Combos by WR:")
            for combo in all_4d[:5]:
                output.append(f"    ✓ {combo['name']:70} | WR: {combo['wr']*100:5.1f}% | P&L: ${combo['pnl']:+8.2f} | Avg: ${combo['avg_pnl']:+.2f} | Closed: {combo['closed']}")
        else:
            output.append("\n  No 4D combos meet minimum trade threshold yet.")
        
        output.append("")
        output.append("=" * 140)
        
        return "\n".join(output)
    
    def save_signal_tiers(self, tiers):
        """Append tiers to SIGNAL_TIERS.json (cumulative history)"""
        filename = "SIGNAL_TIERS.json"
        
        # Extract timestamp from tiers data
        timestamp = tiers.get("generated_at", datetime.now(timezone(timedelta(hours=7))).strftime('%Y-%m-%d %H:%M:%S GMT+7'))
        
        # Build entry (without the full tiers dict, just the summary)
        entry = {
            "timestamp": timestamp,
            "tier1": tiers.get("tier1", []),
            "tier2": tiers.get("tier2", []),
            "tier3": tiers.get("tier3", []),
            "tierx": tiers.get("tierx", []),
            "config_version": tiers.get("config_version", "B (AGREED)")
        }
        
        # Load existing history or create new
        history = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    # Old format (single dict), convert to list
                    history = [history]
            except Exception as e:
                print(f"[TIER-SAVE] Could not read existing {filename}: {e}. Starting fresh.", flush=True)
                history = []
        
        # Append new entry
        history.append(entry)
        
        # Write back
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"[TIER-SAVE] Appended tier entry to {filename} (total entries: {len(history)})", flush=True)
        return filename

if __name__ == "__main__":
    reporter = PECEnhancedReporter()
    report = reporter.generate_report()
    print(report)
    
    # Also save to file
    with open("PEC_ENHANCED_REPORT.txt", "w") as f:
        f.write(report)
    
    print("\n✅ Report saved to PEC_ENHANCED_REPORT.txt")
    
    # Generate and save signal tiers (append-only)
    tiers = reporter.generate_signal_tiers()
    tier_file = reporter.save_signal_tiers(tiers)
    print(f"✅ Signal tiers appended to {tier_file}")
    
    # Cleanup old SIGNAL_TIERS_*.json timestamped files (no longer needed)
    import glob
    old_tier_files = glob.glob("SIGNAL_TIERS_*.json")
    if old_tier_files:
        for f in old_tier_files:
            try:
                os.remove(f)
                print(f"🗑️  Removed old tier file: {f}")
            except Exception as e:
                print(f"[WARN] Could not remove {f}: {e}")
    
    if old_tier_files:
        print(f"✅ Cleaned up {len(old_tier_files)} old SIGNAL_TIERS_*.json files")
