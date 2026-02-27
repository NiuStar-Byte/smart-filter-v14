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
            if not entry_price or entry_price == 0:
                return None
            
            entry = float(entry_price)
            exit_val = float(exit_price)
            notional_position = 1000.0  # $100 entry × 10x leverage = $1,000 notional
            
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
    
    def _format_duration_hms(self, total_seconds: int) -> str:
        """Format duration as HH:MM:SS with zero-padding"""
        try:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except:
            return "-"
    
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
        report.append("─" * 120)
        tf_stats = self._aggregate_by('timeframe')
        for key, stats in sorted(tf_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key:<12} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Direction
        report.append("📈 BY DIRECTION")
        report.append("─" * 120)
        dir_stats = self._aggregate_by('signal_type')
        for key, stats in sorted(dir_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key:<12} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Route
        report.append("🛣️  BY ROUTE")
        report.append("─" * 120)
        route_stats = self._aggregate_by('route')  # lowercase in JSON
        for key, stats in sorted(route_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key:<18} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Regime
        report.append("🌊 BY REGIME")
        report.append("─" * 120)
        regime_stats = self._aggregate_by('regime')
        for key, stats in sorted(regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key:<12} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
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
        
        # === MULTI-DIMENSIONAL AGGREGATES ===
        report.append("=" * 200)
        report.append("📊 MULTI-DIMENSIONAL AGGREGATES")
        report.append("=" * 200)
        report.append("")
        
        # By TimeFrame x Direction
        report.append("🕐📈 BY TIMEFRAME x DIRECTION")
        report.append("─" * 120)
        tf_dir_stats = self._aggregate_by_dimensions(['timeframe', 'signal_type'])
        for key, stats in sorted(tf_dir_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key[0]:<8} | {key[1]:<5} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By TimeFrame x Regime
        report.append("🕐🌊 BY TIMEFRAME x REGIME")
        report.append("─" * 120)
        tf_regime_stats = self._aggregate_by_dimensions(['timeframe', 'regime'])
        for key, stats in sorted(tf_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key[0]:<8} | {key[1]:<6} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Direction x Regime
        report.append("📈🌊 BY DIRECTION x REGIME")
        report.append("─" * 120)
        dir_regime_stats = self._aggregate_by_dimensions(['signal_type', 'regime'])
        for key, stats in sorted(dir_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key[0]:<5} | {key[1]:<6} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Direction x Route
        report.append("📈🛣️  BY DIRECTION x ROUTE")
        report.append("─" * 120)
        dir_route_stats = self._aggregate_by_dimensions(['signal_type', 'route'])
        for key, stats in sorted(dir_route_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key[0]:<5} | {key[1]:<18} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Route x Regime
        report.append("🛣️ 🌊 BY ROUTE x REGIME")
        report.append("─" * 120)
        route_regime_stats = self._aggregate_by_dimensions(['route', 'regime'])
        for key, stats in sorted(route_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key[0]:<18} | {key[1]:<6} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | Closed: {closed:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By TimeFrame x Direction x Route
        report.append("🕐📈🛣️  BY TIMEFRAME x DIRECTION x ROUTE")
        report.append("─" * 140)
        tf_dir_route_stats = self._aggregate_by_dimensions(['timeframe', 'signal_type', 'route'])
        for key, stats in sorted(tf_dir_route_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key[0]:<8} | {key[1]:<5} | {key[2]:<18} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By TimeFrame x Direction x Regime
        report.append("🕐📈🌊 BY TIMEFRAME x DIRECTION x REGIME")
        report.append("─" * 140)
        tf_dir_regime_stats = self._aggregate_by_dimensions(['timeframe', 'signal_type', 'regime'])
        for key, stats in sorted(tf_dir_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key[0]:<8} | {key[1]:<5} | {key[2]:<6} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By Direction x Route x Regime
        report.append("📈🛣️ 🌊 BY DIRECTION x ROUTE x REGIME")
        report.append("─" * 140)
        dir_route_regime_stats = self._aggregate_by_dimensions(['signal_type', 'route', 'regime'])
        for key, stats in sorted(dir_route_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key[0]:<5} | {key[1]:<18} | {key[2]:<6} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By TimeFrame x Direction x Confidence
        report.append("🕐📈💡 BY TIMEFRAME x DIRECTION x CONFIDENCE")
        report.append("─" * 140)
        tf_dir_conf_stats = self._aggregate_by_dimensions(['timeframe', 'signal_type', 'confidence_level'])
        for key, stats in sorted(tf_dir_conf_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key[0]:<8} | {key[1]:<5} | {key[2]:<10} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
        report.append("")
        
        # By TimeFrame x Route x Regime
        report.append("🕐🛣️ 🌊 BY TIMEFRAME x ROUTE x REGIME")
        report.append("─" * 140)
        tf_route_regime_stats = self._aggregate_by_dimensions(['timeframe', 'route', 'regime'])
        for key, stats in sorted(tf_route_regime_stats.items()):
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            report.append(f"{key[0]:<8} | {key[1]:<18} | {key[2]:<6} | Total: {stats['count']:<4} | TP: {stats['tp']:<3} | SL: {stats['sl']:<3} | "
                         f"TIMEOUT: {total_timeout:<3} | WR: {wr:.1f}% | P&L: ${stats['pnl']:+.2f}")
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
        
        # Display Summary
        report.append(f"Total Signals: {total_signals} (Count Win = {total_tp}; Count Loss = {total_sl}; Count TimeOut = {total_timeout}; Count Open = {total_open})")
        report.append(f"Closed Trades: {closed_signals} (TP: {total_tp}, SL: {total_sl}; TimeOut Win = {timeout_wins}; TimeOut Loss = {timeout_losses})")
        report.append(f"Overall Win Rate: {overall_wr:.2f}% >> [ (count TP + Count TimeOut Win) / (Closed Trades) ] = [ ({total_tp}+{timeout_wins}) / {closed_signals} ]")
        report.append(f"Total P&L: ${total_pnl:+.2f}")
        report.append("")
        
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

if __name__ == "__main__":
    reporter = PECEnhancedReporter()
    report = reporter.generate_report()
    print(report)
    
    # Also save to file
    with open("PEC_ENHANCED_REPORT.txt", "w") as f:
        f.write(report)
    
    print("\n✅ Report saved to PEC_ENHANCED_REPORT.txt")
