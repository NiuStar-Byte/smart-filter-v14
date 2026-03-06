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

# === SYMBOL GROUPING (5D DIMENSION) ===
SYMBOL_GROUPS = {
    "MAIN_BLOCKCHAIN": [
        "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT",
        "AVAX-USDT", "BNB-USDT", "XLM-USDT", "LINK-USDT", "POL-USDT"
    ],
    "TOP_ALTS": [
        "ZKJ-USDT", "ROAM-USDT", "XAUT-USDT", "SAHARA-USDT"
    ],
    "MID_ALTS": [
        "XPL-USDT", "DOT-USDT", "FUEL-USDT", "VIRTUAL-USDT", "BERA-USDT",
        "CROSS-USDT", "FUN-USDT", "ENA-USDT", "SOL-USDT", "AVAX-USDT"
    ]
}

class PECEnhancedReporter:
    def __init__(self, sent_signals_file=None):
        # Prefer CUMULATIVE files (immutable hourly snapshots)
        # Fall back to ARCHIVE (Foundation) or LIVE (current accumulation)
        if sent_signals_file is None:
            workspace = "/Users/geniustarigan/.openclaw/workspace"
            
            # Try CUMULATIVE first (hourly immutable snapshot)
            import glob
            cumulative_files = sorted(glob.glob(os.path.join(workspace, "SENT_SIGNALS_CUMULATIVE_*.jsonl")))
            if cumulative_files:
                sent_signals_file = cumulative_files[-1]
                print(f"[INFO] Using CUMULATIVE file: {os.path.basename(sent_signals_file)}", flush=True)
            else:
                # Fallback to ARCHIVE (Foundation baseline)
                sent_signals_file = os.path.join(workspace, "SENT_SIGNALS_ARCHIVE_2026-03-05.jsonl")
                print(f"[INFO] Using ARCHIVE file (Foundation): SENT_SIGNALS_ARCHIVE_2026-03-05.jsonl", flush=True)
        
        self.sent_signals_file = sent_signals_file
        self.signals = []
        self.load_signals()
    
    def load_signals(self):
        """Load ALL signals from BOTH archive + live files (complete history)"""
        workspace = "/Users/geniustarigan/.openclaw/workspace"
        
        # Load from ARCHIVE FIRST (FOUNDATION signals)
        archive_file = os.path.join(workspace, "SENT_SIGNALS_ARCHIVE_2026-03-05.jsonl")
        if os.path.exists(archive_file):
            try:
                with open(archive_file, 'r') as f:
                    count = 0
                    for line in f:
                        try:
                            signal = json.loads(line.strip())
                            self.signals.append(signal)
                            count += 1
                        except:
                            pass
                print(f"[INFO] Loaded {count} signals from archive", flush=True)
            except Exception as e:
                print(f"[WARN] Error loading archive: {e}")
        
        # Load from LIVE FILE (accumulating NEW signals)
        if not os.path.exists(self.sent_signals_file):
            print(f"[WARN] {self.sent_signals_file} not found")
            return
        
        try:
            with open(self.sent_signals_file, 'r') as f:
                count = 0
                for line in f:
                    try:
                        signal = json.loads(line.strip())
                        self.signals.append(signal)
                        count += 1
                    except:
                        pass
                print(f"[INFO] Loaded {count} signals from live file", flush=True)
        except Exception as e:
            print(f"[WARN] Error loading signals: {e}")
    
    def get_symbol_group(self, symbol: str) -> str:
        """Map a symbol to its group"""
        for group_name, symbols in SYMBOL_GROUPS.items():
            if symbol in symbols:
                return group_name
        return "LOW_ALTS"  # Default for unmapped symbols
    
    def get_gmt7_time(self, utc_time_str):
        """Convert UTC time string to GMT+7 (Bangkok) time
        Always include date for consistent, transparent timestamp display across all signals.
        """
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
            
            # Always show date for consistency (e.g., "Feb-28 09:50:05", "Mar-01 01:54:44")
            return gmt7.strftime('%b-%d %H:%M:%S')
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
        detail_lines.append("=" * 290)
        detail_lines.append("📋 DETAILED SIGNAL LIST: FIXED POSITION SIZE $100, LEVERAGE 10x")
        detail_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
        detail_lines.append(f"Total Signals Loaded: {len(self.signals)}")
        detail_lines.append("")
        detail_lines.append("⚠️  NOTE: Signals marked 'STALE_TIMEOUT' in Data Quality column are EXCLUDED from Aggregates/Summary/Hierarchy")
        detail_lines.append("    (They appear here for audit trail only, but don't affect backtest P&L calculations)")
        detail_lines.append("")
        
        # Column headers
        detail_lines.append("─" * 290)
        detail_lines.append(f"{'Symbol':<12} {'TF':<8} {'Dir':<6} {'Route':<20} {'Regime':<6} {'Conf':<6} {'Sym Grp':<12} "
                           f"{'Status':<10} {'Entry':<12} {'Exit':<12} {'PnL':<10} "
                           f"{'Fired Time':<12} {'Exit Time/TimeOut':<18} {'Duration':<12} {'Data Quality':<28}")
        detail_lines.append("─" * 290)
        
        for signal in sorted(self.signals, key=lambda s: s.get('fired_time_utc', ''), reverse=True):
            symbol = signal.get('symbol', 'N/A')[:11]
            tf = signal.get('timeframe', 'N/A')[:7]
            direction = signal.get('signal_type', 'N/A')[:5]
            route = signal.get('route', 'N/A')[:19]  # FIXED: lowercase 'route'
            regime = signal.get('regime', 'N/A')[:5]
            confidence = f"{signal.get('confidence', 0):.0f}%"[:5]
            symbol_group = self.get_symbol_group(symbol)[:11]
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
            
            # Add data quality flag column
            data_quality_flag = signal.get('data_quality_flag', '')
            if data_quality_flag:
                if 'STALE_TIMEOUT' in data_quality_flag:
                    # Extract hours overdue from flag
                    hours_match = data_quality_flag.split('_')[-2]  # Gets the "Xh" part
                    quality_str = f"⚠️  {data_quality_flag[:27]}"  # Truncate if too long
                else:
                    quality_str = "✓ CLEAN"
            else:
                quality_str = "✓ CLEAN"
            
            detail_lines.append(f"{symbol:<12} {tf:<8} {direction:<6} {route:<20} {regime:<6} {confidence:<6} {symbol_group:<12} "
                              f"{status:<10} {entry:<12} {exit_str:<12} {pnl_str:<10} "
                              f"{fired_time:<12} {exit_time:<18} {duration:<12} {quality_str:<28}")
        
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
        
        # By Symbol Group
        report.append("💰 BY SYMBOL GROUP")
        report.append("─" * 175)
        report.append(f"{'Symbol Group':<20} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 175)
        
        # Aggregate by symbol group
        symbol_group_stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout_win': 0, 'timeout_loss': 0, 'pnl': 0.0})
        for signal in self.signals:
            symbol = signal.get('symbol', 'UNKNOWN')
            group = self.get_symbol_group(symbol)
            
            symbol_group_stats[group]['count'] += 1
            
            status = signal.get('status', 'OPEN')
            if status == 'TP_HIT':
                symbol_group_stats[group]['tp'] += 1
            elif status == 'SL_HIT':
                symbol_group_stats[group]['sl'] += 1
            elif status == 'TIMEOUT':
                if signal.get('actual_exit_price'):
                    pnl_calc = self._calculate_pnl_usd(
                        signal.get('entry_price'),
                        signal.get('actual_exit_price'),
                        signal.get('signal_type')
                    )
                    if pnl_calc is not None:
                        if pnl_calc > 0:
                            symbol_group_stats[group]['timeout_win'] += 1
                        elif pnl_calc < 0:
                            symbol_group_stats[group]['timeout_loss'] += 1
            
            # P&L calculation
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(
                    signal.get('entry_price'),
                    signal.get('actual_exit_price'),
                    signal.get('signal_type')
                )
                if pnl_calc is not None:
                    symbol_group_stats[group]['pnl'] += pnl_calc
        
        for group in sorted(symbol_group_stats.keys()):
            stats = symbol_group_stats[group]
            closed = stats['tp'] + stats['sl'] + stats['timeout_win'] + stats['timeout_loss']
            open_count = stats['count'] - closed
            win_count = stats['tp'] + stats['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stats['timeout_win'] + stats['timeout_loss']
            
            # Duration metrics
            group_signals = [s for s in self.signals if self.get_symbol_group(s.get('symbol', 'UNKNOWN')) == group]
            avg_tp_dur = self._calculate_avg_duration_by_status(group_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(group_signals, 'SL_HIT')
            
            report.append(f"{group:<20} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stats['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
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
        
        # By TimeFrame x Symbol Group
        report.append("🕐💰 BY TIMEFRAME x SYMBOL_GROUP")
        report.append("─" * 205)
        report.append(f"{'TF':<8} | {'Symbol Group':<18} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 205)
        tf_symbol_stats = self._aggregate_by_dimensions_with_symbol(['timeframe'])
        for key, stat in sorted(tf_symbol_stats.items()):
            closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
            open_count = stat['count'] - closed
            win_count = stat['tp'] + stat['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stat['timeout_win'] + stat['timeout_loss']
            
            # Duration metrics
            combo_signals = [s for s in self.signals if s.get('timeframe') == key[0] and self.get_symbol_group(s.get('symbol', 'UNKNOWN')) == key[1]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<8} | {key[1]:<18} | {stat['count']:<6} | {stat['tp']:<4} | {stat['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stat['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By Direction x Symbol Group
        report.append("📈💰 BY DIRECTION x SYMBOL_GROUP")
        report.append("─" * 205)
        report.append(f"{'Dir':<6} | {'Symbol Group':<18} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 205)
        dir_symbol_stats = self._aggregate_by_dimensions_with_symbol(['signal_type'])
        for key, stat in sorted(dir_symbol_stats.items()):
            closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
            open_count = stat['count'] - closed
            win_count = stat['tp'] + stat['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stat['timeout_win'] + stat['timeout_loss']
            
            # Duration metrics
            combo_signals = [s for s in self.signals if s.get('signal_type') == key[0] and self.get_symbol_group(s.get('symbol', 'UNKNOWN')) == key[1]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<6} | {key[1]:<18} | {stat['count']:<6} | {stat['tp']:<4} | {stat['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stat['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
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
        
        # By Route x Symbol Group
        report.append("🛣️ 💰 BY ROUTE x SYMBOL_GROUP")
        report.append("─" * 225)
        report.append(f"{'Route':<20} | {'Symbol Group':<18} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 225)
        route_symbol_stats = self._aggregate_by_dimensions_with_symbol(['route'])
        for key, stat in sorted(route_symbol_stats.items()):
            closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
            open_count = stat['count'] - closed
            win_count = stat['tp'] + stat['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stat['timeout_win'] + stat['timeout_loss']
            
            # Duration metrics
            combo_signals = [s for s in self.signals if s.get('route') == key[0] and self.get_symbol_group(s.get('symbol', 'UNKNOWN')) == key[1]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<20} | {key[1]:<18} | {stat['count']:<6} | {stat['tp']:<4} | {stat['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stat['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
        report.append("")
        
        # By Regime x Symbol Group
        report.append("🌊💰 BY REGIME x SYMBOL_GROUP")
        report.append("─" * 225)
        report.append(f"{'Regime':<8} | {'Symbol Group':<18} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<8} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<10} | {'Avg TP Duration':<17} | {'Avg SL Duration':<17}")
        report.append("─" * 225)
        regime_symbol_stats = self._aggregate_by_dimensions_with_symbol(['regime'])
        for key, stat in sorted(regime_symbol_stats.items()):
            closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
            open_count = stat['count'] - closed
            win_count = stat['tp'] + stat['timeout_win']
            wr = (win_count / closed * 100) if closed > 0 else 0
            total_timeout = stat['timeout_win'] + stat['timeout_loss']
            
            # Duration metrics
            combo_signals = [s for s in self.signals if s.get('regime') == key[0] and self.get_symbol_group(s.get('symbol', 'UNKNOWN')) == key[1]]
            avg_tp_dur = self._calculate_avg_duration_by_status(combo_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(combo_signals, 'SL_HIT')
            
            report.append(f"{key[0]:<8} | {key[1]:<18} | {stat['count']:<6} | {stat['tp']:<4} | {stat['sl']:<4} | {total_timeout:<8} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | ${stat['pnl']:>+8.2f} | {avg_tp_dur:<17} | {avg_sl_dur:<17}")
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
        report.append("=" * 132)
        report.append("📊 SUMMARY")
        report.append("=" * 132)
        
        # Count signals by status (EXCLUDING stale timeouts from backtest)
        total_signals = len(self.signals)
        total_tp = sum(1 for s in self.signals if s.get('status') == 'TP_HIT' and not (s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag')))
        total_sl = sum(1 for s in self.signals if s.get('status') == 'SL_HIT' and not (s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag')))
        
        # Separate TIMEOUT into wins and losses based on P&L (EXCLUDING stale)
        timeout_wins = 0
        timeout_losses = 0
        for s in self.signals:
            # Skip stale timeouts from metrics
            if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'):
                continue
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
        
        # RECALCULATE total P&L using notional position of $1,000 (EXCLUDING stale timeouts)
        total_pnl = 0.0
        for s in self.signals:
            # Skip stale timeouts from backtest P&L
            if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'):
                continue
            if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(
                    s.get('entry_price'),
                    s.get('actual_exit_price'),
                    s.get('signal_type')
                )
                if pnl_calc is not None:
                    total_pnl += pnl_calc
        
        # Calculate average P&L per signal
        avg_pnl_per_signal = total_pnl / total_signals if total_signals > 0 else 0
        avg_pnl_per_trade = total_pnl / closed_signals if closed_signals > 0 else 0
        
        # Calculate average durations for summary (TP/SL only - stale timeouts already excluded by status)
        # Note: Stale timeouts have status='TIMEOUT', not 'TP_HIT'/'SL_HIT', so they never appear in these calculations
        avg_tp_duration_summary = self._calculate_avg_duration_by_status(self.signals, 'TP_HIT')
        avg_sl_duration_summary = self._calculate_avg_duration_by_status(self.signals, 'SL_HIT')
        
        # Calculate ACTUAL MAX TIMEOUT WINDOW by timeframe (from clean timeout signals only, excluding stale)
        # Count ALL clean timeouts (whether same-date or cross-date) and find max per TF
        # Then limit to expected maximum (3h45m for 15min, 5h for 30min, 5h for 1h)
        max_timeout_by_tf = {'15min': 0, '30min': 0, '1h': 0}
        expected_max_by_tf = {'15min': 15*15*60, '30min': 10*30*60, '1h': 5*60*60}  # Expected maximums
        
        for s in self.signals:
            # Skip stale timeouts - only calculate from clean timeouts
            if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'):
                continue
            if s.get('status') == 'TIMEOUT' and s.get('fired_time_utc') and s.get('closed_at'):
                try:
                    fired = datetime.fromisoformat(s.get('fired_time_utc').replace('Z', '+00:00'))
                    closed = datetime.fromisoformat(s.get('closed_at').replace('Z', '+00:00'))
                    delta = closed - fired
                    duration_seconds = int(delta.total_seconds())
                    
                    tf = s.get('timeframe', '')
                    if tf in max_timeout_by_tf:
                        # Only count if within expected maximum (real timeout window)
                        if duration_seconds <= expected_max_by_tf[tf]:
                            max_timeout_by_tf[tf] = max(max_timeout_by_tf[tf], duration_seconds)
                except:
                    pass
        
        # Format timeout windows - show DESIGNED maximum (what system is set for)
        # These are the theoretical limits per timeframe design
        timeout_window_15min = "3h 45m"   # 15 bars × 15min = 225min = 3h45m (designed)
        timeout_window_30min = "5h 0m"    # 10 bars × 30min = 300min = 5h (designed)
        timeout_window_1h = "5h 0m"       # 5 bars × 60min = 300min = 5h (designed)
        
        # Also calculate actual max from clean signals within expected limits
        actual_max_15min = self._format_duration_hm(max_timeout_by_tf['15min']) if max_timeout_by_tf['15min'] > 0 else "None"
        actual_max_30min = self._format_duration_hm(max_timeout_by_tf['30min']) if max_timeout_by_tf['30min'] > 0 else "None"
        actual_max_1h = self._format_duration_hm(max_timeout_by_tf['1h']) if max_timeout_by_tf['1h'] > 0 else "None"
        
        # Calculate fired time range and count per date
        signals_by_date = defaultdict(list)
        all_fired_times = []
        
        for s in self.signals:
            fired_utc = s.get('fired_time_utc')
            if fired_utc:
                try:
                    dt = datetime.fromisoformat(fired_utc.replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    
                    # Convert to GMT+7
                    gmt7 = dt.astimezone(timezone(timedelta(hours=7)))
                    date_str = gmt7.strftime('%Y-%m-%d')
                    signals_by_date[date_str].append(gmt7)
                    all_fired_times.append(gmt7)
                except:
                    pass
        
        # Get overall earliest and latest fired times (for reference)
        if all_fired_times:
            beginning_fired = min(all_fired_times).strftime('%Y-%m-%d %H:%M:%S')
            last_fired = max(all_fired_times).strftime('%Y-%m-%d %H:%M:%S')
        else:
            beginning_fired = "N/A"
            last_fired = "N/A"
        
        # Build per-date summary (each date on its own line with its own beginning/last times)
        date_summary_lines = []
        today_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).strftime('%Y-%m-%d')
        
        for date_str in sorted(signals_by_date.keys()):
            times = signals_by_date[date_str]
            count = len(times)
            date_earliest = min(times)
            date_latest = max(times)
            beginning_time_str = date_earliest.strftime('%H:%M:%S')
            last_time_str = date_latest.strftime('%H:%M:%S')
            
            # Add immutable label for past dates, accumulating for today
            if date_str == today_date:
                status_label = "🔄 still accumulating"
            else:
                status_label = "✓ IMMUTABLE"
            
            date_line = f"  {date_str}: {count} fired | Beginning Fired Time: {beginning_time_str} | Last Fired Time: {last_time_str} | {status_label}"
            date_summary_lines.append(date_line)
        
        # Display Summary (with stale timeout exclusion note) - Match user's exact template
        stale_timeout_count = sum(1 for s in self.signals if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'))
        
        report.append("")
        report.append("🔒 FOUNDATION BASELINE (IMMUTABLE - Locked at commit c535c34)")
        report.append("Total Signals: 853 | Closed: 830 | WR: 25.7%")
        report.append("LONG WR: 29.6% | SHORT WR: 46.2% | P&L: $-5498.59")
        report.append("")
        report.append(f"Total Signals (Foundation + New): {total_signals} (Count Win = {total_tp}; Count Loss = {total_sl}; Count TimeOut = {total_timeout}; Count Open = {total_open}; Stale Timeouts Excluded = {stale_timeout_count})")
        report.append(f"Closed Trades (Clean Data): {closed_signals} (TP: {total_tp}, SL: {total_sl}; TimeOut Win = {timeout_wins}; TimeOut Loss = {timeout_losses})")
        report.append(f"Overall Win Rate: {overall_wr:.2f}% >> [ (count TP + Count TimeOut Win) / (Closed Trades) ] = [ ({total_tp}+{timeout_wins}) / {closed_signals} ]")
        report.append(f"Total P&L (Clean Data): ${total_pnl:+.2f} (Avg P&L per Signal = ${avg_pnl_per_signal:+.2f}; Avg P&L per Closed Trade = ${avg_pnl_per_trade:+.2f})")
        report.append(f"Avg TP Duration (Clean): {avg_tp_duration_summary} | Avg SL Duration (Clean): {avg_sl_duration_summary}")
        report.append(f"Max TIMEOUT Window (Designed): 15min={timeout_window_15min} | 30min={timeout_window_30min} | 1h={timeout_window_1h}")
        report.append(f"Max TIMEOUT Actual (Within Limit): 15min={actual_max_15min} | 30min={actual_max_30min} | 1h={actual_max_1h}")
        
        # Add fired time range and count per date (each date on its own line)
        if date_summary_lines:
            report.append("Total Fired per Date:")
            report.extend(date_summary_lines)
        
        # === NEW: HOURLY BREAKDOWN FOR TODAY (DYNAMIC) ===
        today_str = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).strftime('%Y-%m-%d')
        signals_by_hour = defaultdict(int)
        
        if today_str in signals_by_date:
            today_times = signals_by_date[today_str]
            for gmt7_dt in today_times:
                hour_key = gmt7_dt.strftime('%H:%M-%H:%M').split('-')[0]  # Get hour as "HH"
                hour_bucket = f"{gmt7_dt.hour:02d}:00-{(gmt7_dt.hour+1):02d}:00"
                signals_by_hour[hour_bucket] += 1
        
        if signals_by_hour and today_str:  # Show hourly breakdown only for current accumulating date
            report.append(f"Total Fired Today per Hour ({today_str}):")
            current_hour = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).hour
            current_hour_bucket = f"{current_hour:02d}:00-{(current_hour+1):02d}:00"
            
            # Build list of ALL hours from 00:00 to current_hour (inclusive)
            all_hour_buckets = []
            for hour in range(current_hour + 1):  # 0 to current_hour
                hour_bucket = f"{hour:02d}:00-{(hour+1):02d}:00"
                all_hour_buckets.append(hour_bucket)
            
            for hour_bucket in all_hour_buckets:
                count = signals_by_hour.get(hour_bucket, 0)
                
                # Add status label: current hour is accumulating, past hours are immutable
                if hour_bucket == current_hour_bucket:
                    status_label = "🔄 still accumulating"
                else:
                    status_label = "✓ IMMUTABLE"
                
                report.append(f"  {hour_bucket}: {count} fired | {status_label}")
        
        if stale_timeout_count > 0:
            report.append("")
            report.append(f"⚠️  DATA QUALITY NOTE: {stale_timeout_count} signals marked as 'STALE_TIMEOUT' (closed >150% past deadline)")
            report.append(f"These are EXCLUDED from all above metrics to preserve backtest accuracy. See DETAILED SIGNAL LIST for individual stale timeout records.")
        
        report.append("")
        
        # === NEW: HIERARCHY RANKING SECTION ===
        report.append("=" * 200)
        report.append("🎯 HIERARCHY RANKING - 5D / 4D / 3D / 2D PERFORMANCE TRACKING")
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
            # SKIP stale timeouts (data quality issue) - exclude from backtest P&L
            if signal.get('data_quality_flag') and 'STALE_TIMEOUT' in signal.get('data_quality_flag'):
                continue
            
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
            # SKIP stale timeouts (data quality issue) - exclude from backtest P&L
            if signal.get('data_quality_flag') and 'STALE_TIMEOUT' in signal.get('data_quality_flag'):
                continue
            
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
    
    def _aggregate_by_dimensions_with_symbol(self, dimensions):
        """Aggregate by 4D dimensions PLUS symbol_group (5D)"""
        stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout_win': 0, 'timeout_loss': 0, 'pnl': 0.0})
        
        for signal in self.signals:
            # SKIP stale timeouts (data quality issue) - exclude from backtest P&L
            if signal.get('data_quality_flag') and 'STALE_TIMEOUT' in signal.get('data_quality_flag'):
                continue
            
            # Build tuple key from dimensions + symbol_group
            key_parts = []
            for dim in dimensions:
                if dim == 'confidence_level':
                    conf = signal.get('confidence', 0)
                    if conf >= 76:
                        key_parts.append('HIGH')
                    elif 51 <= conf < 76:
                        key_parts.append('MID')
                    else:
                        key_parts.append('LOW')
                else:
                    key_parts.append(str(signal.get(dim, 'N/A')))
            
            # Add symbol_group as 5th dimension
            symbol = signal.get('symbol', 'UNKNOWN')
            symbol_group = self.get_symbol_group(symbol)
            key_parts.append(symbol_group)
            
            key = tuple(key_parts)
            
            stats[key]['count'] += 1
            
            status = signal.get('status', 'OPEN')
            if status == 'TP_HIT':
                stats[key]['tp'] += 1
            elif status == 'SL_HIT':
                stats[key]['sl'] += 1
            elif status == 'TIMEOUT':
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
    
    def _check_tier_qualification(self, wr, avg_pnl, closed_trades, tier_level):
        """
        Check if a combo qualifies for a specific tier level.
        Returns: tier_name or None
        """
        th = TIER_THRESHOLDS
        
        if tier_level == 1:
            min_trades = th.get("tier1_min_trades", 60)
            min_wr = th.get("tier1_wr", 0.60)
            min_pnl = th.get("tier1_pnl", 5.50)
            if closed_trades >= min_trades and wr >= min_wr and avg_pnl >= min_pnl:
                return "Tier-1"
        
        elif tier_level == 2:
            min_trades = th.get("tier2_min_trades", 50)
            min_wr = th.get("tier2_wr", 0.50)
            min_pnl = th.get("tier2_pnl", 3.50)
            if closed_trades >= min_trades and wr >= min_wr and avg_pnl >= min_pnl:
                return "Tier-2"
        
        elif tier_level == 3:
            min_trades = th.get("tier3_min_trades", 40)
            min_wr = th.get("tier3_wr", 0.40)
            min_pnl = th.get("tier3_pnl", 2.00)
            if closed_trades >= min_trades and wr >= min_wr and avg_pnl >= min_pnl:
                return "Tier-3"
        
        return None
    
    def generate_signal_tiers(self):
        """
        Generate SIGNAL_TIERS.json using CONSENSUS CASCADE strategy:
        1. Evaluate 5D first (finest granularity)
        2. If 5D qualifies for ANY tier (1/2/3), assign and STOP
        3. If 5D fails, try 4D
        4. If 4D fails, try 3D
        5. If 3D fails, try 2D
        6. If all fail, assign Tier-X
        
        Within each tier check: verify Tier-1 first (hardest), then Tier-2, then Tier-3
        """
        tiers = {'tier1': [], 'tier2': [], 'tier3': [], 'tierx': []}
        tier_details = {}  # Store which level each combo qualified at
        
        th = TIER_THRESHOLDS
        print(f"[TIER-CONFIG] Consensus Cascade + Option B", flush=True)
        print(f"  Tier-1: {th.get('tier1_wr')*100:.0f}% WR, ${th.get('tier1_pnl'):.2f}+ avg, {th.get('tier1_min_trades')}+ trades", flush=True)
        print(f"  Tier-2: {th.get('tier2_wr')*100:.0f}% WR, ${th.get('tier2_pnl'):.2f}+ avg, {th.get('tier2_min_trades')}+ trades", flush=True)
        print(f"  Tier-3: {th.get('tier3_wr')*100:.0f}% WR, ${th.get('tier3_pnl'):.2f}+ avg, {th.get('tier3_min_trades')}+ trades", flush=True)
        
        # ===== CONSENSUS CASCADE LOGIC =====
        # Build all combos with their stats at each dimension
        all_combos_by_dimension = {
            '5D': {},
            '4D': {},
            '3D': {},
            '2D': {}
        }
        
        # Get 5D stats
        stats_5d = self._aggregate_by_dimensions_with_symbol(['timeframe', 'signal_type', 'route', 'regime'])
        for key, stat in stats_5d.items():
            closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
            win_count = stat['tp'] + stat['timeout_win']
            wr = (win_count / closed) if closed > 0 else 0
            avg_pnl = stat['pnl'] / closed if closed > 0 else 0
            combo_name = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
            all_combos_by_dimension['5D'][combo_name] = {
                'wr': wr, 'avg_pnl': avg_pnl, 'pnl': stat['pnl'], 'closed': closed
            }
        
        # Get 4D stats
        stats_4d = self._aggregate_by_dimensions(['timeframe', 'signal_type', 'route', 'regime'])
        for key, stat in stats_4d.items():
            closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
            win_count = stat['tp'] + stat['timeout_win']
            wr = (win_count / closed) if closed > 0 else 0
            avg_pnl = stat['pnl'] / closed if closed > 0 else 0
            combo_name = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}"
            all_combos_by_dimension['4D'][combo_name] = {
                'wr': wr, 'avg_pnl': avg_pnl, 'pnl': stat['pnl'], 'closed': closed
            }
        
        # Get 3D stats (all types)
        combos_3d = [
            (['timeframe', 'signal_type', 'route'], 'TF_DIR_ROUTE'),
            (['timeframe', 'signal_type', 'regime'], 'TF_DIR_REGIME'),
            (['signal_type', 'route', 'regime'], 'DIR_ROUTE_REGIME'),
            (['timeframe', 'route', 'regime'], 'TF_ROUTE_REGIME'),
        ]
        for dimensions, label in combos_3d:
            stats = self._aggregate_by_dimensions(dimensions)
            for key, stat in stats.items():
                closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
                win_count = stat['tp'] + stat['timeout_win']
                wr = (win_count / closed) if closed > 0 else 0
                avg_pnl = stat['pnl'] / closed if closed > 0 else 0
                combo_name = f"{label}_{key[0]}_{key[1]}_{key[2]}"
                all_combos_by_dimension['3D'][combo_name] = {
                    'wr': wr, 'avg_pnl': avg_pnl, 'pnl': stat['pnl'], 'closed': closed
                }
        
        # Get 2D stats
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
                closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
                win_count = stat['tp'] + stat['timeout_win']
                wr = (win_count / closed) if closed > 0 else 0
                avg_pnl = stat['pnl'] / closed if closed > 0 else 0
                combo_name = f"{label}_{key[0]}_{key[1]}"
                all_combos_by_dimension['2D'][combo_name] = {
                    'wr': wr, 'avg_pnl': avg_pnl, 'pnl': stat['pnl'], 'closed': closed
                }
        
        # ===== CONSENSUS CASCADE: Assign tiers =====
        all_combo_names = set()
        all_combo_names.update(all_combos_by_dimension['5D'].keys())
        all_combo_names.update(all_combos_by_dimension['4D'].keys())
        all_combo_names.update(all_combos_by_dimension['3D'].keys())
        all_combo_names.update(all_combos_by_dimension['2D'].keys())
        
        for combo_name in sorted(all_combo_names):
            assigned_tier = None
            assigned_level = None
            
            # CASCADE: Check 5D → 4D → 3D → 2D
            for dimension in ['5D', '4D', '3D', '2D']:
                if combo_name in all_combos_by_dimension[dimension]:
                    stats = all_combos_by_dimension[dimension][combo_name]
                    wr = stats['wr']
                    avg_pnl = stats['avg_pnl']
                    closed = stats['closed']
                    
                    # Check Tier-1, then Tier-2, then Tier-3
                    for tier_level in [1, 2, 3]:
                        tier_name = self._check_tier_qualification(wr, avg_pnl, closed, tier_level)
                        if tier_name:
                            assigned_tier = tier_name
                            assigned_level = dimension
                            break
                    
                    if assigned_tier:
                        break
            
            # Assign to tier list (STORE CLEAN COMBO NAMES WITHOUT SUFFIX for tier_lookup.py matching)
            if assigned_tier == "Tier-1":
                tiers['tier1'].append(combo_name)
            elif assigned_tier == "Tier-2":
                tiers['tier2'].append(combo_name)
            elif assigned_tier == "Tier-3":
                tiers['tier3'].append(combo_name)
            else:
                tiers['tierx'].append(combo_name)
        
        # Add timestamp and config version
        tiers['generated_at'] = datetime.now(timezone(timedelta(hours=7))).strftime('%Y-%m-%d %H:%M:%S GMT+7')
        tiers['config_version'] = 'C (CONSENSUS CASCADE + OPTION B)'
        
        return tiers
    
    def _generate_hierarchy_ranking(self):
        """Generate 5D, 4D, 3D, 2D hierarchy ranking for decision-making (top to bottom)"""
        output = []
        min_trades = TIER_THRESHOLDS.get("min_trades", 25)
        
        # ===== 5D COMBOS (MOST GRANULAR) =====
        output.append("📊 5-DIMENSIONAL COMBOS (TimeFrame × Direction × Route × Regime × Symbol_Group)")
        output.append("─" * 160)
        
        stats_5d = self._aggregate_by_dimensions_with_symbol(['timeframe', 'signal_type', 'route', 'regime'])
        
        all_5d = []
        for key, stat in stats_5d.items():
            closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
            if closed >= min_trades:
                win_count = stat['tp'] + stat['timeout_win']
                wr = (win_count / closed) if closed > 0 else 0
                avg_pnl = stat['pnl'] / closed if closed > 0 else 0
                all_5d.append({
                    'name': f"TF_DIR_ROUTE_REGIME_SG_{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}",
                    'wr': wr,
                    'pnl': stat['pnl'],
                    'avg_pnl': avg_pnl,
                    'closed': closed,
                    'tp': stat['tp'],
                    'sl': stat['sl']
                })
        
        all_5d.sort(key=lambda x: (x['wr'], x['pnl']), reverse=True)
        
        if all_5d:
            output.append("\n  Top 10 5D Combos by WR:")
            for combo in all_5d[:10]:
                output.append(f"    ✓ {combo['name']:80} | WR: {combo['wr']*100:5.1f}% | P&L: ${combo['pnl']:+8.2f} | Avg: ${combo['avg_pnl']:+.2f} | Closed: {combo['closed']}")
        else:
            output.append("\n  No 5D combos meet minimum trade threshold yet.")
        
        output.append("")
        
        # ===== 4D COMBOS =====
        output.append("📊 4-DIMENSIONAL COMBOS (TimeFrame × Direction × Route × Regime)")
        output.append("─" * 160)
        
        stats_4d = self._aggregate_by_dimensions(['timeframe', 'signal_type', 'route', 'regime'])
        
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
        
        # ===== 3D COMBOS =====
        output.append("📊 3-DIMENSIONAL COMBOS (Sample: TF_DIR_ROUTE, TF_DIR_REGIME, DIR_ROUTE_REGIME, TF_ROUTE_REGIME)")
        output.append("─" * 160)
        
        combos_3d = [
            (['timeframe', 'signal_type', 'route'], 'TF_DIR_ROUTE'),
            (['timeframe', 'signal_type', 'regime'], 'TF_DIR_REGIME'),
            (['signal_type', 'route', 'regime'], 'DIR_ROUTE_REGIME'),
            (['timeframe', 'route', 'regime'], 'TF_ROUTE_REGIME'),
        ]
        
        all_3d = []
        for dimensions, label in combos_3d:
            stats = self._aggregate_by_dimensions(dimensions)
            
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
        
        # ===== 2D COMBOS (BROADEST) =====
        output.append("📊 2-DIMENSIONAL COMBOS (TF_DIR, TF_REGIME, DIR_REGIME, DIR_ROUTE, ROUTE_REGIME)")
        output.append("─" * 160)
        
        combos_2d = [
            (['timeframe', 'signal_type'], 'TF_DIR'),
            (['timeframe', 'regime'], 'TF_REGIME'),
            (['signal_type', 'regime'], 'DIR_REGIME'),
            (['signal_type', 'route'], 'DIR_ROUTE'),
            (['route', 'regime'], 'ROUTE_REGIME'),
        ]
        
        all_2d = []
        for dimensions, label in combos_2d:
            stats = self._aggregate_by_dimensions(dimensions)
            
            for key, stat in stats.items():
                closed = stat['tp'] + stat['sl'] + stat['timeout_win'] + stat['timeout_loss']
                if closed >= min_trades:
                    win_count = stat['tp'] + stat['timeout_win']
                    wr = (win_count / closed) if closed > 0 else 0
                    avg_pnl = stat['pnl'] / closed if closed > 0 else 0
                    all_2d.append({
                        'name': f"{label}_{key[0]}_{key[1]}",
                        'wr': wr,
                        'pnl': stat['pnl'],
                        'avg_pnl': avg_pnl,
                        'closed': closed,
                        'tp': stat['tp'],
                        'sl': stat['sl']
                    })
        
        all_2d.sort(key=lambda x: (x['wr'], x['pnl']), reverse=True)
        
        if all_2d:
            output.append("\n  Top 8 2D Combos by WR:")
            for combo in all_2d[:8]:
                output.append(f"    ✓ {combo['name']:45} | WR: {combo['wr']*100:5.1f}% | P&L: ${combo['pnl']:+8.2f} | Avg: ${combo['avg_pnl']:+.2f} | Closed: {combo['closed']}")
        else:
            output.append("\n  No 2D combos meet minimum trade threshold yet.")
        
        output.append("")
        output.append("=" * 160)
        
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
