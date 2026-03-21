#!/usr/bin/env python3
"""
PEC Enhanced Reporter - Multi-Dimensional Signal Tracking
Breaks down PEC results by: TimeFrame, Direction, Confidence, Regime, Route

CRITICAL ARCHITECTURE (Option B - Correct Separation of Concerns):
- Daemon writes raw signals to SIGNALS_MASTER.jsonl (NO P&L calculation)
- Reporter READS signals and RECALCULATES P&L with Phase 1-3 logic
- Single source of truth: Reporter's _calculate_pnl_usd() function
- All P&L values use consistent, verified calculation logic

This prevents daemon from pre-calculating P&L with old/broken logic.

Enhanced Features:
- Position size & leverage header
- Extended columns: Route, Exit Price, Exit Time, Duration
- 5D aggregates: TF, Direction, Route, Regime, Confidence
- Dynamic tier configuration from tier_config.py
- Phase 1-3 P&L fixes applied at report-time (not fire-time)
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
        # Use SIGNALS_MASTER.jsonl (REAL-TIME source - includes foundation + new signals)
        # Falls back to CLEAN for foundation-only baseline if needed
        if sent_signals_file is None:
            workspace = "/Users/geniustarigan/.openclaw/workspace"
            
            # Primary: SIGNALS_MASTER.jsonl (live, includes both FOUNDATION + NEW_LIVE)
            signals_master = os.path.join(workspace, "SIGNALS_MASTER.jsonl")
            if os.path.exists(signals_master) and os.path.getsize(signals_master) > 1000:
                sent_signals_file = signals_master
                print(f"[INFO] Using SIGNALS_MASTER.jsonl (live data): {signals_master}", flush=True)
            else:
                # Fallback: SIGNALS_MASTER_CLEAN_2538 (foundation-only if live unavailable)
                signals_clean = os.path.join(workspace, "SIGNALS_MASTER_CLEAN_2538.jsonl")
                if os.path.exists(signals_clean) and os.path.getsize(signals_clean) > 1000:
                    sent_signals_file = signals_clean
                    print(f"[INFO] Using SIGNALS_MASTER_CLEAN_2538.jsonl (foundation baseline): {signals_clean}", flush=True)
                else:
                    # Fallback: CUMULATIVE (immutable hourly snapshots for historical data)
                    import glob
                    cumulative_files = sorted(glob.glob(os.path.join(workspace, "SENT_SIGNALS_CUMULATIVE_*.jsonl")))
                    if cumulative_files:
                        sent_signals_file = cumulative_files[-1]
                        print(f"[INFO] Using CUMULATIVE: {os.path.basename(sent_signals_file)}", flush=True)
                    else:
                        # Last resort: ARCHIVE (Foundation baseline)
                        sent_signals_file = os.path.join(workspace, "SENT_SIGNALS_ARCHIVE_2026-03-05.jsonl")
                        print(f"[INFO] Using ARCHIVE file: SENT_SIGNALS_ARCHIVE_2026-03-05.jsonl", flush=True)
        
        self.sent_signals_file = sent_signals_file
        self.signals = []
        self.load_signals()
    
    def load_signals(self):
        """Load signals from SIGNALS_MASTER.jsonl (single source of truth)
        
        If using SIGNALS_MASTER: all signals in one file (complete history)
        If using CUMULATIVE: load cumulative + append any newer signals from SIGNALS_MASTER
        """
        workspace = "/Users/geniustarigan/.openclaw/workspace"
        
        if not os.path.exists(self.sent_signals_file):
            print(f"[WARN] {self.sent_signals_file} not found")
            return
        
        try:
            with open(self.sent_signals_file, 'r') as f:
                count = 0
                for line in f:
                    try:
                        signal = json.loads(line.strip())
                        # Normalize field names: convert 'direction' → 'signal_type' if needed
                        if 'direction' in signal and 'signal_type' not in signal:
                            signal['signal_type'] = signal['direction']
                        self.signals.append(signal)
                        count += 1
                    except:
                        pass
                print(f"[INFO] Loaded {count} signals from {os.path.basename(self.sent_signals_file)}", flush=True)
        except Exception as e:
            print(f"[WARN] Error loading signals: {e}")
        
        # If NOT using SIGNALS_MASTER already, try to append newer signals from it
        # This ensures we always have the latest signals for today
        if "SIGNALS_MASTER" not in self.sent_signals_file:
            signals_master = os.path.join(workspace, "SIGNALS_MASTER.jsonl")
            if os.path.exists(signals_master):
                try:
                    # Get the latest fired_time from currently loaded signals
                    if self.signals:
                        latest_fired = max((datetime.fromisoformat(s.get('fired_time_utc', '').replace('Z', '+00:00')) 
                                          for s in self.signals if s.get('fired_time_utc')), 
                                         default=datetime.min.replace(tzinfo=timezone.utc))
                    else:
                        latest_fired = datetime.min.replace(tzinfo=timezone.utc)
                    
                    # Load newer signals from SIGNALS_MASTER
                    appended_count = 0
                    with open(signals_master, 'r') as f:
                        for line in f:
                            try:
                                signal = json.loads(line.strip())
                                fired_utc_str = signal.get('fired_time_utc', '')
                                if fired_utc_str:
                                    fired_dt = datetime.fromisoformat(fired_utc_str.replace('Z', '+00:00'))
                                    # Only add if newer than what we already have
                                    if fired_dt > latest_fired:
                                        # Check for duplicate uuid
                                        signal_uuid = signal.get('signal_uuid')
                                        if signal_uuid and not any(s.get('signal_uuid') == signal_uuid for s in self.signals):
                                            if 'direction' in signal and 'signal_type' not in signal:
                                                signal['signal_type'] = signal['direction']
                                            self.signals.append(signal)
                                            appended_count += 1
                            except:
                                pass
                    if appended_count > 0:
                        print(f"[INFO] Appended {appended_count} newer signals from SIGNALS_MASTER.jsonl", flush=True)
                except Exception as e:
                    print(f"[WARN] Error appending from SIGNALS_MASTER: {e}")
    
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
        """Calculate P&L USD using notional position of $1000 ($100 margin × 10x leverage)"""
        try:
            if not entry_price or entry_price == 0 or not exit_price or exit_price == 0:
                return None
            
            entry = float(entry_price)
            exit_val = float(exit_price)
            notional_position = 1000.0  # $100 margin × 10x leverage = $1000 notional
            
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
        detail_lines.append("📋 DETAILED SIGNAL LIST: FIXED POSITION SIZE $100, LEVERAGE 10x, NOTIONAL $1000")
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
            
            # P&L - RECALCULATE using notional position of $1000 ($100 margin × 10x leverage)
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
        
        # === SECTION 1 & 2 (Summary Statistics) ===
        # FOUNDATION_CUTOFF_TIME: LOCKED to 2026-03-19T18:03:17.605610 (last signal in clean baseline)
        # Foundation: All signals fired <= this time (IMMUTABLE snapshot)
        # New: All signals fired > this time (dynamic, calculated as Total - Foundation)
        # Parse cutoff as naive datetime (signals are stored as naive UTC times)
        cutoff_naive = datetime.fromisoformat('2026-03-19T18:03:17.605610')
        
        foundation_signals = []
        new_signals = []
        for s in self.signals:
            try:
                # Parse fired_time_utc - comes as naive datetime string (assumed UTC)
                fired_str = s.get('fired_time_utc', '')
                if not fired_str:
                    foundation_signals.append(s)
                    continue
                
                # Remove Z or timezone info to get naive datetime
                fired_str = fired_str.replace('Z', '').replace('+00:00', '')
                fired = datetime.fromisoformat(fired_str)
                
                # Compare naive datetimes (both assumed UTC)
                if fired <= cutoff_naive:
                    foundation_signals.append(s)
                else:
                    new_signals.append(s)
            except Exception as e:
                # If parse fails, default to foundation (safe fallback)
                foundation_signals.append(s)
        
        # Calculate foundation stats DYNAMICALLY (not hardcoded)
        foundation_stats = self._analyze_signal_group(foundation_signals)
        
        # === DISPLAY FOUNDATION BASELINE (LOCKED - immutable snapshot) ===
        report.append("=" * 200)
        report.append("🔒 FOUNDATION BASELINE (LOCKED - All signals fired <= 2026-03-19T18:03:17.605610)")
        report.append("=" * 200)
        report.append(f"Total Signals: {foundation_stats['total']} | Closed: {foundation_stats['closed']} | WR: {foundation_stats['wr']:.1f}%")
        report.append(f"LONG WR: {self._calculate_wr_by_direction(foundation_signals, 'LONG'):.1f}% | SHORT WR: {self._calculate_wr_by_direction(foundation_signals, 'SHORT'):.1f}% | P&L: ${foundation_stats['total_pnl']:+,.2f}")
        report.append("(This baseline is immutable - NEW signals are calculated as TOTAL - FOUNDATION)")
        report.append("")
        
        # SECTION 1: Foundation + New
        combined = foundation_signals + new_signals
        combined_stats = self._analyze_signal_group(combined)
        
        report.append("=" * 200)
        report.append("📊 SECTION 1: TOTAL SIGNALS (Foundation + New)")
        report.append("=" * 200)
        report.append(f"Total Signals Loaded: {combined_stats['total']}")
        report.append("")
        report.append("SIGNAL BREAKDOWN (All signals shown for audit trail):")
        report.append(f"  INCLUDED IN METRICS (TP/SL/TIMEOUT/OPEN):")
        report.append(f"    • TP_HIT: {combined_stats['tp']}")
        report.append(f"    • SL_HIT: {combined_stats['sl']}")
        report.append(f"    • TIMEOUT: {combined_stats['timeout']}")
        report.append(f"    • OPEN: {combined_stats['open']}")
        report.append(f"    Subtotal (Counted in WR & P&L): {combined_stats['tp'] + combined_stats['sl'] + combined_stats['timeout'] + combined_stats['open']}")
        report.append(f"")
        report.append(f"  EXCLUDED FROM METRICS (Not counted in WR or P&L):")
        report.append(f"    • REJECTED_NOT_SENT_TELEGRAM: {combined_stats['rejected']} (never sent to traders)")
        report.append(f"    • STALE_TIMEOUT: {combined_stats['stale']} ⚠️  DATA QUALITY ISSUE - COMPLETELY EXCLUDED")
        report.append(f"    Subtotal (Excluded from all calculations): {combined_stats['rejected'] + combined_stats['stale']}")
        report.append(f"")
        report.append(f"BREAKDOWN VERIFICATION:")
        report.append(f"  Included ({combined_stats['tp'] + combined_stats['sl'] + combined_stats['timeout'] + combined_stats['open']}) + Excluded ({combined_stats['rejected'] + combined_stats['stale']}) = Total ({combined_stats['total_accounted']})")
        report.append(f"  ✓ Verified: All {combined_stats['total']} signals in audit trail" if combined_stats['total'] == combined_stats['total_accounted'] else f"  ✗ MISMATCH: {combined_stats['total']} loaded but {combined_stats['total_accounted']} accounted")
        report.append("")
        report.append("CLOSED TRADES ANALYSIS (Backtest Signals Only):")
        report.append(f"  Closed Trades (Clean Data): {combined_stats['closed']}")
        report.append(f"    • TP_HIT: {combined_stats['tp']}")
        report.append(f"    • SL_HIT: {combined_stats['sl']}")
        report.append(f"    • TimeOut: {combined_stats['timeout']}")
        report.append(f"      - TimeOut Win: {combined_stats['timeout_win']} (approximate)")
        report.append(f"      - TimeOut Loss: {combined_stats['timeout_loss']} (approximate)")
        report.append(f"")
        report.append(f"Overall Win Rate: {combined_stats['wr']:.2f}%")
        report.append(f"Calculation: ({combined_stats['tp']} TP + {combined_stats['timeout_win']} TIMEOUT_WIN) / {combined_stats['closed']} Closed = {combined_stats['wins']} / {combined_stats['closed']} = {combined_stats['wr']:.2f}%")
        report.append(f"")
        report.append(f"Total P&L (Clean Data): ${combined_stats['total_pnl']:+,.2f}")
        report.append(f"Avg P&L per Signal: ${combined_stats['avg_pnl_signal']:+.2f}")
        report.append(f"Avg P&L per Closed Trade: ${combined_stats['avg_pnl_closed']:+.2f}")
        report.append(f"")
        report.append(f"P&L BREAKDOWN (STALE_TIMEOUT completely excluded):")
        report.append(f"")
        report.append(f"  INCLUDED IN TOTAL P&L (Counted in metrics):")
        report.append(f"    • TP_HIT:    {combined_stats['tp_pnl']:+12,.2f}")
        report.append(f"    • SL_HIT:    {combined_stats['sl_pnl']:+12,.2f}")
        report.append(f"    • TIMEOUT:   {combined_stats['timeout_pnl']:+12,.2f}")
        report.append(f"    • OPEN:      {combined_stats['open_pnl']:+12,.2f} (unrealized)")
        report.append(f"    Subtotal (Backtest P&L): {combined_stats['tp_pnl'] + combined_stats['sl_pnl'] + combined_stats['timeout_pnl'] + combined_stats['open_pnl']:+12,.2f}")
        report.append(f"")
        report.append(f"  EXCLUDED FROM TOTAL P&L (Not counted in any metric):")
        report.append(f"    • REJECTED:  {combined_stats['rejected_pnl']:+12,.2f} (never sent to traders, excluded from WR)")
        report.append(f"    • STALE:     ⚠️  NOT CALCULATED (data quality - completely excluded)")
        report.append(f"    Subtotal (Excluded P&L): {combined_stats['rejected_pnl']:+12,.2f}")
        report.append(f"")
        report.append(f"  VALIDATION:")
        included_sum = combined_stats['tp_pnl'] + combined_stats['sl_pnl'] + combined_stats['timeout_pnl'] + combined_stats['open_pnl']
        report.append(f"    Included in Total P&L: {included_sum:+12,.2f}")
        report.append(f"    Total P&L Reported:    {combined_stats['total_pnl']:+12,.2f}")
        report.append(f"    ✓ Verified: P&L matches Total P&L" if abs(included_sum - combined_stats['total_pnl']) < 0.01 else f"    ✗ MISMATCH: Included {included_sum:+.2f} ≠ Total {combined_stats['total_pnl']:+.2f}")
        report.append(f"")
        report.append(f"Average P&L per Count:")
        report.append(f"  Avg P&L TP per Count TP: ${combined_stats['avg_tp_pnl']:+.2f}" if combined_stats['tp'] > 0 else f"  Avg P&L TP per Count TP: N/A (0 TP trades)")
        report.append(f"  Avg P&L SL per Count SL: ${combined_stats['avg_sl_pnl']:+.2f}" if combined_stats['sl'] > 0 else f"  Avg P&L SL per Count SL: N/A (0 SL trades)")
        report.append("")
        
        # SECTION 2: New Only (Show NEW_LIVE signals - all signals after initial foundation)
        # "NEW" = signals fired on Mar 21+ onwards (not FOUNDATION, and after rebuild cutoff)
        # NOTE: Must convert UTC fired_time to GMT+7 for proper date comparison
        new_signals_by_origin = []
        for s in self.signals:
            if s.get('signal_origin') != 'FOUNDATION':
                try:
                    # Parse UTC time and convert to GMT+7
                    fired_utc_str = s.get('fired_time_utc', '')
                    if fired_utc_str:
                        fired_utc = datetime.fromisoformat(fired_utc_str.replace('Z', '+00:00'))
                        fired_gmt7 = fired_utc + timedelta(hours=7)
                        fired_gmt7_date = fired_gmt7.strftime('%Y-%m-%d')
                        # Include if fired on or after 2026-03-21 GMT+7
                        if fired_gmt7_date >= '2026-03-21':
                            new_signals_by_origin.append(s)
                except:
                    pass
        new_stats = self._analyze_signal_group(new_signals_by_origin)
        
        report.append("=" * 200)
        report.append("📊 SECTION 2: TOTAL SIGNALS (NEW ONLY - Mar 21+ onwards)")
        report.append("=" * 200)
        report.append(f"Total Signals Loaded (NEW ONLY): {new_stats['total']}")
        report.append("")
        report.append("SIGNAL BREAKDOWN (All signals shown for audit trail):")
        report.append(f"  INCLUDED IN METRICS (TP/SL/TIMEOUT/OPEN):")
        report.append(f"    • TP_HIT: {new_stats['tp']}")
        report.append(f"    • SL_HIT: {new_stats['sl']}")
        report.append(f"    • TIMEOUT: {new_stats['timeout']}")
        report.append(f"    • OPEN: {new_stats['open']}")
        report.append(f"    Subtotal (Counted in WR & P&L): {new_stats['tp'] + new_stats['sl'] + new_stats['timeout'] + new_stats['open']}")
        report.append(f"")
        report.append(f"  EXCLUDED FROM METRICS (Not counted in WR or P&L):")
        report.append(f"    • REJECTED_NOT_SENT_TELEGRAM: {new_stats['rejected']} (never sent to traders)")
        report.append(f"    • STALE_TIMEOUT: {new_stats['stale']} ⚠️  DATA QUALITY ISSUE - COMPLETELY EXCLUDED")
        report.append(f"    Subtotal (Excluded from all calculations): {new_stats['rejected'] + new_stats['stale']}")
        report.append(f"")
        report.append(f"BREAKDOWN VERIFICATION:")
        report.append(f"  Included ({new_stats['tp'] + new_stats['sl'] + new_stats['timeout'] + new_stats['open']}) + Excluded ({new_stats['rejected'] + new_stats['stale']}) = Total ({new_stats['total_accounted']})")
        report.append(f"  ✓ Verified: All {new_stats['total']} signals in audit trail" if new_stats['total'] == new_stats['total_accounted'] else f"  ✗ MISMATCH: {new_stats['total']} loaded but {new_stats['total_accounted']} accounted")
        report.append("")
        report.append("CLOSED TRADES ANALYSIS (Backtest Signals Only):")
        report.append(f"  Closed Trades (Clean Data): {new_stats['closed']}")
        report.append(f"    • TP_HIT: {new_stats['tp']}")
        report.append(f"    • SL_HIT: {new_stats['sl']}")
        report.append(f"    • TimeOut: {new_stats['timeout']}")
        report.append(f"      - TimeOut Win: {new_stats['timeout_win']} (approximate)")
        report.append(f"      - TimeOut Loss: {new_stats['timeout_loss']} (approximate)")
        report.append(f"")
        report.append(f"Overall Win Rate: {new_stats['wr']:.2f}%")
        if new_stats['closed'] > 0:
            report.append(f"Calculation: ({new_stats['tp']} TP + {new_stats['timeout_win']} TIMEOUT_WIN) / {new_stats['closed']} Closed = {new_stats['wins']} / {new_stats['closed']} = {new_stats['wr']:.2f}%")
        else:
            report.append(f"Note: Insufficient closed trades for reliable WR")
        report.append(f"")
        report.append(f"Total P&L (Clean Data): ${new_stats['total_pnl']:+,.2f}")
        report.append(f"Avg P&L per Signal: ${new_stats['avg_pnl_signal']:+.2f}")
        report.append(f"Avg P&L per Closed Trade: ${new_stats['avg_pnl_closed']:+.2f}")
        report.append(f"")
        report.append(f"P&L BREAKDOWN (STALE_TIMEOUT completely excluded):")
        report.append(f"")
        report.append(f"  INCLUDED IN TOTAL P&L (Counted in metrics):")
        report.append(f"    • TP_HIT:    {new_stats['tp_pnl']:+12,.2f}")
        report.append(f"    • SL_HIT:    {new_stats['sl_pnl']:+12,.2f}")
        report.append(f"    • TIMEOUT:   {new_stats['timeout_pnl']:+12,.2f}")
        report.append(f"    • OPEN:      {new_stats['open_pnl']:+12,.2f} (unrealized)")
        report.append(f"    Subtotal (Backtest P&L): {new_stats['tp_pnl'] + new_stats['sl_pnl'] + new_stats['timeout_pnl'] + new_stats['open_pnl']:+12,.2f}")
        report.append(f"")
        report.append(f"  EXCLUDED FROM TOTAL P&L (Not counted in any metric):")
        report.append(f"    • REJECTED:  {new_stats['rejected_pnl']:+12,.2f} (never sent to traders, excluded from WR)")
        report.append(f"    • STALE:     ⚠️  NOT CALCULATED (data quality - completely excluded)")
        report.append(f"    Subtotal (Excluded P&L): {new_stats['rejected_pnl']:+12,.2f}")
        report.append(f"")
        report.append(f"  VALIDATION:")
        new_included_sum = new_stats['tp_pnl'] + new_stats['sl_pnl'] + new_stats['timeout_pnl'] + new_stats['open_pnl']
        report.append(f"    Included in Total P&L: {new_included_sum:+12,.2f}")
        report.append(f"    Total P&L Reported:    {new_stats['total_pnl']:+12,.2f}")
        report.append(f"    ✓ Verified: P&L matches Total P&L" if abs(new_included_sum - new_stats['total_pnl']) < 0.01 else f"    ✗ MISMATCH: Included {new_included_sum:+.2f} ≠ Total {new_stats['total_pnl']:+.2f}")
        report.append(f"")
        report.append(f"Average P&L per Count:")
        report.append(f"  Avg P&L TP per Count TP: ${new_stats['avg_tp_pnl']:+.2f}" if new_stats['tp'] > 0 else f"  Avg P&L TP per Count TP: N/A (0 TP trades)")
        report.append(f"  Avg P&L SL per Count SL: ${new_stats['avg_sl_pnl']:+.2f}" if new_stats['sl'] > 0 else f"  Avg P&L SL per Count SL: N/A (0 SL trades)")
        report.append(f"")
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
            
            # RECALCULATE P&L using notional position of $1000 ($100 margin × 10x leverage)
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
            # Skip stale timeouts from metrics (overdue quality issues)
            if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'):
                continue
            if s.get('status') == 'STALE_TIMEOUT':
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
        total_rejected = sum(1 for s in self.signals if s.get('status') == 'REJECTED_NOT_SENT_TELEGRAM')
        total_stale_status = sum(1 for s in self.signals if s.get('status') == 'STALE_TIMEOUT')
        
        # Closed trades = TP + SL + TIMEOUT_WIN + TIMEOUT_LOSS
        closed_signals = total_tp + total_sl + timeout_wins + timeout_losses
        
        # Win Rate = (TP + TIMEOUT_WIN) / Closed Trades
        win_count = total_tp + timeout_wins
        overall_wr = (win_count / closed_signals * 100) if closed_signals > 0 else 0
        
        # Calculate total P&L (EXCLUDING stale timeouts)
        # USE STORED PNL_USD when available (may be based on max_bar closing price for timeouts)
        # Fall back to recalculation if stored value missing
        # Stale timeouts = zero P&L (overdue quality issues)
        total_pnl = 0.0
        for s in self.signals:
            # SKIP stale timeouts completely (overdue quality issues, zero P&L)
            if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'):
                continue
            if s.get('status') == 'STALE_TIMEOUT':
                continue
            
            if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                # RECALCULATE P&L with Phase 1-3 logic (ignore daemon's stored pnl_usd)
                # This ensures all P&L uses consistent, correct calculation logic
                pnl_calc = self._calculate_pnl_usd(
                    s.get('entry_price'),
                    s.get('actual_exit_price'),
                    s.get('signal_type')
                )
                if pnl_calc is not None:
                    total_pnl += pnl_calc
        
        # Calculate P&L breakdown by outcome type (EXCLUDING stale timeouts)
        pnl_tp = 0.0
        pnl_sl = 0.0
        pnl_timeout_win = 0.0
        pnl_timeout_loss = 0.0
        
        for s in self.signals:
            # SKIP stale timeouts completely (overdue quality issues, zero P&L)
            if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'):
                continue
            if s.get('status') == 'STALE_TIMEOUT':
                continue
            
            status = s.get('status', 'OPEN')
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                # RECALCULATE P&L with Phase 1-3 logic (ignore daemon's stored pnl_usd)
                pnl_val_calc = self._calculate_pnl_usd(
                    s.get('entry_price'),
                    s.get('actual_exit_price'),
                    s.get('signal_type')
                )
                pnl_val = pnl_val_calc if pnl_val_calc is not None else 0.0
                
                # Categorize by outcome type
                if status == 'TP_HIT':
                    pnl_tp += pnl_val
                elif status == 'SL_HIT':
                    pnl_sl += pnl_val
                elif status == 'TIMEOUT':
                    # Separate TIMEOUT by P&L sign
                    if pnl_val > 0:
                        pnl_timeout_win += pnl_val
                    elif pnl_val < 0:
                        pnl_timeout_loss += pnl_val
        
        # Calculate average P&L per count for each outcome type
        avg_pnl_per_tp = pnl_tp / total_tp if total_tp > 0 else 0
        avg_pnl_per_sl = pnl_sl / total_sl if total_sl > 0 else 0
        avg_pnl_per_timeout_win = pnl_timeout_win / timeout_wins if timeout_wins > 0 else 0
        avg_pnl_per_timeout_loss = pnl_timeout_loss / timeout_losses if timeout_losses > 0 else 0
        
        # Calculate average P&L per signal
        avg_pnl_per_signal = total_pnl / total_signals if total_signals > 0 else 0
        avg_pnl_per_trade = total_pnl / closed_signals if closed_signals > 0 else 0
        
        # Calculate average durations for summary (TP/SL only - stale timeouts already excluded by status)
        # Note: Stale timeouts have status='TIMEOUT', not 'TP_HIT'/'SL_HIT', so they never appear in these calculations
        avg_tp_duration_summary = self._calculate_avg_duration_by_status(self.signals, 'TP_HIT')
        avg_sl_duration_summary = self._calculate_avg_duration_by_status(self.signals, 'SL_HIT')
        
        # Calculate ACTUAL MAX TIMEOUT WINDOW by timeframe
        # EXCLUDING stale timeouts (quality issues - overdue signals that inflate averages)
        # Only calculate from clean timeouts within expected design limits (with 10% tolerance for timing slippage)
        max_timeout_clean = {'15min': 0, '30min': 0, '1h': 0}
        expected_max_by_tf = {'15min': 15*15*60, '30min': 10*30*60, '1h': 5*60*60}  # Expected maximums (in seconds)
        tolerance = 0.10  # Allow 10% tolerance for minor timing variations
        
        for s in self.signals:
            # SKIP stale timeouts completely (data quality issues)
            if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'):
                continue
            # Also skip STALE_TIMEOUT status
            if s.get('status') == 'STALE_TIMEOUT':
                continue
            
            if s.get('status') == 'TIMEOUT' and s.get('fired_time_utc') and s.get('closed_at'):
                try:
                    fired = datetime.fromisoformat(s.get('fired_time_utc').replace('Z', '+00:00'))
                    closed = datetime.fromisoformat(s.get('closed_at').replace('Z', '+00:00'))
                    delta = closed - fired
                    duration_seconds = int(delta.total_seconds())
                    
                    tf = s.get('timeframe', '')
                    if tf in max_timeout_clean:
                        # Only count if within expected limit + 10% tolerance (clean timeout)
                        # E.g., 1h design window is 5h = 18000s, with 10% tolerance = 19800s (5h 30m)
                        threshold = int(expected_max_by_tf[tf] * (1 + tolerance))
                        if duration_seconds <= threshold:
                            max_timeout_clean[tf] = max(max_timeout_clean[tf], duration_seconds)
                except:
                    pass
        
        # Format timeout windows - show DESIGNED maximum (what system is set for)
        # These are the theoretical limits per timeframe design
        timeout_window_15min = "3h 45m"   # 15 bars × 15min = 225min = 3h45m (designed)
        timeout_window_30min = "5h 0m"    # 10 bars × 30min = 300min = 5h (designed)
        timeout_window_1h = "5h 0m"       # 5 bars × 60min = 300min = 5h (designed)
        
        # Format actual max (clean within limit, stale excluded)
        actual_max_15min = self._format_duration_hm(max_timeout_clean['15min']) if max_timeout_clean['15min'] > 0 else "None"
        actual_max_30min = self._format_duration_hm(max_timeout_clean['30min']) if max_timeout_clean['30min'] > 0 else "None"
        actual_max_1h = self._format_duration_hm(max_timeout_clean['1h']) if max_timeout_clean['1h'] > 0 else "None"
        
        # Calculate fired time range and count per date (with dedup and telegram tracking)
        signals_by_date = defaultdict(list)
        signals_by_date_dedup = defaultdict(set)  # Track unique UUIDs per date
        signals_by_date_tele = defaultdict(list)  # Track telegram-sent only
        all_fired_times = []
        
        # Read SIGNALS_MASTER for ALL dates to get accurate Unique/Tele metrics
        today_str = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).strftime('%Y-%m-%d')
        master_stats_all = defaultdict(lambda: {'total': 0, 'unique': set(), 'tele': 0})
        
        if os.path.exists(os.path.join("/Users/geniustarigan/.openclaw/workspace", "SIGNALS_MASTER.jsonl")):
            try:
                with open(os.path.join("/Users/geniustarigan/.openclaw/workspace", "SIGNALS_MASTER.jsonl"), 'r') as f:
                    for line in f:
                        try:
                            sig = json.loads(line)
                            fired_utc = sig.get('fired_time_utc', '')
                            if fired_utc:
                                # Extract date from fired_time_utc (format: YYYY-MM-DDTHH:MM:SS...)
                                date_str = fired_utc.split('T')[0]
                                master_stats_all[date_str]['total'] += 1
                                
                                uuid = sig.get('signal_uuid')
                                if uuid:
                                    master_stats_all[date_str]['unique'].add(uuid)
                                
                                if sig.get('sent_time_utc') and sig.get('status') != 'REJECTED_NOT_SENT_TELEGRAM':
                                    master_stats_all[date_str]['tele'] += 1
                        except:
                            pass
            except Exception as e:
                pass
        
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
                    
                    # Track unique UUIDs (dedup)
                    uuid = s.get('signal_uuid')
                    if uuid:
                        signals_by_date_dedup[date_str].add(uuid)
                    
                    # Track telegram-sent signals (have sent_time_utc and not REJECTED)
                    if s.get('sent_time_utc') and s.get('status') != 'REJECTED_NOT_SENT_TELEGRAM':
                        signals_by_date_tele[date_str].append(s)
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
            
            # Calculate metrics from loaded signals for this date (handles timezone shifts)
            signals_for_date = [s for s in self.signals if s.get('fired_time_utc')]
            signals_for_date = [s for s in signals_for_date if 'T' in s.get('fired_time_utc', '')]
            
            # Re-parse and group to match the date_str (which is already in GMT+7)
            date_signals = []
            for s in signals_for_date:
                try:
                    dt = datetime.fromisoformat(s.get('fired_time_utc').replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    gmt7 = dt.astimezone(timezone(timedelta(hours=7)))
                    if gmt7.strftime('%Y-%m-%d') == date_str:
                        date_signals.append(s)
                except:
                    pass
            
            total_fired = len(times)  # Use times count (already grouped by GMT+7 date)
            
            # For today (Mar 06), use SIGNALS_MASTER stats if available (more complete than loaded signals)
            if date_str == today_date and date_str in master_stats_all:
                unique_uuids = master_stats_all[date_str]['unique']
                unique_fired = len(unique_uuids)
                tele_sent = master_stats_all[date_str]['tele']
            else:
                # For past dates, use loaded signals (which have all the data)
                unique_uuids = set(s.get('signal_uuid') for s in date_signals if s.get('signal_uuid'))
                unique_fired = len(unique_uuids) if unique_uuids else None  # None means not available (no UUIDs in data)
                tele_sent = sum(1 for s in date_signals if s.get('sent_time_utc') and s.get('status') != 'REJECTED_NOT_SENT_TELEGRAM')
            
            # Format metrics string: show "-" for Unique if not available
            unique_str = str(unique_fired) if unique_fired else "-"
            metrics_str = f"Total Fired={total_fired} | Unique Fired={unique_str} | Tele Sent={tele_sent}"
            
            date_earliest = min(times)
            date_latest = max(times)
            beginning_time_str = date_earliest.strftime('%H:%M:%S')
            last_time_str = date_latest.strftime('%H:%M:%S')
            
            # Add immutable label for past dates, accumulating for today
            if date_str == today_date:
                status_label = "🔄 still accumulating"
            else:
                status_label = "✓ IMMUTABLE"
            
            date_line = f"  {date_str}: {metrics_str} | {beginning_time_str}-{last_time_str} | {status_label}"
            date_summary_lines.append(date_line)
        
        # Display Summary (with stale timeout exclusion note) - Match user's exact template
        stale_timeout_count = sum(1 for s in self.signals if s.get('data_quality_flag') and 'STALE_TIMEOUT' in s.get('data_quality_flag'))
        
        report.append("")
        report.append("🔒 FOUNDATION BASELINE (IMMUTABLE - Locked at commit 8f58cec)")
        report.append(f"Total Signals: {foundation_stats['total']} | Closed: {foundation_stats['closed']} | WR: {foundation_stats['wr']:.1f}%")
        report.append(f"LONG WR: {self._calculate_wr_by_direction(foundation_signals, 'LONG'):.1f}% | SHORT WR: {self._calculate_wr_by_direction(foundation_signals, 'SHORT'):.1f}% | P&L: ${foundation_stats['total_pnl']:+,.2f}")
        report.append("")
        # Total stale timeouts = flagged TIMEOUT signals + STALE_TIMEOUT status
        total_stale_all = stale_timeout_count + total_stale_status  # 151 + 1 = 152
        report.append(f"Total Signals (Foundation + New): {total_signals} (Count Win = {total_tp}; Count Loss = {total_sl}; Count TimeOut = {total_timeout}; Count Open = {total_open}; Count Rejected = {total_rejected}; Stale Timeouts Excluded = {total_stale_all})")
        report.append(f"Closed Trades (Clean Data): {closed_signals} (TP: {total_tp}, SL: {total_sl}; TimeOut Win = {timeout_wins}; TimeOut Loss = {timeout_losses})")
        report.append(f"Overall Win Rate: {overall_wr:.2f}% >> [ (count TP + Count TimeOut Win) / (Closed Trades) ] = [ ({total_tp}+{timeout_wins}) / {closed_signals} ]")
        report.append(f"Total P&L (Clean Data): ${total_pnl:+.2f} (Avg P&L per Signal = ${avg_pnl_per_signal:+.2f}; Avg P&L per Closed Trade = ${avg_pnl_per_trade:+.2f})")
        report.append(f"Total P&L TP = ${pnl_tp:+.2f}; Total P&L SL = ${pnl_sl:+.2f}; Total P&L TIMEOUT Win = ${pnl_timeout_win:+.2f}; Total P&L TIMEOUT Loss = ${pnl_timeout_loss:+.2f}")
        report.append(f"Avg P&L TP per Count TP = ${avg_pnl_per_tp:+.2f}; Avg P&L SL per Count SL = ${avg_pnl_per_sl:+.2f}; Avg P&L TIMEOUT Win per Count TIMEOUT Win = ${avg_pnl_per_timeout_win:+.2f}; Avg P&L TIMEOUT Loss per Count TIMEOUT Loss = ${avg_pnl_per_timeout_loss:+.2f}")
        report.append(f"Avg TP Duration (Clean): {avg_tp_duration_summary} | Avg SL Duration (Clean): {avg_sl_duration_summary}")
        report.append(f"Max TIMEOUT Window (Designed): 15min={timeout_window_15min} | 30min={timeout_window_30min} | 1h={timeout_window_1h}")
        report.append(f"Max TIMEOUT Actual (Clean Data): 15min={actual_max_15min} | 30min={actual_max_30min} | 1h={actual_max_1h}")
        
        if stale_timeout_count > 0:
            report.append("")
            report.append(f"⚠️  DATA QUALITY ALERT: {stale_timeout_count} signals marked as 'STALE_TIMEOUT' (closed >150% past deadline)")
            report.append(f"EXCLUDED from all metrics to preserve backtest accuracy.")
            report.append(f"⚠️  CRITICAL: 1h timeframe shows None because ALL {sum(1 for s in self.signals if s.get('status') == 'TIMEOUT' and s.get('timeframe') == '1h')} 1h timeouts exceed 5h design limit!")
            report.append(f"These signals should be reviewed for data/trade quality issues.")
        
        report.append("")
        
        # === FIRED BY DATE SECTION (PLACED BELOW SUMMARY) ===
        report.append("=" * 132)
        report.append("🔥 FIRED BY DATE")
        report.append("=" * 132)
        
        # Add fired time range and count per date (each date on its own line)
        if date_summary_lines:
            report.extend(date_summary_lines)
        
        # === HOURLY BREAKDOWN FOR TODAY (DYNAMIC) ===
        today_gmt7 = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).strftime('%Y-%m-%d')
        signals_by_hour = defaultdict(list)
        signals_by_hour_dedup = defaultdict(set)
        signals_by_hour_tele = defaultdict(int)
        
        # For today, read directly from SIGNALS_MASTER for accurate hourly breakdown
        if os.path.exists(os.path.join("/Users/geniustarigan/.openclaw/workspace", "SIGNALS_MASTER.jsonl")):
            try:
                with open(os.path.join("/Users/geniustarigan/.openclaw/workspace", "SIGNALS_MASTER.jsonl"), 'r') as f:
                    for line in f:
                        try:
                            sig = json.loads(line)
                            fired_utc = sig.get('fired_time_utc')
                            if fired_utc:
                                dt = datetime.fromisoformat(fired_utc.replace('Z', '+00:00'))
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                gmt7 = dt.astimezone(timezone(timedelta(hours=7)))
                                gmt7_date = gmt7.strftime('%Y-%m-%d')
                                
                                # CRITICAL FIX: Compare GMT+7 dates, not UTC vs GMT+7
                                if gmt7_date == today_gmt7:
                                    hour_bucket = f"{gmt7.hour:02d}:00-{(gmt7.hour+1):02d}:00"
                                    
                                    signals_by_hour[hour_bucket].append(sig)
                                    
                                    # Track unique UUIDs per hour
                                    uuid = sig.get('signal_uuid')
                                    if uuid:
                                        signals_by_hour_dedup[hour_bucket].add(uuid)
                                    
                                    # Track telegram-sent per hour
                                    if sig.get('sent_time_utc') and sig.get('status') != 'REJECTED_NOT_SENT_TELEGRAM':
                                        signals_by_hour_tele[hour_bucket] += 1
                        except:
                            pass
            except Exception as e:
                pass
        
        if signals_by_hour:  # Show hourly breakdown only for current accumulating date
            report.append("")
            report.append(f"⏰ TODAY'S BREAKDOWN BY HOUR ({today_gmt7}):")
            current_hour = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).hour
            current_hour_bucket = f"{current_hour:02d}:00-{(current_hour+1):02d}:00"
            
            # Build list of ALL hours from 00:00 to current_hour (inclusive)
            all_hour_buckets = []
            for hour in range(current_hour + 1):  # 0 to current_hour
                hour_bucket = f"{hour:02d}:00-{(hour+1):02d}:00"
                all_hour_buckets.append(hour_bucket)
            
            for hour_bucket in all_hour_buckets:
                total_fired = len(signals_by_hour.get(hour_bucket, []))
                unique_fired = len(signals_by_hour_dedup.get(hour_bucket, set()))
                tele_sent = signals_by_hour_tele.get(hour_bucket, 0)
                
                # Add status label: current hour is accumulating, past hours are immutable
                if hour_bucket == current_hour_bucket:
                    status_label = "🔄 still accumulating"
                else:
                    status_label = "✓ IMMUTABLE"
                
                report.append(f"  {hour_bucket}: Total={total_fired} | Unique={unique_fired} | TeleSent={tele_sent} | {status_label}")
        
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
    
    def _calculate_wr_by_direction(self, signals, direction):
        """Calculate win rate for a specific direction (LONG or SHORT)"""
        dir_signals = [s for s in signals if s.get('signal_type') == direction]
        if not dir_signals:
            return 0.0
        
        wins = 0
        closed = 0
        
        for s in dir_signals:
            status = s.get('status')
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                closed += 1
                if status == 'TP_HIT':
                    wins += 1
                elif status == 'TIMEOUT':
                    pnl = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                    if pnl and pnl > 0:
                        wins += 1
        
        return (wins / closed * 100) if closed > 0 else 0.0
    
    def _analyze_signal_group(self, signals):
        """Analyze a group of signals (for SECTION 1 & 2) with FULL TRANSPARENCY
        
        Every signal is accounted for in the breakdown:
        - TP_HIT, SL_HIT, TIMEOUT, OPEN (backtest signals)
        - REJECTED_NOT_SENT_TELEGRAM (never sent to traders)
        - STALE_TIMEOUT (stale, excluded from backtest)
        
        Sum of all categories = Total loaded signals
        """
        tp = sum(1 for s in signals if s.get('status') == 'TP_HIT')
        sl = sum(1 for s in signals if s.get('status') == 'SL_HIT')
        timeout = sum(1 for s in signals if s.get('status') == 'TIMEOUT')
        open_trades = sum(1 for s in signals if s.get('status') == 'OPEN')
        rejected = sum(1 for s in signals if s.get('status') == 'REJECTED_NOT_SENT_TELEGRAM')
        stale = sum(1 for s in signals if s.get('status') == 'STALE_TIMEOUT')
        
        # Verify: sum of all categories = total signals (no hidden signals)
        total_accounted = tp + sl + timeout + open_trades + rejected + stale
        
        # RECALCULATE all P&L with Phase 1-3 logic (ignore daemon's stored pnl_usd)
        # CRITICAL: EXCLUDE STALE_TIMEOUT signals completely from all calculations
        # STALE_TIMEOUT = data quality issues, not valid for backtest metrics
        timeout_win = 0
        total_pnl = 0.0
        tp_pnl = 0.0
        sl_pnl = 0.0
        timeout_pnl = 0.0
        open_pnl = 0.0
        rejected_pnl = 0.0
        # NOTE: stale_pnl is NOT tracked - STALE_TIMEOUT excluded from all calculations
        
        for s in signals:
            status = s.get('status')
            
            # SKIP STALE_TIMEOUT completely - do not include in any calculation
            if status == 'STALE_TIMEOUT':
                continue
            
            pnl_calc = self._calculate_pnl_usd(
                s.get('entry_price'),
                s.get('actual_exit_price'),
                s.get('signal_type')
            )
            pnl_val = pnl_calc if pnl_calc is not None else 0.0
            
            total_pnl += pnl_val
            
            # Track P&L for each status group explicitly (STALE excluded)
            if status == 'TP_HIT':
                tp_pnl += pnl_val
            elif status == 'SL_HIT':
                sl_pnl += pnl_val
            elif status == 'TIMEOUT':
                timeout_pnl += pnl_val
                if pnl_val > 0:
                    timeout_win += 1
            elif status == 'OPEN':
                open_pnl += pnl_val
            elif status == 'REJECTED_NOT_SENT_TELEGRAM':
                rejected_pnl += pnl_val
        
        timeout_loss = timeout - timeout_win
        closed = tp + sl + timeout
        
        avg_pnl_signal = total_pnl / len(signals) if signals else 0
        avg_pnl_closed = total_pnl / closed if closed > 0 else 0
        avg_tp_pnl = tp_pnl / tp if tp > 0 else 0
        avg_sl_pnl = sl_pnl / sl if sl > 0 else 0
        
        wins = tp + timeout_win
        wr = (wins / closed * 100) if closed > 0 else 0
        
        return {
            'total': len(signals),
            'total_accounted': total_accounted,  # Must equal total
            'tp': tp,
            'sl': sl,
            'timeout': timeout,
            'open': open_trades,
            'rejected': rejected,  # Not sent to traders (excluded from metrics)
            'stale': stale,  # Data quality issues (EXCLUDED from all calculations)
            'closed': closed,
            'timeout_win': timeout_win,
            'timeout_loss': timeout_loss,
            'total_pnl': total_pnl,  # EXCLUDES STALE_TIMEOUT
            'avg_pnl_signal': avg_pnl_signal,
            'avg_pnl_closed': avg_pnl_closed,
            'tp_pnl': tp_pnl,
            'sl_pnl': sl_pnl,
            'timeout_pnl': timeout_pnl,
            'open_pnl': open_pnl,
            'rejected_pnl': rejected_pnl,  # Included in total_pnl (never sent to traders)
            # NOTE: stale_pnl NOT tracked - STALE_TIMEOUT completely excluded from calculations
            'avg_tp_pnl': avg_tp_pnl,
            'avg_sl_pnl': avg_sl_pnl,
            'wr': wr,
            'wins': wins,
        }
    
    def _aggregate_by(self, dimension):
        """Aggregate statistics by dimension (all lowercase field names)
        
        CRITICAL: Exclude both STALE_TIMEOUT and REJECTED_NOT_SENT_TELEGRAM from all aggregates
        Only valid backtest signals (TP_HIT, SL_HIT, TIMEOUT, OPEN) are aggregated
        """
        stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout_win': 0, 'timeout_loss': 0, 'pnl': 0.0})
        
        for signal in self.signals:
            status = signal.get('status', 'OPEN')
            
            # SKIP invalid/non-backtest signals - exclude from all aggregates
            if status == 'STALE_TIMEOUT':
                continue  # Data quality issue - completely excluded
            if status == 'REJECTED_NOT_SENT_TELEGRAM':
                continue  # Never sent to traders - excluded from all aggregates
            
            # Get the key value for this dimension
            key = signal.get(dimension, 'N/A')
            
            stats[key]['count'] += 1
            
            # Status already extracted above for skip checks - reuse it
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
            
            # RECALCULATE P&L using notional position of $1000 ($100 margin × 10x leverage)
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
        """Aggregate statistics by multiple dimensions (tuple of field names)
        
        CRITICAL: Exclude both STALE_TIMEOUT and REJECTED_NOT_SENT_TELEGRAM from all aggregates
        Only valid backtest signals (TP_HIT, SL_HIT, TIMEOUT, OPEN) are aggregated
        """
        stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout_win': 0, 'timeout_loss': 0, 'pnl': 0.0})
        
        for signal in self.signals:
            status = signal.get('status', 'OPEN')
            
            # SKIP invalid/non-backtest signals - exclude from all aggregates
            if status == 'STALE_TIMEOUT':
                continue  # Data quality issue - completely excluded
            if status == 'REJECTED_NOT_SENT_TELEGRAM':
                continue  # Never sent to traders - excluded from all aggregates
            
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
            
            # Status already extracted above for skip checks - reuse it
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
            
            # RECALCULATE P&L using notional position of $1000 ($100 margin × 10x leverage)
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
        """Aggregate by 4D dimensions PLUS symbol_group (5D)
        
        CRITICAL: Exclude both STALE_TIMEOUT and REJECTED_NOT_SENT_TELEGRAM from all aggregates
        Only valid backtest signals (TP_HIT, SL_HIT, TIMEOUT, OPEN) are aggregated
        """
        stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout_win': 0, 'timeout_loss': 0, 'pnl': 0.0})
        
        for signal in self.signals:
            status = signal.get('status', 'OPEN')
            
            # SKIP invalid/non-backtest signals - exclude from all aggregates
            if status == 'STALE_TIMEOUT':
                continue  # Data quality issue - completely excluded
            if status == 'REJECTED_NOT_SENT_TELEGRAM':
                continue  # Never sent to traders - excluded from all aggregates
            
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
            
            # Status already extracted above for skip checks - reuse it
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
            
            # RECALCULATE P&L using notional position of $1000 ($100 margin × 10x leverage)
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
