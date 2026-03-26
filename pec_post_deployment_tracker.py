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
    
    def _calculate_rr_metrics(self, signals_list):
        """Calculate Risk:Reward metrics: Max, Min, Avg
        RR = |TP - Entry| / |Entry - SL|
        Works for both LONG and SHORT
        """
        try:
            rr_values = []
            for s in signals_list:
                entry = s.get('entry_price')
                # Support BOTH schemas: old (tp_price/sl_price) + new (tp_target/sl_target)
                tp = s.get('tp_price') or s.get('tp_target')
                sl = s.get('sl_price') or s.get('sl_target')
                
                if entry and tp and sl:
                    try:
                        entry_f = float(entry)
                        tp_f = float(tp)
                        sl_f = float(sl)
                        
                        # Calculate RR using absolute values (works for LONG and SHORT)
                        reward = abs(tp_f - entry_f)
                        risk = abs(entry_f - sl_f)
                        
                        if risk > 0:  # Avoid division by zero
                            rr = reward / risk
                            rr_values.append(rr)
                    except:
                        pass
            
            if not rr_values:
                return None, None, None
            
            highest_rr = max(rr_values)
            avg_rr = sum(rr_values) / len(rr_values)
            lowest_rr = min(rr_values)
            
            return round(highest_rr, 2), round(avg_rr, 2), round(lowest_rr, 2)
        except:
            return None, None, None

    def _get_symbol_group(self, symbol):
        """Map symbol to group"""
        main_blockchain = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "AVAX-USDT", "BNB-USDT", "XLM-USDT", "LINK-USDT", "POL-USDT"]
        top_alts = ["ZKJ-USDT", "ROAM-USDT", "XAUT-USDT", "SAHARA-USDT"]
        mid_alts = ["XPL-USDT", "DOT-USDT", "FUEL-USDT", "VIRTUAL-USDT", "BERA-USDT", "CROSS-USDT", "FUN-USDT", "ENA-USDT"]
        
        if symbol in main_blockchain:
            return "MAIN_BLOCKCHAIN"
        elif symbol in top_alts:
            return "TOP_ALTS"
        elif symbol in mid_alts:
            return "MID_ALTS"
        else:
            return "LOW_ALTS"
    
    def _get_confidence_level(self, confidence):
        """Convert confidence % to level"""
        if confidence >= 73:
            return "HIGH"
        elif confidence >= 66:
            return "MID"
        else:
            return "LOW"

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
        included_count = tp + sl + timeout + open_trades
        excluded_count = rejected + stale
        
        # ===== DETAILED BREAKDOWN SECTION =====
        report.append("📋 SIGNAL BREAKDOWN")
        report.append("─" * 200)
        report.append("")
        report.append(f"INCLUDED IN METRICS (TP/SL/TIMEOUT/OPEN - Counted in WR & P&L):")
        report.append(f"  • TP_HIT:     {tp:>5} signals")
        report.append(f"  • SL_HIT:     {sl:>5} signals")
        report.append(f"  • TIMEOUT:    {timeout:>5} signals")
        report.append(f"  • OPEN:       {open_trades:>5} signals (unrealized)")
        report.append(f"  ──────────────────────")
        report.append(f"  Subtotal (Counted in WR & P&L): {included_count}")
        report.append("")
        report.append(f"EXCLUDED FROM METRICS (Not counted in WR or P&L):")
        report.append(f"  • REJECTED_NOT_SENT_TELEGRAM: {rejected:>5} signals (never sent to traders)")
        report.append(f"  • STALE_TIMEOUT:              {stale:>5} signals ⚠️ DATA QUALITY ISSUE - COMPLETELY EXCLUDED")
        report.append(f"  ──────────────────────")
        report.append(f"  Subtotal (Excluded from all calculations): {excluded_count}")
        report.append("")
        report.append(f"BREAKDOWN VERIFICATION:")
        report.append(f"  Included ({included_count}) + Excluded ({excluded_count}) = Total ({total})")
        if included_count + excluded_count == total:
            report.append(f"  ✓ Verified: All {total} signals accounted for")
        else:
            report.append(f"  ✗ ERROR: Mismatch detected! {included_count} + {excluded_count} ≠ {total}")
        report.append("")
        report.append(f"Loaded {total} post-deployment signals")
        report.append("")
        report.append("─" * 200)
        report.append("")
        
        # ===== CLOSED TRADES ANALYSIS SECTION =====
        report.append("🎯 CLOSED TRADES ANALYSIS (Backtest Signals Only)")
        report.append("─" * 200)
        
        closed = tp + sl + timeout
        
        # Separate timeout into wins/losses
        timeout_wins = 0
        timeout_losses = 0
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
                    else:
                        timeout_losses += 1
        
        if closed > 0:
            wins = tp + timeout_wins
            wr = (wins / closed * 100) if closed > 0 else 0
            
            report.append(f"Closed Trades (Clean Data): {closed}")
            report.append(f"  • TP_HIT:               {tp:>5} signals (profit targets hit)")
            report.append(f"  • SL_HIT:               {sl:>5} signals (stop losses hit)")
            report.append(f"  • TIMEOUT:              {timeout:>5} signals")
            report.append(f"    - Timeout Win:       {timeout_wins:>5} (closed > entry)")
            report.append(f"    - Timeout Loss:      {timeout_losses:>5} (closed < entry)")
            report.append("")
            report.append(f"Overall Win Rate: {wr:.2f}%")
            report.append(f"Calculation: ({tp} TP + {timeout_wins} TIMEOUT_WIN) / {closed} Closed = {wins} / {closed} = {wr:.2f}%")
            
            # Add Win Rate v2 (TP & SL ONLY - excludes TIMEOUT)
            tp_sl_closed = tp + sl
            if tp_sl_closed > 0:
                wr_v2 = (tp / tp_sl_closed * 100) if tp_sl_closed > 0 else 0
                report.append(f"Win Rate (based on TP & SL ONLY): {wr_v2:.2f}%")
                report.append(f"Calculation: ({tp} TP) / ({tp} TP + {sl} SL) = {tp} / {tp_sl_closed} = {wr_v2:.2f}%")
        else:
            report.append(f"Closed Trades (Clean Data): {closed}")
            report.append(f"  (No closed trades yet - awaiting first completions)")
        
        report.append("")
        
        # ===== P&L BREAKDOWN SECTION =====
        report.append("💰 P&L BREAKDOWN")
        report.append("─" * 200)
        
        # Calculate P&L by category (INCLUDED)
        pnl_tp = 0.0
        pnl_sl = 0.0
        pnl_timeout = 0.0
        pnl_timeout_win = 0.0
        pnl_timeout_loss = 0.0
        pnl_open = 0.0
        
        for s in self.signals:
            status = s.get('status', 'OPEN')
            direction = s.get('signal_type', 'LONG')
            
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                if pnl_calc:
                    if status == 'TP_HIT':
                        pnl_tp += pnl_calc
                    elif status == 'SL_HIT':
                        pnl_sl += pnl_calc
                    elif status == 'TIMEOUT':
                        pnl_timeout += pnl_calc
                        # Separate timeout into wins and losses
                        if pnl_calc > 0:
                            pnl_timeout_win += pnl_calc
                        else:
                            pnl_timeout_loss += pnl_calc
            elif status == 'OPEN':
                pnl_open += 0.0  # Unrealized, counted as 0
        
        # Calculate P&L by category (EXCLUDED)
        pnl_rejected = 0.0
        pnl_stale = 0.0
        
        for s in self.signals:
            status = s.get('status', 'OPEN')
            direction = s.get('signal_type', 'LONG')
            
            if status == 'REJECTED_NOT_SENT_TELEGRAM':
                pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                if pnl_calc:
                    pnl_rejected += pnl_calc
            elif status == 'STALE_TIMEOUT':
                # Stale: don't calculate P&L
                pass
        
        total_pnl_included = pnl_tp + pnl_sl + pnl_timeout + pnl_open
        total_pnl_excluded = pnl_rejected + pnl_stale
        total_pnl = total_pnl_included + total_pnl_excluded
        
        report.append(f"INCLUDED IN TOTAL P&L (Counted in metrics):")
        report.append(f"  • TP_HIT:   ${pnl_tp:>+12.2f}")
        report.append(f"  • SL_HIT:   ${pnl_sl:>+12.2f}")
        report.append(f"  • TIMEOUT:  ${pnl_timeout:>+12.2f}")
        report.append(f"    - Timeout Win:  ${pnl_timeout_win:>+10.2f}")
        report.append(f"    - Timeout Loss: ${pnl_timeout_loss:>+10.2f}")
        report.append(f"  • OPEN:     ${pnl_open:>+12.2f} (unrealized)")
        report.append(f"  ──────────────────────────────")
        report.append(f"  Subtotal (Backtest P&L): ${total_pnl_included:>+12.2f}")
        report.append("")
        report.append(f"EXCLUDED FROM TOTAL P&L (Not counted in any metric):")
        report.append(f"  • REJECTED: ${pnl_rejected:>+12.2f} (never sent to traders, excluded from WR)")
        report.append(f"  • STALE:    ⚠️ NOT CALCULATED (data quality - completely excluded)")
        report.append(f"  ──────────────────────────────")
        report.append(f"  Subtotal (Excluded P&L): ${total_pnl_excluded:>+12.2f}")
        report.append("")
        report.append(f"VALIDATION:")
        report.append(f"  Included in Total P&L: ${total_pnl_included:>+12.2f}")
        report.append(f"  Total P&L Reported:    ${total_pnl:>+12.2f}")
        if abs(total_pnl - total_pnl_included) < 0.01:
            report.append(f"  ✓ Verified: P&L matches")
        report.append("")
        
        # Per-signal averages
        if closed > 0:
            avg_pnl_per_closed = total_pnl_included / closed
            report.append(f"Average P&L per Closed Trade: ${avg_pnl_per_closed:>+.2f}")
        
        if included_count > 0:
            avg_pnl_per_signal = total_pnl_included / included_count
            report.append(f"Average P&L per Signal (Included): ${avg_pnl_per_signal:>+.2f}")
        
        if tp > 0:
            avg_pnl_tp = pnl_tp / tp
            report.append(f"Average P&L per TP_HIT: ${avg_pnl_tp:>+.2f}")
        
        if sl > 0:
            avg_pnl_sl = pnl_sl / sl
            report.append(f"Average P&L per SL_HIT: ${avg_pnl_sl:>+.2f}")
        
        report.append("")
        report.append("─" * 200)
        report.append("")
        
        # ===== OVERALL RR METRICS SECTION =====
        report.append("📊 OVERALL RISK:REWARD (RR) METRICS")
        report.append("─" * 200)
        
        # Calculate RR for ALL included signals
        all_rr_values = []
        for s in self.signals:
            status = s.get('status')
            if status not in ['REJECTED_NOT_SENT_TELEGRAM', 'STALE_TIMEOUT']:
                entry = s.get('entry_price')
                tp = s.get('tp_price') or s.get('tp_target')
                sl = s.get('sl_price') or s.get('sl_target')
                
                if entry and tp and sl:
                    try:
                        entry_f = float(entry)
                        tp_f = float(tp)
                        sl_f = float(sl)
                        
                        reward = abs(tp_f - entry_f)
                        risk = abs(entry_f - sl_f)
                        
                        if risk > 0:
                            rr = reward / risk
                            all_rr_values.append(rr)
                    except:
                        pass
        
        if all_rr_values:
            highest_rr = max(all_rr_values)
            avg_rr = sum(all_rr_values) / len(all_rr_values)
            lowest_rr = min(all_rr_values)
            
            report.append(f"Highest RR: {highest_rr:.2f}")
            report.append(f"Avg RR:     {avg_rr:.2f}")
            report.append(f"Lowest RR:  {lowest_rr:.2f}")
            report.append("")
            report.append(f"Total signals analyzed: {len(all_rr_values)}")
        else:
            report.append("No RR data available")
        
        report.append("")
        report.append("─" * 200)
        report.append("")
        
        # BY TIMEFRAME (Enhanced table)
        report.append("🕐 BY TIMEFRAME")
        report.append("─" * 260)
        report.append(f"{'TF':<8} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<10} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<12} | {'Max RR':<8} | {'Avg RR':<8} | {'Min RR':<8} | {'Avg TP Dur':<12} | {'Avg SL Dur':<12}")
        report.append("─" * 260)
        
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
            
            # Calculate RR metrics for this timeframe
            max_rr, avg_rr, min_rr = self._calculate_rr_metrics(tf_signals)
            max_rr_str = f"{max_rr}" if max_rr is not None else "N/A"
            avg_rr_str = f"{avg_rr}" if avg_rr is not None else "N/A"
            min_rr_str = f"{min_rr}" if min_rr is not None else "N/A"
            
            report.append(f"{tf:<8} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {timeout_str:<10} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | {pnl_str:>10} | {max_rr_str:>6} | {avg_rr_str:>6} | {min_rr_str:>6} | {avg_tp_dur_str:<12} | {avg_sl_dur_str:<12}")
        
        report.append("─" * 260)
        report.append("")
        
        # BY DIRECTION
        report.append("📈 BY DIRECTION")
        report.append("─" * 260)
        report.append(f"{'Direction':<12} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<10} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<12} | {'Max RR':<8} | {'Avg RR':<8} | {'Min RR':<8} | {'Avg TP Dur':<12} | {'Avg SL Dur':<12}")
        report.append("─" * 260)
        
        dir_stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'timeout_win': 0, 'timeout_loss': 0, 'open': 0, 'pnl': 0.0})
        for s in self.signals:
            direction = s.get('signal_type', 'N/A')
            status = s.get('status', 'OPEN')
            dir_stats[direction]['count'] += 1
            
            if status == 'TP_HIT':
                dir_stats[direction]['tp'] += 1
            elif status == 'SL_HIT':
                dir_stats[direction]['sl'] += 1
            elif status == 'TIMEOUT':
                dir_stats[direction]['timeout'] += 1
                if s.get('actual_exit_price'):
                    pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                    if pnl_calc and pnl_calc > 0:
                        dir_stats[direction]['timeout_win'] += 1
                    elif pnl_calc and pnl_calc < 0:
                        dir_stats[direction]['timeout_loss'] += 1
            elif status == 'OPEN':
                dir_stats[direction]['open'] += 1
            
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                if pnl_calc:
                    dir_stats[direction]['pnl'] += pnl_calc
        
        for direction in sorted(dir_stats.keys()):
            stats = dir_stats[direction]
            closed = stats['tp'] + stats['sl'] + stats['timeout']
            open_count = stats['open']
            timeout_str = f"{stats['timeout_win']}W/{stats['timeout_loss']}L" if stats['timeout'] > 0 else "-"
            wr = (((stats['tp'] + stats['timeout_win']) / closed) * 100) if closed > 0 else 0
            pnl_str = f"${stats['pnl']:+.2f}"
            
            dir_signals = [s for s in self.signals if s.get('signal_type') == direction]
            max_rr, avg_rr, min_rr = self._calculate_rr_metrics(dir_signals)
            max_rr_str = f"{max_rr}" if max_rr is not None else "N/A"
            avg_rr_str = f"{avg_rr}" if avg_rr is not None else "N/A"
            min_rr_str = f"{min_rr}" if min_rr is not None else "N/A"
            
            avg_tp_dur = self._calculate_avg_duration_by_status(dir_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(dir_signals, 'SL_HIT')
            avg_tp_dur_str = self._format_duration_hm(avg_tp_dur) if avg_tp_dur else "N/A"
            avg_sl_dur_str = self._format_duration_hm(avg_sl_dur) if avg_sl_dur else "N/A"
            
            report.append(f"{direction:<12} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {timeout_str:<10} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | {pnl_str:>10} | {max_rr_str:>6} | {avg_rr_str:>6} | {min_rr_str:>6} | {avg_tp_dur_str:<12} | {avg_sl_dur_str:<12}")
        
        report.append("─" * 260)
        report.append("")
        
        # BY ROUTE
        report.append("🛣️ BY ROUTE")
        report.append("─" * 280)
        report.append(f"{'Route':<20} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<10} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<12} | {'Max RR':<8} | {'Avg RR':<8} | {'Min RR':<8} | {'Avg TP Dur':<12} | {'Avg SL Dur':<12}")
        report.append("─" * 280)
        
        route_stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'timeout_win': 0, 'timeout_loss': 0, 'open': 0, 'pnl': 0.0})
        for s in self.signals:
            route = s.get('route', 'N/A')
            status = s.get('status', 'OPEN')
            direction = s.get('signal_type', 'LONG')
            route_stats[route]['count'] += 1
            
            if status == 'TP_HIT':
                route_stats[route]['tp'] += 1
            elif status == 'SL_HIT':
                route_stats[route]['sl'] += 1
            elif status == 'TIMEOUT':
                route_stats[route]['timeout'] += 1
                if s.get('actual_exit_price'):
                    pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                    if pnl_calc and pnl_calc > 0:
                        route_stats[route]['timeout_win'] += 1
                    elif pnl_calc and pnl_calc < 0:
                        route_stats[route]['timeout_loss'] += 1
            elif status == 'OPEN':
                route_stats[route]['open'] += 1
            
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                if pnl_calc:
                    route_stats[route]['pnl'] += pnl_calc
        
        for route in sorted(route_stats.keys()):
            stats = route_stats[route]
            closed = stats['tp'] + stats['sl'] + stats['timeout']
            open_count = stats['open']
            timeout_str = f"{stats['timeout_win']}W/{stats['timeout_loss']}L" if stats['timeout'] > 0 else "-"
            wr = (((stats['tp'] + stats['timeout_win']) / closed) * 100) if closed > 0 else 0
            pnl_str = f"${stats['pnl']:+.2f}"
            
            route_signals = [s for s in self.signals if s.get('route') == route]
            max_rr, avg_rr, min_rr = self._calculate_rr_metrics(route_signals)
            max_rr_str = f"{max_rr}" if max_rr is not None else "N/A"
            avg_rr_str = f"{avg_rr}" if avg_rr is not None else "N/A"
            min_rr_str = f"{min_rr}" if min_rr is not None else "N/A"
            
            avg_tp_dur = self._calculate_avg_duration_by_status(route_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(route_signals, 'SL_HIT')
            avg_tp_dur_str = self._format_duration_hm(avg_tp_dur) if avg_tp_dur else "N/A"
            avg_sl_dur_str = self._format_duration_hm(avg_sl_dur) if avg_sl_dur else "N/A"
            
            report.append(f"{route:<20} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {timeout_str:<10} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | {pnl_str:>10} | {max_rr_str:>6} | {avg_rr_str:>6} | {min_rr_str:>6} | {avg_tp_dur_str:<12} | {avg_sl_dur_str:<12}")
        
        report.append("─" * 280)
        report.append("")
        
        # BY REGIME
        report.append("🌊 BY REGIME")
        report.append("─" * 260)
        report.append(f"{'Regime':<12} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<10} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<12} | {'Max RR':<8} | {'Avg RR':<8} | {'Min RR':<8} | {'Avg TP Dur':<12} | {'Avg SL Dur':<12}")
        report.append("─" * 260)
        
        regime_stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'timeout_win': 0, 'timeout_loss': 0, 'open': 0, 'pnl': 0.0})
        for s in self.signals:
            regime = s.get('regime', 'N/A')
            status = s.get('status', 'OPEN')
            direction = s.get('signal_type', 'LONG')
            regime_stats[regime]['count'] += 1
            
            if status == 'TP_HIT':
                regime_stats[regime]['tp'] += 1
            elif status == 'SL_HIT':
                regime_stats[regime]['sl'] += 1
            elif status == 'TIMEOUT':
                regime_stats[regime]['timeout'] += 1
                if s.get('actual_exit_price'):
                    pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                    if pnl_calc and pnl_calc > 0:
                        regime_stats[regime]['timeout_win'] += 1
                    elif pnl_calc and pnl_calc < 0:
                        regime_stats[regime]['timeout_loss'] += 1
            elif status == 'OPEN':
                regime_stats[regime]['open'] += 1
            
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                if pnl_calc:
                    regime_stats[regime]['pnl'] += pnl_calc
        
        for regime in sorted(regime_stats.keys()):
            stats = regime_stats[regime]
            closed = stats['tp'] + stats['sl'] + stats['timeout']
            open_count = stats['open']
            timeout_str = f"{stats['timeout_win']}W/{stats['timeout_loss']}L" if stats['timeout'] > 0 else "-"
            wr = (((stats['tp'] + stats['timeout_win']) / closed) * 100) if closed > 0 else 0
            pnl_str = f"${stats['pnl']:+.2f}"
            
            regime_signals = [s for s in self.signals if s.get('regime') == regime]
            max_rr, avg_rr, min_rr = self._calculate_rr_metrics(regime_signals)
            max_rr_str = f"{max_rr}" if max_rr is not None else "N/A"
            avg_rr_str = f"{avg_rr}" if avg_rr is not None else "N/A"
            min_rr_str = f"{min_rr}" if min_rr is not None else "N/A"
            
            avg_tp_dur = self._calculate_avg_duration_by_status(regime_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(regime_signals, 'SL_HIT')
            avg_tp_dur_str = self._format_duration_hm(avg_tp_dur) if avg_tp_dur else "N/A"
            avg_sl_dur_str = self._format_duration_hm(avg_sl_dur) if avg_sl_dur else "N/A"
            
            report.append(f"{regime:<12} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {timeout_str:<10} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | {pnl_str:>10} | {max_rr_str:>6} | {avg_rr_str:>6} | {min_rr_str:>6} | {avg_tp_dur_str:<12} | {avg_sl_dur_str:<12}")
        
        report.append("─" * 260)
        report.append("")
        
        # BY SYMBOL GROUP
        report.append("💰 BY SYMBOL GROUP")
        report.append("─" * 280)
        report.append(f"{'Symbol Group':<20} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<10} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<12} | {'Max RR':<8} | {'Avg RR':<8} | {'Min RR':<8} | {'Avg TP Dur':<12} | {'Avg SL Dur':<12}")
        report.append("─" * 280)
        
        symbol_group_stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'timeout_win': 0, 'timeout_loss': 0, 'open': 0, 'pnl': 0.0})
        for s in self.signals:
            symbol = s.get('symbol', 'UNKNOWN')
            group = self._get_symbol_group(symbol)
            status = s.get('status', 'OPEN')
            direction = s.get('signal_type', 'LONG')
            symbol_group_stats[group]['count'] += 1
            
            if status == 'TP_HIT':
                symbol_group_stats[group]['tp'] += 1
            elif status == 'SL_HIT':
                symbol_group_stats[group]['sl'] += 1
            elif status == 'TIMEOUT':
                symbol_group_stats[group]['timeout'] += 1
                if s.get('actual_exit_price'):
                    pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                    if pnl_calc and pnl_calc > 0:
                        symbol_group_stats[group]['timeout_win'] += 1
                    elif pnl_calc and pnl_calc < 0:
                        symbol_group_stats[group]['timeout_loss'] += 1
            elif status == 'OPEN':
                symbol_group_stats[group]['open'] += 1
            
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                if pnl_calc:
                    symbol_group_stats[group]['pnl'] += pnl_calc
        
        for group in sorted(symbol_group_stats.keys()):
            stats = symbol_group_stats[group]
            closed = stats['tp'] + stats['sl'] + stats['timeout']
            open_count = stats['open']
            timeout_str = f"{stats['timeout_win']}W/{stats['timeout_loss']}L" if stats['timeout'] > 0 else "-"
            wr = (((stats['tp'] + stats['timeout_win']) / closed) * 100) if closed > 0 else 0
            pnl_str = f"${stats['pnl']:+.2f}"
            
            group_signals = [s for s in self.signals if self._get_symbol_group(s.get('symbol', 'UNKNOWN')) == group]
            max_rr, avg_rr, min_rr = self._calculate_rr_metrics(group_signals)
            max_rr_str = f"{max_rr}" if max_rr is not None else "N/A"
            avg_rr_str = f"{avg_rr}" if avg_rr is not None else "N/A"
            min_rr_str = f"{min_rr}" if min_rr is not None else "N/A"
            
            avg_tp_dur = self._calculate_avg_duration_by_status(group_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(group_signals, 'SL_HIT')
            avg_tp_dur_str = self._format_duration_hm(avg_tp_dur) if avg_tp_dur else "N/A"
            avg_sl_dur_str = self._format_duration_hm(avg_sl_dur) if avg_sl_dur else "N/A"
            
            report.append(f"{group:<20} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {timeout_str:<10} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | {pnl_str:>10} | {max_rr_str:>6} | {avg_rr_str:>6} | {min_rr_str:>6} | {avg_tp_dur_str:<12} | {avg_sl_dur_str:<12}")
        
        report.append("─" * 280)
        report.append("")
        
        # BY CONFIDENCE LEVEL
        report.append("💡 BY CONFIDENCE LEVEL")
        report.append("─" * 280)
        report.append(f"{'Confidence':<20} | {'Total':<6} | {'TP':<4} | {'SL':<4} | {'TIMEOUT':<10} | {'Closed':<7} | {'Open':<6} | {'WR':<8} | {'P&L':<12} | {'Max RR':<8} | {'Avg RR':<8} | {'Min RR':<8} | {'Avg TP Dur':<12} | {'Avg SL Dur':<12}")
        report.append("─" * 280)
        
        conf_stats = defaultdict(lambda: {'count': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'timeout_win': 0, 'timeout_loss': 0, 'open': 0, 'pnl': 0.0})
        for s in self.signals:
            confidence = s.get('confidence', 0)
            conf_level = self._get_confidence_level(confidence)
            status = s.get('status', 'OPEN')
            direction = s.get('signal_type', 'LONG')
            conf_stats[conf_level]['count'] += 1
            
            if status == 'TP_HIT':
                conf_stats[conf_level]['tp'] += 1
            elif status == 'SL_HIT':
                conf_stats[conf_level]['sl'] += 1
            elif status == 'TIMEOUT':
                conf_stats[conf_level]['timeout'] += 1
                if s.get('actual_exit_price'):
                    pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                    if pnl_calc and pnl_calc > 0:
                        conf_stats[conf_level]['timeout_win'] += 1
                    elif pnl_calc and pnl_calc < 0:
                        conf_stats[conf_level]['timeout_loss'] += 1
            elif status == 'OPEN':
                conf_stats[conf_level]['open'] += 1
            
            if status in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                pnl_calc = self._calculate_pnl_usd(s.get('entry_price'), s.get('actual_exit_price'), direction)
                if pnl_calc:
                    conf_stats[conf_level]['pnl'] += pnl_calc
        
        for conf_level in ['HIGH', 'MID', 'LOW']:
            if conf_level not in conf_stats:
                continue
            stats = conf_stats[conf_level]
            closed = stats['tp'] + stats['sl'] + stats['timeout']
            open_count = stats['open']
            timeout_str = f"{stats['timeout_win']}W/{stats['timeout_loss']}L" if stats['timeout'] > 0 else "-"
            wr = (((stats['tp'] + stats['timeout_win']) / closed) * 100) if closed > 0 else 0
            pnl_str = f"${stats['pnl']:+.2f}"
            
            conf_signals = [s for s in self.signals if self._get_confidence_level(s.get('confidence', 0)) == conf_level]
            max_rr, avg_rr, min_rr = self._calculate_rr_metrics(conf_signals)
            max_rr_str = f"{max_rr}" if max_rr is not None else "N/A"
            avg_rr_str = f"{avg_rr}" if avg_rr is not None else "N/A"
            min_rr_str = f"{min_rr}" if min_rr is not None else "N/A"
            
            avg_tp_dur = self._calculate_avg_duration_by_status(conf_signals, 'TP_HIT')
            avg_sl_dur = self._calculate_avg_duration_by_status(conf_signals, 'SL_HIT')
            avg_tp_dur_str = self._format_duration_hm(avg_tp_dur) if avg_tp_dur else "N/A"
            avg_sl_dur_str = self._format_duration_hm(avg_sl_dur) if avg_sl_dur else "N/A"
            
            report.append(f"{conf_level:<20} | {stats['count']:<6} | {stats['tp']:<4} | {stats['sl']:<4} | {timeout_str:<10} | {closed:<7} | {open_count:<6} | {wr:>6.1f}% | {pnl_str:>10} | {max_rr_str:>6} | {avg_rr_str:>6} | {min_rr_str:>6} | {avg_tp_dur_str:<12} | {avg_sl_dur_str:<12}")
        
        report.append("─" * 280)
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
