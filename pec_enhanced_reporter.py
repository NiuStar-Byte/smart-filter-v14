#!/usr/bin/env python3
"""
PEC Enhanced Reporter - Multi-Dimensional Signal Tracking
Breaks down PEC results by: TimeFrame, Direction, Confidence, Regime, Route
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
            return
        
        with open(self.sent_signals_file, 'r') as f:
            for line in f:
                try:
                    signal = json.loads(line.strip())
                    self.signals.append(signal)
                except json.JSONDecodeError:
                    continue
    
    def get_gmt7_time(self, utc_time_str):
        """Convert UTC time string to GMT+7 (UTC+7). Display format: HH:MM:SS"""
        if not utc_time_str:
            return "N/A"
        try:
            # Parse UTC time string (ISO format)
            # Example: "2026-02-27T06:33:00" or "2026-02-27T06:33:00.123456"
            if utc_time_str.endswith('Z'):
                utc_time_str = utc_time_str[:-1] + '+00:00'
            
            # Parse to datetime
            dt_utc = datetime.fromisoformat(utc_time_str)
            
            # If naive (no timezone), assume UTC
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            
            # Convert to GMT+7 by adding 7 hours
            gmt7_tz = timezone(timedelta(hours=7))
            dt_gmt7 = dt_utc.astimezone(gmt7_tz)
            
            # Return only HH:MM:SS in GMT+7
            return dt_gmt7.strftime("%H:%M:%S")
        except Exception as e:
            return "N/A"
    
    def get_confidence_group(self, confidence):
        """Categorize confidence into groups"""
        if confidence >= 76:
            return "HIGH (>=76%)"
        elif confidence >= 51:
            return "MID (51-75%)"
        else:
            return "LOW (<=50%)"
    
    def calculate_stats(self, signals_list):
        """Calculate stats for a list of signals"""
        if not signals_list:
            return {
                'total': 0, 'open': 0, 'tp': 0, 'sl': 0, 'timeout': 0,
                'wr': 0, 'pnl': 0, 'avg_pnl': 0
            }
        
        total = len(signals_list)
        open_count = sum(1 for s in signals_list if s.get('status') == 'OPEN')
        tp_count = sum(1 for s in signals_list if s.get('status') == 'TP_HIT')
        sl_count = sum(1 for s in signals_list if s.get('status') == 'SL_HIT')
        timeout_count = sum(1 for s in signals_list if s.get('status') == 'TIMEOUT')
        
        closed = tp_count + sl_count + timeout_count
        wr = (tp_count / closed * 100) if closed > 0 else 0
        
        pnl_vals = [s.get('pnl_usd', 0) for s in signals_list if s.get('pnl_usd') is not None]
        total_pnl = sum(pnl_vals)
        avg_pnl = total_pnl / len(pnl_vals) if pnl_vals else 0
        
        return {
            'total': total,
            'open': open_count,
            'tp': tp_count,
            'sl': sl_count,
            'timeout': timeout_count,
            'closed': closed,
            'wr': wr,
            'pnl': total_pnl,
            'avg_pnl': avg_pnl
        }
    
    def format_stat_row(self, label, stats, indent=0):
        """Format a single stat row"""
        prefix = "  " * indent
        wr_str = f"{stats['wr']:.1f}%" if stats['closed'] > 0 else "N/A"
        pnl_str = f"${stats['pnl']:+.4f}" if stats['pnl'] else "$0.0000"
        
        return (f"{prefix}{label:<35} | "
                f"T:{stats['total']:<2} O:{stats['open']:<2} "
                f"✅:{stats['tp']:<2} ❌:{stats['sl']:<2} "
                f"⏱️:{stats['timeout']:<2} | "
                f"WR: {wr_str:<6} | PnL: {pnl_str}")
    
    def generate_report(self):
        """Generate comprehensive multi-dimensional report"""
        report = []
        report.append("=" * 180)
        report.append(f"PEC ENHANCED REPORT - {datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
        report.append("=" * 180)
        report.append("")
        
        # Overall stats
        overall_stats = self.calculate_stats(self.signals)
        report.append(f"📊 OVERALL METRICS:")
        report.append(self.format_stat_row("ALL SIGNALS", overall_stats))
        report.append("")
        
        # BY TIMEFRAME
        report.append("━" * 180)
        report.append("📈 BY TIMEFRAME:")
        report.append("━" * 180)
        
        for tf in ['15min', '30min', '1h']:
            tf_signals = [s for s in self.signals if s.get('timeframe') == tf]
            if tf_signals:
                stats = self.calculate_stats(tf_signals)
                report.append(self.format_stat_row(f"{tf}", stats, indent=0))
        report.append("")
        
        # BY DIRECTION
        report.append("━" * 180)
        report.append("📍 BY DIRECTION:")
        report.append("━" * 180)
        
        for direction in ['LONG', 'SHORT']:
            dir_signals = [s for s in self.signals if s.get('signal_type') == direction]
            if dir_signals:
                stats = self.calculate_stats(dir_signals)
                report.append(self.format_stat_row(f"{direction}", stats, indent=0))
        report.append("")
        
        # BY TIMEFRAME + DIRECTION + CONFIDENCE (3-dimensional)
        report.append("━" * 180)
        report.append("🎯 BY TIMEFRAME × DIRECTION × CONFIDENCE:")
        report.append("━" * 180)
        
        for tf in ['15min', '30min', '1h']:
            tf_signals = [s for s in self.signals if s.get('timeframe') == tf]
            if tf_signals:
                report.append(f"{tf}:")
                for direction in ['LONG', 'SHORT']:
                    dir_signals = [s for s in tf_signals if s.get('signal_type') == direction]
                    if dir_signals:
                        report.append(f"  {direction}:")
                        for conf_name, conf_min, conf_max in [("High", 76, 100), ("Medium", 51, 75), ("Low", 0, 50)]:
                            filtered = [s for s in dir_signals 
                                       if conf_min <= s.get('confidence', 0) <= conf_max]
                            if filtered:
                                stats = self.calculate_stats(filtered)
                                report.append(self.format_stat_row(f"    {conf_name}", stats, indent=2))
                report.append("")
        
        # BY DIRECTION + MARKET REGIME + ROUTE (3-dimensional)
        report.append("━" * 180)
        report.append("⚔️  BY DIRECTION × MARKET REGIME × ROUTE:")
        report.append("━" * 180)
        
        directions = ['LONG', 'SHORT']
        regimes = sorted(set(s.get('regime') for s in self.signals if s.get('regime')))
        
        for direction in directions:
            dir_signals = [s for s in self.signals if s.get('signal_type') == direction]
            if dir_signals:
                report.append(f"{direction}:")
                for regime in regimes:
                    regime_signals = [s for s in dir_signals if s.get('regime') == regime]
                    if regime_signals:
                        report.append(f"  {regime}:")
                        for route in sorted(set(s.get('route') for s in regime_signals if s.get('route'))):
                            filtered = [s for s in regime_signals if s.get('route') == route]
                            if filtered:
                                stats = self.calculate_stats(filtered)
                                report.append(self.format_stat_row(f"    {route}", stats, indent=3))
                report.append("")
        
        # BY CONFIDENCE GROUP
        report.append("━" * 180)
        report.append("💪 BY CONFIDENCE LEVEL:")
        report.append("━" * 180)
        
        for conf_group in ["HIGH (>=76%)", "MID (51-75%)", "LOW (<=50%)"]:
            conf_signals = [s for s in self.signals 
                           if self.get_confidence_group(s.get('confidence', 0)) == conf_group]
            if conf_signals:
                stats = self.calculate_stats(conf_signals)
                report.append(self.format_stat_row(f"{conf_group}", stats, indent=0))
        report.append("")
        
        # BY REGIME
        report.append("━" * 180)
        report.append("🌊 BY MARKET REGIME:")
        report.append("━" * 180)
        
        regimes = set(s.get('regime') for s in self.signals if s.get('regime'))
        for regime in sorted(regimes):
            regime_signals = [s for s in self.signals if s.get('regime') == regime]
            if regime_signals:
                stats = self.calculate_stats(regime_signals)
                report.append(self.format_stat_row(f"{regime}", stats, indent=0))
        report.append("")
        
        # BY ROUTE
        report.append("━" * 180)
        report.append("🛣️  BY SIGNAL ROUTE:")
        report.append("━" * 180)
        
        routes = set(s.get('route') for s in self.signals if s.get('route'))
        for route in sorted(routes):
            route_signals = [s for s in self.signals if s.get('route') == route]
            if route_signals:
                stats = self.calculate_stats(route_signals)
                report.append(self.format_stat_row(f"{route}", stats, indent=0))
        report.append("")
        
        # DETAILED SIGNAL LIST
        report.append("━" * 180)
        report.append("📋 DETAILED SIGNAL LIST:")
        report.append("━" * 180)
        report.append(f"{'Symbol':<12} {'TF':<8} {'Dir':<6} {'Conf':<8} {'Regime':<6} "
                     f"{'Status':<10} {'Entry':<12} {'PnL':<10} {'Time GMT+7':<12}")
        report.append("─" * 180)
        
        for signal in sorted(self.signals, key=lambda s: s.get('fired_time_utc', ''), reverse=True):
            symbol = signal.get('symbol', 'N/A')[:11]
            tf = signal.get('timeframe', 'N/A')[:7]
            direction = signal.get('signal_type', 'N/A')[:5]
            confidence = f"{signal.get('confidence', 0):.0f}%"
            regime = signal.get('regime', 'N/A')[:5]
            status = signal.get('status', 'OPEN')[:9]
            entry = f"{signal.get('entry_price', 0):.6f}"[:11]
            
            pnl_usd = signal.get('pnl_usd')
            if pnl_usd is not None:
                pnl_str = f"${pnl_usd:+.4f}"
            else:
                pnl_str = "OPEN"
            pnl_str = pnl_str[:11]
            
            time_gmt7 = self.get_gmt7_time(signal.get('fired_time_utc'))
            
            report.append(f"{symbol:<12} {tf:<8} {direction:<6} {confidence:<8} {regime:<6} "
                         f"{status:<10} {entry:<12} {pnl_str:<10} {time_gmt7:<12}")
        
        report.append("")
        report.append("=" * 180)
        
        return "\n".join(report)
    
    def print_report(self):
        """Print the report to console"""
        print(self.generate_report())
    
    def save_report(self, filename="PEC_ENHANCED_REPORT.txt"):
        """Save report to file"""
        with open(filename, 'w') as f:
            f.write(self.generate_report())
        print(f"✅ Report saved to {filename}")

if __name__ == "__main__":
    reporter = PECEnhancedReporter()
    reporter.print_report()
    reporter.save_report()
