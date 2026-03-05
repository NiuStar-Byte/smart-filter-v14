#!/usr/bin/env python3
"""
PEC IMMUTABLE REPORTER
Reads from SIGNALS_LEDGER_IMMUTABLE.jsonl (immutable source of truth)
- Past dates: LOCKED and immutable
- Today: Open for accumulation only
- No changes except signal accumulation
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

LEDGER_FILE = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_LEDGER_IMMUTABLE.jsonl"
PAST_DATES_LOCKED = ["2026-02-27", "2026-02-28", "2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"]

class ImmutablePECReporter:
    def __init__(self):
        self.signals = []
        self.load_signals()
        
    def load_signals(self):
        """Load all signals from immutable ledger"""
        if not os.path.exists(LEDGER_FILE):
            print(f"[ERROR] Ledger file not found: {LEDGER_FILE}")
            return
        
        with open(LEDGER_FILE, "r") as f:
            count = 0
            for line in f:
                try:
                    signal = json.loads(line.strip())
                    self.signals.append(signal)
                    count += 1
                except:
                    pass
        
        print(f"[INFO] Loaded {count} signals from immutable ledger")
    
    def get_utc_date(self, signal):
        """Extract UTC date from signal"""
        fired = signal.get("fired_time_utc") or signal.get("fired_time", "")
        if fired:
            return fired[:10]
        return "unknown"
    
    def get_utc_hour(self, signal):
        """Extract UTC hour from signal"""
        fired = signal.get("fired_time_utc") or signal.get("fired_time", "")
        if fired and len(fired) > 13:
            return int(fired[11:13])
        return -1
    
    def get_signal_status(self, signal):
        """Determine signal status (TP_HIT, SL_HIT, TIMEOUT_WIN, TIMEOUT_LOSS, OPEN)"""
        status = signal.get("status", "OPEN")
        actual_exit = signal.get("actual_exit_price")
        entry = signal.get("entry_price")
        
        if status == "OPEN":
            return "OPEN"
        elif status == "TP_HIT":
            return "TP_HIT"
        elif status == "SL_HIT":
            return "SL_HIT"
        elif status == "TIMEOUT":
            if actual_exit and entry:
                pnl = ((actual_exit - entry) / entry) * 100
                return "TIMEOUT_WIN" if pnl > 0 else "TIMEOUT_LOSS"
            return "TIMEOUT"
        else:
            return status
    
    def generate_report(self):
        """Generate immutable report"""
        report = []
        
        # Header
        report.append("=" * 120)
        report.append("📊 PEC IMMUTABLE REPORTER - LEDGER-BASED SIGNAL TRACKING")
        report.append("=" * 120)
        report.append(f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"Source: SIGNALS_LEDGER_IMMUTABLE.jsonl")
        report.append(f"Total Signals: {len(self.signals)}")
        report.append("")
        
        # Group by date
        by_date = defaultdict(list)
        by_date_hour = defaultdict(lambda: defaultdict(list))
        
        for signal in self.signals:
            date = self.get_utc_date(signal)
            hour = self.get_utc_hour(signal)
            by_date[date].append(signal)
            if hour >= 0:
                by_date_hour[date][hour].append(signal)
        
        # TOTAL FIRED PER DATE (IMMUTABLE for past dates)
        report.append("📅 TOTAL FIRED PER DATE")
        report.append("=" * 120)
        
        for date in sorted(by_date.keys()):
            signals_on_date = by_date[date]
            count = len(signals_on_date)
            
            times = []
            for s in signals_on_date:
                fired = s.get("fired_time_utc") or s.get("fired_time", "")
                if fired:
                    times.append(fired[11:19])
            
            earliest = min(times) if times else "N/A"
            latest = max(times) if times else "N/A"
            
            status_str = ">> immutable" if date in PAST_DATES_LOCKED else ">> OPEN (accumulating)"
            report.append(f"  {date}: {count} fired | Begin: {earliest} | Last: {latest} | {status_str}")
        
        report.append("")
        
        # HOURLY BREAKDOWN FOR TODAY (MAR-05)
        today = "2026-03-05"
        if today in by_date_hour:
            report.append("⏰ TOTAL FIRED TODAY HOURLY (UTC)")
            report.append("=" * 120)
            
            for hour in sorted(by_date_hour[today].keys()):
                signals_in_hour = by_date_hour[today][hour]
                count = len(signals_in_hour)
                
                times = []
                for s in signals_in_hour:
                    fired = s.get("fired_time_utc") or s.get("fired_time", "")
                    if fired:
                        times.append(fired[11:19])
                
                earliest = min(times) if times else "N/A"
                latest = max(times) if times else "N/A"
                
                hour_str = f"{hour:02d}:00-{hour:02d}:59"
                report.append(f"  {hour_str} UTC: {count} fired | Begin: {earliest} | Last: {latest}")
            
            report.append("")
        
        # SUMMARY STATISTICS
        report.append("📊 SUMMARY STATISTICS")
        report.append("=" * 120)
        
        tp_count = sum(1 for s in self.signals if self.get_signal_status(s) == "TP_HIT")
        sl_count = sum(1 for s in self.signals if self.get_signal_status(s) == "SL_HIT")
        timeout_win = sum(1 for s in self.signals if self.get_signal_status(s) == "TIMEOUT_WIN")
        timeout_loss = sum(1 for s in self.signals if self.get_signal_status(s) == "TIMEOUT_LOSS")
        open_count = sum(1 for s in self.signals if self.get_signal_status(s) == "OPEN")
        
        closed_count = tp_count + sl_count + timeout_win + timeout_loss
        wr = (tp_count + timeout_win) / closed_count * 100 if closed_count > 0 else 0
        
        report.append(f"Total Signals: {len(self.signals)}")
        report.append(f"  TP_HIT: {tp_count}")
        report.append(f"  SL_HIT: {sl_count}")
        report.append(f"  TIMEOUT_WIN: {timeout_win}")
        report.append(f"  TIMEOUT_LOSS: {timeout_loss}")
        report.append(f"  OPEN: {open_count}")
        report.append(f"  Closed: {closed_count}")
        report.append(f"")
        report.append(f"Win Rate (TP + TIMEOUT_WIN) / Closed: {wr:.1f}%")
        report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    reporter = ImmutablePECReporter()
    report = reporter.generate_report()
    
    # Print to console
    print(report)
    
    # Save to text file
    output_file = "/Users/geniustarigan/.openclaw/workspace/PEC_IMMUTABLE_REPORT.txt"
    with open(output_file, "w") as f:
        f.write(report)
    
    print(f"\n✓ Report saved to {output_file}")
