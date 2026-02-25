"""
Clean, informative logging for SmartFilter signal processing.
Removes verbose per-symbol checks, focuses on actual events.
"""

import sys
from datetime import datetime

class SignalLogger:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.signal_count = {"15min": 0, "30min": 0, "1h": 0}
        self.sent_count = {"15min": 0, "30min": 0, "1h": 0}
        self.rejected_count = 0
        self.start_time = datetime.now()

    def signal_generated(self, symbol: str, tf: str, bias: str, score: int, entry_price: float):
        """Log when a signal is generated (passes filters)"""
        if not self.enabled:
            return
        self.signal_count[tf] += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[SIGNAL] {symbol:15} {tf:6} {bias:6} @ {entry_price:12.2f} (Score: {score}/19) | {now}", flush=True)

    def signal_sent(self, symbol: str, tf: str, msg_id: str):
        """Log when signal reaches Telegram"""
        if not self.enabled:
            return
        self.sent_count[tf] += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[✅ SENT]  {symbol:15} {tf:6} → Telegram (ID: {msg_id[:8]}...) | {now}", flush=True)

    def signal_rejected(self, symbol: str, tf: str, reason: str):
        """Log when signal is rejected"""
        if not self.enabled:
            return
        self.rejected_count += 1
        print(f"[⛔ REJECT] {symbol:15} {tf:6} — {reason}", flush=True)

    def cycle_summary(self):
        """Print summary at end of cycle"""
        if not self.enabled:
            return
        total_generated = sum(self.signal_count.values())
        total_sent = sum(self.sent_count.values())
        elapsed = (datetime.now() - self.start_time).total_seconds()
        cycle_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*70, flush=True)
        print(f"CYCLE SUMMARY ({elapsed:.1f}s) | {cycle_time}", flush=True)
        print(f"  Generated: 15m={self.signal_count['15min']} | 30m={self.signal_count['30min']} | 1h={self.signal_count['1h']} (Total: {total_generated})", flush=True)
        print(f"  Sent:      15m={self.sent_count['15min']} | 30m={self.sent_count['30min']} | 1h={self.sent_count['1h']} (Total: {total_sent})", flush=True)
        print(f"  Rejected:  {self.rejected_count}", flush=True)
        print("="*70 + "\n", flush=True)

    def reset(self):
        """Reset counters for next cycle"""
        self.signal_count = {"15min": 0, "30min": 0, "1h": 0}
        self.sent_count = {"15min": 0, "30min": 0, "1h": 0}
        self.rejected_count = 0
        self.start_time = datetime.now()
