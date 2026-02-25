#!/usr/bin/env python3
"""
signal_tracking_enhanced.py

Enhanced signal tracking for SmartFilter Project-3.
Adds Telegram send status and rejection reason tracking.
Does NOT modify existing signals_fired.jsonl - creates SENT_ONLY variant.

Functions:
1. mark_signal_sent() - Updates a signal in the store with telegram_sent=true
2. mark_signal_rejected() - Updates a signal with telegram_sent=false + reason
3. generate_sent_only_file() - Creates signals_fired_SENT_ONLY.jsonl from sent signals
4. get_rejection_stats() - Diagnostic: how many rejected and why
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional, Dict

class SignalTracker:
    """
    Enhanced tracking for signal lifecycle.
    Safe: does not modify original signals_fired.jsonl
    """
    
    def __init__(self, signals_jsonl="signals_fired.jsonl"):
        self.signals_path = Path(signals_jsonl)
        self.status_file = self.signals_path.parent / "signal_status.jsonl"
        self._ensure_status_file()
    
    def _ensure_status_file(self):
        """Create signal_status.jsonl if missing"""
        if not self.status_file.exists():
            self.status_file.touch()
            print(f"[SignalTracker] Created status file: {self.status_file}", flush=True)
    
    def mark_signal_sent(self, signal_uuid: str, telegram_message_id: Optional[str] = None):
        """
        Mark a signal as successfully sent to Telegram.
        
        Args:
            signal_uuid: UUID of the signal
            telegram_message_id: Optional Telegram message ID for tracking
        """
        status = {
            "uuid": signal_uuid,
            "telegram_sent": True,
            "rejection_reason": None,
            "marked_at_utc": datetime.now(timezone.utc).isoformat(),
            "telegram_message_id": telegram_message_id
        }
        
        try:
            with open(self.status_file, 'a') as f:
                f.write(json.dumps(status) + "\n")
            print(f"[SignalTracker] ✅ Marked sent: {signal_uuid}", flush=True)
        except Exception as e:
            print(f"[SignalTracker] ERROR marking sent: {e}", flush=True)
    
    def mark_signal_rejected(self, signal_uuid: str, rejection_reason: str):
        """
        Mark a signal as rejected (not sent to Telegram).
        
        Args:
            signal_uuid: UUID of the signal
            rejection_reason: Why was it rejected? (e.g., "RR_TOO_LOW", "DUPLICATE", "FILTERS_OK_FALSE", "COOLDOWN_BLOCKED")
        """
        status = {
            "uuid": signal_uuid,
            "telegram_sent": False,
            "rejection_reason": rejection_reason,
            "marked_at_utc": datetime.now(timezone.utc).isoformat(),
            "telegram_message_id": None
        }
        
        try:
            with open(self.status_file, 'a') as f:
                f.write(json.dumps(status) + "\n")
            print(f"[SignalTracker] ❌ Marked rejected: {signal_uuid} - {rejection_reason}", flush=True)
        except Exception as e:
            print(f"[SignalTracker] ERROR marking rejected: {e}", flush=True)
    
    def get_signal_status(self, signal_uuid: str) -> Optional[Dict]:
        """Get status of a signal"""
        if not self.status_file.exists():
            return None
        
        try:
            with open(self.status_file, 'r') as f:
                for line in f:
                    try:
                        status = json.loads(line)
                        if status.get("uuid") == signal_uuid:
                            return status
                    except:
                        pass
        except Exception as e:
            print(f"[SignalTracker] ERROR reading status: {e}", flush=True)
        
        return None
    
    def generate_sent_only_file(self, output_path: Optional[str] = None):
        """
        Generate signals_fired_SENT_ONLY.jsonl containing only sent signals.
        Safe: reads from both files, creates new file.
        
        Args:
            output_path: Where to save sent-only file (default: signals_fired_SENT_ONLY.jsonl)
        """
        if output_path is None:
            output_path = self.signals_path.parent / "signals_fired_SENT_ONLY.jsonl"
        
        # Load all signals
        signals = {}
        try:
            with open(self.signals_path, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line)
                        signals[sig.get("uuid")] = sig
                    except:
                        pass
        except Exception as e:
            print(f"[SignalTracker] ERROR loading signals: {e}", flush=True)
            return
        
        # Load all statuses
        sent_uuids = set()
        rejected_uuids = set()
        try:
            with open(self.status_file, 'r') as f:
                for line in f:
                    try:
                        status = json.loads(line)
                        uuid = status.get("uuid")
                        if status.get("telegram_sent"):
                            sent_uuids.add(uuid)
                        else:
                            rejected_uuids.add(uuid)
                    except:
                        pass
        except Exception as e:
            print(f"[SignalTracker] ERROR loading status: {e}", flush=True)
        
        # Write sent-only signals
        sent_count = 0
        try:
            with open(output_path, 'w') as f:
                for uuid, sig in signals.items():
                    if uuid in sent_uuids:
                        sig["telegram_sent"] = True
                        f.write(json.dumps(sig) + "\n")
                        sent_count += 1
        except Exception as e:
            print(f"[SignalTracker] ERROR writing sent-only file: {e}", flush=True)
            return
        
        print(f"[SignalTracker] ✅ Generated {output_path}")
        print(f"   Sent signals: {sent_count} / {len(signals)}")
        print(f"   Rejected: {len(rejected_uuids)}")
    
    def get_rejection_stats(self) -> Dict:
        """Get statistics on rejections by reason"""
        stats = defaultdict(int)
        total_rejected = 0
        
        try:
            with open(self.status_file, 'r') as f:
                for line in f:
                    try:
                        status = json.loads(line)
                        if not status.get("telegram_sent"):
                            reason = status.get("rejection_reason", "UNKNOWN")
                            stats[reason] += 1
                            total_rejected += 1
                    except:
                        pass
        except Exception as e:
            print(f"[SignalTracker] ERROR reading stats: {e}", flush=True)
            return {}
        
        return dict(stats)
    
    def get_diagnostic_report(self) -> str:
        """Generate human-readable diagnostic report"""
        # Load counts
        total_signals = 0
        signals_by_tf = defaultdict(int)
        try:
            with open(self.signals_path, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line)
                        total_signals += 1
                        tf = sig.get("timeframe", "unknown")
                        signals_by_tf[tf] += 1
                    except:
                        pass
        except:
            pass
        
        # Load status counts
        sent_count = 0
        rejection_stats = self.get_rejection_stats()
        total_rejected = sum(rejection_stats.values())
        
        try:
            with open(self.status_file, 'r') as f:
                for line in f:
                    try:
                        status = json.loads(line)
                        if status.get("telegram_sent"):
                            sent_count += 1
                    except:
                        pass
        except:
            pass
        
        # Build report
        report = "\n" + "="*70 + "\n"
        report += "SMARTFILTER SIGNAL TRACKING DIAGNOSTIC REPORT\n"
        report += "="*70 + "\n\n"
        
        report += f"Total Signals Generated: {total_signals}\n"
        report += f"By Timeframe:\n"
        for tf in sorted(signals_by_tf.keys()):
            count = signals_by_tf[tf]
            pct = 100 * count / total_signals if total_signals > 0 else 0
            report += f"   {tf:6}: {count:5} ({pct:5.1f}%)\n"
        
        report += f"\nTelegram Delivery:\n"
        report += f"   Sent:     {sent_count:5}\n"
        report += f"   Rejected: {total_rejected:5}\n"
        if total_signals > 0:
            send_rate = 100 * sent_count / total_signals
            report += f"   Send Rate: {send_rate:.1f}%\n"
        
        report += f"\nRejection Reasons:\n"
        for reason in sorted(rejection_stats.keys()):
            count = rejection_stats[reason]
            pct = 100 * count / total_rejected if total_rejected > 0 else 0
            report += f"   {reason:30}: {count:5} ({pct:5.1f}%)\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report


def get_signal_tracker(signals_jsonl="signals_fired.jsonl") -> SignalTracker:
    """Helper to get tracker instance"""
    return SignalTracker(signals_jsonl)
