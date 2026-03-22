"""
Signal Sent Tracker for PEC (Position Entry Confidence)

CRITICAL: Track ONLY signals that actually fired to Telegram
- signals_fired.jsonl = ALL signals (generated, may or may not reach Telegram)
- SENT_SIGNALS.jsonl = ONLY Telegram-delivered signals (for PEC to track)
- PEC uses SENT_SIGNALS.jsonl ONLY
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class SignalSentTracker:
    """Track signals that were actually sent to Telegram"""
    
    def __init__(self, sent_signals_path: str = "SENT_SIGNALS.jsonl"):
        self.sent_signals_path = sent_signals_path
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create file if doesn't exist (with parent directory creation)"""
        if not os.path.exists(self.sent_signals_path):
            try:
                # Create parent directory if needed
                parent_dir = os.path.dirname(self.sent_signals_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                
                # Create empty file
                with open(self.sent_signals_path, 'w') as f:
                    pass  # Create empty file
                
                print(f"[INIT] Created SENT_SIGNALS.jsonl at {self.sent_signals_path}", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to create SENT_SIGNALS.jsonl: {e}", flush=True)
                raise
    
    def log_sent_signal(self, 
                       signal_uuid: str,
                       symbol: str,
                       timeframe: str,
                       signal_type: str,
                       entry_price: float,
                       tp_target: float,
                       sl_target: float,
                       tp_pct: float,
                       sl_pct: float,
                       achieved_rr: float,
                       score: int,
                       max_score: int,
                       confidence: float,
                       route: str,
                       regime: str,
                       telegram_msg_id: str,
                       fired_time_utc: str,
                       passed_filters: list = None,
                       failed_filters: list = None,
                       passed_filter_count: int = 0,
                       failed_filter_count: int = 0) -> bool:
        """
        Log a signal that was SENT to Telegram
        
        Returns: True if logged successfully, False otherwise
        """
        import sys
        try:
            sent_record = {
                "uuid": signal_uuid,
                "symbol": symbol,
                "timeframe": timeframe,
                "signal_type": signal_type,
                "entry_price": float(entry_price),
                "tp_target": float(tp_target),
                "sl_target": float(sl_target),
                "tp_pct": float(tp_pct),
                "sl_pct": float(sl_pct),
                "achieved_rr": float(achieved_rr),
                "score": int(score),
                "max_score": int(max_score),
                "confidence": float(confidence),
                "route": str(route) if route else "NONE",
                "regime": str(regime) if regime else "UNKNOWN",
                "telegram_msg_id": str(telegram_msg_id),
                "fired_time_utc": str(fired_time_utc),
                "sent_time_utc": datetime.utcnow().isoformat(),
                "passed_filters": passed_filters or [],
                "failed_filters": failed_filters or [],
                "passed_filter_count": int(passed_filter_count),
                "failed_filter_count": int(failed_filter_count),
                "status": "OPEN",  # OPEN, TP_HIT, SL_HIT, TIMEOUT
                "closed_at": None,
                "actual_exit_price": None,
                "pnl_usd": None,
                "pnl_pct": None,
                "signal_origin": "NEW"  # Auto-tagged: Daemon fires NEW signals only. FOUNDATION signals are pre-tagged in base file.
            }
            
            with open(self.sent_signals_path, 'a') as f:
                f.write(json.dumps(sent_record) + '\n')
                f.flush()  # Force write to disk immediately
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[✅ PEC SENT] {symbol:15} {timeframe:6} to SENT_SIGNALS.jsonl | {now}", flush=True)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to log sent signal: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    def update_signal_execution(self, 
                               signal_uuid: str,
                               status: str,  # TP_HIT, SL_HIT, TIMEOUT
                               exit_price: float,
                               pnl_usd: float,
                               pnl_pct: float) -> bool:
        """
        Update signal with execution results
        status: TP_HIT, SL_HIT, TIMEOUT
        """
        try:
            records = []
            found = False
            
            # Read all records
            if os.path.exists(self.sent_signals_path):
                with open(self.sent_signals_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if record.get('uuid') == signal_uuid:
                                record['status'] = status
                                record['actual_exit_price'] = float(exit_price)
                                record['pnl_usd'] = float(pnl_usd)
                                record['pnl_pct'] = float(pnl_pct)
                                record['closed_at'] = datetime.utcnow().isoformat()
                                found = True
                            records.append(record)
            
            if found:
                # Write back updated records
                with open(self.sent_signals_path, 'w') as f:
                    for record in records:
                        f.write(json.dumps(record) + '\n')
                return True
            else:
                print(f"[WARN] Signal {signal_uuid} not found in SENT_SIGNALS", flush=True)
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to update signal execution: {e}", flush=True)
            return False
    
    def get_open_signals(self) -> list:
        """Get all OPEN signals (not yet TP/SL/TIMEOUT)"""
        open_signals = []
        try:
            if os.path.exists(self.sent_signals_path):
                with open(self.sent_signals_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if record.get('status') == 'OPEN':
                                open_signals.append(record)
        except Exception as e:
            print(f"[ERROR] Failed to read open signals: {e}", flush=True)
        
        return open_signals
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        try:
            total = 0
            open_count = 0
            tp_hit = 0
            sl_hit = 0
            timeout = 0
            total_pnl = 0.0
            winning_trades = 0
            losing_trades = 0
            
            if os.path.exists(self.sent_signals_path):
                with open(self.sent_signals_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            total += 1
                            status = record.get('status', 'OPEN')
                            
                            if status == 'OPEN':
                                open_count += 1
                            elif status == 'TP_HIT':
                                tp_hit += 1
                            elif status == 'SL_HIT':
                                sl_hit += 1
                            elif status == 'TIMEOUT':
                                timeout += 1
                            
                            pnl = record.get('pnl_usd', 0)
                            if pnl:
                                total_pnl += pnl
                                if pnl > 0:
                                    winning_trades += 1
                                else:
                                    losing_trades += 1
            
            win_rate = (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0
            
            return {
                "total_sent": total,
                "open": open_count,
                "tp_hit": tp_hit,
                "sl_hit": sl_hit,
                "timeout": timeout,
                "closed": tp_hit + sl_hit + timeout,
                "total_pnl_usd": round(total_pnl, 2),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate_pct": round(win_rate, 1)
            }
        except Exception as e:
            print(f"[ERROR] Failed to calculate stats: {e}", flush=True)
            return {}


def get_signal_sent_tracker(path: str = "SENT_SIGNALS.jsonl") -> SignalSentTracker:
    """Factory function to get tracker instance"""
    return SignalSentTracker(path)
