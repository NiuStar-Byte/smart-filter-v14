#!/usr/bin/env python3
"""
signals_master_writer.py

Write signals to SIGNALS_MASTER.jsonl (single source of truth)
Called after Telegram alert succeeds
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class SignalsMasterWriter:
    """Append sent signals to SIGNALS_MASTER.jsonl"""
    
    def __init__(self, master_path: str = "SIGNALS_MASTER.jsonl"):
        self.master_path = master_path
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create file if doesn't exist"""
        if not os.path.exists(self.master_path):
            try:
                parent_dir = os.path.dirname(self.master_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                with open(self.master_path, 'w') as f:
                    pass
                print(f"[INIT] Created SIGNALS_MASTER.jsonl at {self.master_path}", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to create SIGNALS_MASTER.jsonl: {e}", flush=True)
    
    def write_signal(self, signal_dict: Dict[str, Any]) -> bool:
        """
        Append signal to SIGNALS_MASTER.jsonl with canonical 29-field schema
        
        Args:
            signal_dict: Complete signal data with all fields
        
        Returns: True if written successfully, False otherwise
        """
        try:
            # Normalize to 29-field canonical schema
            master_record = {
                # Core Identity (3)
                "signal_uuid": signal_dict.get('signal_uuid', ''),
                "symbol": signal_dict.get('symbol', ''),
                "timeframe": signal_dict.get('timeframe', ''),
                
                # Trade Setup (6)
                "signal_type": signal_dict.get('signal_type') or signal_dict.get('direction', ''),
                "entry_price": float(signal_dict.get('entry_price', 0)),
                "tp_target": float(signal_dict.get('tp_target', 0)),
                "sl_target": float(signal_dict.get('sl_target', 0)),
                "tp_pct": float(signal_dict.get('tp_pct', 0)),
                "sl_pct": float(signal_dict.get('sl_pct', 0)),
                "achieved_rr": float(signal_dict.get('achieved_rr', 0)),
                
                # Signal Metadata (6)
                "score": int(signal_dict.get('score', 0)),
                "max_score": int(signal_dict.get('max_score', 19)),
                "confidence": float(signal_dict.get('confidence', 0)),
                "route": signal_dict.get('route', ''),
                "regime": signal_dict.get('regime', ''),
                "weighted_score": float(signal_dict.get('weighted_score', 0)),
                
                # Telegram Delivery (2)
                "telegram_msg_id": str(signal_dict.get('telegram_msg_id', '')),
                "sent_time_utc": signal_dict.get('sent_time_utc') or datetime.utcnow().isoformat(),
                
                # Timing (2)
                "fired_time_utc": signal_dict.get('fired_time_utc', ''),
                "fired_time_jakarta": signal_dict.get('fired_time_jakarta', ''),
                
                # Trade Status (5)
                "status": signal_dict.get('status', 'OPEN'),
                "closed_at": signal_dict.get('closed_at'),
                "actual_exit_price": signal_dict.get('actual_exit_price'),
                "pnl_usd": signal_dict.get('pnl_usd'),
                "pnl_pct": signal_dict.get('pnl_pct'),
                
                # Data Quality (2)
                "signal_origin": signal_dict.get('signal_origin', 'NEW_LIVE'),
                "data_quality_flag": signal_dict.get('data_quality_flag', ''),
            }
            
            # Append to SIGNALS_MASTER.jsonl
            with open(self.master_path, 'a') as f:
                f.write(json.dumps(master_record) + '\n')
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to write to SIGNALS_MASTER.jsonl: {e}", flush=True)
            return False


def get_signals_master_writer(path: str = "SIGNALS_MASTER.jsonl") -> SignalsMasterWriter:
    """Factory function"""
    return SignalsMasterWriter(path)
