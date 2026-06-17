#!/usr/bin/env python3
"""
signals_master_writer.py

Write signals to COMPLETE_SIGNALS.jsonl (single source of truth - ATOMIC)
Called after Telegram alert succeeds

CRITICAL:
- Path: /Users/geniustarigan/.openclaw/workspace/COMPLETE_SIGNALS.jsonl (passed from main.py)
- Write Strategy: ATOMIC (temp file + verify JSON + atomic replace)
- All fields: symbol_group, confidence_level, status, fired_time_utc, etc.
- All trackers read from this SINGLE file
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from signal_file_lock import get_signal_file_lock

class SignalsMasterWriter:
    """Atomically write signals to COMPLETE_SIGNALS.jsonl (single source of truth)
    
    All writes are atomic: temp file -> verify JSON -> atomic replace
    All fields required for trackers (symbol_group, confidence_level, status, etc.) are included
    """
    
    def __init__(self, master_path: str = "COMPLETE_SIGNALS.jsonl"):
        self.master_path = master_path
        self.file_lock = get_signal_file_lock(master_path)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create file if doesn't exist (COMPLETE_SIGNALS.jsonl only)"""
        if not os.path.exists(self.master_path):
            try:
                parent_dir = os.path.dirname(self.master_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                with open(self.master_path, 'w') as f:
                    pass
                print(f"[INIT] Created COMPLETE_SIGNALS.jsonl at {self.master_path}", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to create COMPLETE_SIGNALS.jsonl: {e}", flush=True)
    
    def write_signal(self, signal_dict: Dict[str, Any]) -> bool:
        """
        Atomically write signal to COMPLETE_SIGNALS.jsonl with canonical schema
        
        Args:
            signal_dict: Complete signal data with all fields
        
        Returns: True if written successfully, False otherwise
        """
        # DEBUG: Log IMMEDIATELY at function entry
        symbol = signal_dict.get('symbol', '?')
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"[🔴 WRITE_SIGNAL] START: {signal_dict.get('signal_type', '?')} {symbol}", flush=True)
        sys.stdout.flush()
        try:
            # Normalize to canonical schema (includes instrumentation fields for filter analysis)
            # CRITICAL: main.py creates signals with 'uuid' field, so check both
            uuid_value = signal_dict.get('uuid') or signal_dict.get('signal_uuid', '')
            master_record = {
                # Core Identity (3)
                "signal_uuid": uuid_value,
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
                
                # Signal Metadata (7)
                "score": int(signal_dict.get('score', 0)),
                "max_score": int(signal_dict.get('max_score', 19)),
                "confidence": float(signal_dict.get('confidence', 0)),
                "route": signal_dict.get('route', ''),
                "regime": signal_dict.get('regime', ''),
                "weighted_score": float(signal_dict.get('weighted_score', 0)),
                "tier": signal_dict.get('tier', 'Tier-X'),
                
                # Tier Dimensions (2) - CRITICAL FOR TIER MATCHING
                "symbol_group": signal_dict.get('symbol_group', 'UNKNOWN'),
                "confidence_level": signal_dict.get('confidence_level', 'UNKNOWN'),
                
                # Telegram Delivery (2)
                "telegram_msg_id": str(signal_dict.get('telegram_msg_id', '')),
                "sent_time_utc": signal_dict.get('sent_time_utc') or datetime.utcnow().isoformat(),
                
                # Timing (2)
                "fired_time_utc": signal_dict.get('fired_time_utc', ''),
                "fired_time_jakarta": signal_dict.get('fired_time_jakarta', ''),
                
                # Instrumentation - Filter Analysis (4)
                "passed_filters": signal_dict.get('passed_filters', []),
                "failed_filters": signal_dict.get('failed_filters', []),
                "passed_filter_count": int(signal_dict.get('passed_filter_count', 0)),
                "failed_filter_count": int(signal_dict.get('failed_filter_count', 0)),
                
                # MTF Alignment (2) - NEW: Store band for tracker visibility
                "mtf_alignment_band": signal_dict.get('mtf_alignment_band', 'unassigned'),
                "mtf_alignment_score": int(signal_dict.get('mtf_alignment_score', 0)) if signal_dict.get('mtf_alignment_score') is not None else 0,
                
                # Trade Status (5)
                "status": signal_dict.get('status', 'OPEN'),
                "closed_at": signal_dict.get('closed_at'),
                "actual_exit_price": signal_dict.get('actual_exit_price'),
                "pnl_usd": signal_dict.get('pnl_usd'),
                "pnl_pct": signal_dict.get('pnl_pct'),
                
                # Exit Timestamps (2) - When TP/SL were hit (populated by PEC executor)
                "tp_hit_time": signal_dict.get('tp_hit_time'),  # ISO timestamp when TP was hit
                "sl_hit_time": signal_dict.get('sl_hit_time'),  # ISO timestamp when SL was hit
                
                # Risk/Reward Variants (3) - Multiple representations for analysis flexibility
                "max_weights": int(signal_dict.get('max_weights', 0)),  # Max possible weighted score
                "atr_rr": float(signal_dict.get('atr_rr', 0)),  # Explicit ATR-based RR calculation
                "mtf_consensus": signal_dict.get('mtf_consensus', ''),  # LONG/SHORT/MIXED from MTF alignment
                
                # Derived Status Flags (3) - Computed from existing fields for convenience
                "timeout_win": signal_dict.get('timeout_win', False),  # Timeout closed profitably
                "timeout_loss": signal_dict.get('timeout_loss', False),  # Timeout closed at loss
                "stale_timeout": signal_dict.get('stale_timeout', False),  # Exceeded design window
                
                # Data Quality (2)
                "signal_origin": signal_dict.get('signal_origin', 'NEW_LIVE'),
                "data_quality_flag": signal_dict.get('data_quality_flag', ''),
            }
            
            # ATOMIC WRITE to COMPLETE_SIGNALS.jsonl (single source of truth)
            # Strategy: Acquire exclusive lock → write to temp file → verify JSON → atomic replace
            print(f"[WRITE_DEBUG] Acquiring write lock for {self.master_path}", flush=True)
            
            with self.file_lock.write_lock(timeout_sec=10):
                print(f"[WRITE_DEBUG] Write lock acquired, performing atomic write to {self.master_path}", flush=True)
                
                temp_path = self.master_path + '.tmp'
                try:
                    # Step 1: Read existing file to temp (preserving order)
                    existing_lines = []
                    try:
                        with open(self.master_path, 'r') as f:
                            existing_lines = f.readlines()
                    except FileNotFoundError:
                        pass
                    
                    # Step 2: Write to temp file (existing + new)
                    with open(temp_path, 'w') as f:
                        for line in existing_lines:
                            f.write(line)
                        json_line = json.dumps(master_record) + '\n'
                        f.write(json_line)
                    
                    # Step 3: Verify temp file is valid JSON
                    with open(temp_path, 'r') as f:
                        for line in f:
                            json.loads(line)  # Will raise if invalid
                    
                    # Step 4: Atomic replace (os.replace is atomic on all platforms)
                    os.replace(temp_path, self.master_path)
                    print(f"[WRITE_DEBUG] Successfully wrote {symbol} via atomic operation", flush=True)
                    print(f"[WRITE_SUCCESS] {symbol} written to {os.path.basename(self.master_path)} ✓ (ATOMIC+LOCKED)", flush=True)
                    
                except Exception as e:
                    # Clean up temp file on error
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    raise e
            return True
            
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to write to SIGNALS_MASTER.jsonl: {e}", flush=True)
            print(f"[ERROR-TRACE] {traceback.format_exc()}", flush=True)
            return False


def get_signals_master_writer(path: str = "SIGNALS_MASTER.jsonl") -> SignalsMasterWriter:
    """Factory function"""
    return SignalsMasterWriter(path)
