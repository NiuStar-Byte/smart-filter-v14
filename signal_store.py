#!/usr/bin/env python3
"""
signal_store.py (REVISED 2026-02-23: Hybrid UUID+Timestamp)

Permanent signal storage for Smart Filter.
Stores all fired signals to JSONL format with complete metadata.

HYBRID DEDUP STRATEGY:
- UUID: includes timestamp for unique identification per occurrence
- Dedup Fingerprint: excludes timestamp for duplicate blocking within 120s
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime, timezone
import pandas as pd

class SignalStore:
    """Manages permanent signal storage with hybrid dedup."""
    
    def __init__(self, jsonl_path: str = "signals_fired.jsonl"):
        self.path = jsonl_path
        self._ensure_file_exists()
        self._last_signal_per_fingerprint = {}
    
    def _ensure_file_exists(self):
        """Create JSONL file if it doesn't exist."""
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                pass
            print(f"[SignalStore] Created new signals file: {self.path}", flush=True)
    
    def _get_signal_fingerprint(self, signal_dict: Dict) -> str:
        """
        Create deterministic fingerprint for DEDUPLICATION (NO timestamp).
        SAME signal = SAME fingerprint (regardless of time)
        """
        try:
            symbol = signal_dict.get('symbol', '').upper()
            timeframe = signal_dict.get('timeframe', '').lower()
            entry_price = float(signal_dict.get('entry_price', 0))
            signal_type = str(signal_dict.get('signal_type', '')).upper()
            score = int(signal_dict.get('score', 0))
            
            price_rounded = round(entry_price, 8)
            fingerprint = f"{symbol}_{timeframe}_{price_rounded:.8f}_{signal_type}_{score}"
            return fingerprint
        
        except Exception as e:
            print(f"[SignalStore] Fingerprint error: {e}", flush=True)
            return ""
    
    def _check_duplicate(self, signal_dict: Dict) -> bool:
        """
        Check if signal is duplicate using fingerprint.
        If SAME fingerprint fires within 120 seconds = BLOCK
        Returns True if should REJECT (is duplicate).
        """
        try:
            fingerprint = self._get_signal_fingerprint(signal_dict)
            if not fingerprint:
                return False
            
            fired_time_str = signal_dict.get('fired_time_utc')
            symbol = signal_dict.get('symbol')
            timeframe = signal_dict.get('timeframe')
            entry_price = signal_dict.get('entry_price')
            
            if fingerprint in self._last_signal_per_fingerprint:
                last = self._last_signal_per_fingerprint[fingerprint]
                try:
                    last_time = pd.Timestamp(last['time'], tz='UTC')
                    curr_time = pd.Timestamp(fired_time_str, tz='UTC')
                    time_diff = (curr_time - last_time).total_seconds()
                except Exception:
                    time_diff = 200
                
                if time_diff < 120:
                    print(f"[SignalStore] DUPLICATE BLOCKED: {symbol} {timeframe} @ {entry_price} (fired {time_diff:.1f}s ago)", flush=True)
                    return True
            
            self._last_signal_per_fingerprint[fingerprint] = {'time': fired_time_str}
            return False
        
        except Exception as e:
            print(f"[SignalStore] Dedup check error: {e}", flush=True)
            return False
    
    def append_signal(self, signal_dict: Dict) -> Optional[str]:
        """Append signal to JSONL. Returns: signal_uuid if successful, None if duplicate rejected."""
        try:
            required = ['uuid', 'symbol', 'timeframe', 'signal_type', 'fired_time_utc', 'entry_price', 'tp_target', 'sl_target', 'achieved_rr', 'score', 'confidence']
            
            for field in required:
                if field not in signal_dict or signal_dict[field] is None:
                    print(f"[SignalStore] WARNING: Missing required field '{field}'", flush=True)
                    return None
            
            # CHECK FOR DUPLICATES BEFORE STORING
            if self._check_duplicate(signal_dict):
                return None
            
            signal_dict['stored_at_utc'] = datetime.now(timezone.utc).isoformat()
            signal_dict['version'] = '1.1'
            
            with open(self.path, 'a') as f:
                json.dump(signal_dict, f)
                f.write('\n')
            
            print(f"[SignalStore] Signal stored: {signal_dict['uuid'][:8]}...", flush=True)
            return signal_dict['uuid']
        
        except Exception as e:
            print(f"[SignalStore] ERROR appending signal: {e}", flush=True)
            return None
    
    def load_all_signals(self) -> List[Dict]:
        """Load all signals from JSONL file."""
        return self.load_signals()
    
    def load_signals(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                    symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None) -> List[Dict]:
        """Load signals within date range and optional filters."""
        try:
            signals = []
            
            if not os.path.exists(self.path):
                print(f"[SignalStore] File not found: {self.path}", flush=True)
                return signals
            
            start_ts = None
            end_ts = None
            if start_date:
                start_ts = pd.Timestamp(start_date, tz='UTC')
            if end_date:
                end_ts = pd.Timestamp(end_date, tz='UTC')
            
            with open(self.path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        signal = json.loads(line)
                        
                        if start_ts or end_ts:
                            try:
                                signal_ts = pd.Timestamp(signal.get('fired_time_utc'), tz='UTC')
                                if start_ts and signal_ts < start_ts:
                                    continue
                                if end_ts and signal_ts > end_ts:
                                    continue
                            except Exception:
                                continue
                        
                        if symbols and signal.get('symbol') not in symbols:
                            continue
                        
                        if timeframes and signal.get('timeframe') not in timeframes:
                            continue
                        
                        signals.append(signal)
                    
                    except json.JSONDecodeError as e:
                        print(f"[SignalStore] WARNING: JSON parse error on line {line_num}: {e}", flush=True)
                        continue
            
            print(f"[SignalStore] Loaded {len(signals)} signals", flush=True)
            return signals
        
        except Exception as e:
            print(f"[SignalStore] ERROR loading signals: {e}", flush=True)
            return []
    
    def get_signal_by_uuid(self, uuid: str) -> Optional[Dict]:
        """Retrieve specific signal by UUID."""
        try:
            if not os.path.exists(self.path):
                return None
            
            with open(self.path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        signal = json.loads(line)
                        if signal.get('uuid') == uuid:
                            return signal
                    except json.JSONDecodeError:
                        continue
            
            return None
        
        except Exception as e:
            print(f"[SignalStore] ERROR retrieving signal {uuid}: {e}", flush=True)
            return None
    
    def get_signal_count(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> int:
        """Get count of signals in date range."""
        signals = self.load_signals(start_date=start_date, end_date=end_date)
        return len(signals)
    
    def export_to_csv(self, output_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Export signals to CSV for analysis."""
        try:
            signals = self.load_signals(start_date=start_date, end_date=end_date)
            
            if not signals:
                print(f"[SignalStore] No signals to export", flush=True)
                return False
            
            df = pd.DataFrame(signals)
            df.to_csv(output_path, index=False)
            print(f"[SignalStore] Exported {len(signals)} signals to {output_path}", flush=True)
            return True
        
        except Exception as e:
            print(f"[SignalStore] ERROR exporting to CSV: {e}", flush=True)
            return False

_store = None

def get_signal_store(jsonl_path: str = "signals_fired.jsonl") -> SignalStore:
    """Get or create global signal store instance."""
    global _store
    if _store is None:
        _store = SignalStore(jsonl_path)
    return _store
