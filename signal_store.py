#!/usr/bin/env python3
"""
signal_store.py

Permanent signal storage system for Smart Filter.
Stores all fired signals to JSONL format with complete metadata.
Allows reliable querying for PEC backtesting.

Format: signals_fired.jsonl (one JSON dict per line)
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime, timezone
import pandas as pd

class SignalStore:
    """
    Manages permanent signal storage and retrieval.
    Uses JSONL format for append-only, queryable storage.
    """
    
    def __init__(self, jsonl_path: str = "signals_fired.jsonl"):
        """
        Initialize signal store.
        
        Args:
            jsonl_path: Path to signals_fired.jsonl file
        """
        self.path = jsonl_path
        self._ensure_file_exists()
        self._last_signal_per_key = {}  # Simple: track last signal per symbol+timeframe
    
    def _ensure_file_exists(self):
        """Create JSONL file if it doesn't exist."""
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                pass  # Create empty file
            print(f"[SignalStore] Created new signals file: {self.path}", flush=True)
    
    def _check_duplicate(self, signal_dict: Dict) -> bool:
        """
        Simple duplicate check: same symbol+tf+price within 120 seconds = block.
        Returns True if should REJECT (is duplicate).
        """
        try:
            symbol = signal_dict.get('symbol')
            timeframe = signal_dict.get('timeframe')
            entry_price = float(signal_dict.get('entry_price', 0))
            fired_time_str = signal_dict.get('fired_time_utc')
            
            key = f"{symbol}_{timeframe}"
            
            # Check if this exact signal was sent recently
            if key in self._last_signal_per_key:
                last = self._last_signal_per_key[key]
                # Price within 0.1% and time within 120 seconds
                price_diff = abs(last['price'] - entry_price) / max(entry_price, 0.00000001)
                try:
                    last_time = pd.Timestamp(last['time'], tz='UTC')
                    curr_time = pd.Timestamp(fired_time_str, tz='UTC')
                    time_diff = (curr_time - last_time).total_seconds()
                except Exception:
                    time_diff = 200  # If time parse fails, don't block
                
                if price_diff < 0.001 and time_diff < 120:
                    print(f"[SignalStore] DUPLICATE BLOCKED: {symbol} {timeframe} @ {entry_price:.8f} (fired {time_diff:.1f}s ago)", flush=True)
                    return True
            
            # Update cache with this signal
            self._last_signal_per_key[key] = {'price': entry_price, 'time': fired_time_str}
            return False
        
        except Exception as e:
            print(f"[SignalStore] Dedup check error (allowing signal): {e}", flush=True)
            return False
    
    def append_signal(self, signal_dict: Dict) -> Optional[str]:
        """
        Append a complete signal to JSONL file.
        
        Args:
            signal_dict: Complete signal data including:
                - uuid: Unique signal ID
                - symbol: Trading pair (e.g., BTC-USDT)
                - timeframe: Market timeframe (15min, 30min, 1h)
                - signal_type: LONG or SHORT
                - fired_time_utc: When signal was fired (ISO format)
                - entry_price: Entry price at signal time
                - tp_target: Take profit target price
                - sl_target: Stop loss target price
                - tp_pct: TP as percentage of entry
                - sl_pct: SL as percentage of entry
                - achieved_rr: Risk-reward ratio
                - fib_ratio: Fibonacci level used for TP (if applicable)
                - atr_value: ATR at signal time
                - score: Filter score (0-20)
                - max_score: Maximum possible score
                - confidence: Confidence percentage (0-100)
                - route: Trade route/confirmation
                - regime: Market regime (UPTREND, DOWNTREND, RANGE, etc.)
                - passed_gatekeepers: Number of gatekeepers passed
                - max_gatekeepers: Total gatekeepers
        
        Returns:
            signal_uuid if successful, None if failed
        """
        try:
            # Validate required fields
            required = [
                'uuid', 'symbol', 'timeframe', 'signal_type',
                'fired_time_utc', 'entry_price', 'tp_target', 'sl_target',
                'achieved_rr', 'score', 'confidence'
            ]
            
            for field in required:
                if field not in signal_dict or signal_dict[field] is None:
                    print(f"[SignalStore] WARNING: Missing required field '{field}'", flush=True)
                    return None
            
            # CHECK FOR DUPLICATES BEFORE STORING
            if self._check_duplicate(signal_dict):
                return None  # Reject duplicate, don't store
            
            # Add metadata
            signal_dict['stored_at_utc'] = datetime.now(timezone.utc).isoformat()
            signal_dict['version'] = '1.0'
            
            # Append to JSONL
            with open(self.path, 'a') as f:
                json.dump(signal_dict, f)
                f.write('\n')
            
            print(f"[SignalStore] Signal stored: {signal_dict['uuid'][:8]}... ({signal_dict['symbol']} {signal_dict['timeframe']} {signal_dict['signal_type']})", flush=True)
            return signal_dict['uuid']
        
        except Exception as e:
            print(f"[SignalStore] ERROR appending signal: {e}", flush=True)
            return None
    
    def load_all_signals(self) -> List[Dict]:
        """
        Load all signals from JSONL file.
        
        Returns:
            List of all signal dicts
        """
        return self.load_signals()
    
    def load_signals(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                    symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None) -> List[Dict]:
        """
        Load signals within date range and optional filters.
        
        Args:
            start_date: ISO format start date (e.g., "2026-02-22")
            end_date: ISO format end date (e.g., "2026-02-25")
            symbols: Filter by symbols (e.g., ["BTC-USDT", "ETH-USDT"])
            timeframes: Filter by timeframes (e.g., ["15min", "30min"])
        
        Returns:
            List of signal dicts matching criteria
        """
        try:
            signals = []
            
            if not os.path.exists(self.path):
                print(f"[SignalStore] File not found: {self.path}", flush=True)
                return signals
            
            # Parse date range if provided
            start_ts = None
            end_ts = None
            if start_date:
                start_ts = pd.Timestamp(start_date, tz='UTC')
            if end_date:
                end_ts = pd.Timestamp(end_date, tz='UTC')
            
            # Read JSONL
            with open(self.path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        signal = json.loads(line)
                        
                        # Date filtering
                        if start_ts or end_ts:
                            try:
                                signal_ts = pd.Timestamp(signal.get('fired_time_utc'), tz='UTC')
                                if start_ts and signal_ts < start_ts:
                                    continue
                                if end_ts and signal_ts > end_ts:
                                    continue
                            except Exception:
                                print(f"[SignalStore] WARNING: Could not parse timestamp on line {line_num}", flush=True)
                                continue
                        
                        # Symbol filtering
                        if symbols and signal.get('symbol') not in symbols:
                            continue
                        
                        # Timeframe filtering
                        if timeframes and signal.get('timeframe') not in timeframes:
                            continue
                        
                        signals.append(signal)
                    
                    except json.JSONDecodeError as e:
                        print(f"[SignalStore] WARNING: JSON parse error on line {line_num}: {e}", flush=True)
                        continue
            
            print(f"[SignalStore] Loaded {len(signals)} signals from {self.path}", flush=True)
            return signals
        
        except Exception as e:
            print(f"[SignalStore] ERROR loading signals: {e}", flush=True)
            return []
    
    def get_signal_by_uuid(self, uuid: str) -> Optional[Dict]:
        """
        Retrieve specific signal by UUID.
        
        Args:
            uuid: Signal UUID to retrieve
        
        Returns:
            Signal dict if found, None if not found
        """
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
        """
        Get count of signals in date range.
        
        Args:
            start_date: Optional ISO date start
            end_date: Optional ISO date end
        
        Returns:
            Count of signals
        """
        signals = self.load_signals(start_date=start_date, end_date=end_date)
        return len(signals)
    
    def get_signals_by_batch(self, start_date: str, end_date: str, 
                            limit: Optional[int] = None) -> List[Dict]:
        """
        Get signals for a specific backtest batch.
        
        Args:
            start_date: Batch start date (ISO format)
            end_date: Batch end date (ISO format)
            limit: Optional limit (e.g., 50 for Batch 1)
        
        Returns:
            List of signals within date range
        """
        signals = self.load_signals(start_date=start_date, end_date=end_date)
        
        if limit:
            signals = signals[:limit]
        
        print(f"[SignalStore] Batch: {len(signals)} signals from {start_date} to {end_date}", flush=True)
        return signals
    
    def export_to_csv(self, output_path: str, start_date: Optional[str] = None, 
                     end_date: Optional[str] = None):
        """
        Export signals to CSV for analysis.
        
        Args:
            output_path: CSV output file path
            start_date: Optional start date filter
            end_date: Optional end date filter
        """
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


# Global instance
_store = None

def get_signal_store(jsonl_path: str = "signals_fired.jsonl") -> SignalStore:
    """Get or create global signal store instance."""
    global _store
    if _store is None:
        _store = SignalStore(jsonl_path)
    return _store
