#!/usr/bin/env python3
"""
signal_enrichment.py

Enrich signals with derived and calculated fields AFTER they are created.
Called when signals are updated (e.g., when they close in PEC executor).

Enrichment adds:
1. Derived status flags (timeout_win, timeout_loss, stale_timeout)
2. Exit timestamps (tp_hit_time, sl_hit_time) when applicable
3. Risk/Reward variants (max_weights, atr_rr, mtf_consensus) if available

COMPLETE_SIGNALS.jsonl becomes the single source of truth with all attributes.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from signal_file_lock import get_signal_file_lock


class SignalEnricher:
    """Enrich signals with derived and missing fields"""
    
    STALE_TIMEOUT_HOURS = 5  # Signals older than this are stale timeouts
    
    @staticmethod
    def enrich_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a signal dict with derived fields
        
        Args:
            signal: Signal dict from COMPLETE_SIGNALS.jsonl
        
        Returns:
            Enriched signal dict with all derived fields
        """
        status = signal.get('status', 'OPEN')
        pnl = float(signal.get('pnl_usd') or 0)
        fired_time_str = signal.get('fired_time_utc', '')
        closed_time_str = signal.get('closed_at')
        
        # Compute derived status flags
        timeout_win = False
        timeout_loss = False
        stale_timeout = False
        
        if status == 'TIMEOUT':
            # Timeout signals: compute win/loss based on P&L
            timeout_win = pnl > 0
            timeout_loss = pnl < 0
            
            # Check if timeout is stale (exceeded design window)
            if fired_time_str and closed_time_str:
                try:
                    fired = datetime.fromisoformat(fired_time_str.replace('Z', '+00:00'))
                    closed = datetime.fromisoformat(closed_time_str.replace('Z', '+00:00'))
                    duration = closed - fired
                    stale_timeout = duration > timedelta(hours=SignalEnricher.STALE_TIMEOUT_HOURS)
                except:
                    pass  # If parse fails, leave as False
        
        # Determine exit times (if TP or SL was hit)
        tp_hit_time = None
        sl_hit_time = None
        
        if status == 'TP_HIT' and closed_time_str:
            tp_hit_time = closed_time_str
        elif status == 'SL_HIT' and closed_time_str:
            sl_hit_time = closed_time_str
        elif status == 'TIMEOUT' and closed_time_str:
            # For timeout: if profit, assume TP direction hit; if loss, assume SL direction hit
            if pnl > 0:
                tp_hit_time = closed_time_str
            elif pnl < 0:
                sl_hit_time = closed_time_str
        
        # Enrich signal with all derived fields
        enriched = signal.copy()
        enriched['timeout_win'] = timeout_win
        enriched['timeout_loss'] = timeout_loss
        enriched['stale_timeout'] = stale_timeout
        enriched['tp_hit_time'] = tp_hit_time
        enriched['sl_hit_time'] = sl_hit_time
        
        # If not already present, set default values for optional fields
        if 'max_weights' not in enriched:
            enriched['max_weights'] = signal.get('max_weights', 0)
        if 'atr_rr' not in enriched:
            enriched['atr_rr'] = signal.get('atr_rr', 0)
        if 'mtf_consensus' not in enriched:
            enriched['mtf_consensus'] = signal.get('mtf_consensus', '')
        
        return enriched
    
    @staticmethod
    def enrich_signals_file(file_path: str, output_path: Optional[str] = None) -> int:
        """
        Enrich all signals in a file and optionally write to new file
        Uses file locking to prevent concurrent access issues.
        
        Args:
            file_path: Path to COMPLETE_SIGNALS.jsonl
            output_path: If provided, write enriched signals here (otherwise in-place)
        
        Returns:
            Count of signals enriched
        """
        enriched_count = 0
        file_lock = get_signal_file_lock(file_path)
        
        try:
            # Acquire read lock for reading signals
            with file_lock.read_lock(timeout_sec=5):
                print(f"[ENRICH] Read lock acquired, reading {file_path}", flush=True)
                signals = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                signals.append(json.loads(line))
                            except:
                                pass
            
            # Enrich each signal (not under lock, safe operation)
            enriched_signals = []
            for sig in signals:
                enriched = SignalEnricher.enrich_signal(sig)
                enriched_signals.append(enriched)
                enriched_count += 1
            
            # Write back with exclusive lock
            write_path = output_path or file_path
            if write_path == file_path:
                # Writing back to same file - use write lock
                with file_lock.write_lock(timeout_sec=10):
                    print(f"[ENRICH] Write lock acquired, writing {enriched_count} enriched signals", flush=True)
                    temp_path = file_path + '.tmp'
                    with open(temp_path, 'w') as f:
                        for sig in enriched_signals:
                            f.write(json.dumps(sig) + '\n')
                    os.replace(temp_path, file_path)
            else:
                # Writing to different file - no lock needed
                with open(write_path, 'w') as f:
                    for sig in enriched_signals:
                        f.write(json.dumps(sig) + '\n')
            
            print(f"[ENRICH] ✅ Enriched {enriched_count} signals", flush=True)
            return enriched_count
            
        except Exception as e:
            print(f"[ERROR] Signal enrichment failed: {e}", flush=True)
            return 0


def get_signal_enricher() -> SignalEnricher:
    """Factory function"""
    return SignalEnricher()
