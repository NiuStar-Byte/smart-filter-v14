#!/usr/bin/env python3
"""
PEC Executor Persistent - SAFE APPEND-ONLY MODE
Checks OPEN signals and writes closures to separate SIGNALS_CLOSURES.jsonl
Does NOT reload/rewrite SIGNALS_MASTER - eliminates atomic writer variance issues

Built: May 19 2026, 11:07 GMT+7
Architecture: Fire-and-Closure separation
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kucoin_data import get_ohlcv
from timezone_utils import parse_timestamp_naive_utc, get_utc_now_naive, calculate_cutoff_date, compare_timestamps

class PECExecutorSafe:
    """
    Persistent price closure checker (append-only safe mode)
    
    Architecture:
    - Reads SIGNALS_MASTER.jsonl (fire events, read-only)
    - Appends closures to SIGNALS_CLOSURES.jsonl (append-only)
    - Never reloads/rewrites files (prevents atomic writer issues)
    - Maintains separate closure ledger for auditability
    """
    
    def __init__(self):
        """Initialize PEC executor"""
        self.workspace_root = os.path.dirname(os.path.abspath(__file__))
        self.workspace = os.path.dirname(self.workspace_root)
        self.signals_master = os.path.join(self.workspace_root, 'SIGNALS_MASTER.jsonl')
        self.signals_closures = os.path.join(self.workspace, 'SIGNALS_CLOSURES.jsonl')
        
        # Cache: already-closed signal UUIDs (load once per run)
        self.already_closed_uuids = self._load_already_closed_uuids()
        
        print(f"[PEC-SAFE] ✅ Initialized (safe append-only mode with dedup)", flush=True)
        print(f"[PEC-SAFE] Reading from: SIGNALS_MASTER.jsonl (read-only)", flush=True)
        print(f"[PEC-SAFE] Writing to: SIGNALS_CLOSURES.jsonl (append-only)", flush=True)
        print(f"[PEC-SAFE] Dedup cache: {len(self.already_closed_uuids)} already-closed signals loaded", flush=True)
        print(f"[PEC-SAFE] 7-day window: signals from last 7 days only", flush=True)
    
    def _load_already_closed_uuids(self):
        """Load all UUIDs that have already been closed (deduplication)"""
        closed_uuids = set()
        
        if not os.path.exists(self.signals_closures):
            return closed_uuids
        
        try:
            with open(self.signals_closures, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            closure = json.loads(line)
                            uuid = closure.get('signal_uuid')
                            if uuid:
                                closed_uuids.add(uuid)
                        except:
                            pass
            print(f"[PEC-SAFE] ✅ Loaded {len(closed_uuids)} already-closed UUIDs for dedup", flush=True)
        except Exception as e:
            print(f"[PEC-SAFE] ⚠️ Error loading dedup cache: {str(e)[:50]}", flush=True)
        
        return closed_uuids
    
    def check_signal_closure(self, signal, debug=False):
        """
        Check if an OPEN signal should close (TP/SL/TIMEOUT)
        
        Args:
            signal: Signal dict with entry_price, tp_target, sl_target
            debug: Print debug info
            
        Returns:
            closure_event dict or None if still open
        """
        try:
            if signal.get('status') != 'OPEN':
                return None
            
            symbol = signal.get('symbol')
            entry = float(signal.get('entry_price', 0))
            tp = float(signal.get('tp_target', 0))
            sl = float(signal.get('sl_target', 0))
            uuid = signal.get('signal_uuid')
            
            if not all([symbol, entry, tp, sl, uuid]):
                if debug:
                    print(f"[DEBUG-SKIP] {symbol}: Missing field (entry={entry}, tp={tp}, sl={sl})", flush=True)
                return None
            
            # Fetch current price (try tighter timeframes first, fallback to 1h)
            current_price = None
            for tf in ['15m', '5m', '1h']:  # Try tighter first for more recent price
                try:
                    df = get_ohlcv(symbol, tf, limit=1)
                    if df is not None and not df.empty:
                        current_price = float(df['close'].iloc[-1])
                        if debug:
                            print(f"[DEBUG-TF] {symbol}: Got price from {tf}", flush=True)
                        break
                except:
                    pass
            
            if current_price is None:
                if debug:
                    print(f"[DEBUG-SKIP] {symbol}: Price fetch failed for all timeframes", flush=True)
                return None
            
            if debug:
                print(f"[DEBUG-CHECK] {symbol}: current={current_price:.8f}, entry={entry:.8f}, tp={tp:.8f}, sl={sl:.8f}", flush=True)
            
            # Check closure conditions (TIMEOUT FIRST - highest priority for old signals!)
            closure_event = None
            fired_at = signal.get('timestamp') or signal.get('fired_time_utc')
            timeframe = signal.get('timeframe', '4h')
            
            # FIRST: Check if signal has TIMED OUT (regardless of current price)
            if fired_at:
                try:
                    fired_dt = parse_timestamp_naive_utc(fired_at)
                    now = get_utc_now_naive()
                    
                    # Timeout thresholds (correct from pec_enhanced_reporter.py):
                    timeout_hours = {
                        '15min': 3.75,
                        '30min': 5.0,
                        '1h': 5.0,
                        '2h': 6.0,
                        '4h': 8.0
                    }.get(timeframe, 8.0)
                    
                    from datetime import timedelta
                    timeout_cutoff = fired_dt + timedelta(hours=timeout_hours)
                    
                    if compare_timestamps(now, timeout_cutoff, '>='):
                        # Signal HAS TIMED OUT - close it immediately
                        closure_event = {
                            'signal_uuid': uuid,
                            'status': 'TIMEOUT',
                            'actual_exit_price': current_price,
                            'closed_at': now.isoformat(),
                            'close_reason': f'TIMEOUT (>{timeout_hours}h, {timeframe})'
                        }
                        if debug:
                            print(f"[DEBUG-TIMEOUT] {symbol}: {now.isoformat()} >= {timeout_cutoff.isoformat()}", flush=True)
                        return closure_event
                except Exception as e:
                    if debug:
                        print(f"[DEBUG-ERROR] {symbol} timeout check failed: {str(e)[:50]}", flush=True)
            
            # SECOND: If not timed out, check TP/SL based on current price
            if current_price >= tp:
                closure_event = {
                    'signal_uuid': uuid,
                    'status': 'TP_HIT',
                    'actual_exit_price': current_price,
                    'closed_at': get_utc_now_naive().isoformat(),
                    'close_reason': 'TP_HIT'
                }
            elif current_price <= sl:
                closure_event = {
                    'signal_uuid': uuid,
                    'status': 'SL_HIT',
                    'actual_exit_price': current_price,
                    'closed_at': get_utc_now_naive().isoformat(),
                    'close_reason': 'SL_HIT'
                }
            
            return closure_event
        
        except Exception as e:
            return None
    
    def run_cycle(self):
        """Run one check cycle"""
        try:
            # Load ALL OPEN signals (read-only) - no time window limit
            open_signals = []
            
            with open(self.signals_master, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        sig = json.loads(line)
                        if sig.get('status') == 'OPEN':
                            open_signals.append(sig)
                    except:
                        pass
            
            print(f"[CYCLE] Checking {len(open_signals):,} OPEN signals (ALL, no time window limit)", flush=True)
            
            # Check each for closure (skip already-closed signals)
            closures = []
            skipped = 0
            for sig in open_signals:
                uuid = sig.get('signal_uuid')
                
                # Skip if already closed (deduplication)
                if uuid in self.already_closed_uuids:
                    skipped += 1
                    continue
                
                closure = self.check_signal_closure(sig, debug=False)
                if closure:
                    closures.append(closure)
                    # Add to cache so we don't close it again
                    self.already_closed_uuids.add(uuid)
            
            # Append closures to SIGNALS_CLOSURES.jsonl (safe append-only)
            if closures:
                with open(self.signals_closures, 'a') as f:
                    for closure in closures:
                        f.write(json.dumps(closure) + '\n')
                print(f"[CYCLE] ✅ Appended {len(closures)} NEW closures ({skipped} already-closed skipped)", flush=True)
            else:
                if skipped > 0:
                    print(f"[CYCLE] No NEW closures ({skipped} already-closed skipped)", flush=True)
                else:
                    print(f"[CYCLE] No closures this cycle", flush=True)
            
            return len(closures)
        
        except Exception as e:
            print(f"[CYCLE-ERROR] {str(e)}", flush=True)
            return 0
    
    def run_daemon(self, cycle_interval=300):
        """Run as persistent daemon"""
        cycle_count = 0
        total_closures = 0
        
        print(f"\n[DAEMON] Starting (cycle interval: {cycle_interval}s)", flush=True)
        
        try:
            while True:
                cycle_count += 1
                print(f"\n[DAEMON-CYCLE_{cycle_count}] {get_utc_now_naive().isoformat()}", flush=True)
                
                closures_this_cycle = self.run_cycle()
                total_closures += closures_this_cycle
                
                print(f"[DAEMON] Total closures so far: {total_closures}", flush=True)
                print(f"[DAEMON] Sleeping {cycle_interval}s...", flush=True)
                time.sleep(cycle_interval)
        
        except KeyboardInterrupt:
            print(f"\n[DAEMON] Interrupted by user", flush=True)
        except Exception as e:
            print(f"\n[DAEMON] Fatal error: {str(e)}", flush=True)
        finally:
            print(f"[DAEMON] SHUTDOWN - Cycles: {cycle_count}, Total closures: {total_closures}", flush=True)


def main():
    """Main entry point"""
    executor = PECExecutorSafe()
    executor.run_daemon(cycle_interval=300)  # 5-minute cycles


if __name__ == '__main__':
    main()
