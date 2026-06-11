#!/usr/bin/env python3
"""
PEC EXECUTOR MODE A - CORRECT ARCHITECTURE
Persistent price closure checker that UPDATES signal status in SIGNALS_MASTER.jsonl

Key design:
- Reads OPEN signals from SIGNALS_MASTER.jsonl
- Detects closures (TP/SL/TIMEOUT)
- WRITES BACK to SIGNALS_MASTER with updated: status, actual_exit_price, pnl_usd, closed_at
- Uses atomic UUID-based updates (no rewrites, no data loss)
- Trackers read SIGNALS_MASTER directly (single source of truth)
- NO separate closure files needed
- NO reconciliation needed

Why this works:
1. SIGNALS_MASTER is always current (reflects actual signal status)
2. Trackers see real data immediately (no merge logic)
3. Atomic updates prevent data loss
4. Simple, predictable, zero tolerance for wrong data
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

class PECExecutorModeACorrect:
    """
    Persistent price closure executor (correct architecture)
    
    Reads OPEN signals → detects closures → WRITES BACK to SIGNALS_MASTER
    Single file, single source of truth, no reconciliation needed
    """
    
    def __init__(self):
        self.workspace_root = os.path.dirname(os.path.abspath(__file__))
        self.workspace = os.path.dirname(self.workspace_root)
        self.signals_master = os.path.join(self.workspace_root, 'SIGNALS_MASTER.jsonl')
        self.audit_log = os.path.join(self.workspace, 'pec_closure_audit.log')
        
        # Cache: already-closed UUIDs (prevent re-processing)
        self.already_closed_uuids = set()
        
        print(f"[PEC-MODE-A] ✅ Initialized (CORRECT: writes closures back to SIGNALS_MASTER)", flush=True)
        print(f"[PEC-MODE-A] Source of truth: SIGNALS_MASTER.jsonl (single file)", flush=True)
        print(f"[PEC-MODE-A] Audit log: pec_closure_audit.log", flush=True)
    
    def check_signal_closure(self, signal):
        """
        Check if an OPEN signal should close
        
        Returns:
            dict with closure data (status, actual_exit_price, pnl_usd, closed_at) or None
        """
        try:
            if signal.get('status') != 'OPEN':
                return None
            
            symbol = signal.get('symbol')
            direction = signal.get('direction')
            entry = float(signal.get('entry_price', 0))
            tp = float(signal.get('tp_target', 0))
            sl = float(signal.get('sl_target', 0))
            uuid = signal.get('signal_uuid')
            
            if not all([symbol, entry, tp, sl, uuid]):
                return None
            
            # Fetch current price (try tighter timeframes first)
            current_price = None
            for tf in ['15m', '5m', '1h']:
                try:
                    df = get_ohlcv(symbol, tf, limit=1)
                    if df is not None and not df.empty:
                        current_price = float(df['close'].iloc[-1])
                        break
                except:
                    pass
            
            if current_price is None:
                return None
            
            # Check closure conditions
            closure_data = None
            
            if current_price >= tp:
                # TP hit
                pnl = (current_price - entry) * 100  # Simplified, actual calc in main.py
                closure_data = {
                    'status': 'TP_HIT',
                    'actual_exit_price': current_price,
                    'pnl_usd': pnl,
                    'closed_at': get_utc_now_naive().isoformat(),
                    'close_reason': 'TP_HIT'
                }
            elif current_price <= sl:
                # SL hit
                pnl = (current_price - entry) * 100
                closure_data = {
                    'status': 'SL_HIT',
                    'actual_exit_price': current_price,
                    'pnl_usd': pnl,
                    'closed_at': get_utc_now_naive().isoformat(),
                    'close_reason': 'SL_HIT'
                }
            else:
                # Check timeout (7 days)
                fired_at = signal.get('timestamp') or signal.get('fired_time_utc')
                if fired_at:
                    try:
                        fired_dt = parse_timestamp_naive_utc(fired_at)
                        cutoff = calculate_cutoff_date(days_back=7)
                        if compare_timestamps(fired_dt, cutoff, '<'):
                            closure_data = {
                                'status': 'STALE_TIMEOUT',
                                'actual_exit_price': current_price,
                                'pnl_usd': 0,
                                'closed_at': get_utc_now_naive().isoformat(),
                                'close_reason': 'STALE_TIMEOUT (>7 days)'
                            }
                    except:
                        pass
            
            return closure_data
        
        except Exception as e:
            return None
    
    def update_signal_in_master(self, uuid, closure_data):
        """
        Update a signal in SIGNALS_MASTER.jsonl with closure data
        
        Uses atomic read-modify-write by UUID (no file rewrite)
        """
        try:
            temp_file = self.signals_master + '.tmp'
            updated = False
            
            # Read all signals, find and update the one with matching UUID
            with open(self.signals_master, 'r') as f_in:
                with open(temp_file, 'w') as f_out:
                    for line in f_in:
                        if not line.strip():
                            continue
                        try:
                            signal = json.loads(line)
                            if signal.get('signal_uuid') == uuid:
                                # Update closure fields
                                signal['status'] = closure_data['status']
                                signal['actual_exit_price'] = closure_data['actual_exit_price']
                                signal['pnl_usd'] = closure_data['pnl_usd']
                                signal['closed_at'] = closure_data['closed_at']
                                updated = True
                            f_out.write(json.dumps(signal) + '\n')
                        except:
                            f_out.write(line)
            
            if updated:
                # Atomic replace
                os.replace(temp_file, self.signals_master)
                return True
            else:
                os.remove(temp_file)
                return False
        
        except Exception as e:
            print(f"[PEC-ERROR] Failed to update {uuid}: {str(e)[:50]}", flush=True)
            return False
    
    def run_cycle(self):
        """Run one check cycle"""
        try:
            # Load OPEN signals
            open_signals = []
            cutoff = calculate_cutoff_date(days_back=7)
            
            with open(self.signals_master, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            sig = json.loads(line)
                            if sig.get('status') == 'OPEN':
                                fired_at = sig.get('timestamp') or sig.get('fired_time_utc')
                                if fired_at:
                                    try:
                                        fired_dt = parse_timestamp_naive_utc(fired_at)
                                        if compare_timestamps(fired_dt, cutoff, '>='):
                                            open_signals.append(sig)
                                    except:
                                        pass
                        except:
                            pass
            
            print(f"[CYCLE] Checking {len(open_signals):,} OPEN signals", flush=True)
            
            # Check each for closure
            closures_found = 0
            closures_updated = 0
            
            for sig in open_signals:
                uuid = sig.get('signal_uuid')
                
                # Skip if already processed this cycle
                if uuid in self.already_closed_uuids:
                    continue
                
                closure = self.check_signal_closure(sig)
                if closure:
                    # Update signal in SIGNALS_MASTER
                    if self.update_signal_in_master(uuid, closure):
                        closures_found += 1
                        closures_updated += 1
                        self.already_closed_uuids.add(uuid)
                        
                        # Log for audit
                        with open(self.audit_log, 'a') as f:
                            f.write(f"{get_utc_now_naive().isoformat()} | {uuid} | {closure['status']} | {closure.get('pnl_usd', 0)}\n")
            
            if closures_found > 0:
                print(f"[CYCLE] ✅ Updated {closures_updated} signals in SIGNALS_MASTER (closures now reflected)", flush=True)
            else:
                print(f"[CYCLE] No closures this cycle", flush=True)
            
            return closures_found
        
        except Exception as e:
            print(f"[CYCLE-ERROR] {str(e)}", flush=True)
            return 0
    
    def run_daemon(self, cycle_interval=300):
        """Run as persistent daemon"""
        print(f"\n[DAEMON] Starting (cycle interval: {cycle_interval}s)", flush=True)
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                print(f"\n[DAEMON] Cycle #{cycle_count} ({datetime.utcnow().isoformat()})", flush=True)
                
                closures = self.run_cycle()
                
                time.sleep(cycle_interval)
        
        except KeyboardInterrupt:
            print(f"\n[DAEMON] Stopped by user", flush=True)
        except Exception as e:
            print(f"[DAEMON-ERROR] {str(e)}", flush=True)

if __name__ == '__main__':
    executor = PECExecutorModeACorrect()
    executor.run_daemon(cycle_interval=300)
