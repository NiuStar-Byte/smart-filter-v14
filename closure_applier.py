#!/usr/bin/env python3
"""
CLOSURE APPLIER - THE MISSING LINK
Reads closures from SIGNALS_CLOSURES.jsonl and applies them back to SIGNALS_MASTER.jsonl
Ensures SIGNALS_MASTER always reflects actual signal status
Trackers then read correct data from SIGNALS_MASTER

Architecture:
- pec_executor_safe: Detects closures → appends to SIGNALS_CLOSURES (clean)
- closure_applier: Applies closures → updates SIGNALS_MASTER atomically (every 5 min)
- Trackers: Read SIGNALS_MASTER (single source of truth, always current)

Zero tolerance: SIGNALS_MASTER must always show actual status
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

class ClosureApplier:
    """
    Applies closure data from SIGNALS_CLOSURES.jsonl back to SIGNALS_MASTER.jsonl
    Ensures SIGNALS_MASTER is always the single source of truth
    """
    
    def __init__(self):
        self.workspace_root = os.path.dirname(os.path.abspath(__file__))
        self.workspace = os.path.dirname(self.workspace_root)
        self.signals_master = os.path.join(self.workspace_root, 'SIGNALS_MASTER.jsonl')
        self.signals_closures = os.path.join(self.workspace, 'SIGNALS_CLOSURES.jsonl')
        self.audit_log = os.path.join(self.workspace, 'closure_applier_audit.log')
        
        # Track which closures have been applied
        self.last_applied_count = 0
        
        print(f"[APPLIER] ✅ Initialized", flush=True)
        print(f"  Reading closures from: SIGNALS_CLOSURES.jsonl", flush=True)
        print(f"  Applying to: SIGNALS_MASTER.jsonl", flush=True)
        print(f"  Audit log: closure_applier_audit.log", flush=True)
    
    def load_closures(self):
        """Load all closure events from SIGNALS_CLOSURES.jsonl"""
        closures_by_uuid = {}
        
        if not os.path.exists(self.signals_closures):
            return closures_by_uuid
        
        try:
            with open(self.signals_closures, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            closure = json.loads(line)
                            uuid = closure.get('signal_uuid')
                            if uuid:
                                closures_by_uuid[uuid] = closure
                        except:
                            pass
        except:
            pass
        
        return closures_by_uuid
    
    def apply_closures(self):
        """
        Apply all pending closures to SIGNALS_MASTER.jsonl
        Uses atomic file replacement to ensure data safety
        """
        closures = self.load_closures()
        
        if not closures:
            print(f"[APPLIER] No pending closures", flush=True)
            return 0
        
        print(f"[APPLIER] Loaded {len(closures):,} closures from SIGNALS_CLOSURES.jsonl", flush=True)
        
        # Read all signals, apply closures where found
        signals = []
        closures_applied = 0
        
        try:
            with open(self.signals_master, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            signal = json.loads(line)
                            uuid = signal.get('signal_uuid')
                            
                            # If this signal has a closure event, apply it
                            if uuid in closures:
                                closure = closures[uuid]
                                
                                # Only apply if signal is still OPEN (avoid double-applying)
                                if signal.get('status') == 'OPEN':
                                    signal['status'] = closure['status']
                                    signal['actual_exit_price'] = closure['actual_exit_price']
                                    signal['closed_at'] = closure['closed_at']
                                    
                                    # Calculate P&L if not in closure
                                    if 'pnl_usd' not in closure:
                                        entry = float(signal.get('entry_price', 0))
                                        exit_price = float(closure['actual_exit_price'])
                                        direction = signal.get('direction', 'LONG')
                                        
                                        if direction == 'LONG':
                                            pnl = (exit_price - entry) / entry * 100
                                        else:  # SHORT
                                            pnl = (entry - exit_price) / entry * 100
                                        signal['pnl_usd'] = pnl
                                    else:
                                        signal['pnl_usd'] = closure['pnl_usd']
                                    
                                    closures_applied += 1
                                    
                                    # Log for audit
                                    with open(self.audit_log, 'a') as f:
                                        f.write(f"{datetime.utcnow().isoformat()} | APPLIED | {uuid} | {signal['status']} | {signal.get('symbol')}\n")
                            
                            signals.append(signal)
                        except:
                            pass
            
            if closures_applied == 0:
                print(f"[APPLIER] No NEW closures to apply (all already applied)", flush=True)
                return 0
            
            # Write back atomically
            temp_file = self.signals_master + '.tmp'
            with open(temp_file, 'w') as f:
                for signal in signals:
                    f.write(json.dumps(signal) + '\n')
            
            # Atomic replace
            os.replace(temp_file, self.signals_master)
            
            print(f"[APPLIER] ✅ Applied {closures_applied:,} closures to SIGNALS_MASTER.jsonl", flush=True)
            print(f"[APPLIER] ✅ SIGNALS_MASTER now reflects actual signal status (single source of truth)", flush=True)
            
            return closures_applied
        
        except Exception as e:
            print(f"[APPLIER-ERROR] {str(e)}", flush=True)
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return 0
    
    def run_cycle(self):
        """Run one cycle of closure application"""
        try:
            applied = self.apply_closures()
            return applied
        except Exception as e:
            print(f"[CYCLE-ERROR] {str(e)}", flush=True)
            return 0
    
    def run_daemon(self, cycle_interval=300):
        """Run as daemon, apply closures every cycle_interval seconds"""
        import time
        
        print(f"\n[DAEMON] Starting closure applier (cycle interval: {cycle_interval}s)", flush=True)
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                print(f"\n[DAEMON] Cycle #{cycle_count} ({datetime.utcnow().isoformat()})", flush=True)
                
                self.run_cycle()
                
                time.sleep(cycle_interval)
        
        except KeyboardInterrupt:
            print(f"\n[DAEMON] Stopped by user", flush=True)
        except Exception as e:
            print(f"[DAEMON-ERROR] {str(e)}", flush=True)

if __name__ == '__main__':
    applier = ClosureApplier()
    applier.run_daemon(cycle_interval=300)
