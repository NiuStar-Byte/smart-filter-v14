#!/usr/bin/env python3
"""
PEC Persistent Executor: Single long-running loop
- Runs continuously, updates every 5 minutes
- No spawning, no child processes, no daemon overhead
- Self-healing: catches/logs errors, keeps running
- One process = one point of responsibility

Start once: python3 pec_executor_persistent.py
(then leave it running in the background)
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import traceback

# Import from smart-filter
sys.path.insert(0, '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main')
from pec_executor import PECExecutor

class PersistentExecutor:
    def __init__(self):
        self.executor = PECExecutor()
        self.update_interval = 5 * 60  # 5 minutes in seconds
        self.workspace_root = "/Users/geniustarigan/.openclaw/workspace"
        self.signals_master_path = os.path.join(self.workspace_root, 'SIGNALS_MASTER.jsonl')
        self.startup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def log(self, msg: str, level: str = "INFO"):
        """Print timestamped log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {msg}", flush=True)
    
    def run_once(self):
        """Execute one update cycle"""
        try:
            self.log("Running update cycle...")
            summary = self.executor.update_signals()
            
            # Report summary
            total_checked = summary.get('total_checked', 0)
            tp_hits = len(summary.get('tp_hits', []))
            sl_hits = len(summary.get('sl_hits', []))
            timeouts = len(summary.get('timeouts', []))
            stale = len(summary.get('stale_timeouts', []))
            
            if tp_hits + sl_hits + timeouts + stale > 0:
                self.log(f"Update complete: {tp_hits}x TP, {sl_hits}x SL, {timeouts}x TIMEOUT, {stale}x STALE", level="UPDATE")
            else:
                self.log(f"Update complete: no status changes", level="UPDATE")
            
            return True
        except Exception as e:
            self.log(f"ERROR in update cycle: {str(e)}", level="ERROR")
            self.log(traceback.format_exc(), level="ERROR")
            return False
    
    def run_forever(self):
        """Main loop - runs until killed"""
        self.log(f"PEC Persistent Executor started", level="START")
        self.log(f"Workspace: {self.workspace_root}", level="START")
        self.log(f"Update interval: {self.update_interval}s (5 minutes)", level="START")
        self.log(f"Source: {self.signals_master_path}", level="START")
        self.log(f"Stop with: pkill -f pec_executor_persistent.py", level="START")
        
        cycle_count = 0
        errors_in_row = 0
        
        while True:
            try:
                cycle_count += 1
                
                # Run update
                success = self.run_once()
                
                if success:
                    errors_in_row = 0
                else:
                    errors_in_row += 1
                    if errors_in_row >= 5:
                        self.log(f"5 consecutive errors - still running but check logs", level="WARN")
                        errors_in_row = 0  # Reset counter to avoid spam
                
                # Wait for next cycle
                self.log(f"Sleeping for {self.update_interval}s until next cycle...", level="DEBUG")
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                self.log("Shutdown signal received (SIGINT)", level="STOP")
                break
            except Exception as e:
                self.log(f"FATAL: Unhandled exception in main loop: {e}", level="FATAL")
                self.log(traceback.format_exc(), level="FATAL")
                # Don't exit - keep running and try again
                time.sleep(60)  # Wait 1 minute before retrying
        
        self.log(f"PEC Persistent Executor stopped", level="STOP")

if __name__ == '__main__':
    executor = PersistentExecutor()
    executor.run_forever()
