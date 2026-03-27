#!/usr/bin/env python3
"""
PEC Unified Supervisor: Single point of control for BOTH daemons
- Starts both main.py and pec_executor_persistent.py as subprocesses
- Monitors both continuously (health checks every 10 seconds)
- Auto-restarts either if it dies
- Alerts immediately if restart fails 5x in a row
- Logs everything to single file
- STOPS ONLY when explicitly killed

This eliminates the "one runs, one stops" problem.
Both processes are now a UNIT - they live and die together.

Start once: python3 pec_unified_supervisor.py
Stop: pkill -f pec_unified_supervisor
Monitor: tail -f pec_supervisor.log
"""

import subprocess
import time
import os
import sys
from datetime import datetime
from pathlib import Path

class ProcessManager:
    def __init__(self):
        self.workspace_root = "/Users/geniustarigan/.openclaw/workspace"
        self.log_file = os.path.join(self.workspace_root, 'pec_supervisor.log')
        
        # Process definitions
        self.processes = {
            'signal_generator': {
                'cmd': ['python3', 'smart-filter-v14-main/main.py'],
                'name': 'Signal Generator (main.py)',
                'cwd': self.workspace_root,
                'log': 'main_daemon.log',
                'proc': None,
                'restarts': 0,
                'last_check': None
            },
            'signal_executor': {
                'cmd': ['python3', 'pec_executor_persistent.py'],
                'name': 'Signal Executor (pec_executor_persistent.py)',
                'cwd': self.workspace_root,
                'log': 'pec_persistent.log',
                'proc': None,
                'restarts': 0,
                'last_check': None
            }
        }
        
        self.check_interval = 10  # seconds
        self.restart_threshold = 5  # alert after 5 restarts in a row
        self.startup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def log(self, msg: str, level: str = "INFO"):
        """Write to supervisor log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level:8}] {msg}"
        print(line, flush=True)
        
        with open(self.log_file, 'a') as f:
            f.write(line + '\n')
    
    def start_process(self, key: str) -> bool:
        """Start a process, return True if successful"""
        proc_def = self.processes[key]
        
        try:
            stdout_file = os.path.join(self.workspace_root, proc_def['log'])
            stdout = open(stdout_file, 'a')
            
            proc_def['proc'] = subprocess.Popen(
                proc_def['cmd'],
                cwd=proc_def['cwd'],
                stdout=stdout,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group
            )
            
            self.log(f"✅ Started {proc_def['name']} (PID {proc_def['proc'].pid})", level="START")
            proc_def['last_check'] = time.time()
            return True
        except Exception as e:
            self.log(f"❌ Failed to start {proc_def['name']}: {e}", level="ERROR")
            return False
    
    def check_process(self, key: str) -> bool:
        """Check if process is alive, restart if dead"""
        proc_def = self.processes[key]
        
        if proc_def['proc'] is None:
            # Never started
            return self.start_process(key)
        
        # Check if process is still alive
        if proc_def['proc'].poll() is None:
            # Process is running
            proc_def['last_check'] = time.time()
            return True
        
        # Process died
        self.log(f"⚠️  {proc_def['name']} died (PID {proc_def['proc'].pid})", level="WARN")
        proc_def['restarts'] += 1
        
        # Try to restart
        if proc_def['restarts'] <= self.restart_threshold:
            self.log(f"🔄 Restarting {proc_def['name']} (attempt {proc_def['restarts']})", level="RESTART")
            success = self.start_process(key)
            if success:
                proc_def['restarts'] = 0  # Reset counter on success
            return success
        else:
            self.log(f"🚨 CRITICAL: {proc_def['name']} crashed {self.restart_threshold}x - MANUAL INTERVENTION REQUIRED", level="CRITICAL")
            return False
    
    def health_check(self):
        """Check both processes"""
        all_healthy = True
        
        for key in self.processes:
            if not self.check_process(key):
                all_healthy = False
        
        return all_healthy
    
    def stop_all(self):
        """Stop both processes gracefully"""
        self.log("Stopping all processes...", level="STOP")
        
        for key in self.processes:
            proc_def = self.processes[key]
            if proc_def['proc'] is not None:
                try:
                    os.killpg(os.getpgid(proc_def['proc'].pid), 15)  # SIGTERM to process group
                    proc_def['proc'].wait(timeout=5)
                    self.log(f"✅ Stopped {proc_def['name']}", level="STOP")
                except Exception as e:
                    self.log(f"⚠️  Could not stop {proc_def['name']}: {e}", level="WARN")
    
    def run(self):
        """Main supervisor loop"""
        self.log("=" * 80, level="START")
        self.log("PEC UNIFIED SUPERVISOR started", level="START")
        self.log(f"Managing: Signal Generator (main.py) + Signal Executor (pec_executor_persistent.py)", level="START")
        self.log(f"Health check interval: {self.check_interval}s", level="START")
        self.log(f"Critical restart threshold: {self.restart_threshold}x", level="START")
        self.log(f"Stop with: pkill -f pec_unified_supervisor", level="START")
        self.log("=" * 80, level="START")
        
        # Start both processes
        for key in self.processes:
            self.start_process(key)
            time.sleep(1)
        
        # Main loop
        try:
            while True:
                time.sleep(self.check_interval)
                
                # Check health
                all_healthy = self.health_check()
                
                if not all_healthy:
                    # Log status after unhealthy check
                    status = {k: "✅ RUNNING" if self.processes[k]['proc'] and self.processes[k]['proc'].poll() is None else "❌ DEAD" 
                              for k in self.processes}
                    self.log(f"Status: {status['signal_generator']} / {status['signal_executor']}", level="CHECK")
        
        except KeyboardInterrupt:
            self.log("Interrupt signal received", level="STOP")
            self.stop_all()
            self.log("Supervisor exiting", level="STOP")
            sys.exit(0)

if __name__ == '__main__':
    manager = ProcessManager()
    manager.run()
