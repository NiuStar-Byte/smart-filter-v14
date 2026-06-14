#!/usr/bin/env python3
"""
STARTUP VERIFICATION SCRIPT
===========================
Runs on Mac startup via LaunchAgent.
Ensures ALL critical files exist and ALL services are running.
STRICT: No exceptions. If anything fails, alert immediately.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
import time

WORKSPACE = "/Users/geniustarigan/.openclaw/workspace"

# CRITICAL FILES THAT MUST EXIST
CRITICAL_FILES = [
    "pec_post_deployment_tracker.py",
    "pec_post_deployment_tracker_v2.py",
    "pec_enhanced_reporter.py",
    "MTF_alignment_comparison_tracker_v2.py",
    "validate_tiers_all.py",
    "manual_daily_combo_refresh.py",
    "filter_effectiveness_analyzer_detailed.py",
    "asterdex_live_tracker_baseline_v2.py",
    "asterdex_snapshot_detailed.py",
    "pec_executor_persistent.py",
]

# CRITICAL SERVICES THAT MUST BE RUNNING
CRITICAL_SERVICES = [
    ("main.py", "Signal Generation"),
    ("asterdex_entry_poster.py", "Entry Posting"),
    ("pec_executor_persistent.py", "Position Closure"),
]

def log_startup(message, level="INFO"):
    """Log to both stdout and startup log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S GMT+7")
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line, flush=True)
    
    # Write to persistent log
    log_file = os.path.join(WORKSPACE, ".startup_verification.log")
    try:
        with open(log_file, "a") as f:
            f.write(log_line + "\n")
    except:
        pass

def check_critical_files():
    """Verify all critical tracker files exist"""
    log_startup("=" * 80)
    log_startup("CHECKING CRITICAL TRACKER FILES")
    log_startup("=" * 80)
    
    missing_files = []
    for filename in CRITICAL_FILES:
        filepath = os.path.join(WORKSPACE, filename)
        if os.path.exists(filepath):
            log_startup(f"✅ {filename}")
        else:
            log_startup(f"❌ MISSING: {filename}", "ERROR")
            missing_files.append(filename)
    
    if missing_files:
        log_startup(f"\n⚠️ CRITICAL: {len(missing_files)} files missing!", "ERROR")
        log_startup("Files not found:", "ERROR")
        for f in missing_files:
            log_startup(f"  - {f}", "ERROR")
        return False
    
    log_startup(f"\n✅ All {len(CRITICAL_FILES)} critical files present\n")
    return True

def check_services():
    """Verify all critical services are running"""
    log_startup("=" * 80)
    log_startup("CHECKING CRITICAL SERVICES")
    log_startup("=" * 80)
    
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=10
        )
        ps_output = result.stdout
    except Exception as e:
        log_startup(f"❌ Failed to check processes: {e}", "ERROR")
        return False
    
    running_services = []
    missing_services = []
    
    for service_name, description in CRITICAL_SERVICES:
        if service_name in ps_output:
            running_services.append((service_name, description))
            log_startup(f"✅ {description} ({service_name}) - RUNNING")
        else:
            missing_services.append((service_name, description))
            log_startup(f"❌ {description} ({service_name}) - NOT RUNNING", "ERROR")
    
    if missing_services:
        log_startup(f"\n⚠️ CRITICAL: {len(missing_services)} services not running!", "ERROR")
        log_startup("Attempting to start missing services...", "WARN")
        start_services(missing_services)
        return False
    
    log_startup(f"\n✅ All {len(CRITICAL_SERVICES)} critical services running\n")
    return True

def start_services(services_to_start):
    """Start missing services"""
    for service_name, description in services_to_start:
        log_startup(f"🚀 Starting {description}...")
        script_path = os.path.join(WORKSPACE, service_name)
        
        if not os.path.exists(script_path):
            log_startup(f"❌ Cannot start {service_name} - file not found", "ERROR")
            continue
        
        try:
            # Start in background with nohup
            subprocess.Popen(
                [f"python3", script_path],
                cwd=WORKSPACE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            log_startup(f"✅ Started {description}")
        except Exception as e:
            log_startup(f"❌ Failed to start {service_name}: {e}", "ERROR")

def run_quick_trackers():
    """Run quick health checks on critical trackers"""
    log_startup("=" * 80)
    log_startup("RUNNING TRACKER HEALTH CHECKS")
    log_startup("=" * 80)
    
    trackers_to_check = [
        ("validate_tiers_all.py", "Tier Validation"),
        ("filter_effectiveness_analyzer_detailed.py", "Filter Analysis"),
    ]
    
    for tracker_name, description in trackers_to_check:
        log_startup(f"Running {description}...")
        script_path = os.path.join(WORKSPACE, tracker_name)
        
        try:
            result = subprocess.run(
                ["python3", script_path],
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Extract first few lines of output
                output_lines = result.stdout.split('\n')[:3]
                log_startup(f"✅ {description} - OK")
                for line in output_lines:
                    if line.strip():
                        log_startup(f"   {line}")
            else:
                log_startup(f"❌ {description} - FAILED (exit code {result.returncode})", "ERROR")
                log_startup(f"   Error: {result.stderr[:200]}", "ERROR")
        except subprocess.TimeoutExpired:
            log_startup(f"⚠️ {description} - TIMEOUT (>30s)", "WARN")
        except Exception as e:
            log_startup(f"❌ {description} - ERROR: {e}", "ERROR")

def main():
    """Main startup verification sequence"""
    log_startup("\n" + "=" * 80)
    log_startup("🚀 SYSTEM STARTUP VERIFICATION")
    log_startup(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    log_startup(f"   Workspace: {WORKSPACE}")
    log_startup("=" * 80 + "\n")
    
    # Check 1: Critical files
    files_ok = check_critical_files()
    
    # Check 2: Critical services
    services_ok = check_services()
    
    # Check 3: Quick tracker health
    if files_ok and services_ok:
        run_quick_trackers()
    
    # Summary
    log_startup("\n" + "=" * 80)
    if files_ok and services_ok:
        log_startup("✅ STARTUP VERIFICATION COMPLETE - ALL SYSTEMS OPERATIONAL")
    else:
        log_startup("⚠️ STARTUP VERIFICATION COMPLETED WITH WARNINGS", "WARN")
    log_startup("=" * 80 + "\n")

if __name__ == "__main__":
    main()
