#!/usr/bin/env python3
"""
Hourly PEC Enhanced Reporter - Captures signal performance snapshots every hour
Stores reports in REPORTS/ directory for historical tracking and anomaly detection
"""

import os
import subprocess
import json
from datetime import datetime
from pathlib import Path

def run_hourly_reporter():
    """Run pec_enhanced_reporter.py and save output to hourly file"""
    
    # Setup
    workspace = "/Users/geniustarigan/.openclaw/workspace"
    reports_dir = os.path.join(workspace, "REPORTS")
    
    # Create REPORTS directory if doesn't exist
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate filename
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")
    report_file = os.path.join(reports_dir, f"pec_report_{timestamp}.txt")
    
    print(f"[HOURLY REPORTER] Running at {now.strftime('%Y-%m-%d %H:%M:%S GMT+7')}", flush=True)
    
    try:
        # Run PEC reporter and capture output
        result = subprocess.run(
            ["python3", os.path.join(workspace, "pec_enhanced_reporter.py")],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # Write report to file
            with open(report_file, 'w') as f:
                f.write(f"REPORT GENERATED: {now.strftime('%Y-%m-%d %H:%M:%S GMT+7')}\n")
                f.write("=" * 100 + "\n\n")
                f.write(result.stdout)
            
            print(f"✅ Report saved: {report_file}", flush=True)
            
            # Extract signal count for summary
            try:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Total signals:' in line or 'FOUNDATION' in line or 'NEW' in line:
                        print(f"   {line.strip()}", flush=True)
            except:
                pass
        else:
            print(f"❌ Reporter error: {result.stderr[:200]}", flush=True)
    
    except Exception as e:
        print(f"❌ Exception running hourly reporter: {e}", flush=True)
    
    # Cleanup: Keep only last 24 hours of reports (keep ~25 files for 24h + margin)
    try:
        reports = sorted([f for f in os.listdir(reports_dir) if f.startswith('pec_report_')])
        if len(reports) > 25:
            to_delete = reports[:-25]
            for f in to_delete:
                os.remove(os.path.join(reports_dir, f))
            print(f"🧹 Cleanup: Removed {len(to_delete)} old reports", flush=True)
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}", flush=True)

if __name__ == "__main__":
    run_hourly_reporter()
