#!/usr/bin/env python3
"""
PEC Enhanced Reporter - Hourly Snapshot Script
Runs every hour to track backtest progress
"""

import subprocess
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

def run_hourly_report():
    """Run reporter and save snapshot"""
    
    workspace = '/Users/geniustarigan/.openclaw/workspace'
    reports_dir = os.path.join(workspace, 'pec_hourly_reports')
    
    # Create directory if needed
    Path(reports_dir).mkdir(exist_ok=True)
    
    # Get current time in GMT+7
    now_gmt7 = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7)))
    timestamp = now_gmt7.strftime('%Y-%m-%d_%H-00')
    
    report_file = os.path.join(reports_dir, f'{timestamp}-report.txt')
    
    print(f"[REPORTER] Running PEC Enhanced Reporter at {now_gmt7.strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"[REPORTER] Output: {report_file}")
    
    # Run reporter and capture output
    try:
        result = subprocess.run(
            ['python3', 'pec_enhanced_reporter.py'],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Save full output
        with open(report_file, 'w') as f:
            f.write(f"Timestamp: {now_gmt7.strftime('%Y-%m-%d %H:%M:%S GMT+7')}\n")
            f.write("=" * 120 + "\n\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n[STDERR]\n")
                f.write(result.stderr)
        
        # Extract key metrics
        lines = result.stdout.split('\n')
        summary = {}
        
        for line in lines:
            if 'Total Signals:' in line and 'Closed:' in line and 'WR:' in line:
                # Parse: "Total Signals: 2540 | Closed: 1917 | WR: 23.0%"
                parts = line.split('|')
                if len(parts) >= 3:
                    summary['total'] = parts[0].split(':')[-1].strip()
                    summary['closed'] = parts[1].split(':')[-1].strip()
                    summary['wr'] = parts[2].split(':')[-1].strip()
            
            # Get P&L from FOUNDATION BASELINE section only (first occurrence)
            if 'Total P&L (Clean Data):' in line and '$' in line and 'Avg P&L' not in line:
                if 'summary' not in locals() or 'pnl' not in summary:
                    # Extract dollar amount
                    pnl_part = line.split('$')[-1].strip()
                    # Remove any trailing text after the number
                    pnl = pnl_part.split()[0] if pnl_part else 'N/A'
                    summary['pnl'] = pnl
        
        # Save summary as JSON for tracking
        summary_file = os.path.join(reports_dir, f'{timestamp}-summary.json')
        summary['timestamp'] = now_gmt7.isoformat()
        summary['report_file'] = report_file
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print console summary
        print(f"\n[SUMMARY]")
        for key, value in summary.items():
            if key not in ['timestamp', 'report_file']:
                print(f"  {key}: {value}")
        
        print(f"✓ Report saved: {report_file}")
        print(f"✓ Summary saved: {summary_file}")
        
        return True
    
    except Exception as e:
        print(f"[ERROR] Failed to run reporter: {e}")
        return False

if __name__ == '__main__':
    run_hourly_report()
