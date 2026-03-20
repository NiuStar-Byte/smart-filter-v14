#!/usr/bin/env python3
"""
Track PEC Backtest Progress Over Time
Shows how many signals have been closed/processed each hour
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

def track_progress():
    """Read all hourly summaries and show progress"""
    
    reports_dir = '/Users/geniustarigan/.openclaw/workspace/pec_hourly_reports'
    
    if not os.path.exists(reports_dir):
        print("No reports directory found yet. Run pec_reporter_hourly.py first.")
        return
    
    # Load all summary files
    summaries = []
    for file in sorted(os.listdir(reports_dir)):
        if file.endswith('-summary.json'):
            try:
                with open(os.path.join(reports_dir, file), 'r') as f:
                    summary = json.load(f)
                    summaries.append(summary)
            except:
                pass
    
    if not summaries:
        print("No summary reports found.")
        return
    
    print("=" * 100)
    print("PEC BACKTEST PROGRESS TRACKING")
    print("=" * 100)
    print()
    print(f"{'Timestamp':<25} {'Total':<8} {'Closed':<8} {'WR':<10} {'P&L':<15} {'Progress':<10}")
    print("-" * 100)
    
    # Show each hourly snapshot
    for summary in summaries:
        timestamp = summary.get('timestamp', 'N/A')
        total = summary.get('total', 'N/A')
        closed = summary.get('closed', 'N/A')
        wr = summary.get('wr', 'N/A')
        pnl = summary.get('pnl', 'N/A')
        
        # Calculate progress
        try:
            closed_int = int(closed)
            total_int = int(total)
            progress = f"{100*closed_int/total_int:.1f}%"
        except:
            progress = "N/A"
        
        # Truncate timestamp for display
        ts_short = timestamp.split('T')[0] + ' ' + timestamp.split('T')[1][:5] if 'T' in timestamp else timestamp
        
        print(f"{ts_short:<25} {total:<8} {closed:<8} {wr:<10} {pnl:<15} {progress:<10}")
    
    print()
    print("=" * 100)
    
    # Show final stats
    if summaries:
        latest = summaries[-1]
        first = summaries[0]
        
        print(f"\nFinal Status:")
        print(f"  Total Signals: {latest.get('total', 'N/A')}")
        print(f"  Closed: {latest.get('closed', 'N/A')}")
        print(f"  Win Rate: {latest.get('wr', 'N/A')}")
        print(f"  Total P&L: {latest.get('pnl', 'N/A')}")
        
        # Show if progress is being made
        try:
            closed_now = int(latest['closed'])
            closed_first = int(first['closed'])
            if closed_now > closed_first:
                print(f"\n✓ Progress: {closed_now - closed_first} new signals closed since first report")
            else:
                print(f"\n⚠ No new signals closed since first report")
        except:
            pass

if __name__ == '__main__':
    track_progress()
