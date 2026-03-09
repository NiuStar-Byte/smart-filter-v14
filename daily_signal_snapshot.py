#!/usr/bin/env python3
"""
Daily Signal Snapshot
Copies SENT_SIGNALS.jsonl to timestamped cumulative file.
Counts total and closed signals. Runs daily at 23:00 GMT+7.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Configuration
WORKSPACE = "/Users/geniustarigan/.openclaw/workspace"
SENT_SIGNALS_FILE = f"{WORKSPACE}/SENT_SIGNALS.jsonl"

# Get today's date in YYYY-MM-DD format (Jakarta timezone)
today = datetime.now().strftime("%Y-%m-%d")
CUMULATIVE_FILE = f"{WORKSPACE}/SENT_SIGNALS_CUMULATIVE_{today}.jsonl"

def run_snapshot():
    """Run daily snapshot."""
    
    # Check if SENT_SIGNALS.jsonl exists
    if not os.path.exists(SENT_SIGNALS_FILE):
        print(f"Error: {SENT_SIGNALS_FILE} not found")
        return False
    
    try:
        # Read all signals from SENT_SIGNALS.jsonl
        signals = []
        total_count = 0
        closed_count = 0
        
        with open(SENT_SIGNALS_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    signal = json.loads(line)
                    signals.append(signal)
                    total_count += 1
                    
                    # Count closed signals (status is not OPEN and not null)
                    status = signal.get('status')
                    if status and status.upper() != 'OPEN':
                        closed_count += 1
        
        # Write cumulative file
        with open(CUMULATIVE_FILE, 'w') as f:
            for signal in signals:
                f.write(json.dumps(signal) + '\n')
        
        # Get file creation timestamp
        save_time = datetime.now().isoformat()
        
        # Output summary
        print(f"File created successfully: {CUMULATIVE_FILE}")
        print(f"Total signal count: {total_count}")
        print(f"Closed signal count: {closed_count}")
        print(f"Timestamp of save: {save_time}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = run_snapshot()
    exit(0 if success else 1)
