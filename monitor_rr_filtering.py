#!/usr/bin/env python3
"""
monitor_rr_filtering.py - Real-time RR filtering monitoring

Monitors daemon logs for [RR_FILTER] messages to verify:
1. RR filtering is working correctly
2. MIN_ACCEPTED_RR threshold is being enforced
3. No extreme RR signals are being passed through

Usage:
  # Real-time monitoring (follows log)
  python3 monitor_rr_filtering.py --live

  # Analyze existing log
  python3 monitor_rr_filtering.py --file /path/to/daemon.log

  # Dashboard with refresh
  python3 monitor_rr_filtering.py --dashboard
"""

import sys
import time
import json
import re
import os
from collections import defaultdict, deque
from datetime import datetime
import subprocess

LOG_PATTERNS = {
    'rr_accepted': r'\[RR_FILTER\].*signal ACCEPTED.*RR ([\d.]+)',
    'rr_rejected': r'\[RR_FILTER\].*signal REJECTED.*RR ([\d.]+).*MIN ([\d.]+)',
    'symbol': r'(\w+-USDT)',
    'timeframe': r'(15min|30min|1h|4h)',
}

def parse_rr_log_line(line):
    """Extract RR info from a log line"""
    result = {}
    
    # Check if this is an RR_FILTER line
    if '[RR_FILTER]' not in line:
        return None
    
    # Extract status (ACCEPTED or REJECTED)
    if 'ACCEPTED' in line:
        result['status'] = 'ACCEPTED'
        match = re.search(LOG_PATTERNS['rr_accepted'], line)
        if match:
            result['rr'] = float(match.group(1))
    elif 'REJECTED' in line:
        result['status'] = 'REJECTED'
        match = re.search(LOG_PATTERNS['rr_rejected'], line)
        if match:
            result['rr'] = float(match.group(1))
            result['min_rr'] = float(match.group(2))
    else:
        return None
    
    # Extract symbol
    symbol_match = re.search(LOG_PATTERNS['symbol'], line)
    if symbol_match:
        result['symbol'] = symbol_match.group(1)
    
    # Extract timeframe
    tf_match = re.search(LOG_PATTERNS['timeframe'], line)
    if tf_match:
        result['timeframe'] = tf_match.group(1)
    
    return result if 'status' in result else None

def analyze_log_file(log_path):
    """Analyze a log file for RR filtering statistics"""
    stats = {
        'total_checked': 0,
        'accepted': 0,
        'rejected': 0,
        'rr_accepted': [],
        'rr_rejected': [],
        'by_timeframe': defaultdict(lambda: {'accepted': 0, 'rejected': 0}),
        'by_symbol': defaultdict(lambda: {'accepted': 0, 'rejected': 0}),
    }
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                parsed = parse_rr_log_line(line)
                if not parsed:
                    continue
                
                stats['total_checked'] += 1
                
                if parsed['status'] == 'ACCEPTED':
                    stats['accepted'] += 1
                    stats['rr_accepted'].append(parsed['rr'])
                    if 'timeframe' in parsed:
                        stats['by_timeframe'][parsed['timeframe']]['accepted'] += 1
                    if 'symbol' in parsed:
                        stats['by_symbol'][parsed['symbol']]['accepted'] += 1
                
                elif parsed['status'] == 'REJECTED':
                    stats['rejected'] += 1
                    stats['rr_rejected'].append(parsed['rr'])
                    if 'timeframe' in parsed:
                        stats['by_timeframe'][parsed['timeframe']]['rejected'] += 1
                    if 'symbol' in parsed:
                        stats['by_symbol'][parsed['symbol']]['rejected'] += 1
    
    except FileNotFoundError:
        print(f"[ERROR] Log file not found: {log_path}")
        return None
    
    return stats

def print_stats(stats):
    """Pretty print RR filtering statistics"""
    if not stats or stats['total_checked'] == 0:
        print("[INFO] No RR filtering data found in logs")
        return
    
    acceptance_rate = (stats['accepted'] / stats['total_checked'] * 100) if stats['total_checked'] > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"RR FILTERING STATISTICS")
    print(f"{'='*80}")
    print(f"\nTotal Signals Checked: {stats['total_checked']}")
    print(f"  ✓ Accepted: {stats['accepted']} ({acceptance_rate:.1f}%)")
    print(f"  ✗ Rejected: {stats['rejected']} ({100-acceptance_rate:.1f}%)")
    
    if stats['rr_accepted']:
        print(f"\nACCEPTED RR VALUES:")
        print(f"  Min: {min(stats['rr_accepted']):.2f}")
        print(f"  Avg: {sum(stats['rr_accepted'])/len(stats['rr_accepted']):.2f}")
        print(f"  Max: {max(stats['rr_accepted']):.2f}")
    
    if stats['rr_rejected']:
        print(f"\nREJECTED RR VALUES:")
        print(f"  Min: {min(stats['rr_rejected']):.2f}")
        print(f"  Avg: {sum(stats['rr_rejected'])/len(stats['rr_rejected']):.2f}")
        print(f"  Max: {max(stats['rr_rejected']):.2f}")
    
    print(f"\nBY TIMEFRAME:")
    for tf in ['15min', '30min', '1h', '4h']:
        if tf in stats['by_timeframe']:
            data = stats['by_timeframe'][tf]
            total = data['accepted'] + data['rejected']
            accept_rate = (data['accepted'] / total * 100) if total > 0 else 0
            print(f"  {tf:>5}: {data['accepted']:>3} accept, {data['rejected']:>3} reject ({accept_rate:>5.1f}%)")
    
    print(f"\nTOP SYMBOLS (by signal count):")
    sorted_symbols = sorted(stats['by_symbol'].items(), 
                           key=lambda x: x[1]['accepted'] + x[1]['rejected'], 
                           reverse=True)[:10]
    for symbol, data in sorted_symbols:
        total = data['accepted'] + data['rejected']
        accept_rate = (data['accepted'] / total * 100) if total > 0 else 0
        print(f"  {symbol:>12}: {data['accepted']:>2} accept, {data['rejected']:>2} reject ({accept_rate:>5.1f}%)")
    
    print(f"\n{'='*80}\n")

def monitor_live():
    """Monitor daemon logs in real-time"""
    print("[INFO] Starting live RR filtering monitor...")
    print("[INFO] Watching main.py output for [RR_FILTER] messages")
    print("[INFO] Press Ctrl+C to stop\n")
    
    # Try to find running daemon process and follow its output
    try:
        # Look for running main.py process
        result = subprocess.run(
            "ps aux | grep 'python.*main.py' | grep -v grep",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print("[INFO] Found running daemon process")
            print(result.stdout)
        else:
            print("[WARN] No running daemon process found")
            print("[INFO] Monitoring for new processes...")
    
    except Exception as e:
        print(f"[WARN] Could not check for running daemon: {e}")
    
    # Real-time stats
    stats = {
        'total': 0,
        'accepted': 0,
        'rejected': 0,
        'recent': deque(maxlen=20)
    }
    
    print("\n[RR_FILTER] Real-Time Monitoring Active")
    print("-" * 80)
    
    # For now, periodically analyze the main log file if it exists
    log_file = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/daemon.log"
    last_size = 0
    
    try:
        while True:
            time.sleep(5)  # Check every 5 seconds
            
            # Check if log file exists and has grown
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)
                
                if current_size > last_size:
                    # File has new content
                    analyzed = analyze_log_file(log_file)
                    if analyzed:
                        new_checked = analyzed['total_checked'] - stats['total']
                        new_accepted = analyzed['accepted'] - stats['accepted']
                        new_rejected = analyzed['rejected'] - stats['rejected']
                        
                        if new_checked > 0:
                            accept_rate = (new_accepted / new_checked * 100)
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            print(f"[{timestamp}] +{new_checked} signals | " +
                                  f"✓ {new_accepted} ({accept_rate:.0f}%) | " +
                                  f"✗ {new_rejected} ({100-accept_rate:.0f}%)")
                            
                            stats['total'] = analyzed['total_checked']
                            stats['accepted'] = analyzed['accepted']
                            stats['rejected'] = analyzed['rejected']
                    
                    last_size = current_size
            else:
                print(f"[WARN] Log file not found: {log_file}")
                print("[HINT] Run daemon and output to file: python3 main.py > daemon.log 2>&1")
                break
    
    except KeyboardInterrupt:
        print("\n\n[STOP] Monitoring stopped by user")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor RR filtering in daemon logs')
    parser.add_argument('--file', type=str, help='Log file to analyze')
    parser.add_argument('--live', action='store_true', help='Monitor live (follow log file)')
    parser.add_argument('--dashboard', action='store_true', help='Show dashboard with refresh')
    
    args = parser.parse_args()
    
    if args.live:
        monitor_live()
    elif args.dashboard:
        # Dashboard mode - refresh every 30 seconds
        print("[INFO] Starting RR filtering dashboard (refreshes every 30s)")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                log_file = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/daemon.log"
                if os.path.exists(log_file):
                    stats = analyze_log_file(log_file)
                    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}\n")
                    print_stats(stats)
                else:
                    print(f"[ERROR] Log file not found: {log_file}")
                    print("Run daemon with: python3 main.py > daemon.log 2>&1")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n[STOP] Dashboard stopped")
    elif args.file:
        stats = analyze_log_file(args.file)
        print_stats(stats)
    else:
        # Default: try common log file
        log_file = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/daemon.log"
        if os.path.exists(log_file):
            stats = analyze_log_file(log_file)
            print_stats(stats)
        else:
            print("[INFO] No arguments provided and no default log file found")
            print("\nUsage:")
            print("  python3 monitor_rr_filtering.py --file /path/to/daemon.log")
            print("  python3 monitor_rr_filtering.py --live")
            print("  python3 monitor_rr_filtering.py --dashboard")

if __name__ == "__main__":
    main()
