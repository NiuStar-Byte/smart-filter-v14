#!/usr/bin/env python3
"""
🔒 PHASE 3B SIMPLE TRACKER - Count actual logs from daemon
Parses what's ACTUALLY in main_daemon.log (not what we wish was there)
"""

import subprocess
from datetime import datetime
from collections import defaultdict
import sys
import time
import os
import re

def parse_phase3b_actual():
    """Parse ACTUAL Phase 3B logs from main_daemon.log"""
    
    metrics = {
        "total_phase3b_checks": 0,
        "approved": 0,        # ✅ TREND_CONT with score > 0
        "rejected": 0,        # ❌ NONE, AMBIGUOUS, or score 0
        "by_route": defaultdict(int),
        "by_tf": defaultdict(int),
        "by_symbol": defaultdict(int),
    }
    
    try:
        with open("main_daemon.log", 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if "[PHASE3B-SCORE]" not in line:
                continue
            
            metrics["total_phase3b_checks"] += 1
            
            # Parse: [PHASE3B-SCORE] 30min KAITO-USDT: ✅ TREND_CONT: LONG ... (score: 60.0)
            parts = line.split()
            idx = [i for i, p in enumerate(parts) if "[PHASE3B-SCORE]" in p][0]
            
            tf = parts[idx + 1]  # 15min, 30min, 1h
            symbol = parts[idx + 2].rstrip(':')
            
            # Extract score from "(score: X.X)" at end of line
            score_match = re.search(r'\(score:\s*([-\d.]+)\)', line)
            score = 0.0
            if score_match:
                try:
                    score = float(score_match.group(1))
                except:
                    score = 0.0
            
            # Approve if score > 0 (signal will be sent)
            if score > 0:
                metrics["approved"] += 1
                route_indicator = "APPROVED"
            else:
                metrics["rejected"] += 1
                route_indicator = "REJECTED"
            
            # Extract route name
            route = "UNKNOWN"
            if "TREND_CONT" in line or "TREND CONTINUATION" in line:
                route = "TREND_CONT"
            elif "REVERSAL" in line:
                route = "REVERSAL"
            elif "AMBIGUOUS" in line:
                route = "AMBIGUOUS"
            elif "NONE" in line:
                route = "NONE"
            
            metrics["by_route"][route] += 1
            metrics["by_tf"][tf] += 1
            metrics["by_symbol"][symbol] += 1
        
        return metrics
    
    except Exception as e:
        print(f"❌ Error parsing logs: {e}")
        return None

def print_report(metrics):
    """Print clean Phase 3B report"""
    
    if not metrics:
        print("❌ No metrics available")
        return
    
    print("\n" + "="*100)
    print("🔒 PHASE 3B SIMPLE TRACKER - Actual Log Analysis")
    print("="*100)
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print("="*100)
    print()
    
    total = metrics["total_phase3b_checks"]
    approved = metrics["approved"]
    rejected = metrics["rejected"]
    
    if total == 0:
        print("⏳ No Phase 3B checks recorded yet")
        print("   Waiting for daemon to process signals...")
        print()
        print("="*100 + "\n")
        return
    
    approved_pct = (approved / total * 100) if total > 0 else 0
    rejected_pct = (rejected / total * 100) if total > 0 else 0
    
    print(f"📊 PHASE 3B CHECKS (Total: {total})")
    print("─" * 100)
    print(f"  ✅ Approved:  {approved:>6d} ({approved_pct:>5.1f}%)")
    print(f"  ❌ Rejected:  {rejected:>6d} ({rejected_pct:>5.1f}%)")
    print()
    
    print(f"🛣️  ROUTE BREAKDOWN:")
    print("─" * 100)
    for route, count in sorted(metrics["by_route"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {route:<20} : {count:>6d} ({pct:>5.1f}%)")
    print()
    
    print(f"⏱️  BY TIMEFRAME:")
    print("─" * 100)
    for tf in ["15min", "30min", "1h"]:
        count = metrics["by_tf"].get(tf, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {tf:<20} : {count:>6d} ({pct:>5.1f}%)")
    print()
    
    print(f"💰 TOP 5 SYMBOLS:")
    print("─" * 100)
    top_symbols = sorted(metrics["by_symbol"].items(), key=lambda x: x[1], reverse=True)[:5]
    for symbol, count in top_symbols:
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {symbol:<20} : {count:>6d} ({pct:>5.1f}%)")
    
    print()
    print("="*100)
    print("📌 INTERPRETATION:")
    print("="*100)
    print(f"  Phase 3B has processed {total} signal checks")
    print(f"  Approval rate: {approved_pct:.1f}%")
    
    # Check if logs are stale (all TREND_CONT with score 0 = old logs from before fix)
    if approved_pct == 0.0 and metrics["by_route"].get("TREND_CONT", 0) == total:
        print("  ⚠️  LOGS ARE STALE - All TREND_CONT have score 0 (from before fix was deployed)")
        print("  📌 SOLUTION: Restart daemon with fresh code")
        print("     Run: bash RESTART_DAEMON.sh")
    elif approved_pct > 60:
        print("  ✅ Good - Quality gates are working, filtering out weak signals")
    elif approved_pct > 40:
        print("  ⚠️  Moderate - Gates filtering moderately, check if too strict")
    else:
        print("  ❌ Low - Gates might be too strict, rejecting too many")
    print()
    print("="*100 + "\n")

if __name__ == "__main__":
    once_mode = "--once" in sys.argv
    
    if once_mode:
        metrics = parse_phase3b_actual()
        print_report(metrics)
    else:
        try:
            while True:
                os.system('clear' if os.name != 'nt' else 'cls')
                metrics = parse_phase3b_actual()
                print_report(metrics)
                print("⏱️  Refreshing in 5 seconds... (Press Ctrl+C to stop)")
                time.sleep(5)
        except KeyboardInterrupt:
            os.system('clear' if os.name != 'nt' else 'cls')
            print("✅ Stopped\n")
