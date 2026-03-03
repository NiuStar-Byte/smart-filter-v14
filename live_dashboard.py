#!/usr/bin/env python3
"""
🚀 LIVE MONITORING DASHBOARD - All Metrics in Real-Time
Updates every 5 seconds with clear screen
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def get_phase2_stats():
    """Get Phase 2-FIXED stats"""
    try:
        result = subprocess.run(['python3', 'track_phase2_fixed.py', '--once'], 
                              capture_output=True, text=True, timeout=10)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Total signals' in line or 'Closed:' in line or 'WR:' in line:
                return line.strip()
    except:
        return "⏳ Loading..."
    return ""

def get_ab_test_stats():
    """Get A/B test comparison"""
    try:
        result = subprocess.run(['python3', 'COMPARE_AB_TEST.py', '--once'], 
                              capture_output=True, text=True, timeout=10)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Overall Win Rate' in line or 'Total Signals' in line:
                return line.strip()
    except:
        return "⏳ Loading..."
    return ""

def get_pec_stats():
    """Get PEC enhanced reporter stats"""
    try:
        result = subprocess.run(['python3', 'pec_enhanced_reporter.py'], 
                              capture_output=True, text=True, timeout=10)
        lines = result.stdout.split('\n')
        stats = []
        for i, line in enumerate(lines):
            if 'Total' in line and '|' in line:
                stats.append(line.strip())
                if len(stats) >= 3:
                    break
        return stats[0] if stats else "⏳ Loading..."
    except:
        return "⏳ Loading..."

def get_phase3_stats():
    """Get Phase 3 tracker stats"""
    try:
        result = subprocess.run(['python3', 'PHASE3_TRACKER.py'], 
                              capture_output=True, text=True, timeout=10)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Overall Win Rate' in line or 'signals' in line.lower():
                return line.strip()
    except:
        return "⏳ Loading..."
    return ""

def get_phase3b_stats():
    """Get Phase 3B reversal stats"""
    try:
        result = subprocess.run(['python3', 'track_phase3b.py', '--once'], 
                              capture_output=True, text=True, timeout=10)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Total Checked' in line or 'Approved' in line:
                return line.strip()
    except:
        return "⏳ Loading..."
    return ""

def get_daemon_status():
    """Check if daemon is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'python3 main.py'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            return "✅ RUNNING"
        else:
            return "❌ STOPPED"
    except:
        return "⚠️  UNKNOWN"

def get_signal_count():
    """Get current signal count from SENT_SIGNALS.jsonl"""
    try:
        with open('SENT_SIGNALS.jsonl', 'r') as f:
            count = len([line for line in f if line.strip()])
        return count
    except:
        return 0

def run_dashboard():
    """Run the live dashboard"""
    iteration = 0
    
    while True:
        clear_screen()
        iteration += 1
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print("=" * 120)
        print(f"🚀 LIVE TRADING DASHBOARD - {now} (Refresh {iteration})")
        print("=" * 120)
        print()
        
        # System Status
        daemon_status = get_daemon_status()
        signal_count = get_signal_count()
        print(f"📡 DAEMON: {daemon_status} | 📊 TOTAL SIGNALS: {signal_count}")
        print()
        
        # Phase 2-FIXED
        print("┌─ 📈 PHASE 2-FIXED PERFORMANCE")
        print("│")
        phase2 = get_phase2_stats()
        if phase2:
            print(f"│  {phase2}")
        print("│  Status: Direction-aware gates active (BEAR+SHORT recovery monitoring)")
        print("└─" + "─" * 115)
        print()
        
        # A/B Test
        print("┌─ 🧪 A/B TEST COMPARISON (Phase 1 vs Phase 2-FIXED)")
        print("│")
        ab_test = get_ab_test_stats()
        if ab_test:
            print(f"│  {ab_test}")
        print("│  Baseline: 1,205 signals | 29.66% WR | Target: >29% WR by Mar 10")
        print("└─" + "─" * 115)
        print()
        
        # PEC Enhanced
        print("┌─ 📊 PEC ENHANCED REPORT (All Signals)")
        print("│")
        pec = get_pec_stats()
        if pec:
            print(f"│  {pec}")
        print("│  Breakdown: Timeframe, Direction, Route analysis")
        print("└─" + "─" * 115)
        print()
        
        # Phase 3
        print("┌─ 🛣️  PHASE 3 TRACKER (Route Optimization)")
        print("│")
        phase3 = get_phase3_stats()
        if phase3:
            print(f"│  {phase3}")
        print("│  Status: Route optimization analysis (historical)")
        print("└─" + "─" * 115)
        print()
        
        # Phase 3B
        print("┌─ 🔄 PHASE 3B REVERSAL QUALITY GATES")
        print("│")
        phase3b = get_phase3b_stats()
        if phase3b:
            print(f"│  {phase3b}")
        print("│  Status: 4-gate quality validation (RQ1-RQ4) active")
        print("└─" + "─" * 115)
        print()
        
        # Footer
        print("=" * 120)
        print("⏱️  Next refresh in 5 seconds... (Press Ctrl+C to stop)")
        print("=" * 120)
        
        # Wait 5 seconds
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            clear_screen()
            print("✅ Dashboard stopped.")
            sys.exit(0)

if __name__ == "__main__":
    try:
        run_dashboard()
    except KeyboardInterrupt:
        clear_screen()
        print("✅ Dashboard stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
