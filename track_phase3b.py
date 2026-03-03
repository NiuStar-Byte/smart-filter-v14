#!/usr/bin/env python3
"""
track_phase3b.py - Phase 3B Monitoring Tool (AUTO-REFRESH EVERY 5 SECONDS)

✅ LIVE MONITORING DASHBOARD - Updates every 5 seconds automatically

Tracks REVERSAL signal quality, filtering rates, and direction-regime impact:
  1. Quality gate results (RQ1, RQ2, RQ3, RQ4)
  2. REVERSAL approval rates (target: 60-80%)
  3. Route decisions (REVERSAL vs TREND_CONTINUATION)
  4. Direction-Regime impact analysis
  5. AUTO-REFRESHES every 5 seconds (no manual re-run needed)

Usage:
  python3 track_phase3b.py              # Full report (auto-refresh every 5s) ✅ DEFAULT
  python3 track_phase3b.py --once       # Single report (no auto-refresh)
  python3 track_phase3b.py --watch      # Live tail of daemon logs
  python3 track_phase3b.py --short      # SHORT reversals only (auto-refresh)
  python3 track_phase3b.py --reversal   # REVERSAL decisions only (auto-refresh)
"""

import json
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import time
import os

def parse_phase3b_logs(log_file="main_daemon.log", hours=24):
    """
    Parse main_daemon.log for Phase 3B metrics.
    
    Extracts:
    - [PHASE3B-RQ] lines: Reversal quality gate results
    - [PHASE3B-FALLBACK] lines: Signals re-routed due to quality check
    - [PHASE3B-APPROVED] lines: Signals approved by quality check
    - [PHASE3B-SCORE] lines: Route scoring decisions
    """
    
    metrics = {
        "total_reversals_checked": 0,
        "reversals_approved": 0,
        "reversals_rejected": 0,
        "rejections_by_combo": defaultdict(int),
        "gate_results": {
            "RQ1_pass_count": 0,
            "RQ2_pass_count": 0,
            "RQ3_pass_count": 0,
            "RQ4_pass_count": 0,
        },
        "by_direction_regime": defaultdict(lambda: {
            "checked": 0,
            "approved": 0,
            "rejected": 0,
            "avg_strength": 0
        }),
        "route_decisions": defaultdict(int),
        "signals_by_tf": defaultdict(lambda: {"checked": 0, "approved": 0}),
        "latest_signals": []
    }
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        for line in lines:
            # Parse [PHASE3B-RQ] lines
            if "[PHASE3B-RQ]" in line:
                metrics["total_reversals_checked"] += 1
                
                # Extract TF, symbol, direction
                parts = line.split()
                idx = parts.index("[PHASE3B-RQ]") + 1
                tf = parts[idx]  # 15min, 30min, 1h
                symbol = parts[idx + 1]
                direction = parts[idx + 2]
                
                # Extract gate results (RQ1✓ RQ2✗ RQ3✓ RQ4✓)
                gate_str = parts[idx + 3]
                gates = gate_str.split(',')
                for gate in gates:
                    if '✓' in gate:
                        gate_name = gate.split('=')[0].strip()
                        if gate_name in metrics["gate_results"]:
                            metrics["gate_results"][f"{gate_name}_pass_count"] += 1
                
                # Extract strength
                strength_str = parts[-1] if parts[-1].endswith('%') else "0%"
                try:
                    strength = float(strength_str.rstrip('%'))
                except:
                    strength = 0
                
                # Track by combo
                combo = f"{direction}_{symbol.split('-')[0]}"  # e.g., SHORT_BTC
                metrics["by_direction_regime"][combo]["checked"] += 1
                metrics["by_direction_regime"][combo]["avg_strength"] += strength
                
                # Track by TF
                metrics["signals_by_tf"][tf]["checked"] += 1
            
            # Parse [PHASE3B-FALLBACK] lines
            elif "[PHASE3B-FALLBACK]" in line:
                metrics["reversals_rejected"] += 1
                parts = line.split()
                idx = parts.index("[PHASE3B-FALLBACK]") + 1
                tf = parts[idx]
                symbol = parts[idx + 1]
                
                # Extract failure reason (in parentheses)
                import re
                match = re.search(r'\(([^)]+)\)', line)
                if match:
                    reason = match.group(1)
                    metrics["rejections_by_combo"][reason] += 1
                
                metrics["signals_by_tf"][tf]["checked"] += 1
            
            # Parse [PHASE3B-APPROVED] lines
            elif "[PHASE3B-APPROVED]" in line:
                metrics["reversals_approved"] += 1
                parts = line.split()
                idx = parts.index("[PHASE3B-APPROVED]") + 1
                tf = parts[idx]
                symbol = parts[idx + 1]
                
                metrics["signals_by_tf"][tf]["approved"] += 1
            
            # Parse [PHASE3B-SCORE] lines for route decisions
            elif "[PHASE3B-SCORE]" in line:
                # Extract route name (REVERSAL or TREND_CONTINUATION)
                import re
                match = re.search(r': (REVERSAL|TREND_CONTINUATION)', line)
                if match:
                    route = match.group(1)
                    metrics["route_decisions"][route] += 1
                
                # Store latest for summary
                if len(metrics["latest_signals"]) < 20:
                    metrics["latest_signals"].append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "log": line.strip()
                    })
        
        # Calculate averages
        for combo in metrics["by_direction_regime"]:
            checked = metrics["by_direction_regime"][combo]["checked"]
            if checked > 0:
                metrics["by_direction_regime"][combo]["avg_strength"] /= checked
                metrics["by_direction_regime"][combo]["approved"] = sum(
                    1 for signal in metrics["latest_signals"]
                    if combo.split('_')[0] in signal["log"] and "APPROVED" in signal["log"]
                )
    
    except FileNotFoundError:
        print(f"[ERROR] Log file not found: {log_file}")
    except Exception as e:
        print(f"[ERROR] Error parsing logs: {e}")
    
    return metrics


def print_phase3b_report(metrics, auto_refresh=False):
    """Print comprehensive Phase 3B monitoring report."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    refresh_indicator = "🔄 AUTO-REFRESH (every 5 seconds)" if auto_refresh else "📊 SINGLE REPORT"
    
    print("\n" + "="*90)
    print("📊 PHASE 3B PERFORMANCE REPORT (Last 24 Hours)")
    print(f"   {refresh_indicator}")
    print(f"   Last updated: {timestamp}")
    print("="*90)
    
    # Summary stats
    total = metrics["total_reversals_checked"]
    approved = metrics["reversals_approved"]
    rejected = metrics["reversals_rejected"]
    
    if total > 0:
        approval_rate = (approved / total) * 100
        rejection_rate = (rejected / total) * 100
    else:
        approval_rate = rejection_rate = 0
    
    print(f"\n📈 REVERSALS OVERVIEW:")
    print(f"  Total Checked:     {total}")
    print(f"  ✅ Approved:       {approved} ({approval_rate:.1f}%)")
    print(f"  ❌ Rejected:       {rejected} ({rejection_rate:.1f}%)")
    
    # Gate performance
    print(f"\n🚪 GATE PERFORMANCE:")
    for gate_name, count in sorted(metrics["gate_results"].items()):
        gate_short = gate_name.replace("_pass_count", "")
        pct = (count / total * 100) if total else 0
        print(f"  {gate_short:4} Pass Rate: {count:3} ({pct:5.1f}%)")
    
    # Rejection reasons
    if metrics["rejections_by_combo"]:
        print(f"\n❌ REJECTION REASONS (Top 5):")
        top_reasons = sorted(metrics["rejections_by_combo"].items(), key=lambda x: x[1], reverse=True)[:5]
        for reason, count in top_reasons:
            print(f"  • {reason}: {count} times")
    
    # By direction-regime
    if metrics["by_direction_regime"]:
        print(f"\n🔄 BY DIRECTION-REGIME COMBO (Top 10):")
        combos = sorted(
            metrics["by_direction_regime"].items(),
            key=lambda x: x[1]["checked"],
            reverse=True
        )[:10]
        for combo, data in combos:
            checked = data["checked"]
            approved = data["approved"]
            strength = data["avg_strength"]
            approval_pct = (approved / checked * 100) if checked else 0
            print(f"  {combo:12} | Checked: {checked:3} | Approved: {approved:3} ({approval_pct:5.1f}%) | Avg Strength: {strength:5.1f}%")
    
    # By timeframe
    if metrics["signals_by_tf"]:
        print(f"\n⏱️  BY TIMEFRAME:")
        for tf in ["15min", "30min", "1h"]:
            if tf in metrics["signals_by_tf"]:
                data = metrics["signals_by_tf"][tf]
                checked = data["checked"]
                approved = data["approved"]
                approval_pct = (approved / checked * 100) if checked else 0
                print(f"  {tf:6} | Checked: {checked:3} | Approved: {approved:3} ({approval_pct:5.1f}%)")
    
    # Route distribution
    if metrics["route_decisions"]:
        print(f"\n🛣️  ROUTE DISTRIBUTION:")
        total_routes = sum(metrics["route_decisions"].values())
        for route, count in sorted(metrics["route_decisions"].items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_routes * 100) if total_routes else 0
            print(f"  {route:25} | {count:4} ({pct:5.1f}%)")
    
    print("\n" + "="*90 + "\n")


def watch_phase3b_logs(log_file="main_daemon.log", interval=5):
    """
    Watch logs in real-time and show Phase 3B decisions.
    """
    
    print(f"🔍 Watching {log_file} for Phase 3B decisions (Ctrl+C to exit)...\n")
    
    try:
        process = subprocess.Popen(
            ["tail", "-f", log_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        for line in process.stdout:
            if "[PHASE3B" in line:
                # Colorize different Phase 3B event types
                if "[PHASE3B-RQ]" in line:
                    print(f"🚪 {line.strip()}")
                elif "[PHASE3B-FALLBACK]" in line:
                    print(f"↩️  {line.strip()}")
                elif "[PHASE3B-APPROVED]" in line:
                    print(f"✅ {line.strip()}")
                elif "[PHASE3B-SCORE]" in line:
                    print(f"📊 {line.strip()}")
                else:
                    print(f"ℹ️  {line.strip()}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Watch mode terminated by user")
        process.terminate()
    except Exception as e:
        print(f"[ERROR] {e}")


def generate_report(auto_refresh=True, hours=24, focus_short=False, focus_reversal=False):
    """Generate Phase 3B report with optional auto-refresh every 5 seconds."""
    cycle = 1
    while True:
        # Clear screen for auto-refresh (not on first run)
        if auto_refresh and cycle > 1:
            os.system('clear')
        
        metrics = parse_phase3b_logs(hours=hours)
        print_phase3b_report(metrics, auto_refresh=auto_refresh)
        
        if focus_short:
            print("\n🔽 SHORT REVERSALS FOCUS:")
            for combo, data in metrics["by_direction_regime"].items():
                if "SHORT" in combo:
                    checked = data["checked"]
                    approved = data["approved"]
                    strength = data["avg_strength"]
                    approval_pct = (approved / checked * 100) if checked else 0
                    print(f"  {combo:12} | {checked:3} checked | {approved:3} approved ({approval_pct:5.1f}%) | Strength: {strength:5.1f}%")
        
        if focus_reversal:
            print("\n🔄 REVERSAL DECISIONS FOCUS:")
            print(f"  REVERSAL routes: {metrics['route_decisions'].get('REVERSAL', 0)}")
            print(f"  TREND_CONTINUATION routes: {metrics['route_decisions'].get('TREND_CONTINUATION', 0)}")
        
        # Auto-refresh logic
        if auto_refresh:
            print(f"\n" + "="*90)
            print(f"⏱️  Next update in 5 seconds... (press Ctrl+C to stop)")
            print(f"="*90)
            try:
                time.sleep(5)
                cycle += 1
            except KeyboardInterrupt:
                print(f"\n\n👋 Stopped auto-refresh monitoring")
                break
        else:
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3B Monitoring Tool")
    parser.add_argument("--watch", action="store_true", help="Watch logs in real-time")
    parser.add_argument("--once", action="store_true", help="Single report (no auto-refresh)")
    parser.add_argument("--hours", type=int, default=24, help="Report window (hours)")
    parser.add_argument("--short", action="store_true", help="Show SHORT signals only")
    parser.add_argument("--reversal", action="store_true", help="Show REVERSAL decisions only")
    
    args = parser.parse_args()
    
    if args.watch:
        watch_phase3b_logs()
    elif args.once:
        metrics = parse_phase3b_logs(hours=args.hours)
        print_phase3b_report(metrics, auto_refresh=False)
        
        if args.short:
            print("\n🔽 SHORT REVERSALS FOCUS:")
            for combo, data in metrics["by_direction_regime"].items():
                if "SHORT" in combo:
                    checked = data["checked"]
                    approved = data["approved"]
                    strength = data["avg_strength"]
                    approval_pct = (approved / checked * 100) if checked else 0
                    print(f"  {combo:12} | {checked:3} checked | {approved:3} approved ({approval_pct:5.1f}%) | Strength: {strength:5.1f}%")
        
        if args.reversal:
            print("\n🔄 REVERSAL DECISIONS FOCUS:")
            print(f"  REVERSAL routes: {metrics['route_decisions'].get('REVERSAL', 0)}")
            print(f"  TREND_CONTINUATION routes: {metrics['route_decisions'].get('TREND_CONTINUATION', 0)}")
    else:
        # Default: auto-refresh every 5 seconds
        generate_report(auto_refresh=True, hours=args.hours, focus_short=args.short, focus_reversal=args.reversal)
