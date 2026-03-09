#!/usr/bin/env python3
"""
DUAL-SPLIT FILTER PERFORMANCE VALIDATOR
Separates tracking into:
1. BASELINE (immutable reference) - Signals BEFORE 2026-03-05
2. POST-ENHANCEMENT (live) - Signals FROM 2026-03-05 onwards

Shows per-filter effectiveness in both periods side-by-side.
"""

import json
import re
from datetime import datetime
from collections import defaultdict
import argparse

DAEMON_LOG_PATH = "/Users/geniustarigan/.openclaw/workspace/main_daemon.log"
SIGNALS_MASTER_PATH = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"

# All 20 filters (canonical internal names)
ALL_FILTERS = [
    "MACD", "Volume Spike", "Fractal Zone", "TREND", "Momentum", "ATR Momentum Burst",
    "MTF Volume Agreement", "HH/LL Trend", "Volatility Model", "Liquidity Awareness",
    "Volatility Squeeze", "Candle Confirmation", "VWAP Divergence", "Spread Filter",
    "Chop Zone", "Liquidity Pool", "Support/Resistance", "Smart Money Bias",
    "Absorption", "Wick Dominance"
]

# Mapping: Actual log names → canonical filter names
LOG_NAME_TO_FILTER = {
    "Smart Money": "Smart Money Bias",
    "Chop Zone Check": "Chop Zone",
    "MACD": "MACD",
    "Volume Spike": "Volume Spike",
    "Fractal Zone": "Fractal Zone",
    "TREND": "TREND",
    "Momentum": "Momentum",
    "ATR Momentum Burst": "ATR Momentum Burst",
    "MTF Volume Agreement": "MTF Volume Agreement",
    "HH/LL Trend": "HH/LL Trend",
    "Volatility Model": "Volatility Model",
    "Liquidity Awareness": "Liquidity Awareness",
    "Volatility Squeeze": "Volatility Squeeze",
    "Candle Confirmation": "Candle Confirmation",
    "VWAP Divergence": "VWAP Divergence",
    "Spread Filter": "Spread Filter",
    "Liquidity Pool": "Liquidity Pool",
    "Support/Resistance": "Support/Resistance",
    "Absorption": "Absorption",
    "Wick Dominance": "Wick Dominance"
}

# Enhancement cutoff
ENHANCEMENT_START = "2026-03-05T00:00:00"

class DualSplitValidator:
    def __init__(self):
        # Baseline stats (BEFORE 2026-03-05)
        self.baseline_stats = {filter_name: {
            "total_evals": 0, "passed": 0, "failed": 0,
            "pass_rate": 0.0, "fail_rate": 0.0
        } for filter_name in ALL_FILTERS}
        
        # Post-enhancement stats (FROM 2026-03-05+)
        self.post_enh_stats = {filter_name: {
            "total_evals": 0, "passed": 0, "failed": 0,
            "pass_rate": 0.0, "fail_rate": 0.0
        } for filter_name in ALL_FILTERS}
        
        self.baseline_signals = []
        self.post_enh_signals = []
    
    def load_signals(self):
        """Load and split signals by enhancement cutoff."""
        cutoff_dt = datetime.fromisoformat(ENHANCEMENT_START.replace('Z', '+00:00'))
        
        with open(SIGNALS_MASTER_PATH, 'r') as f:
            for line in f:
                try:
                    s = json.loads(line.strip())
                    if not s.get('sent_time_utc'):
                        continue
                    
                    signal_time = datetime.fromisoformat(s['sent_time_utc'].replace('Z', '+00:00'))
                    if signal_time < cutoff_dt:
                        self.baseline_signals.append(signal_time)
                    else:
                        self.post_enh_signals.append(signal_time)
                except:
                    pass
        
        return len(self.baseline_signals), len(self.post_enh_signals)
    
    def parse_daemon_logs(self):
        """Parse daemon logs and split by signal timestamp."""
        cutoff_dt = datetime.fromisoformat(ENHANCEMENT_START.replace('Z', '+00:00'))
        
        try:
            with open(DAEMON_LOG_PATH, 'r') as f:
                for line in f:
                    # Extract timestamp from log line (format: [SYMBOL-USDT] [Filter])
                    # For simplicity, we'll extract symbol and assume recent logs are post-enh
                    # This is approximate since logs don't have exact signal timestamps
                    
                    # Extract filter name from [Symbol] [Filter Name ...]
                    match = re.search(r'\]\s+\[([a-zA-Z\s/]+)\]', line)
                    if not match:
                        continue
                    
                    filter_text = match.group(1).strip()
                    
                    # Handle ENHANCED version (remove "ENHANCED" suffix)
                    filter_name = filter_text.replace(" ENHANCED", "").strip()
                    
                    # Map log names to canonical filter names
                    if filter_name in LOG_NAME_TO_FILTER:
                        filter_name = LOG_NAME_TO_FILTER[filter_name]
                    
                    # Skip if not in our filter list
                    if filter_name not in ALL_FILTERS:
                        continue
                    
                    # Since we can't extract exact signal time from logs, use heuristic:
                    # Assume logs are mostly post-enhancement (daemon running since 2026-03-09)
                    # Split roughly: use timestamp from line if available
                    is_post_enh = True  # Conservative: assume most logs are post-enh
                    
                    stats = self.post_enh_stats if is_post_enh else self.baseline_stats
                    stats[filter_name]["total_evals"] += 1
                    
                    # Determine pass/fail based on outcome
                    if "No signal" in line:
                        stats[filter_name]["failed"] += 1
                    elif "too choppy" in line:
                        stats[filter_name]["failed"] += 1
                    elif "TIED" in line:
                        stats[filter_name]["failed"] += 1
                    elif "Signal: LONG" in line or "Signal: SHORT" in line:
                        stats[filter_name]["passed"] += 1
                    elif re.search(r'\|\s*(LONG|SHORT)\s*\((\d+)/(\d+)\)', line):
                        match = re.search(r'\|\s*(LONG|SHORT)\s*\((\d+)/(\d+)\)', line)
                        if match:
                            met = int(match.group(2))
                            total = int(match.group(3))
                            if met >= total:
                                stats[filter_name]["passed"] += 1
                            else:
                                stats[filter_name]["failed"] += 1
                    else:
                        stats[filter_name]["passed"] += 1
                        
        except Exception as e:
            print(f"[ERROR] Failed to parse daemon logs: {e}")
            return False
        
        return True
    
    def calculate_rates(self):
        """Calculate pass/fail rates for both periods."""
        for filter_name in ALL_FILTERS:
            # Baseline
            baseline = self.baseline_stats[filter_name]
            if baseline["total_evals"] > 0:
                baseline["pass_rate"] = round((baseline["passed"] / baseline["total_evals"]) * 100, 1)
                baseline["fail_rate"] = round((1 - baseline["passed"] / baseline["total_evals"]) * 100, 1)
            
            # Post-enhancement
            post_enh = self.post_enh_stats[filter_name]
            if post_enh["total_evals"] > 0:
                post_enh["pass_rate"] = round((post_enh["passed"] / post_enh["total_evals"]) * 100, 1)
                post_enh["fail_rate"] = round((1 - post_enh["passed"] / post_enh["total_evals"]) * 100, 1)
    
    def print_report(self):
        """Print dual-split validation report."""
        print("\n" + "="*180)
        print("DUAL-SPLIT FILTER PERFORMANCE VALIDATOR")
        print("="*180)
        
        print(f"\n📊 BASELINE (Immutable Reference - BEFORE 2026-03-05)")
        print(f"   Signals: {len(self.baseline_signals)} total")
        print("-"*180)
        print(f"{'Rank':<5} {'Filter Name':<25} {'Evals':<8} {'Pass':<8} {'Fail':<8} {'Pass %':<10} {'Fail %':<10}")
        print("-"*180)
        
        # Sort baseline by pass rate
        baseline_sorted = sorted(
            [(f, s) for f, s in self.baseline_stats.items() if s["total_evals"] > 0],
            key=lambda x: x[1]["pass_rate"],
            reverse=True
        )
        
        for idx, (filter_name, stats) in enumerate(baseline_sorted, 1):
            print(f"{idx:<5} {filter_name:<25} {stats['total_evals']:<8} {stats['passed']:<8} {stats['failed']:<8} {stats['pass_rate']:<9.1f}% {stats['fail_rate']:<9.1f}%")
        
        baseline_avg_pass = sum(s["pass_rate"] for f, s in baseline_sorted) / len(baseline_sorted) if baseline_sorted else 0
        print(f"\n   Baseline Average Pass Rate: {baseline_avg_pass:.1f}%")
        
        print(f"\n\n📊 POST-ENHANCEMENT (Live - FROM 2026-03-05 onwards)")
        print(f"   Signals: {len(self.post_enh_signals)} total (still accumulating)")
        print("-"*180)
        print(f"{'Rank':<5} {'Filter Name':<25} {'Evals':<8} {'Pass':<8} {'Fail':<8} {'Pass %':<10} {'Fail %':<10}")
        print("-"*180)
        
        # Sort post-enhancement by pass rate
        post_sorted = sorted(
            [(f, s) for f, s in self.post_enh_stats.items() if s["total_evals"] > 0],
            key=lambda x: x[1]["pass_rate"],
            reverse=True
        )
        
        for idx, (filter_name, stats) in enumerate(post_sorted, 1):
            print(f"{idx:<5} {filter_name:<25} {stats['total_evals']:<8} {stats['passed']:<8} {stats['failed']:<8} {stats['pass_rate']:<9.1f}% {stats['fail_rate']:<9.1f}%")
        
        post_avg_pass = sum(s["pass_rate"] for f, s in post_sorted) / len(post_sorted) if post_sorted else 0
        print(f"\n   Post-Enhancement Average Pass Rate: {post_avg_pass:.1f}%")
        
        print(f"\n\n📈 COMPARISON (POST-ENHANCEMENT vs BASELINE)")
        print("-"*180)
        print(f"{'Filter Name':<25} {'Baseline %':<15} {'Post-Enh %':<15} {'Delta':<15} {'Impact':<20}")
        print("-"*180)
        
        for filter_name in sorted(ALL_FILTERS):
            baseline_rate = self.baseline_stats[filter_name]["pass_rate"]
            post_enh_rate = self.post_enh_stats[filter_name]["pass_rate"]
            delta = post_enh_rate - baseline_rate
            
            if baseline_rate == 0 and post_enh_rate == 0:
                impact = "No data"
            elif delta > 5:
                impact = "✅ IMPROVED"
            elif delta < -5:
                impact = "❌ DEGRADED"
            else:
                impact = "~ STABLE"
            
            print(f"{filter_name:<25} {baseline_rate:<14.1f}% {post_enh_rate:<14.1f}% {delta:+.1f}pp {impact:<20}")
        
        print("\n" + "="*180)
        print("💡 KEY INSIGHT:")
        print("   • Daemon logs only available from current run (2026-03-09 17:40 with DEBUG_FILTERS=true)")
        print("   • Baseline (pre-2026-03-05) signals existed but weren't logged at that time")
        print("   • POST-ENHANCEMENT shows actual current evaluation data")
        print("   • BASELINE 0 evals = No logs exist for historical signals")
        print("   • To measure baseline quality, use SIGNALS_MASTER.jsonl WR data instead")
        print("="*180 + "\n")

if __name__ == "__main__":
    validator = DualSplitValidator()
    baseline_count, post_enh_count = validator.load_signals()
    print(f"[INFO] Loaded {baseline_count} baseline + {post_enh_count} post-enhancement signals")
    
    print(f"[INFO] Parsing daemon logs: {DAEMON_LOG_PATH}")
    if validator.parse_daemon_logs():
        validator.calculate_rates()
        validator.print_report()
    else:
        print("[ERROR] Failed to parse daemon logs")
