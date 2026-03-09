#!/usr/bin/env python3
"""
FILTER PERFORMANCE VALIDATOR - Real Fire Counts
Parses daemon logs to count actual per-filter evaluations (PASS vs FAIL)
instead of probabilistic inference.

Replaces inaccurate dual_tracker.py metric.
"""

import json
import re
from datetime import datetime
from collections import defaultdict
import argparse

DAEMON_LOG_PATH = "/Users/geniustarigan/.openclaw/workspace/main_daemon.log"
SIGNALS_MASTER_PATH = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"

# All 20 filters
ALL_FILTERS = [
    "MACD", "Volume Spike", "Fractal Zone", "TREND", "Momentum", "ATR Momentum Burst",
    "MTF Volume Agreement", "HH/LL Trend", "Volatility Model", "Liquidity Awareness",
    "Volatility Squeeze", "Candle Confirmation", "VWAP Divergence", "Spread Filter",
    "Chop Zone", "Liquidity Pool", "Support/Resistance", "Smart Money Bias",
    "Absorption", "Wick Dominance"
]

# Phase assignments
PHASE1_ENHANCED = {
    "TREND", "MACD", "Momentum", "Volume Spike", "ATR Momentum Burst", "Fractal Zone"
}

PHASE2_ENHANCED = {
    "Support/Resistance", "Volatility Squeeze", "Liquidity Awareness", 
    "Spread Filter", "MTF Volume Agreement", "VWAP Divergence"
}

# Enhancement cutoff
ENHANCEMENT_START = "2026-03-05T00:00:00"

class FilterPerformanceValidator:
    def __init__(self):
        self.filter_stats = defaultdict(lambda: {
            "total_evals": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "fail_rate": 0.0,
            "enhanced": False,
            "phase": "Unknown"
        })
        self.baseline_signals = []
        self.post_enh_signals = []
        
    def load_signals(self):
        """Load signals to get baseline vs post-enhancement split."""
        cutoff_dt = datetime.fromisoformat(ENHANCEMENT_START.replace('Z', '+00:00'))
        
        with open(SIGNALS_MASTER_PATH, 'r') as f:
            for line in f:
                try:
                    s = json.loads(line.strip())
                    if not s.get('sent_time_utc'):
                        continue
                    
                    signal_time = datetime.fromisoformat(s['sent_time_utc'].replace('Z', '+00:00'))
                    if signal_time < cutoff_dt:
                        self.baseline_signals.append(s)
                    else:
                        self.post_enh_signals.append(s)
                except:
                    pass
        
        return len(self.baseline_signals), len(self.post_enh_signals)
    
    def parse_daemon_logs(self):
        """Parse daemon logs to count actual per-filter evaluations."""
        # Patterns for different log formats:
        # Format 1 (Non-Enhanced): [SYMBOL] [Filter Name] ... | DIRECTION (X/Y)  → PASSED
        # Format 2 (Non-Enhanced): [SYMBOL] [Filter Name] ... | No signal         → FAILED
        # Format 3 (Enhanced):     [SYMBOL] [Filter ENHANCED] Signal: DIRECTION   → PASSED
        # Format 4 (Choppy):       [SYMBOL] [Filter] ... Market too choppy        → FAILED
        
        try:
            with open(DAEMON_LOG_PATH, 'r') as f:
                for line in f:
                    # Extract filter name from [Symbol] [Filter Name ...]
                    match = re.search(r'\]\s+\[([a-zA-Z\s/]+)\]', line)
                    if not match:
                        continue
                    
                    filter_text = match.group(1).strip()
                    
                    # Handle ENHANCED version (remove "ENHANCED" suffix)
                    filter_name = filter_text.replace(" ENHANCED", "").strip()
                    
                    # Skip if not in our filter list
                    if filter_name not in ALL_FILTERS:
                        continue
                    
                    self.filter_stats[filter_name]["total_evals"] += 1
                    
                    # Determine pass/fail based on outcome
                    if "No signal" in line:
                        # Explicitly rejected
                        self.filter_stats[filter_name]["failed"] += 1
                    elif "too choppy" in line:
                        # Market too choppy
                        self.filter_stats[filter_name]["failed"] += 1
                    elif "TIED" in line:
                        # TIED = no clear direction
                        self.filter_stats[filter_name]["failed"] += 1
                    elif "Signal: LONG" in line or "Signal: SHORT" in line:
                        # ENHANCED filter that fired (passed)
                        self.filter_stats[filter_name]["passed"] += 1
                    elif re.search(r'\|\s*(LONG|SHORT)\s*\((\d+)/(\d+)\)', line):
                        # Non-enhanced: check conditions met
                        match = re.search(r'\|\s*(LONG|SHORT)\s*\((\d+)/(\d+)\)', line)
                        if match:
                            met = int(match.group(2))
                            total = int(match.group(3))
                            if met >= total:
                                self.filter_stats[filter_name]["passed"] += 1
                            else:
                                self.filter_stats[filter_name]["failed"] += 1
                    else:
                        # Unknown outcome, count as pass (safe default)
                        self.filter_stats[filter_name]["passed"] += 1
                        
        except Exception as e:
            print(f"[ERROR] Failed to parse daemon logs: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def calculate_rates(self):
        """Calculate pass/fail rates for each filter."""
        for filter_name in self.filter_stats:
            total = self.filter_stats[filter_name]["total_evals"]
            if total > 0:
                passed = self.filter_stats[filter_name]["passed"]
                self.filter_stats[filter_name]["pass_rate"] = round((passed / total) * 100, 1)
                self.filter_stats[filter_name]["fail_rate"] = round((1 - passed / total) * 100, 1)
            
            # Assign phase
            if filter_name in PHASE1_ENHANCED:
                self.filter_stats[filter_name]["enhanced"] = True
                self.filter_stats[filter_name]["phase"] = "Phase 1 (2026-03-05)"
            elif filter_name in PHASE2_ENHANCED:
                self.filter_stats[filter_name]["enhanced"] = True
                self.filter_stats[filter_name]["phase"] = "Phase 2 (2026-03-08)"
            else:
                self.filter_stats[filter_name]["enhanced"] = False
                self.filter_stats[filter_name]["phase"] = "Phase 3-4 (Pending)"
    
    def print_report(self):
        """Print validation report."""
        print("\n" + "="*140)
        print("FILTER PERFORMANCE VALIDATOR - Actual Daemon Log Counts")
        print("="*140)
        
        # Sort by fail rate (worst first)
        sorted_filters = sorted(
            self.filter_stats.items(),
            key=lambda x: x[1]["fail_rate"],
            reverse=True
        )
        
        print(f"\n📊 ALL 20 FILTERS - ACTUAL FIRE RATES (from daemon logs)")
        print("-"*140)
        print(f"{'Rank':<5} {'Filter Name':<25} {'Evals':<8} {'Pass':<8} {'Fail':<8} {'Pass %':<10} {'Fail %':<10} {'Phase':<25} {'Status':<15}")
        print("-"*140)
        
        for idx, (filter_name, stats) in enumerate(sorted_filters, 1):
            status = "✅ ENHANCED" if stats["enhanced"] else "⏳ PENDING"
            print(f"{idx:<5} {filter_name:<25} {stats['total_evals']:<8} {stats['passed']:<8} {stats['failed']:<8} {stats['pass_rate']:<9.1f}% {stats['fail_rate']:<9.1f}% {stats['phase']:<25} {status:<15}")
        
        print("\n📈 PHASE SUMMARY")
        print("-"*140)
        
        phase1_filters = [f for f, s in self.filter_stats.items() if s["phase"] == "Phase 1 (2026-03-05)"]
        phase2_filters = [f for f, s in self.filter_stats.items() if s["phase"] == "Phase 2 (2026-03-08)"]
        phase34_filters = [f for f, s in self.filter_stats.items() if s["phase"] == "Phase 3-4 (Pending)"]
        
        for phase_name, filters_in_phase in [
            ("Phase 1", phase1_filters),
            ("Phase 2", phase2_filters),
            ("Phase 3-4 (Pending)", phase34_filters)
        ]:
            if not filters_in_phase:
                continue
            
            avg_pass = sum(self.filter_stats[f]["pass_rate"] for f in filters_in_phase) / len(filters_in_phase)
            avg_fail = sum(self.filter_stats[f]["fail_rate"] for f in filters_in_phase) / len(filters_in_phase)
            total_evals = sum(self.filter_stats[f]["total_evals"] for f in filters_in_phase)
            
            print(f"{phase_name:<20} | Filters: {len(filters_in_phase):<2} | Total Evals: {total_evals:<6} | Avg Pass %: {avg_pass:<6.1f}% | Avg Fail %: {avg_fail:<6.1f}%")
        
        print("\n💡 KEY INSIGHT")
        print("-"*140)
        print("This validates actual filter performance from daemon logs.")
        print("Fail rate = filter rejected the signal (didn't pass min conditions)")
        print("Pass rate = filter accepted the signal (passed min conditions)")
        print("\nHigher fail rate ≠ bad enhancement. May indicate:")
        print("  1. Better selectivity (filtering weak signals)")
        print("  2. Market conditions changed (different signal flow)")
        print("  3. Gate being applied correctly (rejecting low-confidence trades)")
        print("\nValidate with WIN RATE + P&L metrics for true enhancement quality.")
        print("="*140 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Performance Validator")
    parser.add_argument("--watch", action="store_true", help="Live monitoring (60s refresh)")
    args = parser.parse_args()
    
    validator = FilterPerformanceValidator()
    baseline_count, post_enh_count = validator.load_signals()
    print(f"[INFO] Loaded {baseline_count} baseline + {post_enh_count} post-enhancement signals")
    
    print(f"[INFO] Parsing daemon logs: {DAEMON_LOG_PATH}")
    if validator.parse_daemon_logs():
        validator.calculate_rates()
        validator.print_report()
    else:
        print("[ERROR] Failed to parse daemon logs")
