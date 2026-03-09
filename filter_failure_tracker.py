#!/usr/bin/env python3
"""
FILTER FAILURE TRACKER (2026-03-08 21:47 GMT+7)

Identifies which filters pass/fail most frequently during signal generation.
Tracks all 20 filters across signals and aggregates failure rates to guide next enhancement priorities.

Usage:
  python3 filter_failure_tracker.py                    # Analyze SENT_SIGNALS.jsonl
  python3 filter_failure_tracker.py --watch            # Live monitoring (10s refresh)
  python3 filter_failure_tracker.py --export-csv       # Export to filter_failure_analysis.csv
"""

import json
import os
import sys
import time
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ===== CONFIGURATION =====
SIGNALS_JSONL_PATH = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"
OUTPUT_CSV = "/Users/geniustarigan/.openclaw/workspace/filter_failure_analysis.csv"
SIGNALS_MASTER_PATH = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"

# All 20 filters in the system (from smart_filter.py)
ALL_FILTERS = [
    "MACD",                    # Weight: 5.0
    "Volume Spike",            # Weight: 5.0
    "Fractal Zone",            # Weight: 4.8
    "TREND",                   # Weight: 4.7
    "Momentum",                # Weight: 4.9
    "ATR Momentum Burst",      # Weight: 4.3
    "MTF Volume Agreement",    # Weight: 5.0 (ENHANCED 2026-03-08)
    "HH/LL Trend",             # Weight: 4.1
    "Volatility Model",        # Weight: 3.9
    "Liquidity Awareness",     # Weight: 5.0 (ENHANCED 2026-03-08)
    "Volatility Squeeze",      # Weight: 3.7 (ENHANCED 2026-03-08)
    "Candle Confirmation",     # Weight: 5.0 (GATEKEEPER)
    "VWAP Divergence",         # Weight: 3.5 (ENHANCED 2026-03-08)
    "Spread Filter",           # Weight: 5.0 (ENHANCED 2026-03-08)
    "Chop Zone",               # Weight: 3.3
    "Liquidity Pool",          # Weight: 3.1
    "Support/Resistance",      # Weight: 5.0 (ENHANCED 2026-03-08)
    "Smart Money Bias",        # Weight: 2.9
    "Absorption",              # Weight: 2.7
    "Wick Dominance"           # Weight: 2.5
]

# Enhanced filters (6 so far, 2026-03-08)
ENHANCED_FILTERS = {
    "Support/Resistance": "36d93ef",
    "Volatility Squeeze": "ef7b495",
    "Liquidity Awareness": "8897c66",
    "Spread Filter": "a953a63",
    "MTF Volume Agreement": "5ae4b96",
    "VWAP Divergence": "13ae855"
}

# ===== FILTER ANALYSIS CLASS =====
class FilterFailureTracker:
    def __init__(self):
        self.filter_stats = {f: {"pass": 0, "fail": 0, "total": 0} for f in ALL_FILTERS}
        self.filter_reasons = defaultdict(list)  # Capture failure reasons
        self.signals_analyzed = 0
        self.last_analysis_time = None
        
    def analyze_signals(self, limit=None):
        """
        Read SENT_SIGNALS.jsonl and extract filter pass/fail data.
        """
        if not os.path.exists(SIGNALS_JSONL_PATH):
            print(f"[ERROR] {SIGNALS_JSONL_PATH} not found")
            return False
        
        signals_count = 0
        with open(SIGNALS_JSONL_PATH, 'r') as f:
            for line in f:
                if limit and signals_count >= limit:
                    break
                try:
                    signal = json.loads(line.strip())
                    self._extract_filter_data(signal)
                    signals_count += 1
                except json.JSONDecodeError:
                    continue
        
        self.signals_analyzed = signals_count
        self.last_analysis_time = datetime.now()
        print(f"\n[OK] Analyzed {signals_count} signals")
        return True
    
    def _extract_filter_data(self, signal):
        """
        Extract filter pass/fail from signal metadata.
        Signals contain 'filter_results' or similar debug info.
        """
        # Check if signal has debug/filter data
        if "filters_evaluated" in signal:
            filters_data = signal.get("filters_evaluated", {})
            for filter_name, result in filters_data.items():
                if filter_name in self.filter_stats:
                    if result.get("passed", False):
                        self.filter_stats[filter_name]["pass"] += 1
                    else:
                        self.filter_stats[filter_name]["fail"] += 1
                    self.filter_stats[filter_name]["total"] += 1
        
        # Alternative: Check for inline filter results in debug_info
        if "debug_info" in signal and "filter_evaluation" in signal["debug_info"]:
            debug = signal["debug_info"]["filter_evaluation"]
            for filter_name, status in debug.items():
                if filter_name in self.filter_stats:
                    self.filter_stats[filter_name]["pass" if status else "fail"] += 1
                    self.filter_stats[filter_name]["total"] += 1
    
    def calculate_failure_rates(self):
        """
        Calculate pass rate and failure rate for each filter.
        Returns sorted by failure rate (worst first).
        """
        results = []
        for filter_name in ALL_FILTERS:
            stats = self.filter_stats[filter_name]
            total = stats["total"]
            if total == 0:
                continue
            
            pass_rate = stats["pass"] / total * 100 if total > 0 else 0
            fail_rate = stats["fail"] / total * 100 if total > 0 else 0
            
            is_enhanced = "✅" if filter_name in ENHANCED_FILTERS else "⏳"
            
            results.append({
                "filter": filter_name,
                "pass": stats["pass"],
                "fail": stats["fail"],
                "total": total,
                "pass_rate": pass_rate,
                "fail_rate": fail_rate,
                "enhanced": is_enhanced
            })
        
        # Sort by failure rate (descending) - worst filters first
        results.sort(key=lambda x: x["fail_rate"], reverse=True)
        return results
    
    def print_report(self):
        """
        Print comprehensive filter failure analysis.
        """
        results = self.calculate_failure_rates()
        
        print("\n" + "="*100)
        print("FILTER FAILURE ANALYSIS REPORT")
        print("="*100)
        print(f"Signals Analyzed: {self.signals_analyzed}")
        print(f"Last Updated: {self.last_analysis_time}")
        print(f"Enhanced Filters: {len(ENHANCED_FILTERS)} / 20")
        print(f"Remaining: {20 - len(ENHANCED_FILTERS)} filters to enhance\n")
        
        # ===== SECTION 1: Failure Rate Ranking (Worst First) =====
        print("📊 FILTER FAILURE RATE RANKING (Worst Filters First)")
        print("-" * 100)
        print(f"{'#':<3} {'Filter Name':<25} {'Enhanced':<10} {'Pass':<8} {'Fail':<8} {'Total':<8} {'Pass %':<10} {'Fail %':<10}")
        print("-" * 100)
        
        for idx, result in enumerate(results, 1):
            print(f"{idx:<3} {result['filter']:<25} {result['enhanced']:<10} "
                  f"{result['pass']:<8} {result['fail']:<8} {result['total']:<8} "
                  f"{result['pass_rate']:<10.1f} {result['fail_rate']:<10.1f}")
        
        # ===== SECTION 2: BOTTLENECK FILTERS (Top 8 Failures) =====
        print("\n" + "="*100)
        print("🎯 BOTTLENECK FILTERS (Top 8 - Priority for Next Enhancements)")
        print("="*100)
        
        bottleneck_filters = results[:8]
        for idx, result in enumerate(bottleneck_filters, 1):
            status = "✅ ENHANCED" if result["enhanced"] == "✅" else "⏳ NEXT TARGET"
            print(f"\n{idx}. {result['filter']} ({status})")
            print(f"   Failure Rate: {result['fail_rate']:.1f}% ({result['fail']} failures out of {result['total']} tests)")
            print(f"   Pass Rate: {result['pass_rate']:.1f}%")
            if result["filter"] in ENHANCED_FILTERS:
                print(f"   GitHub Commit: {ENHANCED_FILTERS[result['filter']]}")
        
        # ===== SECTION 3: Already Enhanced Filters Performance =====
        print("\n" + "="*100)
        print("✅ ENHANCED FILTERS PERFORMANCE (Post-Deployment Tracking)")
        print("="*100)
        
        enhanced_results = [r for r in results if r["enhanced"] == "✅"]
        for result in enhanced_results:
            print(f"• {result['filter']:<25} Pass: {result['pass_rate']:>6.1f}% | Fail: {result['fail_rate']:>6.1f}% | "
                  f"({result['pass']}/{result['total']})")
        
        # ===== SECTION 4: Candidates for Next Enhancement =====
        print("\n" + "="*100)
        print("⏳ CANDIDATES FOR NEXT 8 ENHANCEMENTS (Highest Failure Rate - Target 7-8 filters)")
        print("="*100)
        
        not_enhanced = [r for r in results if r["enhanced"] != "✅"]
        if not_enhanced:
            avg_fail_rate = sum(r["fail_rate"] for r in not_enhanced) / len(not_enhanced)
            expected_improvement = sum(r["fail_rate"] for r in not_enhanced[:8]) / 8 if len(not_enhanced) >= 8 else avg_fail_rate
            
            print(f"Average Failure Rate (non-enhanced): {avg_fail_rate:.1f}%")
            print(f"Expected Avg Improvement (Top 8): ~{expected_improvement * 0.4:.1f}% (40% reduction)")
            print(f"Expected New Baseline: {avg_fail_rate - (expected_improvement * 0.4):.1f}%\n")
            
            print("Top 8 Priority Targets:")
            for idx, result in enumerate(not_enhanced[:8], 1):
                print(f"{idx}. {result['filter']:<25} Failure Rate: {result['fail_rate']:>6.1f}% ({result['fail']}/{result['total']})")
        
        # ===== SECTION 5: Scoring System Context =====
        print("\n" + "="*100)
        print("📋 SYSTEM CONTEXT")
        print("="*100)
        print(f"MIN_SCORE Threshold: 12 (out of ~19 filters, 1 gatekeeper)")
        print(f"Expected Pass Count: ~12 filters per signal")
        print(f"Expected Fail Count: ~7-8 filters per signal (BOTTLENECKS)")
        print(f"Gatekeeper: Candle Confirmation (hard gate, blocks if fails)")
        print(f"\nInterpretation:")
        print(f"  • Filters with >40% failure rate: Critical targets for enhancement")
        print(f"  • Filters with 20-40% failure rate: Important to enhance")
        print(f"  • Filters with <20% failure rate: Working well, may skip")
        
    def export_to_csv(self):
        """
        Export detailed analysis to CSV for spreadsheet analysis.
        """
        results = self.calculate_failure_rates()
        
        with open(OUTPUT_CSV, 'w') as f:
            f.write("Rank,Filter Name,Pass,Fail,Total,Pass Rate %,Fail Rate %,Enhanced,Weight,GitHub Commit\n")
            
            for idx, result in enumerate(results, 1):
                filter_name = result["filter"]
                enhanced = "YES" if result["enhanced"] == "✅" else "NO"
                commit = ENHANCED_FILTERS.get(filter_name, "")
                
                # Approximate weights from smart_filter.py
                weights = {
                    "MACD": 5.0, "Volume Spike": 5.0, "Fractal Zone": 4.8, "TREND": 4.7,
                    "Momentum": 4.9, "ATR Momentum Burst": 4.3, "MTF Volume Agreement": 5.0,
                    "HH/LL Trend": 4.1, "Volatility Model": 3.9, "Liquidity Awareness": 5.0,
                    "Volatility Squeeze": 3.7, "Candle Confirmation": 5.0, "VWAP Divergence": 3.5,
                    "Spread Filter": 5.0, "Chop Zone": 3.3, "Liquidity Pool": 3.1,
                    "Support/Resistance": 5.0, "Smart Money Bias": 2.9, "Absorption": 2.7,
                    "Wick Dominance": 2.5
                }
                weight = weights.get(filter_name, 0)
                
                f.write(f"{idx},{filter_name},{result['pass']},{result['fail']},{result['total']},"
                        f"{result['pass_rate']:.1f},{result['fail_rate']:.1f},{enhanced},{weight},{commit}\n")
        
        print(f"\n✅ Exported to {OUTPUT_CSV}")

# ===== INSTRUMENTED SMART FILTER PATCH =====
def generate_instrumentation_patch():
    """
    Generates a patch for smart_filter.py to export filter results as JSON.
    This should be added to the filter evaluation to capture detailed data.
    """
    patch = """
    # ===== ADD THIS TO SmartFilter.evaluate() METHOD =====
    # Export filter results for failure tracking
    filter_results = {}
    for filter_name in self.filter_names:
        method_name = f"_check_{filter_name.lower().replace(' ', '_').replace('/', '_')}"
        if hasattr(self, method_name):
            try:
                result = getattr(self, method_name)()
                filter_results[filter_name] = {
                    "passed": result if isinstance(result, bool) else result[0],
                    "value": result if isinstance(result, (int, float)) else (result[1] if isinstance(result, tuple) else None)
                }
            except Exception as e:
                filter_results[filter_name] = {"passed": False, "error": str(e)[:50]}
    
    # Store in signal data for tracking
    signal_data["filters_evaluated"] = filter_results
    """
    return patch

# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser(description="Filter Failure Tracker")
    parser.add_argument("--watch", action="store_true", help="Live monitoring (10s refresh)")
    parser.add_argument("--export-csv", action="store_true", help="Export to CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit analysis to N signals")
    
    args = parser.parse_args()
    
    tracker = FilterFailureTracker()
    
    try:
        if args.watch:
            print("[INFO] Live monitoring mode (10s refresh). Press Ctrl+C to stop.\n")
            while True:
                tracker = FilterFailureTracker()  # Fresh analysis each cycle
                tracker.analyze_signals(limit=args.limit)
                tracker.print_report()
                if args.export_csv:
                    tracker.export_to_csv()
                print("\n[INFO] Next update in 10s... (Ctrl+C to stop)")
                time.sleep(10)
        else:
            tracker.analyze_signals(limit=args.limit)
            tracker.print_report()
            if args.export_csv:
                tracker.export_to_csv()
    
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
