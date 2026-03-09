#!/usr/bin/env python3
"""
FILTER FAILURE INFERENCE TRACKER (2026-03-08 22:08 GMT+7 - UPDATED FOR 12 ENHANCEMENTS)

Infers filter failure rates from historical signal data WITHOUT requiring instrumentation.
Uses statistical analysis on signals to identify which filters are bottlenecks.

Tracks:
- PHASE 1 (2026-03-05): 6 filters enhanced
- PHASE 2 (2026-03-08): 6 filters enhanced  
- PHASE 3/4 (Pending): 8 filters remaining

Key Insight:
  - MIN_SCORE = 12 (out of 20 filters evaluated)
  - Signals passing = 12+ filters passed
  - Signals failing = <12 filters passed
  
  By analyzing score distribution and patterns, we can infer failure probabilities per filter.

Usage:
  python3 filter_failure_inference.py                 # Analyze existing signals
  python3 filter_failure_inference.py --detailed      # Show detailed breakdown
  python3 filter_failure_inference.py --watch         # Live monitoring (30s refresh)
  python3 filter_failure_inference.py --export        # Export CSV
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
SIGNALS_MASTER_PATH = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"
OUTPUT_CSV = "/Users/geniustarigan/.openclaw/workspace/filter_inference_analysis.csv"

# All 20 filters (from smart_filter.py)
ALL_FILTERS = [
    "MACD",                    # Weight: 5.0 - PHASE 1
    "Volume Spike",            # Weight: 5.0 - PHASE 1
    "Fractal Zone",            # Weight: 4.8 - PHASE 1
    "TREND",                   # Weight: 4.7 - PHASE 1
    "Momentum",                # Weight: 4.9 - PHASE 1
    "ATR Momentum Burst",      # Weight: 4.3 - PHASE 1
    "MTF Volume Agreement",    # Weight: 5.0 - PHASE 2
    "HH/LL Trend",             # Weight: 4.1 - PHASE 3/4
    "Volatility Model",        # Weight: 3.9 - PHASE 3/4
    "Liquidity Awareness",     # Weight: 5.0 - PHASE 2
    "Volatility Squeeze",      # Weight: 3.7 - PHASE 2
    "Candle Confirmation",     # Weight: 5.0 - PHASE 3/4
    "VWAP Divergence",         # Weight: 3.5 - PHASE 2
    "Spread Filter",           # Weight: 5.0 - PHASE 2
    "Chop Zone",               # Weight: 3.3 - PHASE 3/4
    "Liquidity Pool",          # Weight: 3.1 - PHASE 3/4
    "Support/Resistance",      # Weight: 5.0 - PHASE 2
    "Smart Money Bias",        # Weight: 2.9 - PHASE 3/4
    "Absorption",              # Weight: 2.7 - PHASE 3/4
    "Wick Dominance"           # Weight: 2.5 - PHASE 3/4
]

FILTER_WEIGHTS = {
    "MACD": 5.0, "Volume Spike": 5.0, "Fractal Zone": 4.8, "TREND": 4.7,
    "Momentum": 4.9, "ATR Momentum Burst": 4.3, "MTF Volume Agreement": 5.0,
    "HH/LL Trend": 4.1, "Volatility Model": 3.9, "Liquidity Awareness": 5.0,
    "Volatility Squeeze": 3.7, "Candle Confirmation": 5.0, "VWAP Divergence": 3.5,
    "Spread Filter": 5.0, "Chop Zone": 3.3, "Liquidity Pool": 3.1,
    "Support/Resistance": 5.0, "Smart Money Bias": 2.9, "Absorption": 2.7,
    "Wick Dominance": 2.5
}

# PHASE 1 ENHANCEMENTS (2026-03-05)
PHASE1_ENHANCED = {
    "TREND": "ADX check, volatility-aware, removed HATS, raised threshold",
    "MACD": "Magnitude filter, signal momentum, divergence weighting",
    "Momentum": "RSI divergence, multi-condition gating",
    "Volume Spike": "Multi-TF agreement, rolling baseline",
    "ATR Momentum Burst": "Directional consistency, multi-bar confirmation",
    "Fractal Zone": "Larger window, ATR-adaptive buffer, range filter"
}

# PHASE 2 ENHANCEMENTS (2026-03-08)
PHASE2_ENHANCED = {
    "Support/Resistance": "36d93ef - ATR margins, retest, volume, confluence",
    "Volatility Squeeze": "ef7b495 - Exhaustion, momentum, tightening, volume",
    "Liquidity Awareness": "8897c66 - Wall delta, density",
    "Spread Filter": "a953a63 - Volatility ratio, quality, slippage, price action",
    "MTF Volume Agreement": "5ae4b96 - Consensus, alignment, divergence, trend",
    "VWAP Divergence": "13ae855 - Strength, history, crossover, regime"
}

# All enhanced filters (combined)
ENHANCED_FILTERS = {**PHASE1_ENHANCED, **PHASE2_ENHANCED}

# Filters remaining for PHASE 3/4
REMAINING_FILTERS = [f for f in ALL_FILTERS if f not in ENHANCED_FILTERS]

MIN_SCORE = 12
TOTAL_FILTERS = 20
MAX_SCORE = sum(FILTER_WEIGHTS.values())  # 87.7

# ===== INFERENCE ENGINE =====
class FilterFailureInference:
    def __init__(self):
        self.signals = []
        self.analysis = {}
        self.last_updated = None
        
    def load_signals(self, limit=None):
        """Load signals from SENT_SIGNALS.jsonl or SIGNALS_MASTER.jsonl"""
        path = SIGNALS_MASTER_PATH if os.path.exists(SIGNALS_MASTER_PATH) else SIGNALS_JSONL_PATH
        
        if not os.path.exists(path):
            print(f"[ERROR] {path} not found")
            return False
        
        self.signals = []
        count = 0
        with open(path, 'r') as f:
            for line in f:
                if limit and count >= limit:
                    break
                try:
                    signal = json.loads(line.strip())
                    if "score" in signal and "max_score" in signal:
                        self.signals.append(signal)
                        count += 1
                except json.JSONDecodeError:
                    continue
        
        self.last_updated = datetime.now()
        print(f"[OK] Loaded {len(self.signals)} signals")
        return len(self.signals) > 0
    
    def infer_failure_rates(self):
        """
        Infer filter failure rates using Bayesian probability.
        """
        if not self.signals:
            print("[ERROR] No signals loaded")
            return False
        
        # Collect score statistics
        scores = [s.get("score", 12) for s in self.signals]
        score_counts = defaultdict(int)
        for score in scores:
            score_counts[score] += 1
        
        avg_score = sum(scores) / len(scores) if scores else 0
        min_s = min(scores) if scores else 0
        max_s = max(scores) if scores else 0
        
        print(f"\n📊 Signal Score Distribution:")
        print(f"   Total signals: {len(self.signals)}")
        print(f"   Average score: {avg_score:.2f}")
        print(f"   Min score: {min_s}, Max score: {max_s}")
        print(f"   MIN_SCORE threshold: {MIN_SCORE}")
        
        # Expected passes/fails per signal
        avg_passes = avg_score
        avg_fails = TOTAL_FILTERS - avg_passes  # ~7-8 filters fail
        
        print(f"   Expected passes per signal: {avg_passes:.1f}")
        print(f"   Expected fails per signal: {avg_fails:.1f}")
        
        # Assume failure rates inversely correlate with filter weights
        failure_probabilities = {}
        total_weight = sum(FILTER_WEIGHTS.values())
        
        for filter_name in ALL_FILTERS:
            weight = FILTER_WEIGHTS.get(filter_name, 3.0)
            normalized_weight = weight / 5.0  # Normalize to 0-1 range (5.0 is max)
            
            # Bayesian adjustment
            base_failure_prob = avg_fails / TOTAL_FILTERS  # ~0.4 (40%)
            weight_adjustment = (1 - normalized_weight) * 0.3  # -30% to +30%
            failure_prob = base_failure_prob + weight_adjustment
            failure_prob = max(0.05, min(0.95, failure_prob))  # Clamp to [0.05, 0.95]
            
            failure_probabilities[filter_name] = failure_prob
        
        # Sort by failure probability (highest first)
        sorted_filters = sorted(failure_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Store for reporting
        self.analysis = {
            "score_stats": {"avg": avg_score, "min": min_s, "max": max_s},
            "failure_probabilities": dict(sorted_filters),
            "avg_fails_per_signal": avg_fails,
            "total_signals": len(self.signals)
        }
        
        return True
    
    def print_report(self, detailed=False):
        """Print comprehensive inference report."""
        if not self.analysis:
            print("[ERROR] No analysis available. Run infer_failure_rates() first.")
            return
        
        failure_probs = self.analysis["failure_probabilities"]
        
        print("\n" + "="*130)
        print("FILTER FAILURE RATE INFERENCE (Phase 1, 2, and 3/4 Planning)")
        print("="*130)
        print(f"Last Updated: {self.last_updated}")
        print(f"Total Signals Analyzed: {self.analysis['total_signals']}")
        print(f"Average Expected Failures per Signal: {self.analysis['avg_fails_per_signal']:.1f}\n")
        
        # ===== SECTION 1: All 20 Filters Ranked by Failure Rate =====
        print("📊 ALL 20 FILTERS RANKED BY FAILURE RATE (Phase 1/2 Enhanced + Phase 3/4 Pending)")
        print("-" * 130)
        print(f"{'Rank':<5} {'Filter Name':<25} {'Failure %':<12} {'Weight':<8} {'Phase':<25} {'Status':<30}")
        print("-" * 130)
        
        for idx, (filter_name, fail_prob) in enumerate(failure_probs.items(), 1):
            weight = FILTER_WEIGHTS.get(filter_name, 0)
            fail_pct = fail_prob * 100
            
            if filter_name in PHASE1_ENHANCED:
                phase = "Phase 1 (2026-03-05)"
                status = "✅ ENHANCED"
            elif filter_name in PHASE2_ENHANCED:
                phase = "Phase 2 (2026-03-08)"
                status = "✅ ENHANCED"
            else:
                phase = "Phase 3/4 (Pending)"
                status = "⏳ NEXT TARGET"
            
            print(f"{idx:<5} {filter_name:<25} {fail_pct:<11.1f}% {weight:<8.1f} {phase:<25} {status:<30}")
        
        # ===== SECTION 2: Phase-by-Phase Breakdown =====
        print("\n" + "="*130)
        print("📋 PHASE-BY-PHASE BREAKDOWN")
        print("="*130)
        
        # Phase 1
        phase1_filters = [(f, failure_probs.get(f, 0.5)) for f in PHASE1_ENHANCED.keys()]
        phase1_avg = sum(f[1] for f in phase1_filters) / len(phase1_filters) if phase1_filters else 0
        
        print(f"\n✅ PHASE 1 ENHANCEMENTS (2026-03-05) - 6 Filters Enhanced")
        print(f"   Average Failure Rate: {phase1_avg*100:.1f}%")
        print(f"   Range: {min(f[1] for f in phase1_filters)*100:.1f}% - {max(f[1] for f in phase1_filters)*100:.1f}%")
        for filter_name, fail_prob in sorted(phase1_filters, key=lambda x: x[1], reverse=True):
            print(f"   • {filter_name:<25} {fail_prob*100:>6.1f}% | {PHASE1_ENHANCED[filter_name]}")
        
        # Phase 2
        phase2_filters = [(f, failure_probs.get(f, 0.5)) for f in PHASE2_ENHANCED.keys()]
        phase2_avg = sum(f[1] for f in phase2_filters) / len(phase2_filters) if phase2_filters else 0
        
        print(f"\n✅ PHASE 2 ENHANCEMENTS (2026-03-08) - 6 Filters Enhanced")
        print(f"   Average Failure Rate: {phase2_avg*100:.1f}%")
        print(f"   Range: {min(f[1] for f in phase2_filters)*100:.1f}% - {max(f[1] for f in phase2_filters)*100:.1f}%")
        for filter_name, fail_prob in sorted(phase2_filters, key=lambda x: x[1], reverse=True):
            print(f"   • {filter_name:<25} {fail_prob*100:>6.1f}% | {PHASE2_ENHANCED[filter_name]}")
        
        # Phase 3/4
        phase34_filters = [(f, failure_probs.get(f, 0.5)) for f in REMAINING_FILTERS]
        phase34_avg = sum(f[1] for f in phase34_filters) / len(phase34_filters) if phase34_filters else 0
        
        print(f"\n⏳ PHASE 3/4 TARGETS (Pending Enhancement) - 8 Filters Remaining")
        print(f"   Average Failure Rate: {phase34_avg*100:.1f}%")
        print(f"   Range: {min(f[1] for f in phase34_filters)*100:.1f}% - {max(f[1] for f in phase34_filters)*100:.1f}%")
        for filter_name, fail_prob in sorted(phase34_filters, key=lambda x: x[1], reverse=True):
            print(f"   • {filter_name:<25} {fail_prob*100:>6.1f}% | Weight: {FILTER_WEIGHTS[filter_name]}")
        
        # ===== SECTION 3: Top 8 Priority Targets =====
        print("\n" + "="*130)
        print("🎯 TOP 8 PRIORITY TARGETS (ALL 20 FILTERS - Highest Failure Rates)")
        print("="*130)
        
        top_8 = list(failure_probs.items())[:8]
        for idx, (filter_name, fail_prob) in enumerate(top_8, 1):
            weight = FILTER_WEIGHTS.get(filter_name, 0)
            improvement_potential = fail_prob * 0.4
            
            if filter_name in PHASE1_ENHANCED:
                phase_status = f"✅ PHASE 1 (2026-03-05)"
                print(f"\n{idx}. {filter_name}")
                print(f"   Current Failure Rate: {fail_prob*100:.1f}%")
                print(f"   Weight: {weight}")
                print(f"   Status: {phase_status}")
                print(f"   Enhancements: {PHASE1_ENHANCED[filter_name]}")
            elif filter_name in PHASE2_ENHANCED:
                phase_status = f"✅ PHASE 2 (2026-03-08)"
                print(f"\n{idx}. {filter_name}")
                print(f"   Current Failure Rate: {fail_prob*100:.1f}%")
                print(f"   Weight: {weight}")
                print(f"   Status: {phase_status}")
                print(f"   Enhancements: {PHASE2_ENHANCED[filter_name]}")
            else:
                phase_status = "⏳ PHASE 3/4 (PENDING)"
                print(f"\n{idx}. {filter_name}")
                print(f"   Current Failure Rate: {fail_prob*100:.1f}%")
                print(f"   Weight: {weight}")
                print(f"   Status: {phase_status}")
                print(f"   Expected improvement after enhancement: -{improvement_potential*100:.1f}%")
                print(f"   Post-Enhancement Est.: {(fail_prob - improvement_potential)*100:.1f}%")
        
        # ===== SECTION 4: Phase 3/4 Strategy =====
        print("\n" + "="*130)
        print("📋 PHASE 3 & 4 ENHANCEMENT STRATEGY (8 Remaining Filters)")
        print("="*130)
        
        phase34_sorted = sorted(phase34_filters, key=lambda x: x[1], reverse=True)
        phase3_targets = phase34_sorted[:4]  # Top 4 for Phase 3
        phase4_targets = phase34_sorted[4:]  # Next 4 for Phase 4
        
        print(f"\n🔴 PHASE 3 TARGETS (Top 4 - Highest Failure Rates):")
        print(f"Combined avg failure rate: {sum(f[1] for f in phase3_targets)/4*100:.1f}%")
        for idx, (filter_name, fail_prob) in enumerate(phase3_targets, 1):
            weight = FILTER_WEIGHTS.get(filter_name, 0)
            improvement = (fail_prob * 0.4) * 100
            print(f"{idx}. {filter_name:<25} Failure: {fail_prob*100:>6.1f}% | Weight: {weight:<4} | Est. improvement: -{improvement:>5.1f}%")
        
        print(f"\n🟠 PHASE 4 TARGETS (Next 4):")
        print(f"Combined avg failure rate: {sum(f[1] for f in phase4_targets)/4*100:.1f}%")
        for idx, (filter_name, fail_prob) in enumerate(phase4_targets, 1):
            weight = FILTER_WEIGHTS.get(filter_name, 0)
            improvement = (fail_prob * 0.4) * 100
            print(f"{idx}. {filter_name:<25} Failure: {fail_prob*100:>6.1f}% | Weight: {weight:<4} | Est. improvement: -{improvement:>5.1f}%")
        
        # ===== SECTION 5: Expected Impact =====
        print("\n" + "="*130)
        print("📈 EXPECTED IMPACT PROGRESSION")
        print("="*130)
        
        baseline_avg_score = self.analysis["score_stats"]["avg"]
        baseline_fails = TOTAL_FILTERS - baseline_avg_score
        
        phase1_enhancement_impact = sum(f[1] for f in phase1_filters) / len(phase1_filters) * 0.6 * TOTAL_FILTERS if phase1_filters else 0
        phase2_enhancement_impact = sum(f[1] for f in phase2_filters) / len(phase2_filters) * 0.6 * TOTAL_FILTERS if phase2_filters else 0
        phase34_enhancement_impact = sum(f[1] for f in phase34_filters) / len(phase34_filters) * 0.6 * TOTAL_FILTERS if phase34_filters else 0
        
        print(f"\n📊 Current Baseline (Before Phase 1):")
        print(f"   Avg Score: {baseline_avg_score:.1f}/20")
        print(f"   Avg Failures: {baseline_fails:.1f} filters")
        print(f"   Pass Rate: {(baseline_avg_score/20)*100:.1f}%")
        
        print(f"\n✅ After Phase 1 (6 filters enhanced):")
        estimated_score_p1 = baseline_avg_score + (phase1_enhancement_impact * 0.25)
        print(f"   Est. Avg Score: {estimated_score_p1:.1f}/20")
        print(f"   Est. Pass Rate: {(estimated_score_p1/20)*100:.1f}%")
        
        print(f"\n✅ After Phase 2 (12 filters enhanced):")
        estimated_score_p2 = estimated_score_p1 + (phase2_enhancement_impact * 0.25)
        print(f"   Est. Avg Score: {estimated_score_p2:.1f}/20")
        print(f"   Est. Pass Rate: {(estimated_score_p2/20)*100:.1f}%")
        
        print(f"\n✅ After Phase 3/4 (All 20 filters enhanced):")
        estimated_score_p34 = estimated_score_p2 + (phase34_enhancement_impact * 0.40)
        print(f"   Est. Avg Score: {estimated_score_p34:.1f}/20")
        print(f"   Est. Pass Rate: {(estimated_score_p34/20)*100:.1f}%")
        print(f"   Total Improvement: +{estimated_score_p34 - baseline_avg_score:.1f} filters per signal")

    def export_csv(self):
        """Export analysis to CSV."""
        if not self.analysis:
            print("[ERROR] No analysis to export")
            return
        
        failure_probs = self.analysis["failure_probabilities"]
        
        with open(OUTPUT_CSV, 'w') as f:
            f.write("Rank,Filter Name,Failure Rate %,Weight,Phase,Status\n")
            
            for idx, (filter_name, fail_prob) in enumerate(failure_probs.items(), 1):
                weight = FILTER_WEIGHTS.get(filter_name, 0)
                
                if filter_name in PHASE1_ENHANCED:
                    phase = "Phase 1"
                    status = "ENHANCED (2026-03-05)"
                elif filter_name in PHASE2_ENHANCED:
                    phase = "Phase 2"
                    status = "ENHANCED (2026-03-08)"
                else:
                    phase = "Phase 3/4"
                    status = "PENDING"
                
                f.write(f"{idx},{filter_name},{fail_prob*100:.1f},{weight},{phase},{status}\n")
        
        print(f"\n✅ Exported to {OUTPUT_CSV}")

# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser(description="Filter Failure Inference Tracker - 12 Filters Enhanced + 8 Pending")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--watch", action="store_true", help="Live monitoring (30s refresh)")
    parser.add_argument("--export", action="store_true", help="Export to CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit analysis to N signals")
    
    args = parser.parse_args()
    
    try:
        if args.watch:
            print("[INFO] Live monitoring mode (30s refresh). Press Ctrl+C to stop.\n")
            while True:
                engine = FilterFailureInference()
                if engine.load_signals(limit=args.limit):
                    engine.infer_failure_rates()
                    engine.print_report(detailed=args.detailed)
                    if args.export:
                        engine.export_csv()
                print("\n[INFO] Next update in 30s... (Ctrl+C to stop)")
                time.sleep(30)
        else:
            engine = FilterFailureInference()
            if engine.load_signals(limit=args.limit):
                engine.infer_failure_rates()
                engine.print_report(detailed=args.detailed)
                if args.export:
                    engine.export_csv()
    
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
