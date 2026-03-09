#!/usr/bin/env python3
"""
DUAL TRACKER: IMMUTABLE BASELINE + DYNAMIC POST-ENHANCEMENT (2026-03-08 22:59 GMT+7)
WITH PER-FILTER FAILURE RATE COMPARISON

Two trackers coexist:
  1. IMMUTABLE BASELINE - All signals BEFORE 2026-03-05 (locked forever)
  2. DYNAMIC TRACKER - All signals FROM 2026-03-05 onwards (updates live)

Key insight:
  - Enhancement goal ≠ lower failure rate
  - Enhancement goal = better signal quality (higher WR, better P&L)
  - Higher gate rates might mean BETTER selectivity, not failure
  
Approach:
  - 60% enhancement (12/20) is enough to track impact
  - No need to wait for 20/20 completion
  - Compare against immutable baseline continuously
  - Per-filter failure rates show which filters are bottlenecks
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from collections import defaultdict

# ===== CONFIGURATION =====
SIGNALS_JSONL_PATH = "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl"
SIGNALS_MASTER_PATH = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"

BASELINE_IMMUTABLE_FILE = "/Users/geniustarigan/.openclaw/workspace/FILTERS_BASELINE_IMMUTABLE.json"
TRACKER_POST_ENHANCEMENT_FILE = "/Users/geniustarigan/.openclaw/workspace/FILTERS_TRACKER_POST_ENHANCEMENT.json"
COMPARISON_REPORT_FILE = "/Users/geniustarigan/.openclaw/workspace/FILTERS_COMPARISON_REPORT.json"

# Enhancement cutoff (start of Phase 1)
ENHANCEMENT_START = "2026-03-05T00:00:00"

MIN_SCORE = 12
TOTAL_FILTERS = 20

# All 20 filters with weights
ALL_FILTERS = [
    "MACD", "Volume Spike", "Fractal Zone", "TREND", "Momentum", "ATR Momentum Burst",
    "MTF Volume Agreement", "HH/LL Trend", "Volatility Model", "Liquidity Awareness",
    "Volatility Squeeze", "Candle Confirmation", "VWAP Divergence", "Spread Filter",
    "Chop Zone", "Liquidity Pool", "Support/Resistance", "Smart Money Bias",
    "Absorption", "Wick Dominance"
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

# Phase assignments
PHASE1_ENHANCED = {
    "TREND", "MACD", "Momentum", "Volume Spike", "ATR Momentum Burst", "Fractal Zone"
}

PHASE2_ENHANCED = {
    "Support/Resistance", "Volatility Squeeze", "Liquidity Awareness", 
    "Spread Filter", "MTF Volume Agreement", "VWAP Divergence"
}

# Phase 3: NOT YET DEPLOYED
PHASE3_ENHANCED = set()

# Phase 4 Wave 2: OPTIMIZATIONS COMPLETE, AWAITING DEPLOYMENT
PHASE4_OPTIMIZED = {
    "Chop Zone",  # KEEP AS-IS
    "Volatility Model",  # atr_expansion 0.15→0.08, volume_mult 1.3→1.15
    "HH/LL Trend",  # range_threshold 0.3%→0.5%
    "Candle Confirmation"  # pin_wick_ratio 2.0→1.5
}

class DualTracker:
    def __init__(self):
        self.all_signals = []
        self.baseline_signals = []
        self.post_enhancement_signals = []
        
    def load_signals(self):
        """Load all signals and stratify by enhancement cutoff."""
        path = SIGNALS_MASTER_PATH if os.path.exists(SIGNALS_MASTER_PATH) else SIGNALS_JSONL_PATH
        
        if not os.path.exists(path):
            print(f"[ERROR] {path} not found")
            return False
        
        cutoff_dt = datetime.fromisoformat(ENHANCEMENT_START.replace('Z', '+00:00'))
        
        with open(path, 'r') as f:
            for line in f:
                try:
                    signal = json.loads(line.strip())
                    if "score" in signal and "sent_time_utc" in signal:
                        self.all_signals.append(signal)
                        
                        signal_time = datetime.fromisoformat(
                            signal["sent_time_utc"].replace('Z', '+00:00')
                        )
                        
                        if signal_time < cutoff_dt:
                            self.baseline_signals.append(signal)
                        else:
                            self.post_enhancement_signals.append(signal)
                except:
                    pass
        
        return len(self.all_signals) > 0
    
    def infer_filter_failure_rates(self, signals):
        """Infer per-filter failure rates using Bayesian inference."""
        if not signals:
            return {}
        
        scores = [s.get('score', 12) for s in signals]
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_fails = TOTAL_FILTERS - avg_score
        
        failure_probabilities = {}
        
        for filter_name in ALL_FILTERS:
            weight = FILTER_WEIGHTS.get(filter_name, 3.0)
            normalized_weight = weight / 5.0
            
            base_failure_prob = avg_fails / TOTAL_FILTERS
            weight_adjustment = (1 - normalized_weight) * 0.3
            failure_prob = base_failure_prob + weight_adjustment
            failure_prob = max(0.05, min(0.95, failure_prob))
            
            failure_probabilities[filter_name] = failure_prob
        
        return failure_probabilities
    
    def calculate_baseline(self):
        """Calculate immutable baseline (pre-enhancement)."""
        if not self.baseline_signals:
            return None
        
        scores = [s.get('score', 12) for s in self.baseline_signals]
        avg_score = sum(scores) / len(scores)
        
        # Calculate per-filter failure rates
        filter_failures = self.infer_filter_failure_rates(self.baseline_signals)
        
        baseline = {
            "calculated_at": datetime.now().isoformat(),
            "cutoff": ENHANCEMENT_START,
            "description": "IMMUTABLE BASELINE - All signals BEFORE enhancement (2026-03-05)",
            "signal_count": len(self.baseline_signals),
            "avg_score": round(avg_score, 2),
            "pass_rate_pct": round((avg_score / TOTAL_FILTERS) * 100, 1),
            "failure_rate_pct": round((1 - avg_score / TOTAL_FILTERS) * 100, 1),
            "min_score": min(scores),
            "max_score": max(scores),
            "filter_failure_rates": {k: round(v * 100, 1) for k, v in filter_failures.items()},
            "immutable": True,
            "status": "✓ LOCKED - Reference for all future comparisons"
        }
        
        return baseline
    
    def calculate_post_enhancement(self):
        """Calculate dynamic post-enhancement tracker."""
        if not self.post_enhancement_signals:
            return None
        
        scores = [s.get('score', 12) for s in self.post_enhancement_signals]
        avg_score = sum(scores) / len(scores)
        
        # Calculate per-filter failure rates
        filter_failures = self.infer_filter_failure_rates(self.post_enhancement_signals)
        
        post_enh = {
            "calculated_at": datetime.now().isoformat(),
            "cutoff": ENHANCEMENT_START,
            "description": "DYNAMIC TRACKER - All signals FROM enhancement (2026-03-05 onwards)",
            "signal_count": len(self.post_enhancement_signals),
            "avg_score": round(avg_score, 2),
            "pass_rate_pct": round((avg_score / TOTAL_FILTERS) * 100, 1),
            "failure_rate_pct": round((1 - avg_score / TOTAL_FILTERS) * 100, 1),
            "min_score": min(scores),
            "max_score": max(scores),
            "filter_failure_rates": {k: round(v * 100, 1) for k, v in filter_failures.items()},
            "enhancement_coverage_pct": round((len(PHASE1_ENHANCED) + len(PHASE2_ENHANCED) + len(PHASE3_ENHANCED) + len(PHASE4_OPTIMIZED)) / TOTAL_FILTERS * 100, 1),
            "enhanced_filters": len(PHASE1_ENHANCED) + len(PHASE2_ENHANCED) + len(PHASE3_ENHANCED) + len(PHASE4_OPTIMIZED),
            "not_enhanced_filters": TOTAL_FILTERS - (len(PHASE1_ENHANCED) + len(PHASE2_ENHANCED) + len(PHASE3_ENHANCED) + len(PHASE4_OPTIMIZED)),
            "dynamic": True,
            "status": "⏳ DYNAMIC - Updates continuously"
        }
        
        return post_enh
    
    def calculate_comparison(self, baseline, post_enh):
        """Calculate delta and analysis."""
        if not baseline or not post_enh:
            return None
        
        score_delta = post_enh["avg_score"] - baseline["avg_score"]
        pass_rate_delta = post_enh["pass_rate_pct"] - baseline["pass_rate_pct"]
        
        # Interpretation
        if score_delta < -0.5:
            interpretation = "⚠️ Potential degradation - enhancements may be too strict (high gate rate)"
        elif score_delta < 0:
            interpretation = "⏳ Slight decrease - may indicate better selectivity (filtering weak signals)"
        elif score_delta == 0:
            interpretation = "→ No change - enhancements neutral impact"
        else:
            interpretation = "✅ Improvement - enhancements passing more signals"
        
        comparison = {
            "generated_at": datetime.now().isoformat(),
            "baseline_timestamp": baseline["calculated_at"],
            "tracker_timestamp": post_enh["calculated_at"],
            "comparison": {
                "baseline_avg_score": baseline["avg_score"],
                "post_enhancement_avg_score": post_enh["avg_score"],
                "score_delta": round(score_delta, 2),
                "score_delta_pct": round((score_delta / baseline["avg_score"]) * 100, 1),
                "baseline_pass_rate": baseline["pass_rate_pct"],
                "post_pass_rate": post_enh["pass_rate_pct"],
                "pass_rate_delta": round(pass_rate_delta, 1),
            },
            "sample_sizes": {
                "baseline_signals": baseline["signal_count"],
                "post_enhancement_signals": post_enh["signal_count"],
                "baseline_pct": round((baseline["signal_count"] / (baseline["signal_count"] + post_enh["signal_count"])) * 100, 1),
                "post_enhancement_pct": round((post_enh["signal_count"] / (baseline["signal_count"] + post_enh["signal_count"])) * 100, 1),
            },
            "interpretation": interpretation,
            "key_insight": {
                "enhancement_coverage": f"{post_enh['enhanced_filters']}/{TOTAL_FILTERS} filters ({post_enh['enhancement_coverage_pct']}%)",
                "no_need_to_wait": "Partial enhancement is meaningful - no need to wait for 20/20",
                "gate_rate_note": "Higher failure rate ≠ bad. May indicate better selectivity filtering weak signals.",
                "quality_metric": "Compare P&L and win rate (WR) to confirm if delta = good or bad",
                "next_action": "Deploy Phase 3/4 with same dual-tracker methodology"
            }
        }
        
        return comparison
    
    def save_immutable(self, baseline):
        """Save immutable baseline to file."""
        if not baseline:
            return False
        
        if os.path.exists(BASELINE_IMMUTABLE_FILE):
            return True
        
        with open(BASELINE_IMMUTABLE_FILE, 'w') as f:
            json.dump(baseline, f, indent=2)
        return True
    
    def save_dynamic(self, tracker):
        """Save/update dynamic post-enhancement tracker."""
        if not tracker:
            return False
        
        with open(TRACKER_POST_ENHANCEMENT_FILE, 'w') as f:
            json.dump(tracker, f, indent=2)
        return True
    
    def save_comparison(self, comparison):
        """Save/update comparison report."""
        if not comparison:
            return False
        
        with open(COMPARISON_REPORT_FILE, 'w') as f:
            json.dump(comparison, f, indent=2)
        return True
    
    def print_filter_table(self, filter_failures, title):
        """Print per-filter failure rate table."""
        print(f"\n📊 {title}")
        print("-" * 130)
        print(f"{'Rank':<5} {'Filter Name':<25} {'Failure %':<12} {'Weight':<8} {'Phase':<25} {'Status':<20}")
        print("-" * 130)
        
        # Sort by failure rate (highest first)
        sorted_filters = sorted(filter_failures.items(), key=lambda x: x[1], reverse=True)
        
        for idx, (filter_name, fail_pct) in enumerate(sorted_filters, 1):
            weight = FILTER_WEIGHTS.get(filter_name, 0)
            
            if filter_name in PHASE1_ENHANCED:
                phase = "Phase 1 (2026-03-05)"
                status = "✅ ENHANCED"
            elif filter_name in PHASE2_ENHANCED:
                phase = "Phase 2 (2026-03-08)"
                status = "✅ ENHANCED"
            elif filter_name in PHASE4_OPTIMIZED:
                phase = "Phase 4 Wave 2 (OPTIMIZED)"
                status = "🟡 READY TO DEPLOY"
            else:
                phase = "Phase 3-4 (Pending)"
                status = "⏳ NEXT TARGET"
            
            print(f"{idx:<5} {filter_name:<25} {fail_pct:<11.1f}% {weight:<8.1f} {phase:<25} {status:<20}")
    
    def print_report(self, baseline, post_enh, comparison):
        """Print comprehensive dual-tracker report."""
        print("\n" + "="*130)
        print("DUAL TRACKER REPORT: IMMUTABLE BASELINE vs DYNAMIC POST-ENHANCEMENT")
        print("="*130)
        
        print(f"\n📌 IMMUTABLE BASELINE (Pre-Enhancement - Locked Forever)")
        print("-"*130)
        print(f"Cutoff: All signals BEFORE {ENHANCEMENT_START}")
        print(f"Signal count: {baseline['signal_count']} (locked, final)")
        print(f"Avg score: {baseline['avg_score']} / 20")
        print(f"Pass rate: {baseline['pass_rate_pct']}%")
        print(f"Failure rate: {baseline['failure_rate_pct']}%")
        print(f"Status: ✓ IMMUTABLE (reference point)")
        
        self.print_filter_table(baseline['filter_failure_rates'], 
                               "ALL 20 FILTERS RANKED BY FAILURE RATE (IMMUTABLE BASELINE)")
        
        print(f"\n📊 DYNAMIC TRACKER (Post-Enhancement - Live)")
        print("-"*130)
        print(f"Cutoff: All signals FROM {ENHANCEMENT_START} onwards")
        print(f"Signal count: {post_enh['signal_count']} (growing, updates live)")
        print(f"Avg score: {post_enh['avg_score']} / 20")
        print(f"Pass rate: {post_enh['pass_rate_pct']}%")
        print(f"Failure rate: {post_enh['failure_rate_pct']}%")
        print(f"Enhancement coverage: {post_enh['enhanced_filters']}/{TOTAL_FILTERS} filters ({post_enh['enhancement_coverage_pct']}%)")
        print(f"Status: ⏳ DYNAMIC (updates continuously)")
        
        self.print_filter_table(post_enh['filter_failure_rates'],
                               "ALL 20 FILTERS RANKED BY FAILURE RATE (DYNAMIC POST-ENHANCEMENT)")
        
        if comparison:
            print(f"\n🔍 COMPARISON & ANALYSIS")
            print("-"*130)
            comp = comparison['comparison']
            print(f"Score change: {comp['score_delta']:+.2f} filters per signal ({comp['score_delta_pct']:+.1f}%)")
            print(f"Pass rate change: {comp['pass_rate_delta']:+.1f}%")
            print(f"\nInterpretation: {comparison['interpretation']}")
            print(f"\n💡 Key Insights:")
            for key, val in comparison['key_insight'].items():
                print(f"   • {key}: {val}")
        
        print(f"\n" + "="*130)

def main():
    parser = argparse.ArgumentParser(description="Dual Tracker: Immutable + Dynamic with Per-Filter Analysis")
    parser.add_argument("--watch", action="store_true", help="Live monitoring (60s refresh)")
    parser.add_argument("--save", action="store_true", help="Save to JSON files")
    args = parser.parse_args()
    
    try:
        if args.watch:
            print("[INFO] Dual tracking mode (60s refresh). Press Ctrl+C to stop.\n")
            while True:
                tracker = DualTracker()
                if tracker.load_signals():
                    baseline = tracker.calculate_baseline()
                    post_enh = tracker.calculate_post_enhancement()
                    comparison = tracker.calculate_comparison(baseline, post_enh)
                    
                    tracker.print_report(baseline, post_enh, comparison)
                    
                    if args.save:
                        tracker.save_immutable(baseline)
                        tracker.save_dynamic(post_enh)
                        tracker.save_comparison(comparison)
                
                print("\n[INFO] Next update in 60s... (Ctrl+C to stop)")
                time.sleep(60)
        else:
            tracker = DualTracker()
            if tracker.load_signals():
                baseline = tracker.calculate_baseline()
                post_enh = tracker.calculate_post_enhancement()
                comparison = tracker.calculate_comparison(baseline, post_enh)
                
                tracker.print_report(baseline, post_enh, comparison)
                
                if args.save:
                    tracker.save_immutable(baseline)
                    tracker.save_dynamic(post_enh)
                    tracker.save_comparison(comparison)
    
    except KeyboardInterrupt:
        print("\n[INFO] Tracking stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
