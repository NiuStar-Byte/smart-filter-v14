#!/usr/bin/env python3
"""
FILTER INSTRUMENTATION PATCH (2026-03-08 21:47 GMT+7)

Adds detailed filter evaluation logging to main.py and smart_filter.py to capture:
- Which filters pass/fail for each signal
- Filter score contributions
- Failure reasons/gates

This creates SIGNALS_MASTER_WITH_FILTERS.jsonl with complete filter metadata for analysis.

Usage:
  python3 filter_instrumentation_patch.py --show           # Show patch code
  python3 filter_instrumentation_patch.py --apply          # Apply patch to smart_filter.py
  python3 filter_instrumentation_patch.py --status         # Check if instrumented
"""

import os
import json
import re
import sys
import argparse
from pathlib import Path

SMART_FILTER_PATH = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/smart_filter.py"
MAIN_PATH = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/main.py"

# ===== INSTRUMENTATION PATCH CONTENT =====
SMART_FILTER_PATCH = '''
    # ===== FILTER EVALUATION INSTRUMENTATION (2026-03-08) =====
    # This block captures detailed filter pass/fail for each signal
    # Enables post-hoc analysis to identify bottleneck filters
    
    def _export_filter_evaluation(self, results_dict):
        """
        Export detailed filter evaluation data for tracking.
        results_dict: {"filter_name": (passed: bool, score_contributed: float, reason: str)}
        """
        filter_export = {}
        for filter_name, (passed, score, reason) in results_dict.items():
            filter_export[filter_name] = {
                "passed": bool(passed),
                "score": float(score) if score else 0,
                "reason": str(reason)[:100] if reason else ""
            }
        return filter_export
    
    def evaluate_with_instrumentation(self, direction="LONG"):
        """
        Enhanced evaluate() that captures filter-by-filter results.
        Returns: (is_valid, total_score, max_score, filter_results_dict)
        """
        filter_results = {}  # {filter_name: (passed, score, reason)}
        
        # Evaluate all filters
        try:
            # LONG filters
            if direction == "LONG":
                checks = [
                    ("MACD", self._check_macd),
                    ("Volume Spike", self._check_volume_spike),
                    ("Fractal Zone", self._check_fractal_zone),
                    ("TREND", self._check_trend),
                    ("Momentum", self._check_momentum),
                    ("ATR Momentum Burst", self._check_atr_momentum_burst),
                    ("MTF Volume Agreement", self._check_mtf_volume_agreement),
                    ("HH/LL Trend", self._check_hh_ll_trend),
                    ("Volatility Model", self._check_volatility_model),
                    ("Liquidity Awareness", self._check_liquidity_awareness),
                    ("Volatility Squeeze", self._check_volatility_squeeze),
                    ("Candle Confirmation", self._check_candle_confirmation),
                    ("VWAP Divergence", self._check_vwap_divergence),
                    ("Spread Filter", self._check_spread_filter),
                    ("Chop Zone", self._check_chop_zone),
                    ("Liquidity Pool", self._check_liquidity_pool),
                    ("Support/Resistance", self._check_support_resistance),
                    ("Smart Money Bias", self._check_smart_money_bias),
                    ("Absorption", self._check_absorption),
                    ("Wick Dominance", self._check_wick_dominance),
                ]
            else:  # SHORT
                checks = [
                    ("MACD", self._check_macd_short),
                    ("Volume Spike", self._check_volume_spike),
                    ("Fractal Zone", self._check_fractal_zone_short),
                    ("TREND", self._check_trend_short),
                    ("Momentum", self._check_momentum_short),
                    ("ATR Momentum Burst", self._check_atr_momentum_burst),
                    ("MTF Volume Agreement", self._check_mtf_volume_agreement),
                    ("HH/LL Trend", self._check_hh_ll_trend_short),
                    ("Volatility Model", self._check_volatility_model_short),
                    ("Liquidity Awareness", self._check_liquidity_awareness),
                    ("Volatility Squeeze", self._check_volatility_squeeze_short),
                    ("Candle Confirmation", self._check_candle_confirmation),
                    ("VWAP Divergence", self._check_vwap_divergence_short),
                    ("Spread Filter", self._check_spread_filter),
                    ("Chop Zone", self._check_chop_zone_short),
                    ("Liquidity Pool", self._check_liquidity_pool_short),
                    ("Support/Resistance", self._check_support_resistance_short),
                    ("Smart Money Bias", self._check_smart_money_bias_short),
                    ("Absorption", self._check_absorption_short),
                    ("Wick Dominance", self._check_wick_dominance_short),
                ]
            
            total_score = 0
            for filter_name, check_func in checks:
                try:
                    result = check_func()
                    
                    # Extract passed status and score
                    if isinstance(result, bool):
                        passed = result
                        score = self.filter_weights_long.get(filter_name, 0) if direction == "LONG" else self.filter_weights_short.get(filter_name, 0)
                    elif isinstance(result, tuple) and len(result) >= 1:
                        passed = result[0]
                        score = self.filter_weights_long.get(filter_name, 0) if direction == "LONG" else self.filter_weights_short.get(filter_name, 0)
                    else:
                        passed = bool(result)
                        score = self.filter_weights_long.get(filter_name, 0) if direction == "LONG" else self.filter_weights_short.get(filter_name, 0)
                    
                    if passed:
                        total_score += score
                    
                    filter_results[filter_name] = (passed, score if passed else 0, "pass" if passed else "fail")
                    
                except Exception as e:
                    filter_results[filter_name] = (False, 0, f"error: {str(e)[:50]}")
            
            # Check gatekeeper
            gatekeeper_name = "Candle Confirmation"
            is_gatekeeper_passed = filter_results.get(gatekeeper_name, (True, 0, ""))[0]
            
            # Determine if valid
            is_valid = (total_score >= self.min_score) and is_gatekeeper_passed
            max_score = sum(self.filter_weights_long.values())
            
            return is_valid, total_score, max_score, filter_results
        
        except Exception as e:
            print(f"[ERROR] Filter evaluation instrumentation failed: {str(e)}", flush=True)
            return False, 0, 19, {}
    
    # ===== END INSTRUMENTATION BLOCK =====
'''

MAIN_PATCH = '''
    # ===== IN create_and_store_signal() METHOD ===== 
    # Add this after SmartFilter evaluation:
    
    # Capture detailed filter results (NEW: 2026-03-08)
    is_valid, score, max_score, filter_results = sf.evaluate_with_instrumentation(direction)
    
    # Export filter evaluation to signal metadata
    if filter_results:
        signal_data["filters_evaluated"] = sf._export_filter_evaluation(filter_results)
        signal_data["score_breakdown"] = {
            f: {"passed": passed, "weight": score}
            for f, (passed, score, _) in filter_results.items()
        }
'''

# ===== UTILITIES =====
def show_patch():
    """Display the instrumentation patch."""
    print("\n" + "="*100)
    print("FILTER INSTRUMENTATION PATCH FOR smart_filter.py")
    print("="*100)
    print(SMART_FILTER_PATCH)
    
    print("\n" + "="*100)
    print("FILTER INSTRUMENTATION PATCH FOR main.py (in create_and_store_signal)")
    print("="*100)
    print(MAIN_PATCH)
    
    print("\n" + "="*100)
    print("USAGE INSTRUCTIONS")
    print("="*100)
    print("""
1. Add the SmartFilter instrumentation method to smart_filter.py:
   - Locate the SmartFilter class
   - Add the _export_filter_evaluation() and evaluate_with_instrumentation() methods
   - Keep existing evaluate() method for backward compatibility

2. Update main.py in create_and_store_signal():
   - Replace: is_valid, score, max_score = sf.evaluate(direction)
   - With: is_valid, score, max_score, filter_results = sf.evaluate_with_instrumentation(direction)
   - Then capture: signal_data["filters_evaluated"] = sf._export_filter_evaluation(filter_results)

3. Run filter_failure_tracker.py after 50+ signals:
   python3 filter_failure_tracker.py --watch

4. Monitor which filters have highest failure rates and prioritize next enhancements.
    """)

def check_status():
    """Check if smart_filter.py is already instrumented."""
    if not os.path.exists(SMART_FILTER_PATH):
        print(f"[ERROR] {SMART_FILTER_PATH} not found")
        return False
    
    with open(SMART_FILTER_PATH, 'r') as f:
        content = f.read()
    
    if "evaluate_with_instrumentation" in content:
        print("[✅] smart_filter.py is ALREADY instrumented")
        return True
    else:
        print("[⏳] smart_filter.py is NOT yet instrumented")
        print("\nTo add instrumentation:")
        print("  python3 filter_instrumentation_patch.py --show")
        print("  (then manually add the patch code)")
        return False

def apply_patch():
    """Apply instrumentation patch to smart_filter.py."""
    if not os.path.exists(SMART_FILTER_PATH):
        print(f"[ERROR] {SMART_FILTER_PATH} not found")
        return False
    
    with open(SMART_FILTER_PATH, 'r') as f:
        content = f.read()
    
    if "evaluate_with_instrumentation" in content:
        print("[INFO] Already instrumented, skipping patch")
        return True
    
    # Find insertion point: end of SmartFilter class before final methods
    # Insert before the evaluate() method
    insertion_marker = "    def evaluate("
    
    if insertion_marker not in content:
        print(f"[ERROR] Could not find insertion point in {SMART_FILTER_PATH}")
        return False
    
    # Insert instrumentation code
    new_content = content.replace(
        insertion_marker,
        SMART_FILTER_PATCH + "\n\n" + insertion_marker
    )
    
    # Backup original
    backup_path = SMART_FILTER_PATH + ".backup_2026_03_08"
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"[OK] Backup created: {backup_path}")
    
    # Write patched version
    with open(SMART_FILTER_PATH, 'w') as f:
        f.write(new_content)
    
    print(f"[OK] Instrumentation patch applied to {SMART_FILTER_PATH}")
    return True

# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser(description="Filter Instrumentation Patch Utility")
    parser.add_argument("--show", action="store_true", help="Show patch code")
    parser.add_argument("--apply", action="store_true", help="Apply patch (backup original)")
    parser.add_argument("--status", action="store_true", help="Check instrumentation status")
    
    args = parser.parse_args()
    
    if args.show:
        show_patch()
    elif args.status:
        check_status()
    elif args.apply:
        if apply_patch():
            print("\n[INFO] Next: Restart daemon and monitor with filter_failure_tracker.py")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
