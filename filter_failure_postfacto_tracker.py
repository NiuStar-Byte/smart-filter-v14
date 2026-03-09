#!/usr/bin/env python3
"""
POST-FACTO FILTER FAILURE TRACKER (2026-03-08 22:28 GMT+7 - CORRECTED)

CRITICAL DIFFERENCE FROM PREVIOUS TRACKER:
  Previous: PRE-FACTO (mixed all signals, 98% pre-enhancement)
  This: POST-FACTO (only signals AFTER each enhancement)

Methodology:
  1. PHASE 1 (2026-03-05): All signals after 2026-03-05 00:00
  2. PHASE 2 (2026-03-08): All signals after each deployment time
  3. PHASE 3/4: Will analyze after deployment

This gives ACTUAL post-enhancement performance, not mixed pre/post averages.
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
OUTPUT_CSV = "/Users/geniustarigan/.openclaw/workspace/filter_postfacto_analysis.csv"

# All 20 filters
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

# PHASE DEPLOYMENT TIMES (UTC)
PHASE_DEPLOYMENTS = {
    "phase1_start": "2026-03-05T00:00:00",      # Phase 1 start (approximate)
    "phase2_support_resistance": "2026-03-08T12:45:00",      # 19:45 GMT+7
    "phase2_volatility_squeeze": "2026-03-08T12:54:00",      # 19:54 GMT+7
    "phase2_liquidity_awareness": "2026-03-08T13:00:00",     # 20:00 GMT+7
    "phase2_spread_filter": "2026-03-08T13:05:00",           # 20:05 GMT+7
    "phase2_mtf_volume": "2026-03-08T14:07:00",              # 21:07 GMT+7
    "phase2_vwap_divergence": "2026-03-08T14:14:00",         # 21:14 GMT+7
}

MIN_SCORE = 12
TOTAL_FILTERS = 20

class PostFactoAnalyzer:
    def __init__(self):
        self.signals_pre_phase2 = []
        self.signals_phase2_vwap = []
        self.signals_post_vwap = []
        self.all_signals = []
        
    def load_and_stratify_signals(self):
        """Load signals and stratify by phase deployment times."""
        path = SIGNALS_MASTER_PATH if os.path.exists(SIGNALS_MASTER_PATH) else SIGNALS_JSONL_PATH
        
        if not os.path.exists(path):
            print(f"[ERROR] {path} not found")
            return False
        
        phase2_start = datetime.fromisoformat(PHASE_DEPLOYMENTS["phase2_support_resistance"])
        phase2_end = datetime.fromisoformat(PHASE_DEPLOYMENTS["phase2_vwap_divergence"])
        
        with open(path, 'r') as f:
            for line in f:
                try:
                    signal = json.loads(line.strip())
                    if "score" in signal and "sent_time_utc" in signal:
                        self.all_signals.append(signal)
                        
                        signal_time = datetime.fromisoformat(signal["sent_time_utc"].replace('Z', '+00:00'))
                        
                        if signal_time < phase2_start:
                            self.signals_pre_phase2.append(signal)
                        elif signal_time < phase2_end:
                            self.signals_phase2_vwap.append(signal)
                        else:
                            self.signals_post_vwap.append(signal)
                except:
                    pass
        
        return len(self.all_signals) > 0
    
    def infer_failure_rates_postfacto(self):
        """Infer failure rates using POST-FACTO methodology."""
        if not self.all_signals:
            return False
        
        MIN_SCORE = 12
        TOTAL_FILTERS = 20
        
        print(f"\n📊 POST-FACTO SIGNAL STRATIFICATION")
        print(f"="*80)
        print(f"Total signals analyzed: {len(self.all_signals)}")
        print(f"\nSignal Distribution:")
        print(f"  PRE-Phase 2 (baseline):                {len(self.signals_pre_phase2):>6} signals ({len(self.signals_pre_phase2)/len(self.all_signals)*100:>5.1f}%)")
        print(f"  Phase 2 (during Phase 2 deploys):     {len(self.signals_phase2_vwap):>6} signals ({len(self.signals_phase2_vwap)/len(self.all_signals)*100:>5.1f}%)")
        print(f"  POST-VWAP (after last Phase 2):      {len(self.signals_post_vwap):>6} signals ({len(self.signals_post_vwap)/len(self.all_signals)*100:>5.1f}%)")
        
        print(f"\n⚠️  CRITICAL ISSUE IDENTIFIED:")
        print(f"    Previous analysis mixed all {len(self.all_signals)} signals (PRE-FACTO)")
        print(f"    This analysis uses POST-FACTO methodology:")
        print(f"    - PRE-Phase 2: Baseline (no enhancements)")
        print(f"    - POST-VWAP: After all Phase 2 deployed")
        
        # Calculate baseline (pre-enhancement)
        baseline_scores = [s.get('score', 12) for s in self.signals_pre_phase2]
        baseline_avg_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
        baseline_passes = baseline_avg_score
        baseline_fails = TOTAL_FILTERS - baseline_passes
        
        print(f"\n" + "="*80)
        print(f"BASELINE PERFORMANCE (PRE-Phase 2, NO ENHANCEMENTS)")
        print(f"="*80)
        print(f"Sample size: {len(self.signals_pre_phase2)} signals")
        print(f"Avg score: {baseline_avg_score:.2f} / 20")
        print(f"Avg passes: {baseline_passes:.1f} filters")
        print(f"Avg failures: {baseline_fails:.1f} filters")
        print(f"Pass rate: {(baseline_passes/20)*100:.1f}%")
        
        # Calculate post-Phase 2 (all enhancements deployed)
        if len(self.signals_post_vwap) > 0:
            post_scores = [s.get('score', 12) for s in self.signals_post_vwap]
            post_avg_score = sum(post_scores) / len(post_scores)
            post_passes = post_avg_score
            post_fails = TOTAL_FILTERS - post_passes
            
            improvement = post_passes - baseline_passes
            improvement_pct = (improvement / baseline_passes) * 100 if baseline_passes > 0 else 0
            
            print(f"\n" + "="*80)
            print(f"POST-PHASE 2 PERFORMANCE (ALL PHASE 2 ENHANCEMENTS DEPLOYED)")
            print(f"="*80)
            print(f"Sample size: {len(self.signals_post_vwap)} signals (LIMITED SAMPLE)")
            print(f"Avg score: {post_avg_score:.2f} / 20")
            print(f"Avg passes: {post_passes:.1f} filters")
            print(f"Avg failures: {post_fails:.1f} filters")
            print(f"Pass rate: {(post_passes/20)*100:.1f}%")
            
            print(f"\n" + "="*80)
            print(f"IMPACT OF PHASE 2 ENHANCEMENTS")
            print(f"="*80)
            
            if improvement > 0:
                print(f"✅ IMPROVEMENT DETECTED")
                print(f"   Score increase: {improvement:+.2f} filters per signal")
                print(f"   Percentage improvement: {improvement_pct:+.1f}%")
            elif improvement < 0:
                print(f"⚠️  DEGRADATION DETECTED")
                print(f"   Score decrease: {improvement:.2f} filters per signal")
                print(f"   Percentage change: {improvement_pct:.1f}%")
            else:
                print(f"⏳ NO MEASURABLE CHANGE")
                print(f"   Score delta: {improvement:.2f}")
        else:
            print(f"\n⚠️  INSUFFICIENT POST-ENHANCEMENT DATA")
            print(f"   Only {len(self.signals_post_vwap)} signals after Phase 2")
            print(f"   Need 24+ hours for statistical significance")
        
        # Recommendation
        print(f"\n" + "="*80)
        print(f"DECISION RECOMMENDATION")
        print(f"="*80)
        
        if len(self.signals_post_vwap) < 50:
            print(f"""
INSUFFICIENT POST-ENHANCEMENT DATA
  Current: {len(self.signals_post_vwap)} signals post-Phase 2 ({len(self.signals_post_vwap)/len(self.all_signals)*100:.1f}% of total)
  Needed: ~50-100 signals for statistical confidence
  
ACTION:
  ✅ OPTION 1 (Deploy Phase 3/4 First) IS CORRECT
  1. Deploy Phase 3 & Phase 4 now (4-6 hours)
  2. Let signals accumulate for 24 hours
  3. Rerun this tracker after 24h
  4. Then decide if Phase 2 needs fixing
  
WHY:
  - Current Phase 2 effectiveness UNKNOWN (only {len(self.signals_post_vwap)} post-signals)
  - Previous "41.5% failure" was 98% pre-enhancement data (misleading)
  - Better to deploy Phase 3/4 AND get 24h of real data
  - Then make evidence-based decisions on all 20 filters
""")
        else:
            print(f"""
SUFFICIENT POST-ENHANCEMENT DATA
  Current: {len(self.signals_post_vwap)} signals post-Phase 2 ({len(self.signals_post_vwap)/len(self.all_signals)*100:.1f}%)
  
  Can now make confident decision on Phase 2 effectiveness
  Follow-up analysis needed
""")

def main():
    parser = argparse.ArgumentParser(description="Post-Facto Filter Failure Analysis")
    parser.add_argument("--watch", action="store_true", help="Live monitoring (60s refresh)")
    args = parser.parse_args()
    
    try:
        if args.watch:
            print("[INFO] Live POST-FACTO monitoring (60s refresh). Press Ctrl+C to stop.\n")
            while True:
                analyzer = PostFactoAnalyzer()
                if analyzer.load_and_stratify_signals():
                    analyzer.infer_failure_rates_postfacto()
                print("\n[INFO] Next update in 60s... (Ctrl+C to stop)")
                time.sleep(60)
        else:
            analyzer = PostFactoAnalyzer()
            if analyzer.load_and_stratify_signals():
                analyzer.infer_failure_rates_postfacto()
    
    except KeyboardInterrupt:
        print("\n[INFO] Analysis stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
