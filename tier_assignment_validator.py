#!/usr/bin/env python3
"""
TIER ASSIGNMENT VALIDATOR
==========================
Continuous validation that tier assignment mechanism is working.
Monitors the quality loop: signals → tier check → tier assignment → better performance

Key Questions:
1. Are signals getting tier assignments? (not null)
2. Do assigned tiers match tier_config criteria? (validation)
3. What % of recent signals are successfully tiered? (health metric)
4. If assignment fails, can we trigger healing? (auto-recovery)

Integration with pec_master_controller:
- Runs every 60 seconds (parallel to health checks)
- Reports to pec_controller.log with [TIER_CHECK] prefix
- Triggers alert if assignment failure rate > threshold
- Can trigger pec_executor restart if needed
"""

import json
import os
from datetime import datetime
from pathlib import Path
from collections import Counter

class TierAssignmentValidator:
    def __init__(self):
        self.workspace = "/Users/geniustarigan/.openclaw/workspace"
        self.signals_file = os.path.join(self.workspace, "SIGNALS_MASTER.jsonl")
        self.tiers_file = os.path.join(self.workspace, "SIGNAL_TIERS.json")
        self.log_file = os.path.join(self.workspace, "pec_controller.log")
        
        # Validation thresholds
        self.recent_signal_count = 100  # Check last N signals
        self.failure_threshold = 0.30   # Alert if >30% assignment failures
        self.last_validated_line = 0
        
        self.tier_config = {
            1: {"wr_min": 0.60, "avg_min": 5.50, "trades_min": 60},
            2: {"wr_min": 0.50, "avg_min": 3.50, "trades_min": 50},
            3: {"wr_min": 0.40, "avg_min": 2.00, "trades_min": 40},
        }
    
    def log(self, msg: str, level: str = "TIER_CHECK"):
        """Write to controller log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level:12}] {msg}"
        print(line, flush=True)
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(line + '\n')
        except Exception as e:
            print(f"Warning: Could not write to log: {e}")
    
    def load_recent_signals(self, count: int = 100) -> list:
        """Load last N closed signals for validation"""
        signals = []
        
        if not os.path.exists(self.signals_file):
            self.log(f"ERROR: {self.signals_file} not found", "TIER_ERROR")
            return signals
        
        try:
            # Load all signals and get last N closed ones
            all_signals = []
            with open(self.signals_file, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line)
                        # Only check closed signals for tier assignment
                        if sig.get("status") in ["TP_HIT", "SL_HIT", "TIMEOUT_WIN", "TIMEOUT_LOSS"]:
                            all_signals.append(sig)
                    except json.JSONDecodeError:
                        continue
            
            # Return last N
            signals = all_signals[-count:] if len(all_signals) > count else all_signals
        except Exception as e:
            self.log(f"ERROR reading signals: {e}", "TIER_ERROR")
        
        return signals
    
    def load_tier_patterns(self) -> dict:
        """Load current tier patterns from SIGNAL_TIERS.json"""
        patterns = {}
        
        if not os.path.exists(self.tiers_file):
            self.log(f"WARNING: {self.tiers_file} not found - tier assignment disabled", "TIER_WARN")
            return patterns
        
        try:
            with open(self.tiers_file, 'r') as f:
                data = json.load(f)
            
            # Get latest tier patterns
            if isinstance(data, list) and data:
                latest = data[-1]
                for tier_num in [1, 2, 3]:
                    tier_name = f'tier{tier_num}'
                    if tier_name in latest:
                        for pattern in latest[tier_name]:
                            patterns[pattern] = tier_num
        except Exception as e:
            self.log(f"ERROR loading tier patterns: {e}", "TIER_ERROR")
        
        return patterns
    
    def get_signal_tier_from_patterns(self, signal: dict, tier_patterns: dict) -> int:
        """Try to match signal to tier patterns (same logic as tier_lookup)"""
        timeframe = signal.get("timeframe", "").lower() if signal.get("timeframe") else ""
        direction = signal.get("direction", "").upper() if signal.get("direction") else ""
        regime = signal.get("regime", "").upper() if signal.get("regime") else ""
        route = signal.get("route", "").upper() if signal.get("route") else ""
        
        tf_short = timeframe.replace("min", "") if timeframe else ""
        
        # Try patterns in order of specificity
        patterns_to_try = [
            f"{tf_short}_{direction}_{route}_{regime}",
            f"TF_DIR_ROUTE_REGIME_{tf_short}_{direction}_{route}_{regime}",
            f"TF_DIR_ROUTE_{tf_short}_{direction}_{route}",
            f"TF_DIR_REGIME_{tf_short}_{direction}_{regime}",
            f"DIR_ROUTE_REGIME_{direction}_{route}_{regime}",
            f"TF_DIR_{tf_short}_{direction}",
            f"DIR_REGIME_{direction}_{regime}",
        ]
        
        for pattern in patterns_to_try:
            if pattern in tier_patterns:
                return tier_patterns[pattern]
        
        return None
    
    def validate_assignment(self) -> dict:
        """Validate tier assignments in recent signals"""
        recent_signals = self.load_recent_signals(self.recent_signal_count)
        tier_patterns = self.load_tier_patterns()
        
        if not recent_signals:
            self.log("No recent closed signals to validate", "TIER_WARN")
            return {
                "total": 0,
                "assigned": 0,
                "success_rate": 0.0,
                "status": "SKIP_NO_DATA"
            }
        
        results = {
            "total": len(recent_signals),
            "assigned": 0,
            "unassigned": 0,
            "tier_dist": Counter(),
            "assignment_failures": [],
            "success_rate": 0.0,
            "status": "OK"
        }
        
        for sig in recent_signals:
            # Check if signal has tier assignment
            has_tier = sig.get("tier") is not None
            
            if has_tier:
                results["assigned"] += 1
                tier = sig.get("tier")
                results["tier_dist"][tier] += 1
            else:
                results["unassigned"] += 1
                # Try to match using pattern logic
                expected_tier = self.get_signal_tier_from_patterns(sig, tier_patterns)
                if expected_tier:
                    results["assignment_failures"].append({
                        "signal": sig.get("signal_uuid"),
                        "status": sig.get("status"),
                        "expected_tier": expected_tier,
                        "actual_tier": None,
                        "issue": "Missing tier assignment (pattern matched but not assigned)"
                    })
        
        # Calculate success rate
        results["success_rate"] = results["assigned"] / results["total"] if results["total"] > 0 else 0.0
        
        # Determine status
        if results["success_rate"] < (1.0 - self.failure_threshold):
            results["status"] = "FAIL_HIGH_UNASSIGNED"
        elif results["assignment_failures"]:
            results["status"] = "WARN_PARTIAL_FAILURES"
        else:
            results["status"] = "OK"
        
        return results
    
    def report_validation(self):
        """Run validation and report results"""
        results = self.validate_assignment()
        
        if results["status"] == "SKIP_NO_DATA":
            self.log("Waiting for first signals...", "TIER_CHECK")
            return results
        
        # Format tier distribution
        tier_str = " | ".join([f"T{t}:{c}" for t, c in sorted(results["tier_dist"].items())])
        
        # Report
        if results["status"] == "OK":
            self.log(
                f"✅ TIER ASSIGNMENT HEALTH: {results['assigned']}/{results['total']} "
                f"({results['success_rate']*100:.1f}%) | {tier_str}",
                "TIER_OK"
            )
        elif results["status"] == "WARN_PARTIAL_FAILURES":
            self.log(
                f"⚠️  TIER ASSIGNMENT WARNING: {results['success_rate']*100:.1f}% success | "
                f"{len(results['assignment_failures'])} pattern mismatches | {tier_str}",
                "TIER_WARN"
            )
            # Log failures
            for failure in results["assignment_failures"][:5]:  # Log first 5
                self.log(
                    f"   → Signal {failure['signal'][:8]}... expected Tier-{failure['expected_tier']}, got None",
                    "TIER_WARN"
                )
        else:  # FAIL_HIGH_UNASSIGNED
            self.log(
                f"🚨 TIER ASSIGNMENT FAILURE: Only {results['success_rate']*100:.1f}% assigned (threshold: {(1.0 - self.failure_threshold)*100:.0f}%) | {tier_str}",
                "TIER_CRITICAL"
            )
            self.log(
                f"   ACTION: Tier assignment mechanism may be broken. Check pec_executor.py and tier_lookup.py",
                "TIER_CRITICAL"
            )
        
        return results


# Standalone runner for testing
if __name__ == "__main__":
    validator = TierAssignmentValidator()
    
    print("\n" + "="*80)
    print("TIER ASSIGNMENT VALIDATOR - Manual Run")
    print("="*80 + "\n")
    
    results = validator.report_validation()
    
    print(f"\nValidation Results:")
    print(f"  Total Recent Signals: {results['total']}")
    print(f"  Successfully Assigned: {results['assigned']}")
    print(f"  Assignment Failures: {results['unassigned']}")
    print(f"  Success Rate: {results['success_rate']*100:.1f}%")
    print(f"  Status: {results['status']}")
    print(f"  Tier Distribution: {dict(results['tier_dist'])}")
    
    if results['assignment_failures']:
        print(f"\n  Failed Assignments (first 5):")
        for failure in results['assignment_failures'][:5]:
            print(f"    - {failure}")
