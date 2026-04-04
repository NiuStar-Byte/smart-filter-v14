#!/usr/bin/env python3
"""
HEALTH CHECK: Tier Field Completeness Monitor

Validates that recent signals have ALL required tier-matching fields:
- symbol_group (MAIN_BLOCKCHAIN, TOP_ALTS, MID_ALTS, LOW_ALTS)
- confidence_level (HIGH, MID, LOW)
- tier (Tier-1, Tier-2, Tier-3, Tier-X)

Alerts if:
- Missing fields (symbol_group: None/UNKNOWN)
- Incomplete records
- Systematic failures (>5% missing)
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

class TierFieldHealthCheck:
    def __init__(self, signals_file="SIGNALS_MASTER.jsonl"):
        self.signals_file = signals_file
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_checked": 0,
            "complete_records": 0,
            "missing_symbol_group": 0,
            "missing_confidence_level": 0,
            "missing_tier": 0,
            "null_symbol_group": 0,
            "unknown_symbol_group": 0,
            "recent_sample": []
        }
    
    def check_file(self, last_n=100, last_minutes=5):
        """Check last N signals for field completeness, prioritizing recent signals"""
        if not os.path.exists(self.signals_file):
            return {"error": f"{self.signals_file} not found"}
        
        try:
            signals = []
            with open(self.signals_file, 'r') as f:
                for line in f:
                    try:
                        signals.append(json.loads(line.strip()))
                    except:
                        pass
            
            # Filter: only signals from last N minutes (if possible)
            from datetime import datetime, timedelta, timezone
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=last_minutes)
            
            recent_by_time = []
            for signal in signals[-last_n:]:
                try:
                    fired = signal.get('fired_time_utc')
                    if fired:
                        fired_dt = datetime.fromisoformat(fired.replace('Z', '+00:00'))
                        if fired_dt >= cutoff_time:
                            recent_by_time.append(signal)
                except:
                    recent_by_time.append(signal)  # Include if timestamp parsing fails
            
            # Use time-filtered signals, fallback to last_n if filtering finds nothing
            recent_signals = recent_by_time if recent_by_time else signals[-last_n:]
            self.results["total_checked"] = len(recent_signals)
            
            for signal in recent_signals:
                # Check all required fields
                symbol_group = signal.get('symbol_group')
                confidence_level = signal.get('confidence_level')
                tier = signal.get('tier')
                
                # Validate
                is_complete = all([
                    symbol_group is not None,
                    symbol_group not in [None, 'None', 'UNKNOWN'],
                    confidence_level is not None,
                    confidence_level not in [None, 'None'],
                    tier is not None
                ])
                
                if is_complete:
                    self.results["complete_records"] += 1
                else:
                    if symbol_group is None or symbol_group == 'None':
                        self.results["missing_symbol_group"] += 1
                        self.results["null_symbol_group"] += 1
                    if symbol_group == 'UNKNOWN':
                        self.results["unknown_symbol_group"] += 1
                    if confidence_level is None or confidence_level == 'None':
                        self.results["missing_confidence_level"] += 1
                    if tier is None:
                        self.results["missing_tier"] += 1
                
                # Sample recent failures
                if not is_complete and len(self.results["recent_sample"]) < 5:
                    self.results["recent_sample"].append({
                        "symbol": signal.get('symbol'),
                        "timeframe": signal.get('timeframe'),
                        "symbol_group": symbol_group,
                        "confidence_level": confidence_level,
                        "tier": tier,
                        "fired_time": signal.get('fired_time_utc')
                    })
            
            # Calculate completion rate
            completion_rate = (self.results["complete_records"] / self.results["total_checked"] * 100) if self.results["total_checked"] > 0 else 0
            self.results["completion_rate_pct"] = round(completion_rate, 1)
            
            # Health status
            if completion_rate >= 95:
                self.results["status"] = "✅ HEALTHY"
            elif completion_rate >= 80:
                self.results["status"] = "⚠️ DEGRADED"
            else:
                self.results["status"] = "🔴 CRITICAL"
            
            return self.results
        
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    def report(self):
        """Print formatted health report"""
        print("\n" + "="*80)
        print("TIER FIELD HEALTH CHECK")
        print("="*80)
        print(f"Timestamp: {self.results.get('timestamp')}")
        print(f"Status: {self.results.get('status', '?')}")
        print(f"Monitor Window: Last 5 minutes of signals (recent activity)")
        print()
        print("COMPLETION METRICS:")
        print(f"  Total signals checked: {self.results.get('total_checked', 0)} (recent)")
        print(f"  Complete records: {self.results.get('complete_records', 0)}")
        print(f"  Completion rate: {self.results.get('completion_rate_pct', 0)}%")
        print()
        print("FIELD DEFECTS:")
        print(f"  Missing symbol_group: {self.results.get('missing_symbol_group', 0)}")
        print(f"    - Null/None values: {self.results.get('null_symbol_group', 0)}")
        print(f"    - UNKNOWN values: {self.results.get('unknown_symbol_group', 0)}")
        print(f"  Missing confidence_level: {self.results.get('missing_confidence_level', 0)}")
        print(f"  Missing tier: {self.results.get('missing_tier', 0)}")
        
        if self.results.get("recent_sample"):
            print()
            print("RECENT FAILURE SAMPLES (up to 5):")
            for i, sample in enumerate(self.results["recent_sample"], 1):
                print(f"  {i}. {sample['symbol']} {sample['timeframe']}")
                print(f"     - symbol_group: {sample['symbol_group']}")
                print(f"     - confidence_level: {sample['confidence_level']}")
                print(f"     - tier: {sample['tier']}")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    workspace = "/Users/geniustarigan/.openclaw/workspace"
    signals_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")
    
    checker = TierFieldHealthCheck(signals_file)
    results = checker.check_file(last_n=100)
    checker.report()
    
    # Exit with error code if critical
    if results.get("status") == "🔴 CRITICAL":
        sys.exit(1)
    elif results.get("status") == "⚠️ DEGRADED":
        sys.exit(2)
    else:
        sys.exit(0)
