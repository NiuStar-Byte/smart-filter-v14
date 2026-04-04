#!/usr/bin/env python3
"""
CHECK_AUTHORITATIVE_TIER_SOURCE.py

Micro-check for system health monitoring:
Verify that SIGNAL_TIERS_APPEND.jsonl (from manual_daily_combo_refresh) is current
This is the AUTHORITATIVE source for today's tier assignments
"""

import json
import os
from datetime import datetime, timedelta
import sys

def check_tier_file_freshness():
    """
    Verify that tier file was generated TODAY from manual_daily_combo_refresh.py
    If stale, system must regenerate to avoid using yesterday's tiers
    """
    
    tier_file = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/SIGNAL_TIERS_APPEND.jsonl"
    
    # Check file exists
    if not os.path.exists(tier_file):
        print(f"❌ CRITICAL: {tier_file} NOT FOUND")
        print(f"   Action: Run manual_daily_combo_refresh.py to generate fresh tiers")
        return False
    
    # Check file size (should be 20KB+)
    file_size = os.path.getsize(tier_file)
    if file_size < 5000:
        print(f"❌ CRITICAL: Tier file too small ({file_size} bytes, expected >5KB)")
        print(f"   This indicates corrupted or incomplete tier data")
        return False
    
    # Check file modification time
    mod_time = os.path.getmtime(tier_file)
    mod_datetime = datetime.fromtimestamp(mod_time)
    now = datetime.now()
    age_minutes = (now - mod_datetime).total_seconds() / 60
    
    # Threshold: tier file should be regenerated daily (max 24 hours old)
    max_age_minutes = 24 * 60
    
    if age_minutes > max_age_minutes:
        print(f"⚠️  WARNING: Tier file is {age_minutes:.0f} minutes old")
        print(f"   Expected refresh: Every 24 hours")
        print(f"   Action: Run manual_daily_combo_refresh.py")
        return False
    
    # Check metadata: valid_from date matches today
    try:
        with open(tier_file, 'r') as f:
            metadata = json.loads(f.readline())
        
        valid_from = metadata.get('valid_from', 'UNKNOWN')
        tier1_count = len(metadata.get('tier1', []))
        tier2_count = len(metadata.get('tier2', []))
        tier3_count = len(metadata.get('tier3', []))
        
        print(f"✅ Tier file healthy:")
        print(f"   Age: {age_minutes:.0f} minutes")
        print(f"   Valid from: {valid_from}")
        print(f"   Combos: Tier-1={tier1_count}, Tier-2={tier2_count}, Tier-3={tier3_count}")
        
        # Verify metadata has at least SOME tiers
        if tier2_count == 0 and tier3_count == 0:
            print(f"⚠️  WARNING: No Tier-2 or Tier-3 combos found (may be very fresh data)")
            if tier1_count == 0:
                print(f"❌ CRITICAL: All tier levels empty!")
                return False
        
        return True
    
    except Exception as e:
        print(f"❌ ERROR reading tier file: {e}")
        return False

def verify_tier_signals_in_ledger():
    """
    Spot check: Verify that RECENT signals in SIGNALS_MASTER.jsonl have tier field
    Sample last 10 signals - they should all have 'tier' field
    """
    
    signals_file = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"
    
    if not os.path.exists(signals_file):
        print(f"⚠️  Cannot verify signal tiers (file not found)")
        return None
    
    try:
        with open(signals_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"⚠️  No signals in ledger yet")
            return None
        
        # Check last 10 signals
        recent_signals = lines[-10:]
        tier_count = 0
        no_tier_count = 0
        
        for line in recent_signals:
            signal = json.loads(line)
            if 'tier' in signal and signal['tier']:
                tier_count += 1
            else:
                no_tier_count += 1
        
        if tier_count == len(recent_signals):
            print(f"✅ All recent signals have tier field ({tier_count}/{tier_count})")
            return True
        elif tier_count > 0:
            print(f"⚠️  Mixed: Some signals have tier ({tier_count}/{len(recent_signals)})")
            print(f"   Note: Old signals may not have tier, but new ones should")
            return True
        else:
            print(f"❌ CRITICAL: No recent signals have tier field (0/{len(recent_signals)})")
            print(f"   Action: Verify signals_master_writer.py includes 'tier' field")
            return False
    
    except Exception as e:
        print(f"⚠️  Error checking signal tiers: {e}")
        return None

if __name__ == "__main__":
    print("=" * 70)
    print("AUTHORITATIVE TIER SOURCE HEALTH CHECK")
    print("=" * 70)
    print()
    
    print("[1/2] Checking tier file (SIGNAL_TIERS_APPEND.jsonl)...")
    tier_file_ok = check_tier_file_freshness()
    print()
    
    print("[2/2] Checking signal ledger for tier field...")
    signal_tier_ok = verify_tier_signals_in_ledger()
    print()
    
    print("=" * 70)
    if tier_file_ok and signal_tier_ok:
        print("✅ SYSTEM HEALTHY: Tiers from authoritative source")
        sys.exit(0)
    elif tier_file_ok is False or signal_tier_ok is False:
        print("❌ CRITICAL ISSUE DETECTED")
        sys.exit(1)
    else:
        print("⚠️  WARNING: Could not fully verify (check manually)")
        sys.exit(2)
