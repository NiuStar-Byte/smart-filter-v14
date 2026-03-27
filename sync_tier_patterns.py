#!/usr/bin/env python3
"""
TIER PATTERN SYNCHRONIZATION
=============================
Syncs SIGNAL_TIERS.json with actual tier-qualifying combos from pec_enhanced_reporter
Runs continuously in pec_controller - self-heals quality loop on divergence

Architecture: Always sync = sync going forward = quality signal loop never breaks
"""

import json
import os
import re
from datetime import datetime, timezone, timedelta

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
REPORT_FILE = os.path.join(WORKSPACE, "PEC_ENHANCED_REPORT.txt")
TIERS_FILE = os.path.join(WORKSPACE, "SIGNAL_TIERS.json")
QUALIFYING_FILE = os.path.join(WORKSPACE, "TIER_QUALIFYING_COMBOS.json")
SYNC_LOG_FILE = os.path.join(WORKSPACE, "tier_sync.log")

TIER_THRESHOLDS = {
    1: {"wr": 60, "pnl": 5.50, "trades": 60},
    2: {"wr": 50, "pnl": 3.50, "trades": 50},
    3: {"wr": 40, "pnl": 2.00, "trades": 40},
}


def extract_qualifying_combos():
    """Extract tier-qualifying combos from PEC_ENHANCED_REPORT.txt"""
    tier_1 = []
    tier_2 = []
    tier_3 = []
    
    if not os.path.exists(REPORT_FILE):
        return tier_1, tier_2, tier_3
    
    try:
        with open(REPORT_FILE, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if '✓' not in line or 'WR:' not in line:
                continue
            
            wr_match = re.search(r'WR:\s+([\d\.]+)%', line)
            avg_match = re.search(r'Avg:\s+\$\s*([\d\.\+\-]+)', line)
            closed_match = re.search(r'Closed:\s+(\d+)', line)
            
            if not (wr_match and avg_match and closed_match):
                continue
            
            wr = float(wr_match.group(1))
            avg = float(avg_match.group(1))
            closed = int(closed_match.group(1))
            
            combo_match = re.search(r'✓\s+(.+?)\s+\|', line)
            if not combo_match:
                continue
            
            combo = combo_match.group(1).strip()
            
            if wr >= 60 and avg >= 5.50 and closed >= 60:
                tier_1.append(combo)
            elif wr >= 50 and avg >= 3.50 and closed >= 50:
                tier_2.append(combo)
            elif wr >= 40 and avg >= 2.00 and closed >= 40:
                tier_3.append(combo)
    
    except Exception as e:
        log(f"[ERROR] Failed to extract combos: {e}")
    
    return tier_1, tier_2, tier_3


def load_current_tiers():
    """Load current tier patterns from SIGNAL_TIERS.json"""
    tier_1 = []
    tier_2 = []
    tier_3 = []
    
    if not os.path.exists(TIERS_FILE):
        return tier_1, tier_2, tier_3
    
    try:
        with open(TIERS_FILE, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list) and data:
            latest = data[-1]
            tier_1 = latest.get('tier1', [])
            tier_2 = latest.get('tier2', [])
            tier_3 = latest.get('tier3', [])
    except Exception as e:
        log(f"[ERROR] Failed to load current tiers: {e}")
    
    return tier_1, tier_2, tier_3


def sync_tiers():
    """
    Sync SIGNAL_TIERS.json with proven combos
    Returns: (synced, changes_made)
    """
    log("[SYNC] Starting tier pattern synchronization...")
    
    # Extract proven combos
    new_tier_1, new_tier_2, new_tier_3 = extract_qualifying_combos()
    
    # Load current tiers
    old_tier_1, old_tier_2, old_tier_3 = load_current_tiers()
    
    # Check if sync needed
    t1_changed = set(new_tier_1) != set(old_tier_1)
    t2_changed = set(new_tier_2) != set(old_tier_2)
    t3_changed = set(new_tier_3) != set(old_tier_3)
    
    changes_made = t1_changed or t2_changed or t3_changed
    
    if not changes_made:
        log("[SYNC] No tier pattern changes - already in sync ✓")
        return True, False
    
    log(f"[SYNC] Tier patterns diverged - syncing:")
    log(f"[SYNC]   Tier-1: {len(old_tier_1)} → {len(new_tier_1)} combos")
    log(f"[SYNC]   Tier-2: {len(old_tier_2)} → {len(new_tier_2)} combos")
    log(f"[SYNC]   Tier-3: {len(old_tier_3)} → {len(new_tier_3)} combos")
    
    # Load entire SIGNAL_TIERS.json for update
    try:
        if os.path.exists(TIERS_FILE):
            with open(TIERS_FILE, 'r') as f:
                tiers_data = json.load(f)
        else:
            tiers_data = []
        
        # Ensure it's a list
        if not isinstance(tiers_data, list):
            tiers_data = [tiers_data]
        
        # Get latest entry or create new
        if tiers_data and isinstance(tiers_data[-1], dict):
            latest_entry = tiers_data[-1].copy()
        else:
            latest_entry = {}
        
        # Update tier lists
        latest_entry['tier1'] = new_tier_1
        latest_entry['tier2'] = new_tier_2
        latest_entry['tier3'] = new_tier_3
        latest_entry['synced_at'] = datetime.now(timezone(timedelta(hours=7))).isoformat()
        
        # If list is empty or last entry is different, append new entry
        if not tiers_data or tiers_data[-1] != latest_entry:
            tiers_data.append(latest_entry)
        
        # Write back
        with open(TIERS_FILE, 'w') as f:
            json.dump(tiers_data, f, indent=2)
        
        log("[SYNC] ✅ SIGNAL_TIERS.json updated with proven patterns")
        log(f"[SYNC] New tier composition:")
        log(f"[SYNC]   Tier-1: {len(new_tier_1)} combos")
        log(f"[SYNC]   Tier-2: {len(new_tier_2)} combos")
        log(f"[SYNC]   Tier-3: {len(new_tier_3)} combos")
        
        return True, True
    
    except Exception as e:
        log(f"[SYNC] ❌ Failed to update SIGNAL_TIERS.json: {e}")
        return False, False


def log(message):
    """Write to sync log with timestamp"""
    gmt7_tz = timezone(timedelta(hours=7))
    timestamp = datetime.now(gmt7_tz).strftime("%Y-%m-%d %H:%M:%S GMT+7")
    log_msg = f"[{timestamp}] {message}"
    
    # Print to console
    print(log_msg, flush=True)
    
    # Append to log file
    try:
        with open(SYNC_LOG_FILE, 'a') as f:
            f.write(log_msg + '\n')
    except:
        pass


if __name__ == "__main__":
    import sys
    
    if "--check" in sys.argv:
        # Just check status without syncing
        new_t1, new_t2, new_t3 = extract_qualifying_combos()
        old_t1, old_t2, old_t3 = load_current_tiers()
        
        print(f"\nTier Pattern Status:")
        print(f"  Tier-1: {len(old_t1)} current → {len(new_t1)} proven (delta: {len(new_t1) - len(old_t1)})")
        print(f"  Tier-2: {len(old_t2)} current → {len(new_t2)} proven (delta: {len(new_t2) - len(old_t2)})")
        print(f"  Tier-3: {len(old_t3)} current → {len(new_t3)} proven (delta: {len(new_t3) - len(old_t3)})")
        
        if set(old_t1) == set(new_t1) and set(old_t2) == set(new_t2) and set(old_t3) == set(new_t3):
            print(f"\n✅ SYNCED - No divergence detected")
        else:
            print(f"\n⚠️ DIVERGED - Sync needed")
    else:
        # Perform sync
        synced, changed = sync_tiers()
        exit(0 if synced else 1)
