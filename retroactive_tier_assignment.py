#!/usr/bin/env python3
"""
RETROACTIVE TIER ASSIGNMENT
Assigns tier field to all signals that lack it (Mar 25-26 signals)

Purpose:
- Populate tier field for historical signals using get_signal_tier()
- Write updated signals back to SIGNALS_MASTER.jsonl
- Enable pec_enhanced_reporter to breakdown by tier status

Process:
1. Load SIGNALS_MASTER.jsonl
2. For each signal without tier: get_signal_tier() from combo
3. Write updated signal back
4. Create audit log

Result: All signals (past + present) have tier field populated
"""

import json
import os
from pathlib import Path
from datetime import datetime
import sys

# Import tier lookup
sys.path.insert(0, '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main')
from tier_lookup import get_signal_tier


def load_signals_master():
    """Load SIGNALS_MASTER.jsonl"""
    signals = []
    filepath = Path.home() / '.openclaw/workspace/SIGNALS_MASTER.jsonl'
    
    if not filepath.exists():
        print(f"ERROR: {filepath} not found")
        return []
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                signals.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    
    return signals


def get_tier_for_signal(signal):
    """Get tier for a signal using its combo"""
    try:
        tf = signal.get('timeframe', '')
        direction = signal.get('direction', '')
        route = signal.get('route', '')
        regime = signal.get('regime', '')
        alts_type = signal.get('alts_type', '')
        
        # Call tier lookup
        tier = get_signal_tier(
            timeframe=tf,
            direction=direction,
            route=route,
            regime=regime,
            symbol_group=alts_type
        )
        
        return tier
    except Exception as e:
        print(f"[ERROR] Failed to get tier for signal: {e}")
        return "Tier-X"


def assign_tiers_retroactively(signals):
    """Assign tiers to signals that lack them"""
    updated = 0
    skipped = 0
    audit = []
    
    print("  [Optimized] Loading tier lookup once...")
    from tier_lookup import get_tier_lookup
    tier_lookup = get_tier_lookup()
    
    for i, signal in enumerate(signals):
        tier = signal.get('tier')
        fired_date = signal.get('fired_time_utc', '')[:10]
        
        # Skip if already has tier
        if tier and tier != 'None':
            skipped += 1
            continue
        
        # Get tier from combo (using pre-loaded lookup, no reload)
        try:
            tf = signal.get('timeframe', '')
            direction = signal.get('direction', '')
            route = signal.get('route', '')
            regime = signal.get('regime', '')
            alts_type = signal.get('alts_type', '')
            
            new_tier = tier_lookup.get_tier(tf, direction, route, regime, alts_type)
        except Exception as e:
            new_tier = "Tier-X"
        
        # Update signal
        signal['tier'] = new_tier
        updated += 1
        
        # Log update (sample every 100)
        if updated % 100 == 0:
            audit.append({
                'signal_uuid': signal.get('signal_uuid')[:12],
                'fired_date': fired_date,
                'status': signal.get('status'),
                'timeframe': signal.get('timeframe'),
                'direction': signal.get('direction'),
                'assigned_tier': new_tier
            })
        
        if updated % 500 == 0:
            print(f"  Processed {i+1:,}/{len(signals):,} signals... ({updated:,} updated, {skipped:,} skipped)")
    
    return signals, updated, skipped, audit


def write_signals_master(signals):
    """Write updated signals back to SIGNALS_MASTER.jsonl"""
    filepath = Path.home() / '.openclaw/workspace/SIGNALS_MASTER.jsonl'
    
    with open(filepath, 'w') as f:
        for signal in signals:
            f.write(json.dumps(signal) + '\n')
    
    print(f"✓ Wrote {len(signals):,} signals to {filepath}")


def main():
    print("=" * 80)
    print("RETROACTIVE TIER ASSIGNMENT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print("=" * 80)
    print()
    
    print("Step 1: Loading SIGNALS_MASTER.jsonl...")
    signals = load_signals_master()
    print(f"  Loaded {len(signals):,} signals")
    print()
    
    print("Step 2: Assigning tiers to signals without tier field...")
    signals, updated, skipped, audit = assign_tiers_retroactively(signals)
    print(f"  Updated: {updated:,} signals")
    print(f"  Skipped: {skipped:,} signals (already have tier)")
    print()
    
    print("Step 3: Writing updated signals back to SIGNALS_MASTER.jsonl...")
    write_signals_master(signals)
    print()
    
    print("Step 4: Creating audit log...")
    audit_path = Path.home() / '.openclaw/workspace/retroactive_tier_assignment_audit.json'
    with open(audit_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_signals': len(signals),
            'updated': updated,
            'skipped': skipped,
            'sample_updates': audit
        }, f, indent=2)
    print(f"  Audit log: {audit_path}")
    print()
    
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Verify tier field is now populated in SIGNALS_MASTER.jsonl")
    print("2. Add dimensional breakdown to pec_enhanced_reporter: 'BY TIER STATUS'")
    print("3. pec_enhanced_reporter will now answer Q1 & Q2 automatically")
    print()


if __name__ == '__main__':
    main()
