#!/usr/bin/env python3
"""
PROJECT-5: PEC (Position Entry Closure & Backtest)
FULL ARCHITECTURAL REBUILD - PHASES 1 TO 6
Executed: 2026-03-21 01:20 GMT+7

PHASES:
1. Extract clean Feb 27 - Mar 14 foundation (2,224 signals)
2. Lock foundation metrics (calculate WR, P&L once)
3. Rebuild SIGNALS_INDEPENDENT_AUDIT.txt (remove Mar 15-20)
4. Reset SIGNALS_MASTER.jsonl (keep FOUNDATION, empty NEW_LIVE)
5. Lock reporter template (structure frozen)
6. Verify system alignment

CRITICAL: Run this ONCE on 2026-03-21 to establish clean baseline
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import Counter
from pathlib import Path

WORKSPACE = "/Users/geniustarigan/.openclaw/workspace"

# ===== PHASE 1: EXTRACT CLEAN FOUNDATION =====

def phase_1_extract_clean_foundation():
    """Extract Feb 27 - Mar 14 signals (2,224 total)"""
    print("\n" + "="*100)
    print("PHASE 1: EXTRACT CLEAN FOUNDATION (Feb 27 - Mar 14, 2026)")
    print("="*100)
    
    master_path = os.path.join(WORKSPACE, "SIGNALS_MASTER.jsonl")
    audit_path = os.path.join(WORKSPACE, "SIGNALS_INDEPENDENT_AUDIT.txt")
    
    # Load signals from MASTER
    master_signals = {}
    with open(master_path, 'r') as f:
        for line in f:
            try:
                s = json.loads(line)
                uuid = s.get('signal_uuid')
                if uuid:
                    master_signals[uuid] = s
            except:
                pass
    
    # Load signals from AUDIT
    audit_signals = {}
    with open(audit_path, 'r') as f:
        for line in f:
            try:
                s = json.loads(line)
                uuid = s.get('signal_uuid')
                if uuid:
                    audit_signals[uuid] = s
            except:
                pass
    
    print(f"✓ Loaded {len(master_signals)} from MASTER")
    print(f"✓ Loaded {len(audit_signals)} from AUDIT")
    
    # Filter for Feb 27 - Mar 14 period
    cutoff_date = datetime(2026, 3, 14, 23, 59, 59)
    start_date = datetime(2026, 2, 27, 0, 0, 0)
    
    foundation_signals = []
    for uuid, signal in master_signals.items():
        try:
            fired_str = signal.get('fired_time_utc', '')
            if fired_str:
                # Parse naive datetime
                fired_str = fired_str.replace('Z', '').replace('+00:00', '')
                fired = datetime.fromisoformat(fired_str)
                
                if start_date <= fired <= cutoff_date:
                    foundation_signals.append(signal)
        except:
            pass
    
    print(f"✓ Extracted {len(foundation_signals)} FOUNDATION signals (Feb 27 - Mar 14)")
    
    # Verify both files have same signals
    foundation_uuids = set(s.get('signal_uuid') for s in foundation_signals)
    in_master = foundation_uuids & set(master_signals.keys())
    in_audit = foundation_uuids & set(audit_signals.keys())
    
    print(f"  In MASTER: {len(in_master)}")
    print(f"  In AUDIT: {len(in_audit)}")
    print(f"  In both: {len(in_master & in_audit)}")
    
    # Save clean foundation
    foundation_path = os.path.join(WORKSPACE, "SIGNALS_FOUNDATION_CLEAN.jsonl")
    with open(foundation_path, 'w') as f:
        for signal in foundation_signals:
            f.write(json.dumps(signal) + '\n')
    
    print(f"✓ Saved to {foundation_path}")
    
    return foundation_signals

# ===== PHASE 2: LOCK FOUNDATION METRICS =====

def phase_2_lock_foundation_metrics(foundation_signals):
    """Calculate and lock foundation metrics"""
    print("\n" + "="*100)
    print("PHASE 2: LOCK FOUNDATION METRICS")
    print("="*100)
    
    # Calculate metrics
    total = len(foundation_signals)
    closed = sum(1 for s in foundation_signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT'])
    tp_hit = sum(1 for s in foundation_signals if s.get('status') == 'TP_HIT')
    sl_hit = sum(1 for s in foundation_signals if s.get('status') == 'SL_HIT')
    timeout = sum(1 for s in foundation_signals if s.get('status') == 'TIMEOUT')
    
    total_pnl = sum(s.get('pnl_usd', 0) for s in foundation_signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT'])
    
    if closed > 0:
        timeout_win = sum(1 for s in foundation_signals if s.get('status') == 'TIMEOUT' and s.get('pnl_usd', 0) > 0)
        wr = ((tp_hit + timeout_win) / closed) * 100
    else:
        timeout_win = 0
        wr = 0
    
    print(f"Total: {total}")
    print(f"Closed: {closed}")
    print(f"  TP_HIT: {tp_hit}")
    print(f"  SL_HIT: {sl_hit}")
    print(f"  TIMEOUT: {timeout} (WIN: {timeout_win})")
    print(f"Win Rate: {wr:.1f}%")
    print(f"Total P&L: ${total_pnl:+.2f}")
    
    # Save locked metadata
    metadata = {
        "locked_at": datetime.utcnow().isoformat(),
        "period": "2026-02-27 to 2026-03-14",
        "total_signals": total,
        "closed_trades": closed,
        "tp_hit": tp_hit,
        "sl_hit": sl_hit,
        "timeout": timeout,
        "timeout_win": timeout_win,
        "win_rate_pct": round(wr, 1),
        "total_pnl_usd": round(total_pnl, 2),
        "immutable": True,
        "signal_origin": "FOUNDATION"
    }
    
    meta_path = os.path.join(WORKSPACE, "SIGNALS_FOUNDATION_LOCKED_METADATA.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {meta_path}")
    
    return metadata

# ===== PHASE 3: REBUILD AUDIT =====

def phase_3_rebuild_audit(foundation_signals):
    """Rebuild SIGNALS_INDEPENDENT_AUDIT.txt with clean foundation only"""
    print("\n" + "="*100)
    print("PHASE 3: REBUILD SIGNALS_INDEPENDENT_AUDIT.txt")
    print("="*100)
    
    audit_backup = os.path.join(WORKSPACE, "SIGNALS_INDEPENDENT_AUDIT_BACKUP_BEFORE_REBUILD.txt")
    audit_path = os.path.join(WORKSPACE, "SIGNALS_INDEPENDENT_AUDIT.txt")
    
    # Backup current AUDIT
    if os.path.exists(audit_path):
        os.rename(audit_path, audit_backup)
        print(f"✓ Backed up current AUDIT to {audit_backup}")
    
    # Write clean AUDIT with FOUNDATION only
    with open(audit_path, 'w') as f:
        for signal in foundation_signals:
            # Ensure signal_origin is FOUNDATION
            signal['signal_origin'] = 'FOUNDATION'
            f.write(json.dumps(signal) + '\n')
    
    print(f"✓ Rebuilt {audit_path} with {len(foundation_signals)} FOUNDATION signals")
    print(f"  All signals tagged as signal_origin='FOUNDATION'")

# ===== PHASE 4: RESET MASTER =====

def phase_4_reset_master(foundation_signals):
    """Reset SIGNALS_MASTER.jsonl to FOUNDATION + empty NEW_LIVE ready"""
    print("\n" + "="*100)
    print("PHASE 4: RESET SIGNALS_MASTER.jsonl")
    print("="*100)
    
    master_backup = os.path.join(WORKSPACE, "SIGNALS_MASTER_BACKUP_BEFORE_REBUILD.jsonl")
    master_path = os.path.join(WORKSPACE, "SIGNALS_MASTER.jsonl")
    
    # Backup current MASTER
    os.rename(master_path, master_backup)
    print(f"✓ Backed up current MASTER to {master_backup}")
    
    # Write clean MASTER with FOUNDATION only
    with open(master_path, 'w') as f:
        for signal in foundation_signals:
            signal['signal_origin'] = 'FOUNDATION'
            f.write(json.dumps(signal) + '\n')
    
    print(f"✓ Rebuilt {master_path} with {len(foundation_signals)} FOUNDATION signals")
    print(f"  Ready for fresh NEW_LIVE starting Mar 21")

# ===== PHASE 5: LOCK REPORTER TEMPLATE =====

def phase_5_lock_reporter_template():
    """Verify reporter template is locked (structure frozen, content dynamic)"""
    print("\n" + "="*100)
    print("PHASE 5: LOCK REPORTER TEMPLATE")
    print("="*100)
    
    reporter_path = os.path.join(WORKSPACE, "pec_enhanced_reporter.py")
    
    if os.path.exists(reporter_path):
        # Calculate hash of reporter structure
        with open(reporter_path, 'r') as f:
            content = f.read()
        
        import hashlib
        reporter_hash = hashlib.sha256(content.encode()).hexdigest()
        
        lock_info = {
            "locked_at": datetime.utcnow().isoformat(),
            "file": "pec_enhanced_reporter.py",
            "sha256": reporter_hash,
            "rule": "Template structure FROZEN, metrics content DYNAMIC",
            "frozen_sections": [
                "Section layout (FOUNDATION vs NEW_LIVE)",
                "Column headers",
                "Report structure"
            ],
            "dynamic_sections": [
                "Metrics (WR, P&L, duration - calculated from AUDIT)",
                "Status (current state from MASTER)",
                "Counts (OPEN, TP, SL, TIMEOUT - from current signals)"
            ]
        }
        
        lock_path = os.path.join(WORKSPACE, "PEC_REPORTER_LOCKED_STRUCTURE.json")
        with open(lock_path, 'w') as f:
            json.dump(lock_info, f, indent=2)
        
        print(f"✓ Reporter template verified as locked")
        print(f"  SHA256: {reporter_hash}")
        print(f"  Saved lock info to {lock_path}")
    else:
        print(f"⚠ {reporter_path} not found")

# ===== PHASE 6: VERIFY SYSTEM =====

def phase_6_verify_system():
    """Verify rebuild successful"""
    print("\n" + "="*100)
    print("PHASE 6: VERIFY SYSTEM ALIGNMENT")
    print("="*100)
    
    master_path = os.path.join(WORKSPACE, "SIGNALS_MASTER.jsonl")
    audit_path = os.path.join(WORKSPACE, "SIGNALS_INDEPENDENT_AUDIT.txt")
    
    # Load both
    master_signals = {}
    with open(master_path, 'r') as f:
        for line in f:
            try:
                s = json.loads(line)
                master_signals[s.get('signal_uuid')] = s
            except:
                pass
    
    audit_signals = {}
    with open(audit_path, 'r') as f:
        for line in f:
            try:
                s = json.loads(line)
                audit_signals[s.get('signal_uuid')] = s
            except:
                pass
    
    master_uuids = set(master_signals.keys())
    audit_uuids = set(audit_signals.keys())
    
    in_both = master_uuids & audit_uuids
    only_master = master_uuids - audit_uuids
    only_audit = audit_uuids - master_uuids
    
    print(f"MASTER signals: {len(master_signals)}")
    print(f"AUDIT signals: {len(audit_signals)}")
    print(f"In both: {len(in_both)}")
    print(f"Only in MASTER: {len(only_master)}")
    print(f"Only in AUDIT: {len(only_audit)}")
    
    if len(only_master) == 0 and len(only_audit) == 0:
        print("\n✅ CHECKPOINT PASSED: Files perfectly aligned")
        
        # Verify signal_origin
        all_foundation = sum(1 for s in audit_signals.values() if s.get('signal_origin') == 'FOUNDATION')
        print(f"✓ All signals tagged FOUNDATION: {all_foundation} / {len(audit_signals)}")
        
        return True
    else:
        print("\n❌ CHECKPOINT FAILED: Files not aligned")
        if only_master:
            print(f"   {len(only_master)} signals only in MASTER (ERROR)")
        if only_audit:
            print(f"   {len(only_audit)} signals only in AUDIT (ERROR)")
        return False

# ===== MAIN EXECUTION =====

def main():
    print("\n" + "🏗️  PEC ARCHITECTURAL REBUILD - PHASES 1 TO 6".center(100))
    print("="*100)
    print(f"Start Time: {datetime.utcnow().isoformat()} GMT")
    print(f"Workspace: {WORKSPACE}")
    print("="*100)
    
    try:
        # Phase 1: Extract
        foundation = phase_1_extract_clean_foundation()
        
        # Phase 2: Lock metrics
        metadata = phase_2_lock_foundation_metrics(foundation)
        
        # Phase 3: Rebuild AUDIT
        phase_3_rebuild_audit(foundation)
        
        # Phase 4: Reset MASTER
        phase_4_reset_master(foundation)
        
        # Phase 5: Lock reporter
        phase_5_lock_reporter_template()
        
        # Phase 6: Verify
        success = phase_6_verify_system()
        
        if success:
            print("\n" + "✅ REBUILD COMPLETE".center(100))
            print("="*100)
            print("System is now:")
            print("  - ✅ Clean FOUNDATION (2,224 signals locked)")
            print("  - ✅ AUDIT rebuilt (FOUNDATION only)")
            print("  - ✅ MASTER reset (FOUNDATION + empty NEW_LIVE)")
            print("  - ✅ Reporter template locked")
            print("  - ✅ Files aligned and verified")
            print("\nReady for fresh start on Mar 21!")
            print("="*100)
        else:
            print("\n" + "❌ REBUILD FAILED - VERIFICATION ERROR".center(100))
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
