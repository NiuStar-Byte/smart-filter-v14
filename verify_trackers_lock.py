#!/usr/bin/env python3
"""
🔒 TRACKERS LOCK VERIFICATION - Daily Hash Check
Ensures no accidental modifications to locked trackers
Run daily: python3 verify_trackers_lock.py
"""

import hashlib
import json
from datetime import datetime

# 🔒 LOCKED TRACKER HASHES (2026-03-05 11:53 GMT+7)
LOCKED_TRACKERS = {
    'COMPARE_AB_TEST_LOCKED.py': '424d239eadb70a1026ab105c9d9602bae44c8257b78a5cebbb12d95e51211ba3',
    'PHASE3_TRACKER.py': 'a64a12b7bb2a32534030e2b4bb37ac8e7350e3a01061646bf13e6475361ab329',
    'track_rr_comparison.py': 'ad27f98315f14cfb38f27ea7b65a99f62ff396da39aa12c2bdcc7a04d9c8e056',
    'pec_enhanced_reporter.py': '02913b861199f99dbd9bfe822d42619e8e4bcc500adcfd5fc3ca46d9981dced2',
}

FOUNDATION_SIGNALS = 853
SENT_SIGNALS_FILE = 'SENT_SIGNALS.jsonl'

def verify_tracker_hash(filename, expected_hash):
    """Verify tracker file hash"""
    try:
        with open(filename, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        
        match = actual_hash == expected_hash
        return match, actual_hash
    except FileNotFoundError:
        return None, None

def verify_foundation_count():
    """Verify FOUNDATION baseline unchanged (853 signals)"""
    try:
        foundation_count = 0
        with open(SENT_SIGNALS_FILE, 'r') as f:
            for line in f:
                try:
                    sig = json.loads(line.strip())
                    if sig.get('signal_origin') == 'FOUNDATION':
                        foundation_count += 1
                except:
                    continue
        
        match = foundation_count == FOUNDATION_SIGNALS
        return match, foundation_count
    except FileNotFoundError:
        return None, None

def main():
    print("\n" + "="*100)
    print("🔒 TRACKERS LOCK VERIFICATION")
    print("="*100)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print()
    
    all_ok = True
    
    # Verify tracker hashes
    print("📊 TRACKER INTEGRITY CHECK")
    print("─"*100)
    
    for tracker, expected_hash in LOCKED_TRACKERS.items():
        match, actual_hash = verify_tracker_hash(tracker, expected_hash)
        
        if match is None:
            status = "❌ MISSING"
            print(f"{tracker:<40} {status}")
            all_ok = False
        elif match:
            print(f"{tracker:<40} ✅ INTACT")
        else:
            print(f"{tracker:<40} ⚠️  MODIFIED!")
            print(f"   Expected: {expected_hash}")
            print(f"   Actual:   {actual_hash}")
            print(f"   ACTION: git checkout {tracker}")
            all_ok = False
    
    print()
    
    # Verify FOUNDATION baseline
    print("📊 FOUNDATION BASELINE CHECK")
    print("─"*100)
    
    match, count = verify_foundation_count()
    
    if match is None:
        print(f"{SENT_SIGNALS_FILE:<40} ❌ NOT FOUND")
        all_ok = False
    elif match:
        print(f"FOUNDATION signals: {count} ✅ INTACT (locked at 853)")
    else:
        print(f"FOUNDATION signals: {count} ❌ CORRUPTED (should be 853)")
        print(f"   ACTION: git checkout {SENT_SIGNALS_FILE}")
        all_ok = False
    
    print()
    print("="*100)
    
    if all_ok:
        print("✅ ALL CHECKS PASSED - Trackers locked & verified")
        print("="*100 + "\n")
        return 0
    else:
        print("❌ VERIFICATION FAILED - Issues detected")
        print("   Run: git checkout <filename> to restore")
        print("="*100 + "\n")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
