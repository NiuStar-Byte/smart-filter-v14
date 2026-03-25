#!/usr/bin/env python3
"""
verify_rr_fix.py - Verify that the RR calculation fix is in place

Checks:
1. calculations.py uses entry_price (not current_price) for RR calculation
2. pec_config.py has MIN_ACCEPTED_RR and other RR settings correct
3. Cleanup script has been run and extreme RR signals are marked
4. Reporter will exclude extreme RR signals from calculations
"""

import os
import re

WORKSPACE = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main"

def check_calculations_py():
    """Verify calculations.py uses entry_price for RR"""
    print("\n[CHECK 1] Verify calculations.py uses entry_price for RR calculation...")
    
    calc_file = os.path.join(WORKSPACE, "calculations.py")
    if not os.path.exists(calc_file):
        print("  ✗ FAIL: calculations.py not found")
        return False
    
    with open(calc_file, 'r') as f:
        content = f.read()
    
    # Check for the fixed code
    checks = [
        ('reward = tp - entry_price' in content, "LONG: reward = tp - entry_price"),
        ('risk = entry_price - sl' in content, "LONG: risk = entry_price - sl"),
        ('reward = entry_price - tp' in content, "SHORT: reward = entry_price - tp"),
        ('risk = sl - entry_price' in content, "SHORT: risk = sl - entry_price"),
    ]
    
    all_pass = True
    for check, desc in checks:
        if check:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc}")
            all_pass = False
    
    return all_pass

def check_pec_config():
    """Verify pec_config.py has correct RR settings"""
    print("\n[CHECK 2] Verify pec_config.py RR settings...")
    
    config_file = os.path.join(WORKSPACE, "pec_config.py")
    if not os.path.exists(config_file):
        print("  ✗ FAIL: pec_config.py not found")
        return False
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    checks = [
        ('MIN_ACCEPTED_RR' in content, "MIN_ACCEPTED_RR defined"),
        ("1.25" in content, "Fallback RR is 1.25:1"),
        ("MIN_ACCEPTED_RR = float(os.getenv" in content, "MIN_ACCEPTED_RR from environment"),
    ]
    
    all_pass = True
    for check, desc in checks:
        if check:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc}")
            all_pass = False
    
    # Extract the actual value
    match = re.search(r'MIN_ACCEPTED_RR = float\(os\.getenv\("MIN_ACCEPTED_RR", "([^"]+)"\)\)', content)
    if match:
        min_rr = float(match.group(1))
        print(f"  → MIN_ACCEPTED_RR = {min_rr}:1")
    
    return all_pass

def check_cleanup_run():
    """Verify cleanup script has been run"""
    print("\n[CHECK 3] Verify RR cleanup has been run...")
    
    audit_file = os.path.join(WORKSPACE, "cleanup_audit_extreme_rr.log")
    backup_file = os.path.join(WORKSPACE, "SIGNALS_MASTER.jsonl.backup_before_rr_cleanup*")
    
    # Check for backup
    import glob
    backups = glob.glob(backup_file)
    
    if backups:
        print(f"  ✓ Backup created: {os.path.basename(backups[0])}")
    else:
        print(f"  ✗ No backup found (cleanup may not have run)")
    
    # Check for audit trail
    if os.path.exists(audit_file):
        with open(audit_file, 'r') as f:
            audit_content = f.read()
        
        # Extract statistics
        match = re.search(r'Signals Flagged: (\d+)', audit_content)
        if match:
            flagged = int(match.group(1))
            print(f"  ✓ Cleanup completed: {flagged} extreme RR signals marked")
        
        return True
    else:
        print(f"  ✗ No audit trail found (cleanup not run yet)")
        return False

def check_reporter_excludes():
    """Verify reporter excludes EXTREME_RR signals"""
    print("\n[CHECK 4] Verify reporter excludes EXTREME_RR signals...")
    
    reporter_file = os.path.join(WORKSPACE, "pec_enhanced_reporter.py")
    if not os.path.exists(reporter_file):
        print("  ✗ FAIL: pec_enhanced_reporter.py not found")
        return False
    
    with open(reporter_file, 'r') as f:
        content = f.read()
    
    checks = [
        ('EXTREME_RR' in content, "Reporter checks for EXTREME_RR flag"),
        ('STALE_TIMEOUT' in content, "Reporter handles STALE_TIMEOUT"),
        ('data_quality_flag' in content, "Reporter reads data_quality_flag"),
    ]
    
    all_pass = True
    for check, desc in checks:
        if check:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc}")
            all_pass = False
    
    return all_pass

def check_git_commit():
    """Verify fix commit is in history"""
    print("\n[CHECK 5] Verify git commit (1b7067a)...")
    
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'log', '--oneline', '-10'],
            cwd=WORKSPACE,
            capture_output=True,
            text=True
        )
        
        if '1b7067a' in result.stdout or 'entry_price' in result.stdout:
            print(f"  ✓ Fix commit found in history")
            # Show the commit
            commit_result = subprocess.run(
                ['git', 'log', '--oneline', '-1'],
                cwd=WORKSPACE,
                capture_output=True,
                text=True
            )
            if commit_result.stdout:
                print(f"  → Latest: {commit_result.stdout.strip()}")
            return True
        else:
            print(f"  ✗ Fix commit not found (check git log)")
            return False
    except Exception as e:
        print(f"  ✗ Error checking git: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("RR FIX VERIFICATION")
    print("="*80)
    
    checks = [
        check_calculations_py(),
        check_pec_config(),
        check_cleanup_run(),
        check_reporter_excludes(),
        check_git_commit(),
    ]
    
    print("\n" + "="*80)
    if all(checks):
        print("✓ ALL CHECKS PASSED - RR fix is properly implemented!")
        print("="*80 + "\n")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Review above for details")
        print("="*80 + "\n")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
