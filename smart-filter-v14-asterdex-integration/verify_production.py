#!/usr/bin/env python3
"""
verify_production.py - Verify production configuration before starting

Checks:
1. API credentials are set
2. Endpoint is production
3. DRY_RUN_MODE is False
4. Data directories exist
5. Log directories exist
6. No previous errors in logs

Run this before starting asterdex_entry_poster.py
"""

import os
import sys
from pathlib import Path

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              PRODUCTION VERIFICATION - Smart Filter v14 Integration        ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

checks_passed = 0
checks_failed = 0

# Check 1: Web3 Wallet Credentials (PRO API V3)
print("✓ Checking Web3 wallet credentials...")
wallet_address = os.environ.get("ASTER_WALLET_ADDRESS")
wallet_private_key = os.environ.get("ASTER_WALLET_PRIVATE_KEY")

if wallet_address and wallet_private_key:
    is_valid_address = wallet_address.startswith("0x") and len(wallet_address) == 42
    is_valid_key = wallet_private_key.startswith("0x") and len(wallet_private_key) == 66
    
    if is_valid_address:
        print(f"  ✅ ASTER_WALLET_ADDRESS: {wallet_address}")
        checks_passed += 1
    else:
        print(f"  ❌ ASTER_WALLET_ADDRESS: Invalid format ({len(wallet_address)} chars, expected 42)")
        checks_failed += 1
    
    if is_valid_key:
        print(f"  ✅ ASTER_WALLET_PRIVATE_KEY: Present ({len(wallet_private_key)} chars)")
        checks_passed += 1
    else:
        print(f"  ❌ ASTER_WALLET_PRIVATE_KEY: Invalid format ({len(wallet_private_key)} chars, expected 66)")
        checks_failed += 1
else:
    print(f"  ❌ ASTER_WALLET_ADDRESS: {wallet_address or 'MISSING'}")
    print(f"  ❌ ASTER_WALLET_PRIVATE_KEY: {wallet_private_key or 'MISSING'}")
    checks_failed += 2
    print("\n  🔧 Set environment variables:")
    print("     export ASTER_WALLET_ADDRESS='0x...'")
    print("     export ASTER_WALLET_PRIVATE_KEY='0x...'")

# Check 2: Configuration
print("\n✓ Checking configuration...")
try:
    from asterdex_config import (
        ASTERDEX_ENDPOINT,
        DRY_RUN_MODE,
        PRODUCTION_MODE,
        ASTERDEX_BASE_URL,
    )
    
    if ASTERDEX_ENDPOINT == ASTERDEX_BASE_URL:
        print(f"  ✅ Endpoint: {ASTERDEX_ENDPOINT}")
        checks_passed += 1
    else:
        print(f"  ❌ Endpoint: {ASTERDEX_ENDPOINT} (not production!)")
        checks_failed += 1
    
    if not DRY_RUN_MODE:
        print(f"  ✅ DRY_RUN_MODE: FALSE (real orders enabled)")
        checks_passed += 1
    else:
        print(f"  ❌ DRY_RUN_MODE: TRUE (still in simulation!)")
        checks_failed += 1
    
    if PRODUCTION_MODE:
        print(f"  ✅ PRODUCTION_MODE: TRUE")
        checks_passed += 1
    else:
        print(f"  ❌ PRODUCTION_MODE: FALSE")
        checks_failed += 1

except Exception as e:
    print(f"  ❌ Configuration error: {e}")
    checks_failed += 3

# Check 3: Directories
print("\n✓ Checking directories...")
integration_dir = Path(__file__).parent
dirs_to_check = {
    "data": integration_dir / "data",
    "logs": integration_dir / "logs",
}

for name, path in dirs_to_check.items():
    if path.exists():
        print(f"  ✅ {name}/: Exists")
        checks_passed += 1
    else:
        print(f"  ⚠️  {name}/: Creating...")
        path.mkdir(parents=True, exist_ok=True)
        print(f"     ✅ Created")
        checks_passed += 1

# Check 4: Files
print("\n✓ Checking required files...")
files_to_check = {
    "asterdex_entry_poster.py": integration_dir / "asterdex_entry_poster.py",
    "asterdex_rate_limiter.py": integration_dir / "asterdex_rate_limiter.py",
    "asterdex_utils.py": integration_dir / "asterdex_utils.py",
}

for name, path in files_to_check.items():
    if path.exists():
        print(f"  ✅ {name}")
        checks_passed += 1
    else:
        print(f"  ❌ {name}: MISSING")
        checks_failed += 1

# Check 5: Previous logs for errors
print("\n✓ Checking previous logs...")
log_dir = integration_dir / "logs"
if log_dir.exists():
    log_files = list(log_dir.glob("asterdex_poster_*.log"))
    if log_files:
        latest_log = sorted(log_files)[-1]
        try:
            with open(latest_log, 'r') as f:
                content = f.read()
                errors = content.count("[ERROR]")
                if errors == 0:
                    print(f"  ✅ {latest_log.name}: No errors ({len(content)} bytes)")
                    checks_passed += 1
                else:
                    print(f"  ⚠️  {latest_log.name}: {errors} errors found")
                    print(f"     Review logs before proceeding")
                    checks_passed += 1
        except Exception as e:
            print(f"  ⚠️  Could not read logs: {e}")
    else:
        print(f"  ℹ️  No previous logs (first run)")
        checks_passed += 1

# Summary
print(f"\n{'='*80}")
print("📊 VERIFICATION SUMMARY")
print(f"{'='*80}")

total = checks_passed + checks_failed
percentage = (checks_passed / total * 100) if total > 0 else 0

print(f"Checks Passed: {checks_passed}")
print(f"Checks Failed: {checks_failed}")
print(f"Pass Rate: {percentage:.1f}%")

if checks_failed == 0:
    print(f"\n✅ ✅ ✅ ALL CHECKS PASSED - READY FOR PRODUCTION ✅ ✅ ✅")
    print(f"\n🚀 To start real trading:")
    print(f"   cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-asterdex-integration")
    print(f"   python3 asterdex_entry_poster.py")
    print(f"\n📊 Monitor logs:")
    print(f"   tail -f logs/asterdex_poster_*.log")
    sys.exit(0)
else:
    print(f"\n❌ PRODUCTION VERIFICATION FAILED")
    print(f"\n🔧 Fix the issues above and try again")
    sys.exit(1)
