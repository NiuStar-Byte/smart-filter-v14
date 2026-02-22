#!/usr/bin/env python3
"""
validate_batch1_ready.py

Validates that all components are ready for Batch 1 (50-signal backtest).
Checks:
1. signal_store.py - JSONL storage working
2. pec_config.py - Configuration loaded correctly
3. pec_engine.py - Modified to accept dynamic parameters
4. pec_backtest_v2.py - JSONL-based backtest ready
5. signals_fired.jsonl - Signals being captured
"""

import os
import sys
import json
from datetime import datetime, timezone

def check_imports():
    """Verify all modules can be imported."""
    print("\n[CHECK 1] Module Imports...")
    errors = []
    
    modules = [
        'pec_config',
        'signal_store',
        'pec_engine',
        'pec_backtest_v2'
    ]
    
    for mod in modules:
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except ImportError as e:
            print(f"  ✗ {mod}: {str(e)}")
            errors.append(f"{mod}: {str(e)}")
    
    return len(errors) == 0, errors


def check_config():
    """Verify PEC config is correct."""
    print("\n[CHECK 2] PEC Configuration...")
    try:
        from pec_config import MIN_ACCEPTED_RR, MAX_BARS_BY_TF, SIGNALS_JSONL_PATH
        
        print(f"  ✓ MIN_ACCEPTED_RR = {MIN_ACCEPTED_RR}")
        print(f"  ✓ MAX_BARS_BY_TF = {MAX_BARS_BY_TF}")
        print(f"  ✓ SIGNALS_JSONL_PATH = {SIGNALS_JSONL_PATH}")
        
        # Verify config values are sensible
        errors = []
        if MIN_ACCEPTED_RR < 1.0:
            errors.append("MIN_ACCEPTED_RR < 1.0")
        if MAX_BARS_BY_TF.get("15min", 0) < 5:
            errors.append("MAX_BARS for 15min too low")
        
        if errors:
            print(f"  ⚠ Config warnings: {', '.join(errors)}")
            return True, errors  # Warnings, not blocking
        
        return True, []
    except Exception as e:
        print(f"  ✗ Config load failed: {str(e)}")
        return False, [str(e)]


def check_signal_store():
    """Verify signal store can write/read."""
    print("\n[CHECK 3] Signal Store (JSONL)...")
    try:
        from signal_store import SignalStore
        import tempfile
        
        # Create test store in temp directory
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as tmp:
            tmp_path = tmp.name
        
        store = SignalStore(tmp_path)
        
        # Write test signal
        test_signal = {
            "uuid": "test-uuid-123",
            "symbol": "BTC-USDT",
            "timeframe": "15min",
            "signal_type": "LONG",
            "fired_time_utc": datetime.now(timezone.utc).isoformat(),
            "entry_price": 42500.00,
            "tp_target": 42750.00,
            "sl_target": 42420.00,
            "tp_pct": 0.588,
            "sl_pct": -0.188,
            "achieved_rr": 3.125,
            "score": 18,
            "confidence": 90.0,
            "route": "LONG_CONFIRMED"
        }
        
        store.append_signal(test_signal)
        
        # Read back
        signals = store.load_all_signals()
        if signals and signals[-1].get('uuid') == 'test-uuid-123':
            print(f"  ✓ Signal storage working (wrote & read back test signal)")
            os.unlink(tmp_path)
            return True, []
        else:
            print(f"  ✗ Could not verify signal write/read")
            os.unlink(tmp_path)
            return False, ["Write/read verification failed"]
    
    except Exception as e:
        print(f"  ✗ Signal store error: {str(e)}")
        return False, [str(e)]


def check_pec_engine():
    """Verify pec_engine.py has dynamic parameters."""
    print("\n[CHECK 4] PEC Engine (Dynamic Parameters)...")
    try:
        from pec_engine import find_realistic_exit
        import inspect
        
        # Check function signature
        sig = inspect.signature(find_realistic_exit)
        params = list(sig.parameters.keys())
        
        required = ['max_bars', 'tp_price', 'sl_price']
        found = [p for p in required if p in params]
        
        if len(found) == len(required):
            print(f"  ✓ find_realistic_exit has dynamic parameters: {found}")
            return True, []
        else:
            missing = set(required) - set(found)
            print(f"  ✗ Missing parameters: {missing}")
            return False, [f"Missing parameters: {missing}"]
    
    except Exception as e:
        print(f"  ✗ PEC engine check failed: {str(e)}")
        return False, [str(e)]


def check_pec_backtest_v2():
    """Verify pec_backtest_v2.py exists and loads JSONL."""
    print("\n[CHECK 5] PEC Backtest V2 (JSONL)...")
    try:
        from pec_backtest_v2 import load_signals_from_jsonl, run_pec_backtest_v2
        import inspect
        
        # Check function signatures
        sig1 = inspect.signature(load_signals_from_jsonl)
        sig2 = inspect.signature(run_pec_backtest_v2)
        
        print(f"  ✓ load_signals_from_jsonl function exists")
        print(f"  ✓ run_pec_backtest_v2 function exists")
        
        return True, []
    
    except Exception as e:
        print(f"  ✗ PEC backtest V2 check failed: {str(e)}")
        return False, [str(e)]


def check_signal_capture():
    """Check if signals are being captured to signals_fired.jsonl."""
    print("\n[CHECK 6] Signal Capture (monitoring)...")
    try:
        from pec_config import SIGNALS_JSONL_PATH
        
        if os.path.exists(SIGNALS_JSONL_PATH):
            with open(SIGNALS_JSONL_PATH, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            
            signal_count = len(lines)
            if signal_count > 0:
                print(f"  ✓ {SIGNALS_JSONL_PATH} exists with {signal_count} signals")
                
                # Parse first signal for validation
                try:
                    first_sig = json.loads(lines[0])
                    print(f"    Sample: {first_sig.get('symbol')} {first_sig.get('timeframe')} @ {first_sig.get('fired_time_utc')}")
                except:
                    pass
                
                return True, []
            else:
                print(f"  ⚠ {SIGNALS_JSONL_PATH} exists but is empty (signals not yet firing)")
                return True, ["Signals not yet firing - this is normal on first run"]
        else:
            print(f"  ⚠ {SIGNALS_JSONL_PATH} not found (signals not yet firing)")
            return True, ["Signals not yet firing - this is expected initially"]
    
    except Exception as e:
        print(f"  ✗ Signal capture check failed: {str(e)}")
        return False, [str(e)]


def main():
    """Run all validation checks."""
    print("="*60)
    print("BATCH 1 READINESS VALIDATION")
    print("="*60)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    
    checks = [
        ("Imports", check_imports),
        ("Config", check_config),
        ("Signal Store", check_signal_store),
        ("PEC Engine", check_pec_engine),
        ("PEC Backtest V2", check_pec_backtest_v2),
        ("Signal Capture", check_signal_capture),
    ]
    
    results = []
    for name, check_func in checks:
        passed, errors = check_func()
        results.append((name, passed, errors))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    blocking_errors = []
    warnings = []
    
    for name, passed, errors in results:
        if passed:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            blocking_errors.extend(errors)
    
    print("\n" + "="*60)
    
    if blocking_errors:
        print("❌ BATCH 1 NOT READY - Blocking Issues:")
        for err in blocking_errors:
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("✅ BATCH 1 READY TO GO!")
        print("\nNext steps:")
        print("1. Confirm signals are firing to signals_fired.jsonl")
        print("2. Accumulate 50 signals")
        print("3. Run: python3 -m pec_backtest_v2")
        print("\nMonitor: tail -f signals_fired.jsonl")
        sys.exit(0)


if __name__ == "__main__":
    main()
