#!/usr/bin/env python3
"""
validate_signal_completeness.py

FIELD COMPLETENESS AUDITOR
Validates that all signals in COMPLETE_SIGNALS.jsonl have required fields.
Used to detect and prevent missing field issues.

Usage:
    python3 validate_signal_completeness.py
"""

import json
import sys
from datetime import datetime

# === ALL REQUIRED FIELDS (FROM signal_store.py) ===
REQUIRED_FIELDS = {
    # Core identification
    'uuid': str,
    'symbol': str,
    'timeframe': str,
    
    # Signal type & direction
    'signal_type': str,
    'direction': str,
    
    # Timing
    'fired_time_utc': str,
    
    # Price & targets
    'entry_price': (int, float),
    'tp_target': (int, float),
    'sl_target': (int, float),
    'tp_pct': (int, float),
    'sl_pct': (int, float),
    
    # Risk metrics
    'achieved_rr': (int, float),
    'atr_value': (int, float),
    
    # Scoring
    'score': int,
    'max_score': int,
    'confidence': (int, float),
    
    # Analysis
    'route': str,
    'regime': str,
    
    # Gatekeepers
    'passed_gatekeepers': int,
    'max_gatekeepers': int,
    
    # Filters
    'passed_filters': list,
    'failed_filters': list,
    'passed_filter_count': int,
    'failed_filter_count': int,
    
    # Telegram
    'telegram_msg_id': str,
    
    # MTF Analysis
    'mtf_alignment_band': str,
    'mtf_alignment_score': int,
    
    # === CRITICAL: Tier & Signal Classification ===
    'status': str,
    'symbol_group': str,
    'confidence_level': str,
    'tier': str,
    
    # === PEC Executor Fields (may be None initially) ===
    'closed_at': (str, type(None)),
    'actual_exit_price': (int, float, type(None)),
    'pnl_usd': (int, float, type(None)),
}

def validate_signal(signal_dict, line_num):
    """
    Check if signal has all required fields.
    Returns: (is_valid, error_list)
    """
    errors = []
    
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in signal_dict:
            errors.append(f"Line {line_num}: MISSING FIELD '{field}'")
        elif signal_dict[field] is None and expected_type != type(None):
            # None is allowed only for optional fields
            if field not in ['closed_at', 'actual_exit_price', 'pnl_usd', 'fib_ratio']:
                errors.append(f"Line {line_num}: NULL VALUE for '{field}'")
    
    return len(errors) == 0, errors

def main():
    signals_file = "/Users/geniustarigan/.openclaw/workspace/COMPLETE_SIGNALS.jsonl"
    
    print("=" * 80)
    print("FIELD COMPLETENESS VALIDATOR")
    print("=" * 80)
    print(f"\nChecking: {signals_file}")
    print(f"Scan time: {datetime.utcnow().isoformat()} GMT+7\n")
    
    total_signals = 0
    valid_signals = 0
    incomplete_signals = []
    missing_fields_set = set()
    
    try:
        with open(signals_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_signals += 1
                
                try:
                    signal = json.loads(line)
                    is_valid, errors = validate_signal(signal, line_num)
                    
                    if is_valid:
                        valid_signals += 1
                    else:
                        incomplete_signals.append((line_num, signal.get('symbol', 'UNKNOWN'), errors))
                        
                        # Track which fields are missing
                        for error in errors:
                            if "MISSING FIELD" in error:
                                field_name = error.split("'")[1]
                                missing_fields_set.add(field_name)
                
                except json.JSONDecodeError as e:
                    incomplete_signals.append((line_num, 'JSON_ERROR', [f"Parse error: {e}"]))
                    total_signals -= 1  # Don't count parse errors
    
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {signals_file}")
        return 1
    
    # === REPORT ===
    print(f"\n📊 COMPLETENESS REPORT")
    print(f"{'─' * 80}")
    print(f"Total signals analyzed: {total_signals}")
    print(f"✅ Complete signals: {valid_signals} ({100*valid_signals/total_signals:.1f}%)" if total_signals > 0 else "✅ Complete: 0")
    print(f"❌ Incomplete signals: {len(incomplete_signals)}")
    
    if missing_fields_set:
        print(f"\n⚠️  MISSING FIELDS DETECTED:")
        for field in sorted(missing_fields_set):
            print(f"   - {field}")
    
    if incomplete_signals and len(incomplete_signals) <= 20:
        print(f"\n❌ INCOMPLETE SIGNALS (first 20):")
        for line_num, symbol, errors in incomplete_signals[:20]:
            print(f"\n  Line {line_num} ({symbol}):")
            for error in errors:
                print(f"    {error}")
    elif incomplete_signals:
        print(f"\n❌ INCOMPLETE SIGNALS (showing first 5 of {len(incomplete_signals)}):")
        for line_num, symbol, errors in incomplete_signals[:5]:
            print(f"\n  Line {line_num} ({symbol}):")
            for error in errors[:3]:  # Show first 3 errors per signal
                print(f"    {error}")
    
    print(f"\n{'─' * 80}")
    
    if valid_signals == total_signals:
        print("✅ ALL SIGNALS COMPLETE - No missing fields detected!")
        return 0
    else:
        print(f"❌ ISSUES FOUND: {len(incomplete_signals)} signals with missing fields")
        return 1

if __name__ == "__main__":
    sys.exit(main())
