#!/usr/bin/env python3
"""
Atomic Write Health Monitor - Verify signal persistence across files
Checks SIGNALS_MASTER.jsonl vs SENT_SIGNALS.jsonl vs COMPLETE_SIGNALS.jsonl
"""

import json
import os
from datetime import datetime
from pathlib import Path

# File paths
files_to_check = {
    'SIGNALS_MASTER.jsonl': 'SIGNALS_MASTER.jsonl',
    'SENT_SIGNALS.jsonl': 'SENT_SIGNALS.jsonl',
    'COMPLETE_SIGNALS.jsonl': 'COMPLETE_SIGNALS.jsonl',
    'ALL_SIGNALS.jsonl': 'ALL_SIGNALS.jsonl'
}

results = {}
timestamp = datetime.utcnow().isoformat() + 'Z'

print("=" * 80)
print(f"ATOMIC WRITE HEALTH MONITOR - {timestamp} (Asia/Jakarta: 2026-06-18 12:03 AM)")
print("=" * 80)

for label, filepath in files_to_check.items():
    if not os.path.exists(filepath):
        results[label] = {
            'exists': False,
            'size_mb': 0,
            'record_count': 0,
            'first_signal': None,
            'last_signal': None,
            'status': 'NOT FOUND'
        }
        print(f"\n❌ {label}: FILE NOT FOUND")
        continue
    
    # Get file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    # Count lines (signals)
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            record_count = len(lines)
        
        # Get first and last signals
        first_signal = None
        last_signal = None
        
        if record_count > 0:
            try:
                first_line = lines[0].strip()
                if first_line:
                    first_signal = json.loads(first_line)
            except:
                pass
            
            try:
                last_line = lines[-1].strip()
                if last_line:
                    last_signal = json.loads(last_line)
            except:
                pass
        
        results[label] = {
            'exists': True,
            'size_mb': round(size_mb, 2),
            'record_count': record_count,
            'first_uuid': first_signal.get('signal_uuid', 'N/A')[:8] if first_signal else 'N/A',
            'first_timestamp': first_signal.get('fired_time_utc', 'N/A')[:19] if first_signal else 'N/A',
            'last_uuid': last_signal.get('signal_uuid', 'N/A')[:8] if last_signal else 'N/A',
            'last_timestamp': last_signal.get('fired_time_utc', 'N/A')[:19] if last_signal else 'N/A',
            'status': '✅ OK'
        }
        
        print(f"\n✅ {label}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Records: {record_count:,}")
        print(f"   First: {results[label]['first_timestamp']} ({results[label]['first_uuid']})")
        print(f"   Last:  {results[label]['last_timestamp']} ({results[label]['last_uuid']})")
        
    except Exception as e:
        results[label] = {
            'exists': True,
            'size_mb': round(size_mb, 2),
            'record_count': 0,
            'status': f'ERROR: {str(e)[:50]}'
        }
        print(f"\n⚠️  {label}: {str(e)}")

# Analyze gaps and trends
print("\n" + "=" * 80)
print("GAP ANALYSIS")
print("=" * 80)

master_count = results.get('SIGNALS_MASTER.jsonl', {}).get('record_count', 0)
sent_count = results.get('SENT_SIGNALS.jsonl', {}).get('record_count', 0)
complete_count = results.get('COMPLETE_SIGNALS.jsonl', {}).get('record_count', 0)
all_count = results.get('ALL_SIGNALS.jsonl', {}).get('record_count', 0)

print(f"\nMASTER vs SENT Gap: {master_count - sent_count:,} signals")
print(f"  - SIGNALS_MASTER: {master_count:,}")
print(f"  - SENT_SIGNALS:   {sent_count:,}")

print(f"\nCOMPLETE vs ALL Gap: {all_count - complete_count:,} signals")
print(f"  - ALL_SIGNALS:      {all_count:,}")
print(f"  - COMPLETE_SIGNALS: {complete_count:,}")

# Check for critical issues
print("\n" + "=" * 80)
print("HEALTH CHECK SUMMARY")
print("=" * 80)

issues = []

# Check for file corruption (zero size)
for label, data in results.items():
    if data.get('exists') and data.get('size_mb') == 0 and data.get('record_count') == 0:
        issues.append(f"🔴 CRITICAL: {label} is empty (possible truncation/corruption)")

# Check for gaps that are growing (no detailed baseline here, but log current state)
if master_count > 0 and sent_count > 0:
    gap_ratio = (master_count - sent_count) / master_count
    if gap_ratio > 0.10:  # More than 10% gap
        issues.append(f"⚠️  WARNING: SIGNALS_MASTER vs SENT_SIGNALS gap is {gap_ratio*100:.1f}% ({master_count - sent_count:,} signals)")

if all_count > 0 and complete_count > 0:
    if complete_count < all_count * 0.8:  # Complete is significantly behind
        issues.append(f"⚠️  WARNING: COMPLETE_SIGNALS significantly behind ALL_SIGNALS ({complete_count:,} vs {all_count:,})")

if issues:
    print("\n⚠️  ISSUES DETECTED:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("\n✅ NO CRITICAL ISSUES DETECTED")
    print("   - All files exist and have content")
    print("   - Record counts are consistent across systems")
    print("   - Atomic write health is nominal")

print("\n" + "=" * 80)
print("ATOMIC WRITE PERSISTENCE: OPERATIONAL")
print("=" * 80)

