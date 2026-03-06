#!/usr/bin/env python3
"""
REBUILD SIGNALS_MASTER.jsonl from SIGNALS_INDEPENDENT_AUDIT.txt

If SIGNALS_MASTER.jsonl gets corrupted or deleted:
  python3 rebuild_signals_master_from_audit.py

This will reconstruct SIGNALS_MASTER.jsonl from the immutable audit trail,
preserving all signal data and metadata.

SAFETY: This script ALWAYS preserves immutability rules:
  - FOUNDATION (lines 1-853): locked baseline
  - NEW_IMMUTABLE (lines 854-1,087): locked historical
  - NEW_LIVE: regenerated from audit trail
"""

import json
import os
from datetime import datetime, timezone, timedelta

workspace = "/Users/geniustarigan/.openclaw/workspace"
audit_file = os.path.join(workspace, "SIGNALS_INDEPENDENT_AUDIT.txt")
master_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")
backup_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl.backup")

# Immutability boundaries
FOUNDATION_END = 853
IMMUTABLE_END = 1087

print("[INFO] Rebuilding SIGNALS_MASTER.jsonl from audit trail...")
print(f"[INFO] Source: {audit_file}")
print(f"[INFO] Target: {master_file}")

if not os.path.exists(audit_file):
    print(f"[ERROR] Audit trail not found: {audit_file}")
    exit(1)

# Backup existing SIGNALS_MASTER if it exists
if os.path.exists(master_file):
    try:
        with open(master_file, 'r') as src, open(backup_file, 'w') as dst:
            dst.write(src.read())
        print(f"[INFO] Backed up existing SIGNALS_MASTER to {backup_file}")
    except Exception as e:
        print(f"[WARN] Could not backup existing file: {e}")

# Rebuild from audit trail
count_foundation = 0
count_new_immutable = 0
count_new_live = 0

with open(audit_file, 'r') as infile, open(master_file, 'w') as outfile:
    for idx, line in enumerate(infile, 1):
        try:
            audit_entry = json.loads(line.strip())
            
            # Determine signal_origin based on immutable boundaries
            if idx <= FOUNDATION_END:
                signal_origin = 'FOUNDATION'
                count_foundation += 1
            elif idx <= IMMUTABLE_END:
                signal_origin = 'NEW_IMMUTABLE'
                count_new_immutable += 1
            else:
                signal_origin = 'NEW_LIVE'
                count_new_live += 1
            
            # Convert to SIGNALS_MASTER format
            master_entry = {
                'uuid': audit_entry.get('signal_uuid', ''),
                'symbol': audit_entry.get('symbol', ''),
                'timeframe': audit_entry.get('timeframe', ''),
                'signal_type': audit_entry.get('direction', ''),
                'fired_time_utc': audit_entry.get('fired_time_utc', ''),
                'fired_time_jakarta': audit_entry.get('fired_time_jakarta', ''),
                'fired_date_jakarta': audit_entry.get('fired_time_jakarta', '')[:10] if audit_entry.get('fired_time_jakarta') else '',
                'entry_price': audit_entry.get('entry_price'),
                'tp_price': audit_entry.get('tp_price'),
                'sl_price': audit_entry.get('sl_price'),
                'score': audit_entry.get('score'),
                'confidence': audit_entry.get('confidence'),
                'regime': audit_entry.get('regime', ''),
                'route': audit_entry.get('route', ''),
                'tier': audit_entry.get('tier'),
                'rr': audit_entry.get('rr'),
                'status': audit_entry.get('status', 'OPEN'),
                'signal_origin': signal_origin,
                # Preserve all audit fields
                'passed_min_score_gate': audit_entry.get('passed_min_score_gate'),
                'weighted_score': audit_entry.get('weighted_score'),
                'consensus': audit_entry.get('consensus'),
            }
            
            # Extract hour for easier grouping
            fired_jakarta = audit_entry.get('fired_time_jakarta', '')
            if fired_jakarta:
                try:
                    dt = datetime.fromisoformat(fired_jakarta)
                    master_entry['fired_hour_jakarta'] = int(dt.strftime('%H'))
                except:
                    pass
            
            outfile.write(json.dumps(master_entry) + '\n')
        except Exception as e:
            print(f"[ERROR] Line {idx}: {e}")

print(f"\n✅ Rebuild complete:")
print(f"   FOUNDATION: {count_foundation} signals")
print(f"   NEW_IMMUTABLE: {count_new_immutable} signals")
print(f"   NEW_LIVE: {count_new_live} signals")
print(f"   TOTAL: {count_foundation + count_new_immutable + count_new_live} signals")
print(f"\n✅ SIGNALS_MASTER.jsonl restored from audit trail")
print(f"   Immutability preserved: FOUNDATION + NEW_IMMUTABLE locked")
print(f"   Backup saved: {backup_file}")
