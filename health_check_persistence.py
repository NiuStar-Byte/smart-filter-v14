#!/usr/bin/env python3
"""
Enhanced Health Check: Signal Persistence + Field Writing Monitoring

Detects:
1. Missing symbol_group/confidence_level fields
2. Signals sent to Telegram but NOT persisted to SIGNALS_MASTER.jsonl
3. Write failures and file corruption
4. Signal generation vs persistence rate mismatch
"""

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

def check_signal_persistence():
    """Check for signals fired but not persisted"""
    
    workspace = "/Users/geniustarigan/.openclaw/workspace"
    master_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")
    sent_signals_file = os.path.join(workspace, "SENT_SIGNALS.jsonl")
    
    print("=" * 100)
    print("ENHANCED HEALTH CHECK: SIGNAL PERSISTENCE + FIELD WRITING")
    print("=" * 100)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print()
    
    # ===== PART 1: FIELD COMPLETION CHECK =====
    print("PART 1: FIELD COMPLETION CHECK")
    print("-" * 100)
    
    # Get last 100 signals from SIGNALS_MASTER
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    recent_signals = []
    for line in lines[-100:]:
        if not line.strip():
            continue
        try:
            recent_signals.append(json.loads(line))
        except:
            pass
    
    # Check field completion
    field_defects = {
        'symbol_group': 0,
        'confidence_level': 0,
        'tier': 0
    }
    
    samples = []
    for sig in recent_signals:
        if sig.get('symbol_group') is None or sig.get('symbol_group') == 'UNKNOWN':
            field_defects['symbol_group'] += 1
            if len(samples) < 3:
                samples.append({
                    'symbol': sig.get('symbol'),
                    'timeframe': sig.get('timeframe'),
                    'fired_time': sig.get('fired_time_utc', '?')[:19],
                    'field': 'symbol_group',
                    'value': sig.get('symbol_group')
                })
        
        if sig.get('confidence_level') is None or sig.get('confidence_level') == 'UNKNOWN':
            field_defects['confidence_level'] += 1
            if len(samples) < 6:
                samples.append({
                    'symbol': sig.get('symbol'),
                    'timeframe': sig.get('timeframe'),
                    'fired_time': sig.get('fired_time_utc', '?')[:19],
                    'field': 'confidence_level',
                    'value': sig.get('confidence_level')
                })
    
    completion_rate = ((len(recent_signals) - field_defects['symbol_group']) / len(recent_signals) * 100) if recent_signals else 0
    
    print(f"Total signals checked: {len(recent_signals)} (last 100)")
    print(f"Complete records: {len(recent_signals) - field_defects['symbol_group']}")
    print(f"Completion rate: {completion_rate:.1f}%")
    print()
    print("Field Defects:")
    print(f"  Missing symbol_group: {field_defects['symbol_group']}")
    print(f"  Missing confidence_level: {field_defects['confidence_level']}")
    print()
    
    if samples:
        print("Failure Samples:")
        for s in samples[:5]:
            print(f"  {s['symbol']} {s['timeframe']} | {s['fired_time']} | {s['field']}: {s['value']}")
        print()
    
    # ===== PART 2: SIGNAL PERSISTENCE CHECK =====
    print()
    print("PART 2: SIGNAL PERSISTENCE CHECK (Telegram → File)")
    print("-" * 100)
    
    # Check SENT_SIGNALS vs SIGNALS_MASTER
    if not os.path.exists(sent_signals_file):
        print("⚠️  SENT_SIGNALS.jsonl not found - skipping persistence check")
        print()
    else:
        with open(sent_signals_file, 'r') as f:
            sent_lines = f.readlines()
        
        sent_signals = {}
        for line in sent_lines[-500:]:  # Last 500 sent signals
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sig_uuid = record.get('signal_uuid', '')
                sent_signals[sig_uuid] = record
            except:
                pass
        
        # Get all UUIDs from SIGNALS_MASTER
        master_uuids = set()
        with open(master_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    master_uuids.add(json.loads(line).get('signal_uuid', ''))
                except:
                    pass
        
        # Find sent but not persisted
        orphaned = []
        for sig_uuid, sent_record in sent_signals.items():
            if sig_uuid not in master_uuids:
                orphaned.append(sent_record)
        
        print(f"Signals sent (last 500): {len(sent_signals)}")
        print(f"Signals in SIGNALS_MASTER: {len(master_uuids)}")
        print(f"Orphaned signals (sent but NOT persisted): {len(orphaned)}")
        print()
        
        if orphaned:
            print(f"🔴 CRITICAL: {len(orphaned)} signals fired to Telegram but NOT written to file!")
            print()
            print("Recent Orphaned Signals (up to 5):")
            for sig in orphaned[-5:]:
                print(f"  {sig.get('symbol')} {sig.get('timeframe')} {sig.get('signal_type')}")
                print(f"    Fired: {sig.get('fired_time_utc', '?')[:19]} | UUID: {sig.get('signal_uuid', '?')[:12]}")
                print(f"    Route: {sig.get('route')} | Confidence: {sig.get('confidence')}%")
            print()
            print("Possible Causes:")
            print("  1. Multiple main.py processes (file lock conflicts)")
            print("  2. write_signal() exception silently caught")
            print("  3. _master_writer_ready flag False during processing")
            print("  4. File system permission/disk space issues")
            print()
        else:
            print("✅ All sent signals properly persisted to SIGNALS_MASTER.jsonl")
            print()
    
    # ===== PART 3: GENERATION vs PERSISTENCE RATE =====
    print()
    print("PART 3: SIGNAL GENERATION vs PERSISTENCE RATE")
    print("-" * 100)
    
    now = datetime.utcnow()
    window_start = now - timedelta(hours=1)
    
    # Count signals generated in last hour
    recent_master = [s for s in recent_signals if 
        datetime.fromisoformat(s.get('fired_time_utc', '').replace('Z', '+00:00')) >= window_start]
    
    print(f"Signals in SIGNALS_MASTER (last hour): {len(recent_master)}")
    print(f"Expected rate: ~100+ signals/hour")
    
    if len(recent_master) < 50:
        print("⚠️  LOW SIGNAL RATE - Check if main.py is running or stalled")
    else:
        print("✅ Normal signal generation rate")
    print()
    
    # ===== PART 4: PROCESS HEALTH =====
    print()
    print("PART 4: PROCESS HEALTH CHECK")
    print("-" * 100)
    
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
        lines = result.stdout.split('\n')
        
        main_py_count = sum(1 for line in lines if 'python' in line and 'main.py' in line and 'grep' not in line)
        
        print(f"Active main.py processes: {main_py_count}")
        
        if main_py_count == 0:
            print("🔴 ERROR: No main.py processes running!")
        elif main_py_count == 1:
            print("✅ Single main.py instance (healthy)")
        else:
            print(f"⚠️  WARN: {main_py_count} main.py instances (should be 1, may cause conflicts)")
            print()
            print("Running instances:")
            for line in lines:
                if 'python' in line and 'main.py' in line and 'grep' not in line:
                    parts = line.split()
                    if len(parts) >= 8:
                        print(f"  PID: {parts[1]} | Memory: {parts[5]} | Started: {parts[8:10]}")
        print()
    except Exception as e:
        print(f"Could not check processes: {e}")
        print()
    
    # ===== PART 5: FILE INTEGRITY =====
    print()
    print("PART 5: FILE INTEGRITY CHECK")
    print("-" * 100)
    
    # Check last line is valid JSON
    try:
        with open(master_file, 'r') as f:
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            
            # Read last 1KB and find last complete line
            read_size = min(1024, file_size)
            f.seek(file_size - read_size)
            content = f.read()
            
            last_line = content.split('\n')[-2]  # -2 because last is usually empty
            json.loads(last_line)
        
        print(f"✅ SIGNALS_MASTER.jsonl: Valid JSON, {file_size:,} bytes")
    except json.JSONDecodeError:
        print(f"🔴 ERROR: Last line not valid JSON - file may be corrupted")
        print(f"  Last line: {last_line[:100]}")
    except Exception as e:
        print(f"⚠️  Could not verify file: {e}")
    
    print()
    
    # ===== OVERALL STATUS =====
    print()
    print("=" * 100)
    print("OVERALL STATUS")
    print("=" * 100)
    
    issues = []
    if completion_rate < 90:
        issues.append(f"Field completion only {completion_rate:.1f}% (target: >95%)")
    if orphaned and len(orphaned) > 0:
        issues.append(f"{len(orphaned)} orphaned signals (sent but not persisted)")
    if len(recent_master) < 50:
        issues.append("Low signal generation rate (<50 signals/hour)")
    if main_py_count != 1:
        issues.append(f"{main_py_count} main.py processes (should be 1)")
    
    if issues:
        print("🔴 ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ HEALTHY: All checks passed")
    
    print()

if __name__ == "__main__":
    check_signal_persistence()
