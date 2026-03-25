#!/usr/bin/env python3
"""
cleanup_extreme_rr_signals.py - Mark signals with extreme RR for exclusion

Purpose:
- Identifies signals with RR outside acceptable bounds (0.5-4.0)
- Marks them with 'EXTREME_RR' flag in data_quality_flag
- Creates audit trail of what was flagged and why
- Does NOT delete signals - preserves data integrity

Background:
- Before 2026-03-25: Market-driven RR calc used current_price instead of entry_price
- This created fake extreme RR values (6.3, 0.03, etc.)
- Fixed in commit 1b7067a, but historical signals need manual marking

Strategy:
1. Scan all signals in SIGNALS_MASTER.jsonl
2. Calculate actual RR = (TP - Entry) / (Entry - SL)
3. If RR < 0.5 or RR > 4.0: Add 'EXTREME_RR' to data_quality_flag
4. Back up original file
5. Write cleaned file
6. Log audit trail
"""

import json
import os
from datetime import datetime
import shutil

SIGNALS_FILE = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/SIGNALS_MASTER.jsonl"
AUDIT_TRAIL_FILE = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/cleanup_audit_extreme_rr.log"

def calculate_rr(entry, tp, sl):
    """Calculate RR from entry/tp/sl prices"""
    try:
        entry_f = float(entry or 0)
        tp_f = float(tp or 0)
        sl_f = float(sl or 0)
        
        if entry_f and tp_f and sl_f:
            reward = tp_f - entry_f
            risk = entry_f - sl_f
            if risk > 0:
                return round(reward / risk, 2)
    except:
        pass
    return None

def cleanup_extreme_rr():
    """Mark signals with extreme RR (< 0.5 or > 4.0)"""
    
    if not os.path.exists(SIGNALS_FILE):
        print(f"[ERROR] {SIGNALS_FILE} not found")
        return False
    
    # Create backup
    backup_file = f"{SIGNALS_FILE}.backup_before_rr_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(SIGNALS_FILE, backup_file)
    print(f"[BACKUP] Created: {backup_file}")
    
    # Open audit trail
    audit = open(AUDIT_TRAIL_FILE, 'a')
    audit.write(f"\n\n{'='*80}\n")
    audit.write(f"RR CLEANUP RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}\n")
    audit.write(f"{'='*80}\n")
    
    signals_processed = 0
    signals_flagged = 0
    extreme_rr_signals = []
    
    # Process file
    with open(SIGNALS_FILE, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    
    for line_num, line in enumerate(lines, 1):
        try:
            signal = json.loads(line)
            signals_processed += 1
            
            entry = float(signal.get('entry_price') or 0)
            tp = float(signal.get('tp_target') or signal.get('tp_price') or 0)
            sl = float(signal.get('sl_target') or signal.get('sl_price') or 0)
            
            rr = calculate_rr(entry, tp, sl)
            
            if rr is not None and (rr < 0.5 or rr > 4.0):
                # Mark as extreme
                if 'data_quality_flag' not in signal or signal['data_quality_flag'] is None:
                    signal['data_quality_flag'] = 'EXTREME_RR'
                elif 'EXTREME_RR' not in signal['data_quality_flag']:
                    signal['data_quality_flag'] = f"{signal['data_quality_flag']},EXTREME_RR"
                
                signals_flagged += 1
                extreme_rr_signals.append({
                    'symbol': signal.get('symbol'),
                    'timeframe': signal.get('timeframe'),
                    'direction': signal.get('signal_type'),
                    'fired_time': signal.get('fired_time_utc'),
                    'entry': entry,
                    'tp': tp,
                    'sl': sl,
                    'rr': rr,
                    'achieved_rr_stored': signal.get('achieved_rr'),
                    'status': signal.get('status')
                })
            
            cleaned_lines.append(json.dumps(signal) + '\n')
        
        except json.JSONDecodeError as e:
            print(f"[WARN] Line {line_num}: JSON parse error - {e}")
            cleaned_lines.append(line)
        except Exception as e:
            print(f"[ERROR] Line {line_num}: {e}")
            cleaned_lines.append(line)
    
    # Write cleaned file
    with open(SIGNALS_FILE, 'w') as f:
        f.writelines(cleaned_lines)
    
    # Log results
    audit.write(f"\nSignals Processed: {signals_processed}\n")
    audit.write(f"Signals Flagged: {signals_flagged}\n")
    audit.write(f"Backup Location: {backup_file}\n\n")
    
    if extreme_rr_signals:
        audit.write(f"FLAGGED SIGNALS (RR < 0.5 or > 4.0):\n")
        audit.write(f"{'-'*80}\n")
        for sig in sorted(extreme_rr_signals, key=lambda x: x['rr']):
            audit.write(f"\n{sig['symbol']} {sig['timeframe']} {sig['direction']}\n")
            audit.write(f"  Fired: {sig['fired_time']}\n")
            audit.write(f"  Entry: {sig['entry']:.8f}, TP: {sig['tp']:.8f}, SL: {sig['sl']:.8f}\n")
            audit.write(f"  Calculated RR: {sig['rr']:.2f}\n")
            audit.write(f"  Stored achieved_rr: {sig['achieved_rr_stored']}\n")
            audit.write(f"  Status: {sig['status']}\n")
    
    audit.write(f"\nCLEANUP COMPLETE\n")
    audit.close()
    
    # Console output
    print(f"\n{'='*80}")
    print(f"RR CLEANUP COMPLETE")
    print(f"{'='*80}")
    print(f"Signals Processed: {signals_processed}")
    print(f"Signals Flagged: {signals_flagged}")
    print(f"Backup: {backup_file}")
    print(f"Audit Trail: {AUDIT_TRAIL_FILE}")
    print(f"\nFlagged signals now have 'EXTREME_RR' in data_quality_flag")
    print(f"Reporter will automatically EXCLUDE these from calculations")
    
    return True

if __name__ == "__main__":
    cleanup_extreme_rr()
