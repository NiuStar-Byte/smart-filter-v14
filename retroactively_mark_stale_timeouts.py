#!/usr/bin/env python3
"""
Retroactively Mark STALE_TIMEOUT Signals
Scans all TIMEOUT signals in SIGNALS_MASTER.jsonl and flags those >150% overdue
This fixes the blind spot where old signals weren't checked by pec_executor
"""

import json
from datetime import datetime, timezone
import os

def main():
    workspace = "/Users/geniustarigan/.openclaw/workspace"
    signals_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")
    backup_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl.backup_before_stale_fix")
    
    if not os.path.exists(signals_file):
        print(f"[ERROR] {signals_file} not found")
        return
    
    # TF-specific max_bars from pec_executor
    MAX_BARS = {
        '15min': 15,
        '30min': 10,
        '1h': 5
    }
    
    # TF-specific bar duration (minutes)
    TF_MINUTES = {
        '15min': 15,
        '30min': 30,
        '1h': 60
    }
    
    # STALE threshold: > 150% of max_bars
    STALE_MULTIPLIER = 1.5
    
    print("\n" + "=" * 120)
    print("🔄 RETROACTIVELY MARKING STALE_TIMEOUT SIGNALS")
    print("=" * 120)
    print()
    
    # Read all signals
    signals = []
    with open(signals_file, 'r') as f:
        for line in f:
            try:
                signal = json.loads(line.strip())
                signals.append(signal)
            except:
                pass
    
    print(f"[INFO] Loaded {len(signals)} signals from SIGNALS_MASTER.jsonl")
    print()
    
    # Filter TIMEOUT signals only (exclude existing STALE_TIMEOUT)
    timeout_signals = [s for s in signals if s.get('status') == 'TIMEOUT']
    print(f"[INFO] Found {len(timeout_signals)} TIMEOUT signals to check")
    print()
    
    # Check each TIMEOUT signal
    stale_count = 0
    stale_details = []
    
    for sig in timeout_signals:
        try:
            tf = sig.get('timeframe')
            if tf not in MAX_BARS:
                continue
            
            fired_str = sig.get('fired_time_utc')
            closed_str = sig.get('closed_at')
            
            if not fired_str or not closed_str:
                continue
            
            # Parse times
            fired = datetime.fromisoformat(fired_str.replace('Z', '+00:00'))
            closed = datetime.fromisoformat(closed_str.replace('Z', '+00:00'))
            
            if fired.tzinfo is None:
                fired = fired.replace(tzinfo=timezone.utc)
            if closed.tzinfo is None:
                closed = closed.replace(tzinfo=timezone.utc)
            
            # Calculate bars elapsed
            time_delta_minutes = (closed - fired).total_seconds() / 60
            bars_elapsed = int(time_delta_minutes / TF_MINUTES[tf])
            
            # Get thresholds
            max_bars = MAX_BARS[tf]
            stale_threshold = max_bars * STALE_MULTIPLIER
            
            # Check if STALE
            if bars_elapsed > stale_threshold:
                hours_overdue = int((bars_elapsed - max_bars) * TF_MINUTES[tf] / 60)
                
                sig['status'] = 'STALE_TIMEOUT'
                sig['pnl_usd'] = 0.0  # Zero out P&L for stale
                sig['data_quality_flag'] = f"STALE_TIMEOUT_{hours_overdue}h_overdue"
                
                stale_count += 1
                stale_details.append({
                    'symbol': sig.get('symbol'),
                    'tf': tf,
                    'bars': bars_elapsed,
                    'hours_overdue': hours_overdue,
                    'max_bars': max_bars
                })
        except Exception as e:
            pass
    
    print(f"🔴 FOUND {stale_count} STALE_TIMEOUT signals to mark")
    print()
    
    if stale_details:
        print("Breakdown by TF:")
        tf_count = {}
        for detail in stale_details:
            tf = detail['tf']
            tf_count[tf] = tf_count.get(tf, 0) + 1
        
        for tf in ['15min', '30min', '1h']:
            count = tf_count.get(tf, 0)
            if count > 0:
                print(f"  {tf}: {count} signals")
        
        print()
        print("Sample stale signals:")
        for detail in stale_details[:10]:
            print(f"  {detail['symbol']} {detail['tf']}: {detail['bars']} bars ({detail['hours_overdue']}h overdue, max={detail['max_bars']})")
        
        if len(stale_details) > 10:
            print(f"  ... and {len(stale_details) - 10} more")
    
    # Backup original file
    if stale_count > 0:
        print()
        print(f"📦 Creating backup: {os.path.basename(backup_file)}")
        with open(signals_file, 'r') as f_in:
            with open(backup_file, 'w') as f_out:
                f_out.write(f_in.read())
        
        # Write updated signals
        print(f"✍️  Writing updated signals with STALE_TIMEOUT flags...")
        with open(signals_file, 'w') as f:
            for sig in signals:
                f.write(json.dumps(sig) + '\n')
        
        print()
        print("✅ COMPLETE")
        print(f"   {stale_count} TIMEOUT signals marked as STALE_TIMEOUT")
        print(f"   Original backed up to: SIGNALS_MASTER.jsonl.backup_before_stale_fix")
        print()
        print("⚠️  IMPACT ON METRICS:")
        print(f"   These {stale_count} signals will now be EXCLUDED from all backtest metrics")
        print(f"   Estimated P&L recovery: Review new pec_enhanced_reporter.py output")
    else:
        print("✅ No changes needed - all TIMEOUT signals are within thresholds")
    
    print()

if __name__ == '__main__':
    main()
