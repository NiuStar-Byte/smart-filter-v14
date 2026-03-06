#!/usr/bin/env python3
"""
Backfill SIGNALS_MASTER.jsonl with all signals from SENT_SIGNALS.jsonl
Ensures SIGNALS_MASTER contains the complete history
"""

import json
from datetime import datetime, timezone, timedelta
import os

def backfill_signals_master():
    workspace = "/Users/geniustarigan/.openclaw/workspace"
    master_path = os.path.join(workspace, "SIGNALS_MASTER.jsonl")
    sent_signals_path = os.path.join(workspace, "SENT_SIGNALS.jsonl")
    
    # Load SIGNALS_MASTER to get what we already have
    master_uuids = set()
    master_signals = []
    
    if os.path.exists(master_path):
        with open(master_path) as f:
            for line in f:
                sig = json.loads(line)
                uuid = sig.get('signal_uuid')
                if uuid:
                    master_uuids.add(uuid)
                master_signals.append(sig)
    
    print(f"[INFO] SIGNALS_MASTER: {len(master_uuids)} unique signals")
    
    # Load SENT_SIGNALS and find ones NOT in MASTER
    sent_count = 0
    appended_count = 0
    
    if os.path.exists(sent_signals_path):
        with open(sent_signals_path) as f:
            for line in f:
                sig = json.loads(line)
                uuid = sig.get('uuid') or sig.get('signal_uuid')
                sent_count += 1
                
                if uuid not in master_uuids:
                    # Convert SENT_SIGNALS schema to SIGNALS_MASTER schema (29 fields)
                    master_record = {
                        "signal_uuid": uuid or '',
                        "symbol": sig.get('symbol', ''),
                        "timeframe": sig.get('timeframe', ''),
                        "signal_type": sig.get('signal_type', ''),
                        "entry_price": float(sig.get('entry_price', 0)),
                        "tp_target": float(sig.get('tp_target', 0)),
                        "sl_target": float(sig.get('sl_target', 0)),
                        "tp_pct": float(sig.get('tp_pct', 0)),
                        "sl_pct": float(sig.get('sl_pct', 0)),
                        "achieved_rr": float(sig.get('achieved_rr', 0)),
                        "score": int(sig.get('score', 0)),
                        "max_score": int(sig.get('max_score', 19)),
                        "confidence": float(sig.get('confidence', 0)),
                        "route": sig.get('route', ''),
                        "regime": sig.get('regime', ''),
                        "weighted_score": float(sig.get('confidence', 0)) * int(sig.get('score', 0)) / 100 if int(sig.get('max_score', 1)) else 0,
                        "telegram_msg_id": sig.get('telegram_msg_id', ''),
                        "sent_time_utc": sig.get('sent_time_utc', datetime.utcnow().isoformat()),
                        "fired_time_utc": sig.get('fired_time_utc', ''),
                        "fired_time_jakarta": sig.get('fired_time_jakarta', ''),
                        "status": sig.get('status', 'OPEN'),
                        "closed_at": sig.get('closed_at'),
                        "actual_exit_price": sig.get('actual_exit_price'),
                        "pnl_usd": sig.get('pnl_usd'),
                        "pnl_pct": sig.get('pnl_pct'),
                        "signal_origin": "BACKFILLED_SENT_SIGNALS",
                        "data_quality_flag": sig.get('data_quality_flag', ''),
                    }
                    master_signals.append(master_record)
                    master_uuids.add(uuid)
                    appended_count += 1
    
    print(f"[INFO] SENT_SIGNALS: {sent_count} signals")
    print(f"[INFO] Appended: {appended_count} new signals to SIGNALS_MASTER")
    
    # Sort by fired_time_utc before writing
    master_signals.sort(key=lambda s: s.get('fired_time_utc', ''))
    
    # Write back to SIGNALS_MASTER
    with open(master_path, 'w') as f:
        for sig in master_signals:
            f.write(json.dumps(sig) + '\n')
    
    print(f"[INFO] SIGNALS_MASTER now has {len(master_signals)} total signals")
    print(f"[INFO] Backfill complete!")

if __name__ == '__main__':
    backfill_signals_master()
