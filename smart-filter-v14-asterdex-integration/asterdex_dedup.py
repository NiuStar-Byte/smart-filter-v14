#!/usr/bin/env python3
"""
ASTERDEX DEDUPLICATION - Remove duplicate positions from tracker
Keeps first occurrence, removes exact duplicates
"""

import json
from pathlib import Path

TRACKER_FILE = Path("ASTERDEX_POSITIONS_LIVE.jsonl")

positions = []
if TRACKER_FILE.exists():
    with open(TRACKER_FILE) as f:
        for line in f:
            if line.strip():
                pos = json.loads(line.strip())
                positions.append(pos)

print(f"[INFO] Loaded {len(positions)} positions from tracker")

# Create dedup key: symbol + entry_order_id (most reliable unique identifier)
# Falls back to symbol + opened_timestamp (without fractional seconds)
seen = {}
unique_positions = []
duplicate_count = 0

for pos in positions:
    # Use entry_order_id if available, otherwise use symbol + opened date (no fractional seconds)
    if pos.get('entry_order_id'):
        dedup_key = f"{pos['symbol']}_{pos['entry_order_id']}"
    else:
        # Strip fractional seconds for matching
        opened_base = pos['opened'].split('.')[0] if '.' in pos['opened'] else pos['opened']
        dedup_key = f"{pos['symbol']}_{pos['entry_price']}_{opened_base}"
    
    if dedup_key not in seen:
        seen[dedup_key] = True
        unique_positions.append(pos)
    else:
        duplicate_count += 1
        print(f"[DUP] Removing duplicate: {pos['symbol']} entry_id={pos.get('entry_order_id', 'N/A')} opened {pos['opened']}")

print(f"\n[RESULT] Duplicates removed: {duplicate_count}")
print(f"[RESULT] Unique positions: {len(unique_positions)}")

# Save deduplicated positions
with open(TRACKER_FILE, 'w') as f:
    for pos in sorted(unique_positions, key=lambda x: x.get('opened', '')):
        f.write(json.dumps(pos) + '\n')

print(f"[OK] Saved {len(unique_positions)} deduplicated positions to {TRACKER_FILE}")
