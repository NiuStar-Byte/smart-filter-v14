#!/usr/bin/env python3
"""
Extract tier-qualifying combos from PEC_ENHANCED_REPORT.txt
Store them for SIGNAL_TIERS.json synchronization
"""

import re
import json
from datetime import datetime, timezone, timedelta

REPORT_FILE = "/Users/geniustarigan/.openclaw/workspace/PEC_ENHANCED_REPORT.txt"
OUTPUT_FILE = "/Users/geniustarigan/.openclaw/workspace/TIER_QUALIFYING_COMBOS.json"

TIER_THRESHOLDS = {
    1: {"wr": 60, "pnl": 5.50, "trades": 60},
    2: {"wr": 50, "pnl": 3.50, "trades": 50},
    3: {"wr": 40, "pnl": 2.00, "trades": 40},
}

tier_1 = []
tier_2 = []
tier_3 = []

try:
    with open(REPORT_FILE, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Look for lines with ✓ and performance metrics
        if '✓' not in line or 'WR:' not in line:
            continue
        
        # Extract metrics
        wr_match = re.search(r'WR:\s+([\d\.]+)%', line)
        avg_match = re.search(r'Avg:\s+\$\s*([\d\.\+\-]+)', line)
        closed_match = re.search(r'Closed:\s+(\d+)', line)
        
        if not (wr_match and avg_match and closed_match):
            continue
        
        wr = float(wr_match.group(1))
        avg = float(avg_match.group(1))
        closed = int(closed_match.group(1))
        
        # Extract combo name (between ✓ and |)
        combo_match = re.search(r'✓\s+(.+?)\s+\|', line)
        if not combo_match:
            continue
        
        combo = combo_match.group(1).strip()
        
        # Classify by tier
        if wr >= 60 and avg >= 5.50 and closed >= 60:
            tier_1.append({
                'combo': combo,
                'wr': round(wr, 1),
                'avg': round(avg, 2),
                'closed': closed
            })
        elif wr >= 50 and avg >= 3.50 and closed >= 50:
            tier_2.append({
                'combo': combo,
                'wr': round(wr, 1),
                'avg': round(avg, 2),
                'closed': closed
            })
        elif wr >= 40 and avg >= 2.00 and closed >= 40:
            tier_3.append({
                'combo': combo,
                'wr': round(wr, 1),
                'avg': round(avg, 2),
                'closed': closed
            })

except Exception as e:
    print(f"Error reading report: {e}")
    exit(1)

# Sort by WR descending
tier_1.sort(key=lambda x: -x['wr'])
tier_2.sort(key=lambda x: -x['wr'])
tier_3.sort(key=lambda x: -x['wr'])

# Save to file
output_data = {
    'generated': datetime.now(timezone(timedelta(hours=7))).isoformat(),
    'source': 'PEC_ENHANCED_REPORT.txt',
    'tier_1': tier_1,
    'tier_2': tier_2,
    'tier_3': tier_3,
    'summary': {
        'tier_1_count': len(tier_1),
        'tier_2_count': len(tier_2),
        'tier_3_count': len(tier_3),
        'total_qualifying': len(tier_1) + len(tier_2) + len(tier_3)
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=2)

# Print to console
print("\n" + "="*130)
print("🎯 TIER-QUALIFYING COMBOS (From Closed Signal Performance)")
print("="*130)
print(f"Source: PEC_ENHANCED_REPORT.txt | Generated: {output_data['generated']}")

if tier_1:
    print(f"\n🥇 TIER-1 ({len(tier_1)} combos - 60% WR, $5.50+ avg, 60+ trades):")
    for c in tier_1:
        print(f"  • {c['combo']:<80} WR: {c['wr']:>6.1f}% | Avg: ${c['avg']:>7.2f} | Closed: {c['closed']:>3}")

if tier_2:
    print(f"\n🥈 TIER-2 ({len(tier_2)} combos - 50% WR, $3.50+ avg, 50+ trades):")
    for c in tier_2:
        print(f"  • {c['combo']:<80} WR: {c['wr']:>6.1f}% | Avg: ${c['avg']:>7.2f} | Closed: {c['closed']:>3}")

if tier_3:
    print(f"\n🥉 TIER-3 ({len(tier_3)} combos - 40% WR, $2.00+ avg, 40+ trades):")
    for c in tier_3:
        print(f"  • {c['combo']:<80} WR: {c['wr']:>6.1f}% | Avg: ${c['avg']:>7.2f} | Closed: {c['closed']:>3}")

print(f"\n{'='*130}")
print(f"SUMMARY: {len(tier_1)} Tier-1 + {len(tier_2)} Tier-2 + {len(tier_3)} Tier-3 = {output_data['summary']['total_qualifying']} total qualifying combos")
print(f"Saved to: {OUTPUT_FILE}")
print("="*130 + "\n")
