#!/bin/bash

# track_instrumentation_real.sh
# Simplified instrumentation tracker - reads directly from SIGNALS_MASTER.jsonl
# Shows which filters are passing and their correlation with win rate

SIGNALS_MASTER="/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"

if [ ! -f "$SIGNALS_MASTER" ]; then
    echo "ERROR: SIGNALS_MASTER.jsonl not found at $SIGNALS_MASTER"
    exit 1
fi

echo "=== Filter Instrumentation Analysis ==="
echo "Source: $SIGNALS_MASTER"
echo "Last updated: $(date)"
echo ""

# Extract all unique filter names from passed_filters
echo "Extracting instrumented signals..."
python3 << 'EOF'
import json
from collections import defaultdict
from datetime import datetime
import re

master_file = "/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl"

# All 20 filters (from smart_filter.py)
all_filters = [
    "MACD", "Volume Spike", "Fractal Zone", "TREND", "Momentum",
    "ATR Momentum Burst", "MTF Volume Agreement", "HH/LL Trend", "Volatility Model",
    "Liquidity Awareness", "Volatility Squeeze", "Candle Confirmation",
    "VWAP Divergence", "Spread Filter", "Chop Zone", "Liquidity Pool",
    "Support/Resistance", "Smart Money Bias", "Absorption", "Wick Dominance"
]

# Load current weights from smart_filter.py
weights = {}
try:
    smart_filter_path = "/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/smart_filter.py"
    with open(smart_filter_path, 'r') as f:
        content = f.read()
    match = re.search(r'self\.filter_weights_long = \{(.*?)\}', content, re.DOTALL)
    if match:
        lines = match.group(1).split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':')
                filter_name = parts[0].strip().strip('"').strip("'")
                try:
                    weight_str = parts[1].split('#')[0].strip().rstrip(',')
                    weight = float(weight_str)
                    weights[filter_name] = weight
                except (ValueError, IndexError):
                    pass
except Exception as e:
    pass

filter_stats = defaultdict(lambda: {"passed": 0, "won": 0, "lost": 0})
total_signals = 0
instrumented_signals = 0
closed_signals = 0

try:
    with open(master_file, 'r') as f:
        for line in f:
            try:
                signal = json.loads(line.strip())
                total_signals += 1
                
                # Check if instrumented
                if 'passed_filters' not in signal:
                    continue
                
                instrumented_signals += 1
                status = signal.get('status', 'UNKNOWN')
                
                # Track if signal closed
                if status in ['TP_HIT', 'SL_HIT']:
                    closed_signals += 1
                    is_win = (status == 'TP_HIT')
                    
                    # Count each passed filter
                    for filt in signal.get('passed_filters', []):
                        filter_stats[filt]['passed'] += 1
                        if is_win:
                            filter_stats[filt]['won'] += 1
                        else:
                            filter_stats[filt]['lost'] += 1
            
            except json.JSONDecodeError:
                continue

except FileNotFoundError:
    print(f"ERROR: Cannot read {master_file}")
    exit(1)

print(f"\n{'='*110}")
print(f"FILTER EFFECTIVENESS ANALYSIS - ALL 20 FILTERS")
print(f"{'='*110}")
print(f"Total signals: {total_signals} | Instrumented: {instrumented_signals} | Closed (TP/SL): {closed_signals}")
print(f"Baseline WR: {(closed_signals / max(1, instrumented_signals))*100:.1f}%")
print(f"{'='*110}\n")

# Calculate and sort by effectiveness
effectiveness = []
baseline_wr = closed_signals / max(1, instrumented_signals)

for filter_name in all_filters:
    stats = filter_stats[filter_name]
    
    if stats['passed'] == 0:
        effectiveness.append({
            'name': filter_name,
            'passed': 0,
            'won': 0,
            'lost': 0,
            'wr': None,
            'effectiveness': None
        })
    else:
        filter_wr = stats['won'] / max(1, stats['passed'])
        effectiveness_pp = (filter_wr - baseline_wr) * 100
        
        effectiveness.append({
            'name': filter_name,
            'passed': stats['passed'],
            'won': stats['won'],
            'lost': stats['lost'],
            'wr': filter_wr,
            'effectiveness': effectiveness_pp
        })

# Sort by effectiveness (descending, nulls last)
effectiveness.sort(key=lambda x: (x['effectiveness'] is None, x['effectiveness'] if x['effectiveness'] is not None else -999), reverse=True)

# Print header
print(f"{'Rank':<5} {'Filter Name':<30} {'Passed':<8} {'Wins':<8} {'WR':<10} {'FA':<8} {'Effectiveness':<16} {'Weight':<8} {'Status':<8}")
print(f"{'-'*145}")

for idx, item in enumerate(effectiveness, 1):
    if item['passed'] == 0:
        status = "○ (0 pass)"
        wr_str = "N/A"
        eff_str = "N/A"
        fa_str = "0.00%"
    else:
        if item['wr'] >= 0.70:
            status = "⭐ High"
        elif item['wr'] >= 0.50:
            status = "✓ Mid"
        else:
            status = "· Low"
        wr_str = f"{item['wr']*100:5.1f}%"
        eff_str = f"{item['effectiveness']:+6.1f}pp"
        fa_pct = (item['passed'] / max(1, closed_signals)) * 100
        fa_str = f"{fa_pct:5.2f}%"
    
    weight = weights.get(item['name'], 'N/A')
    weight_str = f"{weight}" if isinstance(weight, (int, float)) else str(weight)
    print(f"{idx:<5} {item['name']:<30} {item['passed']:<8} {item['won']:<8} {wr_str:<10} {fa_str:<8} {eff_str:<16} {weight_str:<8} {status:<8}")

EOF
