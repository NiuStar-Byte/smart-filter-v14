#!/usr/bin/env python3
"""
Wrap verbose per-symbol logs in DEBUG_FILTERS condition
Keeps [SIGNAL], [WARN], [ERROR], [PEC], CYCLE SUMMARY, [SUMMARY] Reversal
Disables all [symbol] [TREND/MACD/MOMENTUM/etc] logs
"""

import re

with open('smart_filter.py', 'r') as f:
    content = f.read()

# Pattern: per-symbol verbose logs like: print(f"[{self.symbol}] [TREND]...)
# Find lines with per-symbol analysis logs and wrap in if DEBUG_FILTERS:
pattern = r'(\s+)print\(f"\[{self\.symbol}\]'
replacement = r'\1if DEBUG_FILTERS: print(f"[{self.symbol}'

content = re.sub(pattern, replacement, content)

# Also wrap symbol analysis logs like: print(f"[symbol_name] ...)
pattern2 = r'(\s+)print\(f"\[{symbol}\]'
replacement2 = r'\1if DEBUG_FILTERS: print(f"[{symbol}'

content = re.sub(pattern2, replacement2, content)

with open('smart_filter.py', 'w') as f:
    f.write(content)

print("✅ Wrapped verbose per-symbol logs in DEBUG_FILTERS condition")
