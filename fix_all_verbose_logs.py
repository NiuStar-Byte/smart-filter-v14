#!/usr/bin/env python3
"""
Wrap ALL verbose filter analysis print statements in if DEBUG_FILTERS: condition
"""

import re

with open('smart_filter.py', 'r') as f:
    content = f.read()

# Pattern: Find multiline print statements that contain verbose filter log markers
verbose_keywords = [
    r'\[TREND\]', r'\[MACD\]', r'\[Momentum\]', r'\[Chop Zone\]', r'\[Volatility\]',
    r'\[ATR\]', r'\[Volume Spike\]', r'\[Fractal\]', r'\[Candle\]', r'\[Wick\]',
    r'\[HH/LL\]', r'\[Conditions\]', r'\[Met Counts\]', r'\[Diagnostic\]',
    r'\[Liquidity\]', r'\[Support/Resistance\]', r'\[VWAP\]', r'\[Spread\]',
    r'\[Absorption\]', r'\[values:\]', r'\[Not enough data\]'
]

# Build regex to find multi-line print statements
# This is tricky - we need to find:  print( ... ) blocks that span multiple lines
lines = content.split('\n')
result_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Check if this is a print statement
    if 'print(' in line and not line.strip().startswith('#'):
        # Check if it contains verbose keywords
        is_verbose = any(keyword in line for keyword in verbose_keywords)
        
        if not is_verbose and i < len(lines) - 1:
            # Check if next lines are part of the print statement
            j = i + 1
            full_statement = line
            while j < len(lines) and ')' not in full_statement:
                full_statement += '\n' + lines[j]
                is_verbose = any(keyword in lines[j] for keyword in verbose_keywords)
                if is_verbose:
                    break
                j += 1
            
            if is_verbose:
                # This print spans multiple lines and contains verbose keywords
                # Add "if DEBUG_FILTERS:" before it
                indent = len(line) - len(line.lstrip())
                result_lines.append(' ' * indent + 'if DEBUG_FILTERS:')
                
                # Add the print with increased indent
                for k in range(i, j + 1):
                    result_lines.append('    ' + lines[k])
                i = j + 1
                continue
    
    result_lines.append(line)
    i += 1

with open('smart_filter.py', 'w') as f:
    f.write('\n'.join(result_lines))

print("✅ Wrapped verbose multi-line print statements")
