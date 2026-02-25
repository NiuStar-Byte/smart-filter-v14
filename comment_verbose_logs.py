#!/usr/bin/env python3
"""
Comment out verbose per-symbol analysis logs:
- [TREND], [MACD], [Momentum], [Chop Zone], [Volatility], [ATR], [Momentum Burst], 
- [Spread Filter], [Volume Spike], [Fractal Zone], [Candle Confirmation], etc.
- [HH/LL], [Conditions], [Values], [Met Counts], [Diagnostic]

Keep:
- [SIGNAL], [✅ SENT], [PEC], CYCLE SUMMARY, [WARN], [ERROR], [SUMMARY]
"""

import re

with open('smart_filter.py', 'r') as f:
    lines = f.readlines()

verbose_patterns = [
    r'\[TREND\]',
    r'\[MACD\]',
    r'\[Momentum\]',
    r'\[Chop Zone\]',
    r'\[Chop\]',
    r'\[Volatility\]',
    r'\[ATR Momentum Burst\]',
    r'\[ATR\]',
    r'\[Spread Filter\]',
    r'\[Volume Spike\]',
    r'\[Fractal Zone\]',
    r'\[Candle Confirmation\]',
    r'\[Wick Dominance\]',
    r'\[Absorption\]',
    r'\[HH/LL Trend\]',
    r'\[HH/LL\]',
    r'\[Conditions\]',
    r'\[Values\]',
    r'\[Met Counts\]',
    r'\[Diagnostic\]',
    r'\[Not enough data',
    r'\[get_signal_direction',
    r'\[Liquidity',
    r'\[Support/Resistance',
    r'\[VWAP',
    r'\[Volatility Squeeze\]',
    r'\[OrderBookDeltaLog\]',
    r'\[RestingOrderDensityLog\]',
]

# Pattern to match print statements with these verbose logs
output_lines = []
for i, line in enumerate(lines, 1):
    # Check if this line is a print statement with verbose filter logs
    is_verbose = any(pattern in line for pattern in verbose_patterns)
    
    # Also skip lines that are already commented
    is_commented = line.strip().startswith('#')
    
    # Don't comment lines with SIGNAL, SENT, PEC, WARN, ERROR, SUMMARY (keep these)
    has_important = any(x in line for x in ['[SIGNAL]', '[✅ SENT]', '[PEC', 'SUMMARY', '[WARN]', '[ERROR]', 'CYCLE'])
    
    if is_verbose and not is_commented and not has_important and 'print(' in line:
        # Comment out this line
        output_lines.append('            # [LOG DISABLED] ' + line.lstrip())
    else:
        output_lines.append(line)

with open('smart_filter.py', 'w') as f:
    f.writelines(output_lines)

print("✅ Commented out verbose filter analysis logs")
