#!/usr/bin/env python3
"""
Fix 2h block structure: convert score gate from continue to else block
Restructure to match 4h working pattern
"""

import re

with open('/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/main.py', 'r') as f:
    content = f.read()

# Find the 2h block and restructure it
# Pattern: the problematic score check with continue statement

old_code = '''                if isinstance(res2h, dict) and res2h.get("filters_ok") is True:
                    
                    if score_2h is None or score_2h < MIN_SCORE:
                        print(f"[SCORE_GATE] 2h {symbol} REJECTED: score={score_2h} < MIN_SCORE={MIN_SCORE}", flush=True)
                        continue'''

new_code = '''                if isinstance(res2h, dict) and res2h.get("filters_ok") is True:
                    
                    if score_2h is None or score_2h < MIN_SCORE:
                        print(f"[SCORE_GATE] 2h {symbol} REJECTED: score={score_2h} < MIN_SCORE={MIN_SCORE}", flush=True)
                        pass
                    else:'''

# Replace
if old_code in content:
    content = content.replace(old_code, new_code)
    print("✓ Replaced score gate pattern")
else:
    print("✗ Could not find exact pattern to replace")
    exit(1)

# Now we need to indent all code from "# ===== PHASE 2-FIXED:" line until the except block
# Find these lines and add 4 spaces to the beginning of each indented line

lines = content.split('\n')
new_lines = []
in_2h_else = False
indent_count = 0

for i, line in enumerate(lines):
    # Detect when we enter the 2h else block
    if 'else:' in line and i > 2180 and i < 2190:
        new_lines.append(line)
        in_2h_else = True
        print(f"✓ Found else: at line {i+1}")
        continue
    
    # Detect when we exit the 2h else block
    if in_2h_else and (
        'except Exception as e:' in line and 
        line.startswith('            except')
    ):
        in_2h_else = False
        print(f"✓ Found except at line {i+1}, exiting else block")
    
    # If we're in the 2h else block, indent non-empty lines
    if in_2h_else and line.strip() and not line.strip().startswith('#'):
        # Only indent if it's not already super-indented
        if line.startswith('                    '):  # 20 spaces - indent it to 24
            line = '    ' + line
            indent_count += 1
    
    new_lines.append(line)

print(f"✓ Indented {indent_count} lines")

# Write back
with open('/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/main.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("✓ File updated")
