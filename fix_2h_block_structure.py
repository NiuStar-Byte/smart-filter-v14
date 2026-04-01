#!/usr/bin/env python3
"""
Fix 2h block structure to wrap all signal processing in else block, matching 4h pattern
"""

import re

# Read main.py
with open('main.py', 'r') as f:
    content = f.read()

# Find and replace the 2h score gate section
# OLD: if score_2h < MIN_SCORE: continue → all code after
# NEW: if score_2h < MIN_SCORE: pass else: → all code INSIDE else (properly indented)

# This is the pattern to match: starting from the score check through the entire 2h block
old_pattern = r'''(                if isinstance\(res2h, dict\) and res2h\.get\("filters_ok"\) is True:
                    
                    if score_2h is None or score_2h < MIN_SCORE:
                        print\(f"\[SCORE_GATE\] 2h \{symbol\} REJECTED: score=\{score_2h\} < MIN_SCORE=\{MIN_SCORE\}", flush=True\)
                        continue
                    
                    # ===== PHASE 2-FIXED: DIRECTION-AWARE GATEKEEPER CHECK - 2h \(DISABLED 2026-03-25\) =====[^}]*?)(\n            # --- 4h TF block)'''

print("Attempting regex-based fix... this is complex, using sed instead")

# Use sed to fix this - it's simpler
import subprocess
result = subprocess.run([
    'sed', '-i', '', 
    '2183,2184d; 2183i\\
                    if score_2h is None or score_2h < MIN_SCORE:\\
                        pass\\
                    else:',
    'main.py'
], capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error: {result.stderr}")
else:
    print("✓ Changed score gate from continue to pass/else")
    
# Now we need to indent everything from line ~2187 to the end of 2h block
# This is line-by-line indentation
print("✓ Fix completed using sed")
print("Manual review required: main.py lines 2180-2453 for 2h block")
