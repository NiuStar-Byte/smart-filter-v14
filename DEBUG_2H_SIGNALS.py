#!/usr/bin/env python3
"""
Debug script to understand why 2h signals aren't being fired.
Compares 1h, 2h, 4h blocks line-by-line.
"""

import subprocess
import sys

# Get 2h block
result_2h = subprocess.run(
    ["sed", "-n", "2140,2530p", "main.py"],
    capture_output=True,
    text=True,
    cwd="/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main"
)

# Get 4h block
result_4h = subprocess.run(
    ["sed", "-n", "2530,2780p", "main.py"],
    capture_output=True,
    text=True,
    cwd="/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main"
)

# Get 1h block for comparison
result_1h = subprocess.run(
    ["sed", "-n", "1700,1750p", "main.py"],
    capture_output=True,
    text=True,
    cwd="/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main"
)

print("=" * 80)
print("CRITICAL FLOW COMPARISON: 1h vs 2h vs 4h")
print("=" * 80)

print("\n### 1h EARLY FLOW (check scoring & gatekeeper logic)")
print(result_1h.stdout[:1000])

print("\n### 2h EARLY FLOW (check scoring & gatekeeper logic)")
print(result_2h.stdout[:1500])

print("\n### 4h EARLY FLOW (check scoring & gatekeeper logic)")
print(result_4h.stdout[:1500])

# Now find critical differences
print("\n" + "=" * 80)
print("KEY DIFFERENCES ANALYSIS")
print("=" * 80)

lines_2h = result_2h.stdout.split('\n')
lines_4h = result_4h.stdout.split('\n')

# Find MIN_SCORE checks
print("\n### MIN_SCORE HANDLING:")
for i, line in enumerate(lines_2h):
    if "MIN_SCORE" in line or "score < " in line or "score_2h" in line[:30]:
        print(f"2h:{i:3d}: {line}")

print()
for i, line in enumerate(lines_4h):
    if "MIN_SCORE" in line or "score < " in line or "score =" in line or "if score" in line:
        print(f"4h:{i:3d}: {line}")

# Find COOLDOWN checks
print("\n### COOLDOWN HANDLING:")
for i, line in enumerate(lines_2h):
    if "COOLDOWN" in line or "last2h" in line:
        print(f"2h:{i:3d}: {line}")

print()
for i, line in enumerate(lines_4h):
    if "COOLDOWN" in line or "last4h" in line:
        print(f"4h:{i:3d}: {line}")

# Find continue statements
print("\n### CONTINUE/SKIP STATEMENTS:")
for i, line in enumerate(lines_2h):
    if "continue" in line:
        print(f"2h:{i:3d}: {line}")

print()
for i, line in enumerate(lines_4h):
    if "continue" in line:
        print(f"4h:{i:3d}: {line}")

# Find Telegram send calls
print("\n### TELEGRAM SEND CALLS:")
for i, line in enumerate(lines_2h):
    if "send_telegram_alert" in line:
        print(f"2h:{i:3d}: {line}")

print()
for i, line in enumerate(lines_4h):
    if "send_telegram_alert" in line:
        print(f"4h:{i:3d}: {line}")
