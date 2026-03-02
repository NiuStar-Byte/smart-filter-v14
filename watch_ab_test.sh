#!/bin/bash

# A/B Test Live Monitor - Refreshes every 5 seconds
# Usage: ./watch_ab_test.sh
# Press Ctrl+C to stop

cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main
while true; do
  clear
  python3 COMPARE_AB_TEST.py --once
  sleep 5
done
