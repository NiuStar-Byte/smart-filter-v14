#!/bin/bash
# PEC Auto-Update Runner (for cron job)
# Runs every 5 minutes, updates signal status silently
# Only announces when TP/SL/TIMEOUT hits

cd /Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main
python3 pec_executor.py >> pec_updates.log 2>&1
