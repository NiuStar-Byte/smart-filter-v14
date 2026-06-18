#!/bin/bash
cd /Users/geniustarigan/.openclaw/workspace
source .env
nohup python3 smart-filter-v14-main/smart-filter-v14-asterdex-integration/asterdex_entry_poster.py > /tmp/asterdex_entry_poster.log 2>&1 &
echo "✅ Asterdex Entry Poster started"
sleep 3
ps aux | grep asterdex_entry_poster | grep -v grep | wc -l
