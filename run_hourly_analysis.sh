#!/bin/bash
# Hourly Signal Analysis - Run at :00 every UTC hour
# Add to crontab: 0 * * * * cd /Users/geniustarigan/.openclaw/workspace && /usr/bin/python3 hourly_signal_analysis.py >> hourly_analysis.log 2>&1

cd /Users/geniustarigan/.openclaw/workspace
/usr/bin/python3 hourly_signal_analysis.py >> hourly_analysis.log 2>&1
