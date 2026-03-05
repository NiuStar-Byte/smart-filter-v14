#!/bin/bash
# Setup hourly reporter cron job
# Run this ONCE to enable automatic hourly PEC report snapshots

echo "Setting up hourly PEC reporter cron job..."
(crontab -l 2>/dev/null || echo ""; echo "0 * * * * cd /Users/geniustarigan/.openclaw/workspace && python3 hourly_reporter.py >> hourly_reporter.log 2>&1") | crontab -

echo ""
echo "✅ Cron job installed!"
echo ""
echo "Verify with:"
echo "  crontab -l | grep hourly_reporter"
echo ""
echo "Reports will be captured at:00 (top of every hour) and saved to:"
echo "  /Users/geniustarigan/.openclaw/workspace/REPORTS/pec_report_YYYY-MM-DD_HH-00.txt"
