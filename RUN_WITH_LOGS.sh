#!/bin/bash

echo "Starting SmartFilter with full logging..."
echo "Logs will print to console AND save to smartfilter_run.log"
echo ""

# Run main.py and tee output to both console and file
python3 main.py 2>&1 | tee smartfilter_run.log

echo ""
echo "Run complete. Check smartfilter_run.log for full output."
