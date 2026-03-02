#!/bin/bash

# A/B Testing Results Checker
# Usage: ./CHECK_AB_RESULTS.sh

echo "=========================================================================================================="
echo "🧪 A/B TEST RESULTS CHECKER"
echo "=========================================================================================================="
echo ""
echo "📊 Full PEC Report (All Dimensions):"
echo "   Command: python3 pec_enhanced_reporter.py"
echo ""
python3 pec_enhanced_reporter.py 2>&1 | head -60
echo ""
echo "=========================================================================================================="
echo ""
echo "📋 Quick Metrics Summary:"
echo "=========================================================================================================="
python3 pec_enhanced_reporter.py 2>&1 | grep -A 12 "BY DIRECTION"
echo ""
echo "=========================================================================================================="
echo ""
echo "📌 Baseline Reference:"
echo "   File: PHASE1_BASELINE_A_B_TEST.md"
echo "   Overall WR Target: 31.19% → 35-40%"
echo "   LONG WR Target:    28.0%  → 35%+ (MAIN FIX)"
echo "   SHORT WR Target:   50.0%  → 50%+ (MAINTAIN)"
echo "   P&L Target:        -\$4,288.36 → -\$500 to +\$200"
echo ""
echo "=========================================================================================================="
