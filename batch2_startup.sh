#!/bin/bash

# BATCH 2 STARTUP SCRIPT
# Clears Batch 1 signals, prepares for Batch 2 with new ATR-based 2:1 TP/SL

echo "================================"
echo "BATCH 2 STARTUP - ATR-BASED 2:1"
echo "================================"
echo ""

WORK_DIR="/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main"
SIGNALS_JSONL="$WORK_DIR/signals_fired.jsonl"

# Step 1: Backup Batch 1 signals (for archive)
echo "[1/5] Backing up Batch 1 signals..."
if [ -f "$SIGNALS_JSONL" ]; then
    cp "$SIGNALS_JSONL" "$SIGNALS_JSONL.batch1.backup"
    echo "✅ Backup saved: $SIGNALS_JSONL.batch1.backup"
else
    echo "⚠️ No signals file found (Batch 1 may not have run)"
fi

# Step 2: Clear signals for Batch 2
echo "[2/5] Clearing signals for Batch 2..."
rm -f "$SIGNALS_JSONL"
echo "✅ Signals cleared"

# Step 3: Verify TP/SL changes
echo "[3/5] Verifying ATR-based 2:1 implementation..."
if grep -q "atr_mult_tp = DEFAULT_ATR_MULT_TP" "$WORK_DIR/tp_sl_retracement.py"; then
    echo "✅ tp_sl_retracement.py using ATR-based 2:1"
else
    echo "❌ ERROR: tp_sl_retracement.py not updated!"
    exit 1
fi

# Step 4: Verify pec_batch_automation is ready
echo "[4/5] Checking automation script..."
if [ -f "$WORK_DIR/pec_batch_automation.py" ]; then
    echo "✅ pec_batch_automation.py ready"
else
    echo "❌ ERROR: pec_batch_automation.py not found!"
    exit 1
fi

# Step 5: Summary
echo "[5/5] Batch 2 startup complete"
echo ""
echo "================================"
echo "BATCH 2 READY FOR LAUNCH"
echo "================================"
echo ""
echo "System Configuration:"
echo "  TP/SL Method:      ATR-Based 2:1 RR ✅"
echo "  RR Ratio:          2:1 (Entry ± ATR for SL, Entry ± 2×ATR for TP)"
echo "  Signal Accumulation: ${BATCH_TRIGGER_COUNT:-50} signals"
echo "  Report Generation: Automatic"
echo "  Batch Number:      2 (next)"
echo ""
echo "Next Steps:"
echo "  1. Restart main.py (will use new ATR-based 2:1 TP/SL)"
echo "  2. Start pec_batch_automation.py in background"
echo "  3. Monitor accumulation: python3 count_signals_by_tf.py"
echo "  4. Reports auto-generate when 50 signals reached"
echo ""
echo "Command to start automation:"
echo "  nohup python3 $WORK_DIR/pec_batch_automation.py > pec_batch2.log 2>&1 &"
echo ""
