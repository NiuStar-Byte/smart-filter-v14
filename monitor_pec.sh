#!/bin/bash
# Monitor PEC Executor Persistent - Real-time signal closure tracking
# Usage: watch -n 5 ./monitor_pec.sh
# Or run once: ./monitor_pec.sh

SIGNALS_FILE="/Users/geniustarigan/.openclaw/workspace/COMPLETE_SIGNALS.jsonl"

clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         PEC EXECUTOR PERSISTENT - LIVE MONITORING              ║"
echo "║              $(date '+%Y-%m-%d %H:%M:%S GMT%z')                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# === OPEN SIGNALS COUNT ===
echo "📊 SIGNAL STATUS:"
OPEN_COUNT=$(grep -c '"status": "OPEN"' "$SIGNALS_FILE" 2>/dev/null || echo "0")
CLOSED_COUNT=$(grep -c '"status": "TP_HIT"' "$SIGNALS_FILE" 2>/dev/null || echo "0")
SL_COUNT=$(grep -c '"status": "SL_HIT"' "$SIGNALS_FILE" 2>/dev/null || echo "0")
TIMEOUT_COUNT=$(grep -c '"status": "PEC-TIMEOUT"' "$SIGNALS_FILE" 2>/dev/null || echo "0")
STALE_COUNT=$(grep -c '"status": "STALE_TIMEOUT"' "$SIGNALS_FILE" 2>/dev/null || echo "0")

echo "  🔵 OPEN signals:        $OPEN_COUNT"
echo "  ✅ TP_HIT (closed):     $CLOSED_COUNT"
echo "  ❌ SL_HIT (stopped):    $SL_COUNT"
echo "  ⏱️  PEC-TIMEOUT:         $TIMEOUT_COUNT"
echo "  ⏰ STALE_TIMEOUT:       $STALE_COUNT"
echo ""

# === RECENT CLOSURES (last 10) ===
echo "📝 RECENT CLOSURES (last 10 status changes):"
grep -E '"status": "(TP_HIT|SL_HIT|PEC-TIMEOUT|STALE_TIMEOUT)"' "$SIGNALS_FILE" \
  | tail -10 \
  | jq -r '{symbol: .symbol, status: .status, closed_at: .closed_at, pnl_usd: .pnl_usd, timeframe: .timeframe}' 2>/dev/null \
  | awk '{
    if (NR % 5 == 1) print ""
    printf "  %s", $0
  }'
echo ""
echo ""

# === PROCESS STATUS ===
echo "🔧 PROCESS STATUS:"
PIDS=$(pgrep -f "pec_executor_persistent.py" | grep -v grep)
if [ -z "$PIDS" ]; then
  echo "  ❌ pec_executor_persistent.py is NOT running"
else
  COUNT=$(echo "$PIDS" | wc -l)
  echo "  ✅ pec_executor_persistent.py running ($COUNT process(es))"
  echo "     PIDs: $(echo $PIDS | tr '\n' ', ' | sed 's/,$//')"
fi
echo ""

# === TOTAL SIGNALS ===
TOTAL=$(wc -l < "$SIGNALS_FILE" 2>/dev/null)
echo "📊 TOTAL SIGNALS IN FILE: $TOTAL lines"
echo ""
