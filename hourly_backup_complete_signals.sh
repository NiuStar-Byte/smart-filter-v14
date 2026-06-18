#!/bin/bash
# Hourly backup of COMPLETE_SIGNALS.jsonl
# - Creates .jsonl backups (all previous backups)
# - Creates .txt backups (ONLY last 5 files with all complete signal records)
# Usage: bash hourly_backup_complete_signals.sh (call hourly via cron or launchd)

set -e

WORKSPACE="/Users/geniustarigan/.openclaw/workspace"
SOURCE_FILE="$WORKSPACE/COMPLETE_SIGNALS.jsonl"
BACKUP_DIR="$WORKSPACE/backups/hourly_signals"
BACKUP_TXT_DIR="$WORKSPACE/backups/hourly_signals_txt"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
HOUR=$(date '+%Y-%m-%d_%H%M')

# Create backup directories if needed
mkdir -p "$BACKUP_DIR"
mkdir -p "$BACKUP_TXT_DIR"

# Backup files
BACKUP_FILE="$BACKUP_DIR/COMPLETE_SIGNALS_${TIMESTAMP}.jsonl"
BACKUP_TXT_FILE="$BACKUP_TXT_DIR/COMPLETE_SIGNALS_hourly_backup_${HOUR}.txt"

# Copy with atomic operation
if [ -f "$SOURCE_FILE" ]; then
    # === .jsonl BACKUP (full history) ===
    cp "$SOURCE_FILE" "$BACKUP_FILE"
    
    # === .txt BACKUP (last 5 only) ===
    # Copy COMPLETE_SIGNALS.jsonl to .txt format (same content, different extension)
    cp "$SOURCE_FILE" "$BACKUP_TXT_FILE"
    
    # Count signals for reporting
    OPEN_COUNT=$(grep -c '"status": "OPEN"' "$SOURCE_FILE" || echo "0")
    TOTAL_COUNT=$(wc -l < "$SOURCE_FILE" || echo "0")
    TP_COUNT=$(grep -c '"status": "TP_HIT"' "$SOURCE_FILE" || echo "0")
    SL_COUNT=$(grep -c '"status": "SL_HIT"' "$SOURCE_FILE" || echo "0")
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Hourly backups created:"
    echo "   - .jsonl: $BACKUP_FILE"
    echo "   - .txt:   $BACKUP_TXT_FILE"
    echo "   - Signals: OPEN=$OPEN_COUNT | TP_HIT=$TP_COUNT | SL_HIT=$SL_COUNT | TOTAL=$TOTAL_COUNT"
    
    # === CLEANUP: Keep only last 5 .txt backups ===
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 Rotating .txt backups (keeping last 5)..."
    cd "$BACKUP_TXT_DIR"
    # Count existing .txt files
    TXT_COUNT=$(ls -1 COMPLETE_SIGNALS_hourly_backup_*.txt 2>/dev/null | wc -l)
    if [ $TXT_COUNT -gt 5 ]; then
        # Delete oldest files, keeping only last 5
        TO_DELETE=$((TXT_COUNT - 5))
        ls -1t COMPLETE_SIGNALS_hourly_backup_*.txt | tail -n $TO_DELETE | while read FILE; do
            echo "   - Deleting old backup: $FILE"
            rm -f "$FILE"
        done
    fi
    
    # === CLEANUP: Keep only last 24 .jsonl backups ===
    cd "$BACKUP_DIR"
    ls -t COMPLETE_SIGNALS_*.jsonl 2>/dev/null | tail -n +25 | xargs -r rm -f
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Backup rotation complete"
    echo ""
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ ERROR: $SOURCE_FILE not found"
    exit 1
fi
