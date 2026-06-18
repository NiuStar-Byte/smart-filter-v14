#!/bin/bash

# ============================================================================
# Asterdex Entry Poster - LIVE Real-Time Progress Monitor
# Works from ANY directory - shows live entries posting NOW
# ============================================================================

LOGFILE="/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/smart-filter-v14-asterdex-integration/logs/asterdex_important.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Check if log file exists
if [ ! -f "$LOGFILE" ]; then
    echo -e "${RED}❌ ERROR: Log file not found${NC}"
    echo "Expected: $LOGFILE"
    exit 1
fi

clear
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}    Asterdex Entry Poster - LIVE Real-Time Monitor${NC}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Watching:${NC} ${LOGFILE}"
echo -e "${YELLOW}Press${NC} ${BOLD}Ctrl+C${NC} ${YELLOW}to exit${NC}"
echo ""

# Use tail -f to follow log in real-time, with line processing
tail -f "$LOGFILE" | while IFS= read -r line; do
    # Clear and redisplay header - show steps, filters, results, cooldown
    if [[ $line =~ "Step\|FILTER\|SUCCESS\|FAILURE\|COOLDOWN\|COMPLETE\|ENTRY_FAILED\|TP_FAILED\|SL_FAILED\|margin\|insufficient\|Entry Result\|posted" ]]; then
        # Color-code step lines (match all step variations)
        if [[ $line =~ Step\ 0/5 ]]; then
            echo -e "${BLUE}[STEP 0]${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')"
        elif [[ $line =~ Step\ 1/5 ]]; then
            echo -e "${BLUE}[STEP 1]${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')"
        elif [[ $line =~ Step\ 2/5 ]]; then
            echo -e "${BLUE}[STEP 2]${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')"
        elif [[ $line =~ Step\ 3/5 ]]; then
            echo -e "${BLUE}[STEP 3]${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')"
        elif [[ $line =~ Step\ 4/5 ]]; then
            echo -e "${BLUE}[STEP 4]${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')"
        elif [[ $line =~ Step\ 5/5 ]]; then
            echo -e "${GREEN}[STEP 5]✅${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')"
        elif [[ $line =~ "Entry order posted successfully" ]]; then
            echo -e "${GREEN}[ENTRY SUCCESS]${NC} Order posted"
        elif [[ $line =~ "Entry Result:" ]]; then
            # Show entry result status (SUCCESS or ERROR)
            if [[ $line =~ "SUCCESS" ]]; then
                echo -e "${GREEN}✓ Entry order SUCCESS${NC}"
            else
                echo -e "${RED}✗ Entry order $(echo "$line" | grep -o "ERROR\|TIMEOUT" || echo "FAILED")${NC}"
            fi
        elif [[ $line =~ "TIER FILTER MATCH" ]]; then
            echo -e "${GREEN}✓ TIER MATCH${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')"
        elif [[ $line =~ "MTF STRONG FILTER MATCH" ]]; then
            echo -e "${GREEN}✓ MTF MATCH${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')"
        elif [[ $line =~ "🕐 COOLDOWN:" ]]; then
            echo -e "${MAGENTA}$(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')${NC}"
        elif [[ $line =~ "COMPLETE:" ]] && [[ $line =~ "posted" ]]; then
            echo -e "${GREEN}✅✅✅ $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')${NC}"
        elif [[ $line =~ "CRITICAL FAILURE" ]]; then
            echo -e "${RED}❌ CRITICAL FAILURE - Entry order FAILED${NC}"
        elif [[ $line =~ "ENTRY_FAILED\|Entry order FAILED" ]]; then
            echo -e "${RED}❌ ENTRY_FAILED${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //' | head -c 100)"
        elif [[ $line =~ "TP_FAILED\|TP order FAILED" ]]; then
            echo -e "${RED}❌ TP_FAILED${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //' | head -c 100)"
        elif [[ $line =~ "SL_FAILED\|SL order FAILED" ]]; then
            echo -e "${RED}❌ SL_FAILED${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //' | head -c 100)"
        elif [[ $line =~ "margin is insufficient\|Margin is insufficient" ]]; then
            echo -e "${RED}⚠️  MARGIN INSUFFICIENT${NC}"
        elif [[ $line =~ "SUCCESS" ]] && [[ $line =~ "Leverage\|margin type" ]]; then
            echo -e "${GREEN}✓ SUCCESS${NC} $(echo "$line" | sed 's/.*\[ENTRY_POSTER\] //')"
        else
            # Show other important lines
            if [[ $line =~ "Partial failure\|ERROR\|FAIL" ]]; then
                echo -e "${RED}$(echo "$line")${NC}"
            fi
        fi
    fi
done
