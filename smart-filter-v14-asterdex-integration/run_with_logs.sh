#!/bin/bash
#
# Asterdex Entry Poster - Live Log Monitor
# This script sources credentials and runs asterdex_entry_poster.py with live log output
#
# Usage:
#   ./run_with_logs.sh          # Start fresh (kill any existing instance)
#   ./run_with_logs.sh --attach # Attach to existing running instance
#

WORKSPACE="/Users/geniustarigan/.openclaw/workspace"
SCRIPT_DIR="$WORKSPACE/smart-filter-v14-asterdex-integration"
LOG_FILE="/tmp/asterdex_entry_poster_live.log"
ENV_FILE="$WORKSPACE/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  ASTERDEX ENTRY POSTER - Live Log Monitor${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ ERROR: .env file not found at $ENV_FILE${NC}"
    echo "Please create .env with ASTER_* variables"
    exit 1
fi

# Source environment variables
source "$ENV_FILE"

# Verify credentials are loaded
if [ -z "$ASTER_MAIN_ACCOUNT" ] || [ -z "$ASTER_API_WALLET_ADDRESS" ] || [ -z "$ASTER_API_WALLET_PRIVATE_KEY" ]; then
    echo -e "${RED}❌ ERROR: Wallet credentials not found in .env${NC}"
    echo "Required variables:"
    echo "  - ASTER_MAIN_ACCOUNT"
    echo "  - ASTER_API_WALLET_ADDRESS"
    echo "  - ASTER_API_WALLET_PRIVATE_KEY"
    exit 1
fi

echo -e "${GREEN}✅ Credentials loaded:${NC}"
echo "   Main Account: ${ASTER_MAIN_ACCOUNT:0:10}...${ASTER_MAIN_ACCOUNT: -8}"
echo "   API Wallet:   ${ASTER_API_WALLET_ADDRESS:0:10}...${ASTER_API_WALLET_ADDRESS: -8}"
echo ""

# Handle --attach flag (don't restart, just tail existing logs)
if [ "$1" == "--attach" ]; then
    echo -e "${YELLOW}📊 Attaching to existing asterdex_entry_poster process...${NC}"
    echo ""
    if pgrep -f "asterdex_entry_poster.py" > /dev/null; then
        echo -e "${GREEN}✅ Process found. Tailing live logs:${NC}"
        echo ""
        tail -f "$LOG_FILE"
    else
        echo -e "${RED}❌ No running asterdex_entry_poster process found${NC}"
        echo "Start with: ./run_with_logs.sh"
        exit 1
    fi
    exit 0
fi

# Check if instance already running
EXISTING_PID=$(pgrep -f "asterdex_entry_poster.py" | head -1)
if [ ! -z "$EXISTING_PID" ]; then
    echo -e "${YELLOW}⚠️  asterdex_entry_poster.py already running (PID: $EXISTING_PID)${NC}"
    echo "Killing existing instance..."
    kill -9 $EXISTING_PID
    sleep 2
    echo -e "${GREEN}✅ Previous instance killed${NC}"
    echo ""
fi

# Clear old log and start fresh
> "$LOG_FILE"

# Start asterdex_entry_poster with log redirection
echo -e "${BLUE}🚀 Starting asterdex_entry_poster.py...${NC}"
echo ""

cd "$SCRIPT_DIR"
python3 asterdex_entry_poster.py >> "$LOG_FILE" 2>&1 &
PID=$!

sleep 3

# Verify process started (check if any asterdex_entry_poster is running)
ACTUAL_PID=$(pgrep -f "asterdex_entry_poster.py" | head -1)
if [ ! -z "$ACTUAL_PID" ]; then
    echo -e "${GREEN}✅ Process started successfully (PID: $ACTUAL_PID)${NC}"
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  LIVE LOG MONITOR (Forward logs only)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    echo -e "${YELLOW}⏳ Initial signal scan in progress (wait 10-12 seconds)...${NC}"
    echo -e "${YELLOW}   This checks historical signals for Tier/MTF matches${NC}"
    echo ""
    
    # Wait for initial scan to complete
    sleep 12
    
    echo -e "${GREEN}✅ Scan complete. Now following NEW signals only:${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Follow the log file continuously (shows only new entries from now on)
    tail -f "$LOG_FILE"
else
    echo -e "${RED}❌ Failed to start process${NC}"
    echo "Check log: $LOG_FILE"
    tail -50 "$LOG_FILE"
    exit 1
fi
