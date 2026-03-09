#!/bin/bash

# DUAL TRACKER SHELL WRAPPER
# Quick access to immutable/dynamic tracking

WORKSPACE="/Users/geniustarigan/.openclaw/workspace"
SCRIPT="dual_tracker_immutable_dynamic.py"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Show help
show_help() {
    cat << EOF
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${GREEN}DUAL TRACKER - Immutable Baseline + Dynamic Post-Enhancement${NC}
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${YELLOW}Usage:${NC}
  ./dual_tracker.sh [MODE]

${YELLOW}MODES:${NC}
  ${GREEN}once${NC}       Show report once and exit (default)
  ${GREEN}watch${NC}      Live monitoring - updates every 60 seconds
  ${GREEN}help${NC}       Show this help message

${YELLOW}EXAMPLES:${NC}
  # Quick one-time report
  ./dual_tracker.sh once

  # Live monitoring (keep in terminal, Ctrl+C to stop)
  ./dual_tracker.sh watch

${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
EOF
}

# Default mode
MODE="once"

# Parse arguments
if [[ $# -eq 0 ]]; then
    MODE="once"
elif [[ "$1" == "help" ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
else
    MODE="$1"
fi

# Validate mode
if [[ "$MODE" != "once" && "$MODE" != "watch" ]]; then
    echo "❌ Invalid mode: $MODE"
    echo "Use 'once', 'watch', or 'help'"
    exit 1
fi

# Change to workspace
cd "$WORKSPACE" || exit 1

# Build and run command
if [[ "$MODE" == "watch" ]]; then
    echo -e "${GREEN}🔄 Starting dual tracking - live monitoring (Ctrl+C to stop)${NC}\n"
    python3 "$SCRIPT" --watch --save
else
    echo -e "${GREEN}📊 Running dual tracker report...${NC}\n"
    python3 "$SCRIPT" --save
    echo -e "\n${GREEN}✅ Report complete${NC}"
fi
