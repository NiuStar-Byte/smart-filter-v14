#!/bin/bash

# FILTER FAILURE TRACKER - Shell Wrapper
# Quick access to filter analysis with 2 modes: once or live monitoring

set -e

WORKSPACE="/Users/geniustarigan/.openclaw/workspace"
SCRIPT="filter_failure_inference.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Show usage
show_help() {
    cat << EOF
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${GREEN}FILTER FAILURE TRACKER - Command Line Interface${NC}
${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${YELLOW}Usage:${NC}
  ./filter_tracker.sh [MODE] [OPTIONS]

${YELLOW}MODES:${NC}
  ${GREEN}once${NC}       Show report once and exit (default)
  ${GREEN}watch${NC}      Live monitoring - updates every 5 seconds
  ${GREEN}help${NC}       Show this help message

${YELLOW}OPTIONS:${NC}
  ${GREEN}--export${NC}   Save results to CSV file
  ${GREEN}--limit N${NC}  Analyze only last N signals (faster)

${YELLOW}EXAMPLES:${NC}
  # Quick one-time report
  ./filter_tracker.sh once

  # Live monitoring with CSV updates
  ./filter_tracker.sh watch --export

  # Analyze only last 500 signals
  ./filter_tracker.sh once --limit 500

${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
EOF
}

# Default mode
MODE="once"
ARGS=""

# Parse arguments
if [[ $# -eq 0 ]]; then
    MODE="once"
elif [[ "$1" == "help" ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
else
    MODE="$1"
    shift
    # Pass remaining args to Python script
    ARGS="$@"
fi

# Validate mode
if [[ "$MODE" != "once" && "$MODE" != "watch" ]]; then
    echo -e "${RED}❌ Invalid mode: $MODE${NC}"
    echo "Use 'once', 'watch', or 'help'"
    exit 1
fi

# Change to workspace
cd "$WORKSPACE" || exit 1

# Build Python command
if [[ "$MODE" == "watch" ]]; then
    echo -e "${GREEN}🔄 Starting live monitoring (Ctrl+C to stop)${NC}\n"
    python3 "$SCRIPT" --watch $ARGS
else
    echo -e "${GREEN}📊 Running filter analysis...${NC}\n"
    python3 "$SCRIPT" $ARGS
    echo -e "\n${GREEN}✅ Analysis complete${NC}"
fi
