#!/bin/bash

# Aster Trading Bot Startup Script

set -e

BOT_DIR="/Users/geniustarigan/.openclaw/workspace/aster_bot"

echo "🤖 Starting Aster Trading Bot..."
echo ""

# Check env vars
if [ -z "$ASTER_PRIVATE_KEY" ]; then
    echo "❌ Error: ASTER_PRIVATE_KEY not set"
    echo "   Run: export ASTER_PRIVATE_KEY=0x..."
    exit 1
fi

if [ -z "$ASTER_WALLET_ADDRESS" ]; then
    echo "❌ Error: ASTER_WALLET_ADDRESS not set"
    echo "   Run: export ASTER_WALLET_ADDRESS=0x..."
    exit 1
fi

if [ -z "$ASTER_API_KEY" ]; then
    echo "⚠️  Warning: ASTER_API_KEY not set (optional for public data)"
fi

echo "✅ Environment variables loaded:"
echo "   ASTER_PRIVATE_KEY: ${ASTER_PRIVATE_KEY:0:10}..."
echo "   ASTER_WALLET_ADDRESS: ${ASTER_WALLET_ADDRESS:0:10}..."
echo ""

# Start bot
cd "$BOT_DIR"
python3 aster_bot.py
