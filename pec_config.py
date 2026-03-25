#!/usr/bin/env python3
"""
pec_config.py

PEC (Post-Entry Control) Configuration
Defines parameters for signal firing and backtesting.
"""

import os

# ============================================================================
# RISK-REWARD RATIO FILTERING
# ============================================================================

# Minimum accepted R:R ratio for signal firing
# Signals with lower RR will be filtered out (not fired)
MIN_ACCEPTED_RR = float(os.getenv("MIN_ACCEPTED_RR", "1.25"))

print(f"[PEC_CONFIG] MIN_ACCEPTED_RR = {MIN_ACCEPTED_RR}:1", flush=True)

# ============================================================================
# MAX HOLD BARS BY TIMEFRAME
# ============================================================================
# Determines how many bars to hold before timeout exit
# 2026-03-25 REDESIGN: Progressive reduction in bar count as TF increases
# Rationale: Higher TFs have clearer market structure, need fewer bars
# Result: Timeout duration increases (2h→3h→4h→6h→8h) but bar count decreases (8→6→4→3→2)

MAX_BARS_BY_TF = {
    "15min": 8,      # 8 bars × 15min = 120 min (2h 0m) hold
    "30min": 6,      # 6 bars × 30min = 180 min (3h 0m) hold
    "1h": 4,         # 4 bars × 1h = 240 min (4h 0m) hold
    "2h": 3,         # 3 bars × 2h = 360 min (6h 0m) hold - NEW!
    "4h": 2,         # 2 bars × 4h = 480 min (8h 0m) hold - NEW!
}

def get_max_bars(timeframe: str) -> int:
    """
    Get maximum hold bars for a timeframe.
    
    Args:
        timeframe: Market timeframe (e.g., "15min")
    
    Returns:
        Maximum number of bars to hold
    """
    return MAX_BARS_BY_TF.get(timeframe, 20)  # Default to 20 if unknown


# ============================================================================
# PEC BACKTESTING PARAMETERS
# ============================================================================

# Default fee (KuCoin maker fee)
DEFAULT_MAKER_FEE = float(os.getenv("MAKER_FEE", "0.001"))  # 0.1%

# Exit criteria for backtesting
EXIT_CRITERIA = {
    "TP": "Take Profit hit",
    "SL": "Stop Loss hit",
    "TIMEOUT": f"No TP/SL hit within max bars, exit at close"
}

# Batch sizes for automated backtesting
BATCH_SIZES = {
    1: 50,      # Batch 1: validate first 50 signals
    2: 150,     # Batch 2: optimize on 150 signals
    3: 300,     # Batch 3: comprehensive review on 300 signals
}

# ============================================================================
# SIGNAL STORAGE
# ============================================================================

# Path to signals JSONL file
# FIX 2026-03-06 00:14 GMT+7: Use workspace root, not relative path in submodule
SIGNALS_JSONL_PATH = os.getenv("SIGNALS_JSONL_PATH", "/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl")

# ============================================================================
# LOGGING & TRACKING
# ============================================================================

# Track RR filtering stats
TRACK_RR_FILTERING = True  # Log when signals are filtered due to low RR

# Print signal details when fired
DEBUG_SIGNAL_FIRING = os.getenv("DEBUG_SIGNAL_FIRING", "false").lower() == "true"

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate PEC configuration."""
    if MIN_ACCEPTED_RR < 1.0:
        print("[PEC_CONFIG] WARNING: MIN_ACCEPTED_RR < 1.0 (no profit potential)", flush=True)
    
    for tf, bars in MAX_BARS_BY_TF.items():
        if bars < 3:
            print(f"[PEC_CONFIG] WARNING: MAX_BARS for {tf} is very low ({bars})", flush=True)
    
    print("[PEC_CONFIG] Validation complete", flush=True)


# Run validation on import
validate_config()

# ============================================================================
# REFERENCE: SIGNAL SCHEMA
# ============================================================================
"""
Each signal stored to signals_fired.jsonl contains:

{
    "uuid": "23dee4d1-680a-4435-8445-4c714cfdffd8",
    "symbol": "BTC-USDT",
    "timeframe": "15min",
    "signal_type": "LONG",
    "fired_time_utc": "2026-02-22T15:30:00Z",
    "entry_price": 42500.00,
    "tp_target": 42750.00,
    "sl_target": 42420.00,
    "tp_pct": 0.588,         # (tp - entry) / entry * 100
    "sl_pct": -0.188,        # (sl - entry) / entry * 100
    "achieved_rr": 3.125,    # TP distance / SL distance
    "fib_ratio": 0.618,      # Fibonacci level used for TP
    "atr_value": 80.0,       # ATR at signal time
    "score": 18,             # Filter score (0-20)
    "max_score": 20,
    "confidence": 90.0,      # Confidence %
    "route": "LONG_CONFIRMED",
    "regime": "UPTREND",
    "passed_gatekeepers": 7,
    "max_gatekeepers": 7,
    "stored_at_utc": "2026-02-22T15:30:05Z"
}
"""
