#!/usr/bin/env python3
"""
pec_config.py

PEC (Post-Entry Control) Configuration
Defines parameters for signal firing and backtesting.
"""

import os

# ============================================================================
# RISK-REWARD RATIO FILTERING & ENFORCEMENT
# ============================================================================

# Minimum accepted R:R ratio (never accept losing bets)
# RR < 1.25 means profit potential is < risk, reject it
MIN_ACCEPTED_RR = float(os.getenv("MIN_ACCEPTED_RR", "1.25"))

# Maximum accepted R:R ratio (cap extreme cases)
# RR > 2.75 is too greedy, unrealistic, cap it
MAX_ACCEPTED_RR = float(os.getenv("MAX_ACCEPTED_RR", "2.75"))

print(f"[PEC_CONFIG] RR ENFORCEMENT: MIN = {MIN_ACCEPTED_RR}:1, MAX = {MAX_ACCEPTED_RR}:1", flush=True)

# ============================================================================
# RR VALIDATION & ENFORCEMENT LOGIC
# ============================================================================

def validate_tp_sl_relationship(entry_price, tp_price, sl_price, direction):
    """
    Validate TP/SL relationship matches direction.
    
    LONG: TP must be > Entry, SL must be < Entry
    SHORT: TP must be < Entry, SL must be > Entry
    
    Args:
        entry_price: Entry price
        tp_price: Take Profit price
        sl_price: Stop Loss price
        direction: 'LONG' or 'SHORT'
    
    Returns:
        (is_valid, error_message)
    """
    try:
        entry = float(entry_price)
        tp = float(tp_price)
        sl = float(sl_price)
        
        if direction.upper() == 'LONG':
            if tp <= entry:
                return False, f"LONG TP ({tp}) must be > Entry ({entry})"
            if sl >= entry:
                return False, f"LONG SL ({sl}) must be < Entry ({entry})"
        elif direction.upper() == 'SHORT':
            if tp >= entry:
                return False, f"SHORT TP ({tp}) must be < Entry ({entry})"
            if sl <= entry:
                return False, f"SHORT SL ({sl}) must be > Entry ({entry})"
        else:
            return False, f"Unknown direction: {direction}"
        
        return True, None
    except Exception as e:
        return False, f"Error validating TP/SL: {e}"


def calculate_rr(entry_price, tp_price, sl_price):
    """
    Calculate Risk:Reward ratio.
    RR = (TP - Entry) / (Entry - SL)
    
    Returns: RR value or None if invalid
    """
    try:
        entry = float(entry_price)
        tp = float(tp_price)
        sl = float(sl_price)
        
        reward = tp - entry
        risk = entry - sl
        
        if risk <= 0:
            return None
        
        rr = reward / risk
        return round(rr, 3)
    except:
        return None


def enforce_rr_bounds(entry_price, tp_price, sl_price, direction, calculated_rr=None):
    """
    Enforce RR bounds by adjusting TP or SL if needed.
    
    Returns: (adjusted_tp, adjusted_sl, enforced_rr, action_taken)
    
    Actions:
    - "VALID": RR already within bounds
    - "CAPPED_MAX": RR was > 2.75, adjusted TP closer
    - "RAISED_MIN": RR was < 1.25, adjusted SL closer
    - "DEFAULT_MIN": No RR calculated, set to 1.25
    - "INVALID_RELATIONSHIP": TP/SL relationship wrong for direction
    """
    try:
        entry = float(entry_price)
        tp = float(tp_price)
        sl = float(sl_price)
        
        # Step 1: Validate TP/SL relationship
        is_valid, error_msg = validate_tp_sl_relationship(entry, tp, sl, direction)
        if not is_valid:
            return None, None, None, f"INVALID_RELATIONSHIP: {error_msg}"
        
        # Step 2: Calculate RR
        if calculated_rr is None:
            calculated_rr = calculate_rr(entry, tp, sl)
        
        if calculated_rr is None:
            # No RR available, default to 1.25 by adjusting SL
            if direction.upper() == 'LONG':
                # SL = Entry - (TP - Entry) / 1.25
                sl_adjusted = entry - (tp - entry) / MIN_ACCEPTED_RR
                enforced_rr = calculate_rr(entry, tp, sl_adjusted)
                return tp, round(sl_adjusted, 8), enforced_rr, "DEFAULT_MIN"
            else:  # SHORT
                # SL = Entry + (Entry - TP) / 1.25
                sl_adjusted = entry + (entry - tp) / MIN_ACCEPTED_RR
                enforced_rr = calculate_rr(entry, tp, sl_adjusted)
                return tp, round(sl_adjusted, 8), enforced_rr, "DEFAULT_MIN"
        
        # Step 3: Check bounds
        if calculated_rr < MIN_ACCEPTED_RR:
            # Too low, raise to MIN by adjusting SL
            if direction.upper() == 'LONG':
                sl_adjusted = entry - (tp - entry) / MIN_ACCEPTED_RR
                enforced_rr = MIN_ACCEPTED_RR
                return tp, round(sl_adjusted, 8), enforced_rr, "RAISED_MIN"
            else:  # SHORT
                sl_adjusted = entry + (entry - tp) / MIN_ACCEPTED_RR
                enforced_rr = MIN_ACCEPTED_RR
                return tp, round(sl_adjusted, 8), enforced_rr, "RAISED_MIN"
        
        elif calculated_rr > MAX_ACCEPTED_RR:
            # Too high, cap to MAX by adjusting TP closer
            if direction.upper() == 'LONG':
                tp_adjusted = entry + (entry - sl) * MAX_ACCEPTED_RR
                enforced_rr = MAX_ACCEPTED_RR
                return round(tp_adjusted, 8), sl, enforced_rr, "CAPPED_MAX"
            else:  # SHORT
                tp_adjusted = entry - (sl - entry) * MAX_ACCEPTED_RR
                enforced_rr = MAX_ACCEPTED_RR
                return round(tp_adjusted, 8), sl, enforced_rr, "CAPPED_MAX"
        
        else:
            # Within bounds, no adjustment needed
            return tp, sl, calculated_rr, "VALID"
    
    except Exception as e:
        return None, None, None, f"ERROR: {e}"

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
