#!/usr/bin/env python3
"""
Direction-Aware Regime Adjustments - Phase 2 Redesign

PROBLEM WITH ORIGINAL:
- Applied uniform multipliers (0.60x, 0.50x) at scoring level
- Hard gates already filtered SHORT before scoring penalties applied
- By the time score adjustment happened, SHORT was already 99% reduced

SOLUTION: Replace multiplier-based approach with DIRECTION-AWARE THRESHOLDS
- BULL + LONG: Lower threshold (14) - favor trend-aligned
- BULL + SHORT: Higher threshold (24) - penalize counter-trend
- BEAR + SHORT: Lower threshold (14) - favor trend-aligned
- BEAR + LONG: Higher threshold (24) - penalize counter-trend
- RANGE: Both high (20-22) - require high conviction

This is applied AFTER gates pass, not as gate criteria.

Author: Nox (Phase 2 Redesign)
Date: 2026-03-03
"""

def get_minimum_score_threshold_direction_aware(route: str, regime: str, direction: str) -> int:
    """
    Get minimum score threshold based on REGIME, DIRECTION, and ROUTE.
    
    Strategy:
    - Trend-aligned trades (SHORT in BEAR, LONG in BULL): Lower thresholds
    - Counter-trend trades (SHORT in BULL, LONG in BEAR): Higher thresholds
    - RANGE: Everyone needs high conviction
    
    Args:
        route: 'TREND CONTINUATION', 'REVERSAL', 'AMBIGUOUS', 'NONE'
        regime: 'BULL', 'BEAR', or 'RANGE'
        direction: 'LONG' or 'SHORT'
        
    Returns:
        int: Minimum score threshold (0-100)
    """
    
    # Base thresholds by route (unchanged from Phase 2)
    base_thresholds = {
        "TREND CONTINUATION": 15,    # Safest route
        "NONE": 18,                   # No pattern
        "REVERSAL": 20,               # Riskier
        "AMBIGUOUS": 25,              # Most uncertain
    }
    
    base = base_thresholds.get(route, 16)
    
    # Apply regime + direction adjustments
    if regime == "BULL":
        if direction == "LONG":
            # Trend-aligned: LONG in BULL
            # Lower threshold by 2-3 points (favor it)
            threshold = base - 2
        else:  # SHORT in BULL
            # Counter-trend: SHORT in BULL
            # Higher threshold by 5-8 points (penalize it)
            threshold = base + 7
    
    elif regime == "BEAR":
        if direction == "SHORT":
            # Trend-aligned: SHORT in BEAR
            # Lower threshold by 2-3 points (favor it)
            threshold = base - 2
        else:  # LONG in BEAR
            # Counter-trend: LONG in BEAR
            # Higher threshold by 5-8 points (penalize it)
            threshold = base + 7
    
    else:  # RANGE
        # Both directions risky, require high conviction
        # Add 4-5 points to all
        threshold = base + 4
    
    # Clamp to reasonable range [12, 30]
    threshold = max(12, min(30, threshold))
    
    return threshold


def calculate_direction_aware_threshold(route: str, regime: str, direction: str, debug=False) -> tuple:
    """
    Calculate final minimum score threshold with explanation.
    
    Returns:
        tuple: (int: threshold, str: reason_log)
    """
    threshold = get_minimum_score_threshold_direction_aware(route, regime, direction)
    
    # Build explanation
    if regime == "BULL":
        direction_type = "trend-aligned (favorable)" if direction == "LONG" else "counter-trend (risky)"
    elif regime == "BEAR":
        direction_type = "trend-aligned (favorable)" if direction == "SHORT" else "counter-trend (risky)"
    else:
        direction_type = "uncertain (RANGE)"
    
    reason = f"Route={route} | {regime} + {direction} ({direction_type}) | Threshold={threshold}"
    
    if debug:
        print(f"[THRESHOLD] {reason}")
    
    return threshold, reason


# Configuration for quick reference
DIRECTION_AWARE_THRESHOLDS = {
    "BULL": {
        "LONG": -2,   # Adjustment to base threshold
        "SHORT": +7,  # Adjustment to base threshold
    },
    "BEAR": {
        "SHORT": -2,  # Adjustment to base threshold
        "LONG": +7,   # Adjustment to base threshold
    },
    "RANGE": {
        "LONG": +4,   # Adjustment to base threshold
        "SHORT": +4,  # Adjustment to base threshold
    },
}

ROUTE_BASE_THRESHOLDS = {
    "TREND CONTINUATION": 15,
    "NONE": 18,
    "REVERSAL": 20,
    "AMBIGUOUS": 25,
}


# Example usage in main.py:
"""
# After SmartFilter passes, before dispatching signal:

raw_score = calculate_filter_score(...)  # From SmartFilter
min_threshold, reason = calculate_direction_aware_threshold(route, regime, direction)

if raw_score >= min_threshold:
    print(f"[SIGNAL APPROVED] Score {raw_score:.1f} >= threshold {min_threshold} ({reason})")
    # Fire signal
else:
    print(f"[SIGNAL REJECTED] Score {raw_score:.1f} < threshold {min_threshold} ({reason})")
    # Skip signal
"""
