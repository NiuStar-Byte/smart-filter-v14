#!/usr/bin/env python3
"""
Regime-Aware Score Adjustments - Phase 2 Stage 2

Purpose: Adjust filter scores based on market regime and direction
This ensures filters are context-aware and don't produce false positives
in markets they're not suited for.

Concept:
- BULL regime: Reward LONG, penalize SHORT
- BEAR regime: Reward SHORT, penalize LONG
- RANGE regime: Require higher conviction for both

Author: Nox (Phase 2 Implementation)
Date: 2026-03-02
"""

def adjust_score_for_regime(raw_score: float, regime: str, direction: str, debug=False) -> float:
    """
    Adjust raw filter score based on market regime and entry direction.
    
    Args:
        raw_score: Raw score from filter aggregation (0-100)
        regime: 'BULL', 'BEAR', or 'RANGE'
        direction: 'LONG' or 'SHORT'
        debug: Print adjustment details
        
    Returns:
        float: Adjusted score (0-100)
    """
    
    if regime == "BULL":
        if direction == "LONG":
            # In uptrend, LONG is natural - no penalty
            adjustment_factor = 1.0
            logic = "BULL + LONG = natural trend, no adjustment"
        else:  # SHORT in BULL
            # Shorting in uptrend is counter-trend - penalize
            adjustment_factor = 0.60  # -40% adjustment
            logic = "SHORT in BULL = counter-trend, -40% penalty"
    
    elif regime == "BEAR":
        if direction == "SHORT":
            # In downtrend, SHORT is natural - no penalty
            adjustment_factor = 1.0
            logic = "BEAR + SHORT = natural trend, no adjustment"
        else:  # LONG in BEAR
            # Going long in downtrend is counter-trend - penalize
            adjustment_factor = 0.50  # -50% adjustment (worse than SHORT in BULL)
            logic = "LONG in BEAR = counter-trend, -50% penalty"
    
    else:  # RANGE
        # Choppy market - both directions require higher conviction
        # Don't change score, but will use higher threshold (see below)
        adjustment_factor = 1.0
        logic = f"{direction} in RANGE = choppy market, use higher threshold"
    
    adjusted_score = raw_score * adjustment_factor
    
    if debug or adjustment_factor != 1.0:
        print(f"[REGIME-ADJUST] Raw: {raw_score:.1f} → Adjusted: {adjusted_score:.1f} | "
              f"{direction} in {regime} | {logic}")
    
    return adjusted_score


def get_min_score_by_route_and_regime(route: str, regime: str, direction: str) -> int:
    """
    Return minimum score threshold based on route type and market regime.
    
    Higher threshold = harder to trigger signal
    Lower threshold = easier to trigger signal
    
    Strategy:
    - TREND CONTINUATION: Base threshold (most reliable route)
    - REVERSAL: Higher threshold (riskier route)
    - AMBIGUOUS: Very high threshold (uncertain route)
    - NONE: Very high threshold (no pattern detected)
    
    In RANGE regime: Increase all thresholds (choppy = false signals)
    
    Args:
        route: 'TREND CONTINUATION', 'REVERSAL', 'AMBIGUOUS', 'NONE'
        regime: 'BULL', 'BEAR', 'RANGE'
        direction: 'LONG' or 'SHORT'
        
    Returns:
        int: Minimum score threshold (0-100)
    """
    
    # Base thresholds by route
    base_thresholds = {
        "TREND CONTINUATION": 15,    # Safest route, lowest bar
        "NONE": 18,                   # No pattern = needs higher conviction
        "REVERSAL": 20,               # Riskier route = higher bar
        "AMBIGUOUS": 25,              # Uncertain = highest bar
    }
    
    min_score = base_thresholds.get(route, 16)
    
    # Adjust for regime
    if regime == "RANGE":
        # In choppy markets, require higher conviction
        min_score += 3  # +3 points for all routes in RANGE
    
    return min_score


def calculate_minimum_threshold(route: str, regime: str, direction: str, debug=False) -> tuple:
    """
    Calculate the final minimum score threshold for a signal.
    
    Returns both the threshold and a description of why.
    
    Args:
        route: Signal route
        regime: Market regime
        direction: LONG or SHORT
        debug: Print details
        
    Returns:
        tuple: (int: minimum_threshold, str: reason)
    """
    
    min_score = get_min_score_by_route_and_regime(route, regime, direction)
    
    reason = f"Route={route} | Regime={regime} | MinScore={min_score}"
    
    if debug:
        print(f"[THRESHOLD] {reason}")
    
    return min_score, reason


# Constants for easy reference
ROUTE_BASE_THRESHOLDS = {
    "TREND CONTINUATION": 15,
    "NONE": 18,
    "REVERSAL": 20,
    "AMBIGUOUS": 25,
}

REGIME_ADJUSTMENTS = {
    "BULL": {
        "LONG": 1.0,      # No adjustment
        "SHORT": 0.60,    # -40% penalty
    },
    "BEAR": {
        "LONG": 0.50,     # -50% penalty
        "SHORT": 1.0,     # No adjustment
    },
    "RANGE": {
        "LONG": 1.0,      # Use higher threshold instead
        "SHORT": 1.0,     # Use higher threshold instead
    },
}

RANGE_REGIME_THRESHOLD_BUMP = 3  # Add 3 points to all thresholds in RANGE


# Example usage in main.py:
"""
# When evaluating a signal:
raw_score = calculate_filter_score(...)  # From SmartFilter
adjusted_score = adjust_score_for_regime(raw_score, regime, direction)
min_threshold, reason = calculate_minimum_threshold(route, regime, direction)

if adjusted_score >= min_threshold:
    print(f"[SIGNAL APPROVED] Score {adjusted_score:.1f} >= threshold {min_threshold}")
    # Fire signal
else:
    print(f"[SIGNAL REJECTED] Score {adjusted_score:.1f} < threshold {min_threshold} ({reason})")
    # Skip signal
"""
