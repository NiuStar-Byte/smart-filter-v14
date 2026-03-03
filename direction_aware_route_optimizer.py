# direction_aware_route_optimizer.py
# Phase 3B: Route Scoring and Optimization per Direction + Regime
# Purpose: Score routes based on favorability of direction-regime combination

def calculate_route_score(route, direction, regime, reversal_strength_score=None):
    """
    Calculates route viability score based on direction + regime alignment.
    
    Scoring Logic:
    - TREND_CONTINUATION: Base 50 (always viable fallback)
    - REVERSAL: Base 70 (if quality check passes), adjusted by direction+regime
    
    Direction-Regime Adjustments:
    - SHORT in BEAR: +25 (favorable, SHORT profits from downtrend)
    - SHORT in BULL: -30 (unfavorable, SHORT counter-trend)
    - LONG in BULL: +20 (favorable, LONG profits from uptrend)
    - LONG in BEAR: -25 (unfavorable, LONG counter-trend)
    
    Args:
        route (str): "REVERSAL", "TREND_CONTINUATION", "AMBIGUOUS", or "NONE"
        direction (str): "LONG" or "SHORT"
        regime (str): "BULL", "BEAR", or "RANGE"
        reversal_strength_score (float): 0-100, from reversal_quality_gate
    
    Returns:
        int: Route score (0-100+)
    """
    
    # Base scores
    if route == "TREND_CONTINUATION":
        score = 50
    elif route == "REVERSAL":
        score = 70 if reversal_strength_score is None else reversal_strength_score
    elif route == "AMBIGUOUS":
        score = 0  # Disabled in Phase 3B
    elif route == "NONE":
        score = 0  # Disabled in Phase 3B
    else:
        score = 0
    
    # Direction + Regime multipliers (applied to base score)
    direction_regime_combo = f"{direction}_{regime}"
    
    adjustments = {
        # SHORT combos
        "SHORT_BEAR": 25,      # Favorable: SHORT profits in BEAR
        "SHORT_BULL": -30,     # Unfavorable: SHORT counter-trend in BULL
        "SHORT_RANGE": 10,     # Neutral: range-bound SHORT is OK
        
        # LONG combos
        "LONG_BULL": 20,       # Favorable: LONG profits in BULL
        "LONG_BEAR": -25,      # Unfavorable: LONG counter-trend in BEAR
        "LONG_RANGE": 5,       # Neutral: range-bound LONG is OK
    }
    
    adjustment = adjustments.get(direction_regime_combo, 0)
    
    # Apply adjustment
    # - Favorable combos get bonus to high-scoring routes
    # - Unfavorable combos get penalty to counter-trend routes
    if route == "REVERSAL":
        score += adjustment
    elif route == "TREND_CONTINUATION":
        # Trend continuation gets smaller adjustment (it's always fallback)
        score += adjustment * 0.5
    
    # Ensure score stays in reasonable bounds
    score = max(0, min(100, score))
    
    return score


def get_route_recommendation(route, direction, regime, reversal_strength_score=None):
    """
    Returns a recommendation string for route selection.
    
    Args:
        route (str): The selected route
        direction (str): LONG or SHORT
        regime (str): BULL, BEAR, or RANGE
        reversal_strength_score (float): 0-100 from reversal_quality_gate
    
    Returns:
        str: Human-readable recommendation
    """
    
    combo = f"{direction}_{regime}"
    
    if route == "REVERSAL":
        if combo == "SHORT_BEAR":
            return "✅ REVERSAL: Bearish reversal SHORT in BEAR (favorable)"
        elif combo == "LONG_BULL":
            return "✅ REVERSAL: Bullish reversal LONG in BULL (favorable)"
        elif combo == "SHORT_BULL":
            return "⚠️ REVERSAL: SHORT reversal in BULL (counter-trend, risky)"
        elif combo == "LONG_BEAR":
            return "⚠️ REVERSAL: LONG reversal in BEAR (counter-trend, risky)"
        else:
            return f"⚠️ REVERSAL: {combo} (neutral)"
    
    elif route == "TREND_CONTINUATION":
        if combo == "SHORT_BEAR":
            return "✅ TREND_CONT: SHORT trend continuation in BEAR (favorable)"
        elif combo == "LONG_BULL":
            return "✅ TREND_CONT: LONG trend continuation in BULL (favorable)"
        else:
            return f"⚠️ TREND_CONT: {combo} (fallback route)"
    
    else:
        return f"❌ {route} (disabled in Phase 3B)"


def route_filtering_rules(route, direction, regime):
    """
    Returns filtering rules for a given route + direction + regime combo.
    
    Rules determine if a signal with specific route/direction/regime should be:
    - ALLOW (allowed to be sent)
    - FILTER (rejected)
    - WARN (sent but with warning)
    
    Args:
        route (str): REVERSAL, TREND_CONTINUATION, etc.
        direction (str): LONG or SHORT
        regime (str): BULL, BEAR, RANGE
    
    Returns:
        dict: {
            "action": "ALLOW" | "FILTER" | "WARN",
            "reason": "explanation"
        }
    """
    
    rules = {
        # TREND_CONTINUATION: Always allowed (primary route)
        ("TREND_CONTINUATION", "LONG", "BULL"): {"action": "ALLOW", "reason": "Favorable combo"},
        ("TREND_CONTINUATION", "LONG", "BEAR"): {"action": "ALLOW", "reason": "Fallback OK"},
        ("TREND_CONTINUATION", "LONG", "RANGE"): {"action": "ALLOW", "reason": "Neutral combo"},
        ("TREND_CONTINUATION", "SHORT", "BULL"): {"action": "ALLOW", "reason": "Fallback OK"},
        ("TREND_CONTINUATION", "SHORT", "BEAR"): {"action": "ALLOW", "reason": "Favorable combo"},
        ("TREND_CONTINUATION", "SHORT", "RANGE"): {"action": "ALLOW", "reason": "Neutral combo"},
        
        # REVERSAL: Direction-dependent
        ("REVERSAL", "LONG", "BULL"): {"action": "ALLOW", "reason": "Favorable reversal"},
        ("REVERSAL", "LONG", "BEAR"): {"action": "WARN", "reason": "Counter-trend reversal, risky"},
        ("REVERSAL", "SHORT", "BEAR"): {"action": "ALLOW", "reason": "Favorable reversal"},
        ("REVERSAL", "SHORT", "BULL"): {"action": "WARN", "reason": "Counter-trend reversal, risky"},
        
        # AMBIGUOUS & NONE: Disabled
        ("AMBIGUOUS", _, _): {"action": "FILTER", "reason": "Disabled in Phase 3B"},
        ("NONE", _, _): {"action": "FILTER", "reason": "Disabled in Phase 3B"},
    }
    
    # Lookup with wildcard fallback
    key = (route, direction, regime)
    if key in rules:
        return rules[key]
    
    # Wildcard fallback (catch-all)
    return {"action": "FILTER", "reason": f"Unknown combo: {key}"}


def format_route_log_tag(route, direction, regime, score):
    """
    Formats a log tag for route decision logging.
    
    Returns:
        str: e.g., "[PHASE3B-ROUTE] REVERSAL LONG BEAR (score: 65)"
    """
    return f"[PHASE3B-ROUTE] {route} {direction} {regime} (score: {score})"
