# Dictionary for filter weights for LONG and SHORT signals (from the table)
filter_weights_long = {
    "Fractal Zone": 1, "EMA Cloud": 2, "MACD": 1, "Momentum": 1, "HATS": 2, "Volume Spike": 5, 
    "VWAP Divergence": 0, "MTF Volume Agreement": 4, "HH/LL Trend": 2, "EMA Structure": 1, "Chop Zone": 2, 
    "Candle Confirmation": 5, "Wick Dominance": 1, "Absorption": 0, "Support/Resistance": 1, "Smart Money Bias": 1, 
    "Liquidity Pool": 1, "Spread Filter": 2, "Liquidity Awareness": 2, "Trend Continuation": 2, "Volatility Model": 0, 
    "ATR Momentum Burst": 0, "Volatility Squeeze": 2
}

filter_weights_short = {
    "Fractal Zone": 1, "EMA Cloud": 1, "MACD": 1, "Momentum": 1, "HATS": 4, "Volume Spike": 5, 
    "VWAP Divergence": 0, "MTF Volume Agreement": 4, "HH/LL Trend": 4, "EMA Structure": 1, "Chop Zone": 2, 
    "Candle Confirmation": 5, "Wick Dominance": 1, "Absorption": 0, "Support/Resistance": 1, "Smart Money Bias": 1, 
    "Liquidity Pool": 1, "Spread Filter": 2, "Liquidity Awareness": 2, "Trend Continuation": 2, "Volatility Model": 0, 
    "ATR Momentum Burst": 5, "Volatility Squeeze": 2
}

# Score Calculation - Different thresholds for LONG and SHORT
def calculate_score(results, direction):
    filter_weights = filter_weights_long if direction == 'LONG' else filter_weights_short
    score = 0
    for filter_name, passed in results.items():
        if passed:
            score += filter_weights.get(filter_name, 0)
    return score

# Pass Calculation - Separate PASSES for LONG and SHORT
def calculate_passes(results, direction):
    filter_weights = filter_weights_long if direction == 'LONG' else filter_weights_short
    passes = 0
    for filter_name, passed in results.items():
        if passed:
            passes += 1
    return passes

# Final Signal Validation - Considering thresholds for both directions
def validate_signal(score, passes, direction):
    min_score = 12  # example threshold for LONG
    min_passes = 9  # example threshold for LONG
    
    if direction == 'SHORT':
        min_score = 10  # example threshold for SHORT
        min_passes = 8  # example threshold for SHORT
    
    if score >= min_score and passes >= min_passes:
        return True
    return False

# Function to apply all logic in one place
def apply_balancing_logic(results, directional_decision_func):
    # Get Direction (LONG or SHORT) from smart_filter.py
    direction = directional_decision_func(results)
    if direction is None:
        return None
    
    # Calculate score, passes, and confidence based on the direction
    score = calculate_score(results, direction)
    passes = calculate_passes(results, direction)
    
    # Validate signal based on thresholds
    is_valid = validate_signal(score, passes, direction)
    
    # Output
    return {
        "direction": direction,
        "score": score,
        "passes": passes,
        "valid_signal": is_valid
    }
