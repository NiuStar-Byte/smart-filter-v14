# Market-driven TP/SL enhancement (NEW 2026-03-18)
# This supplements calculations.py with market-structure TP/SL logic

def calculate_tp_sl_from_df(df, entry_price, direction, regime=None):
    """
    MARKET-DRIVEN TP/SL using actual price structure.
    
    Strategy:
    1. Find recent swing highs and lows (last 20 candles)
    2. Use nearest support/resistance as natural TP/SL
    3. Calculate RR from actual market distances
    4. Quality gates: Reject if RR < 0.5 or RR > 4.0
    
    Falls back to ATR if no clear S/R structure found.
    """
    try:
        import pandas as pd
        if df is None or len(df) < 5:
            return None
        
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        current_price = close.iloc[-1]
        
        # Find recent swing highs and lows (last 20 candles)
        lookback = min(20, len(df))
        recent_highs = high.tail(lookback).nlargest(3).values
        recent_lows = low.tail(lookback).nsmallest(3).values
        
        # Filter for actual support/resistance levels
        supports = sorted([x for x in recent_lows if x < current_price], reverse=True)
        resistances = sorted([x for x in recent_highs if x > current_price])
        
        # Determine TP/SL based on direction
        if direction.upper() == "LONG":
            if not supports:
                return None
            
            sl = supports[0]  # Nearest support
            
            if resistances:
                tp = resistances[0]  # Nearest resistance
                source = "market_structure_long"
            else:
                # No resistance, use 1.25:1 ratio from SL distance
                tp_dist = (current_price - sl) * 1.25
                tp = current_price + tp_dist
                source = "hybrid_market_atr_long"
        
        else:  # SHORT
            if not resistances:
                return None
            
            sl = resistances[0]  # Nearest resistance
            
            if supports:
                tp = supports[0]  # Nearest support
                source = "market_structure_short"
            else:
                # No support, use 1.25:1 ratio from SL distance
                tp_dist = (sl - current_price) * 1.25
                tp = current_price - tp_dist
                source = "hybrid_market_atr_short"
        
        # Calculate ACTUAL RR from market structure
        reward = abs(tp - current_price)
        risk = abs(current_price - sl)
        
        if risk == 0:
            return None
        
        achieved_rr = round(reward / risk, 2)
        
        # QUALITY GATES: Reject unrealistic RR
        if achieved_rr < 0.5:
            print(f"[MARKET_DRIVEN] REJECTED {direction} at {entry_price}: RR too low ({achieved_rr} < 0.5)", flush=True)
            return None
        if achieved_rr > 4.0:
            print(f"[MARKET_DRIVEN] REJECTED {direction} at {entry_price}: RR too high ({achieved_rr} > 4.0)", flush=True)
            return None
        
        return {
            'tp': float(tp),
            'sl': float(sl),
            'achieved_rr': achieved_rr,
            'source': source,
            'reward': float(reward),
            'risk': float(risk),
            'fib_levels': None,
            'chosen_ratio': None,
            'sl_capped': False
        }
    
    except Exception as e:
        print(f"[calculate_tp_sl_from_df] Error: {e}", flush=True)
        return None
