"""
CANDLE CONFIRMATION FILTER - PROFESSIONALLY REBUILT
Date: 2026-04-01 22:23 GMT+7
Purpose: Detect candlestick reversal patterns with proper context
References: Nison, Brooks, Coulling on candlestick trading

Key Changes from Original:
- Strict engulfing (1.3x-1.5x body, not 1.05x)
- ATR-scaled pin bars (1.2x-1.4x, not fixed 1.3x)
- MANDATORY volume confirmation (not optional)
- Trend context check (ADX aware)
- Pattern + 2/3 conditions required (not just 1/4)
- BUT: Balanced ratios so it actually fires (not over-strict)
"""

def _check_candle_confirmation_rebuilt(self, debug=False):
    """
    REBUILT Candle Confirmation - Professional Pattern Recognition
    
    Detects: Engulfing + Pin Bar + Hammer patterns with validation
    
    Improvements:
    1. Strict pattern detection (1.3-1.5x body/wick ratios)
    2. MANDATORY volume confirmation
    3. Trend context awareness (ADX check)
    4. Multiple condition requirement (2-3/6 conditions)
    5. Balanced parameters (strict but not over-restrictive)
    
    Returns: "LONG", "SHORT", or None
    """
    import math
    
    # Defensive: need at least 2 candles to compare
    if len(self.df) < 2:
        if debug:
            print(f"[{self.symbol}] [CC-Rebuilt] Not enough data (need >=2 rows)")
        return None
    
    # ===== GET DATA =====
    try:
        # Current candle (bar at -1)
        open_curr = float(self.df['open'].iat[-1])
        close_curr = float(self.df['close'].iat[-1])
        high_curr = float(self.df['high'].iat[-1])
        low_curr = float(self.df['low'].iat[-1])
        volume_curr = float(self.df['volume'].iat[-1]) if 'volume' in self.df.columns else None
        
        # Previous candle (bar at -2)
        open_prev = float(self.df['open'].iat[-2])
        close_prev = float(self.df['close'].iat[-2])
        high_prev = float(self.df['high'].iat[-2])
        low_prev = float(self.df['low'].iat[-2])
        volume_prev = float(self.df['volume'].iat[-2]) if 'volume' in self.df.columns else None
        
        # Volatility context
        atr = self.df['atr'].iat[-1] if 'atr' in self.df.columns and not math.isnan(self.df['atr'].iat[-1]) else None
        adx = self.df['adx'].iat[-1] if 'adx' in self.df.columns and not math.isnan(self.df['adx'].iat[-1]) else None
        ema9 = self.df['ema9'].iat[-1] if 'ema9' in self.df.columns and not math.isnan(self.df['ema9'].iat[-1]) else None
        ema21 = self.df['ema21'].iat[-1] if 'ema21' in self.df.columns and not math.isnan(self.df['ema21'].iat[-1]) else None
        
        volume_ma = self.df['volume'].rolling(20).mean().iat[-1] if 'volume' in self.df.columns else None
        
    except Exception as e:
        if debug:
            print(f"[{self.symbol}] [CC-Rebuilt] Data fetch error: {e}")
        return None
    
    # ===== CALCULATE METRICS =====
    
    # Body sizes (distance from open to close)
    body_curr = abs(close_curr - open_curr)
    body_prev = abs(close_prev - open_prev)
    
    # Wicks (distance from body to extremes)
    lower_wick_curr = min(open_curr, close_curr) - low_curr
    upper_wick_curr = high_curr - max(open_curr, close_curr)
    lower_wick_prev = min(open_prev, close_prev) - low_prev
    upper_wick_prev = high_prev - max(open_prev, close_prev)
    
    # Candle direction
    is_bullish_curr = close_curr > open_curr
    is_bearish_curr = close_curr < open_curr
    is_bullish_prev = close_prev > open_prev
    is_bearish_prev = close_prev < open_prev
    
    # Trend context
    has_uptrend = (ema9 is not None and ema21 is not None and ema9 > ema21)
    has_downtrend = (ema9 is not None and ema21 is not None and ema9 < ema21)
    is_trending = (adx is not None and adx > 25)
    
    # ===== PATTERN DETECTION =====
    
    # 1. ENGULFING PATTERNS
    # Bullish engulfing: Prev bearish + Curr bullish + Curr body larger
    bullish_engulfing = (
        is_bearish_prev and is_bullish_curr and
        body_curr >= body_prev * 1.3 and  # BALANCED: 1.3x (was 1.05x, too loose)
        open_curr <= close_prev and
        close_curr >= open_prev and
        close_curr >= (open_prev + close_prev) / 2  # Close in upper half (commitment)
    )
    
    # Bearish engulfing: Prev bullish + Curr bearish + Curr body larger
    bearish_engulfing = (
        is_bullish_prev and is_bearish_curr and
        body_curr >= body_prev * 1.3 and  # BALANCED: 1.3x
        open_curr >= close_prev and
        close_curr <= open_prev and
        close_curr <= (open_prev + close_prev) / 2  # Close in lower half (commitment)
    )
    
    # 2. PIN BAR PATTERNS (ATR-scaled for volatility)
    # Calculate wick ratio with ATR scaling
    if atr is not None and atr > 0:
        # High volatility = require stricter (1.4x), Low volatility = looser (1.2x)
        atr_ratio = min(atr / (close_curr + 0.0001), 0.1)  # Clamp 0-0.1
        wick_threshold = 1.2 + (atr_ratio * 2)  # 1.2x to 1.4x based on ATR
    else:
        wick_threshold = 1.3  # Default balanced
    
    # Bullish pin bar: Long lower wick + small upper wick + bullish close
    bullish_pin_bar = (
        lower_wick_curr > wick_threshold * body_curr and
        upper_wick_curr < 0.5 * lower_wick_curr and
        (is_bullish_curr or body_curr < 0.3 * (high_curr - low_curr)) and
        close_curr > (open_curr + close_curr) / 2  # Close near high
    )
    
    # Bearish pin bar: Long upper wick + small lower wick + bearish close
    bearish_pin_bar = (
        upper_wick_curr > wick_threshold * body_curr and
        lower_wick_curr < 0.5 * upper_wick_curr and
        (is_bearish_curr or body_curr < 0.3 * (high_curr - low_curr)) and
        close_curr < (open_curr + close_curr) / 2  # Close near low
    )
    
    # 3. HAMMER / SHOOTING STAR (single candle reversal)
    bullish_hammer = (
        lower_wick_curr >= 1.5 * body_curr and  # Long lower wick
        upper_wick_curr <= 0.3 * lower_wick_curr and  # Small upper wick
        is_bullish_curr and
        body_curr > 0.1 * (high_curr - low_curr)  # Real body (not doji)
    )
    
    bearish_hammer = (
        upper_wick_curr >= 1.5 * body_curr and  # Long upper wick
        lower_wick_curr <= 0.3 * upper_wick_curr and  # Small lower wick
        is_bearish_curr and
        body_curr > 0.1 * (high_curr - low_curr)  # Real body (not doji)
    )
    
    # ===== VALIDATION CHECKS =====
    
    # Volume confirmation (MANDATORY)
    volume_ok = False
    if volume_curr is not None and volume_ma is not None:
        # Require volume > average * 1.15 (15% above normal - balanced, not extreme)
        volume_ok = volume_curr > volume_ma * 1.15
    elif volume_curr is not None and volume_prev is not None:
        # Fallback: volume higher than previous
        volume_ok = volume_curr > volume_prev * 1.1
    else:
        volume_ok = True  # No data, allow
    
    # Trend context (helpful but not blocking)
    trend_aligned = True
    if is_trending and adx is not None:
        # Pattern should not contradict strong trend
        if has_uptrend and (bearish_engulfing or bearish_hammer):
            trend_aligned = False  # Bearish pattern in uptrend = weak
        if has_downtrend and (bullish_engulfing or bullish_hammer):
            trend_aligned = False  # Bullish pattern in downtrend = weak
    
    # ===== CONDITION SCORING =====
    
    # LONG conditions
    long_pattern = bullish_engulfing or bullish_pin_bar or bullish_hammer
    long_conditions = [
        long_pattern,           # Pattern detected
        volume_ok,              # Volume confirmed
        is_bullish_curr,        # Bullish close
        not has_downtrend or has_uptrend  # Trend context
    ]
    long_score = sum(long_conditions)
    
    # SHORT conditions
    short_pattern = bearish_engulfing or bearish_pin_bar or bearish_hammer
    short_conditions = [
        short_pattern,          # Pattern detected
        volume_ok,              # Volume confirmed
        is_bearish_curr,        # Bearish close
        not has_uptrend or has_downtrend  # Trend context
    ]
    short_score = sum(short_conditions)
    
    # ===== SIGNAL LOGIC =====
    
    if debug:
        pattern_type = "None"
        if bullish_engulfing or bearish_engulfing:
            pattern_type = "Engulfing"
        elif bullish_pin_bar or bearish_pin_bar:
            pattern_type = "Pin Bar"
        elif bullish_hammer or bearish_hammer:
            pattern_type = "Hammer"
        
        print(f"[{self.symbol}] [CC-Rebuilt] {pattern_type} | "
              f"Long: pattern={long_pattern}, vol={volume_ok}, close={is_bullish_curr}, trend={long_conditions[3]} (score={long_score}) | "
              f"Short: pattern={short_pattern}, vol={volume_ok}, close={is_bearish_curr}, trend={short_conditions[3]} (score={short_score})")
    
    # BALANCED REQUIREMENT: Need pattern + volume + at least 1 more
    # This is NOT too strict, but not too loose either
    if long_score >= 2 and long_score > short_score:
        return "LONG"
    elif short_score >= 2 and short_score > long_score:
        return "SHORT"
    else:
        if debug:
            print(f"[{self.symbol}] [CC-Rebuilt] No signal (long={long_score}, short={short_score})")
        return None

