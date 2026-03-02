#!/usr/bin/env python3
"""
Hard Gatekeeper Module - Phase 2 Stage 2 Implementation

4 Independent gates - ALL must pass before proceeding to score-based filtering.
Each gate tests a different aspect of market structure.

Purpose: Block false positives BEFORE they reach the scoring system

Author: Nox (Phase 2 Implementation)
Date: 2026-03-02
"""

import pandas as pd
import math
from calculations import compute_rsi, compute_atr

class HardGatekeeper:
    """
    4 Independent gates. ALL must pass before proceeding to score-based filters.
    Each gate is independent and tests a different market structure aspect.
    """
    
    @staticmethod
    def gate_1_momentum_price_alignment(df, direction: str, debug=False) -> bool:
        """
        GATE 1: Momentum Divergence Test
        
        Check if price direction aligns with momentum (RSI).
        Purpose: Avoid entries against momentum (high risk)
        
        LONG Entry: Price rising AND RSI not overbought (RSI < 75)
        SHORT Entry: Price falling AND RSI not oversold (RSI > 25)
        
        Args:
            df: DataFrame with OHLCV + indicators
            direction: 'LONG' or 'SHORT'
            debug: Print details
            
        Returns:
            bool: True if momentum aligns with direction
        """
        try:
            close_current = df['close'].iat[-1]
            close_prev = df['close'].iat[-2]
            rsi = compute_rsi(df).iat[-1]
            
            if math.isnan(rsi):
                if debug:
                    print(f"[GATE1] RSI is NaN, skipping momentum check")
                return True  # Can't evaluate, pass through
            
            if direction == "LONG":
                # Price should be rising
                price_rising = close_current > close_prev
                # RSI should not be overbought
                not_overbought = rsi < 75
                
                passed = price_rising and not_overbought
                
                if debug:
                    print(f"[GATE1-LONG] Price: {close_prev:.6f} → {close_current:.6f} "
                          f"({'+' if price_rising else '-'}) | RSI: {rsi:.1f} "
                          f"({'✓' if not_overbought else '✗ overbought'}) | "
                          f"Result: {'PASS' if passed else 'FAIL'}")
                
                return passed
            
            else:  # SHORT
                # Price should be falling
                price_falling = close_current < close_prev
                # RSI should not be oversold
                not_oversold = rsi > 25
                
                passed = price_falling and not_oversold
                
                if debug:
                    print(f"[GATE1-SHORT] Price: {close_prev:.6f} → {close_current:.6f} "
                          f"({'-' if price_falling else '+'}) | RSI: {rsi:.1f} "
                          f"({'✓' if not_oversold else '✗ oversold'}) | "
                          f"Result: {'PASS' if passed else 'FAIL'}")
                
                return passed
        
        except Exception as e:
            print(f"[GATE1] Error: {e}")
            return True  # Fail gracefully
    
    @staticmethod
    def gate_2_volume_confirmation(df, debug=False) -> bool:
        """
        GATE 2: Volume Confirmation Test
        
        Check if current volume is above recent average.
        Purpose: Ensure conviction (avoid low-volume noise trades)
        
        Threshold: Current volume > 120% of 20-day SMA
        
        Args:
            df: DataFrame with OHLCV
            debug: Print details
            
        Returns:
            bool: True if volume confirms the move
        """
        try:
            vol_current = df['volume'].iat[-1]
            vol_ma20 = df['volume'].rolling(20).mean().iat[-1]
            threshold = vol_ma20 * 1.2
            
            if pd.isna(vol_ma20) or pd.isna(vol_current):
                if debug:
                    print(f"[GATE2] Volume data missing, skipping")
                return True
            
            passed = vol_current > threshold
            
            if debug:
                print(f"[GATE2] Volume: {vol_current:,.0f} vs 20d-MA: {vol_ma20:,.0f} "
                      f"(threshold: {threshold:,.0f}) | "
                      f"Result: {'PASS' if passed else 'FAIL'}")
            
            return passed
        
        except Exception as e:
            print(f"[GATE2] Error: {e}")
            return True
    
    @staticmethod
    def gate_3_trend_alignment(df, direction: str, regime: str, debug=False) -> bool:
        """
        GATE 3: Trend Alignment Test
        
        Check if entry direction aligns with market regime.
        Purpose: Don't fight the regime
        
        BULL Regime: LONG OK, SHORT heavily penalized
        BEAR Regime: SHORT OK, LONG heavily penalized
        RANGE Regime: Both allowed but at higher conviction threshold
        
        Args:
            df: DataFrame with EMA200
            direction: 'LONG' or 'SHORT'
            regime: 'BULL', 'BEAR', or 'RANGE'
            debug: Print details
            
        Returns:
            bool: True if direction aligns with regime
        """
        try:
            close = df['close'].iat[-1]
            ema200 = df['ema200'].iat[-1]
            
            if pd.isna(ema200) or pd.isna(close):
                if debug:
                    print(f"[GATE3] EMA200 or close missing, skipping")
                return True
            
            if regime == "BULL":
                if direction == "LONG":
                    # LONG in bull: price should be above EMA200
                    passed = close > ema200
                    if debug:
                        print(f"[GATE3-BULL-LONG] Close: {close:.6f} vs EMA200: {ema200:.6f} | "
                              f"Result: {'PASS (long bullish)' if passed else 'FAIL (long bearish)'}")
                    return passed
                else:  # SHORT in BULL
                    # SHORT in bull: NOT ALLOWED
                    if debug:
                        print(f"[GATE3-BULL-SHORT] SHORT in BULL regime is not recommended | Result: FAIL")
                    return False
            
            elif regime == "BEAR":
                if direction == "SHORT":
                    # SHORT in bear: price should be below EMA200
                    passed = close < ema200
                    if debug:
                        print(f"[GATE3-BEAR-SHORT] Close: {close:.6f} vs EMA200: {ema200:.6f} | "
                              f"Result: {'PASS (short bearish)' if passed else 'FAIL (short bullish)'}")
                    return passed
                else:  # LONG in BEAR
                    # LONG in bear: NOT ALLOWED
                    if debug:
                        print(f"[GATE3-BEAR-LONG] LONG in BEAR regime is not recommended | Result: FAIL")
                    return False
            
            else:  # RANGE
                # In range, both directions are allowed (but will be penalized in scoring)
                if debug:
                    print(f"[GATE3-RANGE] RANGE regime allows both directions (will be penalized in scoring) | Result: PASS")
                return True
        
        except Exception as e:
            print(f"[GATE3] Error: {e}")
            return True
    
    @staticmethod
    def gate_4_candle_structure(df, direction: str, debug=False) -> bool:
        """
        GATE 4: Candle Structure Test
        
        Check if current candle formation is valid (not a spinning top or ambiguous).
        Purpose: Avoid ambiguous candle formations
        
        Requirements:
        - LONG: Close > Open (bullish), body > 50% of range, lower wick < upper wick
        - SHORT: Close < Open (bearish), body > 50% of range, upper wick < lower wick
        - Both: Avoid spinning tops
        
        Args:
            df: DataFrame with OHLCV
            direction: 'LONG' or 'SHORT'
            debug: Print details
            
        Returns:
            bool: True if candle structure is valid
        """
        try:
            open_ = df['open'].iat[-1]
            close = df['close'].iat[-1]
            high = df['high'].iat[-1]
            low = df['low'].iat[-1]
            
            range_ = high - low
            body = abs(close - open_)
            upper_wick = high - max(open_, close)
            lower_wick = min(open_, close) - low
            
            if range_ == 0:
                if debug:
                    print(f"[GATE4] Doji candle (range=0), skipping")
                return True
            
            # Body should be > 50% of range (not spinning top)
            is_solid_candle = body > (range_ * 0.5)
            body_pct = (body / range_) * 100 if range_ > 0 else 0
            
            if direction == "LONG":
                # Bullish candle: close > open, lower wick < upper wick
                close_above_open = close > open_
                wick_alignment = lower_wick < upper_wick
                
                passed = close_above_open and is_solid_candle and wick_alignment
                
                if debug:
                    print(f"[GATE4-LONG] Close: {close:.6f} > Open: {open_:.6f} "
                          f"({'✓' if close_above_open else '✗'}) | "
                          f"Body: {body_pct:.0f}% (need >50%) {'✓' if is_solid_candle else '✗'} | "
                          f"Wicks: Lower {lower_wick:.6f} < Upper {upper_wick:.6f} "
                          f"({'✓' if wick_alignment else '✗'}) | "
                          f"Result: {'PASS' if passed else 'FAIL'}")
                
                return passed
            
            else:  # SHORT
                # Bearish candle: close < open, upper wick < lower wick
                close_below_open = close < open_
                wick_alignment = upper_wick < lower_wick
                
                passed = close_below_open and is_solid_candle and wick_alignment
                
                if debug:
                    print(f"[GATE4-SHORT] Close: {close:.6f} < Open: {open_:.6f} "
                          f"({'✓' if close_below_open else '✗'}) | "
                          f"Body: {body_pct:.0f}% (need >50%) {'✓' if is_solid_candle else '✗'} | "
                          f"Wicks: Upper {upper_wick:.6f} < Lower {lower_wick:.6f} "
                          f"({'✓' if wick_alignment else '✗'}) | "
                          f"Result: {'PASS' if passed else 'FAIL'}")
                
                return passed
        
        except Exception as e:
            print(f"[GATE4] Error: {e}")
            return True
    
    @classmethod
    def check_all_gates(cls, df, direction: str, regime: str, debug=False) -> tuple:
        """
        Check ALL 4 gates. Fail fast on first breach.
        
        Args:
            df: DataFrame with OHLCV + indicators
            direction: 'LONG' or 'SHORT'
            regime: 'BULL', 'BEAR', or 'RANGE'
            debug: Print details for each gate
            
        Returns:
            tuple: (bool: all_passed, dict: gate_results)
        """
        
        gates_results = {}
        
        gates_results["momentum_alignment"] = cls.gate_1_momentum_price_alignment(df, direction, debug)
        gates_results["volume_confirmation"] = cls.gate_2_volume_confirmation(df, debug)
        gates_results["trend_alignment"] = cls.gate_3_trend_alignment(df, direction, regime, debug)
        gates_results["candle_structure"] = cls.gate_4_candle_structure(df, direction, debug)
        
        # ALL gates must pass
        all_passed = all(gates_results.values())
        
        # Summary
        if debug or not all_passed:
            status_str = "✓ ALL GATES PASS" if all_passed else "✗ GATES FAILED"
            passed_count = sum(1 for v in gates_results.values() if v)
            print(f"\n[GATES-SUMMARY] {status_str} ({passed_count}/4 gates passed)\n")
        
        return all_passed, gates_results


# Quick test
if __name__ == "__main__":
    # Simple test with mock data
    import numpy as np
    
    print("[TEST] Creating mock data...")
    dates = pd.date_range('2026-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000000, 5000000, 100),
    }, index=dates)
    
    # Add indicators
    from calculations import add_indicators
    df = add_indicators(df)
    
    print("[TEST] Testing gates...")
    passed, results = HardGatekeeper.check_all_gates(df, "LONG", "BULL", debug=True)
    print(f"\nResult: {passed}")
    print(f"Gate Results: {results}\n")
