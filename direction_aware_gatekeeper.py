#!/usr/bin/env python3
"""
Direction-Aware Gatekeeper Module - Phase 2 Redesign

PROBLEM WITH ORIGINAL PHASE 2:
- Hard gates assumed LONG was "natural" and applied same criteria to SHORT
- Result: SHORT signals filtered even in favorable BEAR regime
- BEAR SHORT signals: 111 (Phase 1) → 1 (Phase 2) - 99% reduction

SOLUTION: Redesigned gates that are regime-aware AND direction-aware
- BEAR regime: Favor SHORT (lower thresholds), penalize LONG (higher thresholds)
- BULL regime: Favor LONG (lower thresholds), penalize SHORT (higher thresholds)
- RANGE regime: Require high conviction for both

Author: Nox (Phase 2 Redesign)
Date: 2026-03-03
"""

import pandas as pd
import math
from calculations import compute_rsi, compute_atr

class DirectionAwareGatekeeper:
    """
    4 Independent gates, now REGIME-AWARE and DIRECTION-AWARE.
    Each gate adapts criteria based on market regime and signal direction.
    """
    
    @staticmethod
    def gate_1_momentum_price_alignment(df, direction: str, regime: str, debug=False) -> bool:
        """
        GATE 1: Momentum-Price Alignment (Direction-Aware)
        
        Checks if price direction aligns with momentum (RSI).
        Criteria adapt per regime:
        - BULL: LONG easy (price rising), SHORT hard (needs reversal signal)
        - BEAR: SHORT easy (price falling), LONG hard (needs reversal signal)
        - RANGE: Both require neutral momentum (RSI 40-60)
        
        Args:
            df: DataFrame with OHLCV + indicators
            direction: 'LONG' or 'SHORT'
            regime: 'BULL', 'BEAR', or 'RANGE'
            debug: Print details
            
        Returns:
            bool: True if gate passes
        """
        try:
            close_current = df['close'].iat[-1]
            close_prev = df['close'].iat[-2]
            rsi = compute_rsi(df).iat[-1]
            
            if math.isnan(rsi):
                return True  # Can't evaluate, pass through
            
            if regime == "BULL":
                if direction == "LONG":
                    # LONG in BULL: Easy, just need price rising + momentum
                    price_rising = close_current > close_prev
                    momentum_ok = rsi < 80  # More lenient in BULL for LONG
                    passed = price_rising and momentum_ok
                    
                    if debug:
                        status = "✓" if passed else "✗"
                        print(f"[GATE1-BULL-LONG] Price: {'↑' if price_rising else '↓'} | RSI: {rsi:.1f} (need <80) | {status}")
                    return passed
                else:  # SHORT in BULL
                    # SHORT in BULL: Hard, needs reversal (price oversold)
                    price_falling = close_current < close_prev
                    oversold = rsi < 30  # Strict: must be oversold
                    passed = price_falling and oversold
                    
                    if debug:
                        status = "✓" if passed else "✗"
                        print(f"[GATE1-BULL-SHORT] Price: {'↓' if price_falling else '↑'} | RSI: {rsi:.1f} (need <30) | {status}")
                    return passed
            
            elif regime == "BEAR":
                if direction == "SHORT":
                    # SHORT in BEAR: Easy, just need price falling + momentum
                    price_falling = close_current < close_prev
                    momentum_ok = rsi < 80  # More lenient in BEAR for SHORT (allow anything not overbought)
                    passed = price_falling and momentum_ok
                    
                    if debug:
                        status = "✓" if passed else "✗"
                        print(f"[GATE1-BEAR-SHORT] Price: {'↓' if price_falling else '↑'} | RSI: {rsi:.1f} (need >20) | {status}")
                    return passed
                else:  # LONG in BEAR
                    # LONG in BEAR: Hard, needs reversal (price overbought)
                    price_rising = close_current > close_prev
                    overbought = rsi > 70  # Strict: must be overbought
                    passed = price_rising and overbought
                    
                    if debug:
                        status = "✓" if passed else "✗"
                        print(f"[GATE1-BEAR-LONG] Price: {'↑' if price_rising else '↓'} | RSI: {rsi:.1f} (need >70) | {status}")
                    return passed
            
            else:  # RANGE
                # RANGE: Both directions need high conviction (neutral momentum)
                price_direction_ok = (direction == "LONG" and close_current > close_prev) or \
                                    (direction == "SHORT" and close_current < close_prev)
                neutral_momentum = 40 < rsi < 60  # Must be neutral zone
                passed = price_direction_ok and neutral_momentum
                
                if debug:
                    status = "✓" if passed else "✗"
                    print(f"[GATE1-RANGE] Price: {'↑' if close_current > close_prev else '↓'} | RSI: {rsi:.1f} (need 40-60) | {status}")
                return passed
        
        except Exception as e:
            if debug:
                print(f"[GATE1] Error: {e}")
            return True
    
    @staticmethod
    def gate_3_trend_alignment_direction_aware(df, direction: str, regime: str, debug=False) -> bool:
        """
        GATE 3: Trend Alignment (Redesigned for Direction-Awareness)
        
        PROBLEM: Original gate assumed LONG is "natural" trend
        
        SOLUTION: Adapt gate logic per regime:
        - BULL regime: LONG aligns with trend (easy), SHORT breaks trend (hard)
        - BEAR regime: SHORT aligns with trend (easy), LONG breaks trend (hard)
        - RANGE regime: Both are risky, require explicit reversal signals
        
        Args:
            df: DataFrame with indicators (close, MA20)
            direction: 'LONG' or 'SHORT'
            regime: 'BULL', 'BEAR', or 'RANGE'
            debug: Print details
            
        Returns:
            bool: True if gate passes
        """
        try:
            close_current = df['close'].iat[-1]
            
            # Calculate trend detection
            if len(df) >= 20:
                ma20 = df['close'].iloc[-20:].mean()
            else:
                return True  # Insufficient data, pass
            
            if regime == "BULL":
                if direction == "LONG":
                    # LONG in BULL: Easy, trend is bullish
                    above_ma = close_current > ma20
                    
                    if debug:
                        print(f"[GATE3-BULL-LONG] Price: {close_current:.6f} vs MA20: {ma20:.6f} | "
                              f"{'above (✓ trend-aligned)' if above_ma else 'below (✗)'}")
                    return above_ma
                else:  # SHORT in BULL
                    # SHORT in BULL: Hard, must be below MA20 + showing reversal
                    below_ma = close_current < ma20
                    
                    # Additional check: Recent lower highs
                    if len(df) >= 5:
                        recent_highs = df['high'].iloc[-5:].values
                        lower_highs = all(recent_highs[i] >= recent_highs[i+1] for i in range(len(recent_highs)-1))
                    else:
                        lower_highs = False
                    
                    passed = below_ma and lower_highs
                    
                    if debug:
                        print(f"[GATE3-BULL-SHORT] Price below MA20: {below_ma} | Lower highs: {lower_highs} | "
                              f"{'✓ reversal pattern' if passed else '✗ no reversal'}")
                    return passed
            
            elif regime == "BEAR":
                if direction == "SHORT":
                    # SHORT in BEAR: Easy, trend is bearish
                    below_ma = close_current < ma20
                    
                    if debug:
                        print(f"[GATE3-BEAR-SHORT] Price: {close_current:.6f} vs MA20: {ma20:.6f} | "
                              f"{'below (✓ trend-aligned)' if below_ma else 'above (✗)'}")
                    return below_ma
                else:  # LONG in BEAR
                    # LONG in BEAR: Hard, must be above MA20 + showing reversal
                    above_ma = close_current > ma20
                    
                    # Additional check: Recent higher lows
                    if len(df) >= 5:
                        recent_lows = df['low'].iloc[-5:].values
                        higher_lows = all(recent_lows[i] <= recent_lows[i+1] for i in range(len(recent_lows)-1))
                    else:
                        higher_lows = False
                    
                    passed = above_ma and higher_lows
                    
                    if debug:
                        print(f"[GATE3-BEAR-LONG] Price above MA20: {above_ma} | Higher lows: {higher_lows} | "
                              f"{'✓ reversal pattern' if passed else '✗ no reversal'}")
                    return passed
            
            else:  # RANGE
                # RANGE: Both need reversal confirmation (neither direction is "natural")
                return False  # In RANGE, both LONG and SHORT are risky - fail this gate
        
        except Exception as e:
            if debug:
                print(f"[GATE3] Error: {e}")
            return True
    
    @staticmethod
    def gate_4_candle_structure_direction_aware(df, direction: str, regime: str, debug=False) -> bool:
        """
        GATE 4: Candle Structure (Direction & Regime Aware)
        
        PROBLEM: Original gate only looked for bull candle patterns
        
        SOLUTION: Use regime-specific patterns:
        - BULL LONG: Bull candle (close near high, small lower wick)
        - BULL SHORT: Bear candle (close near low, small upper wick)
        - BEAR SHORT: Bear candle (close near low)
        - BEAR LONG: Bull candle (close near high)
        - RANGE: Both must have very tight range (no big wicks)
        
        Args:
            df: DataFrame with OHLCV
            direction: 'LONG' or 'SHORT'
            regime: 'BULL', 'BEAR', or 'RANGE'
            debug: Print details
            
        Returns:
            bool: True if gate passes
        """
        try:
            high = df['high'].iat[-1]
            low = df['low'].iat[-1]
            close = df['close'].iat[-1]
            open_ = df['open'].iat[-1]
            
            candle_range = high - low
            if candle_range == 0:
                return True  # No range, indeterminate
            
            body_size = abs(close - open_) / candle_range
            upper_wick = (high - max(close, open_)) / candle_range
            lower_wick = (min(close, open_) - low) / candle_range
            
            if regime == "BULL":
                if direction == "LONG":
                    # Bull candle: large body, close near high, small lower wick
                    is_bull = close > open_
                    body_ok = body_size > 0.4
                    wick_ratio = lower_wick < upper_wick  # Lower wick < upper wick
                    passed = is_bull and body_ok and wick_ratio
                    
                    if debug:
                        print(f"[GATE4-BULL-LONG] Bull candle: {passed} (body: {body_size:.2f}, "
                              f"upper: {upper_wick:.2f}, lower: {lower_wick:.2f})")
                    return passed
                else:  # SHORT in BULL
                    # Bear candle or doji: small body, high upper wick
                    is_bear = close < open_
                    high_upper_wick = upper_wick > 0.4  # Upper wick is prominent
                    passed = is_bear or high_upper_wick  # Either bear or doji-like
                    
                    if debug:
                        print(f"[GATE4-BULL-SHORT] Bear/doji: {passed} (is_bear: {is_bear}, "
                              f"upper_wick: {upper_wick:.2f})")
                    return passed
            
            elif regime == "BEAR":
                if direction == "SHORT":
                    # Bear candle: large body, close near low, small upper wick
                    is_bear = close < open_
                    body_ok = body_size > 0.4
                    wick_ratio = upper_wick < lower_wick  # Upper wick < lower wick
                    passed = is_bear and body_ok and wick_ratio
                    
                    if debug:
                        print(f"[GATE4-BEAR-SHORT] Bear candle: {passed} (body: {body_size:.2f}, "
                              f"upper: {upper_wick:.2f}, lower: {lower_wick:.2f})")
                    return passed
                else:  # LONG in BEAR
                    # Bull candle or doji: small body, high lower wick
                    is_bull = close > open_
                    high_lower_wick = lower_wick > 0.4  # Lower wick is prominent
                    passed = is_bull or high_lower_wick  # Either bull or doji-like
                    
                    if debug:
                        print(f"[GATE4-BEAR-LONG] Bull/doji: {passed} (is_bull: {is_bull}, "
                              f"lower_wick: {lower_wick:.2f})")
                    return passed
            
            else:  # RANGE
                # RANGE: Require tight range, small body, no big wicks
                tight_range = body_size < 0.3 and upper_wick < 0.3 and lower_wick < 0.3
                
                if debug:
                    print(f"[GATE4-RANGE] Tight range: {tight_range} (body: {body_size:.2f})")
                return tight_range
        
        except Exception as e:
            if debug:
                print(f"[GATE4] Error: {e}")
            return True
    
    @staticmethod
    def check_all_gates(df, direction: str, regime: str, debug=False) -> tuple:
        """
        Check ALL 4 gates (compatible with old HardGatekeeper interface).
        
        Returns:
            tuple: (bool: all_passed, dict: gate_results)
        """
        from hard_gatekeeper import HardGatekeeper  # For Gate 2 (volume, non-directional)
        
        gate_results = {}
        
        # GATE 1: Momentum-Price Alignment (direction-aware)
        gate_results["momentum_alignment"] = DirectionAwareGatekeeper.gate_1_momentum_price_alignment(
            df, direction, regime, debug)
        
        # GATE 2: Volume Confirmation (unchanged, non-directional)
        gate_results["volume_confirmation"] = HardGatekeeper.gate_2_volume_confirmation(df, debug)
        
        # GATE 3: Trend Alignment (direction-aware redesign)
        gate_results["trend_alignment"] = DirectionAwareGatekeeper.gate_3_trend_alignment_direction_aware(
            df, direction, regime, debug)
        
        # GATE 4: Candle Structure (direction-aware redesign)
        gate_results["candle_structure"] = DirectionAwareGatekeeper.gate_4_candle_structure_direction_aware(
            df, direction, regime, debug)
        
        # Calculate passed gates
        passed_count = sum(1 for v in gate_results.values() if v)
        
        # FIXED: Apply threshold-based logic instead of ALL gates must pass
        # Favorable combos: Need 3/4 gates (75% pass rate)
        # Unfavorable combos: Need 4/4 gates (100% pass rate)
        if (regime == "BULL" and direction == "LONG") or (regime == "BEAR" and direction == "SHORT"):
            # FAVORABLE combos: Short in BEAR, Long in BULL
            all_passed = passed_count >= 3  # Need 3/4 gates
            threshold_label = "3/4 (FAVORABLE)"
        else:
            # UNFAVORABLE combos: Long in BEAR, Short in BULL, or RANGE
            all_passed = passed_count >= 4  # Need 4/4 gates
            threshold_label = "4/4 (UNFAVORABLE)"
        
        # Summary
        if debug or not all_passed:
            status_str = "✓ GATES PASS" if all_passed else "✗ GATES FAILED"
            regime_dir_str = f"{regime}+{direction}"
            print(f"\n[PHASE2-FIXED] {regime_dir_str}: {status_str} ({passed_count}/4 gates passed, need {threshold_label})\n")
        
        return all_passed, gate_results
    
    @staticmethod
    def run_all_gates(df, direction: str, regime: str, debug=False) -> bool:
        """
        Run all 4 gates. ALL must pass for signal to continue.
        
        Args:
            df: DataFrame with OHLCV + indicators
            direction: 'LONG' or 'SHORT'
            regime: 'BULL', 'BEAR', or 'RANGE'
            debug: Print details
            
        Returns:
            bool: True if ALL gates pass
        """
        gates = [
            ("GATE 1: Momentum-Price Alignment",
             DirectionAwareGatekeeper.gate_1_momentum_price_alignment(df, direction, regime, debug)),
            
            ("GATE 3: Trend Alignment",
             DirectionAwareGatekeeper.gate_3_trend_alignment_direction_aware(df, direction, regime, debug)),
            
            ("GATE 4: Candle Structure",
             DirectionAwareGatekeeper.gate_4_candle_structure_direction_aware(df, direction, regime, debug)),
        ]
        
        # GATE 2 (Volume) is applied separately in main.py as it's not direction-aware
        
        all_pass = all(result for _, result in gates)
        
        if debug or not all_pass:
            print(f"\n[GATES SUMMARY] {direction} in {regime}:")
            for gate_name, result in gates:
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  {gate_name}: {status}")
            print(f"  Overall: {'✓ ALL PASS' if all_pass else '✗ BLOCKED'}\n")
        
        return all_pass
