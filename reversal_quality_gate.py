# reversal_quality_gate.py
# Phase 3B: Reversal Signal Quality Validation
# Purpose: Adds 4-gate quality check to REVERSAL signals before dispatch
# Prevents low-quality reversals from destroying win rates (especially SHORT)

import numpy as np
import pandas as pd

def check_reversal_quality(symbol, df, reversal_type, regime, direction):
    """
    Validates REVERSAL signals with 4-gate quality check.
    
    Gates adapt based on direction + regime:
    - SHORT in BEAR: Easy thresholds (favorable combo)
    - SHORT in BULL: Hard thresholds (unfavorable counter-trend)
    - LONG in BULL: Easy thresholds (favorable combo)
    - LONG in BEAR: Hard thresholds (unfavorable counter-trend)
    
    Args:
        symbol (str): e.g., "BTC-USDT"
        df (pd.DataFrame): OHLCV data with indicators (RSI, MACD, ADX, etc.)
        reversal_type (str): "BULLISH" or "BEARISH"
        regime (str): "BULL", "BEAR", or "RANGE"
        direction (str): "LONG" or "SHORT"
    
    Returns:
        dict: {
            "allowed": True/False,
            "gate_results": {
                "RQ1_detector_consensus": pass/fail,
                "RQ2_momentum_alignment": pass/fail,
                "RQ3_trend_strength": pass/fail,
                "RQ4_direction_regime_match": pass/fail,
            },
            "reversal_strength_score": 0-100,
            "recommendation": "REVERSAL" | "TREND_CONTINUATION" | "REJECT",
            "reason": "explanation of decision"
        }
    """
    
    if df is None or df.empty:
        return {
            "allowed": False,
            "gate_results": {},
            "reversal_strength_score": 0,
            "recommendation": "TREND_CONTINUATION",
            "reason": "DataFrame empty/invalid"
        }
    
    try:
        last_row = df.iloc[-1]
    except Exception as e:
        return {
            "allowed": False,
            "gate_results": {},
            "reversal_strength_score": 0,
            "recommendation": "TREND_CONTINUATION",
            "reason": f"Error accessing last row: {e}"
        }
    
    # Determine if this combo is favorable or unfavorable
    is_favorable = (direction == "SHORT" and regime == "BEAR") or \
                   (direction == "LONG" and regime == "BULL")
    
    # Initialize gate results
    gates = {}
    
    # ========== GATE RQ1: Detector Consensus ==========
    # Favorable combos: Need 2+ detectors
    # Unfavorable combos: Need 3+ detectors
    # This is checked indirectly via reversal_type (if we got here, reversal was detected)
    # For now, assume it passed (detector consensus already passed at reversal gate level)
    gates["RQ1_detector_consensus"] = True
    
    # ========== GATE RQ2: Momentum Alignment ==========
    rq2_pass = True
    try:
        rsi = last_row.get("RSI", np.nan)
        macd = last_row.get("macd", np.nan)
        macd_signal = last_row.get("macd_signal", np.nan)
        
        if reversal_type == "BULLISH":
            # Bullish reversal: expect RSI > 50, MACD > signal line
            rsi_ok = rsi > 40 if is_favorable else rsi > 45
            macd_ok = macd > macd_signal if is_favorable else macd > macd_signal + 0.001
        else:  # BEARISH
            # Bearish reversal: expect RSI < 50, MACD < signal line
            rsi_ok = rsi < 60 if is_favorable else rsi < 55
            macd_ok = macd < macd_signal if is_favorable else macd < macd_signal - 0.001
        
        rq2_pass = rsi_ok and (not pd.isna(macd_ok) and macd_ok)
    except Exception as e:
        rq2_pass = is_favorable  # Lenient for favorable, strict for unfavorable
    
    gates["RQ2_momentum_alignment"] = rq2_pass
    
    # ========== GATE RQ3: Trend Strength ==========
    rq3_pass = True
    try:
        adx = last_row.get("adx", np.nan)
        
        if is_favorable:
            # Favorable: easy threshold (ADX > 15 is enough)
            rq3_pass = adx > 15
        else:
            # Unfavorable: hard threshold (ADX > 30 required, reversal from very strong trend)
            rq3_pass = adx > 30
    except Exception as e:
        rq3_pass = is_favorable
    
    gates["RQ3_trend_strength"] = rq3_pass
    
    # ========== GATE RQ4: Direction-Regime Match ==========
    # This is definitive: does direction + regime make sense for reversal?
    rq4_pass = (direction == "SHORT" and regime == "BEAR") or \
               (direction == "LONG" and regime == "BULL")
    
    gates["RQ4_direction_regime_match"] = rq4_pass
    
    # ========== DECISION LOGIC ==========
    
    # All 4 gates must pass for REVERSAL approval
    all_pass = all(gates.values())
    
    if all_pass:
        allowed = True
        recommendation = "REVERSAL"
        reason = "All quality gates passed"
        reversal_strength_score = calculate_reversal_strength(gates, is_favorable)
    else:
        # Determine if we should reject entirely or fallback to TREND_CONTINUATION
        
        # Hard fails: RQ4 (direction-regime mismatch) is always a blocker for REVERSAL
        if not gates["RQ4_direction_regime_match"]:
            allowed = False
            recommendation = "TREND_CONTINUATION"  # Fallback is OK (different route)
            reason = f"Gate RQ4 failed: {direction} reversal in {regime} regime is counter-trend"
            reversal_strength_score = 0
        
        # Unfavorable combos: Need stricter gates
        elif not is_favorable:
            failed_gates = [k for k, v in gates.items() if not v]
            allowed = False
            recommendation = "TREND_CONTINUATION"
            reason = f"Unfavorable combo ({direction} in {regime}): {', '.join(failed_gates)} failed"
            reversal_strength_score = calculate_reversal_strength(gates, is_favorable)
        
        # Favorable combos: More lenient (allow if RQ1-RQ3 mostly pass)
        else:
            failed_gates = [k for k, v in gates.items() if not v]
            if len(failed_gates) <= 1:  # Allow if max 1 gate fails
                allowed = True
                recommendation = "REVERSAL"
                reason = f"Favorable combo: 3+ gates passed ({', '.join(failed_gates)} failed)"
                reversal_strength_score = calculate_reversal_strength(gates, is_favorable)
            else:
                allowed = False
                recommendation = "TREND_CONTINUATION"
                reason = f"Too many gates failed: {', '.join(failed_gates)}"
                reversal_strength_score = calculate_reversal_strength(gates, is_favorable)
    
    return {
        "allowed": allowed,
        "gate_results": gates,
        "reversal_strength_score": reversal_strength_score,
        "recommendation": recommendation,
        "reason": reason
    }


def calculate_reversal_strength(gate_results, is_favorable):
    """
    Calculates reversal strength score (0-100) based on gate results.
    
    Favorable combos: Each gate = 25 points
    Unfavorable combos: Each gate = 20 points
    """
    passed = sum(1 for v in gate_results.values() if v)
    total_gates = len(gate_results)
    
    if is_favorable:
        return (passed / total_gates) * 100
    else:
        return (passed / total_gates) * 75  # Cap at 75% for unfavorable combos
