# (full file contents with only the _check_candle_confirmation function updated)
# NOTE: This block shows the whole file for context; only the Candle Confirmation function
# (def _check_candle_confirmation) was modified from the original version you provided.
# ... (the earlier imports and class definition remain unchanged)
# For brevity here I include the full file from the original source but with the corrected function.

# smart_filter.py

import datetime
import logging
import math
import os
import requests
import pandas as pd
import numpy as np
from kucoin_orderbook import get_order_wall_delta
from kucoin_density import get_resting_density
from signal_debug_log import export_signal_debug_txt
from calculations import add_indicators, compute_rsi, calculate_cci, calculate_stochrsi, compute_williams_r
from typing import Optional

# ===== SYMBOL REGISTRY ENFORCEMENT =====
try:
    from active_symbols import validate_symbol, SymbolNotRegisteredError
except ImportError:
    # Fallback if active_symbols not available (test environment)
    def validate_symbol(symbol: str, raise_on_invalid: bool = True) -> bool:
        return True
    class SymbolNotRegisteredError(ValueError):
        pass

# Control verbosity (default: quiet mode to avoid Railway logging limits)
DEBUG_FILTERS = os.getenv("DEBUG_FILTERS", "false").lower() == "true"

class SmartFilter:
    """
    Core scanner that evaluates 23+ technical / order-flow filters,
    set min treshold of filters to pass (true), set gate keepers (GK) among filters & min threshold to pass (passes),
    use separate weights of potential LONG / SHORT then decides whether a valid LONG / SHORT signal exists based on total weight,
    use order book & resting density as SuperGK (blocking/passing) signals fired to Telegram,
    """

    def __init__(
        self,
        symbol: str,
        df: pd.DataFrame,
        tf: Optional[str] = None,
        min_score: int = 10,  # FIX: 2026-03-06 00:04 GMT+7 - lowered from 15 (was blocking ALL signals). Gatekeepers + Candle Confirmation handle gating
        required_passed: Optional[int] = None,  # int or None allowed
        volume_multiplier: float = 2.25,
        liquidity_threshold: float = 0.20,
        kwargs: Optional[dict] = None
    ):
        if kwargs is None:
            kwargs = {}

        # ===== ENFORCE SYMBOL REGISTRATION =====
        # Every symbol entering the filter chain must be registered in ACTIVE_SYMBOLS
        try:
            validate_symbol(symbol, raise_on_invalid=True)
        except SymbolNotRegisteredError as e:
            raise SymbolNotRegisteredError(
                f"[SYMBOL VALIDATION] {symbol} is not registered in ACTIVE_SYMBOLS. "
                f"Add it to main.py TOKENS list before processing. Error: {e}"
            )

        self.symbol = symbol
        self.df = add_indicators(df)
        # print(f"[{self.symbol}][{tf}] Indicators:", self.df.tail(5)[['adx','ema200','close']])
                
        # --- Ensure essential columns always exist ---
        for col in ["bid", "ask", "higher_tf_volume"]:
            if self.df is not None and col not in self.df.columns:
                self.df[col] = np.nan

        # --- Debug: Print columns and last row for troubleshooting ---
        # if self.df is not None:
        #    print(f"[DEBUG] {self.symbol} Columns after indicators:", self.df.columns)
        #    print(f"[DEBUG] {self.symbol} Last row after indicators:\n", self.df.iloc[-1])
    
        self.tf = tf
        self.min_score = min_score
        self.required_passed = required_passed
        self.volume_multiplier = volume_multiplier
        self.liquidity_threshold = liquidity_threshold

        # Weights for filters (ENHANCED 2026-03-05: Fractal Zone 4.5→4.8, improved TREND)
        self.filter_weights_long = {
            "MACD": 5.0, "Volume Spike": 5.0, "Fractal Zone": 4.8, "TREND": 4.7, "Momentum": 4.9, "ATR Momentum Burst": 4.3,
            "MTF Volume Agreement": 5.0, "HH/LL Trend": 4.1, "Volatility Model": 3.9,
            "Liquidity Awareness": 5.0, "Volatility Squeeze": 3.7, "Candle Confirmation": 5.0,
            "VWAP Divergence": 3.5, "Spread Filter": 5.0, "Chop Zone": 3.3, "Liquidity Pool": 3.1, "Support/Resistance": 5.0,
            "Smart Money Bias": 2.9, "Absorption": 2.7, "Wick Dominance": 2.5
        }
        
        self.filter_weights_short = {
            "MACD": 5.0, "Volume Spike": 5.0, "Fractal Zone": 4.8, "TREND": 4.7, "Momentum": 4.9, "ATR Momentum Burst": 4.3,
            "MTF Volume Agreement": 5.0, "HH/LL Trend": 4.1, "Volatility Model": 3.9,
            "Liquidity Awareness": 5.0, "Volatility Squeeze": 3.7, "Candle Confirmation": 5.0,
            "VWAP Divergence": 3.5, "Spread Filter": 5.0, "Chop Zone": 3.3, "Liquidity Pool": 3.1, "Support/Resistance": 5.0,
            "Smart Money Bias": 2.9, "Absorption": 2.7, "Wick Dominance": 2.5
        }

        self.filter_names = list(set(self.filter_weights_long.keys()) | set(self.filter_weights_short.keys()))
        
        # Separate gatekeepers for LONG and SHORT to avoid asymmetry
        # LONG: Candle Confirmation + Support/Resistance (tight entry logic)
        # SHORT: Candle Confirmation only (looser, Support/Resistance blocks SHORT)
        self.gatekeepers_long = [
            "Candle Confirmation" # LONG/SHORT deploy Support/Resistance at Filters not at Gatekeeper
        ]
        
        self.gatekeepers_short = [
            "Candle Confirmation"  # LONG/SHORT deploy Support/Resistance at Filters not at Gatekeeper
        ]
        
        # Legacy gatekeepers (for backward compatibility in other methods)
        self.gatekeepers = self.gatekeepers_long
        
        self.soft_gatekeepers = ["Candle Confirmation"]

    # ==== SHARED HELPERS ====
    @staticmethod
    def safe_divide(a, b):
        try:
            return a / b if b else 0.0
        except Exception:
            return 0.0
    
    def get_rolling_avg(self, col, window):
        return self.df[col].rolling(window).mean().iat[-1]
    
    def is_rising(self, col, window=1):
        return self.df[col].iat[-1] > self.df[col].iat[-window-1]
    
    def is_falling(self, col, window=1):
        return self.df[col].iat[-1] < self.df[col].iat[-window-1]
    
    def proximity_to_high_low(self, close, high, low):
        return {
            "to_low": self.safe_divide(close - low, low),
            "to_high": self.safe_divide(high - close, high)
        }
            
    @property
    def filter_weights(self):
        """
        Returns the correct filter weights dict based on the current bias.
        Requires self.bias to be set before accessing this property.
        """
        direction = getattr(self, "bias", None)
        if direction == "LONG":
            return self.filter_weights_long
        elif direction == "SHORT":
            return self.filter_weights_short
        else:
            return {}
    
    def detect_ema_reversal(self, fast_period=6, slow_period=13):
        ema_fast = self.df[f"ema{fast_period}"]
        ema_slow = self.df[f"ema{slow_period}"]
        threshold = 0.001 * ema_slow.iat[-1]
    
        # Cross up: fast EMA moves from below slow EMA to above
        crossed_up = ema_fast.iat[-2] < ema_slow.iat[-2] and ema_fast.iat[-1] > ema_slow.iat[-1]
        # Cross down: fast EMA moves from above slow EMA to below
        crossed_down = ema_fast.iat[-2] > ema_slow.iat[-2] and ema_fast.iat[-1] < ema_slow.iat[-1]
    
        # Ensure nearly_equal only triggers when there is no cross (avoid overlap)
        nearly_equal_up = (
            abs(ema_fast.iat[-1] - ema_slow.iat[-1]) < threshold and
            abs(ema_fast.iat[-2] - ema_slow.iat[-2]) >= threshold and
            ema_fast.iat[-1] > ema_slow.iat[-1] and
            not crossed_down and
            not crossed_up
        )
        nearly_equal_down = (
            abs(ema_fast.iat[-1] - ema_slow.iat[-1]) < threshold and
            abs(ema_fast.iat[-2] - ema_slow.iat[-2]) >= threshold and
            ema_fast.iat[-1] < ema_slow.iat[-1] and
            not crossed_up and
            not crossed_down
        )
    
        # Final, mutually exclusive output
        if crossed_up or nearly_equal_up:
            return "BULLISH_REVERSAL"
        elif crossed_down or nearly_equal_down:
            return "BEARISH_REVERSAL"
        else:
            return "NO_REVERSAL"

    def detect_rsi_reversal(self, threshold_overbought=80, threshold_oversold=20):
        rsi = self.df['RSI']
    
        # Only allow reversal if we move out of oversold or overbought zones (transition event)
        bullish = rsi.iat[-2] <= threshold_oversold and rsi.iat[-1] > threshold_oversold
        bearish = rsi.iat[-2] >= threshold_overbought and rsi.iat[-1] < threshold_overbought
    
        # These conditions cannot both be true for the same bar
        if bullish:
            return "BULLISH_REVERSAL"
        elif bearish:
            return "BEARISH_REVERSAL"
        else:
            return "NO_REVERSAL"

    def detect_engulfing_reversal(self):
        open_ = self.df['open'].iat[-1]
        close = self.df['close'].iat[-1]
        open_prev = self.df['open'].iat[-2]
        close_prev = self.df['close'].iat[-2]
        engulf_threshold = 0.001 * open_prev
    
        # Bullish engulfing: previous candle bearish, current bullish and engulfs
        bullish = (
            close_prev < open_prev and      # prev bearish
            close > open_ and               # current bullish
            open_ < close_prev and          # current open below prev close
            close > open_prev and           # current close above prev open
            not (
                close_prev > open_prev and
                close < open_ and
                open_ > close_prev and
                close < open_prev
            )                              # explicitly exclude bearish overlap
        ) or (
            close_prev < open_prev and
            close > open_ and
            open_ < close_prev and
            abs(close - open_prev) < engulf_threshold and
            not (
                close_prev > open_prev and
                close < open_ and
                open_ > close_prev and
                abs(close - open_prev) < engulf_threshold
            )
        )
    
        # Bearish engulfing: previous candle bullish, current bearish and engulfs
        bearish = (
            close_prev > open_prev and      # prev bullish
            close < open_ and               # current bearish
            open_ > close_prev and          # current open above prev close
            close < open_prev and           # current close below prev open
            not (
                close_prev < open_prev and
                close > open_ and
                open_ < close_prev and
                close > open_prev
            )
        ) or (
            close_prev > open_prev and
            close < open_ and
            open_ > close_prev and
            abs(close - open_prev) < engulf_threshold and
            not (
                close_prev < open_prev and
                close > open_ and
                open_ < close_prev and
                abs(close - open_prev) < engulf_threshold
            )
        )
    
        # Final, mutually exclusive output
        if bullish and not bearish:
            return "BULLISH_REVERSAL"
        elif bearish and not bullish:
            return "BEARISH_REVERSAL"
        else:
            return "NO_REVERSAL"

    def detect_adx_reversal(self, adx_threshold=15):
        logger = logging.getLogger(__name__)
        required = ['adx', 'plus_di', 'minus_di']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            logger.debug(f"ADX columns missing: {missing} | Available columns: {list(self.df.columns)}")
            return "NO_REVERSAL"
    
        adx = self.df['adx']
        plus_di = self.df['plus_di']
        minus_di = self.df['minus_di']
    
        if len(adx) < 2 or len(plus_di) < 2 or len(minus_di) < 2:
            logger.debug("Not enough data to detect ADX reversal (need at least 2 rows).")
            return "NO_REVERSAL"
    
        bullish = (
            adx.iat[-1] > adx_threshold and
            plus_di.iat[-2] < minus_di.iat[-2] and
            plus_di.iat[-1] > minus_di.iat[-1]
        )
        bearish = (
            adx.iat[-1] > adx_threshold and
            plus_di.iat[-2] > minus_di.iat[-2] and
            plus_di.iat[-1] < minus_di.iat[-1]
        )
    
        logger.debug(f"ADX bullish? {bullish} | bearish? {bearish}")
    
        if bullish and not bearish:
            return "BULLISH_REVERSAL"
        elif bearish and not bullish:
            return "BEARISH_REVERSAL"
        else:
            return "NO_REVERSAL"
        
    def detect_stochrsi_reversal(self, overbought=0.8, oversold=0.2):
        required = ['stochrsi_k', 'stochrsi_d']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            return "NO_REVERSAL"
    
        k = self.df['stochrsi_k']
        d = self.df['stochrsi_d']
    
        if len(k) < 2 or len(d) < 2:
            return "NO_REVERSAL"
    
        bullish = k.iat[-2] <= oversold and k.iat[-1] > oversold and k.iat[-1] > d.iat[-1]
        bearish = k.iat[-2] >= overbought and k.iat[-1] < overbought and k.iat[-1] < d.iat[-1]
    
        if bullish:
            return "BULLISH_REVERSAL"
        elif bearish:
            return "BEARISH_REVERSAL"
        else:
            return "NO_REVERSAL"
        
    def detect_cci_reversal(self, overbought=110, oversold=-110):
        if 'cci' not in self.df.columns:
            return "NO_REVERSAL"
    
        cci = self.df['cci']
    
        if len(cci) < 2:
            return "NO_REVERSAL"
    
        bullish = cci.iat[-2] <= oversold and cci.iat[-1] > oversold
        bearish = cci.iat[-2] >= overbought and cci.iat[-1] < overbought
    
        if bullish:
            return "BULLISH_REVERSAL"
        elif bearish:
            return "BEARISH_REVERSAL"
        else:
            return "NO_REVERSAL"

    def explicit_reversal_gate(self):
        # Print all columns present in self.df for diagnostics
        # print("Columns in self.df:", self.df.columns)
        
        # Print the last 10 indicator values for all required columns, handle missing columns gracefully
        indicator_cols = ['ema6', 'ema13', 'RSI', 'adx', 'plus_di', 'minus_di', 'stochrsi_k', 'stochrsi_d', 'cci']
        # print("[INFO] Latest indicator values:")
        for col in indicator_cols:
            if col in self.df.columns:
                # print(f"{col}: {self.df[col].tail(10).values}")
                pass
            else:
                # print(f"{col}: [MISSING]")
                pass
    
        # Diagnostic: print the latest value actually used in reversal detection
        # print("[INFO] Detector check values:")
        if 'RSI' in self.df.columns:
            latest_rsi = self.df['RSI'].iloc[-1]
            # print(f"RSI latest: {latest_rsi} (thresholds: 70/30)")
        if 'adx' in self.df.columns and 'plus_di' in self.df.columns and 'minus_di' in self.df.columns:
            latest_adx = self.df['adx'].iloc[-1]
            latest_plus_di = self.df['plus_di'].iloc[-1]
            latest_minus_di = self.df['minus_di'].iloc[-1]
            # print(f"ADX latest: {latest_adx} (threshold: 10)")
            # print(f"plus_di latest: {latest_plus_di}, minus_di latest: {latest_minus_di}")
            if latest_plus_di > latest_minus_di:
                # print("DI Crossover: Bullish")
                pass
            elif latest_plus_di < latest_minus_di:
                # print("DI Crossover: Bearish")
                pass
            else:
                # print("DI Crossover: Neutral / No crossover")
                pass
    
        # Prepare reversal detectors and log their outputs
        detectors = [
            ("EMA", self.detect_ema_reversal),
            ("RSI", lambda: self.detect_rsi_reversal(threshold_overbought=70, threshold_oversold=30)),
            ("Engulfing", self.detect_engulfing_reversal),
            ("ADX", lambda: self.detect_adx_reversal(adx_threshold=10)),
            ("StochRSI", lambda: self.detect_stochrsi_reversal(overbought=0.7, oversold=0.3)),
            ("CCI", lambda: self.detect_cci_reversal(overbought=100, oversold=-100)),
        ]
        results = []
        for name, func in detectors:
            try:
                result = func()
                # print(f"[DEBUG] {name} reversal result: {result}")
                assert result in ["BULLISH_REVERSAL", "BEARISH_REVERSAL", "NO_REVERSAL"], \
                    f"{name} reversal detector returned unexpected value: {result}"
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Exception in {name} reversal detector:", e)
                results.append("NO_REVERSAL")  # Fallback on error
    
        # Inclusive logic: fire if at least 2 detectors agree and none oppose
        bullish = results.count("BULLISH_REVERSAL")
        bearish = results.count("BEARISH_REVERSAL")
    
        # --- Minimal summary log ---
        if bullish >= 2 and bearish == 0:
            print(f"[SUMMARY] Reversal decision: BULLISH (Detectors: {bullish})")
            return ("REVERSAL", "BULLISH")
        elif bearish >= 2 and bullish == 0:
            print(f"[SUMMARY] Reversal decision: BEARISH (Detectors: {bearish})")
            return ("REVERSAL", "BEARISH")
        elif bullish > 0 and bearish > 0:
            print(f"[SUMMARY] Reversal decision: AMBIGUOUS (Bullish: {bullish}, Bearish: {bearish})")
            return ("AMBIGUOUS", ["BULLISH", "BEARISH"])
        else:
            print("[SUMMARY] Reversal decision: NONE")
            return ("NONE", None)
                    
    def detect_trend_continuation(self):
        required_cols = ['ema6', 'ema13', 'macd', 'RSI']
        for col in required_cols:
            if col not in self.df.columns:
                print(f"[ERROR] '{col}' column missing in DataFrame!")
                return "NO_CONTINUATION"
    
        ema_fast = self.df['ema6'].iat[-1]
        ema_slow = self.df['ema13'].iat[-1]
        macd = self.df['macd'].iat[-1]
        rsi = self.df['RSI'].iat[-1]
        adx = self.df['adx'].iat[-1] if 'adx' in self.df.columns else None
    
        bullish_conditions = [
            ema_fast > ema_slow,
            macd > 0,
            rsi > 50,
            (adx is None or adx > 20)
        ]
        bearish_conditions = [
            ema_fast < ema_slow,
            macd < 0,
            rsi < 50,
            (adx is None or adx > 20)
        ]
    
        # More inclusive: require at least 3 out of 4 conditions
        if sum(bullish_conditions) >= 3:
            return "BULLISH_CONTINUATION"
        elif sum(bearish_conditions) >= 3:
            return "BEARISH_CONTINUATION"
        else:
            return "NO_CONTINUATION"
            
    def explicit_route_gate(self):
        reversal = self.explicit_reversal_gate()
        continuation = self.detect_trend_continuation()
        if reversal[0] in ["REVERSAL", "AMBIGUOUS"]:
            return (reversal[0], reversal[1])
        elif continuation in ["BULLISH_CONTINUATION", "BEARISH_CONTINUATION"]:
            return ("TREND CONTINUATION", continuation)
        else:
            return ("NONE", None)

    def get_signal_direction(self, results_long, results_short, debug: bool = False):
        """
        Robust signal direction decision.
    
        Rules:
        1. Hard gatekeepers have priority: if one side's hard GKs are all passed and the other's are not,
           pick the side that passed its hard GKs.
        2. If both sides pass hard GKs, choose by weighted non-GK score (require a margin to avoid flips).
        3. If neither side passes hard GKs, use a conservative fallback:
           - Compute a weighted_min_score derived from self.min_score scaled by average filter weight.
           - Only pick a side if its weighted score >= weighted_min_score AND exceeds the other side by fallback_margin.
           - Otherwise return "NEUTRAL".
        4. Soft gatekeepers are excluded from the "hard pass" requirement but are available for reporting elsewhere.
    
        Returns: "LONG", "SHORT", or "NEUTRAL".
        """
        # Soft/hard gatekeepers
        soft_gatekeepers = getattr(self, "soft_gatekeepers", ["Volume Spike"])
        hard_gatekeepers = [gk for gk in getattr(self, "gatekeepers", []) if gk not in soft_gatekeepers]
    
        # Hard GK pass booleans
        long_gk_passed = all(results_long.get(gk, False) for gk in hard_gatekeepers) if hard_gatekeepers else True
        short_gk_passed = all(results_short.get(gk, False) for gk in hard_gatekeepers) if hard_gatekeepers else True
    
        # Also expose which gatekeepers passed per side for debugging
        passed_hard_gks_long = [gk for gk in hard_gatekeepers if results_long.get(gk, False)]
        passed_hard_gks_short = [gk for gk in hard_gatekeepers if results_short.get(gk, False)]
        passed_soft_gks_long = [gk for gk in soft_gatekeepers if results_long.get(gk, False)]
        passed_soft_gks_short = [gk for gk in soft_gatekeepers if results_short.get(gk, False)]
    
        # Weighted sum for non-GK filters
        scoring_filters = [f for f in getattr(self, "filter_names", []) if f not in getattr(self, "gatekeepers", [])]
    
        long_score = sum(self.filter_weights_long.get(name, 0.0) for name in scoring_filters if results_long.get(name, False))
        short_score = sum(self.filter_weights_short.get(name, 0.0) for name in scoring_filters if results_short.get(name, False))
    
        # Also compute counts for informational purposes
        long_count = sum(1 for name in scoring_filters if results_long.get(name, False))
        short_count = sum(1 for name in scoring_filters if results_short.get(name, False))
    
        # Compute an approximate weighted_min_score using self.min_score scaled by average weight per scoring filter.
        # This avoids directly comparing "count" to "weight" scales.
        try:
            if scoring_filters:
                total_weight_long = sum(self.filter_weights_long.get(name, 0.0) for name in scoring_filters)
                total_weight_short = sum(self.filter_weights_short.get(name, 0.0) for name in scoring_filters)
                avg_weight_long = total_weight_long / len(scoring_filters)
                avg_weight_short = total_weight_short / len(scoring_filters)
                avg_weight = (avg_weight_long + avg_weight_short) / 2.0 if (avg_weight_long + avg_weight_short) > 0 else 1.0
            else:
                avg_weight = 1.0
        except Exception:
            avg_weight = 1.0
    
        # Weighted minimum score derived from configured min_score (counts) scaled to the weight units
        weighted_min_score = getattr(self, "weighted_min_score", None)
        if weighted_min_score is None:
            weighted_min_score = (getattr(self, "min_score", 0) or 0) * avg_weight
    
        # Margin to require a meaningful difference between sides when making fallback choices (in weight units)
        fallback_margin = getattr(self, "fallback_margin", 2.0)
    
        # Save debug sums for external inspection
        self._debug_sums = {
            "long_gk_passed": long_gk_passed,
            "short_gk_passed": short_gk_passed,
            "passed_hard_gks_long": passed_hard_gks_long,
            "passed_hard_gks_short": passed_hard_gks_short,
            "passed_soft_gks_long": passed_soft_gks_long,
            "passed_soft_gks_short": passed_soft_gks_short,
            "long_score": long_score,
            "short_score": short_score,
            "long_count": long_count,
            "short_count": short_count,
            "weighted_min_score": weighted_min_score,
            "fallback_margin": fallback_margin,
            "hard_gatekeepers": hard_gatekeepers,
            "soft_gatekeepers": soft_gatekeepers,
        }
    
        # Extra debug print to make GK status explicit in logs
        if debug:
            print(
                f"[{self.symbol}] get_signal_direction GK status | "
                f"hard_gatekeepers={hard_gatekeepers} soft_gatekeepers={soft_gatekeepers} | "
                f"long_gk_passed={long_gk_passed} passed_hard_gks_long={passed_hard_gks_long} passed_soft_gks_long={passed_soft_gks_long} | "
                f"short_gk_passed={short_gk_passed} passed_hard_gks_short={passed_hard_gks_short} passed_soft_gks_short={passed_soft_gks_short} | "
                f"long_score={long_score:.2f} short_score={short_score:.2f} weighted_min_score={weighted_min_score:.2f} fallback_margin={fallback_margin:.2f}"
            )
    
        # Decision logic with clear precedence and conservative fallback
        # 1) If one side passes hard GKs and the other does not -> pick that side
        if long_gk_passed and not short_gk_passed:
            if debug:
                print(f"[{self.symbol}] get_signal_direction -> LONG (hard GKs passed for LONG, not for SHORT)")
            return "LONG"
        if short_gk_passed and not long_gk_passed:
            if debug:
                print(f"[{self.symbol}] get_signal_direction -> SHORT (hard GKs passed for SHORT, not for LONG)")
            return "SHORT"
    
        # 2) If both sides pass hard GKs, pick side by weighted score, require a small margin to avoid flip-flopping
        if long_gk_passed and short_gk_passed:
            score_diff = long_score - short_score
            if debug:
                print(f"[{self.symbol}] get_signal_direction (both GKs passed) -> long_score={long_score}, short_score={short_score}, diff={score_diff}")
            if score_diff > fallback_margin:
                return "LONG"
            elif -score_diff > fallback_margin:
                return "SHORT"
            else:
                return "NEUTRAL"
    
        # 3) Neither side passes hard GKs: conservative fallback by weighted score + margin + minimum threshold
        if debug:
            print(f"[{self.symbol}] get_signal_direction (fallback) -> long_score={long_score}, short_score={short_score}, weighted_min_score={weighted_min_score}, fallback_margin={fallback_margin}")
    
        # Only allow fallback if the winning side both exceeds weighted_min_score and exceeds the other by margin
        if long_score >= weighted_min_score and (long_score - short_score) >= fallback_margin:
            if debug:
                print(f"[{self.symbol}] get_signal_direction -> LONG (fallback by weight)")
            return "LONG"
        if short_score >= weighted_min_score and (short_score - long_score) >= fallback_margin:
            if debug:
                print(f"[{self.symbol}] get_signal_direction -> SHORT (fallback by weight)")
            return "SHORT"
    
        # Otherwise neutral
        if debug:
            print(f"[{self.symbol}] get_signal_direction -> NEUTRAL (no decisive condition met)")
        return "NEUTRAL"

    def superGK_check(self, signal_direction, orderbook_result, density_result):
        """
        Super Gatekeeper has been DISABLED inside SmartFilter.
    
        Behavior:
        - This function now always returns True (does not block any signal).
        - Prints a clear diagnostic line so logs show that SuperGK was intentionally bypassed in SmartFilter.
        - main.py remains the authoritative place for final decisions, but this ensures SmartFilter's internal
          SuperGK logic will not block or influence signals.
        """
        try:
            print(f"[{self.symbol}] [SuperGK] DISABLED in SmartFilter - allowing signal ({signal_direction})", flush=True)
        except Exception:
            # Best-effort logging; do not raise
            pass
        return True

    def analyze(self):
        """
        Main analysis routine (inlined SuperGK fetch & robust logging).
        - Evaluates all filters with debug=True.
        - Calls get_signal_direction(..., debug=True) so GK status is logged.
        - Performs two quick attempts to fetch orderbook and density inline (no new function def).
        - Normalizes invalid results to safe defaults and calls self.superGK_check.
        """
        import datetime
        import time
    
        # Defensive checks
        if getattr(self, "df", None) is None or self.df.empty:
            print(f"[{getattr(self, 'symbol', 'UNKNOWN')}] Error: DataFrame empty or not provided.")
            return None
    
        print(f"[{self.symbol}] analyze() start @ {datetime.datetime.utcnow().isoformat()}")
    
        # --- Detect reversal and set route correctly (AMBIGUOUS included) ---
        try:
            reversal_route, reversal_side = self.explicit_reversal_gate()
        except Exception as e:
            print(f"[{self.symbol}] Error running explicit_reversal_gate(): {e}")
            reversal_route, reversal_side = ("NONE", None)
        reversal_detected = reversal_route in ["REVERSAL", "AMBIGUOUS"]
        route = reversal_route if reversal_detected else "TREND CONTINUATION"
    
        # --- Filters list & mapping ---
        filter_names = [
            "Fractal Zone", "TREND", "MACD", "Momentum", "Volume Spike",
            "VWAP Divergence", "MTF Volume Agreement", "HH/LL Trend",
            "Chop Zone", "Candle Confirmation", "Wick Dominance", "Absorption",
            "Support/Resistance", "Smart Money Bias", "Liquidity Pool", "Spread Filter",
            "Liquidity Awareness", "Volatility Model", "ATR Momentum Burst", "Volatility Squeeze"
        ]
    
        filter_function_map = {
            "Fractal Zone": getattr(self, "_check_fractal_zone", None),
            "TREND": getattr(self, "_check_unified_trend", None),
            "MACD": getattr(self, "_check_macd", None),
            "Momentum": getattr(self, "_check_momentum", None),
            "Volume Spike": lambda debug=False: self._check_volume_spike_detector(debug=debug),
            "VWAP Divergence": getattr(self, "_check_vwap_divergence", None),
            "MTF Volume Agreement": getattr(self, "_check_mtf_volume_agreement", None),
            "HH/LL Trend": getattr(self, "_check_hh_ll", None),
            "Chop Zone": getattr(self, "_check_chop_zone", None),
            "Candle Confirmation": getattr(self, "_check_candle_confirmation", None),
            "Wick Dominance": getattr(self, "_check_wick_dominance", None),
            "Absorption": getattr(self, "_check_absorption", None),
            "Support/Resistance": getattr(self, "_check_support_resistance", None),
            "Smart Money Bias": getattr(self, "_check_smart_money_bias", None),
            "Liquidity Pool": getattr(self, "_check_liquidity_pool", None),
            "Spread Filter": getattr(self, "_check_spread_filter", None),
            "Liquidity Awareness": getattr(self, "_check_liquidity_awareness", None),
            "Volatility Model": getattr(self, "_check_volatility_model", None),
            "ATR Momentum Burst": getattr(self, "_check_atr_momentum_burst", None),
            "Volatility Squeeze": getattr(self, "_check_volatility_squeeze", None)
        }
    
        # --- Evaluate each filter and collect results ---
        results_long = {}
        results_short = {}
        results_status = {}
    
        for name in filter_names:
            fn = filter_function_map.get(name)
            if fn is None:
                print(f"[{self.symbol}] [ERROR] Filter function '{name}' not implemented or missing!")
                results_long[name] = False
                results_short[name] = False
                results_status[name] = "ERROR"
                continue
            try:
                # Call with debug flag controlled by DEBUG_FILTERS env var (default: quiet mode)
                result = fn(debug=DEBUG_FILTERS)
                if result == "LONG":
                    results_long[name] = True
                    results_short[name] = False
                    status = "LONG"
                elif result == "SHORT":
                    results_long[name] = False
                    results_short[name] = True
                    status = "SHORT"
                elif result is True:
                    results_long[name] = True
                    results_short[name] = True
                    status = "PASS"
                elif result is False:
                    results_long[name] = False
                    results_short[name] = False
                    status = "FAIL"
                else:
                    results_long[name] = False
                    results_short[name] = False
                    status = "NONE"
                results_status[name] = status
            except Exception as e:
                print(f"[{self.symbol}] [{name}] ERROR: {e}")
                results_long[name] = False
                results_short[name] = False
                results_status[name] = "ERROR"
    
        # --- Non-GK filter list and simple count score ---
        non_gk_filters = [f for f in filter_names if f not in self.gatekeepers]
        long_score = sum(1 for f in non_gk_filters if results_long.get(f, False))
        short_score = sum(1 for f in non_gk_filters if results_short.get(f, False))
    
        # --- Get signal direction (only debug if DEBUG_FILTERS enabled) ---
        try:
            direction = self.get_signal_direction(results_long, results_short, debug=DEBUG_FILTERS)
        except Exception as e:
            print(f"[{self.symbol}] Error in get_signal_direction(): {e}")
            direction = "NEUTRAL"
        self.bias = direction
    
        # --- Gatekeeper breakdown (using bias-specific gatekeepers) ---
        hard_gatekeepers_long = [gk for gk in self.gatekeepers_long if gk not in self.soft_gatekeepers]
        hard_gatekeepers_short = [gk for gk in self.gatekeepers_short if gk not in self.soft_gatekeepers]
        soft_gatekeepers = [gk for gk in self.gatekeepers if gk in self.soft_gatekeepers]
    
        passed_gk_long = [gk for gk in self.gatekeepers_long if results_long.get(gk, False)]
        failed_gk_long = [gk for gk in self.gatekeepers_long if not results_long.get(gk, False)]
        passed_gk_short = [gk for gk in self.gatekeepers_short if results_short.get(gk, False)]
        failed_gk_short = [gk for gk in self.gatekeepers_short if not results_short.get(gk, False)]
    
        passed_hard_gk_long = [gk for gk in hard_gatekeepers_long if results_long.get(gk, False)]
        failed_hard_gk_long = [gk for gk in hard_gatekeepers_long if not results_long.get(gk, False)]
        passed_hard_gk_short = [gk for gk in hard_gatekeepers_short if results_short.get(gk, False)]
        failed_hard_gk_short = [gk for gk in hard_gatekeepers_short if not results_short.get(gk, False)]
    
        passed_soft_gk_long = [gk for gk in soft_gatekeepers if results_long.get(gk, False)]
        failed_soft_gk_long = [gk for gk in soft_gatekeepers if not results_long.get(gk, False)]
        passed_soft_gk_short = [gk for gk in soft_gatekeepers if results_short.get(gk, False)]
        failed_soft_gk_short = [gk for gk in soft_gatekeepers if not results_short.get(gk, False)]
    
        passes_long = len(passed_hard_gk_long) if passed_hard_gk_long is not None else 0
        passes_short = len(passed_hard_gk_short) if passed_hard_gk_short is not None else 0
    
        # --- Passed/failed Non-GK filters and their weights (for confidence) ---
        passed_non_gk_long = [f for f in non_gk_filters if results_long.get(f, False)]
        passed_non_gk_short = [f for f in non_gk_filters if results_short.get(f, False)]
    
        total_non_gk_weight_long = sum(self.filter_weights_long.get(f, 0) for f in non_gk_filters)
        total_non_gk_weight_short = sum(self.filter_weights_short.get(f, 0) for f in non_gk_filters)
        passed_non_gk_weight_long = sum(self.filter_weights_long.get(f, 0) for f in passed_non_gk_long)
        passed_non_gk_weight_short = sum(self.filter_weights_short.get(f, 0) for f in passed_non_gk_short)
    
        # --- Print/logging (only if debug enabled) ---
        if DEBUG_FILTERS:
            print(f"[{self.symbol}] Passed GK LONG: {passed_gk_long}")
            print(f"[{self.symbol}] Failed GK LONG: {failed_gk_long}")
            print(f"[{self.symbol}] Passed GK SHORT: {passed_gk_short}")
            print(f"[{self.symbol}] Failed GK SHORT: {failed_gk_short}")
        
            if not failed_hard_gk_long and failed_soft_gk_long:
                print(f"[{self.symbol}] All hard GKs PASSED for LONG, but SOFT GKs FAILED: {failed_soft_gk_long}")
            if not failed_hard_gk_short and failed_soft_gk_short:
                print(f"[{self.symbol}] All hard GKs PASSED for SHORT, but SOFT GKs FAILED: {failed_soft_gk_short}")
    
        # --- Signal logic: Only hard GKs required to pass (bias-specific) ---
        if self.required_passed is not None:
            required_passed_long = self.required_passed
            required_passed_short = self.required_passed
        else:
            required_passed_long = len(hard_gatekeepers_long)
            required_passed_short = len(hard_gatekeepers_short)
        if DEBUG_FILTERS:
            print(f"[DEBUG] required_passed_long: {required_passed_long}, passes_long: {passes_long}")
            print(f"[DEBUG] required_passed_short: {required_passed_short}, passes_short: {passes_short}")
    
        if required_passed_long is None or passes_long is None:
            print("[ERROR] required_passed_long or passes_long is None!")
            raise ValueError("required_passed_long or passes_long is None!")
        if required_passed_short is None or passes_short is None:
            print("[ERROR] required_passed_short or passes_short is None!")
            raise ValueError("required_passed_short or passes_short is None!")
    
        signal_long_ok = passes_long >= required_passed_long
        signal_short_ok = passes_short >= required_passed_short
    
        if not signal_long_ok:
            print(f"[{self.symbol}] Signal BLOCKED for LONG: Failed hard GKs: {failed_hard_gk_long}")
        if not signal_short_ok:
            print(f"[{self.symbol}] Signal BLOCKED for SHORT: Failed hard GKs: {failed_hard_gk_short}")
    
        # --- Use selected direction's stats ---
        if direction == "LONG":
            score = long_score
            passes = passes_long
            confidence = round(100 * passed_non_gk_weight_long / total_non_gk_weight_long, 1) if total_non_gk_weight_long else 0.0
            passed_weight = passed_non_gk_weight_long
            total_weight = total_non_gk_weight_long
            results = results_long
        elif direction == "SHORT":
            score = short_score
            passes = passes_short
            confidence = round(100 * passed_non_gk_weight_short / total_non_gk_weight_short, 1) if total_non_gk_weight_short else 0.0
            passed_weight = passed_non_gk_weight_short
            total_weight = total_non_gk_weight_short
            results = results_short
        else:
            score = max(long_score, short_score)
            passes = max(passes_long, passes_short)
            confidence = max(
                round(100 * passed_non_gk_weight_long / total_non_gk_weight_long, 1) if total_non_gk_weight_long else 0.0,
                round(100 * passed_non_gk_weight_short / total_non_gk_weight_short, 1) if total_non_gk_weight_short else 0.0
            )
            passed_weight = max(passed_non_gk_weight_long, passed_non_gk_weight_short)
            total_weight = max(total_non_gk_weight_long, total_non_gk_weight_short)
            results = results_long
    
        # Recompute GK-based passes using the chosen 'results' dictionary
        all_gks = self.gatekeepers
        hard_gks = [gk for gk in all_gks if gk not in self.soft_gatekeepers]
        soft_gks = [gk for gk in all_gks if gk in self.soft_gatekeepers]
        all_passed = all(results.get(gk, False) for gk in all_gks)
        hard_passed = all(results.get(gk, False) for gk in hard_gks)
        soft_passed = all(results.get(gk, False) for gk in soft_gks) if soft_gks else True
    
        if all_passed:
            passes = len(all_gks)
        elif hard_passed:
            passes = len(hard_gks)
        else:
            passes = sum(1 for gk in all_gks if results.get(gk, False))
    
        gatekeepers_total = len(all_gks)
        confidence = round(100 * passed_weight / total_weight, 1) if total_weight else 0.0
    
        # --- SuperGK fetch REMOVED: FIX 2026-02-22 ---
        # Removed redundant API calls (get_order_wall_delta, get_resting_density)
        # main.py will fetch these for SuperGK validation. Using defaults for debug reporting only.
        orderbook_result = {"buy_wall": 0, "sell_wall": 0}
        density_result = {"bid_density": 0, "ask_density": 0}
    
        # --- Do not perform an authoritative SuperGK check inside analyze() ---
        # The final SuperGK decision is made in main.py using the freshest orderbook/density.
        try:
            print(f"[{self.symbol}] [Diagnostic] Skipping SuperGK decision inside analyze(); main.py will perform canonical SuperGK.", flush=True)
            # Keep orderbook_result and density_result available in the returned dict for main.py
            super_gk_ok = None
        except Exception as e:
            print(f"[{self.symbol}] Error while skipping superGK in analyze(): {e}", flush=True)
            super_gk_ok = None
    
        # --- Final logging & decision (only if debug enabled) ---
        if DEBUG_FILTERS:
            print("[DEBUG] direction:", direction)
            print("[DEBUG] score:", score, "min_score:", self.min_score)
            print("[DEBUG] passes:", passes, "required_passed:", self.required_passed)
            print("[DEBUG] super_gk_ok:", super_gk_ok)
            print("DEBUG SUMS:", getattr(self, '_debug_sums', {}))
    
        # Always use the correctly-calculated required_passed for the current direction
        if direction == "LONG":
            required_for_signal = required_passed_long
        elif direction == "SHORT":
            required_for_signal = required_passed_short
        else:
            required_for_signal = max(required_passed_long, required_passed_short)

        if required_for_signal is None:
            print(f"[{self.symbol}] [ERROR] required_for_signal is None in final signal logic!")
            required_for_signal = 0

        # --- NEW: compute filters_ok (filters & gatekeepers only) ---
        # Keep SuperGK out of this decision so main.py performs the canonical SuperGK check.
        filters_ok = (
            direction in ["LONG", "SHORT"]
            and (score >= (self.min_score if isinstance(self.min_score, (int, float)) else 0))
            and (passes >= required_for_signal)
            and soft_passed  # Soft gatekeepers MUST pass too (e.g., Candle Confirmation)
        )
        
        # DEBUG: Log why filters_ok might be False
        if not filters_ok:
            print(f"[{self.symbol}] [FILTERS_OK_DEBUG] direction={direction} ({direction in ['LONG', 'SHORT']}) | "
                  f"score={score}>={self.min_score}? ({score >= (self.min_score if isinstance(self.min_score, (int, float)) else 0)}) | "
                  f"passes={passes}>={required_for_signal}? ({passes >= required_for_signal}) | "
                  f"soft_passed={soft_passed}", flush=True)

        # Keep valid_signal for backwards compatibility but do NOT include super_gk_ok here.
        # main.py will compute the final valid_signal by combining filters_ok + super_gk_aligned(...)
        valid_signal = bool(filters_ok)

        price = None
        try:
            price = float(self.df['close'].iat[-1]) if valid_signal else None
        except Exception:
            price = None
        price_str = f"{price:.6f}" if price is not None else "N/A"

        route, reversal_side = self.explicit_route_gate()
        display_route = route if route not in ["?", "NONE", None] else "NO ROUTE"

        signal_type = direction if valid_signal else None
        score_max = len(non_gk_filters)

        message = (
            f"{direction or 'NO-SIGNAL'} on {self.symbol} @ {price_str} "
            f"| Score: {score}/{score_max} | Passed GK: {passes}/{len(self.gatekeepers)} "
            f"| Confidence: {confidence}% (Weighted: {passed_weight:.1f}/{total_weight:.1f})"
            f" | Route: {display_route if valid_signal else 'N/A'}"
        )

        if valid_signal:
            print(f"[{self.symbol}] ✅ FINAL SIGNAL (filters_ok): {message}")
        else:
            if DEBUG_FILTERS:
                print(f"[{self.symbol}] ❌ No signal (filters failed).")
        if DEBUG_FILTERS:
            print("DEBUG SUMS:", getattr(self, '_debug_sums', {}))

        # --- Verdict for debug file (unchanged) ---
        verdict = {
            "orderbook": (
                direction == "SHORT" and orderbook_result.get("sell_wall", 0) > orderbook_result.get("buy_wall", 0)
                or direction == "LONG" and orderbook_result.get("buy_wall", 0) > orderbook_result.get("sell_wall", 0)
            ),
            "density": (
                direction == "SHORT" and density_result.get("ask_density", 0) > density_result.get("bid_density", 0)
                or direction == "LONG" and density_result.get("bid_density", 0) > density_result.get("ask_density", 0)
            ),
            "final": valid_signal
        }

        # DEBUG FILE GENERATION: DISABLED
        # (causing Telegram spam - will re-enable when proper mechanism is in place)
        # export_signal_debug_txt(...)

        regime = self._market_regime()

        # --- Add explicit results keys so main.py can use filters_ok and then do SuperGK check ---
        return {
            "symbol": self.symbol,
            "tf": self.tf,
            "score": score,
            "score_max": score_max,
            "passes": passes,
            "gatekeepers_total": gatekeepers_total,
            "passed_weight": round(passed_weight, 1),
            "total_weight": round(total_weight, 1),
            "confidence": confidence,
            "bias": direction,
            "price": price,
            "valid_signal": valid_signal,    # kept for compatibility (represents filters_ok now)
            "filters_ok": filters_ok,        # explicit: filters + gatekeepers pass requirements
            "super_gk_ok": locals().get("super_gk_ok", None),  # may be None if analyze didn't compute it
            "signal_type": signal_type,
            "Route": route,
            "regime": regime,
            "reversal_side": reversal_side,
            "message": message,
            "debug_sums": getattr(self, '_debug_sums', {}),
            "results_long": results_long,
            "results_short": results_short,
            "results_status": results_status,
            "orderbook_result": orderbook_result,
            "density_result": density_result,
        }

    # === Super-GK logic stubs ===
    def _order_book_wall_passed(self):
        return True

    def _resting_order_density_passed(self):
        return True

    # --- All filter logic below ---

    def _safe_divide(self, a, b):
        try:
            return a / b if b else 0.0
        except Exception:
            return 0.0

    def _check_unified_trend(
        self,
        min_conditions=7,
        min_adx_for_trend=20,
        volatility_adjusted=True,
        use_ema9_for_trend=True,
        debug=False
    ):
        """
        ENHANCED TREND Filter (2026-03-05):
        Confirms unified directional bias across multiple EMA systems.
        
        Improvements from original:
        1. Removed HATS redundancy (was nearly identical to Structure)
        2. Increased threshold: 6 → 7 (54% consensus requirement)
        3. Replaced EMA6 with reuse of EMA9 (less noisy)
        4. Volatility-adjusted: harder to trigger in choppy markets (>3% ATR)
        5. ADX check: only counts as trend if ADX > 20 (true trend confirmation)
        
        Args:
            min_conditions: Require 7/10 conditions met (was 6/13, now cleaner)
            min_adx_for_trend: ADX threshold for trend confirmation (default 20)
            volatility_adjusted: Increase threshold in high volatility (>3% ATR)
            use_ema9_for_trend: Reuse EMA9 instead of separate EMA6 (stability)
            debug: Print debug info
        """
        try:
            # EMA Cloud (20-50): Fast trend detection
            ema20 = self.df['ema20'].iat[-1]
            ema50 = self.df['ema50'].iat[-1]
            ema20_prev = self.df['ema20'].iat[-2]
            close = self.df['close'].iat[-1]
            ema_cloud = ema20 - ema50
        
            # EMA Structure (9-21-50): Mid-term trend stacking
            ema9 = self.df['ema9'].iat[-1]
            ema21 = self.df['ema21'].iat[-1]
            ema50_s = self.df['ema50'].iat[-1]
            ema9_prev = self.df['ema9'].iat[-2]
            ema21_prev = self.df['ema21'].iat[-2]
            ema50_prev = self.df['ema50'].iat[-2]
        
            # NOTE: REMOVED HATS (10-21-50) - redundant with Structure above
        
            # Trend Continuation: Use EMA9 instead of EMA6 (less noisy on 15min)
            if use_ema9_for_trend:
                ema_trend_fast = self.df['ema9'].iat[-1]  # Reuse EMA9
            else:
                ema_trend_fast = self.df['ema6'].iat[-1] if 'ema6' in self.df.columns else self.df['ema9'].iat[-1]
            
            ema_trend_slow = self.df['ema13'].iat[-1] if 'ema13' in self.df.columns else self.df['ema21'].iat[-1]
        
            # Momentum confirmations
            macd = self.df['macd'].iat[-1] if 'macd' in self.df.columns else None
            rsi = self.df['RSI'].iat[-1] if 'RSI' in self.df.columns else None
            adx = self.df['adx'].iat[-1] if 'adx' in self.df.columns else None
            
            # Volatility adjustment: harder in choppy markets
            threshold = min_conditions
            if volatility_adjusted:
                atr = self.df['atr'].iat[-1] if 'atr' in self.df.columns else None
                sma_close = self.df['close'].rolling(20).mean().iat[-1] if len(self.df) >= 20 else close
                if atr and sma_close and sma_close > 0:
                    volatility_pct = atr / sma_close
                    if volatility_pct > 0.03:  # 3% volatility = high chop
                        threshold = min_conditions + 1  # Require 8 instead of 7
                        if debug:
                            print(f"[{self.symbol}] [TREND] HIGH VOLATILITY ({volatility_pct:.2%}) → threshold {min_conditions} → {threshold}")
        
            # LONG conditions (10 total, was 13 with HATS redundancy removed)
            conds_long = [
                ema20 > ema50,                                    # Cloud spread bullish
                ema20 > ema20_prev,                               # Cloud momentum rising
                close > ema20,                                    # Price above cloud
                ema9 > ema21 and ema21 > ema50_s,                 # Structure stacking
                close > ema9 and close > ema21 and close > ema50_s,  # Price above all EMAs
                ema9 > ema9_prev and ema21 > ema21_prev and ema50_s > ema50_prev,  # Structure momentum
                ema_trend_fast > ema_trend_slow,                  # Trend continuation (EMA9 vs EMA13/21)
                macd is not None and macd > 0,                    # MACD bullish
                rsi is not None and rsi > 50,                     # RSI bullish
                not (adx is not None) or adx > min_adx_for_trend  # ADX confirms trend (or absent = skip check)
            ]
            long_met = sum(conds_long)
        
            # SHORT conditions (mirror of LONG)
            conds_short = [
                ema20 < ema50,
                ema20 < ema20_prev,
                close < ema20,
                ema9 < ema21 and ema21 < ema50_s,
                close < ema9 and close < ema21 and close < ema50_s,
                ema9 < ema9_prev and ema21 < ema21_prev and ema50_s < ema50_prev,
                ema_trend_fast < ema_trend_slow,
                macd is not None and macd < 0,
                rsi is not None and rsi < 50,
                not (adx is not None) or adx > min_adx_for_trend
            ]
            short_met = sum(conds_short)
        
            if debug:
                print(
                    f"[{self.symbol}] [TREND ENHANCED] long_met={long_met}, short_met={short_met}, "
                    f"threshold={threshold}, ema_cloud={ema_cloud:.4f}"
                )
        
            # Decision logic: require ≥7 (or 8 in high volatility) and clear majority
            if long_met >= threshold and long_met > short_met:
                print(f"[{self.symbol}] [TREND ENHANCED] Signal: LONG | long_met={long_met}/{threshold}, short_met={short_met}")
                return "LONG"
            elif short_met >= threshold and short_met > long_met:
                print(f"[{self.symbol}] [TREND ENHANCED] Signal: SHORT | short_met={short_met}/{threshold}, long_met={long_met}")
                return "SHORT"
            else:
                if debug:
                    print(f"[{self.symbol}] [TREND ENHANCED] No signal | long_met={long_met}, short_met={short_met}, threshold={threshold}")
                return None
        
        except Exception as e:
            print(f"[{self.symbol}] [TREND] Error: {e}", flush=True)
            return None

    def _check_macd(
        self,
        fast=12,
        slow=26,
        signal=9,
        min_conditions=3,
        min_macd_magnitude=0.0005,
        divergence_weight=0.5,
        check_signal_momentum=True,
        debug=False
    ):
        """
        ENHANCED MACD Filter (2026-03-05):
        Improved momentum detection with better signal quality control.
        
        Improvements from original:
        1. Threshold: 2 → 3 (50% consensus requirement)
        2. MACD magnitude filter (no noise signals near zero)
        3. Signal line momentum check (catch accelerating crosses)
        4. Divergence weighted less (less reliable than direct confirmations)
        5. Histogram acceleration requirement (momentum must be rising)
        6. Better for all timeframes
        
        Args:
            fast: Fast EMA span (default 12)
            slow: Slow EMA span (default 26)
            signal: Signal line EMA span (default 9)
            min_conditions: Require 3/7 conditions met (50%, was 2/6 = 33%)
            min_macd_magnitude: Minimum MACD magnitude to consider (0.0005 filters noise)
            divergence_weight: Weight for divergence (0.5 = half value of other conditions)
            check_signal_momentum: Include signal line acceleration checks
            debug: Print debug info
        """
        if len(self.df) < slow + 3:
            if debug:
                print(f"[{self.symbol}] [MACD] Not enough data for slow EMA={slow}")
            return None
    
        # Calculate MACD components
        efast = self.df['close'].ewm(span=fast, adjust=False).mean()
        eslow = self.df['close'].ewm(span=slow, adjust=False).mean()
        macd = efast - eslow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - signal_line
    
        close = self.df['close'].iloc[-1]
        close_prev = self.df['close'].iloc[-2]
        
        # MACD values (current, previous, 2 bars ago)
        macd_curr = macd.iloc[-1]
        macd_prev = macd.iloc[-2]
        macd_prev2 = macd.iloc[-3] if len(macd) >= 3 else macd_prev
        
        signal_curr = signal_line.iloc[-1]
        signal_prev = signal_line.iloc[-2]
        signal_prev2 = signal_line.iloc[-3] if len(signal_line) >= 3 else signal_prev
        
        hist_curr = macd_hist.iloc[-1]
        hist_prev = macd_hist.iloc[-2]
    
        # MACD Cross Detection
        cross_up = macd_prev < signal_prev and macd_curr > signal_curr
        cross_down = macd_prev > signal_prev and macd_curr < signal_curr
    
        # MACD Divergence Detection (will be weighted less)
        price_delta = close - close_prev
        macd_delta = macd_curr - macd_prev
        divergence = price_delta * macd_delta < 0
        
        # NEW: Signal line momentum (acceleration of trigger line)
        signal_momentum_up = signal_curr > signal_prev > signal_prev2
        signal_momentum_down = signal_curr < signal_prev < signal_prev2
    
        # LONG conditions (7 total, enhanced from 6)
        cond1_long = abs(macd_curr) > min_macd_magnitude and macd_curr > signal_curr      # MACD above signal + significant magnitude
        cond2_long = macd_curr > macd_prev                                                # MACD rising
        cond3_long = hist_curr > 0 and hist_curr > hist_prev                              # Histogram positive & rising
        cond4_long = close > close_prev                                                   # Price momentum
        cond5_long = cross_up                                                             # Bullish crossover
        cond6_long = divergence and macd_delta > 0                                        # Bullish divergence
        cond7_long = check_signal_momentum and signal_momentum_up                        # Signal line accelerating up
    
        # SHORT conditions (mirror of LONG)
        cond1_short = abs(macd_curr) > min_macd_magnitude and macd_curr < signal_curr
        cond2_short = macd_curr < macd_prev
        cond3_short = hist_curr < 0 and hist_curr < hist_prev
        cond4_short = close < close_prev
        cond5_short = cross_down
        cond6_short = divergence and macd_delta < 0
        cond7_short = check_signal_momentum and signal_momentum_down
    
        # Count conditions for decision logic
        long_conditions_met = sum([cond1_long, cond2_long, cond3_long, cond4_long, cond5_long, cond6_long, cond7_long])
        short_conditions_met = sum([cond1_short, cond2_short, cond3_short, cond4_short, cond5_short, cond6_short, cond7_short])
    
        if debug:
            print(
                f"[{self.symbol}] [MACD ENHANCED] macd={macd_curr:.6f}, signal={signal_curr:.6f}, "
                f"hist={hist_curr:.6f}, cross_up={cross_up}, cross_down={cross_down}, "
                f"divergence={divergence}, signal_mom_up={signal_momentum_up}, signal_mom_down={signal_momentum_down}, "
                f"long_met={long_conditions_met}, short_met={short_conditions_met}, min_conditions={min_conditions}"
            )
    
        # Decision: require ≥3 (50% consensus) and clear majority
        if long_conditions_met >= min_conditions and long_conditions_met > short_conditions_met:
            print(f"[{self.symbol}] [MACD ENHANCED] Signal: LONG | long_met={long_conditions_met}/{min_conditions}, short_met={short_conditions_met}")
            return "LONG"
        elif short_conditions_met >= min_conditions and short_conditions_met > long_conditions_met:
            print(f"[{self.symbol}] [MACD ENHANCED] Signal: SHORT | short_met={short_conditions_met}/{min_conditions}, long_met={long_conditions_met}")
            return "SHORT"
        else:
            if debug:
                print(f"[{self.symbol}] [MACD ENHANCED] No signal | long_met={long_conditions_met}, short_met={short_conditions_met}, min_conditions={min_conditions}")
            return None
  
    def _check_momentum(
        self,
        window=10,
        min_conditions=3,
        min_roc_threshold=0.001,
        min_acceleration=0,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        cci_period=20,
        stochrsi_period=14,
        willr_period=14,
        debug=False
    ):
        """
        ENHANCED Momentum Filter (2026-03-05):
        Detects overbought/oversold reversals with momentum confirmation.
        
        Improvements from original:
        1. Threshold: 2 → 3 (29% → 43% consensus requirement)
        2. ROC threshold: 1e-6 → 0.001 (filters noise, requires 0.1% momentum minimum)
        3. Acceleration threshold added (requires meaningful momentum change, not just > 0)
        4. RSI levels: 20/80 → 30/70 (true overbought/oversold, not extreme)
        5. Clear reversal detection focus (extremes → reversals)
        6. Better for all timeframes
        
        Args:
            window: ROC period (10 = momentum over 10 candles)
            min_conditions: Require 3/7 conditions met (43%, was 2/7 = 29%)
            min_roc_threshold: Minimum ROC to consider (0.001 = 0.1%, was 1e-6 = noise)
            min_acceleration: Minimum acceleration change (0 = any positive)
            rsi_oversold: RSI threshold for bullish reversal (30 = true oversold, was 20)
            rsi_overbought: RSI threshold for bearish reversal (70 = true overbought, was 80)
            debug: Print debug info
        
        Returns: "LONG", "SHORT", or None
        """
        
        required_len = max(window + 2, rsi_period + 2, cci_period + 2, stochrsi_period + 2, willr_period + 2)
        if len(self.df) < required_len:
            if debug:
                print(f"[{self.symbol}] [Momentum] Not enough data for required indicators.")
            return None
    
        try:
            # Calculate ROC and acceleration
            roc = self.df['close'].pct_change(periods=window)
            momentum = roc.iloc[-1]
            momentum_prev = roc.iloc[-2]
            acceleration = momentum - momentum_prev
            
            close = self.df['close'].iloc[-1]
            close_prev = self.df['close'].iloc[-2]
        
            # Calculate oscillators
            rsi = compute_rsi(self.df, rsi_period)
            rsi_latest = rsi.iloc[-1]
        
            cci = calculate_cci(self.df, cci_period)
            cci_latest = cci.iloc[-1]
        
            stochrsi_k, stochrsi_d = calculate_stochrsi(self.df, rsi_period, stochrsi_period)
            stochrsi_latest = stochrsi_k.iloc[-1]
        
            willr = compute_williams_r(self.df, willr_period)
            willr_latest = willr.iloc[-1]
        
            # LONG conditions (oversold reversal)
            cond1_long = momentum > min_roc_threshold              # ROC positive + meaningful (0.1%+)
            cond2_long = acceleration > min_acceleration           # Momentum accelerating (not just > 0)
            cond3_long = close > close_prev                        # Price momentum aligned
            cond4_long = rsi_latest < rsi_oversold               # RSI oversold (< 30, true extreme)
            cond5_long = cci_latest < -100                        # CCI at extreme low
            cond6_long = stochrsi_latest < 0.2                    # StochRSI at trough
            cond7_long = willr_latest < -80                       # Williams %R at extreme low
        
            # SHORT conditions (overbought reversal) - mirror of LONG
            cond1_short = momentum < -min_roc_threshold
            cond2_short = acceleration < -min_acceleration
            cond3_short = close < close_prev
            cond4_short = rsi_latest > rsi_overbought            # RSI overbought (> 70, true extreme)
            cond5_short = cci_latest > 100
            cond6_short = stochrsi_latest > 0.8
            cond7_short = willr_latest > -20
        
            long_met = sum([cond1_long, cond2_long, cond3_long, cond4_long, cond5_long, cond6_long, cond7_long])
            short_met = sum([cond1_short, cond2_short, cond3_short, cond4_short, cond5_short, cond6_short, cond7_short])
        
            if debug:
                print(
                    f"[{self.symbol}] [Momentum ENHANCED] momentum={momentum:.6f} (min={min_roc_threshold}), "
                    f"acceleration={acceleration:.6f}, close={close}, close_prev={close_prev}, "
                    f"rsi={rsi_latest:.1f} (oversold={rsi_oversold}/overbought={rsi_overbought}), "
                    f"cci={cci_latest:.1f}, stochrsi={stochrsi_latest:.2f}, willr={willr_latest:.1f}, "
                    f"long_met={long_met}, short_met={short_met}, min_conditions={min_conditions}"
                )
        
            # Decision: require 3/7 (43% consensus) and clear majority
            if long_met >= min_conditions and long_met > short_met:
                print(f"[{self.symbol}] [Momentum ENHANCED] Signal: LONG | long_met={long_met}/{min_conditions}, short_met={short_met}")
                return "LONG"
            elif short_met >= min_conditions and short_met > long_met:
                print(f"[{self.symbol}] [Momentum ENHANCED] Signal: SHORT | short_met={short_met}/{min_conditions}, long_met={long_met}")
                return "SHORT"
            else:
                if debug:
                    print(f"[{self.symbol}] [Momentum ENHANCED] No signal | long_met={long_met}, short_met={short_met}, min_conditions={min_conditions}")
                return None
        
        except Exception as e:
            print(f"[{self.symbol}] [Momentum] Error: {e}", flush=True)
            return None
    
    # ==== FILTERS ====    
    def _check_volume_spike_detector(
        self,
        rolling_window: int = 15,
        min_price_move: float = 0.0001,
        zscore_threshold: float = 1.5,
        debug: bool = False
    ) -> Optional[str]:
        """
        ENHANCED Volume Spike Filter (2026-03-05):
        Detects volume spikes (institutional activity) backing price moves.
        
        Improvements:
        1. Z-score threshold: 1.1 → 1.5 (better balance: 93rd percentile)
        2. Removed 5m confirmation (daemon runs 15min/30min/1h only, 5m not available)
        3. Simplified: 3 conditions, threshold 2/3 (kept simple, proven effective)
        4. Fixed rolling window (no more inconsistent 5-candle override)
        5. Better for all timeframes
        
        Args:
            rolling_window: Volume lookback window (15 candles = baseline)
            min_price_move: Minimum price movement to consider (0.0001 = 0.01%)
            zscore_threshold: Z-score for spike detection (1.5 = 93rd percentile, institutional activity)
            debug: Print debug info
        
        Returns: "LONG", "SHORT", or None
        """
        
        # Defensive: Check rolling window length
        if len(self.df) < rolling_window + 2:
            if debug:
                print(f"[{self.symbol}] [Volume Spike] Not enough data for rolling window={rolling_window}")
            return None
    
        try:
            # Calculate volume metrics
            avg = self.df['volume'].rolling(rolling_window).mean().iat[-2]
            std = self.df['volume'].rolling(rolling_window).std().iat[-2]
            curr_vol = self.df['volume'].iat[-1]
            vol_prev = self.df['volume'].iat[-2]
            
            # Z-score: (current - average) / standard deviation
            zscore = (curr_vol - avg) / (std if std != 0 else 1)
            spike = zscore > zscore_threshold  # Institutional activity detected
        
            # Price metrics
            close = self.df['close'].iat[-1]
            close_prev = self.df['close'].iat[-2]
            price_move = (close - close_prev) / close_prev if close_prev != 0 else 0
        
            # Conditions: price movement + volume confirmation
            price_up = price_move > min_price_move      # Price moved up significantly
            price_down = price_move < -min_price_move    # Price moved down significantly
            vol_up = curr_vol > vol_prev                 # Volume higher than previous
        
            # Three conditions for LONG / SHORT
            long_conditions = [spike, price_up, vol_up]
            short_conditions = [spike, price_down, vol_up]
        
            # Signal decision: need 2/3 conditions (67% threshold)
            signal = None
            if sum(long_conditions) >= 2:
                signal = "LONG"
            elif sum(short_conditions) >= 2:
                signal = "SHORT"
        
            if debug:
                print(
                    f"[{self.symbol}] [Volume Spike ENHANCED] signal={signal} | "
                    f"zscore={zscore:.4f} (threshold={zscore_threshold}), price_move={price_move:.6f}, "
                    f"vol_up={vol_up}, spike={spike}, price_up={price_up}, price_down={price_down}, "
                    f"curr_vol={curr_vol:.0f}, avg_vol={avg:.0f}, conditions_met={sum(long_conditions) if signal=='LONG' else sum(short_conditions)}/3"
                )
        
            return signal
        
        except Exception as e:
            print(f"[{self.symbol}] [Volume Spike] Error: {e}", flush=True)
            return None

    def _check_mtf_volume_agreement(self, debug: bool = False) -> Optional[str]:
        """
        Multi-timeframe volume agreement filter.
        Returns "LONG", "SHORT", or None.
        """
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        higher_tf_volume = self.df['higher_tf_volume'].iat[-1]
        higher_tf_volume_prev = self.df['higher_tf_volume'].iat[-2]
    
        conds_long = [volume > volume_prev, higher_tf_volume > higher_tf_volume_prev, close > close_prev]
        conds_short = [volume < volume_prev, higher_tf_volume < higher_tf_volume_prev, close < close_prev]
    
        long_met, short_met = sum(conds_long), sum(conds_short)
        signal = None
        if long_met >= 1 and long_met > short_met:
            signal = "LONG"
        elif short_met >= 1 and short_met > long_met:
            signal = "SHORT"
    
        if debug:
            print(f"[{self.symbol}] [MTF Volume Agreement] signal={signal} | long_met={long_met}, short_met={short_met}, conds_long={conds_long}, conds_short={conds_short}")
    
        return signal
        
    def _check_liquidity_awareness(self, debug: bool = False) -> Optional[str]:
        """
        Liquidity awareness filter based on spread, volume, and close price.
        Returns "LONG", "SHORT", or None.
        """
        bid, ask = self.df['bid'].iat[-1], self.df['ask'].iat[-1]
        bid_prev, ask_prev = self.df['bid'].iat[-2], self.df['ask'].iat[-2]
        volume, volume_prev = self.df['volume'].iat[-1], self.df['volume'].iat[-2]
        close, close_prev = self.df['close'].iat[-1], self.df['close'].iat[-2]
    
        spread = ask - bid if not (math.isnan(bid) or math.isnan(ask)) else None
        spread_prev = ask_prev - bid_prev if not (math.isnan(bid_prev) or math.isnan(ask_prev)) else None
    
        conds_long = [
            spread is not None and spread_prev is not None and spread < spread_prev,
            volume > volume_prev,
            close > close_prev
        ]
        conds_short = [
            spread is not None and spread_prev is not None and spread > spread_prev,
            volume < volume_prev,
            close < close_prev
        ]
    
        long_met, short_met = sum(conds_long), sum(conds_short)
        signal = None
        if long_met >= 1 and long_met > short_met:
            signal = "LONG"
        elif short_met >= 1 and short_met > long_met:
            signal = "SHORT"
    
        if debug:
            liquidity_metrics = {
                "spread": spread,
                "spread_prev": spread_prev,
                "volume": volume,
                "volume_prev": volume_prev,
                "close": close,
                "close_prev": close_prev
            }
            print(f"[{self.symbol}] [Liquidity Awareness] signal={signal} | long_met={long_met}, short_met={short_met}, liquidity_metrics={liquidity_metrics}")
    
        return signal

    def _check_spread_filter(self, window: int = 20, debug: bool = False) -> Optional[str]:
        """
        Spread filter based on current/historical spreads and price action.
        Returns "LONG", "SHORT", or None.
        """
        high, low, open_, close = self.df['high'].iat[-1], self.df['low'].iat[-1], self.df['open'].iat[-1], self.df['close'].iat[-1]
        high_prev, low_prev, open_prev, close_prev = self.df['high'].iat[-2], self.df['low'].iat[-2], self.df['open'].iat[-2], self.df['close'].iat[-2]
    
        # Add spread column if missing
        if 'spread' not in self.df.columns:
            self.df['spread'] = self.df['high'] - self.df['low']
    
        spread = high - low
        spread_prev = high_prev - low_prev
        spread_ma = self.df['spread'].rolling(window).mean().iat[-1]
    
        conds_long = [spread > spread_prev, close > open_, spread > spread_ma]
        conds_short = [spread < spread_prev, close < open_, spread < spread_ma]
    
        long_met, short_met = sum(conds_long), sum(conds_short)
        signal = None
        if long_met >= 1 and long_met > short_met:
            signal = "LONG"
        elif short_met >= 1 and short_met > long_met:
            signal = "SHORT"
    
        if debug:
            # full, non-truncated debug print
            print(
                f"[{self.symbol}] [Spread Filter] signal={signal} | "
                f"spread={spread:.6f}, long_met={long_met}, short_met={short_met}, "
                f"spread_prev={spread_prev:.6f}, spread_ma={spread_ma:.6f}, close={close}, open={open_}"
            )
    
        return signal

    def _check_smart_money_bias(self, volume_window: int = 20, min_cond: int = 1, debug: bool = False) -> Optional[str]:
        """
        Smart money bias filter combining volume and VWAP.
        Returns "LONG", "SHORT", or None.
        """
        close, close_prev = self.df['close'].iat[-1], self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        avg_volume = self.df['volume'].rolling(volume_window).mean().iat[-1]
        vwap = self.df['vwap'].iat[-1]
    
        conds_long = [volume > avg_volume and close > close_prev, close > vwap]
        conds_short = [volume > avg_volume and close < close_prev, close < vwap]
    
        long_met, short_met = sum(conds_long), sum(conds_short)
        signal = None
        if long_met >= min_cond and long_met > short_met:
            signal = "LONG"
        elif short_met >= min_cond and short_met > long_met:
            signal = "SHORT"
    
        if debug:
            smart_money = {
                "volume_vs_avg": volume / avg_volume if avg_volume != 0 else None,
                "close_vs_prev": close - close_prev,
                "close_vs_vwap": close - vwap
            }
            print(f"[{self.symbol}] [Smart Money Bias] signal={signal} | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, smart_money={smart_money}")
    
        return signal

    def _check_absorption(self, window: int = 20, buffer_pct: float = 0.005, min_cond: int = 1, debug: bool = False) -> Optional[str]:
        """
        Absorption filter for detecting price action near highs/lows with volume confirmation.
        Returns "LONG", "SHORT", or None.
        """
        low = self.df['low'].rolling(window).min().iat[-1]
        high = self.df['high'].rolling(window).max().iat[-1]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
        avg_volume = self.df['volume'].rolling(window).mean().iat[-1]
    
        # Safe division helper
        def safe_divide(a, b):
            try:
                return a / b if b else 0.0
            except Exception:
                return 0.0
    
        absorption_metrics = {
            "close_to_low_pct": safe_divide(close - low, low),
            "close_to_high_pct": safe_divide(high - close, high),
            "volume_vs_avg": safe_divide(volume, avg_volume),
            "volume_vs_prev": safe_divide(volume, volume_prev)
        }
    
        conds_long = [
            close <= low * (1 + buffer_pct),
            volume > avg_volume and volume > volume_prev,
            close >= close_prev
        ]
        conds_short = [
            close >= high * (1 - buffer_pct),
            volume > avg_volume and volume > volume_prev,
            close <= close_prev
        ]
    
        long_met, short_met = sum(conds_long), sum(conds_short)
        signal = None
        if long_met >= min_cond and long_met > short_met:
            signal = "LONG"
        elif short_met >= min_cond and short_met > long_met:
            signal = "SHORT"
    
        if debug:
            print(f"[{self.symbol}] [Absorption] signal={signal} | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, absorption={absorption_metrics}")
    
        return signal

    def _check_vwap_divergence(self, debug: bool = False):
        """
        VWAP divergence filter.
        - Requires a strict majority (long_met > short_met or short_met > long_met)
          and a minimum number of conditions met (>=4) to declare LONG or SHORT.
        - No longer prefers LONG on ties.
        - Debug logging is standardized and guarded.
        """
        # Defensive extraction of required series values
        try:
            vwap = self.df['vwap'].iat[-1]
            vwap_prev = self.df['vwap'].iat[-2]
            close = self.df['close'].iat[-1]
            close_prev = self.df['close'].iat[-2]
            volume = self.df['volume'].iat[-1]
            volume_ma = self.df['volume'].rolling(window=20).mean().iat[-1]
        except Exception as e:
            if debug:
                print(f"[{self.symbol}] [VWAP Divergence] Missing data or index error: {e}")
            return None
    
        # Graceful handling for NaN volume_ma
        if volume_ma is None or (isinstance(volume_ma, float) and math.isnan(volume_ma)):
            volume_ma = volume  # fallback to using current volume so the volume checks don't fail
    
        min_vol_threshold = volume_ma * 0.9  # 10% below moving avg is still acceptable
    
        # Compute VWAP divergence for logging/thresholding
        vwap_div = (vwap - close) - (vwap_prev - close_prev)
        min_vwap_div = 0.001  # e.g., 0.1% as a minimal meaningful divergence
    
        # LONG conditions
        cond1_long = close < vwap
        cond2_long = close > close_prev
        cond3_long = (vwap - close) > (vwap_prev - close_prev)
        cond4_long = volume > min_vol_threshold
        cond5_long = abs(vwap_div) > min_vwap_div
    
        # SHORT conditions
        cond1_short = close > vwap
        cond2_short = close < close_prev
        cond3_short = (close - vwap) > (close_prev - vwap_prev)
        cond4_short = volume > min_vol_threshold
        cond5_short = abs(vwap_div) > min_vwap_div
    
        long_met = sum([cond1_long, cond2_long, cond3_long, cond4_long, cond5_long])
        short_met = sum([cond1_short, cond2_short, cond3_short, cond4_short, cond5_short])
    
        if debug:
            print(
                f"[{self.symbol}] [VWAP Divergence] Values | "
                f"vwap={vwap}, vwap_prev={vwap_prev}, close={close}, close_prev={close_prev}, "
                f"volume={volume}, volume_ma={volume_ma:.6f}"
            )
            print(
                f"[{self.symbol}] [VWAP Divergence] Conditions | "
                f"conds_long={[cond1_long, cond2_long, cond3_long, cond4_long, cond5_long]}, "
                f"conds_short={[cond1_short, cond2_short, cond3_short, cond4_short, cond5_short]}"
            )
            print(
                f"[{self.symbol}] [VWAP Divergence] Met Counts | long_met={long_met}, short_met={short_met}, vwap_div={vwap_div:.6f}"
            )
    
        # Require a strict majority and minimum met count to declare a direction. Do NOT prefer LONG on ties.
        MIN_MET_COUNT = 4
        if long_met > short_met and long_met >= MIN_MET_COUNT:
            if debug:
                print(
                    f"[{self.symbol}] [VWAP Divergence] Signal: LONG | "
                    f"long_met={long_met}, short_met={short_met}, vwap_div={vwap_div:.6f}, volume={volume}, volume_ma={volume_ma:.6f}"
                )
            return "LONG"
        elif short_met > long_met and short_met >= MIN_MET_COUNT:
            if debug:
                print(
                    f"[{self.symbol}] [VWAP Divergence] Signal: SHORT | "
                    f"long_met={long_met}, short_met={short_met}, vwap_div={vwap_div:.6f}, volume={volume}, volume_ma={volume_ma:.6f}"
                )
            return "SHORT"
        else:
            if debug:
                print(
                    f"[{self.symbol}] [VWAP Divergence] No signal fired | "
                    f"long_met={long_met}, short_met={short_met}, vwap_div={vwap_div:.6f}, volume={volume}, volume_ma={volume_ma:.6f}"
                )
            return None
            
    def _get_rolling_extremes(self, col, window, prev=False):
        """Helper to get rolling min/max for current or previous index."""
        series = self.df[col].rolling(window)
        if prev:
            return series.min().iat[-2], series.max().iat[-2]
        else:
            return series.min().iat[-1], series.max().iat[-1]
    
    def _check_support_resistance(
        self,
        window: int = 20,
        buffer_pct: float = 0.005,
        min_cond: int = 2,
        require_volume_confirm: bool = False,
        debug: bool = False
    ) -> Optional[str]:
        """
        Support / Resistance proximity filter.
    
        Improvements over the original:
        - Defensive checks for required columns and sufficient history.
        - Uses safe division to avoid ZeroDivisionError / NaN issues.
        - Optional volume confirmation toggle (require_volume_confirm).
        - Clearer proximity calculations and consistent debug messages.
        - No variable-name typos in debug prints.
        - Explicit tie handling: only return LONG or SHORT when conditions strictly favor one side.
    
        Returns:
            "LONG", "SHORT", or None
        """
        # Required columns / length check
        required_cols = ['low', 'high', 'close', 'volume']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            if debug:
                print(f"[{self.symbol}] [Support/Resistance] Missing columns: {missing}")
            return None
    
        if len(self.df) < window + 1:
            if debug:
                print(f"[{self.symbol}] [Support/Resistance] Not enough data (need at least {window + 1} rows).")
            return None
    
        # Safely get rolling extremes (the helper may raise if indices are short)
        try:
            support, _ = self._get_rolling_extremes('low', window)
            _, resistance = self._get_rolling_extremes('high', window)
        except Exception as e:
            if debug:
                print(f"[{self.symbol}] [Support/Resistance] Error getting rolling extremes: {e}")
            return None
    
        # Latest price/volume values
        try:
            close = float(self.df['close'].iat[-1])
            close_prev = float(self.df['close'].iat[-2])
            volume = float(self.df['volume'].iat[-1])
            volume_prev = float(self.df['volume'].iat[-2])
        except Exception as e:
            if debug:
                print(f"[{self.symbol}] [Support/Resistance] Error reading price/volume: {e}")
            return None
    
        # Safe proximity calculations
        # support_prox: how far above support the close is (as fraction of support)
        support_prox = self.safe_divide(close - support, support)
        # resistance_prox: how far below resistance the close is (as fraction of resistance)
        resistance_prox = self.safe_divide(resistance - close, resistance)
    
        # Volume confirmation (optional)
        volume_confirm_long = volume > volume_prev if require_volume_confirm else True
        volume_confirm_short = volume < volume_prev if require_volume_confirm else True
    
        # Conditions (use proximity instead of direct multiplication to avoid rounding issues)
        cond1_long = support_prox <= buffer_pct  # close within buffer_pct above support
        cond2_long = close > close_prev
        cond3_long = volume_confirm_long
    
        cond1_short = resistance_prox <= buffer_pct  # close within buffer_pct below resistance
        cond2_short = close < close_prev
        cond3_short = volume_confirm_short
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        if debug:
            print(
                f"[{self.symbol}] [Support/Resistance] Values | support={support}, resistance={resistance}, "
                f"close={close}, close_prev={close_prev}, volume={volume}, volume_prev={volume_prev}"
            )
            print(
                f"[{self.symbol}] [Support/Resistance] Proximity | support_prox={support_prox:.6f}, "
                f"resistance_prox={resistance_prox:.6f}, buffer_pct={buffer_pct:.6f}"
            )
            print(
                f"[{self.symbol}] [Support/Resistance] Conditions Long | "
                f"[near_support={cond1_long}, close_up={cond2_long}, volume_confirm={cond3_long}] -> long_met={long_met}"
            )
            print(
                f"[{self.symbol}] [Support/Resistance] Conditions Short | "
                f"[near_resistance={cond1_short}, close_down={cond2_short}, volume_confirm={cond3_short}] -> short_met={short_met}"
            )
    
        # Decision: require strict majority and >= min_cond to declare direction
        if long_met >= min_cond and long_met > short_met:
            if debug:
                print(
                    f"[{self.symbol}] [Support/Resistance] Signal: LONG | support={support}, resistance={resistance}, "
                    f"close={close}, long_met={long_met}, short_met={short_met}, support_prox={support_prox:.6f}"
                )
            return "LONG"
        elif short_met >= min_cond and short_met > long_met:
            if debug:
                print(
                    f"[{self.symbol}] [Support/Resistance] Signal: SHORT | support={support}, resistance={resistance}, "
                    f"close={close}, long_met={long_met}, short_met={short_met}, resistance_prox={resistance_prox:.6f}"
                )
            return "SHORT"
        else:
            if debug:
                print(
                    f"[{self.symbol}] [Support/Resistance] No signal | long_met={long_met}, short_met={short_met}, "
                    f"support_prox={support_prox:.6f}, resistance_prox={resistance_prox:.6f}"
                )
            return None

    def _check_fractal_zone(self, window=50, min_atr_mult=0.5, min_conditions=2, range_volatility_filter=True, debug=False):
        """
        ENHANCED Fractal Zone (2026-03-05):
        Detects structural support/resistance breaks with improved robustness.
        
        Improvements:
        1. Larger window (50 vs 20 candles) for better structure detection
        2. ATR-based adaptive buffer (not fixed 0.5%)
        3. Range volatility filter to skip choppy consolidation
        4. Better for all timeframes
        
        Args:
            window: Lookback for fractal extremes (50 = ~12h on 15min, ~25h on 30min, ~50h on 1h)
            min_atr_mult: Multiplier for ATR-based buffer (0.5 = conservative)
            min_conditions: Require 2+ conditions met (out of 3)
            range_volatility_filter: Skip signals in choppy/consolidating ranges
            debug: Print debug info
        """
        try:
            fractal_low, _ = self._get_rolling_extremes('low', window)
            _, fractal_high = self._get_rolling_extremes('high', window)
            fractal_low_prev, _ = self._get_rolling_extremes('low', window, prev=True)
            _, fractal_high_prev = self._get_rolling_extremes('high', window, prev=True)
            
            close = self.df['close'].iat[-1]
            close_prev = self.df['close'].iat[-2]
            atr = self.df['atr'].iat[-1] if 'atr' in self.df.columns and self.df['atr'].iat[-1] is not None else 0
            
            # ENHANCEMENT 1: Dynamic ATR-based buffer (adaptive to volatility)
            if atr > 0 and fractal_low > 0:
                buffer = (atr / fractal_low) * min_atr_mult  # Scales with volatility
            else:
                buffer = 0.005  # Fallback to 0.5% if ATR not available
            
            # ENHANCEMENT 2: Range volatility filter (skip choppy consolidation)
            fractal_range = fractal_high - fractal_low if fractal_high and fractal_low else 0
            fractal_range_prev = fractal_high_prev - fractal_low_prev if fractal_high_prev and fractal_low_prev else 0
            
            if range_volatility_filter and fractal_range > 0 and fractal_range_prev > 0:
                mean_range = (fractal_range + fractal_range_prev) / 2
                if fractal_range < mean_range * 0.7:  # Current range 30% smaller = choppy
                    if debug:
                        print(
                            f"[{self.symbol}] [Fractal Zone] Skipped (choppy): range={fractal_range:.2f} < "
                            f"mean={mean_range:.2f} (threshold={mean_range * 0.7:.2f})"
                        )
                    return None
            
            # Conditions: 3 checks (structural break + momentum + trend)
            cond1_long = close > fractal_low * (1 + buffer)    # Price above structural low + buffer
            cond2_long = close > close_prev                     # Momentum: higher close
            cond3_long = fractal_low > fractal_low_prev        # Trend: higher lows (bullish structure)
            
            cond1_short = close < fractal_high * (1 - buffer)   # Price below structural high - buffer
            cond2_short = close < close_prev                    # Momentum: lower close
            cond3_short = fractal_high < fractal_high_prev     # Trend: lower highs (bearish structure)
            
            long_met = sum([cond1_long, cond2_long, cond3_long])
            short_met = sum([cond1_short, cond2_short, cond3_short])
            
            if short_met >= min_conditions and short_met > long_met:
                if debug:
                    print(
                        f"[{self.symbol}] [Fractal Zone ENHANCED] Signal: SHORT | "
                        f"short_met={short_met}, long_met={long_met}, buffer={buffer:.4f}, "
                        f"fractal_high={fractal_high:.2f}, fractal_high_prev={fractal_high_prev:.2f}, "
                        f"range={fractal_range:.2f}"
                    )
                return "SHORT"
            elif long_met >= min_conditions and long_met > short_met:
                if debug:
                    print(
                        f"[{self.symbol}] [Fractal Zone ENHANCED] Signal: LONG | "
                        f"long_met={long_met}, short_met={short_met}, buffer={buffer:.4f}, "
                        f"fractal_low={fractal_low:.2f}, fractal_low_prev={fractal_low_prev:.2f}, "
                        f"range={fractal_range:.2f}"
                    )
                return "LONG"
            else:
                if debug:
                    print(
                        f"[{self.symbol}] [Fractal Zone ENHANCED] No signal | "
                        f"short_met={short_met}, long_met={long_met}, buffer={buffer:.4f}"
                    )
                return None
        
        except Exception as e:
            print(f"[{self.symbol}] [Fractal Zone] Error: {e}", flush=True)
            return None

    def _check_hh_ll(self, lookback=3, range_threshold_pct=0.3, debug=False):
        """
        ENHANCED HH/LL Trend with Lookback + Range Check (2026-03-05)
        
        Checks for consistent Higher Highs/Higher Lows over lookback period
        Only fires in markets with meaningful range (excludes choppy/stale moves)
        
        Parameters:
        - lookback: Number of bars to confirm trend (3 = last 3 bars must show pattern)
        - range_threshold_pct: Min High-Low range as % of close (0.3% = 30 bps minimum)
        
        Logic:
        LONG: ≥2 consecutive bars with HH AND ≥2 with HL, High-Low range > threshold
        SHORT: ≥2 consecutive bars with LH AND ≥2 with LL, High-Low range > threshold
        """
        try:
            current_high = self.df['high'].iat[-1]
            current_low = self.df['low'].iat[-1]
            current_close = self.df['close'].iat[-1]
            
            # Calculate current bar's High-Low range
            current_range = current_high - current_low
            current_range_pct = (current_range / current_close) * 100 if current_close > 0 else 0
            
            # Gate: Avoid trading in choppy/stale markets (very small range)
            if current_range_pct < range_threshold_pct:
                if debug:
                    print(f"[{self.symbol}] [HH/LL Trend] SKIP: Range too small ({current_range_pct:.4f}% < {range_threshold_pct}%)")
                return None
            
            # Count Higher Highs and Higher Lows over lookback period
            hh_count = 0
            hl_count = 0
            lh_count = 0
            ll_count = 0
            
            for i in range(1, min(lookback + 1, len(self.df))):
                bar_high = self.df['high'].iat[-i]
                bar_high_prev = self.df['high'].iat[-(i+1)]
                bar_low = self.df['low'].iat[-i]
                bar_low_prev = self.df['low'].iat[-(i+1)]
                
                # Count Higher Highs and Higher Lows
                if bar_high > bar_high_prev:
                    hh_count += 1
                if bar_low > bar_low_prev:
                    hl_count += 1
                
                # Count Lower Highs and Lower Lows
                if bar_high < bar_high_prev:
                    lh_count += 1
                if bar_low < bar_low_prev:
                    ll_count += 1
            
            # Trend confirmation: need majority of lookback bars showing pattern
            hh_threshold = (lookback // 2) + 1  # At least half + 1
            
            if debug:
                print(f"[{self.symbol}] [HH/LL Trend] Lookback={lookback} | Range={current_range_pct:.4f}% | HH={hh_count}, HL={hl_count}, LH={lh_count}, LL={ll_count}")
            
            # LONG: Higher Highs AND Higher Lows trend confirmed
            if hh_count >= hh_threshold and hl_count >= hh_threshold and hh_count > lh_count:
                print(f"[{self.symbol}] [HH/LL Trend] Signal: LONG | HH={hh_count}, HL={hl_count} (lookback={lookback}, range={current_range_pct:.4f}%)")
                return "LONG"
            
            # SHORT: Lower Highs AND Lower Lows trend confirmed
            elif ll_count >= hh_threshold and lh_count >= hh_threshold and ll_count > hl_count:
                print(f"[{self.symbol}] [HH/LL Trend] Signal: SHORT | LH={lh_count}, LL={ll_count} (lookback={lookback}, range={current_range_pct:.4f}%)")
                return "SHORT"
            
            else:
                if debug:
                    print(f"[{self.symbol}] [HH/LL Trend] No signal | HH={hh_count}, HL={hl_count}, LH={lh_count}, LL={ll_count}")
                return None
        
        except Exception as e:
            print(f"[{self.symbol}] [HH/LL Trend] Error: {e}")
            return None

    def _check_liquidity_pool(self, lookback=20, min_cond=2, debug=False):
        close = self.df['close'].iat[-1]
        high = self.df['high'].iat[-1]
        low = self.df['low'].iat[-1]

        # Use helper for previous rolling extremes
        recent_low, recent_high = self._get_rolling_extremes('low', lookback, prev=True)
        # Note: For liquidity pools, recent_high = previous rolling max, recent_low = previous rolling min

        liquidity_pool = {
            "recent_high": recent_high,
            "recent_low": recent_low,
            "close_vs_high": close - recent_high,
            "close_vs_low": close - recent_low
        }

        cond1_long = close > recent_high
        cond2_long = low < recent_low and close > recent_low

        cond1_short = close < recent_low
        cond2_short = high > recent_high and close < recent_high

        long_met = sum([cond1_long, cond2_long])
        short_met = sum([cond1_short, cond2_short])

        if long_met >= min_cond and long_met > short_met:
            print(f"[{self.symbol}] [Liquidity Pool] Signal: LONG | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, liquidity_pool={liquidity_pool}")
            return "LONG"
        elif short_met >= min_cond and short_met > long_met:
            print(f"[{self.symbol}] [Liquidity Pool] Signal: SHORT | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, liquidity_pool={liquidity_pool}")
            return "SHORT"
        else:
            if debug:
                print(f"[{self.symbol}] [Liquidity Pool] No signal fired | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, liquidity_pool={liquidity_pool}")
            return None

    def _check_atr_momentum_burst(
        self,
        threshold_ratio=0.15,
        volume_mult=1.5,
        lookback=3,
        volume_ma_period=10,
        min_atr=0.5,
        min_momentum_bars=2,
        debug=False
    ):
        """
        ENHANCED ATR Momentum Burst (2026-03-05)
        
        Detects sustained momentum bursts with:
        - ATR-scaled thresholds (normalizes to volatility)
        - Strong volume confirmation (50% above average)
        - Lookback period for multi-bar confirmation
        - Directional consistency (no flip-flops)
        
        Parameters:
        - threshold_ratio: Move/ATR ratio > threshold (0.15 = 15% of ATR)
        - volume_mult: Volume must exceed MA by this multiple (1.5 = 50% above avg)
        - lookback: Number of bars to check for momentum (3 = last 3 bars)
        - volume_ma_period: SMA period for volume average (10 = 10-bar MA)
        - min_atr: Minimum ATR to trade (avoid low-volatility symbols)
        - min_momentum_bars: Minimum bars showing momentum (2 = at least 2 of lookback)
        
        Logic:
        LONG: ≥2 of last 3 bars show (move/ATR > 0.15 AND volume > 1.5x avg) 
              AND majority bars are bullish (close > open)
        SHORT: Same but for bearish direction
        """
        try:
            # Get current ATR
            current_atr = self.df['atr'].iat[-1] if 'atr' in self.df.columns else None
            
            # Gate: Require minimum ATR for market quality
            if current_atr is None or current_atr < min_atr:
                if debug:
                    print(f"[{self.symbol}] [ATR Momentum Burst] SKIP: ATR too low ({current_atr} < {min_atr})")
                return None
            
            # Scan lookback period for momentum bars
            momentum_bars_list = []
            directions = []  # Track bullish (+1) or bearish (-1) bars
            
            for i in range(1, min(lookback + 1, len(self.df))):
                bar_open = self.df['open'].iat[-i]
                bar_close = self.df['close'].iat[-i]
                bar_volume = self.df['volume'].iat[-i]
                
                # Calculate metrics for this bar
                pct_move = ((bar_close - bar_open) / bar_open * 100) if bar_open > 0 else 0
                atr_ratio = abs(pct_move) / current_atr if current_atr > 0 else 0
                volume_ma = self.df['volume'].rolling(volume_ma_period).mean().iat[-i]
                
                # Direction: +1 for bullish (close > open), -1 for bearish
                direction = 1 if bar_close > bar_open else (-1 if bar_close < bar_open else 0)
                directions.append(direction)
                
                # Check: ATR-scaled move + strong volume
                if atr_ratio > threshold_ratio and bar_volume > volume_ma * volume_mult:
                    momentum_bars_list.append({
                        'bar': i,
                        'pct_move': pct_move,
                        'atr_ratio': atr_ratio,
                        'volume': bar_volume,
                        'volume_ma': volume_ma,
                        'direction': direction
                    })
            
            if debug:
                print(f"[{self.symbol}] [ATR Momentum Burst] Lookback={lookback} | ATR={current_atr:.6f} | Momentum bars found: {len(momentum_bars_list)}")
                for bar in momentum_bars_list:
                    print(f"  Bar {bar['bar']}: move={bar['pct_move']:.4f}%, ratio={bar['atr_ratio']:.4f}, vol={bar['volume']:.0f} (avg={bar['volume_ma']:.0f})")
            
            # Need minimum momentum bars
            if len(momentum_bars_list) < min_momentum_bars:
                if debug:
                    print(f"[{self.symbol}] [ATR Momentum Burst] No signal | {len(momentum_bars_list)} momentum bars < {min_momentum_bars} required")
                return None
            
            # Check directional consistency
            bullish_count = sum(1 for d in directions if d > 0)
            bearish_count = sum(1 for d in directions if d < 0)
            
            # LONG: Majority of lookback bars are bullish + multiple momentum bars
            if bullish_count >= 2 and bullish_count > bearish_count and len(momentum_bars_list) >= min_momentum_bars:
                momentum_summary = f"{len(momentum_bars_list)} bars, avg_ratio={sum(b['atr_ratio'] for b in momentum_bars_list)/len(momentum_bars_list):.4f}"
                print(f"[{self.symbol}] [ATR Momentum Burst] Signal: LONG | {momentum_summary} | ATR={current_atr:.6f}")
                return "LONG"
            
            # SHORT: Majority of lookback bars are bearish + multiple momentum bars
            elif bearish_count >= 2 and bearish_count > bullish_count and len(momentum_bars_list) >= min_momentum_bars:
                momentum_summary = f"{len(momentum_bars_list)} bars, avg_ratio={sum(b['atr_ratio'] for b in momentum_bars_list)/len(momentum_bars_list):.4f}"
                print(f"[{self.symbol}] [ATR Momentum Burst] Signal: SHORT | {momentum_summary} | ATR={current_atr:.6f}")
                return "SHORT"
            
            else:
                if debug:
                    print(f"[{self.symbol}] [ATR Momentum Burst] No signal | bullish={bullish_count}, bearish={bearish_count}, momentum_bars={len(momentum_bars_list)}")
                return None
        
        except Exception as e:
            print(f"[{self.symbol}] [ATR Momentum Burst] Error: {e}")
            return None


    def _check_volatility_model(
        self,
        atr_col='atr',
        atr_ma_col='atr_ma',
        atr_expansion_pct=0.15,
        lookback=2,
        volume_mult=1.3,
        min_atr=0.5,
        volume_confirm=True,
        direction_threshold=2,
        volume_ma_period=10,
        debug=False
    ):
        """
        ENHANCED VOLATILITY MODEL - MODERATE Rules (2026-03-05)
        
        Detects volatility expansion with balanced/moderate filtering:
        - ATR expansion: 15% above MA (percentage-based, normalized)
        - Lookback: 2 bars, but 1+ bars showing trend OK (flexible)
        - Direction: Need 2 of 3 conditions to fire (ATR, Price, Volume)
        - Volume confirmation: 1.3x average (moderate gate)
        - Min ATR: 0.5 (avoids stale markets)
        
        Parameters:
        - atr_expansion_pct: ATR must be X% above its MA (0.15 = 15%)
        - lookback: Check last N bars for expansion trend (2 = last 2)
        - volume_mult: Volume > MA × this (1.3 = 30% above average)
        - min_atr: Minimum ATR value for market quality
        - volume_confirm: Require volume confirmation (True = enabled)
        - direction_threshold: Need N of 3 conditions (2 = moderate, 3 = strict)
        - volume_ma_period: Period for volume MA (10 = 10-bar MA)
        
        Logic:
        LONG: ATR expanding 15% above MA + 2 of [ATR↑, Price↑, Volume↑]
        SHORT: Same but for bearish direction
        """
        try:
            # Get current values
            current_atr = self.df[atr_col].iat[-1] if atr_col in self.df.columns else None
            current_atr_prev = self.df[atr_col].iat[-2] if atr_col in self.df.columns else None
            current_atr_ma = self.df[atr_ma_col].iat[-1] if atr_ma_col in self.df.columns else None
            current_close = self.df['close'].iat[-1]
            current_close_prev = self.df['close'].iat[-2]
            current_volume = self.df['volume'].iat[-1]
            current_volume_ma = self.df['volume'].rolling(volume_ma_period).mean().iat[-1]
            
            # Gate 1: Minimum ATR requirement (market quality)
            if current_atr is None or current_atr < min_atr:
                if debug:
                    print(f"[{self.symbol}] [Volatility Model] SKIP: ATR too low ({current_atr} < {min_atr})")
                return None
            
            # Gate 2: ATR expansion magnitude (percentage-based, normalized)
            atr_expansion_pct_actual = (current_atr - current_atr_ma) / current_atr_ma if current_atr_ma > 0 else 0
            if atr_expansion_pct_actual < atr_expansion_pct:
                if debug:
                    print(f"[{self.symbol}] [Volatility Model] SKIP: Expansion too small ({atr_expansion_pct_actual:.4f} < {atr_expansion_pct})")
                return None
            
            # Gate 3: Lookback period - check if expansion is consistent (flexible)
            expansion_bars = 0
            for i in range(1, min(lookback + 1, len(self.df))):
                bar_atr = self.df[atr_col].iat[-i]
                bar_atr_ma = self.df[atr_ma_col].iat[-i]
                if bar_atr_ma > 0:
                    bar_expansion_pct = (bar_atr - bar_atr_ma) / bar_atr_ma
                    # Flexible: 80% of target threshold (not strict)
                    if bar_expansion_pct > atr_expansion_pct * 0.8:
                        expansion_bars += 1
            
            # Need at least 1 bar showing expansion (flexible, not strict)
            if expansion_bars < 1:
                if debug:
                    print(f"[{self.symbol}] [Volatility Model] SKIP: No expansion bars ({expansion_bars} < 1)")
                return None
            
            # Direction conditions (MODERATE: need 2 of 3)
            cond_atr_up = current_atr > current_atr_prev if current_atr_prev else False
            cond_price_up = current_close > current_close_prev
            cond_volume_up = current_volume > current_volume_ma
            
            # Count conditions for bullish direction
            long_conditions = sum([cond_atr_up, cond_price_up, cond_volume_up])
            
            # Count conditions for bearish direction
            cond_atr_down = current_atr < current_atr_prev if current_atr_prev else False
            cond_price_down = current_close < current_close_prev
            short_conditions = sum([cond_atr_down, cond_price_down, cond_volume_up])  # Note: volume same for both
            
            if debug:
                print(f"[{self.symbol}] [Volatility Model] Conditions | ATR↑={cond_atr_up}, Price↑={cond_price_up}, Vol↑={cond_volume_up}")
                print(f"  Expansion: {expansion_bars} bars, actual={atr_expansion_pct_actual:.4f} (target={atr_expansion_pct})")
                print(f"  Long conds={long_conditions}, Short conds={short_conditions}")
            
            # LONG: ATR expanding + need 2 of [ATR↑, Price↑, Volume↑]
            # Volume confirmation can be optional, but if enabled, prefer it
            if long_conditions >= direction_threshold and cond_atr_up:
                vol_check = (cond_volume_up or not volume_confirm)  # Pass if vol↑ OR confirm disabled
                if vol_check:
                    print(f"[{self.symbol}] [Volatility Model] Signal: LONG | expansion={atr_expansion_pct_actual:.4f}, conds={long_conditions}/3, ATR={current_atr:.6f}")
                    return "LONG"
            
            # SHORT: ATR expanding + need 2 of [ATR↓, Price↓, Volume↑]
            if short_conditions >= direction_threshold and cond_atr_down:
                vol_check = (cond_volume_up or not volume_confirm)
                if vol_check:
                    print(f"[{self.symbol}] [Volatility Model] Signal: SHORT | expansion={atr_expansion_pct_actual:.4f}, conds={short_conditions}/3, ATR={current_atr:.6f}")
                    return "SHORT"
            
            if debug:
                print(f"[{self.symbol}] [Volatility Model] No signal | long={long_conditions}, short={short_conditions}")
            return None
        
        except Exception as e:
            print(f"[{self.symbol}] [Volatility Model] Error: {e}")
            return None

    def _check_volatility_squeeze(
        self,
        min_cond=2,
        min_squeeze_diff=0.05,
        volume_confirm=True,
        debug=False
    ):
        """
        Enhanced Volatility Squeeze:
        - Minimum squeeze magnitude filter.
        - Optional volume confirmation for breakouts.
        - Detailed logging.
        """
        bb_width = self.df['bb_upper'].iat[-1] - self.df['bb_lower'].iat[-1]
        kc_width = self.df['kc_upper'].iat[-1] - self.df['kc_lower'].iat[-1]
        bb_width_prev = self.df['bb_upper'].iat[-2] - self.df['bb_lower'].iat[-2]
        kc_width_prev = self.df['kc_upper'].iat[-2] - self.df['kc_lower'].iat[-2]
    
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
        volume_ma = self.df['volume'].rolling(10).mean().iat[-1]
    
        squeeze_diff = bb_width - kc_width
        squeeze_firing = (
            squeeze_diff > min_squeeze_diff
            and bb_width_prev < kc_width_prev
        )
    
        squeeze = {
            "bb_width": bb_width,
            "kc_width": kc_width,
            "bb_width_prev": bb_width_prev,
            "kc_width_prev": kc_width_prev,
            "squeeze_firing": squeeze_firing,
            "squeeze_diff": squeeze_diff,
            "min_squeeze_diff": min_squeeze_diff,
        }
        volatility = {
            "close": close,
            "close_prev": close_prev,
            "volume": volume,
            "volume_prev": volume_prev,
            "volume_ma": volume_ma,
        }
    
        cond1_long = squeeze_firing
        cond2_long = close > close_prev
        cond3_long = volume > volume_prev if volume_confirm else True
        cond4_long = volume > volume_ma if volume_confirm else True
    
        cond1_short = squeeze_firing
        cond2_short = close < close_prev
        cond3_short = volume > volume_prev if volume_confirm else True
        cond4_short = volume > volume_ma if volume_confirm else True
    
        long_met = sum([cond1_long, cond2_long, cond3_long, cond4_long])
        short_met = sum([cond1_short, cond2_short, cond3_short, cond4_short])
    
        if debug:
            print(
                f"[{self.symbol}] [Volatility Squeeze] Metrics | squeeze={squeeze}, volatility={volatility}, long_met={long_met}, short_met={short_met}"
            )
    
        if long_met >= min_cond and long_met > short_met:
            print(
                f"[{self.symbol}] [Volatility Squeeze] Signal: LONG | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, squeeze={squeeze}, volatility={volatility}"
            )
            return "LONG"
        elif short_met >= min_cond and short_met > long_met:
            print(
                f"[{self.symbol}] [Volatility Squeeze] Signal: SHORT | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, squeeze={squeeze}, volatility={volatility}"
            )
            return "SHORT"
        else:
            print(
                f"[{self.symbol}] [Volatility Squeeze] No signal fired | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, squeeze={squeeze}, volatility={volatility}"
            )
            return None

    def _check_chop_zone(
        self,
        chop_threshold=40,
        adx_strict=25,
        debug=False
    ):
        """
        Enhanced Chop Zone Check:
        - Configurable choppiness and ADX thresholds.
        - Logs all relevant metrics.
        - Keeps signal logic robust.
        """
        try:
            chop = self.df['chop_zone'].iat[-1]
            ema9 = self.df['ema9'].iat[-1]
            ema21 = self.df['ema21'].iat[-1]
            close = self.df['close'].iat[-1]
            adx = self.df['adx'].iat[-1] if 'adx' in self.df.columns else None
        except (KeyError, IndexError):
            print(f"[{self.symbol}] [Chop Zone Check] Missing required columns or insufficient data.")
            return None
    
        chop_zone = chop
    
        if chop >= chop_threshold:
            print(f"[{self.symbol}] [Chop Zone Check] Market too choppy (chop_zone={chop_zone:.2f} >= {chop_threshold})")
            return None
    
        cond1_long = ema9 > ema21
        cond2_long = adx is not None and adx > adx_strict if adx is not None else True
        cond3_long = close > ema9
    
        cond1_short = ema9 < ema21
        cond2_short = adx is not None and adx > adx_strict if adx is not None else True
        cond3_short = close < ema9
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        if debug:
            print(
                f"[{self.symbol}] [Chop Zone Check] Metrics | chop_zone={chop_zone}, ema9={ema9}, ema21={ema21}, adx={adx}, adx_strict={adx_strict}, long_met={long_met}, short_met={short_met}"
            )
    
        if long_met >= 2 and long_met > short_met:
            print(
                f"[{self.symbol}] [Chop Zone Check] Signal: LONG | long_met={long_met}, short_met={short_met}, chop_zone={chop_zone}, ema9={ema9}, ema21={ema21}, adx={adx}"
            )
            return "LONG"
        elif short_met >= 2 and short_met > long_met:
            print(
                f"[{self.symbol}] [Chop Zone Check] Signal: SHORT | long_met={long_met}, short_met={short_met}, chop_zone={chop_zone}, ema9={ema9}, ema21={ema21}, adx={adx}"
            )
            return "SHORT"
        else:
            print(
                f"[{self.symbol}] [Chop Zone Check] No signal fired | long_met={long_met}, short_met={short_met}, chop_zone={chop_zone}, ema9={ema9}, ema21={ema21}, adx={adx}"
            )
            return None

    def _check_candle_confirmation(
        self,
        min_pin_wick_ratio=2.0,
        require_volume_confirm=False,
        debug=False
    ):
        """
        Enhanced Candle Confirmation (revised to avoid bias toward LONG):
        - Proper engulfing detection requires that the previous candle was of the opposite direction.
        - Engulfing also requires current body to be meaningfully larger than previous body (to avoid micro-engulfs).
        - Pin bar logic uses correct wick/body measurements and includes optional volume confirmation.
        - Returns "LONG", "SHORT", or None.
        """
        # Defensive checks
        if len(self.df) < 2:
            if debug:
                print(f"[{self.symbol}] [Candle Confirmation] Not enough data (need at least 2 rows).")
            return None

        open_ = float(self.df['open'].iat[-1])
        high = float(self.df['high'].iat[-1])
        low = float(self.df['low'].iat[-1])
        close = float(self.df['close'].iat[-1])
        volume = float(self.df['volume'].iat[-1]) if 'volume' in self.df.columns else None

        open_prev = float(self.df['open'].iat[-2])
        close_prev = float(self.df['close'].iat[-2])
        volume_prev = float(self.df['volume'].iat[-2]) if 'volume' in self.df.columns else None
        volume_ma = self.df['volume'].rolling(10).mean().iat[-1] if 'volume' in self.df.columns else None

        # Body sizes
        body = abs(close - open_)
        prev_body = abs(close_prev - open_prev)

        # Determine previous candle direction
        prev_bearish = close_prev < open_prev
        prev_bullish = close_prev > open_prev

        # Engulfing: require previous candle to be opposite, and current body to engulf previous body
        bullish_engulfing = (
            prev_bearish and
            close > open_ and
            open_ < close_prev and
            close > open_prev and
            body > prev_body * 1.05  # require slightly larger body (5% larger) to avoid tiny "engulfs"
        )

        bearish_engulfing = (
            prev_bullish and
            close < open_ and
            open_ > close_prev and
            close < open_prev and
            body > prev_body * 1.05
        )

        # Pin bar logic (lower/upper wick relative to body)
        lower_wick = min(open_, close) - low
        upper_wick = high - max(open_, close)
        # Avoid division by zero: use body or total range as denominator
        denom = body if body > 0 else (high - low if (high - low) > 0 else 1.0)

        bullish_pin_bar = (lower_wick > min_pin_wick_ratio * denom) and (close > open_) and (body <= prev_body * 1.2)
        bearish_pin_bar = (upper_wick > min_pin_wick_ratio * denom) and (close < open_) and (body <= prev_body * 1.2)

        # Candle type for logging
        if bullish_engulfing:
            candle_type = "Bullish Engulfing"
        elif bearish_engulfing:
            candle_type = "Bearish Engulfing"
        elif bullish_pin_bar:
            candle_type = "Bullish Pin Bar"
        elif bearish_pin_bar:
            candle_type = "Bearish Pin Bar"
        else:
            candle_type = "Neutral"

        # Optional volume confirmation (if requested, require volume to be meaningfully above average or prev)
        if require_volume_confirm and volume is not None:
            vol_ok = False
            if volume_ma is not None and not (isinstance(volume_ma, float) and math.isnan(volume_ma)):
                vol_ok = volume > max(volume_ma * 1.05, (volume_prev or 0) * 1.05)
            else:
                vol_ok = volume > (volume_prev or 0) * 1.05
        else:
            vol_ok = True

        # LONG conditions
        cond1_long = bullish_engulfing
        cond2_long = bullish_pin_bar
        cond3_long = close > close_prev
        cond4_long = vol_ok

        # SHORT conditions
        cond1_short = bearish_engulfing
        cond2_short = bearish_pin_bar
        cond3_short = close < close_prev
        cond4_short = vol_ok

        long_met = sum([cond1_long, cond2_long, cond3_long, cond4_long])
        short_met = sum([cond1_short, cond2_short, cond3_short, cond4_short])

        log_info = (
            f"candle_type={candle_type}, open={open_}, close={close}, high={high}, low={low}, "
            f"body={body:.6f}, upper_wick={upper_wick:.6f}, lower_wick={lower_wick:.6f}, "
            f"open_prev={open_prev}, close_prev={close_prev}, prev_body={prev_body:.6f}, volume={volume}, volume_prev={volume_prev}, volume_ma={volume_ma}, "
            f"min_pin_wick_ratio={min_pin_wick_ratio}, require_volume_confirm={require_volume_confirm}"
        )

        if debug:
            print(f"[{self.symbol}] [Candle Confirmation] {log_info}")
            print(f"[{self.symbol}] [Candle Confirmation] conds_long={[cond1_long, cond2_long, cond3_long, cond4_long]}, conds_short={[cond1_short, cond2_short, cond3_short, cond4_short]}, long_met={long_met}, short_met={short_met}")

        # Use a stricter decision rule to avoid one-off candles producing GKs:
        # require at least 2 meaningful confirmations and a strict majority
        if long_met >= 2 and long_met > short_met:
            if debug:
                print(f"[{self.symbol}] [Candle Confirmation] Signal: LONG | {log_info}")
            return "LONG"
        elif short_met >= 2 and short_met > long_met:
            if debug:
                print(f"[{self.symbol}] [Candle Confirmation] Signal: SHORT | {log_info}")
            return "SHORT"
        else:
            if debug:
                print(f"[{self.symbol}] [Candle Confirmation] No signal | long_met={long_met}, short_met={short_met}")
            return None   
            
    def _check_wick_dominance(
        self,
        min_wick_dom_ratio=2.0,
        min_body_ratio=1.5,
        require_volume_confirm=False,
        debug=False
    ):
        """
        Enhanced Wick Dominance:
        - Configurable wick dominance and body ratio thresholds.
        - Optionally requires volume confirmation.
        - Standardized detailed logging.
        """
        open_ = self.df['open'].iat[-1]
        high = self.df['high'].iat[-1]
        low = self.df['low'].iat[-1]
        close = self.df['close'].iat[-1]
        volume = self.df['volume'].iat[-1]
        volume_ma = self.df['volume'].rolling(10).mean().iat[-1] if 'volume' in self.df.columns else None
    
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        body = abs(close - open_)
    
        # Optional volume confirmation
        volume_confirm = volume > volume_ma if require_volume_confirm and volume_ma is not None else True
    
        # Wick dominance logic (configurable thresholds)
        cond1_long = lower_wick > min_wick_dom_ratio * upper_wick
        cond2_long = lower_wick > min_body_ratio * body
        cond3_long = close > open_
        cond4_long = volume_confirm
    
        cond1_short = upper_wick > min_wick_dom_ratio * lower_wick
        cond2_short = upper_wick > min_body_ratio * body
        cond3_short = close < open_
        cond4_short = volume_confirm
    
        long_met = sum([cond1_long, cond2_long, cond3_long, cond4_long])
        short_met = sum([cond1_short, cond2_short, cond3_short, cond4_short])
    
        # Wick dominance for logging
        if lower_wick > upper_wick:
            wick_dom = f"Lower wick dominant ({lower_wick:.2f} > {upper_wick:.2f})"
        elif upper_wick > lower_wick:
            wick_dom = f"Upper wick dominant ({upper_wick:.2f} > {lower_wick:.2f})"
        else:
            wick_dom = f"No dominance (upper={upper_wick:.2f}, lower={lower_wick:.2f})"
    
        log_info = (
            f"wick_dom={wick_dom}, open={open_}, close={close}, high={high}, low={low}, "
            f"body={body:.2f}, upper_wick={upper_wick:.2f}, lower_wick={lower_wick:.2f}, volume={volume}, volume_ma={volume_ma}, "
            f"min_wick_dom_ratio={min_wick_dom_ratio}, min_body_ratio={min_body_ratio}, require_volume_confirm={require_volume_confirm}"
        )
    
        if long_met >= 2 and long_met > short_met:
            if debug:
                print(f"[{self.symbol}] [Wick Dominance] Signal: LONG | long_met={long_met}, short_met={short_met}, {log_info}")
            return "LONG"
        elif short_met >= 2 and short_met > long_met:
            if debug:
                print(f"[{self.symbol}] [Wick Dominance] Signal: SHORT | long_met={long_met}, short_met={short_met}, {log_info}")
            return "SHORT"
        else:
            if debug:
                print(f"[{self.symbol}] [Wick Dominance] No signal fired | long_met={long_met}, short_met={short_met}, {log_info}")
            return None

    def _market_regime(
        self,
        ma_col='ema200',
        adx_col='adx',
        adx_threshold=15,
        range_adx_threshold=10,
        secondary_ma_col=None,
        debug=False
    ):
        """
        Enhanced Market Regime Detector.
        Returns:
            'BULL'      -- strong uptrend
            'BEAR'      -- strong downtrend
            'RANGE'     -- sideways/ranging market
            'NO_REGIME' -- cannot classify (missing data)
        """
        try:
            # Check required columns
            if ma_col not in self.df.columns or adx_col not in self.df.columns:
                if debug:
                    print(f"[Market Regime] Missing required column(s): {ma_col}, {adx_col}")
                return 'NO_REGIME'
    
            close = self.df['close'].iat[-1]
            ma = self.df[ma_col].iat[-1]
            ma_prev = self.df[ma_col].iat[-2]
            adx = self.df[adx_col].iat[-1]
    
            # If any critical value is nan, return NO_REGIME
            if any(map(lambda x: x is None or (isinstance(x, float) and math.isnan(x)), [close, ma, ma_prev, adx])):
                if debug:
                    print(f"[Market Regime] NaN detected in input values: close={close}, ma={ma}, ma_prev={ma_prev}, adx={adx}")
                return 'NO_REGIME'
    
            # Optionally use a secondary MA for confirmation (e.g., EMA50 above EMA200 for bull)
            secondary_trend = True
            if secondary_ma_col and secondary_ma_col in self.df.columns:
                secondary_ma = self.df[secondary_ma_col].iat[-1]
                secondary_ma_prev = self.df[secondary_ma_col].iat[-2]
                # Confirm secondary MA is also trending in the same direction
                if close > ma:
                    secondary_trend = secondary_ma > ma and secondary_ma > secondary_ma_prev
                elif close < ma:
                    secondary_trend = secondary_ma < ma and secondary_ma < secondary_ma_prev
    
            # ADX regime logic
            if adx >= adx_threshold:
                # Strong trend
                if close > ma and ma > ma_prev and secondary_trend:
                    regime = 'BULL'
                elif close < ma and ma < ma_prev and secondary_trend:
                    regime = 'BEAR'
                else:
                    regime = 'RANGE'
            elif adx >= range_adx_threshold:
                # Weak trend/range zone
                regime = 'RANGE'
            else:
                # Choppy/flat, low strength
                regime = 'RANGE'
    
            if debug:
                print(
                    f"[Market Regime] close={close}, ma={ma}, ma_prev={ma_prev}, adx={adx}, "
                    f"secondary_ma_col={secondary_ma_col}, secondary_trend={secondary_trend}, regime={regime}"
                )
            print(f"[SUMMARY] Market regime detected: {regime}")
            return regime
    
        except Exception as e:
            if debug:
                print(f"[Market Regime] Error: {e}")
            return 'NO_REGIME'
    
    def _fetch_order_book(self, depth=100):
        for sym in (self.symbol, self.symbol.replace('-', '/')):
            url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_{depth}?symbol={sym}"
            try:
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json().get('data', {})
                bids = pd.DataFrame(data.get('bids', []), columns=['price', 'size']).astype(float)
                asks = pd.DataFrame(data.get('asks', []), columns=['price', 'size']).astype(float)
                return bids, asks
            except Exception:
                continue
        return None, None

# End of file (modified Candle Confirmation only)
