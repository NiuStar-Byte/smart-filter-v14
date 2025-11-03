# (full file contents with only the _check_candle_confirmation function updated)
# NOTE: This block shows the whole file for context; only the Candle Confirmation function
# (def _check_candle_confirmation) was modified from the original version you provided.
# ... (the earlier imports and class definition remain unchanged)
# For brevity here I include the full file from the original source but with the corrected function.

# smart_filter.py

import datetime
import logging
import math
import requests
import pandas as pd
import numpy as np
from kucoin_orderbook import get_order_wall_delta
from kucoin_density import get_resting_density
from signal_debug_log import export_signal_debug_txt
from calculations import add_indicators, compute_rsi, calculate_cci, calculate_stochrsi, compute_williams_r
from typing import Optional

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
        df3m: Optional[pd.DataFrame] = None,
        df5m: Optional[pd.DataFrame] = None,
        tf: Optional[str] = None,
        min_score: int = 11,
        required_passed: Optional[int] = None,  # int or None allowed
        volume_multiplier: float = 2.0,
        liquidity_threshold: float = 0.25,
        kwargs: Optional[dict] = None
    ):
        if kwargs is None:
            kwargs = {}

        self.symbol = symbol
        self.df = add_indicators(df)
        # print(f"[{self.symbol}][{tf}] Indicators:", self.df.tail(5)[['adx','ema200','close']])
        self.df3m = add_indicators(df3m) if df3m is not None else None
        self.df5m = add_indicators(df5m) if df5m is not None else None

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

        # Weights for filters
        self.filter_weights_long = {
            "MACD": 5.0, "Volume Spike": 5.0, "Fractal Zone": 4.5, "TREND": 4.7, "Momentum": 4.9, "ATR Momentum Burst": 4.3,
            "MTF Volume Agreement": 5.0, "HH/LL Trend": 4.1, "Volatility Model": 3.9,
            "Liquidity Awareness": 5.0, "Volatility Squeeze": 3.7, "Candle Confirmation": 5.0,
            "VWAP Divergence": 3.5, "Spread Filter": 5.0, "Chop Zone": 3.3, "Liquidity Pool": 3.1, "Support/Resistance": 5.0,
            "Smart Money Bias": 2.9, "Absorption": 2.7, "Wick Dominance": 2.5
        }
        
        self.filter_weights_short = {
            "MACD": 5.0, "Volume Spike": 5.0, "Fractal Zone": 4.5, "TREND": 4.7, "Momentum": 4.9, "ATR Momentum Burst": 4.3,
            "MTF Volume Agreement": 5.0, "HH/LL Trend": 4.1, "Volatility Model": 3.9,
            "Liquidity Awareness": 5.0, "Volatility Squeeze": 3.7, "Candle Confirmation": 5.0,
            "VWAP Divergence": 3.5, "Spread Filter": 5.0, "Chop Zone": 3.3, "Liquidity Pool": 3.1, "Support/Resistance": 5.0,
            "Smart Money Bias": 2.9, "Absorption": 2.7, "Wick Dominance": 2.5
        }

        self.filter_names = list(set(self.filter_weights_long.keys()) | set(self.filter_weights_short.keys()))
        
        self.gatekeepers = [
            # "MACD",
            # "Volume Spike",
            # "MTF Volume Agreement",
            # "Liquidity Awareness",
            # "Spread Filter",
            "Candle Confirmation",
            "Support/Resistance"
        ]

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

    # PARTIAL New Super-GK ONLY Liquidity (Density) & OrderBookWall
    def superGK_check(self, signal_direction, orderbook_result, density_result):
        """
        Super Gatekeeper check using order book walls and resting liquidity density.
        Only Liquidity (Density) & OrderBookWall are considered for blocking/passing signals.
        """

        if orderbook_result is None or density_result is None:
            print(f"Signal blocked due to missing order book or density for {self.symbol}")
            return False

        bid_wall = orderbook_result.get('buy_wall', 0)
        ask_wall = orderbook_result.get('sell_wall', 0)

        bid_density = density_result.get('bid_density', 0)
        ask_density = density_result.get('ask_density', 0)

        # Liquidity threshold check
        if bid_density < self.liquidity_threshold or ask_density < self.liquidity_threshold:
            print(f"Signal blocked due to low liquidity for {self.symbol}")
            return False

        # Directional checks
        if signal_direction == "LONG":
            if bid_wall > ask_wall and bid_density > ask_density:
                print(f"Signal passed for LONG: bid_wall & bid_density stronger.")
                return True
            else:
                print(f"Signal blocked for LONG: SuperGK criteria not met for {self.symbol}")
                return False

        elif signal_direction == "SHORT":
            if ask_wall > bid_wall and ask_density > bid_density:
                print(f"Signal passed for SHORT: ask_wall & ask_density stronger.")
                return True
            else:
                print(f"Signal blocked for SHORT: SuperGK criteria not met for {self.symbol}")
                return False

        else:
            print(f"Signal blocked: SuperGK not aligned for {self.symbol} (direction NEUTRAL or undefined)")
            return False

    def analyze(self):
        reversal_detected = False  # <-- Add this here, at the very start of the function
        if self.df.empty:
            print(f"[{self.symbol}] Error: DataFrame empty.")
            return None

        # --- Detect reversal and set route correctly (AMBIGUOUS included) ---
        # reversal = self.explicit_reversal_gate()
        # reversal_detected = reversal in ["LONG", "SHORT"]
        reversal_route, reversal_side = self.explicit_reversal_gate()
        reversal_detected = reversal_route in ["REVERSAL", "AMBIGUOUS"]
        route = reversal_route if reversal_detected else "TREND CONTINUATION"
        
        # List all filter names
        filter_names = [
            "Fractal Zone", "TREND", "MACD", "Momentum", "Volume Spike",
            "VWAP Divergence", "MTF Volume Agreement", "HH/LL Trend",
            "Chop Zone", "Candle Confirmation", "Wick Dominance", "Absorption",
            "Support/Resistance", "Smart Money Bias", "Liquidity Pool", "Spread Filter",
            "Liquidity Awareness", "Volatility Model", "ATR Momentum Burst", "Volatility Squeeze"
        ]

        # Map filter names to functions, handling timeframes if needed
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

        # Prepare LONG and SHORT dictionaries
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
                # Always call with debug=True so all filters log every run
                result = fn(debug=True)
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

        # --- Non-GK filter list ---
        non_gk_filters = [f for f in filter_names if f not in self.gatekeepers]
        
        # --- Calculate score: count of passed NON-GK filters ---
        long_score = sum(1 for f in non_gk_filters if results_long.get(f, False))
        short_score = sum(1 for f in non_gk_filters if results_short.get(f, False))    

        # --- Get signal direction ---
        direction = self.get_signal_direction(results_long, results_short)
        self.bias = direction

        # --- Separate Hard and Soft Gatekeepers ---
        hard_gatekeepers = [gk for gk in self.gatekeepers if gk not in self.soft_gatekeepers]
        soft_gatekeepers = [gk for gk in self.gatekeepers if gk in self.soft_gatekeepers]
        
        # --- Passed/Failed lists for reporting (all GKs, for display/debug only) ---
        passed_gk_long = [gk for gk in self.gatekeepers if results_long.get(gk, False)]
        failed_gk_long = [gk for gk in self.gatekeepers if not results_long.get(gk, False)]
        passed_gk_short = [gk for gk in self.gatekeepers if results_short.get(gk, False)]
        failed_gk_short = [gk for gk in self.gatekeepers if not results_short.get(gk, False)]
        
        # --- Passed/Failed Hard GKs (for signal logic only) ---
        passed_hard_gk_long = [gk for gk in hard_gatekeepers if results_long.get(gk, False)]
        failed_hard_gk_long = [gk for gk in hard_gatekeepers if not results_long.get(gk, False)]
        passed_hard_gk_short = [gk for gk in hard_gatekeepers if results_short.get(gk, False)]
        failed_hard_gk_short = [gk for gk in hard_gatekeepers if not results_short.get(gk, False)]
        
        # --- Passed/Failed Soft GKs (for reporting only) ---
        passed_soft_gk_long = [gk for gk in soft_gatekeepers if results_long.get(gk, False)]
        failed_soft_gk_long = [gk for gk in soft_gatekeepers if not results_long.get(gk, False)]
        passed_soft_gk_short = [gk for gk in soft_gatekeepers if results_short.get(gk, False)]
        failed_soft_gk_short = [gk for gk in soft_gatekeepers if not results_short.get(gk, False)]

        # --- Hard GK pass count (ONLY use for signal logic!) ---
        passes_long = len(passed_hard_gk_long) if passed_hard_gk_long is not None else 0
        passes_short = len(passed_hard_gk_short) if passed_hard_gk_short is not None else 0
                
        # --- Passed/failed Non-GK filters and their weights ---
        passed_non_gk_long = [f for f in non_gk_filters if results_long.get(f, False)]
        passed_non_gk_short = [f for f in non_gk_filters if results_short.get(f, False)]
        
        # Calculate total weights for non-GK filters
        total_non_gk_weight_long = sum(self.filter_weights_long.get(f, 0) for f in non_gk_filters)
        total_non_gk_weight_short = sum(self.filter_weights_short.get(f, 0) for f in non_gk_filters)
        passed_non_gk_weight_long = sum(self.filter_weights_long.get(f, 0) for f in passed_non_gk_long)
        passed_non_gk_weight_short = sum(self.filter_weights_short.get(f, 0) for f in passed_non_gk_short)
        
        # --- Print/logging ---
        print(f"[{self.symbol}] Passed GK LONG: {passed_gk_long}")
        print(f"[{self.symbol}] Failed GK LONG: {failed_gk_long}")
        print(f"[{self.symbol}] Passed GK SHORT: {passed_gk_short}")
        print(f"[{self.symbol}] Failed GK SHORT: {failed_gk_short}")
        
        # Extra debug: Only soft GKs failed, all hard GKs passed
        if not failed_hard_gk_long and failed_soft_gk_long:
            print(f"[{self.symbol}] All hard GKs PASSED for LONG, but SOFT GKs FAILED: {failed_soft_gk_long}")
        if not failed_hard_gk_short and failed_soft_gk_short:
            print(f"[{self.symbol}] All hard GKs PASSED for SHORT, but SOFT GKs FAILED: {failed_soft_gk_short}")
                
        # --- Signal logic: Only hard GKs required to pass ---
        if self.required_passed is not None:
            required_passed_long = self.required_passed
            required_passed_short = self.required_passed
        else:
            required_passed_long = len(hard_gatekeepers)
            required_passed_short = len(hard_gatekeepers)
        print(f"[DEBUG] required_passed_long: {required_passed_long}, passes_long: {passes_long}")
        print(f"[DEBUG] required_passed_short: {required_passed_short}, passes_short: {passes_short}")
        
        if required_passed_long is None or passes_long is None:
            print("[ERROR] required_passed_long or passes_long is None!")
            print(f"required_passed_long: {required_passed_long}, passes_long: {passes_long}")
            # recommended: raise or return here!
            raise ValueError("required_passed_long or passes_long is None!")
        if required_passed_short is None or passes_short is None:
            print("[ERROR] required_passed_short or passes_short is None!")
            print(f"required_passed_short: {required_passed_short}, passes_short: {passes_short}")
            # recommended: raise or return here!
            raise ValueError("required_passed_short or passes_short is None!")
        
        signal_long_ok = passes_long >= required_passed_long
        signal_short_ok = passes_short >= required_passed_short
        
        # print(f"[DEBUG] passes_long: {passes_long} required_passed_long: {required_passed_long}")
        # print(f"[DEBUG] passes_short: {passes_short} required_passed_short: {required_passed_short}")
        
        if not signal_long_ok:
            print(f"[{self.symbol}] Signal BLOCKED for LONG: Failed hard GKs: {failed_hard_gk_long}")
        if not signal_short_ok:
            print(f"[{self.symbol}] Signal BLOCKED for SHORT: Failed hard GKs: {failed_hard_gk_short}")             
                
        confidence_long = round(100 * passed_non_gk_weight_long / total_non_gk_weight_long, 1) if total_non_gk_weight_long else 0.0
        confidence_short = round(100 * passed_non_gk_weight_short / total_non_gk_weight_short, 1) if total_non_gk_weight_short else 0.0

        # --- Use selected direction's stats ---
        if direction == "LONG":
            score = long_score
            passes = passes_long
            confidence = confidence_long
            passed_weight = passed_non_gk_weight_long
            total_weight = total_non_gk_weight_long
            results = results_long    # <--- ADD THIS LINE
        elif direction == "SHORT":
            score = short_score
            passes = passes_short
            confidence = confidence_short
            passed_weight = passed_non_gk_weight_short
            total_weight = total_non_gk_weight_short
            results = results_short   # <--- ADD THIS LINE
        else:
            score = max(long_score, short_score)
            passes = max(passes_long, passes_short)
            confidence = max(confidence_long, confidence_short)
            passed_weight = max(passed_non_gk_weight_long, passed_non_gk_weight_short)
            total_weight = max(total_non_gk_weight_long, total_non_gk_weight_short)
            results = results_long  # or results_short; but results_long is fine as fallback

        # Now, safely use `results` below!
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

        # --- SuperGK check (unchanged) ---
        orderbook_result = get_order_wall_delta(self.symbol)
        density_result = get_resting_density(self.symbol)
        super_gk_ok = self.superGK_check(direction, orderbook_result, density_result)

        print("[DEBUG] direction:", direction)
        print("[DEBUG] score:", score, "min_score:", self.min_score)
        print("[DEBUG] passes:", passes, "required_passed:", self.required_passed)
        print("[DEBUG] super_gk_ok:", super_gk_ok)

        # Always use the correctly-calculated required_passed for the current direction
        if direction == "LONG":
            required_for_signal = required_passed_long
        elif direction == "SHORT":
            required_for_signal = required_passed_short
        else:
            required_for_signal = max(required_passed_long, required_passed_short)
        
        if required_for_signal is None:
            print(f"[{self.symbol}] [ERROR] required_for_signal is None in final signal logic!")
            required_for_signal = 0  # or raise an error
        
        valid_signal = (
            direction in ["LONG", "SHORT"]
            and score >= self.min_score
            and passes >= required_for_signal
            and super_gk_ok
        )

        price = self.df['close'].iat[-1] if valid_signal else None
        price_str = f"{price:.6f}" if price is not None else "N/A"

        # Always determine the route, regardless of signal validity
        route, reversal_side = self.explicit_route_gate()  # Ensure this function returns "REVERSAL", "TREND CONTINUATION", "NONE", or "?"

        # For display purposes, remap ambiguous or empty routes
        display_route = route
        if route in ["?", "NONE", None]:
            display_route = "NO ROUTE"
    
        print(f"[DEBUG] reversal_route: {reversal_route}, reversal_side: {reversal_side}, route: {route}, direction: {direction}, valid_signal: {valid_signal}")

        signal_type = direction if valid_signal else None
        score_max = len(non_gk_filters)

        message = (
            f"{direction or 'NO-SIGNAL'} on {self.symbol} @ {price_str} "
            f"| Score: {score}/{score_max} | Passed GK: {passes}/{len(self.gatekeepers)} "
            f"| Confidence: {confidence}% (Weighted: {passed_weight:.1f}/{total_weight:.1f})"
            f" | Route: {display_route if valid_signal else 'N/A'}"
        )

        if valid_signal:
            print(f"[{self.symbol}] ✅ FINAL SIGNAL: {message}")
        else:
            print(f"[{self.symbol}] ❌ No signal.")
        print("DEBUG SUMS:", getattr(self, '_debug_sums', {}))

        # --- Verdict for debug file (unchanged) ---
        verdict = {
            "orderbook": (
                direction == "SHORT" and orderbook_result["sell_wall"] > orderbook_result["buy_wall"] or
                direction == "LONG" and orderbook_result["buy_wall"] > orderbook_result["sell_wall"]
            ),
            "density": (
                direction == "SHORT" and density_result["ask_density"] > density_result["bid_density"] or
                direction == "LONG" and density_result["bid_density"] > density_result["ask_density"]
            ),
            "final": valid_signal
        }
        
        # Add here:
        regime = self._market_regime()

        export_signal_debug_txt(
            symbol=self.symbol,
            tf=self.tf,
            bias=direction,
            filter_weights_long=self.filter_weights_long,
            filter_weights_short=self.filter_weights_short,
            gatekeepers=self.gatekeepers,
            results_long=results_long,
            results_short=results_short,
            orderbook_result=orderbook_result,
            density_result=density_result
        )

        # Return summary object for main.py
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
            "valid_signal": valid_signal,
            "signal_type": signal_type,
            "Route": route,
            "regime": regime,  # <-- Add this line
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

    def _check_unified_trend(self, min_conditions=6, debug=False):
        # EMA Cloud logic
        ema20 = self.df['ema20'].iat[-1]
        ema50 = self.df['ema50'].iat[-1]
        ema20_prev = self.df['ema20'].iat[-2]
        close = self.df['close'].iat[-1]
        ema_cloud = ema20 - ema50
    
        # EMA Structure logic
        ema9 = self.df['ema9'].iat[-1]
        ema21 = self.df['ema21'].iat[-1]
        ema50_s = self.df['ema50'].iat[-1]
        ema9_prev = self.df['ema9'].iat[-2]
        ema21_prev = self.df['ema21'].iat[-2]
        ema50_prev = self.df['ema50'].iat[-2]
    
        # HATS logic
        fast = self.df['ema10'].iat[-1]
        mid = self.df['ema21'].iat[-1]
        slow = self.df['ema50'].iat[-1]
        fast_prev = self.df['ema10'].iat[-2]
        mid_prev = self.df['ema21'].iat[-2]
        slow_prev = self.df['ema50'].iat[-2]
    
        # Trend Continuation logic
        ema6 = self.df['ema6'].iat[-1] if 'ema6' in self.df.columns else None
        ema13 = self.df['ema13'].iat[-1] if 'ema13' in self.df.columns else None
        macd = self.df['macd'].iat[-1] if 'macd' in self.df.columns else None
        rsi = self.df['RSI'].iat[-1] if 'RSI' in self.df.columns else None
        adx = self.df['adx'].iat[-1] if 'adx' in self.df.columns else None
    
        # LONG conditions (collect all distinctive features)
        conds_long = [
            ema20 > ema50,                       # EMA Cloud spread bullish
            ema20 > ema20_prev,                  # EMA Cloud momentum
            close > ema20,                       # Price above EMA Cloud top
            ema9 > ema21 and ema21 > ema50_s,    # EMA Structure stacking
            close > ema9 and close > ema21 and close > ema50_s, # Price above all EMAs
            ema9 > ema9_prev and ema21 > ema21_prev and ema50_s > ema50_prev, # EMA Structure momentum
            fast > mid and mid > slow,           # HATS stacking
            fast > fast_prev and mid > mid_prev and slow > slow_prev, # HATS momentum
            close > fast,                        # Price above fast EMA (HATS)
            ema6 is not None and ema13 is not None and ema6 > ema13, # Trend continuation EMA
            macd is not None and macd > 0,       # MACD bullish
            rsi is not None and rsi > 50,        # RSI bullish
            adx is None or adx > 20              # ADX confirms trend if present
        ]
        long_met = sum(conds_long)
    
        # SHORT conditions (mirror all features)
        conds_short = [
            ema20 < ema50,
            ema20 < ema20_prev,
            close < ema20,
            ema9 < ema21 and ema21 < ema50_s,
            close < ema9 and close < ema21 and close < ema50_s,
            ema9 < ema9_prev and ema21 < ema21_prev and ema50_s < ema50_prev,
            fast < mid and mid < slow,
            fast < fast_prev and mid < mid_prev and slow < slow_prev,
            close < fast,
            ema6 is not None and ema13 is not None and ema6 < ema13,
            macd is not None and macd < 0,
            rsi is not None and rsi < 50,
            adx is None or adx > 20
        ]
        short_met = sum(conds_short)
    
        if debug:
            print(f"[{self.symbol}] [TREND] long_met={long_met}, short_met={short_met}, min_conditions={min_conditions}, ema_cloud={ema_cloud}")
    
        # Decision logic: at least min_conditions and dominant direction
        if long_met >= min_conditions and long_met > short_met:
            print(f"[{self.symbol}] [TREND] Signal: LONG | long_met={long_met}, short_met={short_met}")
            return "LONG"
        elif short_met >= min_conditions and short_met > long_met:
            print(f"[{self.symbol}] [TREND] Signal: SHORT | long_met={long_met}, short_met={short_met}")
            return "SHORT"
        else:
            print(f"[{self.symbol}] [TREND] No signal fired | long_met={long_met}, short_met={short_met}")
            return None

    def _check_macd(self, fast=12, slow=26, signal=9, min_conditions=2, debug=False):
        """
        Composite MACD filter: MACD, Signal line, Histogram, Price action, MACD Cross, MACD Divergence.
        Returns 'LONG', 'SHORT', or None.
        """
        if len(self.df) < slow + 3:
            if debug:
                print(f"[{self.symbol}] [MACD] Not enough data for slow EMA={slow}")
            return None
    
        efast = self.df['close'].ewm(span=fast, adjust=False).mean()
        eslow = self.df['close'].ewm(span=slow, adjust=False).mean()
        macd = efast - eslow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - signal_line
    
        close = self.df['close'].iloc[-1]
        close_prev = self.df['close'].iloc[-2]
    
        # MACD Cross Detection
        cross_up = macd.iloc[-2] < signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1]
        cross_down = macd.iloc[-2] > signal_line.iloc[-2] and macd.iloc[-1] < signal_line.iloc[-1]
    
        # MACD Divergence Detection
        price_delta = close - close_prev
        macd_delta = macd.iloc[-1] - macd.iloc[-2]
        # True if price and MACD move in opposite directions
        divergence = price_delta * macd_delta < 0
    
        # LONG conditions
        cond1_long = macd.iloc[-1] > signal_line.iloc[-1]                    # MACD above signal
        cond2_long = macd.iloc[-1] > macd.iloc[-2]                           # MACD rising
        cond3_long = macd_hist.iloc[-1] > 0 and macd_hist.iloc[-1] > macd_hist.iloc[-2]
        cond4_long = close > close_prev
        cond5_long = cross_up
        cond6_long = divergence and macd_delta > 0                           # Bullish divergence
    
        # SHORT conditions
        cond1_short = macd.iloc[-1] < signal_line.iloc[-1]                   # MACD below signal
        cond2_short = macd.iloc[-1] < macd.iloc[-2]                          # MACD falling
        cond3_short = macd_hist.iloc[-1] < 0 and macd_hist.iloc[-1] < macd_hist.iloc[-2]
        cond4_short = close < close_prev
        cond5_short = cross_down
        cond6_short = divergence and macd_delta < 0                          # Bearish divergence
    
        long_conditions_met = sum([cond1_long, cond2_long, cond3_long, cond4_long, cond5_long, cond6_long])
        short_conditions_met = sum([cond1_short, cond2_short, cond3_short, cond4_short, cond5_short, cond6_short])
    
        if debug:
            print(f"[{self.symbol}] [MACD] macd={macd.iloc[-1]:.6f}, signal={signal_line.iloc[-1]:.6f}, "
                  f"macd_hist={macd_hist.iloc[-1]:.6f}, close={close}, close_prev={close_prev}, "
                  f"cross_up={cross_up}, cross_down={cross_down}, divergence={divergence}, "
                  f"long_met={long_conditions_met}, short_met={short_conditions_met}")
    
        if long_conditions_met >= min_conditions and long_conditions_met > short_conditions_met:
            print(f"[{self.symbol}] [MACD] Signal: LONG | long_met={long_conditions_met}, short_met={short_conditions_met}")
            return "LONG"
        elif short_conditions_met >= min_conditions and short_conditions_met > long_conditions_met:
            print(f"[{self.symbol}] [MACD] Signal: SHORT | long_met={long_conditions_met}, short_met={short_conditions_met}")
            return "SHORT"
        else:
            return None
  
    def _check_momentum(self, window=10, min_conditions=2, threshold=1e-6,
                        rsi_period=14, rsi_overbought=80, rsi_oversold=20,
                        cci_period=20, stochrsi_period=14, willr_period=14, debug=False):
        """
        Composite momentum filter: ROC, acceleration, price action, RSI, CCI, StochRSI, Williams %R.
        Returns 'LONG', 'SHORT', or None.
        """
        required_len = max(window + 2, rsi_period + 2, cci_period + 2, stochrsi_period + 2, willr_period + 2)
        if len(self.df) < required_len:
            if debug:
                print(f"[{self.symbol}] [Momentum] Not enough data for required indicators.")
            return None
    
        roc = self.df['close'].pct_change(periods=window)
        momentum = roc.iloc[-1]
        momentum_prev = roc.iloc[-2]
        acceleration = momentum - momentum_prev
        close = self.df['close'].iloc[-1]
        close_prev = self.df['close'].iloc[-2]
    
        rsi = compute_rsi(self.df, rsi_period)
        rsi_latest = rsi.iloc[-1]
    
        cci = calculate_cci(self.df, cci_period)
        cci_latest = cci.iloc[-1]
    
        stochrsi_k, stochrsi_d = calculate_stochrsi(self.df, rsi_period, stochrsi_period)
        stochrsi_latest = stochrsi_k.iloc[-1]
    
        willr = compute_williams_r(self.df, willr_period)
        willr_latest = willr.iloc[-1]
    
        # LONG conditions
        cond1_long = momentum > threshold
        cond2_long = acceleration > 0
        cond3_long = close > close_prev
        cond4_long = rsi_latest < rsi_oversold
        cond5_long = cci_latest < -100
        cond6_long = stochrsi_latest < 0.2
        cond7_long = willr_latest < -80
    
        # SHORT conditions
        cond1_short = momentum < -threshold
        cond2_short = acceleration < 0
        cond3_short = close < close_prev
        cond4_short = rsi_latest > rsi_overbought
        cond5_short = cci_latest > 100
        cond6_short = stochrsi_latest > 0.8
        cond7_short = willr_latest > -20
    
        long_met = sum([cond1_long, cond2_long, cond3_long, cond4_long, cond5_long, cond6_long, cond7_long])
        short_met = sum([cond1_short, cond2_short, cond3_short, cond4_short, cond5_short, cond6_short, cond7_short])
    
        if debug:
            print(f"[{self.symbol}] [Momentum] values: "
                  f"momentum={momentum:.6f}, acceleration={acceleration:.6f}, close={close}, close_prev={close_prev}, "
                  f"rsi={rsi_latest:.2f}, cci={cci_latest:.2f}, stochrsi={stochrsi_latest:.2f}, willr={willr_latest:.2f}, "
                  f"long_met={long_met}, short_met={short_met}")
    
        if long_met >= min_conditions and long_met > short_met:
            print(f"[{self.symbol}] [Momentum] Signal: LONG | long_met={long_met}, short_met={short_met}")
            return "LONG"
        elif short_met >= min_conditions and short_met > long_met:
            print(f"[{self.symbol}] [Momentum] Signal: SHORT | long_met={long_met}, short_met={short_met}")
            return "SHORT"
        else:
            return None
    
    # ==== FILTERS ====    
    def _check_volume_spike_detector(
        self,
        rolling_window: int = 15,
        min_price_move: float = 0.0001,
        zscore_threshold: float = 1.1,
        require_5m_trend: bool = True,
        debug: bool = False
    ) -> str | None:
        """
        Detects volume spikes with price move and optional 5m trend confirmation.
        Returns "LONG", "SHORT", or None.
        """
        # --- Parameter overrides by timeframe ---
        if self.tf != "3min":
            rolling_window = 5
            require_5m_trend = False
    
        # Defensive: Check rolling window length
        if len(self.df) < rolling_window + 2:
            if debug:
                print(f"[{self.symbol}] [Volume Spike] Not enough data for rolling window")
            return None
    
        avg = self.df['volume'].rolling(rolling_window).mean().iat[-2]
        std = self.df['volume'].rolling(rolling_window).std().iat[-2]
        curr_vol = self.df['volume'].iat[-1]
        zscore = (curr_vol - avg) / (std if std != 0 else 1)
    
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        price_move = (close - close_prev) / close_prev if close_prev != 0 else 0
    
        price_up = price_move > min_price_move
        price_down = price_move < -min_price_move
        vol_up = curr_vol > self.df['volume'].iat[-2]
        spike = zscore > zscore_threshold
    
        long_conditions = [spike, price_up, vol_up]
        short_conditions = [spike, price_down, vol_up]
    
        signal = None
        if sum(long_conditions) >= 2:
            signal = "LONG"
        elif sum(short_conditions) >= 2:
            signal = "SHORT"
    
        if debug:
            # cleaned-up debug print (previous version had a truncated/unterminated f-string)
            print(
                f"[{self.symbol}] [Volume Spike] signal={signal} | "
                f"zscore={zscore:.6f}, price_move={price_move:.6f}, vol_up={vol_up}, "
                f"spike={spike}, price_up={price_up}, price_down={price_down}, "
                f"curr_vol={curr_vol}, avg_vol={avg}, std={std}"
            )
    
        # 5m volume trend confirmation
        if signal and require_5m_trend:
            df5m = getattr(self, "df5m", None)
            if df5m is None or len(df5m) < 2:
                if debug:
                    print(f"[{self.symbol}] [Volume Spike] Not enough 5m data for trend check.")
                return None
            vol_trend = df5m['volume'].iat[-1] > df5m['volume'].iat[-2]
            if debug:
                print(f"[{self.symbol}] [Volume Spike] 5m volume trend check: {vol_trend} (latest={df5m['volume'].iat[-1]}, prev={df5m['volume'].iat[-2]})")
            if not vol_trend:
                return None
    
        return signal

    def _check_mtf_volume_agreement(self, debug: bool = False) -> str | None:
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
        
    def _check_liquidity_awareness(self, debug: bool = False) -> str | None:
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

    def _check_spread_filter(self, window: int = 20, debug: bool = False) -> str | None:
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

    def _check_smart_money_bias(self, volume_window: int = 20, min_cond: int = 1, debug: bool = False) -> str | None:
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

    def _check_absorption(self, window: int = 20, buffer_pct: float = 0.005, min_cond: int = 1, debug: bool = False) -> str | None:
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
    ) -> str | None:
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

    def _check_fractal_zone(self, buffer_pct=0.005, window=20, min_conditions=2, debug=False):
        fractal_low, _ = self._get_rolling_extremes('low', window)
        _, fractal_high = self._get_rolling_extremes('high', window)
        fractal_low_prev, _ = self._get_rolling_extremes('low', window, prev=True)
        _, fractal_high_prev = self._get_rolling_extremes('high', window, prev=True)
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
    
        cond1_long = close > fractal_low * (1 + buffer_pct)
        cond2_long = close > close_prev
        cond3_long = fractal_low > fractal_low_prev
    
        cond1_short = close < fractal_high * (1 - buffer_pct)
        cond2_short = close < close_prev
        cond3_short = fractal_high < fractal_high_prev
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        if short_met >= min_conditions and short_met > long_met:
            if debug:
                print(
                    f"[{self.symbol}] [Fractal Zone] Signal: SHORT | "
                    f"short_met={short_met}, long_met={long_met}, min_conditions={min_conditions}, "
                    f"fractal_high={fractal_high:.2f}, fractal_high_prev={fractal_high_prev:.2f}"
                )
            return "SHORT"
        elif long_met >= min_conditions and long_met > short_met:
            if debug:
                print(
                    f"[{self.symbol}] [Fractal Zone] Signal: LONG | "
                    f"long_met={long_met}, short_met={short_met}, min_conditions={min_conditions}, "
                    f"fractal_low={fractal_low:.2f}, fractal_low_prev={fractal_low_prev:.2f}"
                )
            return "LONG"
        else:
            if debug:
                print(
                    f"[{self.symbol}] [Fractal Zone] No signal fired | "
                    f"short_met={short_met}, long_met={long_met}, min_conditions={min_conditions}, "
                    f"fractal_high={fractal_high:.2f}, fractal_high_prev={fractal_high_prev:.2f}"
                )
            return None

    def _check_hh_ll(self, debug=False):
        print(f"[{self.symbol}] [HH/LL Trend] Function called")
        high = self.df['high'].iat[-1]
        high_prev = self.df['high'].iat[-2]
        low = self.df['low'].iat[-1]
        low_prev = self.df['low'].iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
    
        print(f"[{self.symbol}] [HH/LL Trend] Values | high={high}, high_prev={high_prev}, low={low}, low_prev={low_prev}, close={close}, close_prev={close_prev}")
    
        hh = high
        ll = low
    
        cond1_long = high > high_prev
        cond2_long = low > low_prev
        cond3_long = close > close_prev
    
        cond1_short = low < low_prev
        cond2_short = high < high_prev
        cond3_short = close < close_prev
    
        print(
            f"[{self.symbol}] [HH/LL Trend] Conditions | "
            f"cond1_long={cond1_long}, cond2_long={cond2_long}, cond3_long={cond3_long}, "
            f"cond1_short={cond1_short}, cond2_short={cond2_short}, cond3_short={cond3_short}"
        )
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        print(f"[{self.symbol}] [HH/LL Trend] Met Counts | long_met={long_met}, short_met={short_met}")
    
        if long_met >= 2 and long_met > short_met:
            print(f"[{self.symbol}] [HH/LL Trend] Signal: LONG | long_met={long_met}, short_met={short_met}, hh={hh}, ll={ll}")
            return "LONG"
        elif short_met >= 2 and short_met > long_met:
            print(f"[{self.symbol}] [HH/LL Trend] Signal: SHORT | long_met={long_met}, short_met={short_met}, hh={hh}, ll={ll}")
            return "SHORT"
        else:
            print(f"[{self.symbol}] [HH/LL Trend] No signal fired | long_met={long_met}, short_met={short_met}, hh={hh}, ll={ll}")
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
        threshold_pct=0.10,
        volume_mult=1.1,
        min_cond=2,
        min_atr=0.5,
        debug=False
    ):
        """
        Enhanced ATR Momentum Burst:
        - Requires minimum ATR for signal quality.
        - Logs all relevant metrics.
        - Keeps logic independent and robust.
        """
        long_met = 0
        short_met = 0
        momentum = []
        atr = []
        volumes = []
        avg_vols = []
        atr_vals = []
    
        for i in [-1, -2]:
            close = self.df['close'].iat[i]
            open_ = self.df['open'].iat[i]
            volume = self.df['volume'].iat[i]
            avg_vol = self.df['volume'].rolling(10).mean().iat[i]
            pct_move = (close - open_) / open_ * 100
            atr_val = self.df['atr'].iat[i] if 'atr' in self.df.columns else None
    
            momentum.append(pct_move)
            atr.append(atr_val)
            volumes.append(volume)
            avg_vols.append(avg_vol)
            atr_vals.append(atr_val)
    
            # Enhancement: Only consider bar if ATR is sufficient
            if atr_val is not None and atr_val < min_atr:
                continue
    
            if pct_move > threshold_pct and volume > avg_vol * volume_mult:
                long_met += 1
            elif pct_move < -threshold_pct and volume > avg_vol * volume_mult:
                short_met += 1
    
        if debug:
            print(
                f"[{self.symbol}] [ATR Momentum Burst] Metrics | momentum={momentum}, atr={atr}, volumes={volumes}, avg_vols={avg_vols}, min_atr={min_atr}"
            )
    
        if long_met >= min_cond and long_met > short_met:
            print(
                f"[{self.symbol}] [ATR Momentum Burst] Signal: LONG | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, atr={atr}, momentum={momentum}"
            )
            return "LONG"
        elif short_met >= min_cond and short_met > long_met:
            print(
                f"[{self.symbol}] [ATR Momentum Burst] Signal: SHORT | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, atr={atr}, momentum={momentum}"
            )
            return "SHORT"
        else:
            print(
                f"[{self.symbol}] [ATR Momentum Burst] No signal fired | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, atr={atr}, momentum={momentum}"
            )
            return None


    def _check_volatility_model(
        self,
        atr_col='atr',
        atr_ma_col='atr_ma',
        min_atr_diff=0.1,
        volume_confirm=False,
        debug=False
    ):
        """
        Enhanced Volatility Model:
        - Configurable minimum ATR difference versus average.
        - Optional volume confirmation.
        - Detailed logging.
        """
        atr = self.df[atr_col].iat[-1]
        atr_prev = self.df[atr_col].iat[-2]
        atr_ma = self.df[atr_ma_col].iat[-1]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_ma = self.df['volume'].rolling(10).mean().iat[-1]
    
        volatility = {
            "atr": atr,
            "atr_prev": atr_prev,
            "atr_ma": atr_ma,
            "close": close,
            "close_prev": close_prev,
            "atr_vs_ma": atr - atr_ma,
            "atr_vs_prev": atr - atr_prev,
            "volume": volume,
            "volume_ma": volume_ma,
            "min_atr_diff": min_atr_diff,
        }
    
        # Enhancement: ATR must exceed MA by min_atr_diff
        cond1_long = atr > atr_prev
        cond2_long = close > close_prev
        cond3_long = (atr - atr_ma) > min_atr_diff
        cond4_long = (volume > volume_ma) if volume_confirm else True
    
        cond1_short = atr < atr_prev
        cond2_short = close < close_prev
        cond3_short = (atr_ma - atr) > min_atr_diff
        cond4_short = (volume > volume_ma) if volume_confirm else True
    
        long_met = sum([cond1_long, cond2_long, cond3_long, cond4_long])
        short_met = sum([cond1_short, cond2_short, cond3_short, cond4_short])
    
        if debug:
            print(
                f"[{self.symbol}] [Volatility Model] Metrics | volatility={volatility}, long_met={long_met}, short_met={short_met}"
            )
    
        if long_met >= 3:
            print(
                f"[{self.symbol}] [Volatility Model] Signal: LONG | long_met={long_met}, short_met={short_met}, volatility={volatility}"
            )
            return "LONG"
        elif short_met >= 3:
            print(
                f"[{self.symbol}] [Volatility Model] Signal: SHORT | long_met={long_met}, short_met={short_met}, volatility={volatility}"
            )
            return "SHORT"
        else:
            print(
                f"[{self.symbol}] [Volatility Model] No signal fired | long_met={long_met}, short_met={short_met}, volatility={volatility}"
            )
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
