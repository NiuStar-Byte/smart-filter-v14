# smart_filter.py

import datetime
import requests
import pandas as pd
import numpy as np
import logging
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
        min_score: int = 7,
        required_passed: Optional[int] = None,  # int or None allowed
        volume_multiplier: float = 2.0,
        liquidity_threshold: float = 0.50,
        kwargs: Optional[dict] = None
    ):
        if kwargs is None:
            kwargs = {}

        self.symbol = symbol
        self.df = add_indicators(df)
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
            "MACD": 4.9, "Volume Spike": 4.7, "Fractal Zone": 4.6, "TREND": 4.5, "Momentum": 4.8, "ATR Momentum Burst": 4.4,
            "MTF Volume Agreement": 4.3, "HH/LL Trend": 3.9, "Volatility Model": 3.8,
            "Liquidity Awareness": 3.6, "Volatility Squeeze": 3.5, "Candle Confirmation": 3.4,
            "VWAP Divergence": 3.3, "Spread Filter": 3.2, "Chop Zone": 3.1, "Liquidity Pool": 2.9, "Support/Resistance": 2.8,
            "Smart Money Bias": 2.7, "Absorption": 2.6, "Wick Dominance": 2.5
        }
        
        self.filter_weights_short = {
            "MACD": 4.9, "Volume Spike": 4.7, "Fractal Zone": 4.6, "TREND": 4.5, "Momentum": 4.8, "ATR Momentum Burst": 4.4,
            "MTF Volume Agreement": 4.3, "HH/LL Trend": 3.9, "Volatility Model": 3.8,
            "Liquidity Awareness": 3.6, "Volatility Squeeze": 3.5, "Candle Confirmation": 3.4,
            "VWAP Divergence": 3.3, "Spread Filter": 3.2, "Chop Zone": 3.1, "Liquidity Pool": 2.9, "Support/Resistance": 2.8,
            "Smart Money Bias": 2.7, "Absorption": 2.6, "Wick Dominance": 2.5
        }

        self.filter_names = list(set(self.filter_weights_long.keys()) | set(self.filter_weights_short.keys()))
        
        self.gatekeepers = [
            "MACD",
            "Volume Spike",
            "MTF Volume Agreement",
            "Liquidity Awareness",
            "Spread Filter",
            "Candle Confirmation",
            "Support/Resistance"
        ]

        self.soft_gatekeepers = ["Volume Spike"]

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

    def detect_rsi_reversal(self, threshold_overbought=65, threshold_oversold=35):
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

    def detect_adx_reversal(self, adx_threshold=25):
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
        
    def detect_cci_reversal(self, overbought=100, oversold=-100):
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
        signals = [
            self.detect_ema_reversal(),
            self.detect_rsi_reversal(),
            self.detect_engulfing_reversal(),
            self.detect_adx_reversal(),
            self.detect_stochrsi_reversal(),
            self.detect_cci_reversal(),
        ]
        print("Reversal detector results:", signals)
        bullish = signals.count("BULLISH_REVERSAL")
        bearish = signals.count("BEARISH_REVERSAL")
    
        if bullish > 0 and bearish == 0:
            return ("REVERSAL", "BULLISH")
        elif bearish > 0 and bullish == 0:
            return ("REVERSAL", "BEARISH")
        elif bullish > 0 and bearish > 0:
            # Ambiguous signal: both bullish and bearish detected
            return ("AMBIGUOUS", ["BULLISH", "BEARISH"])
        else:
            return ("NONE", None)
                    
    def detect_trend_continuation(self):
        # Check required columns
        required_cols = ['ema6', 'ema13', 'macd', 'RSI']
        for col in required_cols:
            if col not in self.df.columns:
                print(f"[ERROR] '{col}' column missing in DataFrame!")
                return "NO_CONTINUATION"  # Or handle as needed
    
        # Safe to extract values now
        ema_fast = self.df['ema6'].iat[-1]
        ema_slow = self.df['ema13'].iat[-1]
        macd = self.df['macd'].iat[-1]
        rsi = self.df['RSI'].iat[-1]
        adx = self.df['adx'].iat[-1] if 'adx' in self.df.columns else None
    
        bullish = (
            ema_fast > ema_slow and
            macd > 0 and
            rsi > 50 and
            (adx is None or adx > 20)
        )
        bearish = (
            ema_fast < ema_slow and
            macd < 0 and
            rsi < 50 and
            (adx is None or adx > 20)
        )
        if bullish:
            return "BULLISH_CONTINUATION"
        elif bearish:
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

    def get_signal_direction(self, results_long, results_short):
        """
        Returns direction "LONG", "SHORT", or "NEUTRAL" based on gatekeeper and scoring filter results.

        1. Checks if all gatekeepers are passed for each direction.
        2. Calculates weighted sum of non-gatekeeper filters for each direction.
        3. Returns direction ONLY if gatekeeper rule and scoring filter are both stronger for one side.
        """

        # --- Gatekeeper Pass Check ---

        # --- Gatekeeper Pass Check with soft GK support ---
        soft_gatekeepers = getattr(self, "soft_gatekeepers", ["Volume Spike"])
        hard_gatekeepers = [gk for gk in self.gatekeepers if gk not in soft_gatekeepers]        
        long_gk_passed = all(results_long.get(gk, False) for gk in hard_gatekeepers)
        short_gk_passed = all(results_short.get(gk, False) for gk in hard_gatekeepers)

        # --- previous = all hard GK ---
        # long_gk_passed = all(results_long.get(gk, False) for gk in self.gatekeepers)
        # short_gk_passed = all(results_short.get(gk, False) for gk in self.gatekeepers)

        # --- Weighted Sum for Non-GK Filters ---
        scoring_filters = [f for f in self.filter_names if f not in self.gatekeepers]

        long_score = sum(
            self.filter_weights_long.get(name, 0)
            for name in scoring_filters
            if results_long.get(name, False)
        )
        short_score = sum(
            self.filter_weights_short.get(name, 0)
            for name in scoring_filters
            if results_short.get(name, False)
        )

        self._debug_sums = {
            "long_gk_passed": long_gk_passed,
            "short_gk_passed": short_gk_passed,
            "long_score": long_score,
            "short_score": short_score
        }

        # --- Direction Decision ---
        if long_gk_passed and not short_gk_passed:
            return "LONG"
        elif short_gk_passed and not long_gk_passed:
            return "SHORT"
        elif long_gk_passed and short_gk_passed:
            if long_score > short_score:
                return "LONG"
            elif short_score > long_score:
                return "SHORT"
            else:
                return "NEUTRAL"
        else:
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
            "Volume Spike": lambda debug=False: self.volume_spike_detector(debug=debug),
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

    def _check_macd(self, fast=12, slow=26, signal=9, min_conditions=3, debug=False):
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
  
    def _check_momentum(self, window=10, min_conditions=3, threshold=1e-6,
                        rsi_period=14, rsi_overbought=70, rsi_oversold=30,
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
    def _check_volume_spike_detector(self,
                             rolling_window=15,
                             min_price_move=0.0002,
                             zscore_threshold=1.2,
                             require_5m_trend=True):
        # --- Parameter overrides by timeframe ---
        if self.tf != "3min":
            rolling_window = 5
            require_5m_trend = False

        if len(self.df) < rolling_window + 2:
            if self.debug:
                print(f"[{self.symbol}] [Volume Spike] Not enough data for rolling window")
            return None

        avg = self.get_rolling_avg('volume', rolling_window)
        std = self.df['volume'].rolling(rolling_window).std().iat[-2]
        curr_vol = self.df['volume'].iat[-1]
        zscore = self.safe_divide(curr_vol - avg, std if std != 0 else 1)

        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        price_move = self.safe_divide(close - close_prev, close_prev if close_prev != 0 else 1)

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

        if self.debug:
            print(f"[{self.symbol}] [Volume Spike] signal={signal} | zscore={zscore:.6f}, price_move={price_move:.6f}, vol_up={vol_up}, spike={spike}, price_up={price_up}, price_down={price_down}, long_conditions={long_conditions}, short_conditions={short_conditions}, require_5m_trend={require_5m_trend}")

        # 5m volume trend confirmation
        if signal and require_5m_trend:
            df5m = self.df5m
            if df5m is None or len(df5m) < 2:
                if self.debug:
                    print(f"[{self.symbol}] [Volume Spike] Not enough 5m data for trend check.")
                return None
            vol_trend = df5m['volume'].iat[-1] > df5m['volume'].iat[-2]
            if self.debug:
                print(f"[{self.symbol}] [Volume Spike] 5m volume trend check: {vol_trend} (latest={df5m['volume'].iat[-1]}, prev={df5m['volume'].iat[-2]})")
            if not vol_trend:
                return None

        return signal

    def _check_mtf_volume_agreement(self):
        # Current TF
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        # Higher TF
        higher_tf_volume = self.df['higher_tf_volume'].iat[-1]
        higher_tf_volume_prev = self.df['higher_tf_volume'].iat[-2]

        conds_long = [volume > volume_prev, higher_tf_volume > higher_tf_volume_prev, close > close_prev]
        conds_short = [volume < volume_prev, higher_tf_volume < higher_tf_volume_prev, close < close_prev]

        long_met, short_met = sum(conds_long), sum(conds_short)
        signal = None
        if long_met >= 2 and long_met > short_met:
            signal = "LONG"
        elif short_met >= 2 and short_met > long_met:
            signal = "SHORT"

        if self.debug:
            print(f"[{self.symbol}] [MTF Volume Agreement] signal={signal} | long_met={long_met}, short_met={short_met}, conds_long={conds_long}, conds_short={conds_short}")

        return signal
        
    def _check_liquidity_awareness(self):
        bid, ask = self.df['bid'].iat[-1], self.df['ask'].iat[-1]
        bid_prev, ask_prev = self.df['bid'].iat[-2], self.df['ask'].iat[-2]
        volume, volume_prev = self.df['volume'].iat[-1], self.df['volume'].iat[-2]
        close, close_prev = self.df['close'].iat[-1], self.df['close'].iat[-2]

        spread = ask - bid
        spread_prev = ask_prev - bid_prev

        conds_long = [spread < spread_prev, volume > volume_prev, close > close_prev]
        conds_short = [spread > spread_prev, volume < volume_prev, close < close_prev]

        long_met, short_met = sum(conds_long), sum(conds_short)
        signal = None
        if long_met >= 2 and long_met > short_met:
            signal = "LONG"
        elif short_met >= 2 and short_met > long_met:
            signal = "SHORT"

        if self.debug:
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

    def _check_spread_filter(self, window=20):
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
        if long_met >= 2 and long_met > short_met:
            signal = "LONG"
        elif short_met >= 2 and short_met > long_met:
            signal = "SHORT"

        if self.debug:
            print(f"[{self.symbol}] [Spread Filter] signal={signal} | spread={spread:.6f}, long_met={long_met}, short_met={short_met}, spread_prev={spread_prev:.6f}, spread_ma={spread_ma:.6f}, close={close:.2f}, open={open_:.2f}")

        return signal

    def _check_smart_money_bias(self, volume_window=20, min_cond=2):
        close, close_prev = self.df['close'].iat[-1], self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        avg_volume = self.get_rolling_avg('volume', volume_window)
        vwap = self.df['vwap'].iat[-1]

        conds_long = [volume > avg_volume and close > close_prev, close > vwap]
        conds_short = [volume > avg_volume and close < close_prev, close < vwap]

        long_met, short_met = sum(conds_long), sum(conds_short)
        signal = None
        if long_met >= min_cond and long_met > short_met:
            signal = "LONG"
        elif short_met >= min_cond and short_met > long_met:
            signal = "SHORT"

        if self.debug:
            smart_money = {
                "volume_vs_avg": self.safe_divide(volume, avg_volume),
                "close_vs_prev": close - close_prev,
                "close_vs_vwap": close - vwap
            }
            print(f"[{self.symbol}] [Smart Money Bias] signal={signal} | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, smart_money={smart_money}")

        return signal

    def _check_absorption(self, window=20, buffer_pct=0.005, min_cond=2):
        low = self.df['low'].rolling(window).min().iat[-1]
        high = self.df['high'].rolling(window).max().iat[-1]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
        avg_volume = self.get_rolling_avg('volume', window)

        absorption_metrics = {
            "close_to_low_pct": self.safe_divide(close - low, low),
            "close_to_high_pct": self.safe_divide(high - close, high),
            "volume_vs_avg": self.safe_divide(volume, avg_volume),
            "volume_vs_prev": self.safe_divide(volume, volume_prev)
        }

        conds_long = [close <= low * (1 + buffer_pct),
                      volume > avg_volume and volume > volume_prev,
                      close >= close_prev]
        conds_short = [close >= high * (1 - buffer_pct),
                       volume > avg_volume and volume > volume_prev,
                       close <= close_prev]

        long_met, short_met = sum(conds_long), sum(conds_short)

        signal = None
        if long_met >= min_cond and long_met > short_met:
            signal = "LONG"
        elif short_met >= min_cond and short_met > long_met:
            signal = "SHORT"

        if self.debug:
            print(f"[{self.symbol}] [Absorption] signal={signal} | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, absorption={absorption_metrics}")

        return signal

    def _check_candle_confirmation(self, debug=False):
        open_ = self.df['open'].iat[-1]
        high = self.df['high'].iat[-1]
        low = self.df['low'].iat[-1]
        close = self.df['close'].iat[-1]
    
        open_prev = self.df['open'].iat[-2]
        close_prev = self.df['close'].iat[-2]
    
        # Engulfing
        bullish_engulfing = close > open_prev and open_ < close_prev
        bearish_engulfing = close < open_prev and open_ > close_prev
    
        # Pin bar / Hammer / Shooting Star
        lower_wick = open_ - low if open_ < close else close - low
        upper_wick = high - close if open_ < close else high - open_
        body = abs(close - open_)
    
        bullish_pin_bar = lower_wick > 2 * body and close > open_
        bearish_pin_bar = upper_wick > 2 * body and close < open_
    
        # Determine candle_type for logging
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
    
        # LONG conditions
        cond1_long = bullish_engulfing
        cond2_long = bullish_pin_bar
        cond3_long = close > close_prev
    
        # SHORT conditions
        cond1_short = bearish_engulfing
        cond2_short = bearish_pin_bar
        cond3_short = close < close_prev
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
        
        # Fix: Only return LONG if long_met > short_met, and vice versa
        if long_met >= 2 and long_met > short_met:
            if debug:
                print(f"[{self.symbol}] [Candle Confirmation] Signal: LONG | long_met={long_met}, short_met={short_met}, candle_type={candle_type}, close={close}, open={open_}")
            return "LONG"
        elif short_met >= 2 and short_met > long_met:
            if debug:
                print(f"[{self.symbol}] [Candle Confirmation] Signal: SHORT | long_met={long_met}, short_met={short_met}, candle_type={candle_type}, close={close}, open={open_}")
            return "SHORT"
        else:
            if debug:
                print(f"[{self.symbol}] [Candle Confirmation] No signal fired | long_met={long_met}, short_met={short_met}, candle_type={candle_type}, close={close}, open={open_}")
            return None
            
    def _check_support_resistance(self, window=20, buffer_pct=0.005, min_cond=2, debug=False):
        support = self.df['low'].rolling(window).min().iat[-1]
        resistance = self.df['high'].rolling(window).max().iat[-1]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
    
        # For logging: proximity metrics
        support_prox = (close - support) / support
        resistance_prox = (resistance - close) / resistance
    
        # LONG conditions
        cond1_long = close <= support * (1 + buffer_pct)
        cond2_long = close > close_prev
        cond3_long = volume > volume_prev
    
        # SHORT conditions
        cond1_short = close >= resistance * (1 - buffer_pct)
        cond2_short = close < close_prev
        cond3_short = volume < volume_prev  # Classic reversal logic
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
        
        if long_met >= min_cond and long_met > short_met:
            print(f"[{self.symbol}] [Support/Resistance] Signal: LONG | support={support}, resistance={resistance}, close={close}, long_met={long_met}, short_met={short_met}, support_prox={support_prox:.4f}")
            return "LONG"
        elif short_met >= min_cond and short_met > long_met:
            print(f"[{self.symbol}] [Support/Resistance] Signal: SHORT | support={support}, resistance={resistance}, close={close}, long_met={long_met}, short_met={short_met}, resistance_prox={resistance_prox:.4f}")
            return "SHORT"
        else:
            if debug:
                print(f"[{self.symbol}] [Support/Resistance] No signal fired | support={support}, resistance={resistance}, close={close}, close_prev={close_prev}, volume={volume}, volume_prev={volume_prev}, long_met={long_met}, short_met={short_met}, support_prox={support_prox:.4f}, resistance_prox={resistance_prox:.4f}")
            return None
    
    def _check_fractal_zone(self, buffer_pct=0.005, window=20, min_conditions=2, debug=False):
        # Calculate fractal highs/lows
        fractal_low = self.df['low'].rolling(window).min().iat[-1]
        fractal_low_prev = self.df['low'].rolling(window).min().iat[-2]
        fractal_high = self.df['high'].rolling(window).max().iat[-1]
        fractal_high_prev = self.df['high'].rolling(window).max().iat[-2]
    
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
    
        # LONG: Strong break above recent range with confirmation
        cond1_long = close > fractal_low * (1 + buffer_pct)
        cond2_long = close > close_prev
        cond3_long = fractal_low > fractal_low_prev
    
        # SHORT: Strong break below recent range with confirmation
        cond1_short = close < fractal_high * (1 - buffer_pct)
        cond2_short = close < close_prev
        cond3_short = fractal_high < fractal_high_prev
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        # Only fire signal if one side strictly beats the other and meets min_conditions
        if short_met >= min_conditions and short_met > long_met:
            print(f"[{self.symbol}] [Fractal Zone] Signal: SHORT | short_met={short_met}, long_met={long_met}, min_conditions={min_conditions}, fractal_high={fractal_high:.2f}, fractal_high_prev={fractal_high_prev:.2f}, close={close:.2f}, close_prev={close_prev:.2f}")
            return "SHORT"
        elif long_met >= min_conditions and long_met > short_met:
            print(f"[{self.symbol}] [Fractal Zone] Signal: LONG | short_met={short_met}, long_met={long_met}, min_conditions={min_conditions}, fractal_low={fractal_low:.2f}, fractal_low_prev={fractal_low_prev:.2f}, close={close:.2f}, close_prev={close_prev:.2f}")
            return "LONG"
        else:
            if debug:
                print(f"[{self.symbol}] [Fractal Zone] No signal fired | short_met={short_met}, long_met={long_met}, min_conditions={min_conditions}, fractal_high={fractal_high:.2f}, fractal_high_prev={fractal_high_prev:.2f}, fractal_low={fractal_low:.2f}, fractal_low_prev={fractal_low_prev:.2f}, close={close:.2f}, close_prev={close_prev:.2f}")
            return None

    def _check_atr_momentum_burst(self, threshold_pct=0.10, volume_mult=1.1, min_cond=2, debug=False):
        long_met = 0
        short_met = 0
        momentum = []
        atr = []
        for i in [-1, -2]:
            close = self.df['close'].iat[i]
            open_ = self.df['open'].iat[i]
            volume = self.df['volume'].iat[i]
            avg_vol = self.df['volume'].rolling(10).mean().iat[i]
            pct_move = (close - open_) / open_ * 100
            momentum.append(pct_move)
            atr_val = self.df['atr'].iat[i] if 'atr' in self.df.columns else None
            atr.append(atr_val)
            if pct_move > threshold_pct and volume > avg_vol * volume_mult:
                long_met += 1
            elif pct_move < -threshold_pct and volume > avg_vol * volume_mult:
                short_met += 1
    
        # Only signal if BOTH bars agree (default: min_cond=2)
        if long_met >= min_cond and long_met > short_met:
            if debug:
                print(f"[{self.symbol}] [ATR Momentum Burst] Signal: LONG | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, atr={atr}, momentum={momentum}, close={close}")
            return "LONG"
        elif short_met >= min_cond and short_met > long_met:
            if debug:
                print(f"[{self.symbol}] [ATR Momentum Burst] Signal: SHORT | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, atr={atr}, momentum={momentum}, close={close}")
            return "SHORT"
        else:
            if debug:
                print(f"[{self.symbol}] [ATR Momentum Burst] No signal fired | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, atr={atr}, momentum={momentum}, close={close}")
            return None

    def _check_hh_ll(self, debug=False):
        high = self.df['high'].iat[-1]
        high_prev = self.df['high'].iat[-2]
        low = self.df['low'].iat[-1]
        low_prev = self.df['low'].iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
    
        # Define hh and ll for logging
        hh = high
        ll = low
    
        # LONG conditions: Higher Highs and Higher Lows
        cond1_long = high > high_prev
        cond2_long = low > low_prev
        cond3_long = close > close_prev
    
        # SHORT conditions: Lower Lows and Lower Highs
        cond1_short = low < low_prev
        cond2_short = high < high_prev
        cond3_short = close < close_prev
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        # Fix: Only return LONG if long_met > short_met, and vice versa
        if long_met >= 2 and long_met > short_met:
            print(f"[{self.symbol}] [HH/LL] Signal: LONG | long_met={long_met}, short_met={short_met}, hh={hh}, ll={ll}")
            return "LONG"
        elif short_met >= 2 and short_met > long_met:
            print(f"[{self.symbol}] [HH/LL] Signal: SHORT | long_met={long_met}, short_met={short_met}, hh={hh}, ll={ll}")
            return "SHORT"
        else:
            return None

    def _check_volatility_model(self, atr_col='atr', atr_ma_col='atr_ma', debug=False):
        atr = self.df[atr_col].iat[-1]
        atr_prev = self.df[atr_col].iat[-2]
        atr_ma = self.df[atr_ma_col].iat[-1]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
    
        # For logging: volatility metrics
        volatility = {
            "atr": atr,
            "atr_prev": atr_prev,
            "atr_ma": atr_ma,
            "close": close,
            "close_prev": close_prev,
            "atr_vs_ma": atr - atr_ma,
            "atr_vs_prev": atr - atr_prev
        }
    
        # LONG conditions: volatility expansion + bullish price
        cond1_long = atr > atr_prev
        cond2_long = close > close_prev
        cond3_long = atr > atr_ma
    
        # SHORT conditions: volatility contraction + bearish price
        cond1_short = atr < atr_prev        # ATR falling
        cond2_short = close < close_prev
        cond3_short = atr < atr_ma          # ATR below average
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        if long_met >= 2:
            print(f"[{self.symbol}] [Volatility Model] Signal: LONG | long_met={long_met}, short_met={short_met}, volatility={volatility}")
            return "LONG"
        elif short_met >= 2:
            print(f"[{self.symbol}] [Volatility Model] Signal: SHORT | long_met={long_met}, short_met={short_met}, volatility={volatility}")
            return "SHORT"
        else:
            return None

    def _check_volatility_squeeze(self, min_cond=2, debug=False):
        bb_width = self.df['bb_upper'].iat[-1] - self.df['bb_lower'].iat[-1]
        kc_width = self.df['kc_upper'].iat[-1] - self.df['kc_lower'].iat[-1]
        bb_width_prev = self.df['bb_upper'].iat[-2] - self.df['bb_lower'].iat[-2]
        kc_width_prev = self.df['kc_upper'].iat[-2] - self.df['kc_lower'].iat[-2]
    
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
    
        squeeze_firing = bb_width > kc_width and bb_width_prev < kc_width_prev
    
        squeeze = {
            "bb_width": bb_width,
            "kc_width": kc_width,
            "bb_width_prev": bb_width_prev,
            "kc_width_prev": kc_width_prev,
            "squeeze_firing": squeeze_firing
        }
        volatility = {
            "close": close,
            "close_prev": close_prev,
            "volume": volume,
            "volume_prev": volume_prev
        }
    
        cond1_long = squeeze_firing
        cond2_long = close > close_prev
        cond3_long = volume > volume_prev
    
        cond1_short = squeeze_firing
        cond2_short = close < close_prev
        cond3_short = volume > volume_prev
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        if long_met >= min_cond and long_met > short_met:
            print(f"[{self.symbol}] [Volatility Squeeze] Signal: LONG | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, squeeze={squeeze}, volatility={volatility}")
            return "LONG"
        elif short_met >= min_cond and short_met > long_met:
            print(f"[{self.symbol}] [Volatility Squeeze] Signal: SHORT | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, squeeze={squeeze}, volatility={volatility}")
            return "SHORT"
        else:
            print(f"[{self.symbol}] [Volatility Squeeze] No signal fired | long_met={long_met}, short_met={short_met}, min_cond={min_cond}, squeeze={squeeze}, volatility={volatility}")
            return None
    
    def _check_vwap_divergence(self, debug=False):
        vwap = self.df['vwap'].iat[-1]
        vwap_prev = self.df['vwap'].iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
    
        # Compute VWAP divergence for logging
        vwap_div = (vwap - close) - (vwap_prev - close_prev)
    
        # LONG conditions
        cond1_long = close < vwap
        cond2_long = close > close_prev
        cond3_long = (vwap - close) > (vwap_prev - close_prev)
    
        # SHORT conditions
        cond1_short = close > vwap
        cond2_short = close < close_prev
        cond3_short = (close - vwap) > (close_prev - vwap_prev)
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        if long_met > short_met and long_met > 0:
            print(f"[{self.symbol}] [VWAP Divergence] Signal: LONG | long_met={long_met}, short_met={short_met}, vwap_div={vwap_div}")
            return "LONG"
        elif short_met > long_met and short_met > 0:
            print(f"[{self.symbol}] [VWAP Divergence] Signal: SHORT | long_met={long_met}, short_met={short_met}, vwap_div={vwap_div}")
            return "SHORT"
        elif long_met == short_met and long_met > 0:
            print(f"[{self.symbol}] [VWAP Divergence] Signal: LONG (tie) | long_met={long_met}, short_met={short_met}, vwap_div={vwap_div}")
            return "LONG"  # Change to "SHORT" or None if desired
        else:
            return None

    def _check_chop_zone(self, chop_threshold=40, debug=False):
        """
        Detects and filters out choppy market conditions using the Choppiness Index,
        and then checks for trend signals using EMAs, ADX, and price.
    
        Returns:
            "LONG" if long conditions are met,
            "SHORT" if short conditions are met,
            None if market is too choppy or no signal.
        """
        # Retrieve required columns safely
        try:
            chop = self.df['chop_zone'].iat[-1]
            ema9 = self.df['ema9'].iat[-1]
            ema21 = self.df['ema21'].iat[-1]
            close = self.df['close'].iat[-1]
            adx = self.df['adx'].iat[-1] if 'adx' in self.df.columns else None
        except (KeyError, IndexError):
            print(f"[{self.symbol}] [Chop Zone Check] Missing required columns or insufficient data.")
            return None
    
        # Define for logging
        chop_zone = chop
    
        # Filter out choppy market
        if chop >= chop_threshold:
            print(f"[{self.symbol}] [Chop Zone Check] Market too choppy (chop_zone={chop_zone:.2f} >= {chop_threshold})")
            return None
    
        # LONG conditions
        cond1_long = ema9 > ema21
        cond2_long = adx is not None and adx > 20 if adx is not None else True  # ADX filter optional
        cond3_long = close > ema9
    
        # SHORT conditions
        cond1_short = ema9 < ema21
        cond2_short = adx is not None and adx > 20 if adx is not None else True
        cond3_short = close < ema9
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        # Only return LONG if long_met > short_met, and vice versa
        if long_met >= 2 and long_met > short_met:
            print(f"[{self.symbol}] [Chop Zone Check] Signal: LONG | long_met={long_met}, short_met={short_met}, chop_zone={chop_zone}")
            return "LONG"
        elif short_met >= 2 and short_met > long_met:
            print(f"[{self.symbol}] [Chop Zone Check] Signal: SHORT | long_met={long_met}, short_met={short_met}, chop_zone={chop_zone}")
            return "SHORT"
        else:
            print(f"[{self.symbol}] [Chop Zone Check] No signal fired | long_met={long_met}, short_met={short_met}, chop_zone={chop_zone}")
            return None

    def _check_liquidity_pool(self, lookback=20, min_cond=2, debug=False):
        close = self.df['close'].iat[-1]
        high = self.df['high'].iat[-1]
        low = self.df['low'].iat[-1]
    
        # Identify recent liquidity pools
        recent_high = self.df['high'].rolling(lookback).max().iat[-2]
        recent_low = self.df['low'].rolling(lookback).min().iat[-2]
    
        # For logging: liquidity pool metrics
        liquidity_pool = {
            "recent_high": recent_high,
            "recent_low": recent_low,
            "close_vs_high": close - recent_high,
            "close_vs_low": close - recent_low
        }
    
        # LONG: break or sweep above recent high
        cond1_long = close > recent_high
        cond2_long = low < recent_low and close > recent_low
    
        # SHORT: break or sweep below recent low
        cond1_short = close < recent_low
        cond2_short = high > recent_high and close < recent_high
    
        long_met = sum([cond1_long, cond2_long])
        short_met = sum([cond1_short, cond2_short])
    
        # Only return if enough conditions met (default: both required)
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
            
    def _check_wick_dominance(self, debug=False):
        open_ = self.df['open'].iat[-1]
        high = self.df['high'].iat[-1]
        low = self.df['low'].iat[-1]
        close = self.df['close'].iat[-1]
    
        # Calculate candle components
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        body = abs(close - open_)
    
        # Determine wick dominance for logging
        if lower_wick > upper_wick:
            wick_dom = f"Lower wick dominant ({lower_wick:.2f} > {upper_wick:.2f})"
        elif upper_wick > lower_wick:
            wick_dom = f"Upper wick dominant ({upper_wick:.2f} > {lower_wick:.2f})"
        else:
            wick_dom = f"No dominance (upper={upper_wick:.2f}, lower={lower_wick:.2f})"
    
        # LONG conditions: dominant lower wick + bullish close
        cond1_long = lower_wick > 2 * upper_wick
        cond2_long = lower_wick > 1.5 * body
        cond3_long = close > open_
    
        # SHORT conditions: dominant upper wick + bearish close
        cond1_short = upper_wick > 2 * lower_wick
        cond2_short = upper_wick > 1.5 * body
        cond3_short = close < open_
    
        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])
    
        if long_met >= 2:
            if debug:
                print(f"[{self.symbol}] [Wick Dominance] Signal: LONG | long_met={long_met}, short_met={short_met}, wick_dom={wick_dom}")
            return "LONG"
        elif short_met >= 2:
            if debug:
                print(f"[{self.symbol}] [Wick Dominance] Signal: SHORT | long_met={long_met}, short_met={short_met}, wick_dom={wick_dom}")
            return "SHORT"
        else:
            if debug:
                print(f"[{self.symbol}] [Wick Dominance] No signal fired | long_met={long_met}, short_met={short_met}, wick_dom={wick_dom}")
            return None

    def _market_regime(self, ma_col='ema200', adx_col='adx', adx_threshold=20):
        """
        Returns 'BULL' if market is in uptrend, 'BEAR' if downtrend, None if ranging.
        """
        close = self.df['close'].iat[-1]
        ma = self.df[ma_col].iat[-1]
        ma_prev = self.df[ma_col].iat[-2]
        adx = self.df[adx_col].iat[-1] if adx_col in self.df.columns else None

        # Optional: require ADX above threshold to confirm trend
        trending = adx is None or adx > adx_threshold

        if close > ma and ma > ma_prev and trending:
            return "BULL"
        elif close < ma and ma < ma_prev and trending:
            return "BEAR"
        else:
            return None
    
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

