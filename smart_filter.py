# smart_filter.py

import requests
import pandas as pd
import numpy as np
from kucoin_orderbook import get_order_wall_delta
from kucoin_density import get_resting_density
import datetime
from signal_debug_log import export_signal_debug_txt

def compute_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iat[-1]

# PREVIOUS COMPUTE RSI for New SuperGK ONLY RSI Density & OrderBookWall
# def compute_rsi(df, period=14):
#    delta = df['close'].diff()
#    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
#    rs = gain / loss
#    rsi = 100 - (100 / (1 + rs))
#    return rsi

# When you get your OHLCV dataframe:
# df['RSI'] = compute_rsi(df)

# PREVIOUS COMPUTE RSI for New SuperGK
# def compute_rsi(df, period=14):
#    delta = df['close'].diff()
#    up = delta.clip(lower=0)
#    down = -1 * delta.clip(upper=0)
#    ema_up = up.ewm(com=period-1, adjust=False).mean()
#    ema_down = down.ewm(com=period-1, adjust=False).mean()
#    rs = ema_up / ema_down
#    return 100 - (100 / (1 + rs)).iat[-1]

def compute_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx.iat[-1]

def add_bollinger_bands(df, price_col='close', window=20, num_std=2):
    df['bb_middle'] = df[price_col].rolling(window).mean()
    df['bb_std'] = df[price_col].rolling(window).std()
    df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
    return df

def add_keltner_channels(df, price_col='close', atr_col='atr', window=20, atr_mult=1.5):
    df['kc_middle'] = df[price_col].ewm(span=window, adjust=False).mean()
    # ATR calculation (if not already present)
    if atr_col not in df.columns:
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window).mean()
    df['kc_upper'] = df['kc_middle'] + atr_mult * df['atr']
    df['kc_lower'] = df['kc_middle'] - atr_mult * df['atr']
    return df
    
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
        df3m: pd.DataFrame = None,
        df5m: pd.DataFrame = None,
        tf: str = None,
        min_score: int = 8,
        required_passed: int = 7,      # NEW logic 100% : now 7 (for 7 gatekeepers)
        volume_multiplier: float = 2.0,
        liquidity_threshold: float = 0.85,   # <-- Set a default value
        kwargs = None
    ):
        if kwargs is None:
            kwargs = {}

        self.symbol = symbol
        self.df = df.copy()
        self.df3m = df3m.copy() if df3m is not None else None
        self.df5m = df5m.copy() if df5m is not None else None
        # Essential EMAs
        self.df["ema6"] = self.df["close"].ewm(span=6).mean()
        self.df["ema9"] = self.df["close"].ewm(span=9).mean()
        self.df["ema10"] = self.df["close"].ewm(span=10).mean()
        self.df["ema13"] = self.df["close"].ewm(span=13).mean()
        self.df["ema21"] = self.df["close"].ewm(span=21).mean()
        self.df["ema50"] = self.df["close"].ewm(span=50).mean()
        self.df["ema200"] = self.df["close"].ewm(span=200).mean()
        
        # Compute RSI as part of initialization or analysis
        self.df['RSI'] = self.compute_rsi(self.df)

        # ATR + ATR_MA
        self.df["atr"] = self.df["high"].sub(self.df["low"]).rolling(14).mean()
        self.df["atr_ma"] = self.df["atr"].rolling(20).mean()

        # Bollinger Bands
        self.df = add_bollinger_bands(self.df)
        # Keltner Channels (needs ATR column)
        self.df = add_keltner_channels(self.df)

        # Chop Zone (simple proxy: rolling std of close, or use your formula)
        self.df["chop_zone"] = self.df["close"].rolling(14).std()

        # Optional: ADX (if required in checks)
        try:
            self.df["adx"] = self.df.apply(lambda row: compute_adx(self.df), axis=1)
        except Exception:
            pass

        # Bid/Ask Columns (if not present, fill with NaN or zeros)
        if "bid" not in self.df.columns:
            self.df["bid"] = np.nan
        if "ask" not in self.df.columns:
            self.df["ask"] = np.nan

        # Higher Timeframe Volume (if not present, fill with NaN)
        if "higher_tf_volume" not in self.df.columns:
            self.df["higher_tf_volume"] = np.nan
    
        self.tf = tf
        self.min_score = min_score
        self.required_passed = required_passed
        self.volume_multiplier = volume_multiplier
        self.liquidity_threshold = liquidity_threshold

        # Weights for filters
        self.filter_weights_long = {
            "MACD": 4.9, "Volume Spike": 4.8, "Fractal Zone": 4.7, "EMA Cloud": 4.6, "Momentum": 4.5, "ATR Momentum Burst": 4.4,
            "MTF Volume Agreement": 4.3, "Trend Continuation": 4.2, "HATS": 4.1, "HH/LL Trend": 3.9, "Volatility Model": 3.8,
            "EMA Structure": 3.7, "Liquidity Awareness": 3.6, "Volatility Squeeze": 3.5, "Candle Confirmation": 3.4,
            "VWAP Divergence": 3.3, "Spread Filter": 3.2, "Chop Zone": 3.1, "Liquidity Pool": 2.9, "Support/Resistance": 2.8,
            "Smart Money Bias": 2.7, "Absorption": 2.6, "Wick Dominance": 2.5
        }

        self.filter_weights_short = {
            "MACD": 4.9, "Volume Spike": 4.8, "Fractal Zone": 4.7, "EMA Cloud": 4.6, "Momentum": 4.5, "ATR Momentum Burst": 4.4,
            "MTF Volume Agreement": 4.3, "Trend Continuation": 4.2, "HATS": 4.1, "HH/LL Trend": 3.9, "Volatility Model": 3.8,
            "EMA Structure": 3.7, "Liquidity Awareness": 3.6, "Volatility Squeeze": 3.5, "Candle Confirmation": 3.4,
            "VWAP Divergence": 3.3, "Spread Filter": 3.2, "Chop Zone": 3.1, "Liquidity Pool": 2.9, "Support/Resistance": 2.8,
            "Smart Money Bias": 2.7, "Absorption": 2.6, "Wick Dominance": 2.5
        }

        self.filter_names = list(set(self.filter_weights_long.keys()) | set(self.filter_weights_short.keys()))
        
        self.gatekeepers = [
            "MACD",
            "Volume Spike",
            "MTF Volume Agreement",
            "Liquidity Awareness",
            "Candle Confirmation",
            "Spread Filter",
            "Support/Resistance"
            ]

        # Directional-aware filters are those with different weights between LONG and SHORT
#        self.directional_aware_filters = [
#            "MACD", "ATR Momentum Burst", "HATS", "Liquidity Awareness",
#            "VWAP Divergence", "Support/Resistance", "Smart Money Bias", "Absorption"
#        ]

        self.df["ema6"] = self.df["close"].ewm(span=6, adjust=False).mean()
        self.df["ema9"] = self.df["close"].ewm(span=9, adjust=False).mean()
        self.df["ema10"] = self.df["close"].ewm(span=10, adjust=False).mean()
        self.df["ema13"] = self.df["close"].ewm(span=13, adjust=False).mean()
        self.df["ema20"] = self.df["close"].ewm(span=20).mean()
        self.df["ema50"] = self.df["close"].ewm(span=50).mean()
        self.df["ema200"] = self.df["close"].ewm(span=200).mean()
        self.df["vwap"] = (self.df["close"] * self.df["volume"]).cumsum() / self.df["volume"].cumsum()

    def compute_rsi(self, df, period=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
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
        # Nearly equal ONLY if change just happened
        nearly_equal = abs(ema_fast.iat[-1] - ema_slow.iat[-1]) < threshold and abs(ema_fast.iat[-2] - ema_slow.iat[-2]) >= threshold

        # Only one can fire at a time
        if crossed_up or (nearly_equal and ema_fast.iat[-1] > ema_slow.iat[-1] and not crossed_down):
            return "BULLISH_REVERSAL"
        elif crossed_down or (nearly_equal and ema_fast.iat[-1] < ema_slow.iat[-1] and not crossed_up):
            return "BEARISH_REVERSAL"
        else:
            return "NO_REVERSAL"

    def detect_rsi_reversal(self, threshold_overbought=65, threshold_oversold=35):
        rsi = self.df['RSI']

        # Sharpened: Only allow reversal if we move from below oversold to above oversold, or from above overbought to below overbought
        bullish = rsi.iat[-2] <= threshold_oversold and rsi.iat[-1] > threshold_oversold
        bearish = rsi.iat[-2] >= threshold_overbought and rsi.iat[-1] < threshold_overbought

        # These conditions are now strictly non-overlapping for any candle
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
            close_prev < open_prev and  # prev bearish
            close > open_ and           # current bullish
            open_ < close_prev and      # current open below prev close
            close > open_prev           # current close above prev open
        ) or (
            close_prev < open_prev and
            close > open_ and
            open_ < close_prev and
            abs(close - open_prev) < engulf_threshold
        )
    
        # Bearish engulfing: previous candle bullish, current bearish and engulfs
        bearish = (
            close_prev > open_prev and  # prev bullish
            close < open_ and           # current bearish
            open_ > close_prev and      # current open above prev close
            close < open_prev           # current close below prev open
        ) or (
            close_prev > open_prev and
            close < open_ and
            open_ > close_prev and
            abs(close - open_prev) < engulf_threshold
        )
    
        # Mutual exclusivity
        if bullish and not bearish:
            return "BULLISH_REVERSAL"
        elif bearish and not bullish:
            return "BEARISH_REVERSAL"
        else:
            return "NO_REVERSAL"

    def detect_adx_reversal(self, adx_threshold=25):
        print("[DEBUG] detect_adx_reversal called")
    
        required = ['adx', 'plus_di', 'minus_di']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            print(f"[DEBUG] ADX columns missing: {missing} | Available columns: {list(self.df.columns)}")
            return "NO_REVERSAL"
    
        adx = self.df['adx']
        plus_di = self.df['plus_di']
        minus_di = self.df['minus_di']
    
        if len(adx) < 2 or len(plus_di) < 2 or len(minus_di) < 2:
            print("[DEBUG] Not enough data to detect ADX reversal (need at least 2 rows).")
            return "NO_REVERSAL"
    
        print("[DEBUG] ADX last 5 values:", adx.tail().to_list())
        print("[DEBUG] plus_di last 5 values:", plus_di.tail().to_list())
        print("[DEBUG] minus_di last 5 values:", minus_di.tail().to_list())
    
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
        print(f"[DEBUG] ADX bullish? {bullish} | bearish? {bearish}")
    
        if bullish and not bearish:
            print("[DEBUG] ADX detected BULLISH_REVERSAL")
            return "BULLISH_REVERSAL"
        elif bearish and not bullish:
            print("[DEBUG] ADX detected BEARISH_REVERSAL")
            return "BEARISH_REVERSAL"
        else:
            print("[DEBUG] ADX detected NO_REVERSAL")
            return "NO_REVERSAL"

    def detect_stochrsi_reversal(self, overbought=0.8, oversold=0.2):
        print("[DEBUG] detect_stochrsi_reversal called")
    
        required = ['stochrsi_k', 'stochrsi_d']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            print(f"[DEBUG] StochRSI columns missing: {missing} | Available columns: {list(self.df.columns)}")
            return "NO_REVERSAL"
    
        k = self.df['stochrsi_k']
        d = self.df['stochrsi_d']
    
        if len(k) < 2 or len(d) < 2:
            print(f"[DEBUG] Not enough data for StochRSI (need at least 2 rows; k={len(k)}, d={len(d)})")
            return "NO_REVERSAL"
    
        print("[DEBUG] StochRSI k last 5 values:", k.tail().to_list())
        print("[DEBUG] StochRSI d last 5 values:", d.tail().to_list())
    
        bullish = k.iat[-2] <= oversold and k.iat[-1] > oversold and k.iat[-1] > d.iat[-1]
        bearish = k.iat[-2] >= overbought and k.iat[-1] < overbought and k.iat[-1] < d.iat[-1]
        print(f"[DEBUG] StochRSI bullish? {bullish} | bearish? {bearish}")
    
        if bullish:
            print("[DEBUG] StochRSI detected BULLISH_REVERSAL")
            return "BULLISH_REVERSAL"
        elif bearish:
            print("[DEBUG] StochRSI detected BEARISH_REVERSAL")
            return "BEARISH_REVERSAL"
        else:
            print("[DEBUG] StochRSI detected NO_REVERSAL")
            return "NO_REVERSAL"

    def detect_cci_reversal(self, overbought=100, oversold=-100):
        print("[DEBUG] detect_cci_reversal called")
    
        if 'cci' not in self.df.columns:
            print("[DEBUG] CCI column missing. Available columns:", list(self.df.columns))
            return "NO_REVERSAL"
    
        cci = self.df['cci']
    
        if len(cci) < 2:
            print("[DEBUG] Not enough data in CCI column (need at least 2 rows). Length:", len(cci))
            return "NO_REVERSAL"
    
        print("[DEBUG] CCI last 5 values:", cci.tail().to_list())
    
        bullish = cci.iat[-2] <= oversold and cci.iat[-1] > oversold
        bearish = cci.iat[-2] >= overbought and cci.iat[-1] < overbought
        print(f"[DEBUG] CCI bullish? {bullish} | bearish? {bearish}")
    
        if bullish:
            print("[DEBUG] CCI detected BULLISH_REVERSAL")
            return "BULLISH_REVERSAL"
        elif bearish:
            print("[DEBUG] CCI detected BEARISH_REVERSAL")
            return "BEARISH_REVERSAL"
        else:
            print("[DEBUG] CCI detected NO_REVERSAL")
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
        if bullish >= 1:
            return ("REVERSAL", "BULLISH")
        elif bearish >= 1:
            return ("REVERSAL", "BEARISH")
        else:
            return ("NONE", None)
            
    def detect_trend_continuation(self):
        # Example criteria for bullish continuation
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
        if reversal[0] == "REVERSAL":
            return ("REVERSAL", reversal[1])
        elif continuation in ["BULLISH_CONTINUATION", "BEARISH_CONTINUATION"]:
            return ("TREND CONTINUATION", continuation)
        else:
            return ("NONE", None)
    
#    def _calculate_all_filters_sum(self, results, direction):
#        """
#        Calculate the sum of weights for all passed filters for the given direction.
#        """
#        weights = self.filter_weights_long if direction == "LONG" else self.filter_weights_short
#        return sum(weights.get(f, 0) for f, v in results.items() if v and f in weights)

#    def _calculate_gatekeeper_sum(self, results, direction):
#        """
#        Calculate the sum of weights for passed gatekeeper filters for the given direction.
#        """
#        weights = self.filter_weights_long if direction == "LONG" else self.filter_weights_short
#        return sum(weights.get(f, 0) for f in self.gatekeepers if results.get(f, False))

#    def _calculate_directional_aware_sum(self, results, direction):
#        """
#        Calculate the sum of weights for passed directional-aware filters for the given direction.
#        """
#        weights = self.filter_weights_long if direction == "LONG" else self.filter_weights_short
#        return sum(weights.get(f, 0) for f in self.directional_aware_filters if results.get(f, False))

    def get_signal_direction(self, results_long, results_short):
        """
        Returns direction "LONG", "SHORT", or "NEUTRAL" based on gatekeeper and scoring filter results.

        1. Checks if all gatekeepers are passed for each direction.
        2. Calculates weighted sum of non-gatekeeper filters for each direction.
        3. Returns direction ONLY if gatekeeper rule and scoring filter are both stronger for one side.
        """

        # --- Gatekeeper Pass Check ---
        long_gk_passed = all(results_long.get(gk, False) for gk in self.gatekeepers)
        short_gk_passed = all(results_short.get(gk, False) for gk in self.gatekeepers)

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

# SIGNAL DIRECTION IF USING DIRECTIONAL AWARE FILTERS
#    def get_signal_direction(self, results_long, results_short):
#        """
#        Calculate three sums for both LONG and SHORT directions and return the direction
#        only if ALL THREE sums are greater for that direction. Otherwise return "NEUTRAL".
        
#        The three sums are:
#        1. Sum of all passed filter weights
#        2. Sum of passed gatekeeper filter weights
#        3. Sum of passed directional-aware filter weights
#        """
#        # Calculate three sums for LONG direction
#        long_all_filters = self._calculate_all_filters_sum(results_long, "LONG")
#        long_gatekeepers = self._calculate_gatekeeper_sum(results_long, "LONG")
#        long_directional = self._calculate_directional_aware_sum(results_long, "LONG")
        
        # Calculate three sums for SHORT direction
#        short_all_filters = self._calculate_all_filters_sum(results_short, "SHORT")
#        short_gatekeepers = self._calculate_gatekeeper_sum(results_short, "SHORT")
#        short_directional = self._calculate_directional_aware_sum(results_short, "SHORT")
        
        # Store sums for debugging (accessible via instance)
#        self._debug_sums = {
#            "long_all_filters": long_all_filters,
#            "long_gatekeepers": long_gatekeepers,
#            "long_directional": long_directional,
#            "short_all_filters": short_all_filters,
#            "short_gatekeepers": short_gatekeepers
#            "short_directional": short_directional
#        }
        
#        # Check if LONG wins all three comparisons
#        if (long_all_filters > short_all_filters and 
#            long_gatekeepers > short_gatekeepers):
#            long_directional > short_directional):
#            return "LONG"
        
        # Check if SHORT wins all three comparisons
#        elif (short_all_filters > long_all_filters and 
#              short_gatekeepers > long_gatekeepers):
#              short_directional > long_directional):
#            return "SHORT"
        
        # If neither direction wins all three, return NEUTRAL
#        else:
#            return "NEUTRAL"

#    PARTIAL New Super-GK ONLY RSI, Liquidity (Density) & OrderBookWall  
#    def superGK_check(self, signal_direction, orderbook_result, density_result):
#        if orderbook_result is None or density_result is None:
#            print(f"Signal blocked due to missing order book or density for {self.symbol}")
#            return False

#        bid_wall = orderbook_result.get('buy_wall', 0)
#        ask_wall = orderbook_result.get('sell_wall', 0)
#        bid_density = density_result.get('bid_density', 0)
#        ask_density = density_result.get('ask_density', 0)

        # Liquidity check
#        if bid_density < self.liquidity_threshold or ask_density < self.liquidity_threshold:
#            print(f"Signal blocked due to low liquidity for {self.symbol}")
#            return False

        # --- RSI Check ---
#        rsi = self.df['RSI'].iat[-1] if 'RSI' in self.df.columns else None
#        if rsi is None:
#            print(f"Signal blocked: RSI data missing for {self.symbol}")
#            return False

        # LONG: bid_wall > ask_wall AND bid_density > ask_density AND RSI < 70
#        if signal_direction == "LONG" and bid_wall > ask_wall and bid_density > ask_density and rsi < 70:
#            print(f"Signal passed for LONG: Wall, density, RSI < 70.")
#            return True
        # SHORT: ask_wall > bid_wall AND ask_density > bid_density AND RSI > 30
#        elif signal_direction == "SHORT" and ask_wall > bid_wall and ask_density > bid_density and rsi > 30:
#            print(f"Signal passed for SHORT: Wall, density, RSI > 30.")
#            return True
#        else:
#            print(
#                f"Signal blocked: SuperGK not aligned for {self.symbol} "
#                f"(bid_wall={bid_wall}, ask_wall={ask_wall}, bid_density={bid_density}, ask_density={ask_density}, RSI={rsi})"
#            )
#            return False

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
            
# COMPLETE New Super-GK ADX, ATR, RSI, Liquidity (Density) & OrderBookWall  
#    def superGK_check(self, signal_direction, orderbook_result, density_result):
        # Check if order book data is valid
#        if (orderbook_result is None 
#            or orderbook_result.get('buy_wall', 0) == 0 
#            or orderbook_result.get('sell_wall', 0) == 0):
#            print(f"Signal blocked due to missing order book data for {self.symbol}")
#            return False

#        bid_wall = orderbook_result.get('buy_wall', 0)
#        ask_wall = orderbook_result.get('sell_wall', 0)
#        midprice = orderbook_result.get('midprice', None)

#        bid_density = density_result.get('bid_density', 0)
#        ask_density = density_result.get('ask_density', 0)

        # Liquidity check
#        if bid_density < self.liquidity_threshold or ask_density < self.liquidity_threshold:
#            print(f"Signal blocked due to low liquidity for {self.symbol}")
#            return False

        # Wall support logic
#        if signal_direction == "LONG" and bid_wall < ask_wall:
#            print(f"Signal blocked due to weak buy-side support for {self.symbol}")
#            return False
#        if signal_direction == "SHORT" and ask_wall < bid_wall:
#            print(f"Signal blocked due to weak sell-side support for {self.symbol}")
#            return False

        # --- ATR RSI & ADX Calculations ---
#        atr = compute_atr(self.df)
#        rsi = compute_rsi(self.df)
#        adx = compute_adx(self.df)
#        price = self.df['close'].iat[-1]
#        atr_pct = atr / price if price else 0

        # --- Dynamic Market Regime Logic ---
#        low_vol_threshold = 0.005   # ATR < 1% of price = low volatility
#        high_vol_threshold = 0.05  # ATR > 3% of price = high volatility
#        bull_rsi = 60
#        bear_rsi = 40

        # --- ADX Filter ---
 #       adx_threshold = 20
 #       if adx < adx_threshold:
 #           print(f"Signal blocked: Trend strength too weak (ADX={adx:.2f})")
 #           return False
    
 #       if atr_pct < low_vol_threshold:
 #           print(f"Signal blocked: Volatility too low (ATR={atr:.4f}, ATR%={atr_pct:.2%})")
 #           return False

 #       if signal_direction == "LONG":
 #           if rsi < bull_rsi:
 #               print(f"Signal blocked: LONG but RSI not bullish enough (RSI={rsi:.2f})")
 #               return False
 #       elif signal_direction == "SHORT":
 #           if rsi > bear_rsi:
 #               print(f"Signal blocked: SHORT but RSI not bearish enough (RSI={rsi:.2f})")
 #               return False

 #       if bear_rsi < rsi < bull_rsi:
 #           print(f"Signal blocked: Market is ranging (RSI={rsi:.2f})")
 #           return False

 #       if atr_pct > high_vol_threshold:
 #           print(f"Signal blocked: Volatility extremely high, unstable market (ATR={atr:.4f}, ATR%={atr_pct:.2%})")
 #           return False

 # Final regime decision
 #       if signal_direction == "LONG" and bid_density > ask_density:
 #           print(f"Signal passed for LONG: Strong bid-side liquidity for {self.symbol}")
 #           return True
 #       elif signal_direction == "SHORT" and ask_density > bid_density:
 #           print(f"Signal passed for SHORT: Strong ask-side liquidity for {self.symbol}")
 #           return True
 #       else:
 #           print(f"Signal blocked due to neutral market conditions for {self.symbol}")
 #           return False

    def analyze(self):
        reversal_detected = False  # <-- Add this here, at the very start of the function
        
        if self.df.empty:
            print(f"[{self.symbol}] Error: DataFrame empty.")
            return None

        # --- Detect reversal but do not block the signal ---
        # reversal = self.explicit_reversal_gate()
        # reversal_detected = reversal in ["LONG", "SHORT"]
        reversal_route, reversal_side = self.explicit_reversal_gate()
        reversal_detected = reversal_route == "REVERSAL"    # <-- PLACE THIS HERE
        route = "REVERSAL" if reversal_detected else "TREND CONTINUATION"
        
        # List all filter names
        filter_names = [
            "Fractal Zone", "EMA Cloud", "MACD", "Momentum", "HATS", "Volume Spike",
            "VWAP Divergence", "MTF Volume Agreement", "HH/LL Trend", "EMA Structure",
            "Chop Zone", "Candle Confirmation", "Wick Dominance", "Absorption",
            "Support/Resistance", "Smart Money Bias", "Liquidity Pool", "Spread Filter",
            "Liquidity Awareness", "Trend Continuation", "Volatility Model",
            "ATR Momentum Burst", "Volatility Squeeze"
        ]

        # Map filter names to functions, handling timeframes if needed
        filter_function_map = {
            "Fractal Zone": getattr(self, "_check_fractal_zone", None),
            "EMA Cloud": getattr(self, "_check_ema_cloud", None),
            "MACD": getattr(self, "_check_macd", None),
            "Momentum": getattr(self, "_check_momentum", None),
            "HATS": getattr(self, "_check_hats", None),
            "Volume Spike": self.volume_surge_confirmed if self.tf == "3min" else getattr(self, "_check_volume_spike", None),
            "VWAP Divergence": getattr(self, "_check_vwap_divergence", None),
            "MTF Volume Agreement": getattr(self, "_check_mtf_volume_agreement", None),
            "HH/LL Trend": getattr(self, "_check_hh_ll", None),
            "EMA Structure": getattr(self, "_check_ema_structure", None),
            "Chop Zone": getattr(self, "_check_chop_zone", None),
            "Candle Confirmation": getattr(self, "_check_candle_close", None),
            "Wick Dominance": getattr(self, "_check_wick_dominance", None),
            "Absorption": getattr(self, "_check_absorption", None),
            "Support/Resistance": getattr(self, "_check_support_resistance", None),
            "Smart Money Bias": getattr(self, "_check_smart_money_bias", None),
            "Liquidity Pool": getattr(self, "_check_liquidity_pool", None),
            "Spread Filter": getattr(self, "_check_spread_filter", None),
            "Liquidity Awareness": getattr(self, "_check_liquidity_awareness", None),
            "Trend Continuation": getattr(self, "_check_trend_continuation", None),
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
                result = fn()
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
                # PER-FILTER STATUS LOGGING REMOVED
            except Exception as e:
                print(f"[{self.symbol}] {name} ERROR: {e}")
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
        
        # --- Calculate weighted sums using the correct dicts ---
        # long_weight_sum = sum(self.filter_weights_long.get(name, 0) for name, passed in results_long.items() if passed)
        # short_weight_sum = sum(self.filter_weights_short.get(name, 0) for name, passed in results_short.items() if passed)

        # print(f"[{self.symbol}] Total LONG weight: {long_weight_sum}")
        # print(f"[{self.symbol}] Total SHORT weight: {short_weight_sum}")
      
        # Weighted sums for info only
        # long_weight_sum = sum(self.filter_weights_long.get(name, 0) for name, passed in results_long.items() if passed)
        # short_weight_sum = sum(self.filter_weights_short.get(name, 0) for name, passed in results_short.items() if passed)

        # print(f"[{self.symbol}] Total LONG weight: {long_weight_sum}")
        # print(f"[{self.symbol}] Total SHORT weight: {short_weight_sum}")
      
        # --- Gatekeeper pass/fail and count ---
        passed_gk_long = [f for f in self.gatekeepers if results_long.get(f, False)]
        passed_gk_short = [f for f in self.gatekeepers if results_short.get(f, False)]
        passes_long = len(passed_gk_long)
        passes_short = len(passed_gk_short)

        # --- Calculate WEIGHTED and CONFIDENCE using only non-GK filters ---
        passed_non_gk_long = [f for f in non_gk_filters if results_long.get(f, False)]
        passed_non_gk_short = [f for f in non_gk_filters if results_short.get(f, False)]

        # Calculate total weights for gatekeepers
        total_non_gk_weight_long = sum(self.filter_weights_long.get(f, 0) for f in non_gk_filters)
        total_non_gk_weight_short = sum(self.filter_weights_short.get(f, 0) for f in non_gk_filters)
        passed_non_gk_weight_long = sum(self.filter_weights_long.get(f, 0) for f in passed_non_gk_long)
        passed_non_gk_weight_short = sum(self.filter_weights_short.get(f, 0) for f in passed_non_gk_short)

        # Add failed GK calculations
        failed_gk_long = [f for f in self.gatekeepers if f not in passed_gk_long]
        failed_gk_short = [f for f in self.gatekeepers if f not in passed_gk_short]

        # Print logic
        print(f"[{self.symbol}] Passed GK LONG: {passed_gk_long}")
        print(f"[{self.symbol}] Failed GK LONG: {failed_gk_long}")
        print(f"[{self.symbol}] Passed GK SHORT: {passed_gk_short}")
        print(f"[{self.symbol}] Failed GK SHORT: {failed_gk_short}")

        confidence_long = round(100 * passed_non_gk_weight_long / total_non_gk_weight_long, 1) if total_non_gk_weight_long else 0.0
        confidence_short = round(100 * passed_non_gk_weight_short / total_non_gk_weight_short, 1) if total_non_gk_weight_short else 0.0

        # --- Use selected direction's stats ---
        if direction == "LONG":
            score = long_score
            passes = passes_long
            confidence = confidence_long
            passed_weight = passed_non_gk_weight_long
            total_weight = total_non_gk_weight_long
        elif direction == "SHORT":
            score = short_score
            passes = passes_short
            confidence = confidence_short
            passed_weight = passed_non_gk_weight_short
            total_weight = total_non_gk_weight_short
        else:
            score = max(long_score, short_score)
            passes = max(passes_long, passes_short)
            confidence = max(confidence_long, confidence_short)
            passed_weight = max(passed_non_gk_weight_long, passed_non_gk_weight_short)
            total_weight = max(total_non_gk_weight_long, total_non_gk_weight_short)

        confidence = round(100 * passed_weight / total_weight, 1) if total_weight else 0.0

        # --- SuperGK check (unchanged) ---
        orderbook_result = get_order_wall_delta(self.symbol)
        density_result = get_resting_density(self.symbol)
        super_gk_ok = self.superGK_check(direction, orderbook_result, density_result)

        print("[DEBUG] direction:", direction)
        print("[DEBUG] score:", score, "min_score:", self.min_score)
        print("[DEBUG] passes:", passes, "required_passed:", self.required_passed)
        print("[DEBUG] super_gk_ok:", super_gk_ok)

        valid_signal = (
            direction in ["LONG", "SHORT"]
            and score >= self.min_score
            and passes >= self.required_passed
            and super_gk_ok
        )

        price = self.df['close'].iat[-1] if valid_signal else None
        price_str = f"{price:.6f}" if price is not None else "N/A"


        price = self.df['close'].iat[-1] if valid_signal else None
        price_str = f"{price:.6f}" if price is not None else "N/A"

        print(f"[DEBUG] reversal_route: {reversal_route}, reversal_side: {reversal_side}, route: {route}, direction: {direction}, valid_signal: {valid_signal}")
        if valid_signal:
            route = "REVERSAL" if reversal_detected else "TREND CONTINUATION"
        else:
            route = None

        signal_type = direction if valid_signal else None

        score_max = len(non_gk_filters)

        message = (
            f"{direction or 'NO-SIGNAL'} on {self.symbol} @ {price_str} "
            f"| Score: {score}/{score_max} | Passed GK: {passes}/{len(self.gatekeepers)} "
            f"| Confidence: {confidence}% (Weighted: {passed_weight:.1f}/{total_weight:.1f})"
            f" | Route: {route if valid_signal else 'N/A'}"
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
            "gatekeepers_total": len(self.gatekeepers),
            "passed_weight": round(passed_weight, 1),
            "total_weight": round(total_weight, 1),
            "confidence": confidence,
            "bias": direction,
            "price": price,
            "valid_signal": valid_signal,
            "signal_type": signal_type,
            "Route": route,
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

    def volume_surge_confirmed(self):
        result = self._check_volume_spike()
        avg_5m = self.df5m['volume'].rolling(10).mean().iat[-1] if self.df5m is not None else 0
        volume_5m_spike = self.df5m['volume'].iat[-1] > avg_5m * self.volume_multiplier if self.df5m is not None else False
        if result in ["LONG", "SHORT"] and self._check_5m_volume_trend() and volume_5m_spike:
            return result
        return None

    def _check_volume_spike(self, zscore_threshold=1.5):
        # Calculate z-score of current volume vs recent (rolling 10)
        avg = self.df['volume'].rolling(10).mean().iat[-1]
        std = self.df['volume'].rolling(10).std().iat[-1]
        zscore = self._safe_divide(self.df['volume'].iat[-1] - avg, std)
    
        # Price direction
        price_up = self.df['close'].iat[-1] > self.df['close'].iat[-2]
        price_down = self.df['close'].iat[-1] < self.df['close'].iat[-2]
    
        # Volume trend
        vol_up = self.df['volume'].iat[-1] > self.df['volume'].iat[-2]

        # use volume multiplier
        volume_spike = self.df['volume'].iat[-1] > avg * self.volume_multiplier
        
        # LONG signal: volume spike + price rising + volume rising
        long_conditions = [zscore > zscore_threshold, price_up, vol_up, volume_spike]
        long_met = sum(long_conditions)
    
        # SHORT signal: volume spike + price falling + volume rising
        short_conditions = [zscore > zscore_threshold, price_down, vol_up, volume_spike]
        short_met = sum(short_conditions)

        # Require at least 2/3 for a signal
        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_5m_volume_trend(self):
        if self.df5m is None or len(self.df5m) < 2:
            return False
        return self.df5m['volume'].iat[-1] > self.df5m['volume'].iat[-2]

    def _check_fractal_zone(self, buffer_pct=0.005, window=20):
        fractal_low = self.df['low'].rolling(window).min().iat[-1]
        fractal_low_prev = self.df['low'].rolling(window).min().iat[-2]
        fractal_high = self.df['high'].rolling(window).max().iat[-1]
        fractal_high_prev = self.df['high'].rolling(window).max().iat[-2]

        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]

        # LONG conditions
        cond1_long = close > fractal_low * (1 + buffer_pct)
        cond2_long = close > close_prev
        cond3_long = fractal_low > fractal_low_prev

        # SHORT conditions
        cond1_short = close < fractal_high * (1 - buffer_pct)
        cond2_short = close < close_prev
        cond3_short = fractal_high < fractal_high_prev

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None
    
    def _check_ema_cloud(self):
        ema20 = self.df['ema20'].iat[-1]
        ema50 = self.df['ema50'].iat[-1]
        ema20_prev = self.df['ema20'].iat[-2]
        close = self.df['close'].iat[-1]

        # LONG conditions
        cond1_long = ema20 > ema50
        cond2_long = ema20 > ema20_prev
        cond3_long = close > ema20

        # SHORT conditions
        cond1_short = ema20 < ema50
        cond2_short = ema20 < ema20_prev
        cond3_short = close < ema20

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_macd(self):
        e12 = self.df['close'].ewm(span=12).mean()
        e26 = self.df['close'].ewm(span=26).mean()
        macd = e12 - e26
        signal = macd.ewm(span=9).mean()

        # LONG conditions
        condition_1_long = macd.iat[-1] > signal.iat[-1]                # MACD above Signal Line
        condition_2_long = macd.iat[-1] > macd.iat[-2]                  # MACD is rising
        condition_3_long = self.df['close'].iat[-1] > self.df['close'].iat[-2]  # Price action rising
        condition_4_long = macd.iat[-1] > signal.iat[-1] and macd.iat[-1] > macd.iat[-2]  # MACD Divergence

        # SHORT conditions
        condition_1_short = macd.iat[-1] < signal.iat[-1]                # MACD below Signal Line
        condition_2_short = macd.iat[-1] < macd.iat[-2]                  # MACD is falling
        condition_3_short = self.df['close'].iat[-1] < self.df['close'].iat[-2]  # Price action falling
        condition_4_short = macd.iat[-1] < signal.iat[-1] and macd.iat[-1] < macd.iat[-2]  # MACD Divergence

        long_conditions_met = sum([condition_1_long, condition_2_long, condition_3_long, condition_4_long])
        short_conditions_met = sum([condition_1_short, condition_2_short, condition_3_short, condition_4_short])

        # If 3 out of 4 conditions are met, we pass the filter
        if long_conditions_met >= 3:
            return "LONG"
        elif short_conditions_met >= 3:
            return "SHORT"
        else:
            return None

    def _check_momentum(self, window=10):
        # Calculate Rate of Change (ROC)
        roc = self.df['close'].pct_change(periods=window)
        momentum = roc.iat[-1]
        momentum_prev = roc.iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]

        # LONG conditions
        cond1_long = momentum > 0
        cond2_long = momentum > momentum_prev
        cond3_long = close > close_prev

        # SHORT conditions
        cond1_short = momentum < 0
        cond2_short = momentum < momentum_prev
        cond3_short = close < close_prev

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_hats(self):
        # Define your moving averages
        fast = self.df['ema10'].iat[-1]
        mid = self.df['ema21'].iat[-1]
        slow = self.df['ema50'].iat[-1]

        fast_prev = self.df['ema10'].iat[-2]
        mid_prev = self.df['ema21'].iat[-2]
        slow_prev = self.df['ema50'].iat[-2]

        close = self.df['close'].iat[-1]

        # LONG conditions
        cond1_long = fast > mid and mid > slow
        cond2_long = fast > fast_prev and mid > mid_prev and slow > slow_prev
        cond3_long = close > fast

        # SHORT conditions
        cond1_short = fast < mid and mid < slow
        cond2_short = fast < fast_prev and mid < mid_prev and slow < slow_prev
        cond3_short = close < fast

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_vwap_divergence(self):
        vwap = self.df['vwap'].iat[-1]
        vwap_prev = self.df['vwap'].iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]

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

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None
            
    def _check_mtf_volume_agreement(self):
        # Current timeframe
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]

        # Higher timeframe (e.g., hourly or daily)
        higher_tf_volume = self.df['higher_tf_volume'].iat[-1]
        higher_tf_volume_prev = self.df['higher_tf_volume'].iat[-2]

        # LONG conditions
        cond1_long = volume > volume_prev
        cond2_long = higher_tf_volume > higher_tf_volume_prev
        cond3_long = close > close_prev

        # SHORT conditions
        cond1_short = volume > volume_prev
        cond2_short = higher_tf_volume > higher_tf_volume_prev
        cond3_short = close < close_prev

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None
        
    def _check_hh_ll(self):
        high = self.df['high'].iat[-1]
        high_prev = self.df['high'].iat[-2]
        low = self.df['low'].iat[-1]
        low_prev = self.df['low'].iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]

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

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_ema_structure(self):
        # Current values
        close = self.df['close'].iat[-1]
        ema9 = self.df['ema9'].iat[-1]
        ema21 = self.df['ema21'].iat[-1]
        ema50 = self.df['ema50'].iat[-1]
        # Previous values
        ema9_prev = self.df['ema9'].iat[-2]
        ema21_prev = self.df['ema21'].iat[-2]
        ema50_prev = self.df['ema50'].iat[-2]

        # LONG conditions
        cond1_long = ema9 > ema21 and ema21 > ema50
        cond2_long = close > ema9 and close > ema21 and close > ema50
        cond3_long = ema9 > ema9_prev and ema21 > ema21_prev and ema50 > ema50_prev

        # SHORT conditions
        cond1_short = ema9 < ema21 and ema21 < ema50
        cond2_short = close < ema9 and close < ema21 and close < ema50
        cond3_short = ema9 < ema9_prev and ema21 < ema21_prev and ema50 < ema50_prev

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_chop_zone(self, chop_threshold=40):
        chop = self.df['chop_zone'].iat[-1]
        ema9 = self.df['ema9'].iat[-1]
        ema21 = self.df['ema21'].iat[-1]
        close = self.df['close'].iat[-1]
        adx = self.df['adx'].iat[-1] if 'adx' in self.df.columns else None

        # Filter out choppy market
        if chop >= chop_threshold:
            return None

        # LONG conditions
        cond1_long = ema9 > ema21 if ema9 and ema21 else False
        cond2_long = adx > 20 if adx else True  # Optional: ADX filter
        cond3_long = close > ema9

        # SHORT conditions
        cond1_short = ema9 < ema21 if ema9 and ema21 else False
        cond2_short = adx > 20 if adx else True
        cond3_short = close < ema9

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_candle_close(self):
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

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_wick_dominance(self):
        open_ = self.df['open'].iat[-1]
        high = self.df['high'].iat[-1]
        low = self.df['low'].iat[-1]
        close = self.df['close'].iat[-1]

        # Calculate candle components
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        body = abs(close - open_)

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
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_absorption(self, window=20, buffer_pct=0.005):
        # Calculate recent low/high for proximity
        low = self.df['low'].rolling(window).min().iat[-1]
        high = self.df['high'].rolling(window).max().iat[-1]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
        avg_volume = self.df['volume'].rolling(window).mean().iat[-1]

        # LONG conditions
        cond1_long = close <= low * (1 + buffer_pct)
        cond2_long = volume > avg_volume and volume > volume_prev
        cond3_long = close >= close_prev

        # SHORT conditions
        cond1_short = close >= high * (1 - buffer_pct)
        cond2_short = volume > avg_volume and volume > volume_prev
        cond3_short = close <= close_prev

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_support_resistance(self, window=20, buffer_pct=0.005):
        # Calculate recent support (local min) and resistance (local max)
        support = self.df['low'].rolling(window).min().iat[-1]
        resistance = self.df['high'].rolling(window).max().iat[-1]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]

        # LONG conditions
        cond1_long = close <= support * (1 + buffer_pct)
        cond2_long = close > close_prev
        cond3_long = volume > volume_prev

        # SHORT conditions
        cond1_short = close >= resistance * (1 - buffer_pct)
        cond2_short = close < close_prev
        cond3_short = volume > volume_prev

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_smart_money_bias(self, volume_window=20):
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
        avg_volume = self.df['volume'].rolling(volume_window).mean().iat[-1]
        vwap = self.df['vwap'].iat[-1]

        # Check for large volume uptick
        large_vol = volume > avg_volume and volume > volume_prev

        # LONG conditions
        cond1_long = large_vol
        cond2_long = close > close_prev
        cond3_long = close > vwap

        # SHORT conditions
        cond1_short = large_vol
        cond2_short = close < close_prev
        cond3_short = close < vwap

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_spread_filter(self, window=20):
        high = self.df['high'].iat[-1]
        low = self.df['low'].iat[-1]
        open_ = self.df['open'].iat[-1]
        close = self.df['close'].iat[-1]

        high_prev = self.df['high'].iat[-2]
        low_prev = self.df['low'].iat[-2]
        open_prev = self.df['open'].iat[-2]
        close_prev = self.df['close'].iat[-2]

        spread = high - low
        spread_prev = high_prev - low_prev
        spread_ma = self.df['spread'].rolling(window).mean().iat[-1] if 'spread' in self.df.columns else self.df['high'].sub(self.df['low']).rolling(window).mean().iat[-1]

        # Add spread column if missing
        if 'spread' not in self.df.columns:
            self.df['spread'] = self.df['high'] - self.df['low']

        # LONG conditions
        cond1_long = spread > spread_prev
        cond2_long = close > open_
        cond3_long = spread > spread_ma

        # SHORT conditions
        cond1_short = spread > spread_prev
        cond2_short = close < open_
        cond3_short = spread > spread_ma

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_liquidity_pool(self, lookback=20):
        close = self.df['close'].iat[-1]
        high = self.df['high'].iat[-1]
        low = self.df['low'].iat[-1]
    
        # Identify recent liquidity pools
        recent_high = self.df['high'].rolling(lookback).max().iat[-2]
        recent_low = self.df['low'].rolling(lookback).min().iat[-2]
    
        # LONG: break or sweep above recent high
        cond1_long = close > recent_high
        cond2_long = low < recent_low and close > recent_low  # sweep and reversal
        cond3_long = close > recent_high

        # SHORT: break or sweep below recent low
        cond1_short = close < recent_low
        cond2_short = high > recent_high and close < recent_high  # sweep and reversal
        cond3_short = close < recent_low

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_liquidity_awareness(self):
        bid = self.df['bid'].iat[-1]
        ask = self.df['ask'].iat[-1]
        bid_prev = self.df['bid'].iat[-2]
        ask_prev = self.df['ask'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]

        spread = ask - bid
        spread_prev = ask_prev - bid_prev

        # LONG conditions
        cond1_long = spread < spread_prev
        cond2_long = volume > volume_prev
        cond3_long = close > close_prev

        # SHORT conditions
        cond1_short = spread > spread_prev
        cond2_short = volume > volume_prev
        cond3_short = close < close_prev

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_trend_continuation(self, ma_col='ema21'):
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        ma = self.df[ma_col].iat[-1]
        ma_prev = self.df[ma_col].iat[-2]

        # LONG conditions
        cond1_long = close > ma
        cond2_long = ma > ma_prev
        cond3_long = close > close_prev

        # SHORT conditions
        cond1_short = close < ma
        cond2_short = ma < ma_prev
        cond3_short = close < close_prev

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_volatility_model(self, atr_col='atr', atr_ma_col='atr_ma'):
        atr = self.df[atr_col].iat[-1]
        atr_prev = self.df[atr_col].iat[-2]
        atr_ma = self.df[atr_ma_col].iat[-1]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]

        # LONG conditions
        cond1_long = atr > atr_prev
        cond2_long = close > close_prev
        cond3_long = atr > atr_ma

        # SHORT conditions
        cond1_short = atr > atr_prev
        cond2_short = close < close_prev
        cond3_short = atr > atr_ma

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_atr_momentum_burst(self, atr_period=14, burst_mult=1.5):
        # ATR must be calculated and present in self.df['atr']
        atr = self.df['atr'].iat[-1]
        atr_prev = self.df['atr'].iat[-2]
        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        price_change = close - close_prev

        # LONG conditions
        cond1_long = close > close_prev
        cond2_long = price_change > burst_mult * atr
        cond3_long = atr > atr_prev

        # SHORT conditions
        cond1_short = close < close_prev
        cond2_short = price_change < -burst_mult * atr
        cond3_short = atr > atr_prev

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
        else:
            return None

    def _check_volatility_squeeze(self):
        bb_width = self.df['bb_upper'].iat[-1] - self.df['bb_lower'].iat[-1]
        kc_width = self.df['kc_upper'].iat[-1] - self.df['kc_lower'].iat[-1]
        bb_width_prev = self.df['bb_upper'].iat[-2] - self.df['bb_lower'].iat[-2]
        kc_width_prev = self.df['kc_upper'].iat[-2] - self.df['kc_lower'].iat[-2]

        close = self.df['close'].iat[-1]
        close_prev = self.df['close'].iat[-2]
        volume = self.df['volume'].iat[-1]
        volume_prev = self.df['volume'].iat[-2]

        # Squeeze fires when BB width expands after contraction below KC width
        squeeze_firing = bb_width > kc_width and bb_width_prev < kc_width_prev

        # LONG conditions
        cond1_long = squeeze_firing
        cond2_long = close > close_prev
        cond3_long = volume > volume_prev

        # SHORT conditions
        cond1_short = squeeze_firing
        cond2_short = close < close_prev
        cond3_short = volume > volume_prev

        long_met = sum([cond1_long, cond2_long, cond3_long])
        short_met = sum([cond1_short, cond2_short, cond3_short])

        if long_met >= 2:
            return "LONG"
        elif short_met >= 2:
            return "SHORT"
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

# def superGK_check(self, symbol, signal_direction):
#    # Fetch order book data (Bid/Ask details)
#    bids, asks = self._fetch_order_book(symbol)

#    # Check if order book data is available
#    if bids is None or asks is None:
#        print(f"Signal blocked due to missing order book data for {symbol}")
#        return False  # Block the signal if no order book data is available

#    # Calculate bid and ask wall sizes (order wall delta)
#    bid_wall = self.get_order_wall_delta(symbol, side='bid')
#    ask_wall = self.get_order_wall_delta(symbol, side='ask')

#    # Check market liquidity (Resting Density)
#    resting_density = self.get_resting_density(symbol)
#    bid_density = resting_density['bid_density']
#    ask_density = resting_density['ask_density']

#    # Market strength check based on liquidity (if density is too low, block)
#    if bid_density < self.liquidity_threshold or ask_density < self.liquidity_threshold:
#        print(f"Signal blocked due to low liquidity for {symbol}")
#        return False  # Block signal due to insufficient liquidity

#    # Further check based on bid/ask wall delta
#    if bid_wall < ask_wall:
#        print(f"Signal blocked due to weak buy-side support for {symbol}")
#        return False  # Block signal if ask wall is stronger

#    # Determine signal based on liquidity and market depth
#    if bid_density > ask_density:
#        print(f"Signal passed for LONG: Strong bid-side liquidity for {symbol}")
#        return True  # Allow LONG signal if bid-side liquidity is stronger
#    elif ask_density > bid_density:
#        print(f"Signal passed for SHORT: Strong ask-side liquidity for {symbol}")
#        return True  # Allow SHORT signal if ask-side liquidity is stronger
#    else:
#        print(f"Signal blocked due to neutral market conditions for {symbol}")
#        return False  # Block signal if market liquidity is balanced

