# smart_filter.py

import requests
import pandas as pd
import numpy as np
from kucoin_orderbook import get_order_wall_delta
from kucoin_density import get_resting_density

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
        min_score: int = 15,
        required_passed: int = 12,      # NEW: now 11 (for 17 gatekeepers)
        volume_multiplier: float = 2.0,
        liquidity_threshold: float = 0.2,   # <-- Set a default value
        kwargs = None
    ):
        if kwargs is None:
            kwargs = {}

        self.symbol = symbol
        self.df = df.copy()
        self.df3m = df3m.copy() if df3m is not None else None
        self.df5m = df5m.copy() if df5m is not None else None
        # Essential EMAs
        self.df["ema9"] = self.df["close"].ewm(span=9).mean()
        self.df["ema10"] = self.df["close"].ewm(span=10).mean()
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
            "MACD": 4.8, "Volume Spike": 4.8, "Fractal Zone": 4.7, "EMA Cloud": 4.6, "Momentum": 4.5, "ATR Momentum Burst": 4.4,
            "MTF Volume Agreement": 4.3, "Trend Continuation": 4.2, "HATS": 4.1, "HH/LL Trend": 3.9, "Volatility Model": 3.8,
            "EMA Structure": 3.7, "Liquidity Awareness": 3.7, "Volatility Squeeze": 3.5, "Candle Confirmation": 3.4,
            "VWAP Divergence": 3.2, "Spread Filter": 3.2, "Chop Zone": 3.1, "Liquidity Pool": 2.9, "Support/Resistance": 2.8,
            "Smart Money Bias": 2.7, "Absorption": 2.6, "Wick Dominance": 2.5
        }

        self.filter_weights_short = {
            "MACD": 4.9, "Volume Spike": 4.8, "Fractal Zone": 4.7, "EMA Cloud": 4.6, "Momentum": 4.5, "ATR Momentum Burst": 4.5,
            "MTF Volume Agreement": 4.3, "Trend Continuation": 4.2, "HATS": 4.0, "HH/LL Trend": 3.9, "Volatility Model": 3.8,
            "EMA Structure": 3.7, "Liquidity Awareness": 3.6, "Volatility Squeeze": 3.5, "Candle Confirmation": 3.4,
            "VWAP Divergence": 3.3, "Spread Filter": 3.2, "Chop Zone": 3.1, "Liquidity Pool": 2.9, "Support/Resistance": 2.9,
            "Smart Money Bias": 2.6, "Absorption": 2.5, "Wick Dominance": 2.5
        }

        self.gatekeepers = [
            "Fractal Zone", "EMA Cloud", "MACD", "Momentum", "HATS",
            "Volume Spike", "VWAP Divergence", "MTF Volume Agreement",
            "HH/LL Trend", "EMA Structure", "Chop Zone",
            "Candle Confirmation", "Trend Continuation",
            "Volatility Model", "Liquidity Awareness",
            "ATR Momentum Burst", "Volatility Squeeze"
        ]

        # Directional-aware filters are those with different weights between LONG and SHORT
        self.directional_aware_filters = [
            "MACD", "ATR Momentum Burst", "HATS", "Liquidity Awareness",
            "VWAP Divergence", "Support/Resistance", "Smart Money Bias", "Absorption"
        ]

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

    def _calculate_all_filters_sum(self, results, direction):
        """
        Calculate the sum of weights for all passed filters for the given direction.
        """
        weights = self.filter_weights_long if direction == "LONG" else self.filter_weights_short
        return sum(weights.get(f, 0) for f, v in results.items() if v and f in weights)

    def _calculate_gatekeeper_sum(self, results, direction):
        """
        Calculate the sum of weights for passed gatekeeper filters for the given direction.
        """
        weights = self.filter_weights_long if direction == "LONG" else self.filter_weights_short
        return sum(weights.get(f, 0) for f in self.gatekeepers if results.get(f, False))

    def _calculate_directional_aware_sum(self, results, direction):
        """
        Calculate the sum of weights for passed directional-aware filters for the given direction.
        """
        weights = self.filter_weights_long if direction == "LONG" else self.filter_weights_short
        return sum(weights.get(f, 0) for f in self.directional_aware_filters if results.get(f, False))
            
    def get_signal_direction(self, results_long, results_short):
        """
        Calculate three sums for both LONG and SHORT directions and return the direction
        only if ALL THREE sums are greater for that direction. Otherwise return "NEUTRAL".
        
        The three sums are:
        1. Sum of all passed filter weights
        2. Sum of passed gatekeeper filter weights
        3. Sum of passed directional-aware filter weights
        """
        # Calculate three sums for LONG direction
        long_all_filters = self._calculate_all_filters_sum(results_long, "LONG")
        long_gatekeepers = self._calculate_gatekeeper_sum(results_long, "LONG")
        long_directional = self._calculate_directional_aware_sum(results_long, "LONG")
        
        # Calculate three sums for SHORT direction
        short_all_filters = self._calculate_all_filters_sum(results_short, "SHORT")
        short_gatekeepers = self._calculate_gatekeeper_sum(results_short, "SHORT")
        short_directional = self._calculate_directional_aware_sum(results_short, "SHORT")
        
        # Store sums for debugging (accessible via instance)
        self._debug_sums = {
            "long_all_filters": long_all_filters,
            "long_gatekeepers": long_gatekeepers,
            "long_directional": long_directional,
            "short_all_filters": short_all_filters,
            "short_gatekeepers": short_gatekeepers,
            "short_directional": short_directional
        }
        
        # Check if LONG wins all three comparisons
        if (long_all_filters > short_all_filters and 
            long_gatekeepers > short_gatekeepers and 
            long_directional > short_directional):
            return "LONG"
        
        # Check if SHORT wins all three comparisons
        elif (short_all_filters > long_all_filters and 
              short_gatekeepers > long_gatekeepers and 
              short_directional > long_directional):
            return "SHORT"
        
        # If neither direction wins all three, return NEUTRAL
        else:
            return "NEUTRAL"

#    PARTIAL New Super-GK ONLY RSI, Liquidity (Density) & OrderBookWall  
    def superGK_check(self, signal_direction, orderbook_result, density_result):
        if orderbook_result is None or density_result is None:
            print(f"Signal blocked due to missing order book or density for {self.symbol}")
            return False

        bid_wall = orderbook_result.get('buy_wall', 0)
        ask_wall = orderbook_result.get('sell_wall', 0)
        bid_density = density_result.get('bid_density', 0)
        ask_density = density_result.get('ask_density', 0)

        # Liquidity check
        if bid_density < self.liquidity_threshold or ask_density < self.liquidity_threshold:
            print(f"Signal blocked due to low liquidity for {self.symbol}")
            return False

        # --- RSI Check ---
        rsi = self.df['RSI'].iat[-1] if 'RSI' in self.df.columns else None
        if rsi is None:
            print(f"Signal blocked: RSI data missing for {self.symbol}")
            return False

        # LONG: bid_wall > ask_wall AND bid_density > ask_density AND RSI < 70
        if signal_direction == "LONG" and bid_wall > ask_wall and bid_density > ask_density and rsi < 70:
            print(f"Signal passed for LONG: Wall, density, RSI < 70.")
            return True
        # SHORT: ask_wall > bid_wall AND ask_density > bid_density AND RSI > 30
        elif signal_direction == "SHORT" and ask_wall > bid_wall and ask_density > bid_density and rsi > 30:
            print(f"Signal passed for SHORT: Wall, density, RSI > 30.")
            return True
        else:
            print(
                f"Signal blocked: SuperGK not aligned for {self.symbol} "
                f"(bid_wall={bid_wall}, ask_wall={ask_wall}, bid_density={bid_density}, ask_density={ask_density}, RSI={rsi})"
            )
            return False

#    PARTIAL New Super-GK ONLY Liquidity (Density) & OrderBookWall
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
        
        # LONG: bid_wall > ask_wall AND bid_density > ask_density
#        if signal_direction == "LONG" and bid_wall > ask_wall and bid_density > ask_density:
#            print(f"Signal passed for LONG: bid_wall & bid_density stronger.")
#            return True
        # SHORT: ask_wall > bid_wall AND ask_density > bid_density
#        elif signal_direction == "SHORT" and ask_wall > bid_wall and ask_density > bid_density:
#            print(f"Signal passed for SHORT: ask_wall & ask_density stronger.")
#            return True
#        else:
#            print(f"Signal blocked: SuperGK not aligned for {self.symbol}")
#            return False
            
#    COMPLETE New Super-GK ADX, ATR, RSI, Liquidity (Density) & OrderBookWall  
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
        if self.df.empty:
            print(f"[{self.symbol}] Error: DataFrame empty.")
            return None

        checks = {
            "Fractal Zone": self._check_fractal_zone,
            "EMA Cloud": self._check_ema_cloud,
            "MACD": self._check_macd,
            "Momentum": self._check_momentum,
            "HATS": self._check_hats,
            "Volume Spike": self.volume_surge_confirmed if self.tf == "3min" else self._check_volume_spike,
            "VWAP Divergence": self._check_vwap_divergence,
            "MTF Volume Agreement": self._check_mtf_volume_agreement,
            "HH/LL Trend": self._check_hh_ll,
            "EMA Structure": self._check_ema_structure,
            "Chop Zone": self._check_chop_zone,
            "Candle Confirmation": self._check_candle_close,
            "Wick Dominance": self._check_wick_dominance,
            "Absorption": self._check_absorption,
            "Support/Resistance": self._check_support_resistance,
            "Smart Money Bias": self._check_smart_money_bias,
            "Liquidity Pool": self._check_liquidity_pool,
            "Spread Filter": self._check_spread_filter,
            "Liquidity Awareness": self._check_liquidity_awareness,
            "Trend Continuation": self._check_trend_continuation,
            "Volatility Model": self._check_volatility_model,
            "ATR Momentum Burst": self._check_atr_momentum_burst,
            "Volatility Squeeze": self._check_volatility_squeeze
        }

        # Run all filters just once (no direction-aware logic)
        results = {}

        for name, fn in checks.items():
            try:
                results[name] = bool(fn())
            except Exception as e:
                print(f"[{self.symbol}] {name} ERROR: {e}")
                results[name] = False

        # For debug/export compatibility, build results_long and results_short as copies
        results_long = dict(results)
        results_short = dict(results)

        # Compute total scores and gatekeeper passes for each direction
        score_long = sum(results_long.values())
        score_short = sum(results_short.values())

        passed_gk_long = [f for f in self.gatekeepers if results_long.get(f, False)]
        passed_gk_short = [f for f in self.gatekeepers if results_short.get(f, False)]
        passes_long = len(passed_gk_long)
        passes_short = len(passed_gk_short)

        # Calculate weighted scores and confidence per direction
        total_gk_weight_long = sum(self.filter_weights_long.get(f, 0) for f in self.gatekeepers)
        total_gk_weight_short = sum(self.filter_weights_short.get(f, 0) for f in self.gatekeepers)
        passed_weight_long = sum(self.filter_weights_long.get(f, 0) for f in passed_gk_long)
        passed_weight_short = sum(self.filter_weights_short.get(f, 0) for f in passed_gk_short)
        confidence_long = round(self._safe_divide(100 * passed_weight_long, total_gk_weight_long), 1) if total_gk_weight_long else 0.0
        confidence_short = round(self._safe_divide(100 * passed_weight_short, total_gk_weight_short), 1) if total_gk_weight_short else 0.0

        # Determine which direction wins (using your preferred logic)
        signal_direction = self.get_signal_direction(results_long, results_short)
        self.bias = signal_direction

        # Select summary stats based on direction
        if signal_direction == "LONG":
            score = score_long
            passes = passes_long
            confidence = confidence_long
            passed_weight = passed_weight_long
            total_gk_weight = total_gk_weight_long
        elif signal_direction == "SHORT":
            score = score_short
            passes = passes_short
            confidence = confidence_short
            passed_weight = passed_weight_short
            total_gk_weight = total_gk_weight_short
        else:
            score = max(score_long, score_short)
            passes = max(passes_long, passes_short)
            confidence = max(confidence_long, confidence_short)
            passed_weight = max(passed_weight_long, passed_weight_short)
            total_gk_weight = max(total_gk_weight_long, total_gk_weight_short)

        confidence = round(self._safe_divide(100 * passed_weight, total_gk_weight), 1) if total_gk_weight else 0.0

        orderbook_result = get_order_wall_delta(self.symbol)
        density_result = get_resting_density(self.symbol)
        super_gk_ok = self.superGK_check(signal_direction, orderbook_result, density_result)
#        super_gk_ok = self.superGK_check(signal_direction)
#        orderbook_ok = self._order_book_wall_passed()
#        resting_density_ok = self._resting_order_density_passed()

        # --- Direction logic: use only PASSED filter weights for each side ---
        signal_direction = self.get_signal_direction(results_long, results_short)

        valid_signal = (
            signal_direction in ["LONG", "SHORT"]
            and score >= self.min_score
            and passes >= self.required_passed
            and super_gk_ok
#            and orderbook_ok
#            and resting_density_ok
        )

        price = self.df['close'].iat[-1] if valid_signal else None
        price_str = f"{price:.6f}" if price is not None else "N/A"
        message = (
            f"{signal_direction or 'NO-SIGNAL'} on {self.symbol} @ {price_str} "
            f"| Score: {score}/23 | Passed: {passes}/{len(self.gatekeepers)} "
            f"| Confidence: {confidence}% (Weighted: {passed_weight:.1f}/{total_gk_weight:.1f})"
        )

        if valid_signal:
            print(f"[{self.symbol}] ✅ FINAL SIGNAL: {message}")
        else:
            print(f"[{self.symbol}] ❌ No signal.")

        print("DEBUG SUMS:", getattr(self, '_debug_sums', {}))
        
        return {
            "symbol": self.symbol,
            "tf": self.tf,
            "score": score,
            "score_max": 23,
            "passes": passes,
            "gatekeepers_total": len(self.gatekeepers),
            "passed_weight": round(passed_weight, 1),
            "total_weight": round(total_gk_weight, 1),
            "confidence": confidence,
            "bias": signal_direction,
            "price": price,
            "valid_signal": valid_signal,
            "message": message,
            "filter_results": results,

            # --- DEBUG SUMS FOR SIGNAL DECISION ---
            "debug_sums": getattr(self, '_debug_sums', {}),

            # --- PATCH FOR DEBUG COMPATIBILITY ---
            "results": results,  # legacy for signal_debug_log.py
            "filter_results_long": results,  # not truly per-direction, but fills debug
            "filter_results_short": results, # not truly per-direction, but fills debug
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
        if result in ["LONG", "SHORT"] and self._check_5m_volume_trend():
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
    
        # LONG signal: volume spike + price rising + volume rising
        long_conditions = [zscore > zscore_threshold, price_up, vol_up]
        long_met = sum(long_conditions)
    
        # SHORT signal: volume spike + price falling + volume rising
        short_conditions = [zscore > zscore_threshold, price_down, vol_up]
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

