import requests
import pandas as pd
import numpy as np

def compute_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iat[-1]

def compute_rsi(df, period=14):
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs)).iat[-1]

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
        min_score: int = 16,
        required_passed: int = 12,      # NEW: now 12 (for 17 gatekeepers)
        volume_multiplier: float = 2.0,
        kwargs = None
    ):
        if kwargs is None:
            kwargs = {}

        self.symbol = symbol
        self.df = df.copy()
        self.df3m = df3m.copy() if df3m is not None else None
        self.df5m = df5m.copy() if df5m is not None else None
        self.tf = tf
        self.min_score = min_score
        self.required_passed = required_passed
        self.volume_multiplier = volume_multiplier

        # Weights for filters
        self.filter_weights_long = {
            "MACD": 4.3, "Volume Spike": 4.6, "Fractal Zone": 4.4, "EMA Cloud": 4.9, "Momentum": 3.8, "ATR Momentum Burst": 3.9,
            "MTF Volume Agreement": 3.8, "Trend Continuation": 4.7, "HATS": 4.6, "HH/LL Trend": 4.5, "Volatility Model": 3.4,
            "EMA Structure": 3.3, "Liquidity Awareness": 3.2, "Volatility Squeeze": 3.1, "Candle Confirmation": 3.8,
            "VWAP Divergence": 2.9, "Spread Filter": 2.7, "Chop Zone": 2.6, "Liquidity Pool": 2.5, "Support/Resistance": 2.4,
            "Smart Money Bias": 2.2, "Absorption": 2.1, "Wick Dominance": 1.9
        }

        self.filter_weights_short = {
            "MACD": 4.1, "Volume Spike": 4.8, "Fractal Zone": 4.6, "EMA Cloud": 4.7, "Momentum": 3.7, "ATR Momentum Burst": 4.1,
            "MTF Volume Agreement": 3.6, "Trend Continuation": 4.5, "HATS": 4.4, "HH/LL Trend": 4.3, "Volatility Model": 3.5,
            "EMA Structure": 3.1, "Liquidity Awareness": 3.4, "Volatility Squeeze": 2.9, "Candle Confirmation": 3.8,
            "VWAP Divergence": 2.7, "Spread Filter": 2.9, "Chop Zone": 2.8, "Liquidity Pool": 2.7, "Support/Resistance": 2.6,
            "Smart Money Bias": 2.4, "Absorption": 1.9, "Wick Dominance": 2.1
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

    def superGK_check(self, signal_direction):
        # Fetch order book data (Bid/Ask details)
        bids, asks = self._fetch_order_book(self.symbol)

        if bids is None or asks is None:
            print(f"Signal blocked due to missing order book data for {self.symbol}")
            return False

        # Calculate bid and ask wall sizes (order wall delta)
        bid_wall = self.get_order_wall_delta(self.symbol, side='bid')
        ask_wall = self.get_order_wall_delta(self.symbol, side='ask')

        # Check market liquidity (Resting Density)
        resting_density = self.get_resting_density(self.symbol)
        bid_density = resting_density['bid_density']
        ask_density = resting_density['ask_density']

        if bid_density < self.liquidity_threshold or ask_density < self.liquidity_threshold:
            print(f"Signal blocked due to low liquidity for {self.symbol}")
            return False

        if bid_wall < ask_wall:
            print(f"Signal blocked due to weak buy-side support for {self.symbol}")
            return False

        # --- ATR & RSI Calculations ---
        atr = compute_atr(self.df)
        rsi = compute_rsi(self.df)
        price = self.df['close'].iat[-1]
        atr_pct = atr / price if price else 0

        # --- Dynamic Market Regime Logic ---
        low_vol_threshold = 0.01   # ATR < 1% of price = low volatility
        high_vol_threshold = 0.03  # ATR > 3% of price = high volatility
        bull_rsi = 60
        bear_rsi = 40

        if atr_pct < low_vol_threshold:
            print(f"Signal blocked: Volatility too low (ATR={atr:.4f}, ATR%={atr_pct:.2%})")
            return False

        if signal_direction == "LONG":
            if rsi < bull_rsi:
                print(f"Signal blocked: LONG but RSI not bullish enough (RSI={rsi:.2f})")
                return False
        elif signal_direction == "SHORT":
            if rsi > bear_rsi:
                print(f"Signal blocked: SHORT but RSI not bearish enough (RSI={rsi:.2f})")
                return False

        if bear_rsi < rsi < bull_rsi:
            print(f"Signal blocked: Market is ranging (RSI={rsi:.2f})")
            return False

        if atr_pct > high_vol_threshold:
            print(f"Signal blocked: Volatility extremely high, unstable market (ATR={atr:.4f}, ATR%={atr_pct:.2%})")
            return False

        if bid_density > ask_density:
            print(f"Signal passed for LONG: Strong bid-side liquidity for {self.symbol}")
            return True
        elif ask_density > bid_density:
            print(f"Signal passed for SHORT: Strong ask-side liquidity for {self.symbol}")
            return True
        else:
            print(f"Signal blocked due to neutral market conditions for {self.symbol}")
            return False
    
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

        super_gk_ok = self.superGK_check(signal_direction)
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
        return self._check_volume_spike() and self._check_5m_volume_trend()

    def _check_volume_spike(self):
        avg = self.df['volume'].rolling(10).mean().iat[-1]
        std = self.df['volume'].rolling(10).std().iat[-1]
        zscore = self._safe_divide(self.df['volume'].iat[-1] - avg, std)
        return zscore > 1.5

    def _check_5m_volume_trend(self):
        if self.df5m is None or len(self.df5m) < 2:
            return False
        return self.df5m['volume'].iat[-1] > self.df5m['volume'].iat[-2]

    def _check_fractal_zone(self, buffer_pct=0.005):
        fractal_low = self.df['low'].rolling(20).min().iat[-1]
        return self.df['close'].iat[-1] > fractal_low * (1 + buffer_pct)
    
    def _check_ema_cloud(self):
        return (
            self.df['ema20'].iat[-1] > self.df['ema50'].iat[-1] and
            self.df['ema20'].iat[-1] > self.df['ema20'].iat[-2]
        )

    def _check_macd(self):
        e12 = self.df['close'].ewm(span=12).mean()
        e26 = self.df['close'].ewm(span=26).mean()
        macd = e12 - e26
        signal = macd.ewm(span=9).mean()
        return macd.iat[-1] > signal.iat[-1] and macd.iat[-1] > macd.iat[-2]

    def _check_momentum(self, roc_window=3):
        return self.df['close'].pct_change(roc_window).iat[-1] > 0

    def _check_hats(self):
        ha_close = (self.df['open'] + self.df['high'] + self.df['low'] + self.df['close']) / 4
        return ha_close.iat[-1] > ha_close.iat[-2]

    def _check_vwap_divergence(self):
        diff = self.df['close'].iat[-1] - self.df['vwap'].iat[-1]
        return diff > 0 and diff / self.df['vwap'].iat[-1] > 0.001 and self.df['volume'].iat[-1] > self.df['volume'].rolling(10).mean().iat[-1]

    def _check_mtf_volume_agreement(self):
        if self.df3m is None or self.df5m is None:
            return False

        def zscore(vols):
            mean = vols.rolling(20).mean().iat[-1]
            std = vols.rolling(20).std().iat[-1]
            return self._safe_divide(vols.iat[-1] - mean, std)

        return zscore(self.df3m['volume']) > 1.2 and zscore(self.df5m['volume']) > 1.2

    def _check_hh_ll(self):
        return (
            self.df['high'].iat[-1] > self.df['high'].iat[-3] and
            self.df['low'].iat[-1] > self.df['low'].iat[-3]
        )

    def _check_ema_structure(self):
        e20, e50, e200 = self.df['ema20'].iat[-1], self.df['ema50'].iat[-1], self.df['ema200'].iat[-1]
        slope20 = e20 > self.df['ema20'].iat[-3]
        slope50 = e50 > self.df['ema50'].iat[-3]
        slope200 = e200 > self.df['ema200'].iat[-3]
        return e20 > e50 > e200 and slope20 and slope50

    def _check_chop_zone(self):
        return (self.df['close'].diff() > 0).sum() > 7

    def _check_candle_close(self):
        body = abs(self.df['close'].iat[-1] - self.df['open'].iat[-1])
        rng = self.df['high'].iat[-1] - self.df['low'].iat[-1]
        avg_body = abs(self.df['close'] - self.df['open']).rolling(5).mean().iat[-1]
        return body > 0.5 * rng and body > avg_body

    def _check_wick_dominance(self):
        body = abs(self.df['close'].iat[-1] - self.df['open'].iat[-1])
        rng = self.df['high'].iat[-1] - self.df['low'].iat[-1]
        lower = min(self.df['close'].iat[-1], self.df['open'].iat[-1]) - self.df['low'].iat[-1]
        return body > 0.6 * rng and lower < 0.2 * rng

    def _check_absorption(self):
        low_under = self.df['low'].iat[-1] < self.df['open'].iat[-1]
        close_up = self.df['close'].iat[-1] > self.df['open'].iat[-1]
        vol_spike = self.df['volume'].iat[-1] > self.volume_multiplier * self.df['volume'].rolling(10).mean().iat[-1]
        return low_under and close_up and vol_spike

    def _check_support_resistance(self):
        swing = self.df['low'].rolling(5).min().iat[-3]
        return abs(self.df['close'].iat[-1] - swing) / swing < 0.01

    def _check_smart_money_bias(self):
        last = self.df['close'].iat[-1]
        open_ = self.df['open'].iat[-1]
        vwap = self.df['vwap'].iat[-1]
        return last > open_ and last > vwap

    def _check_spread_filter(self):
        spread = self.df['high'].iat[-1] - self.df['low'].iat[-1]
        return spread < 0.02 * self.df['close'].iat[-1]

    def _check_liquidity_pool(self):
        bids, asks = self._fetch_order_book(depth=100)
        return bids is not None and asks is not None

    def _check_liquidity_awareness(self, band_pct=0.005, wall_factor=5, history_len=20):
        bids, asks = self._fetch_order_book(depth=100)
        if bids is None or asks is None:
            return True
        mid = (bids['price'].iloc[0] + asks['price'].iloc[0]) / 2
        low, high = mid * (1 - band_pct), mid * (1 + band_pct)
        bid_depth = bids[bids['price'] >= low]['size'].sum()
        ask_depth = asks[asks['price'] <= high]['size'].sum()
        last = self.df['close'].iat[-1]
        opp = ask_depth if last > self.df['open'].iat[-1] else bid_depth

        if not hasattr(self, '_depth_history'):
            self._depth_history = []
        self._depth_history.append(opp)
        if len(self._depth_history) > history_len:
            self._depth_history.pop(0)

        avg = sum(self._depth_history) / len(self._depth_history)
        return self._safe_divide(opp, avg) < wall_factor

    def _check_trend_continuation(self):
        slope = self.df['ema20'].iat[-1] - self.df['ema20'].iat[-3]
        return slope > 0

    def _check_volatility_model(self, low_pct=0.01, high_pct=0.05, period=14):
        tr = pd.concat([
            self.df['high'] - self.df['low'],
            (self.df['high'] - self.df['close'].shift()).abs(),
            (self.df['low'] - self.df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iat[-1]
        atr_pct = self._safe_divide(atr, self.df['close'].iat[-1])
        return low_pct <= atr_pct <= high_pct

    def _check_atr_momentum_burst(self, atr_period=14, burst_threshold=1.5):
        tr = pd.concat([
            self.df['high'] - self.df['low'],
            (self.df['high'] - self.df['close'].shift()).abs(),
            (self.df['low'] - self.df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        momentum = self.df['close'].diff()
        burst = self._safe_divide(momentum.abs().iat[-1], atr.iat[-1])
        return burst > burst_threshold

    def _check_volatility_squeeze(self, window=20):
        bb_high = self.df['close'].rolling(window).max()
        bb_low = self.df['close'].rolling(window).min()
        bb_width = (bb_high - bb_low) / self.df['close']
        return bb_width.iat[-1] < 0.02

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

