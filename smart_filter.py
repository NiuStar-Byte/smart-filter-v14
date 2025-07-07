import requests
import pandas as pd
import numpy as np

class SmartFilter:
    """
    Core scanner that evaluates 23+ technical / order-flow filters,
    then decides whether a valid LONG / SHORT signal exists,
    using pure directional logic from 4 filters per side (June 2025 Golden Rules).
    """

    def __init__(
        self,
        symbol: str,
        df: pd.DataFrame,
        df3m: pd.DataFrame = None,
        df5m: pd.DataFrame = None,
        tf: str = None,
        min_score: int = 12,
        required_passed: int = 10,      # NEW: now 10 (for 17 gatekeepers)
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
            "MACD": 5.0, "Volume Spike": 4.8, "Fractal Zone": 4.5, "EMA Cloud": 5.0, "Momentum": 4.0, "ATR Momentum Burst": 3.8,
            "MTF Volume Agreement": 3.8, "Trend Continuation": 4.7, "HATS": 4.2, "HH/LL Trend": 4.5, "Volatility Model": 3.4,
            "EMA Structure": 3.3, "Liquidity Awareness": 3.2, "Volatility Squeeze": 3.1, "Candle Confirmation": 3.8,
            "VWAP Divergence": 2.8, "Spread Filter": 2.7, "Chop Zone": 2.6, "Liquidity Pool": 2.5, "Support/Resistance": 2.0,
            "Smart Money Bias": 1.9, "Absorption": 1.7, "Wick Dominance": 1.5
        }

        self.filter_weights_short = {
            "MACD": 3.0, "Volume Spike": 4.8, "Fractal Zone": 4.5, "EMA Cloud": 5.0, "Momentum": 4.0, "ATR Momentum Burst": 4.3,
            "MTF Volume Agreement": 3.8, "Trend Continuation": 4.7, "HATS": 4.6, "HH/LL Trend": 4.5, "Volatility Model": 3.4,
            "EMA Structure": 3.3, "Liquidity Awareness": 3.6, "Volatility Squeeze": 3.1, "Candle Confirmation": 3.8,
            "VWAP Divergence": 3.0, "Spread Filter": 2.7, "Chop Zone": 2.6, "Liquidity Pool": 2.5, "Support/Resistance": 2.1,
            "Smart Money Bias": 2.0, "Absorption": 2.0, "Wick Dominance": 1.5
        }

        self.gatekeepers = [
            "Fractal Zone", "EMA Cloud", "MACD", "Momentum", "HATS",
            "Volume Spike", "VWAP Divergence", "MTF Volume Agreement",
            "HH/LL Trend", "EMA Structure", "Chop Zone",
            "Candle Confirmation", "Trend Continuation",
            "Volatility Model", "Liquidity Awareness",
            "ATR Momentum Burst", "Volatility Squeeze"
        ]

        self.df["ema20"] = self.df["close"].ewm(span=20).mean()
        self.df["ema50"] = self.df["close"].ewm(span=50).mean()
        self.df["ema200"] = self.df["close"].ewm(span=200).mean()
        self.df["vwap"] = (self.df["close"] * self.df["volume"]).cumsum() / self.df["volume"].cumsum()

    def get_signal_direction(self, results):
        """
        Calculate the sum of weights for all PASSED filters for both LONG and SHORT,
        and return "LONG", "SHORT" or "NEUTRAL".
        """
        long_sum = sum(
            self.filter_weights_long[f] for f, v in results.items() if v and f in self.filter_weights_long
        )
        short_sum = sum(
            self.filter_weights_short[f] for f, v in results.items() if v and f in self.filter_weights_short
        )
        if long_sum > short_sum:
            return "LONG"
        elif short_sum > long_sum:
            return "SHORT"
        else:
            return "NEUTRAL"

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

        results = {}
        for name, fn in checks.items():
            try:
                results[name] = bool(fn())
            except Exception as e:
                print(f"[{self.symbol}] {name} ERROR: {e}")
                results[name] = False

        score = sum(results.values())
        passed_gk = [f for f in self.gatekeepers if results.get(f, False)]
        passes = len(passed_gk)
        total_gk_weight = sum(self.filter_weights_long[f] for f in self.gatekeepers)
        passed_weight = sum(self.filter_weights_long[f] for f in passed_gk)
        confidence = round(self._safe_divide(100 * passed_weight, total_gk_weight), 1)

        orderbook_ok = self._order_book_wall_passed()
        resting_density_ok = self._resting_order_density_passed()

        # --- Direction logic: use only PASSED filter weights for each side ---
        signal_direction = self.get_signal_direction(results)

        valid_signal = (
            signal_direction in ["LONG", "SHORT"]
            and score >= self.min_score
            and passes >= self.required_passed
            and orderbook_ok
            and resting_density_ok
        )

        price = self.df['close'].iat[-1] if valid_signal else None

        message = (
            f"{signal_direction or 'NO-SIGNAL'} on {self.symbol} @ {price:.6f} "
            f"| Score: {score}/23 | Passed: {passes}/{len(self.gatekeepers)} "
            f"| Confidence: {confidence}% (Weighted: {passed_weight:.1f}/{total_gk_weight:.1f})"
        )

        if valid_signal:
            print(f"[{self.symbol}] ✅ FINAL SIGNAL: {message}")
        else:
            print(f"[{self.symbol}] ❌ No signal.")

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
            "filter_results": results
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
