import requests
import pandas as pd

class SmartFilter:
    def __init__(
        self,
        symbol: str,
        df: pd.DataFrame,
        df3m: pd.DataFrame = None,
        df5m: pd.DataFrame = None,
        tf: str = None,
        min_score: int = 9,
        required_passed: int = 8,
        volume_multiplier: float = 2.0
    ):
        # Initialize state and compute EMAs/VWAP
        self.symbol = symbol
        self.df = df.copy()
        self.df3m = df3m.copy() if df3m is not None else None
        self.df5m = df5m.copy() if df5m is not None else None
        self.tf = tf
        self.min_score = min_score
        self.required_passed = required_passed
        self.volume_multiplier = volume_multiplier

        self.df["ema20"] = self.df["close"].ewm(span=20).mean()
        self.df["ema50"] = self.df["close"].ewm(span=50).mean()
        self.df["ema200"] = self.df["close"].ewm(span=200).mean()
        self.df["vwap"] = (self.df["close"] * self.df["volume"]).cumsum() / self.df["volume"].cumsum()

        # Weights for all filters (now including Liquidity Awareness)
        self.filter_weights = {
            "Fractal Zone": 4.5,
            "EMA Cloud": 4.2,
            "MACD": 5.0,
            "Momentum": 4.0,
            "HATS": 3.6,
            "Volume Spike": 4.8,
            "VWAP Divergence": 2.8,
            "MTF Volume Agreement": 3.8,
            "HH/LL Trend": 3.5,
            "EMA Structure": 3.3,
            "Chop Zone": 2.6,
            "Candle Confirmation": 3.0,
            "Wick Dominance": 1.2,
            "Absorption": 1.5,
            "Support/Resistance": 1.9,
            "Smart Money Bias": 1.8,
            "Liquidity Pool": 2.5,
            "Spread Filter": 2.7,
            # New Liquidity Awareness filter added
            "Liquidity Awareness": 3.2,
            "Trend Continuation": 3.7
        }

        # Top filters used for passed_count & confidence
        self.top_filters = [
            "Fractal Zone", "EMA Cloud", "MACD", "Momentum", "HATS",
            "Volume Spike", "VWAP Divergence", "MTF Volume Agreement",
            "HH/LL Trend", "EMA Structure", "Chop Zone",
            "Candle Confirmation", "Trend Continuation"
        ]

    def analyze(self):
        if self.df.empty:
            print(f"[{self.symbol}] Error: DataFrame empty.")
            return None

        # Run every check
        results = {}
        checks = {
            "Fractal Zone": self._check_fractal_zone,
            "EMA Cloud": self._check_ema_cloud,
            "MACD": self._check_macd,
            "Momentum": self._check_momentum,
            "HATS": self._check_hats,
            "Volume Spike": (
                self.volume_surge_confirmed if self.tf == "3min" else self._check_volume_spike
            ),
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
            # New Liquidity Awareness check
            "Liquidity Awareness": self._check_liquidity_awareness,
            "Trend Continuation": self._check_trend_continuation
        }

        for name, fn in checks.items():
            try:
                results[name] = bool(fn())
            except Exception as e:
                print(f"[{self.symbol}] {name} ERROR: {e}")
                results[name] = False

        # Total score across all filters
        passed_all = [name for name, ok in results.items() if ok]
        score = len(passed_all)

        # Passed among top filters
        passed_top = [f for f in self.top_filters if results.get(f, False)]
        passed_count = len(passed_top)

        # Confidence = passed weight / total top weight
        total_top_weight = sum(self.filter_weights[n] for n in self.top_filters)
        passed_weight = sum(self.filter_weights[n] for n in passed_top)
        confidence = round(100 * passed_weight / total_top_weight, 1)

        # Print breakdown
        print(f"[{self.symbol}] Score: {score}/{len(self.filter_weights)} | "
              f"Passed Top Filters: {passed_count}/{len(self.top_filters)} | "
              f"Confidence: {confidence}%")
        for n in checks:
            mark = '✅' if results[n] else '❌'
            print(f"{n:20} -> {mark} ({self.filter_weights[n]})")

        # Final signal decision
        if score >= self.min_score and passed_count >= self.required_passed:
            price = self.df['close'].iat[-1]
            bias = "LONG" if price > self.df['open'].iat[-1] else "SHORT"
            msg = (
                f"{bias} signal on {self.symbol} @ {price} | "
                f"Confidence: {confidence}% "
                f"(Weighted: {round(passed_weight,1)}/{total_top_weight})",
                self.symbol, bias, price, self.tf,
                f"{score}/{len(self.filter_weights)}",
                f"{passed_count}/{len(self.top_filters)}"
            )
            print(f"[{self.symbol}] ✅ FINAL SIGNAL: {msg[0]}")
            return msg

        print(f"[{self.symbol}] ❌ No signal: thresholds not met.")
        return None

    # --- Helper for Volume Surge Double-Check ---
    def volume_surge_confirmed(self):
        return self._check_volume_spike() and self._check_5m_volume_trend()

    # --- Filter implementations below ---
    def _check_volume_spike(self):
        avg = self.df['volume'].rolling(10).mean().iat[-1]
        return self.df['volume'].iat[-1] > self.volume_multiplier * avg

    def _check_5m_volume_trend(self):
        if self.df5m is None or len(self.df5m) < 2:
            return False
        return self.df5m['volume'].iat[-1] > self.df5m['volume'].iat[-2]

    def _check_fractal_zone(self):
        return self.df['close'].iat[-1] > self.df['low'].rolling(20).min().iat[-1]

    def _check_ema_cloud(self):
        return self.df['ema20'].iat[-1] > self.df['ema50'].iat[-1]

    def _check_macd(self):
        e12 = self.df['close'].ewm(span=12).mean()
        e26 = self.df['close'].ewm(span=26).mean()
        macd = e12 - e26
        sig = macd.ewm(span=9).mean()
        return macd.iat[-1] > sig.iat[-1]

    def _check_momentum(self):
        return self.df['close'].diff().iat[-1] > 0

    def _check_hats(self):
        ha = self.df[['open','high','low','close']].mean(axis=1)
        return ha.iat[-1] > ha.iat[-2]

    def _check_vwap_divergence(self):
        diff = self.df['close'].iat[-1] - self.df['vwap'].iat[-1]
        return diff > 0 and diff / self.df['vwap'].iat[-1] > 0.001

    def _check_mtf_volume_agreement(self):
        if self.df3m is None or self.df5m is None:
            return False
        v3 = self.df3m['volume'].iat[-1] > self.df3m['volume'].rolling(20).mean().iat[-1]
        v5 = self.df5m['volume'].iat[-1] > self.df5m['volume'].rolling(20).mean().iat[-1]
        return v3 and v5

    def _check_hh_ll(self):
        return (
            self.df['high'].iat[-1] > self.df['high'].iat[-3] and
            self.df['low'].iat[-1] > self.df['low'].iat[-3]
        )

    def _check_ema_structure(self):
        cond = (
            self.df['ema20'].iat[-1] > self.df['ema50'].iat[-1] > self.df['ema200'].iat[-1]
        )
        slope = all(
            self.df[f'ema{x}'].iat[-1] > self.df[f'ema{x}'].iat[-2]
            for x in (20, 50, 200)
        )
        return cond and slope

    def _check_chop_zone(self):
        return (self.df['close'].diff() > 0).sum() > 7

    def _check_candle_close(self):
        body = abs(self.df['close'].iat[-1] - self.df['open'].iat[-1])
        rng  = self.df['high'].iat[-1] - self.df['low'].iat[-1]
        return body > 0.5 * rng

    def _check_wick_dominance(self):
        body = abs(self.df['close'].iat[-1] - self.df['open'].iat[-1])
        rng  = self.df['high'].iat[-1] - self.df['low'].iat[-1]
        lower = (
            min(self.df['close'].iat[-1], self.df['open'].iat[-1])
            - self.df['low'].iat[-1]
        )
        return body > 0.6 * rng and lower < 0.2 * rng

    def _check_absorption(self):
        low_under = self.df['low'].iat[-1] < self.df['open'].iat[-1]
        close_up  = self.df['close'].iat[-1] > self.df['open'].iat[-1]
        vol_spike = self.df['volume'].iat[-1] > self.volume_multiplier * self.df['volume'].rolling(10).mean().iat[-1]
        return low_under and close_up and vol_spike

    def _check_support_resistance(self):
        swing = self.df['low'].rolling(5).min().iat[-3]
        return abs(self.df['close'].iat[-1] - swing) / swing < 0.01

    def _check_smart_money_bias(self):
        signed = self.df['volume'] * self.df['close'].diff().apply(lambda x: 1 if x > 0 else -1)
        return signed.iloc[-14:].sum() > 0

    def _check_liquidity_pool(self):
        hi = self.df['high'].rolling(10).max().iat[-2]
        lo = self.df['low'].rolling(10).min().iat[-2]
        return self.df['high'].iat[-1] > hi or self.df['low'].iat[-1] < lo

    def _fetch_order_book(self, depth: int = 100):
        """Helper to fetch top `depth` levels from KuCoin order book."""
        for sym in (self.symbol, self.symbol.replace('-', '/')):
            url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_{depth}?symbol={sym}"
            try:
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json().get('data', {})
                bids = pd.DataFrame(data.get('bids', []), columns=['price','size']).astype(float)
                asks = pd.DataFrame(data.get('asks', []), columns=['price','size']).astype(float)
                return bids, asks
            except Exception:
                continue
        return None, None

    def _check_liquidity_awareness(self, band_pct: float = 0.005, wall_factor: float = 5, history_len: int = 20):
        """
        Block signals if there's a large opposing wall within ±band_pct of mid-price.
        """
        bids, asks = self._fetch_order_book(depth=100)
        if bids is None or asks is None:
            return True
        mid = (bids['price'].iloc[0] + asks['price'].iloc[0]) / 2
        low, high = mid * (1 - band_pct), mid * (1 + band_pct)
        bid_depth = bids[bids['price'] >= low]['size'].sum()
        ask_depth = asks[asks['price'] <= high]['size'].sum()
        last = self.df['close'].iat[-1]
        long = last > self.df['open'].iat[-1]
        opp = ask_depth if long else bid_depth
        # rolling history of opposing depth
        if not hasattr(self, '_depth_history'):
            self._depth_history = []
        self._depth_history.append(opp)
        if len(self._depth_history) > history_len:
            self._depth_history.pop(0)
        avg = sum(self._depth_history) / len(self._depth_history)
        # allow only if wall < wall_factor * average
        return opp / avg < wall_factor
