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
        required_passed: int = 7,
        volume_multiplier: float = 2.0
    ):
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
            "Support/Resistance": 2.4,
            "Smart Money Bias": 1.8,
            "Liquidity Pool": 2.0,
            "Spread Filter": 2.2
        }

    def analyze(self):
        if self.df.empty:
            print(f"[{self.symbol}] Error: DataFrame empty.")
            return None

        results = {}
        filters = {
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
        }

        for name, fn in filters.items():
            try:
                result = fn()
                results[name] = bool(result)
            except Exception as e:
                print(f"[{self.symbol}] {name} ERROR: {e}")
                results[name] = False

        passed_filters = [k for k, v in results.items() if v]
        raw_score = len(passed_filters)

        required_keys = list(results.keys())[:12]
        passed_req = sum(1 for k in required_keys if results[k])

        passed_required = [k for k in required_keys if results[k]]
        weighted_score = sum(self.filter_weights[k] for k in passed_required)
        max_possible_score = sum(self.filter_weights[k] for k in required_keys)
        confidence = round(100 * weighted_score / max_possible_score, 1)

        print(f"[{self.symbol}] Score: {raw_score}/18 | Passed Required: {passed_req}/12 | Confidence: {confidence}%")
        for name in filters.keys():
            status = '✅' if results[name] else '❌'
            weight = self.filter_weights.get(name, 0)
            print(f"{name:20} -> {status}  ({weight})")

        if raw_score >= self.min_score and passed_req >= self.required_passed:
            price = self.df['close'].iat[-1]
            bias = "LONG" if price > self.df['open'].iat[-1] else "SHORT"
            signal = (
                f"{bias} signal on {self.symbol} @ {price} | Confidence: {confidence}% (Weighted: {round(weighted_score,1)}/{max_possible_score})",
                self.symbol, bias, price, self.tf,
                f"{raw_score}/18", f"{passed_req}/12"
            )
            print(f"[{self.symbol}] ✅ FINAL SIGNAL: {signal[0]}")
            return signal

        print(f"[{self.symbol}] ❌ No signal: thresholds not met.")
        failing_filters = [k for k, v in results.items() if not v]
        top_fails = "\n".join(f"- {name}" for name in failing_filters[:5])
        print(f"❗ Top Failing Filters for {self.symbol}:\n{top_fails}")
        return None

    def volume_surge_confirmed(self):
        spike = self._check_volume_spike()
        trend = self._check_5m_volume_trend()
        print(f"[{self.symbol}] Volume Spike: {spike}, 5m Volume Trend: {trend}")
        return spike and trend

    def _check_volume_spike(self):
        avg = self.df['volume'].rolling(10).mean().iat[-1]
        return self.df['volume'].iat[-1] > self.volume_multiplier * avg

    def _check_5m_volume_trend(self):
        if self.df5m is None or len(self.df5m) < 3:
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
        ha = self.df[['open', 'high', 'low', 'close']].mean(axis=1)
        return ha.iat[-1] > ha.iat[-2]

    def _check_vwap_divergence(self):
        diff = self.df['close'].iat[-1] - self.df['vwap'].iat[-1]
        pct = diff / self.df['vwap'].iat[-1]
        return diff > 0 and pct > 0.001

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
        cond = self.df['ema20'].iat[-1] > self.df['ema50'].iat[-1] > self.df['ema200'].iat[-1]
        slope = all(
            self.df[f'ema{x}'].iat[-1] > self.df[f'ema{x}'].iat[-2]
            for x in (20, 50, 200)
        )
        return cond and slope

    def _check_chop_zone(self):
        return (self.df['close'].diff() > 0).sum() > 7

    def _check_candle_close(self):
        body = abs(self.df['close'].iat[-1] - self.df['open'].iat[-1])
        rng = self.df['high'].iat[-1] - self.df['low'].iat[-1]
        return body > 0.5 * rng

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
        signed = self.df['volume'] * self.df['close'].diff().apply(lambda x: 1 if x > 0 else -1)
        recent = signed.iloc[-14:]
        return recent.sum() > 0

    def _check_liquidity_pool(self):
        hi = self.df['high'].rolling(10).max().iat[-2]
        lo = self.df['low'].rolling(10).min().iat[-2]
        return self.df['high'].iat[-1] > hi or self.df['low'].iat[-1] < lo

    def _check_spread_filter(self):
        spread = self.df['high'].iat[-1] - self.df['low'].iat[-1]
        return spread < 0.02 * self.df['close'].iat[-1]
