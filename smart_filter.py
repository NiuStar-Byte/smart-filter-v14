import pandas as pd

# === SMART FILTER V18 (CONTROL GROUP) ===
class SmartFilterV18:
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
            "Support/Resistance": 1.9,
            "Smart Money Bias": 1.8,
            "Liquidity Pool": 2.5,
            "Spread Filter": 2.7
        }

        self.top_filters = [
            "Fractal Zone", "EMA Cloud", "MACD", "Momentum", "HATS", "Volume Spike",
            "VWAP Divergence", "MTF Volume Agreement", "HH/LL Trend", "EMA Structure",
            "Chop Zone", "Candle Confirmation"
        ]

    def analyze(self):
        return self._run_filters(version="V18")

    def _run_filters(self, version="Vxx"):
        if self.df.empty:
            print(f"[{self.symbol}] Error: DataFrame empty.")
            return None

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
            "Spread Filter": self._check_spread_filter
        }

        if version == "V19" and hasattr(self, "_check_trend_continuation"):
            filters["Trend Continuation"] = self._check_trend_continuation

        results = {}
        for name, fn in filters.items():
            try:
                results[name] = bool(fn())
            except Exception as e:
                print(f"[{self.symbol}] {name} ERROR: {e}")
                results[name] = False

        passed_all = [k for k, v in results.items() if v]
        score = len(passed_all)

        passed_top = [k for k in self.top_filters if results.get(k, False)]
        passed_count = len(passed_top)
        top_weight_sum = sum(self.filter_weights[k] for k in self.top_filters)
        passed_weight_sum = sum(self.filter_weights[k] for k in passed_top)
        confidence = round(min(99.9, 100 * passed_weight_sum / top_weight_sum), 1)

        print(f"[{self.symbol}] Score: {score}/{len(filters)} | Passed Top Filters: {passed_count}/{len(self.top_filters)} | Confidence: {confidence}%")
        for name in filters:
            status = "✅" if results[name] else "❌"
            print(f"{name:20} -> {status} ({self.filter_weights.get(name, 0)})")

        if score >= self.min_score and passed_count >= self.required_passed:
            price = self.df['close'].iat[-1]
            bias = "LONG" if price > self.df['open'].iat[-1] else "SHORT"
            signal = (
                f"{bias} signal on {self.symbol} @ {price} | Confidence: {confidence}% (Weighted: {round(passed_weight_sum,1)}/{top_weight_sum})",
                self.symbol, bias, price, self.tf,
                f"{score}/{len(filters)}", f"{passed_count}/{len(self.top_filters)}"
            )
            print(f"[{self.symbol}] ✅ FINAL SIGNAL: {signal[0]}")
            return signal

        print(f"[{self.symbol}] ❌ No signal: thresholds not met.")
        return None

    # === Filter Logic Below ===
    def _check_fractal_zone(self):
        return self.df['close'].iat[-1] > self.df['ema50'].iat[-1]

    def _check_ema_cloud(self):
        return self.df['ema20'].iat[-1] > self.df['ema50'].iat[-1] > self.df['ema200'].iat[-1]

    def _check_macd(self):
        return True

    def _check_momentum(self):
        return True

    def _check_hats(self):
        return True

    def _check_volume_spike(self):
        return self.df['volume'].iat[-1] > self.df['volume'].rolling(20).mean().iat[-1] * self.volume_multiplier

    def volume_surge_confirmed(self):
        return self.df['volume'].iloc[-2] > self.df['volume'].rolling(20).mean().iloc[-2] * self.volume_multiplier

    def _check_vwap_divergence(self):
        return self.df['close'].iat[-1] > self.df['vwap'].iat[-1]

    def _check_mtf_volume_agreement(self):
        if self.df3m is None or self.df5m is None:
            return False
        v3 = self.df3m['volume'].iloc[-1] > self.df3m['volume'].rolling(20).mean().iloc[-1] * 1.5
        v5 = self.df5m['volume'].iloc[-1] > self.df5m['volume'].rolling(20).mean().iloc[-1] * 1.5
        return v3 and v5

    def _check_hh_ll(self):
        return self.df['high'].iat[-1] > self.df['high'].iat[-4]

    def _check_ema_structure(self):
        return self.df['ema20'].iat[-1] > self.df['ema50'].iat[-1]

    def _check_chop_zone(self):
        return True

    def _check_candle_close(self):
        return self.df['close'].iat[-1] > self.df['open'].iat[-1]

    def _check_wick_dominance(self):
        return True

    def _check_absorption(self):
        return True

    def _check_support_resistance(self):
        return True

    def _check_smart_money_bias(self):
        return True

    def _check_liquidity_pool(self):
        return True

    def _check_spread_filter(self):
        return True

# === SMART FILTER V19 (EXPERIMENTAL VERSION WITH TREND CONTINUATION + ENHANCED LOGIC) ===
class SmartFilterV19(SmartFilterV18):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_weights["Trend Continuation"] = 3.7
        self.top_filters.append("Trend Continuation")

    def analyze(self):
        return self._run_filters(version="V19")

    def _check_trend_continuation(self):
        self.df['ema_diff'] = self.df['ema20'] - self.df['ema50']
        ema_slope = self.df['ema_diff'].diff().iat[-1] > 0

        self.df['tr'] = self.df[['high', 'low', 'close']].max(axis=1) - self.df[['high', 'low', 'close']].min(axis=1)
        self.df['dm_plus'] = self.df['high'].diff().clip(lower=0)
        self.df['dm_minus'] = -self.df['low'].diff().clip(upper=0)
        self.df['di_plus'] = 100 * self.df['dm_plus'].ewm(span=14).mean() / self.df['tr'].ewm(span=14).mean()
        self.df['di_minus'] = 100 * self.df['dm_minus'].ewm(span=14).mean() / self.df['tr'].ewm(span=14).mean()
        self.df['dx'] = (abs(self.df['di_plus'] - self.df['di_minus']) / (self.df['di_plus'] + self.df['di_minus'])) * 100
        adx = self.df['dx'].ewm(span=14).mean().iat[-1]

        return ema_slope and adx > 20
