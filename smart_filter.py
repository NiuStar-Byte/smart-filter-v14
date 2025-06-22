import pandas as pd

USE_VOLUME_PREFILTER = True  # Toggle to disable during experiments
VOLUME_ACCELERATION_THRESHOLD = 1.2  # 5m volume must be 20% higher than average

class SmartFilter:
    """
    SmartFilter v14.1 — Applies 18 technical filters + optional 5m volume pre-check on 3m timeframe.
    """

    def __init__(
        self,
        symbol: str,
        df: pd.DataFrame,
        df3m: pd.DataFrame = None,
        df5m: pd.DataFrame = None,
        tf: str = None,
        min_score: int = 9,
        required_passed: int = 7
    ):
        self.symbol = symbol
        self.df = df.copy()
        self.df3m = df3m.copy() if df3m is not None else None
        self.df5m = df5m.copy() if df5m is not None else None
        self.tf = tf
        self.min_score = min_score
        self.required_passed = required_passed

        # Pre-calculate indicators
        self.df["ema20"] = self.df["close"].ewm(span=20).mean()
        self.df["ema50"] = self.df["close"].ewm(span=50).mean()
        self.df["ema200"] = self.df["close"].ewm(span=200).mean()
        self.df["vwap"] = (self.df["close"] * self.df["volume"]).cumsum() / self.df["volume"].cumsum()

    def _volume_precheck_5m(self):
        if self.df5m is None or len(self.df5m) < 4:
            print(f"[DEBUG] df5m is None or too short — cannot check 5m volume.")
            return False
        current_vol = self.df5m['volume'].iat[-1]
        avg_vol = self.df5m['volume'].iloc[-4:-1].mean()
        print(f"[DEBUG] 5m current vol: {current_vol:.2f}, avg vol: {avg_vol:.2f}")
        return current_vol > VOLUME_ACCELERATION_THRESHOLD * avg_vol

    def analyze(self):
        if self.df.empty:
            print(f"[{self.symbol}] Error: DataFrame empty.")
            return None

        print(f"[DEBUG] tf: {self.tf}, USE_VOLUME_PREFILTER: {USE_VOLUME_PREFILTER}")
        if self.tf == '3m' and USE_VOLUME_PREFILTER:
            if not self._volume_precheck_5m():
                print(f"[{self.symbol}] ❌ BLOCKED: 3m signal skipped due to weak 5m volume.")
                return None

        results = {
            "Fractal Zone":            self._check_fractal_zone(),
            "EMA Cloud":               self._check_ema_cloud(),
            "MACD":                    self._check_macd(),
            "Momentum":                self._check_momentum(),
            "HATS":                    self._check_hats(),
            "Volume Spike":            self._check_volume_spike(),
            "VWAP Divergence":         self._check_vwap_divergence(),
            "MTF Volume Agreement":    self._check_mtf_volume_agreement(),
            "HH/LL Trend":             self._check_hh_ll(),
            "EMA Structure":           self._check_ema_structure(),
            "Chop Zone":               self._check_chop_zone(),
            "Candle Confirmation":     self._check_candle_close(),
            "Wick Dominance":          self._check_wick_dominance(),
            "Absorption":              self._check_absorption(),
            "Support/Resistance":      self._check_support_resistance(),
            "Smart Money Bias":        self._check_smart_money_bias(),
            "Liquidity Pool":          self._check_liquidity_pool(),
            "Spread Filter":           self._check_spread_filter(),
        }

        total = sum(results.values())
        required_keys = list(results.keys())[:12]
        passed_req = sum(results[k] for k in required_keys)

        print(f"[{self.symbol}] Score: {total}/18 | Passed Required: {passed_req}/12")
        for name, ok in results.items():
            print(f"{name:20} -> {'✅' if ok else '❌'}")

        if total >= self.min_score and passed_req >= self.required_passed:
            price = self.df['close'].iat[-1]
            bias = "LONG" if price > self.df['open'].iat[-1] else "SHORT"
            signal = (
                f"{bias} signal on {self.symbol} @ {price}",
                self.symbol, bias, price, self.tf,
                f"{total}/18", f"{passed_req}/12"
            )
            print(f"[{self.symbol}] ✅ FINAL SIGNAL: {signal[0]}")
            return signal

        print(f"[{self.symbol}] ❌ No signal: thresholds not met.")
        return None

    # ==== Filter Definitions ====

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
        ha = self.df[['open','high','low','close']].sum(axis=1)/4
        return ha.iat[-1] > ha.iat[-2]

    def _check_volume_spike(self):
        avg = self.df['volume'].rolling(10).mean().iat[-1]
        return self.df['volume'].iat[-1] > 1.5 * avg

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
        vol_spike = self.df['volume'].iat[-1] > 1.5 * self.df['volume'].rolling(10).mean().iat[-1]
        return low_under and close_up and vol_spike

    def _check_support_resistance(self):
        swing = self.df['low'].rolling(5).min().iat[-3]
        return abs(self.df['close'].iat[-1] - swing) / swing < 0.01

    def _check_smart_money_bias(self):
        signed = self.df['volume'] * self.df['close'].diff().apply(lambda x: 1 if x > 0 else -1)
        recent = signed.iloc[-14:]
        return recent.sum() > 0

    def _check_liquidity_pool(self):
        hi = self.df['high'].rolling(10).max

    def _check_spread_filter(self):
        spread = self.df['high'].iat[-1] - self.df['low'].iat[-1]
        return spread < 0.02 * self.df['close'].iat[-1]

