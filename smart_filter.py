import pandas as pd

class SmartFilter:
    """
    SmartFilter v14: Applies 18 technical filters on a primary timeframe DataFrame,
    with optional 3m/5m data for multi-timeframe volume agreement.

    Returns a boolean result per filter, total score, and a final signal tuple when thresholds are met.
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
        self.stack_results = {}
        self.total_score = 0
        self.passed_required = False

        # Calculate standard indicators
        self.df["ema20"] = self.df["close"].ewm(span=20).mean()
        self.df["ema50"] = self.df["close"].ewm(span=50).mean()
        self.df["ema200"] = self.df["close"].ewm(span=200).mean()
        # Calculate intraday VWAP
        self.df["vwap"] = (self.df["close"] * self.df["volume"]).cumsum() / self.df["volume"].cumsum()

    def analyze(self):
        """
        Evaluate all filters. Print and return a signal tuple if thresholds met, else None.
        """
        if self.df.empty:
            print(f"[{self.symbol}] Error: DataFrame is empty.")
            return None

        # Run each filter and store boolean results
        self.stack_results = {
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

        # Compute score
        self.total_score = sum(self.stack_results.values())
        required_keys = list(self.stack_results.keys())[:12]
        passed_required = sum(self.stack_results[k] for k in required_keys)
        self.passed_required = passed_required >= self.required_passed

        # Print results
        print(f"[{self.symbol}] Score: {self.total_score}/18 | Required Passed: {passed_required}/12")
        for name, passed in self.stack_results.items():
            print(f"{name:20} -> {'✅' if passed else '❌'}")

        # Final signal
        if self.total_score >= self.min_score and self.passed_required:
            last_close = self.df['close'].iat[-1]
            bias = "LONG" if last_close > self.df['open'].iat[-1] else "SHORT"
            signal = (
                f"{bias} signal on {self.symbol} @ {last_close}",
                self.symbol,
                bias,
                last_close,
                self.tf,
                f"{self.total_score}/18",
                f"{passed_required}/12"
            )
            print(f"[{self.symbol}] ✅ FINAL SIGNAL: {signal[0]}")
            return signal

        print(f"[{self.symbol}] ❌ No signal: thresholds not met.")
        return None

    # -- Filter implementations --
    def _check_fractal_zone(self):
        return self.df['close'].iat[-1] > self.df['low'].rolling(20).min().iat[-1]

    def _check_ema_cloud(self):
        return self.df['ema20'].iat[-1] > self.df['ema50'].iat[-1]

    def _check_macd(self):
        ema12 = self.df['close'].ewm(span=12).mean()
        ema26 = self.df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd.iat[-1] > signal.iat[-1]

    def _check_momentum(self):
        return self.df['close'].diff().iat[-1] > 0

    def _check_hats(self):
        ha_close = (self.df[['open','high','low','close']].sum(axis=1) / 4)
        return ha_close.iat[-1] > ha_close.iat[-2]

    def _check_volume_spike(self):
        return self.df['volume'].iat[-1] > 1.5 * self.df['volume'].rolling(10).mean().iat[-1]

    def _check_vwap_divergence(self):
        diff_pct = (self.df['close'].iat[-1] - self.df['vwap'].iat[-1]) / self.df['vwap'].iat[-1]
        return (self.df['close'].iat[-1] > self.df['vwap'].iat[-1]) and (diff_pct > 0.001)

    def _check_mtf_volume_agreement(self):
        if self.df3m is None or self.df5m is None:
            return False
        v3 = self.df3m['volume'].iat[-1] > self.df3m['volume'].rolling(20).mean().iat[-1]
        v5 = self.df5m['volume'].iat[-1] > self.df5m['volume'].rolling(20).mean().iat[-1]
        return v3 and v5

    def _check_hh_ll(self):
        return (
            self.df['high'].iat[-1] > self.df['high'].iat[-3]
            and self.df['low'].iat[-1] > self.df['low'].iat[-3]
        )

    def _check_ema_structure(self):
        cond = (
            self.df['ema20'].iat[-1] > self.df['ema50'].iat[-1] > self.df['ema200'].iat[-1]
        )
        slope = (
            self.df['ema20'].iat[-1] > self.df['ema20'].iat[-2] and
            self.df['ema50'].iat[-1] > self.df['ema50'].iat[-2] and
            self.df['ema200'].iat[-1] > self.df['ema200'].iat[-2]
        )
        return cond and slope

    def _check_chop_zone(self):
        positives = (self.df['close'].diff() > 0).sum()
        return positives > 7

    def _check_candle_close(self):
        body = abs(self.df['close'].iat[-1] - self.df['open'].iat[-1])
        rng = self.df['high'].iat[-1] - self.df['low'].iat[-1]
        return body > 0.5 * rng

    def _check_wick_dominance(self):
        body = abs(self.df['close'].iat[-1] - self.df['open'].iat[-1])
        rng = self.df['high'].iat[-1] - self.df['low'].iat[-1]
        lower_wick = min(self.df['close'].iat[-1], self.df['open'].iat[-1]) - self.df['low'].iat[-1]
        return (body > 0.6 * rng) and (lower_wick < 0.2 * rng)

    def _check_absorption(self):
        low_below_open = self.df['low'].iat[-1] < self.df['open'].iat[-1]
        closed_above = self.df['close'].iat[-1] > self.df['open'].iat[-1]
        high_vol = self.df['volume'].iat[-1] > 1.5 * self.df['volume'].rolling(10).mean().iat[-1]
        return low_below_open and closed_above and high_vol

    def _check_support_resistance(self):
        swing_low = self.df['low'].rolling(5).min().iat[-3]
        return abs(self.df['close'].iat[-1] - swing_low) / swing_low < 0.01

        def _check_smart_money_bias(self):
        # Sum of signed volume over the last 14 bars: +1 for up-close, -1 for down-close
        signed = self.df['volume'] * self.df['close'].diff().apply(lambda x: 1 if x > 0 else -1)
        # Use iloc for slicing
        recent = signed.iloc[-14:]
        return recent.sum() > 0
