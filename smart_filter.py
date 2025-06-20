import pandas as pd

class SmartFilter:
    def __init__(self, symbol, df, tf=None, min_score=9, required_passed=7):
        self.symbol = symbol
        self.df = df
        self.tf = tf
        self.min_score = min_score
        self.required_passed = required_passed
        self.result = None
        self.stack_results = {}
        self.total_score = 0
        self.passed_required = True

        self.df["ema20"] = df["close"].ewm(span=20).mean()
        self.df["ema50"] = df["close"].ewm(span=50).mean()
        self.df["ema200"] = df["close"].ewm(span=200).mean()

    def analyze(self):
        if self.df is None or self.df.empty or len(self.df.columns) < 6:
            print(f"[{self.symbol}] DataFrame is invalid or missing columns.")
            return None
        try:
            self.stack_results = {
                "Fractal Zone": self._check_fractal_zone(),
                "EMA Cloud": self._check_ema_cloud(),
                "MACD": self._check_macd(),
                "Momentum": self._check_momentum(),
                "HATS": self._check_hats(),
                "Volume Spike": self._check_volume_spike(),
                "VWAP Divergence": self._optional_dummy(),
                "MTF Volume Agreement": self._optional_dummy(),
                "HH/LL Trend": self._check_hh_ll(),
                "EMA Structure": self._optional_dummy(),
                "Chop Zone": self._check_chop_zone(),
                "Candle Confirmation": self._check_candle_close(),
                "Wick Dominance": self._optional_dummy(),
                "Absorption": self._optional_dummy(),
                "Support/Resistance": self._check_dummy_sr(),
                "Smart Money Bias": self._optional_dummy(),
                "Liquidity Pool": self._check_dummy_liquidity(),
                "Spread Filter": self._check_dummy_volatility(),
            }
            required_items = list(self.stack_results.keys())[:12]
            score = 0
            passed = 0
            for name, result in self.stack_results.items():
                if result:
                    score += 1
                    if name in required_items:
                        passed += 1
            self.total_score = score
            self.passed_required = passed >= self.required_passed
            print(f"[{self.symbol}] Score: {score}/18 | Required Passed: {passed}/12")
            for name, passed_stack in self.stack_results.items():
                print(f"[{name}] → {'✅' if passed_stack else '❌'}")
            if score >= self.min_score and self.passed_required:
                last_close = self.df['close'].iloc[-1]
                trend_bias = "LONG" if last_close > self.df['open'].iloc[-1] else "SHORT"
                signal_text = f"{trend_bias} Signal for {self.symbol} at {last_close}"
                print(f"[{self.symbol}] ✅ FINAL SIGNAL → {signal_text}")
                return signal_text, self.symbol, trend_bias, last_close, self.tf, f"{score}/18", f"{passed}/12"
            else:
                print(f"[{self.symbol}] ❌ No Signal (Score too low or missing required)")
                return None
        except Exception as e:
            print(f"[{self.symbol}] SmartFilter Error: {e}")
            return None

    def _check_fractal_zone(self):
        return self.df['close'].iloc[-1] > self.df['low'].rolling(20).min().iloc[-1]

    def _check_ema_cloud(self):
        return self.df["ema20"].iloc[-1] > self.df["ema50"].iloc[-1]

    def _check_macd(self):
        ema12 = self.df['close'].ewm(span=12).mean()
        ema26 = self.df['close'].ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9).mean()
        return macd_line.iloc[-1] > signal.iloc[-1]

    def _check_momentum(self):
        mom = self.df['close'].diff()
        return mom.iloc[-1] > 0

    def _check_hats(self):
        ha_close = (self.df['open'] + self.df['high'] + self.df['low'] + self.df['close']) / 4
        return ha_close.iloc[-1] > ha_close.iloc[-2]

    def _check_volume_spike(self):
        avg_vol = self.df['volume'].rolling(10).mean()
        return self.df['volume'].iloc[-1] > 1.5 * avg_vol.iloc[-1]

    def _check_chop_zone(self):
        rsi = self.df['close'].rolling(14).apply(lambda x: (x.diff() > 0).sum(), raw=False)
        return rsi.iloc[-1] > 7

    def _check_candle_close(self):
        body = abs(self.df['close'].iloc[-1] - self.df['open'].iloc[-1])
        wick = abs(self.df['high'].iloc[-1] - self.df['low'].iloc[-1])
        return body > 0.5 * wick

    def _check_hh_ll(self):
        return self.df['high'].iloc[-1] > self.df['high'].iloc[-3] and self.df['low'].iloc[-1] > self.df['low'].iloc[-3]

    def _check_dummy_sr(self):
        return True

    def _check_dummy_liquidity(self):
        return True

    def _check_dummy_volatility(self):
        spread = self.df['high'].iloc[-1] - self.df['low'].iloc[-1]
        return spread < (self.df['close'].iloc[-1] * 0.02)

    def _optional_dummy(self):
        return True
