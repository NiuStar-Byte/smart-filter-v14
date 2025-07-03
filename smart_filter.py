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
        min_score: int = 9,
        required_passed: int = 10,      # NEW: now 10 (for 17 gatekeepers)
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
            "Spread Filter": 2.7,
            "Liquidity Awareness": 3.2,
            "Trend Continuation": 3.7,
            "Volatility Model": 3.4,
            "ATR Momentum Burst": 3.9,
            "Volatility Squeeze": 3.1
        }

        # --- Expanded gatekeeper list ---
        self.gatekeepers = [
            "Fractal Zone", "EMA Cloud", "MACD", "Momentum", "HATS",
            "Volume Spike", "VWAP Divergence", "MTF Volume Agreement",
            "HH/LL Trend", "EMA Structure", "Chop Zone",
            "Candle Confirmation", "Trend Continuation",
            "Volatility Model", "Liquidity Awareness",
            "ATR Momentum Burst", "Volatility Squeeze"
        ]

    # ===== Pure Directional Decision Engine (4+4 logic) =====

    def _directional_decision(self, results):
        # 4 for LONG, 4 for SHORT (all based on latest audit)
        long_filters = ["EMA Cloud", "Momentum", "Liquidity Pool", "VWAP Divergence"]
        short_filters = ["MACD", "Candle Confirmation", "Fractal Zone", "ATR Momentum Burst"]
        long_votes = sum(results.get(f, False) for f in long_filters)
        short_votes = sum(results.get(f, False) for f in short_filters)
        if long_votes - short_votes >= 2:
            return "LONG"
        elif short_votes - long_votes >= 2:
            return "SHORT"
        else:
            return None  # VETO

    # ===== Main signal method =====

    def analyze(self):
        if self.df.empty:
            print(f"[{self.symbol}] Error: DataFrame empty.")
            return None

        # Run all checks, collect results
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

        # PASSES + WEIGHTS CALCULATION (for new GK set)
        passed_gk = [f for f in self.gatekeepers if results.get(f, False)]
        passes = len(passed_gk)
        total_gk_weight = sum(self.filter_weights[f] for f in self.gatekeepers)
        passed_weight = sum(self.filter_weights[f] for f in passed_gk)
        confidence = round(self._safe_divide(100 * passed_weight, total_gk_weight), 1)

        # === Super-GK Hard Blockers: Order Book Wall + Resting Order Density ===
        orderbook_ok = self._order_book_wall_passed()
        resting_density_ok = self._resting_order_density_passed()

        # New: Direction decided by 4+4 vote (no bias proposal)
        final_bias = self._directional_decision(results)
        print(f"[{self.symbol}] Directional Votes: LONG={sum(results.get(f, False) for f in ['EMA Cloud', 'Momentum', 'Liquidity Pool', 'VWAP Divergence'])} "
              f"SHORT={sum(results.get(f, False) for f in ['MACD', 'Candle Confirmation', 'Fractal Zone', 'ATR Momentum Burst'])} "
              f"| Final Bias: {final_bias or 'VETO'}")

        valid_signal = (
            final_bias is not None
            and score >= self.min_score
            and passes >= self.required_passed
            and orderbook_ok
            and resting_density_ok
        )
        price = self.df['close'].iat[-1]

        message = (
            f"{final_bias or 'NO-SIGNAL'} on {self.symbol} @ {price:.6f} "
            f"| Score: {score}/23 | Passed: {passes}/{len(self.gatekeepers)} "
            f"| Confidence: {confidence}% (Weighted: {passed_weight:.1f}/{total_gk_weight:.1f})"
        )

        if valid_signal:
            print(f"[{self.symbol}] ✅ FINAL SIGNAL: {message}")
        else:
            print(f"[{self.symbol}] ❌ No signal.")

        signal_time = datetime.utcnow()
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
            "bias": final_bias,
            "price": price,
            "valid_signal": valid_signal,
            "message": message,
            "filter_results": results,
            "Signal Time": signal_time
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
        ha = self.df[['open', 'high', 'low', 'close']].mean(axis=1)
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
        cond = self.df['ema20'].iat[-1] > self.df['ema50'].iat[-1] > self.df['ema200'].iat[-1]
        slope = all(self.df[f'ema{x}'].iat[-1] > self.df[f'ema{x}'].iat[-2] for x in (20, 50, 200))
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
        return self.df['ema20'].iat[-1] > self.df['ema20'].iat[-2]

    def _check_volatility_model(self, low_pct=0.01, high_pct=0.05, period=14):
        high_low = self.df['high'] - self.df['low']
        high_prev = (self.df['high'] - self.df['close'].shift()).abs()
        low_prev = (self.df['low'] - self.df['close'].shift()).abs()
        tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iat[-1]
        last = self.df['close'].iat[-1]
        atr_pct = self._safe_divide(atr, last)
        return low_pct <= atr_pct <= high_pct

    def _check_atr_momentum_burst(self, atr_period=14, burst_threshold=1.5):
        high_low = self.df['high'] - self.df['low']
        high_prev = (self.df['high'] - self.df['close'].shift()).abs()
        low_prev = (self.df['low'] - self.df['close'].shift()).abs()
        tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        momentum = self.df['close'].diff()
        burst = (momentum.abs() / atr).iat[-1]
        return burst > burst_threshold

    def _check_volatility_squeeze(self, window=20):
        rolling_high = self.df['high'].rolling(window).max()
        rolling_low = self.df['low'].rolling(window).min()
        range_pct = (rolling_high - rolling_low) / self.df['close']
        return range_pct.iat[-1] < 0.02

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
