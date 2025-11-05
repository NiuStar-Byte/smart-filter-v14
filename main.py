# main.py
# Revised: refactored single-cycle run into `run_cycle()` and a persistent outer loop,
# added safer SuperGK fetch normalization, environment-driven cycle sleep, and more defensive logging.
# NOTE: This variant FORCE-BYPASSES SuperGK for ALL signals (LONG and SHORT).

print("[INFO] main.py script started.", flush=True)
from signal_debug_log import export_signal_debug_txt, log_fired_signal
import os
import time
import pandas as pd
import random
import pytz
from datetime import datetime
from kucoin_data import get_live_entry_price, DEFAULT_SLIPPAGE
from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert, send_telegram_file
from kucoin_orderbook import get_order_wall_delta
from pec_engine import run_pec_check, export_pec_log
from tp_sl_retracement import calculate_tp_sl
from test_filters import run_all_filter_tests
import math

# --- Configuration ---
TOKENS = [
    "BTC-USDT", "ETH-USDT",
    "BNB-USDT", "XRP-USDT", "SOL-USDT", "ADA-USDT", "XLM-USDT",
    "TON-USDT", "AVAX-USDT", "LINK-USDT", "DOT-USDT", "ARB-USDT",
    "PUMP-USDT", "KAITO-USDT", "MAGIC-USDT", "SUI-USDT", "AERO-USDT",
    "BERA-USDT", "UNI-USDT", "HBAR-USDT", "SAHARA-USDT", "VIRTUAL-USDT",
    "PARTI-USDT", "CFX-USDT", "DOGE-USDT", "VINE-USDT", "PENGU-USDT",
    "WIF-USDT", "EIGEN-USDT", "SPK-USDT", "HYPE-USDT", "WLFI-USDT",
    "POL-USDT", "RAY-USDT", "ZKJ-USDT", "AAVE-USDT", "DYDX-USDT",
    "ONDO-USDT", "ARKM-USDT", "ATH-USDT", "NMR-USDT", "PROMPT-USDT",
    "TURBO-USDT", "ENA-USDT", "BIO-USDT", "ASTER-USDT", "XPL-USDT",
    "AVNT-USDT", "ORDER-USDT", "XAUT-USDT", "ZORA-USDT"
]

COOLDOWN = {"3min": 60, "5min": 60}
last_sent = {}

PEC_BARS = 5
PEC_WINDOW_MINUTES = 720
OHLCV_LIMIT = 1000

# cycle sleep can be controlled via environment variable (seconds)
CYCLE_SLEEP = int(os.getenv("CYCLE_SLEEP", "60"))

# --- SuperGK helper: main_supergk_ok left for compatibility but main will bypass ---
def main_supergk_ok(bias, orderbook_result, density_result, analyzer_result):
    """Legacy helper retained for compatibility. Not used when global bypass is enforced in main."""
    # Keep behavior predictable if called elsewhere: call canonical function if available
    try:
        return super_gk_aligned(bias, orderbook_result, density_result)
    except Exception as e:
        print(f"[SuperGK][MAIN] Error calling super_gk_aligned(): {e}", flush=True)
        return False

def get_local_wib(dt):
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)
    return dt.tz_localize('UTC').tz_convert('Asia/Jakarta').replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')

def get_resting_order_density(symbol, depth=100, top_n=5):
    """
    Wrapper that uses kucoin_density.get_resting_density (network fetch) and ensures
    returned densities are percentages (0..100). Defensive - returns zeros on error.
    """
    try:
        # import here to avoid circular import on module load
        from kucoin_density import get_resting_density as _kd
        res = _kd(symbol, depth=depth, levels=top_n)
        # Ensure fields exist and are numeric
        bid_density = float(res.get("bid_density", 0.0))
        ask_density = float(res.get("ask_density", 0.0))
        # Sanity clamp: density should be between 0 and 100
        bid_density = max(0.0, min(100.0, bid_density))
        ask_density = max(0.0, min(100.0, ask_density))
        # Also return totals for debugging
        return {
            "bid_density": bid_density,
            "ask_density": ask_density,
            "bid_top": float(res.get("bid_top", 0.0)),
            "ask_top": float(res.get("ask_top", 0.0)),
            "bid_total": float(res.get("bid_total", 0.0)),
            "ask_total": float(res.get("ask_total", 0.0))
        }
    except Exception as e:
        print(f"[get_resting_order_density] Error for {symbol}: {e}", flush=True)
        return {"bid_density": 0.0, "ask_density": 0.0, "bid_top": 0.0, "ask_top": 0.0, "bid_total": 0.0, "ask_total": 0.0}

def candle_color(open, close):
    return 'green' if close > open else 'red'

def early_breakout(df, lookback=3):
    if len(df) < lookback:
        return {'valid_signal': False, 'bias': None, 'price': None}

    candles = df.iloc[-lookback:]
    colors = [candle_color(row['open'], row['close']) for _, row in candles.iterrows()]
    if all(c == 'green' for c in colors):
        direction = 'green'
    elif all(c == 'red' for c in colors):
        direction = 'red'
    else:
        return {'valid_signal': False, 'bias': None, 'price': None}

    prev_high = candles['high'].iloc[0]
    prev_low = candles['low'].iloc[0]
    close = candles['close'].iloc[-1]

    if close > prev_high and direction == 'green':
        return {'valid_signal': True, 'bias': 'LONG', 'price': close}
    elif close < prev_low and direction == 'red':
        return {'valid_signal': True, 'bias': 'SHORT', 'price': close}

    return {'valid_signal': False, 'bias': None, 'price': close}

def log_orderbook_and_density(symbol):
    try:
        ob = get_order_wall_delta(symbol)
        # Normalize missing keys
        buy_wall = float(ob.get('buy_wall', 0.0))
        sell_wall = float(ob.get('sell_wall', 0.0))
        wall_delta = float(ob.get('wall_delta', 0.0))
        midprice = ob.get('midprice', 'N/A')
        print(
            f"[OrderBookDeltaLog] {symbol} | buy_wall={buy_wall} | sell_wall={sell_wall} | wall_delta={wall_delta} | midprice={midprice}",
            flush=True
        )
    except Exception as e:
        print(f"[OrderBookDeltaLog] {symbol} ERROR: {e}", flush=True)

    try:
        dens = get_resting_order_density(symbol)
        # dens now guaranteed to be percentages (0..100) by wrapper above
        print(
            f"[RestingOrderDensityLog] {symbol} | bid_density={dens.get('bid_density',0.0):.2f}% | ask_density={dens.get('ask_density',0.0):.2f}% | "
            f"bid_top={dens.get('bid_top',0.0)} | ask_top={dens.get('ask_top',0.0)} | "
            f"bid_total={dens.get('bid_total',0.0)} | ask_total={dens.get('ask_total',0.0)}",
            flush=True
        )
    except Exception as e:
        print(f"[RestingOrderDensityLog] {symbol} ERROR: {e}", flush=True)

# --- SuperGK canonical function retained for reference but not used by main (global bypass enforced below) ---
def super_gk_aligned(bias, orderbook_result, density_result,
                     wall_pct_threshold=None, density_threshold=None,
                     wall_weight=None, density_weight=None, composite_threshold=None):
    """
    Composite-first SuperGK with optional bypasses.
    """
    # Helper to read floats from env with defaults
    def _env_float(name, default):
        v = os.getenv(name)
        try:
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    default_wpct = _env_float("SUPERGK_WALL_PCT_THRESHOLD", 1.0)
    default_dpct = _env_float("SUPERGK_DENSITY_THRESHOLD", 0.5)
    global_wall_w = _env_float("SUPERGK_WALL_WEIGHT", 0.25)
    global_density_w = _env_float("SUPERGK_DENSITY_WEIGHT", 0.75)
    global_comp_th = _env_float("SUPERGK_COMPOSITE_THRESHOLD", 0.01)

    short_wall_w = _env_float("SUPERGK_SHORT_WALL_WEIGHT", os.getenv("SUPERGK_SHORT_WALL_WEIGHT") or global_wall_w)
    short_density_w = _env_float("SUPERGK_SHORT_DENSITY_WEIGHT", os.getenv("SUPERGK_SHORT_DENSITY_WEIGHT") or global_density_w)
    short_comp_th = _env_float("SUPERGK_SHORT_COMPOSITE_THRESHOLD", os.getenv("SUPERGK_SHORT_COMPOSITE_THRESHOLD") or global_comp_th)

    long_wall_w = _env_float("SUPERGK_LONG_WALL_WEIGHT", os.getenv("SUPERGK_LONG_WALL_WEIGHT") or global_wall_w)
    long_density_w = _env_float("SUPERGK_LONG_DENSITY_WEIGHT", os.getenv("SUPERGK_LONG_DENSITY_WEIGHT") or global_density_w)
    long_comp_th = _env_float("SUPERGK_LONG_COMPOSITE_THRESHOLD", os.getenv("SUPERGK_LONG_COMPOSITE_THRESHOLD") or global_comp_th)

    density_dom_factor = _env_float("SUPERGK_DENSITY_DOMINANCE_FACTOR", 1.6)

    wall_pct_threshold = default_wpct if wall_pct_threshold is None else wall_pct_threshold
    density_threshold = default_dpct if density_threshold is None else density_threshold

    if bias == "SHORT":
        wall_weight = short_wall_w if wall_weight is None else wall_weight
        density_weight = short_density_w if density_weight is None else density_weight
        composite_threshold = short_comp_th if composite_threshold is None else composite_threshold
    else:
        wall_weight = long_wall_w if wall_weight is None else wall_weight
        density_weight = long_density_w if density_weight is None else density_weight
        composite_threshold = long_comp_th if composite_threshold is None else composite_threshold

    buy_wall = float(orderbook_result.get('buy_wall', 0.0)) if orderbook_result else 0.0
    sell_wall = float(orderbook_result.get('sell_wall', 0.0)) if orderbook_result else 0.0
    wall_delta = float(orderbook_result.get('wall_delta', buy_wall - sell_wall)) if orderbook_result else 0.0

    bid_density = float(density_result.get('bid_density', 0.0)) if density_result else 0.0
    ask_density = float(density_result.get('ask_density', 0.0)) if density_result else 0.0

    total_wall = max(buy_wall + sell_wall, 1e-9)
    wall_pct = (abs(wall_delta) / total_wall) * 100.0
    wall_sign = "LONG" if wall_delta > 0 else "SHORT" if wall_delta < 0 else "NEUTRAL"

    density_diff = bid_density - ask_density
    density_pct = abs(density_diff)
    density_sign = "LONG" if density_diff > 0 else "SHORT" if density_diff < 0 else "NEUTRAL"

    orderbook_bias = wall_sign if wall_pct >= wall_pct_threshold else "NEUTRAL"
    density_bias = density_sign if density_pct >= density_threshold else "NEUTRAL"

    wall_score = (wall_pct / 100.0)
    density_score = (density_pct / 100.0)

    def sign_to_factor(sign, desired):
        if sign == desired:
            return 1.0
        elif sign == "NEUTRAL":
            return 0.0
        else:
            return -1.0

    wall_factor = sign_to_factor(orderbook_bias, bias)
    density_factor = sign_to_factor(density_bias, bias)

    composite = wall_weight * wall_factor * wall_score + density_weight * density_factor * density_score

    wall_comp = wall_weight * wall_factor * wall_score
    density_comp = density_weight * density_factor * density_score

    return composite >= composite_threshold

# --- run_cycle / main logic (UNCONDITIONAL SUPERGK BYPASS FOR ALL SIGNALS) ---
def run_cycle():
    """
    Single pass over all TOKENS. Returns list of valid_debug dicts collected during this cycle.
    Robustified send logic:
    - Forces SuperGK bypass (mainized elsewhere) by treating super_gk_ok=True.
    - Coerces entry_price to float and falls back safely.
    - Wraps send block in full try/except and prints tracebacks.
    - Emits debug lines showing the precise entry_price and send_telegram_alert return value.
    """
    import traceback

    print("[INFO] Starting Smart Filter cycle (single pass)...", flush=True)
    valid_debugs = []
    now = time.time()

    for idx, symbol in enumerate(TOKENS, start=1):
        print(f"[INFO] Checking {symbol}...", flush=True)
        try:
            df3 = get_ohlcv(symbol, interval="3min", limit=OHLCV_LIMIT)
            df5 = get_ohlcv(symbol, interval="5min", limit=OHLCV_LIMIT)
            if df3 is None or df3.empty or df5 is None or df5.empty:
                print(f"[WARN] Not enough data for {symbol} (df3 or df5 empty). Skipping.", flush=True)
                continue

            early_breakout_3m = early_breakout(df3, lookback=3)
            early_breakout_5m = early_breakout(df5, lookback=3)

            # --- 3min TF ---
            try:
                key3 = f"{symbol}_3min"
                sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=df5, tf="3min")
                regime3 = sf3._market_regime()
                res3 = sf3.analyze()
                if isinstance(res3, dict) and res3.get("filters_ok") is True:
                    last3 = last_sent.get(key3, 0)
                    if now - last3 >= COOLDOWN["3min"]:
                        numbered_signal = f"{idx}.A"

                        log_orderbook_and_density(symbol)
                        try:
                            orderbook_result = get_order_wall_delta(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_order_wall_delta failed for {symbol}: {e}", flush=True)
                            orderbook_result = {"buy_wall": 0, "sell_wall": 0, "wall_delta": 0, "midprice": None}
                        try:
                            density_result = get_resting_order_density(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_resting_order_density failed for {symbol}: {e}", flush=True)
                            density_result = {"bid_density": 0.0, "ask_density": 0.0, "bid_levels": 0, "ask_levels": 0, "midprice": None}

                        bias = res3.get("bias", "NEUTRAL")
                        sf3.bias = bias

                        # GLOBAL BYPASS: SuperGK is fully disabled for ALL signals (LONG & SHORT)
                        print("[SuperGK][MAIN] GLOBAL DISABLED - bypassing SuperGK for ALL signals (LONG and SHORT)", flush=True)
                        super_gk_ok = True

                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok}", flush=True)

                        if not super_gk_ok:
                            # kept for compatibility but should never execute
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                            valid_debugs.append({
                                "symbol": symbol,
                                "tf": "3min",
                                "bias": bias,
                                "filter_weights_long": getattr(sf3, 'filter_weights_long', []),
                                "filter_weights_short": getattr(sf3, 'filter_weights_short', []),
                                "gatekeepers": getattr(sf3, 'gatekeepers', []),
                                "results_long": res3.get("results_long", {}),
                                "results_short": res3.get("results_short", {}),
                                "caption": f"Blocked signal debug for {symbol} 3min",
                                "orderbook_result": orderbook_result,
                                "density_result": density_result,
                                "entry_price": res3.get("price"),
                                "fired_time_utc": datetime.utcnow(),
                                "early_breakout_3m": early_breakout_3m
                            })
                            continue

                        # --- Sending block (robust) ---
                        print(f"[LOG] Sending 3min alert for {res3.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()

                        # Try to fetch live entry price but fall back safely
                        entry_price_raw = None
                        try:
                            entry_price_raw = get_live_entry_price(
                                res3.get("symbol"),
                                bias,
                                tf=res3.get("tf"),
                                slippage=DEFAULT_SLIPPAGE
                            ) or res3.get("price", 0.0)
                        except Exception as e:
                            print(f"[WARN] get_live_entry_price raised: {e} -- falling back to analyzer price", flush=True)
                            entry_price_raw = res3.get("price", 0.0)

                        # Coerce to float defensively
                        try:
                            entry_price = float(entry_price_raw)
                        except Exception:
                            try:
                                entry_price = float(str(entry_price_raw))
                                print(f"[WARN] Coerced entry_price from {entry_price_raw} to {entry_price}", flush=True)
                            except Exception:
                                print(f"[WARN] Failed to coerce entry_price ({entry_price_raw}); defaulting to 0.0", flush=True)
                                entry_price = 0.0

                        # collect signal metadata
                        score = res3.get("score", 0)
                        score_max = res3.get("score_max", 0)
                        passes = res3.get("passes", 0)
                        gatekeepers_total = res3.get("gatekeepers_total", 0)
                        passed_weight = res3.get("passed_weight", 0.0)
                        total_weight = res3.get("total_weight", 0.0)
                        Route = res3.get("Route", None)
                        signal_type = bias
                        tf_val = res3.get("tf", "3min")
                        symbol_val = res3.get("symbol", symbol)
                        try:
                            confidence = round((passed_weight / total_weight) * 100, 1) if total_weight else 0.0
                        except Exception:
                            confidence = 0.0
                        entry_idx = df3.index.get_loc(df3.index[-1])

                        # Calculate TP/SL safely
                        try:
                            tp_sl = calculate_tp_sl(df3, entry_price, signal_type)
                            tp = tp_sl.get('tp')
                            sl = tp_sl.get('sl')
                            fib_levels = tp_sl.get('fib_levels')
                        except Exception as e:
                            print(f"[WARN] calculate_tp_sl failed: {e}", flush=True)
                            tp = sl = fib_levels = None

                        # Append debug object
                        valid_debugs.append({
                            "symbol": symbol_val,
                            "tf": tf_val,
                            "bias": bias,
                            "filter_weights_long": getattr(sf3, 'filter_weights_long', []),
                            "filter_weights_short": getattr(sf3, 'filter_weights_short', []),
                            "gatekeepers": getattr(sf3, 'gatekeepers', []),
                            "results_long": res3.get("results_long", {}),
                            "results_short": res3.get("results_short", {}),
                            "caption": f"Signal debug log for {symbol_val} {tf_val}",
                            "orderbook_result": orderbook_result,
                            "density_result": density_result,
                            "entry_price": entry_price,
                            "fired_time_utc": fired_time_utc,
                            "early_breakout_3m": early_breakout_3m,
                            "tp": tp,
                            "sl": sl,
                            "fib_levels": fib_levels
                        })

                        # Log fired signal and attempt send wrapped in try/except to capture traceback
                        try:
                            log_fired_signal(
                                symbol=symbol_val,
                                tf=tf_val,
                                signal_type=signal_type,
                                entry_idx=entry_idx,
                                fired_time=fired_time_utc,
                                score=score,
                                max_score=score_max,
                                passed=passes,
                                max_passed=gatekeepers_total,
                                weights=passed_weight,
                                max_weights=total_weight,
                                confidence_rate=confidence,
                                entry_price=entry_price
                            )
                        except Exception as e:
                            print(f"[WARN] log_fired_signal raised: {e}", flush=True)
                            traceback.print_exc()

                        regime = sf3._market_regime() if hasattr(sf3, "_market_regime") else None

                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                print(f"[DEBUG] send_telegram_alert about to send: symbol={symbol_val} price={entry_price} (type={type(entry_price)})", flush=True)
                                sent_ok = send_telegram_alert(
                                    numbered_signal=numbered_signal,
                                    symbol=symbol_val,
                                    signal_type=signal_type,
                                    Route=Route,
                                    price=entry_price,
                                    tf=tf_val,
                                    score=score,
                                    passed=passes,
                                    confidence=confidence,
                                    weighted=passed_weight,
                                    score_max=score_max,
                                    gatekeepers_total=gatekeepers_total,
                                    total_weight=total_weight,
                                    reversal_side=res3.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_3m=early_breakout_3m,
                                    tp=tp,
                                    sl=sl
                                )
                                print(f"[DEBUG] send_telegram_alert returned: {sent_ok}", flush=True)
                                if sent_ok:
                                    last_sent[key3] = now
                                else:
                                    print(f"[ERROR] Telegram send failed for {symbol_val} (3min). Not setting cooldown.", flush=True)
                            except Exception as e:
                                print(f"[ERROR] Exception during Telegram send for {symbol_val} (3min): {e}", flush=True)
                                traceback.print_exc()
                        else:
                            print(f"[INFO] DRY_RUN enabled - simulated send for {symbol_val} (3min). Not setting cooldown.", flush=True)
                else:
                    print(f"[INFO] No valid 3min signal for {symbol}.", flush=True)
            except Exception as e:
                print(f"[ERROR] Exception in processing 3min for {symbol}: {e}", flush=True)
                traceback.print_exc()

            # --- 5min TF ---
            try:
                key5 = f"{symbol}_5min"
                sf5 = SmartFilter(symbol, df5, df3m=df3, df5m=df5, tf="5min")
                regime5 = sf5._market_regime()
                res5 = sf5.analyze()
                if isinstance(res5, dict) and res5.get("filters_ok") is True:
                    last5 = last_sent.get(key5, 0)
                    if now - last5 >= COOLDOWN["5min"]:
                        numbered_signal = f"{idx}.B"

                        log_orderbook_and_density(symbol)
                        try:
                            orderbook_result = get_order_wall_delta(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_order_wall_delta failed for {symbol}: {e}", flush=True)
                            orderbook_result = {"buy_wall": 0, "sell_wall": 0, "wall_delta": 0, "midprice": None}
                        try:
                            density_result = get_resting_order_density(symbol) or {}
                        except Exception as e:
                            print(f"[ERROR] get_resting_order_density failed for {symbol}: {e}", flush=True)
                            density_result = {"bid_density": 0.0, "ask_density": 0.0, "bid_levels": 0, "ask_levels": 0, "midprice": None}

                        bias = res5.get("bias", "NEUTRAL")
                        sf5.bias = bias

                        # GLOBAL BYPASS: SuperGK is fully disabled for ALL signals (LONG & SHORT)
                        print("[SuperGK][MAIN] GLOBAL DISABLED - bypassing SuperGK for ALL signals (LONG and SHORT)", flush=True)
                        super_gk_ok = True

                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok}", flush=True)

                        if not super_gk_ok:
                            # kept for compatibility but should never execute
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                            valid_debugs.append({
                                "symbol": symbol,
                                "tf": "5min",
                                "bias": bias,
                                "filter_weights_long": getattr(sf5, 'filter_weights_long', []),
                                "filter_weights_short": getattr(sf5, 'filter_weights_short', []),
                                "gatekeepers": getattr(sf5, 'gatekeepers', []),
                                "results_long": res5.get("results_long", {}),
                                "results_short": res5.get("results_short", {}),
                                "caption": f"Blocked signal debug for {symbol} 5min",
                                "orderbook_result": orderbook_result,
                                "density_result": density_result,
                                "entry_price": res5.get("price"),
                                "fired_time_utc": datetime.utcnow(),
                                "early_breakout_5m": early_breakout_5m
                            })
                            continue

                        # --- Sending block (robust) ---
                        print(f"[LOG] Sending 5min alert for {res5.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()

                        entry_price_raw = None
                        try:
                            entry_price_raw = get_live_entry_price(
                                res5.get("symbol"),
                                bias,
                                tf=res5.get("tf"),
                                slippage=DEFAULT_SLIPPAGE
                            ) or res5.get("price", 0.0)
                        except Exception as e:
                            print(f"[WARN] get_live_entry_price raised: {e} -- falling back to analyzer price", flush=True)
                            entry_price_raw = res5.get("price", 0.0)

                        try:
                            entry_price = float(entry_price_raw)
                        except Exception:
                            try:
                                entry_price = float(str(entry_price_raw))
                                print(f"[WARN] Coerced entry_price from {entry_price_raw} to {entry_price}", flush=True)
                            except Exception:
                                print(f"[WARN] Failed to coerce entry_price ({entry_price_raw}); defaulting to 0.0", flush=True)
                                entry_price = 0.0

                        score = res5.get("score", 0)
                        score_max = res5.get("score_max", 0)
                        passes = res5.get("passes", 0)
                        gatekeepers_total = res5.get("gatekeepers_total", 0)
                        passed_weight = res5.get("passed_weight", 0.0)
                        total_weight = res5.get("total_weight", 0.0)
                        Route = res5.get("Route", None)
                        signal_type = bias
                        tf_val = res5.get("tf", "5min")
                        symbol_val = res5.get("symbol", symbol)
                        try:
                            confidence = round((passed_weight / total_weight) * 100, 1) if total_weight else 0.0
                        except Exception:
                            confidence = 0.0
                        entry_idx = df5.index.get_loc(df5.index[-1])

                        try:
                            tp_sl = calculate_tp_sl(df5, entry_price, signal_type)
                            tp = tp_sl.get('tp')
                            sl = tp_sl.get('sl')
                            fib_levels = tp_sl.get('fib_levels')
                        except Exception as e:
                            print(f"[WARN] calculate_tp_sl failed: {e}", flush=True)
                            tp = sl = fib_levels = None

                        valid_debugs.append({
                            "symbol": symbol_val,
                            "tf": tf_val,
                            "bias": bias,
                            "filter_weights_long": getattr(sf5, 'filter_weights_long', []),
                            "filter_weights_short": getattr(sf5, 'filter_weights_short', []),
                            "gatekeepers": getattr(sf5, 'gatekeepers', []),
                            "results_long": res5.get("results_long", {}),
                            "results_short": res5.get("results_short", {}),
                            "caption": f"Signal debug log for {symbol_val} {tf_val}",
                            "orderbook_result": orderbook_result,
                            "density_result": density_result,
                            "entry_price": entry_price,
                            "fired_time_utc": fired_time_utc,
                            "early_breakout_5m": early_breakout_5m,
                            "tp": tp,
                            "sl": sl,
                            "fib_levels": fib_levels
                        })

                        try:
                            log_fired_signal(
                                symbol=symbol_val,
                                tf=tf_val,
                                signal_type=signal_type,
                                entry_idx=entry_idx,
                                fired_time=fired_time_utc,
                                score=score,
                                max_score=score_max,
                                passed=passes,
                                max_passed=gatekeepers_total,
                                weights=passed_weight,
                                max_weights=total_weight,
                                confidence_rate=confidence,
                                entry_price=entry_price
                            )
                        except Exception as e:
                            print(f"[WARN] log_fired_signal raised: {e}", flush=True)
                            traceback.print_exc()

                        regime = sf5._market_regime() if hasattr(sf5, "_market_regime") else None

                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                print(f"[DEBUG] send_telegram_alert about to send: symbol={symbol_val} price={entry_price} (type={type(entry_price)})", flush=True)
                                sent_ok = send_telegram_alert(
                                    numbered_signal=numbered_signal,
                                    symbol=symbol_val,
                                    signal_type=signal_type,
                                    Route=Route,
                                    price=entry_price,
                                    tf=tf_val,
                                    score=score,
                                    passed=passes,
                                    confidence=confidence,
                                    weighted=passed_weight,
                                    score_max=score_max,
                                    gatekeepers_total=gatekeepers_total,
                                    total_weight=total_weight,
                                    reversal_side=res5.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_5m=early_breakout_5m,
                                    tp=tp,
                                    sl=sl
                                )
                                print(f"[DEBUG] send_telegram_alert returned: {sent_ok}", flush=True)
                                if sent_ok:
                                    last_sent[key5] = now
                                else:
                                    print(f"[ERROR] Telegram send failed for {symbol_val} (5min). Not setting cooldown.", flush=True)
                            except Exception as e:
                                print(f"[ERROR] Exception during Telegram send for {symbol_val} (5min): {e}", flush=True)
                                traceback.print_exc()
                        else:
                            print(f"[INFO] DRY_RUN enabled - simulated send for {symbol_val} (5min). Not setting cooldown.", flush=True)
                else:
                    print(f"[INFO] No valid 5min signal for {symbol}.", flush=True)
            except Exception as e:
                print(f"[ERROR] Exception in processing 5min for {symbol}: {e}", flush=True)
                traceback.print_exc()

        except Exception as e:
            print(f"[FATAL] Exception processing {symbol}: {e}", flush=True)
            traceback.print_exc()

    # End of per-symbol loop
    return valid_debugs

def run():
    """
    Continuous loop that calls run_cycle() then sleeps.
    """
    print("[INFO] Starting Smart Filter engine (LIVE MODE)...\n", flush=True)
    while True:
        try:
            valid_debugs = run_cycle()

            # --- Send up to 2 debug files to Telegram (Signal Debug txt sampling) ---
            try:
                if valid_debugs:
                    print(f"[FIRED] About to send {min(len(valid_debugs), 2)} debug files to Telegram.", flush=True)
                    num = min(len(valid_debugs), 2)
                    for debug_info in random.sample(valid_debugs, num):
                        try:
                            print("[FIRED] LONG filter weights:", debug_info["filter_weights_long"], flush=True)
                            print("[FIRED] SHORT filter weights:", debug_info["filter_weights_short"], flush=True)
                            export_signal_debug_txt(
                                symbol=debug_info["symbol"],
                                tf=debug_info["tf"],
                                bias=debug_info["bias"],
                                filter_weights_long=debug_info["filter_weights_long"],
                                filter_weights_short=debug_info["filter_weights_short"],
                                gatekeepers=debug_info["gatekeepers"],
                                results_long=debug_info.get("results_long", {}),
                                results_short=debug_info.get("results_short", {}),
                                orderbook_result=debug_info.get("orderbook_result"),
                                density_result=debug_info.get("density_result")
                            )
                            send_telegram_file(
                                "signal_debug_temp.txt",
                                caption=debug_info["caption"]
                            )
                        except Exception as e:
                            print(f"[ERROR] Exception in Telegram debug send: {e}", flush=True)
                    # send tracking log at scheduled hours only
                    now_utc = datetime.utcnow()
                    if now_utc.hour in [1, 7, 13, 19] and now_utc.minute == 0:
                        try:
                            send_telegram_file(
                                "signal_tracking.txt",
                                caption=f"Signal logs sent at {now_utc.strftime('%H:%M UTC')}"
                            )
                            print(f"[INFO] Sent signal_tracking.txt at scheduled time {now_utc.strftime('%H:%M UTC')}", flush=True)
                        except Exception as e:
                            print(f"[ERROR] Exception sending signal_tracking.txt: {e}", flush=True)
                    else:
                        print(f"[INFO] Skipped sending signal_tracking.txt (current time: {now_utc.strftime('%H:%M UTC')})", flush=True)
                else:
                    print("[FIRED] valid_debugs is empty — no debug files to send to Telegram.", flush=True)
            except Exception as e:
                print(f"[FATAL] Exception in debug sending block: {e}", flush=True)

            if valid_debugs:
                print(f"[FIRED] Processed {len(valid_debugs)} valid signals this cycle", flush=True)
            else:
                print("[FIRED] No valid signals processed this cycle", flush=True)

            print(f"[INFO] ✅ Cycle complete. Sleeping {CYCLE_SLEEP} seconds...\n", flush=True)
            time.sleep(CYCLE_SLEEP)

        except Exception as e:
            print(f"[FATAL] Exception in main loop: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print("[INFO] Sleeping 10 seconds before retrying main loop...\n", flush=True)
            time.sleep(10)

if __name__ == "__main__":
    print(">>> ENTERED main.py", flush=True)
    if os.getenv("PEC_BACKTEST_ONLY", "false").lower() == "true":
        print(">>> Entering PEC_BACKTEST_ONLY branch", flush=True)
        from pec_backtest import run_pec_backtest
        try:
            print(">>> Calling run_pec_backtest", flush=True)
            run_pec_backtest(TOKENS, get_ohlcv, get_local_wib, PEC_WINDOW_MINUTES, PEC_BARS, OHLCV_LIMIT)
        except Exception as e:
            print(f"EXCEPTION in run_pec_backtest: {e}", flush=True)
            import traceback; traceback.print_exc()
    else:
        print(">>> Entering normal run() branch", flush=True)
        run()
