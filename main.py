# main.py
# Revised: refactored single-cycle run into `run_cycle()` and a persistent outer loop,
# added safer SuperGK fetch normalization, environment-driven cycle sleep, and more defensive logging.
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

# --- SuperGK helper: authoritative check used by main at send-time ---
def main_supergk_ok(bias, orderbook_result, density_result, analyzer_result):
    """Authoritative SuperGK decision performed in main (respects bypass env vars unless forced by main)."""
    # This function remains available for LONG checks; SHORT bypassing is enforced in main.
    # Read bypass flags from env (kept for compatibility if needed elsewhere)
    BYPASS_GLOBAL = os.getenv("SUPERGK_BYPASS", "false").lower() in ("1", "true", "yes", "on")
    BYPASS_SHORT = os.getenv("SUPERGK_BYPASS_SHORT", "false").lower() in ("1", "true", "yes", "on")
    BYPASS_MIN_SCORE_ENABLED = os.getenv("SUPERGK_BYPASS_MIN_SCORE_ENABLE", "false").lower() in ("1", "true", "yes", "on")
    try:
        BYPASS_MIN_SCORE = float(os.getenv("SUPERGK_BYPASS_MIN_SCORE") or 0.0)
    except Exception:
        BYPASS_MIN_SCORE = 0.0

    # Extract signal_score safely from analyzer_result
    debug_sums = analyzer_result.get("debug_sums") if isinstance(analyzer_result, dict) else None
    if debug_sums and isinstance(debug_sums, dict):
        signal_score = float(debug_sums.get("short_score" if bias == "SHORT" else "long_score", 0.0))
    else:
        signal_score = float(analyzer_result.get("score", 0.0)) if isinstance(analyzer_result, dict) else 0.0

    # Bypass logic (used if main calls this; main enforces unconditional SHORT bypass separately)
    if BYPASS_GLOBAL:
        if BYPASS_MIN_SCORE_ENABLED:
            return signal_score >= BYPASS_MIN_SCORE
        return True

    if bias == "SHORT" and BYPASS_SHORT:
        if BYPASS_MIN_SCORE_ENABLED:
            return signal_score >= BYPASS_MIN_SCORE
        return True

    # Otherwise call canonical super_gk_aligned (defined below in this file)
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

# --- SuperGK canonical function (kept inside main for single-file simplicity) ---
def super_gk_aligned(bias, orderbook_result, density_result,
                     wall_pct_threshold=None, density_threshold=None,
                     wall_weight=None, density_weight=None, composite_threshold=None):
    """
    Composite-first SuperGK with optional bypasses.
    """
    # Quick global and SHORT-only bypass
    if os.getenv("SUPERGK_BYPASS", "false").lower() in ("1", "true", "yes", "on"):
        print("[SuperGK] GLOBAL BYPASS enabled via SUPERGK_BYPASS env var — treating as PASSED", flush=True)
        return True
    if bias == "SHORT" and os.getenv("SUPERGK_BYPASS_SHORT", "false").lower() in ("1", "true", "yes", "on"):
        print("[SuperGK] SHORT-BYPASS enabled via SUPERGK_BYPASS_SHORT env var — treating SHORT as PASSED", flush=True)
        return True

    # Helper to read floats from env with defaults
    def _env_float(name, default):
        v = os.getenv(name)
        try:
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    # Global thresholds/defaults
    default_wpct = _env_float("SUPERGK_WALL_PCT_THRESHOLD", 1.0)
    default_dpct = _env_float("SUPERGK_DENSITY_THRESHOLD", 0.5)

    # Global weights and composite threshold (fallback)
    global_wall_w = _env_float("SUPERGK_WALL_WEIGHT", 0.25)
    global_density_w = _env_float("SUPERGK_DENSITY_WEIGHT", 0.75)
    global_comp_th = _env_float("SUPERGK_COMPOSITE_THRESHOLD", 0.01)

    # Per-direction overrides (if provided)
    short_wall_w = _env_float("SUPERGK_SHORT_WALL_WEIGHT", os.getenv("SUPERGK_SHORT_WALL_WEIGHT") or global_wall_w)
    short_density_w = _env_float("SUPERGK_SHORT_DENSITY_WEIGHT", os.getenv("SUPERGK_SHORT_DENSITY_WEIGHT") or global_density_w)
    short_comp_th = _env_float("SUPERGK_SHORT_COMPOSITE_THRESHOLD", os.getenv("SUPERGK_SHORT_COMPOSITE_THRESHOLD") or global_comp_th)

    long_wall_w = _env_float("SUPERGK_LONG_WALL_WEIGHT", os.getenv("SUPERGK_LONG_WALL_WEIGHT") or global_wall_w)
    long_density_w = _env_float("SUPERGK_LONG_DENSITY_WEIGHT", os.getenv("SUPERGK_LONG_DENSITY_WEIGHT") or global_density_w)
    long_comp_th = _env_float("SUPERGK_LONG_COMPOSITE_THRESHOLD", os.getenv("SUPERGK_LONG_COMPOSITE_THRESHOLD") or global_comp_th)

    # Density-dominance factor
    density_dom_factor = _env_float("SUPERGK_DENSITY_DOMINANCE_FACTOR", 1.6)

    # Apply function-args or chosen defaults
    wall_pct_threshold = default_wpct if wall_pct_threshold is None else wall_pct_threshold
    density_threshold = default_dpct if density_threshold is None else density_threshold

    # Choose per-direction weights/thresholds
    if bias == "SHORT":
        wall_weight = short_wall_w if wall_weight is None else wall_weight
        density_weight = short_density_w if density_weight is None else density_weight
        composite_threshold = short_comp_th if composite_threshold is None else composite_threshold
    else:
        wall_weight = long_wall_w if wall_weight is None else wall_weight
        density_weight = long_density_w if density_weight is None else density_weight
        composite_threshold = long_comp_th if composite_threshold is None else composite_threshold

    # Defensive extraction from input dicts
    buy_wall = float(orderbook_result.get('buy_wall', 0.0)) if orderbook_result else 0.0
    sell_wall = float(orderbook_result.get('sell_wall', 0.0)) if orderbook_result else 0.0
    wall_delta = float(orderbook_result.get('wall_delta', buy_wall - sell_wall)) if orderbook_result else 0.0

    bid_density = float(density_result.get('bid_density', 0.0)) if density_result else 0.0
    ask_density = float(density_result.get('ask_density', 0.0)) if density_result else 0.0

    # Normalize wall_delta to percent of total top-wall liquidity (safe denom)
    total_wall = max(buy_wall + sell_wall, 1e-9)
    wall_pct = (abs(wall_delta) / total_wall) * 100.0
    wall_sign = "LONG" if wall_delta > 0 else "SHORT" if wall_delta < 0 else "NEUTRAL"

    # density diff and sign (density_pct kept in same units used elsewhere)
    density_diff = bid_density - ask_density
    density_pct = abs(density_diff)
    density_sign = "LONG" if density_diff > 0 else "SHORT" if density_diff < 0 else "NEUTRAL"

    # Interpret non-neutral using thresholds
    orderbook_bias = wall_sign if wall_pct >= wall_pct_threshold else "NEUTRAL"
    density_bias = density_sign if density_pct >= density_threshold else "NEUTRAL"

    # Diagnostics
    print(f"[SuperGK] bias={bias} | wall_delta={wall_delta:.6f} buy_wall={buy_wall:.6f} sell_wall={sell_wall:.6f} "
          f"wall_pct={wall_pct:.3f}% (th={wall_pct_threshold}) wall_bias={orderbook_bias} | "
          f"bid_density={bid_density:.3f} ask_density={ask_density:.3f} density_pct={density_pct:.3f} (th={density_threshold}) "
          f"density_bias={density_bias} | weights (wall={wall_weight}, density={density_weight}) comp_th={composite_threshold} dom_factor={density_dom_factor}",
          flush=True)

    # If both metrics support the bias -> accept
    if orderbook_bias == bias and density_bias == bias:
        print("[SuperGK] Passed: both metrics support bias.", flush=True)
        return True

    # If both neutral -> block
    if orderbook_bias == "NEUTRAL" and density_bias == "NEUTRAL":
        print("[SuperGK] Blocked: both metrics neutral.", flush=True)
        return False

    # If one supports and the other is neutral -> accept
    if orderbook_bias == bias and density_bias == "NEUTRAL":
        print("[SuperGK] Passed: orderbook supports bias and density is neutral.", flush=True)
        return True
    if density_bias == bias and orderbook_bias == "NEUTRAL":
        print("[SuperGK] Passed: density supports bias and orderbook is neutral.", flush=True)
        return True

    # Density-dominance override: accept when density strongly outweighs wall in favor of bias
    try:
        if density_sign == bias and (wall_pct == 0 or density_pct >= density_dom_factor * wall_pct):
            print(f"[SuperGK] Passed by density dominance (density_pct {density_pct:.3f} >= {density_dom_factor} * wall_pct {wall_pct:.3f}%).", flush=True)
            return True
    except Exception:
        pass

    # Compute composite to resolve conflicts or opposing signs
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

    # Composite diagnostics
    wall_comp = wall_weight * wall_factor * wall_score
    density_comp = density_weight * density_factor * density_score
    print(f"[SuperGK] composite -> wall: {wall_comp:.6f}, density: {density_comp:.6f}, composite={composite:.6f}", flush=True)

    if composite >= composite_threshold:
        print("[SuperGK] Passed: composite meets threshold.", flush=True)
        return True

    print("[SuperGK] Blocked: composite insufficient.", flush=True)
    return False

# --- run_cycle / main logic (uses main_supergk_ok at decision point) ---
def run_cycle():
    """
    Single pass over all TOKENS. Returns list of valid_debug dicts collected during this cycle.
    """
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

                        # AUTHORITATIVE: Force unconditional SHORT bypass here
                        if bias == "SHORT":
                            print("[SuperGK][MAIN] SHORT UNCONDITIONAL BYPASS active — forcing SuperGK PASSED for SHORT signals", flush=True)
                            super_gk_ok = True
                        else:
                            super_gk_ok = main_supergk_ok(bias, orderbook_result, density_result, res3)

                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok}", flush=True)

                        if not super_gk_ok:
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

                        print(f"[LOG] Sending 3min alert for {res3.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()
                        entry_price = get_live_entry_price(
                            res3.get("symbol"),
                            bias,
                            tf=res3.get("tf"),
                            slippage=DEFAULT_SLIPPAGE
                        ) or res3.get("price", 0.0)

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

                        # Calculate TP/SL
                        tp_sl = calculate_tp_sl(df3, entry_price, signal_type)
                        tp = tp_sl.get('tp')
                        sl = tp_sl.get('sl')
                        fib_levels = tp_sl.get('fib_levels')

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

                        regime = sf3._market_regime() if hasattr(sf3, "_market_regime") else None

                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
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
                                if sent_ok:
                                    last_sent[key3] = now
                                else:
                                    print(f"[ERROR] Telegram send failed for {symbol_val} (3min). Not setting cooldown.", flush=True)
                            except Exception as e:
                                print(f"[ERROR] Exception while sending Telegram for {symbol_val} (3min): {e}", flush=True)
                        else:
                            print(f"[INFO] DRY_RUN enabled - simulated send for {symbol_val} (3min). Not setting cooldown.", flush=True)
                else:
                    print(f"[INFO] No valid 3min signal for {symbol}.", flush=True)
            except Exception as e:
                print(f"[ERROR] Exception in processing 3min for {symbol}: {e}", flush=True)

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

                        # AUTHORITATIVE: Force unconditional SHORT bypass here
                        if bias == "SHORT":
                            print("[SuperGK][MAIN] SHORT UNCONDITIONAL BYPASS active — forcing SuperGK PASSED for SHORT signals", flush=True)
                            super_gk_ok = True
                        else:
                            super_gk_ok = main_supergk_ok(bias, orderbook_result, density_result, res5)

                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok}", flush=True)

                        if not super_gk_ok:
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

                        print(f"[LOG] Sending 5min alert for {res5.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()
                        entry_price = get_live_entry_price(
                            res5.get("symbol"),
                            bias,
                            tf=res5.get("tf"),
                            slippage=DEFAULT_SLIPPAGE
                        ) or res5.get("price", 0.0)

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

                        tp_sl = calculate_tp_sl(df5, entry_price, signal_type)
                        tp = tp_sl.get('tp')
                        sl = tp_sl.get('sl')
                        fib_levels = tp_sl.get('fib_levels')

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

                        regime = sf5._market_regime() if hasattr(sf5, "_market_regime") else None

                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
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
                                if sent_ok:
                                    last_sent[key5] = now
                                else:
                                    print(f"[ERROR] Telegram send failed for {symbol_val} (5min). Not setting cooldown.", flush=True)
                            except Exception as e:
                                print(f"[ERROR] Exception while sending Telegram for {symbol_val} (5min): {e}", flush=True)
                        else:
                            print(f"[INFO] DRY_RUN enabled - simulated send for {symbol_val} (5min). Not setting cooldown.", flush=True)
                else:
                    print(f"[INFO] No valid 5min signal for {symbol}.", flush=True)
            except Exception as e:
                print(f"[ERROR] Exception in processing 5min for {symbol}: {e}", flush=True)

        except Exception as e:
            print(f"[FATAL] Exception processing {symbol}: {e}", flush=True)

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
