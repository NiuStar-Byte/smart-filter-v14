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
# Full liquid pairs from kucoin_orderbook.py - 90+ symbols
TOKENS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "AVAX-USDT", 
    "XLM-USDT", "LINK-USDT", "POL-USDT", "BNB-USDT", "SKATE-USDT", "LA-USDT", 
    "SPK-USDT", "ZKJ-USDT", "IP-USDT", "AERO-USDT", "BMT-USDT", "LQTY-USDT", 
    "X-USDT", "RAY-USDT", "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", 
    "FUN-USDT", "CROSS-USDT", "KNC-USDT", "AIN-USDT", "ARK-USDT", "PORTAL-USDT", 
    "ICNT-USDT", "OMNI-USDT", "PARTI-USDT", "VINE-USDT", "ZORA-USDT", "DUCK-USDT", 
    "AUCTION-USDT", "ROAM-USDT", "FUEL-USDT", "TUT-USDT", "VOXEL-USDT", "ALU-USDT", 
    "TURBO-USDT", "PROMPT-USDT", "HIPPO-USDT", "DOGE-USDT", "ALGO-USDT", "DOT-USDT", 
    "NEWT-USDT", "SAHARA-USDT", "PEPE-USDT", "ERA-USDT", "PENGU-USDT", "CFX-USDT", 
    "ENA-USDT", "SUI-USDT", "EIGEN-USDT", "UNI-USDT", "HYPE-USDT", "TON-USDT", 
    "KAS-USDT", "HBAR-USDT", "ONDO-USDT", "VIRTUAL-USDT", "AAVE-USDT", "GALA-USDT", 
    "PUMP-USDT", "WIF-USDT", "BERA-USDT", "DYDX-USDT", "KAITO-USDT", "ARKM-USDT", 
    "ATH-USDT", "NMR-USDT", "ARB-USDT", "WLFI-USDT", "BIO-USDT", "ASTER-USDT", 
    "XPL-USDT", "AVNT-USDT", "ORDER-USDT", "XAUT-USDT"
]

COOLDOWN = {"15min": 120, "30min": 240, "1h": 600}
last_sent = {}

# NOTE: Updated to use 15m, 30m, 1h (removed 3m, 5m - too noisy)
# Based on Test-3 results: 15-min markets are optimal, longer timeframes reduce false signals
PEC_BARS = 5
PEC_WINDOW_MINUTES = 720
OHLCV_LIMIT = 250  # Optimized: covers EMA200 + safety buffer, reduces API calls

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
    """
    Fetch fresh orderbook and resting density and print timestamped logs.
    Adds ts (UTC) and source=main_log so you can confirm these logs were fetched
    immediately prior to sending a signal (not a startup/import dump).
    """
    from datetime import datetime
    ts = datetime.utcnow().isoformat()
    try:
        ob = get_order_wall_delta(symbol)
        buy_wall = float(ob.get('buy_wall', 0.0))
        sell_wall = float(ob.get('sell_wall', 0.0))
        wall_delta = float(ob.get('wall_delta', 0.0))
        midprice = ob.get('midprice', 'N/A')
        print(
            f"[OrderBookDeltaLog] {symbol} | ts={ts} | source=main_log | "
            f"buy_wall={buy_wall} | sell_wall={sell_wall} | wall_delta={wall_delta} | midprice={midprice}",
            flush=True
        )
    except Exception as e:
        print(f"[OrderBookDeltaLog] {symbol} ERROR: {e}", flush=True)

    try:
        dens = get_resting_order_density(symbol)
        print(
            f"[RestingOrderDensityLog] {symbol} | ts={ts} | source=main_log | "
            f"bid_density={dens.get('bid_density',0.0):.2f}% | ask_density={dens.get('ask_density',0.0):.2f}% | "
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
# NOTE: this module expects these symbols to be available in the import environment:
# TOKENS, OHLCV_LIMIT, COOLDOWN, last_sent, DEFAULT_SLIPPAGE
# get_ohlcv, SmartFilter, early_breakout, get_live_entry_price,
# get_order_wall_delta, get_resting_order_density, calculate_tp_sl,
# log_fired_signal, send_telegram_alert
# These are kept external so the function can be copy-pasted into your existing module.

import traceback
from datetime import datetime
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
            # Fetch raw OHLCV for 15m, 30m, and 1h timeframes (removed 3m & 5m - too noisy)
            df15 = get_ohlcv(symbol, interval="15min", limit=OHLCV_LIMIT)
            df30 = get_ohlcv(symbol, interval="30min", limit=OHLCV_LIMIT)
            df1h = get_ohlcv(symbol, interval="1h", limit=OHLCV_LIMIT)
            if df15 is None or df15.empty or df30 is None or df30.empty or df1h is None or df1h.empty:
                print(f"[WARN] Not enough data for {symbol} (df15, df30, or df1h empty). Skipping.", flush=True)
                continue

            early_breakout_15m = early_breakout(df15, lookback=3)
            early_breakout_30m = early_breakout(df30, lookback=3)
            early_breakout_1h = early_breakout(df1h, lookback=3)

            # --- 15min TF block ---
            try:
                key15 = f"{symbol}_15min"
                sf15 = SmartFilter(symbol, df15, df3m=df15, df5m=df30, tf="15min")
                regime15 = sf15._market_regime()
                res15 = sf15.analyze()

                if isinstance(res15, dict) and res15.get("filters_ok") is True:
                    last15 = last_sent.get(key15, 0)
                    if now - last15 >= COOLDOWN["15min"]:
                        numbered_signal = f"{idx}.A"

                        # Fresh orderbook/density logs (main_log)
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

                        bias = res15.get("bias", "NEUTRAL")
                        sf15.bias = bias

                        # RE-ENABLED: SuperGK validation for all signals (FIX: 2026-02-22)
                        print("[SuperGK][MAIN] RE-ENABLED - validating SuperGK for ALL signals (LONG and SHORT)", flush=True)
                        try:
                            super_gk_ok = super_gk_aligned(bias, orderbook_result, density_result)
                        except Exception as e:
                            print(f"[SuperGK][MAIN] Error: {e}, defaulting to True for safety", flush=True)
                            super_gk_ok = True
                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok}", flush=True)

                        if not super_gk_ok:
                            # legacy compatibility block (should not execute with global bypass)
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                            valid_debugs.append({
                                "symbol": symbol,
                                "tf": "15min",
                                "bias": bias,
                                "filter_weights_long": getattr(sf15, 'filter_weights_long', []),
                                "filter_weights_short": getattr(sf15, 'filter_weights_short', []),
                                "gatekeepers": getattr(sf15, 'gatekeepers', []),
                                "results_long": res15.get("results_long", {}),
                                "results_short": res15.get("results_short", {}),
                                "caption": f"Blocked signal debug for {symbol} 15min",
                                "orderbook_result": orderbook_result,
                                "density_result": density_result,
                                "entry_price": res15.get("price"),
                                "fired_time_utc": datetime.utcnow(),
                                "early_breakout_15m": early_breakout_15m
                            })
                            continue

                        # --- Prepare to send ---
                        print(f"[LOG] Sending 15min alert for {res15.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()

                        # Live entry price preferred, fall back to analyzer price
                        entry_price_raw = None
                        try:
                            entry_price_raw = get_live_entry_price(
                                res15.get("symbol"),
                                bias,
                                tf=res15.get("tf"),
                                slippage=DEFAULT_SLIPPAGE
                            ) or res15.get("price", 0.0)
                        except Exception as e:
                            print(f"[WARN] get_live_entry_price raised: {e} -- falling back to analyzer price", flush=True)
                            entry_price_raw = res15.get("price", 0.0)

                        # Coerce defensively
                        try:
                            entry_price = float(entry_price_raw)
                        except Exception:
                            try:
                                entry_price = float(str(entry_price_raw))
                                print(f"[WARN] Coerced entry_price from {entry_price_raw} to {entry_price}", flush=True)
                            except Exception:
                                print(f"[WARN] Failed to coerce entry_price ({entry_price_raw}); defaulting to 0.0", flush=True)
                                entry_price = 0.0

                        # Debug: final entry price vs analyzer price
                        print(f"[DEBUG] entry_price_final={entry_price:.8f} analyzer_price={res15.get('price')}", flush=True)

                        # Collect signal metadata
                        score = res15.get("score", 0)
                        score_max = res15.get("score_max", 0)
                        passes = res15.get("passes", 0)
                        gatekeepers_total = res15.get("gatekeepers_total", 0)
                        passed_weight = res15.get("passed_weight", 0.0)
                        total_weight = res15.get("total_weight", 0.0)
                        Route = res15.get("Route", None)
                        signal_type = bias
                        tf_val = res15.get("tf", "15min")
                        symbol_val = res15.get("symbol", symbol)
                        try:
                            confidence = round((passed_weight / total_weight) * 100, 1) if total_weight else 0.0
                        except Exception:
                            confidence = 0.0
                        entry_idx = df15.index.get_loc(df15.index[-1])

                        # Recompute TP/SL against the final live entry_price so TP/SL match execution price
                        tp_sl = None
                        tp = None
                        sl = None
                        fib_levels = None
                        try:
                            tp_sl = calculate_tp_sl(df15, entry_price, signal_type)
                            tp = tp_sl.get('tp')
                            sl = tp_sl.get('sl')
                            fib_levels = tp_sl.get('fib_levels')
                            # Debug: show tp_sl summary
                            print(f"[DEBUG] calculate_tp_sl -> tp={tp} sl={sl} chosen_ratio={tp_sl.get('chosen_ratio')} achieved_rr={tp_sl.get('achieved_rr')}", flush=True)
                            print(f"[DEBUG] tp_sl full dict for {symbol_val}: {tp_sl}", flush=True)
                        except Exception as e:
                            print(f"[WARN] calculate_tp_sl failed: {e}", flush=True)
                            tp = sl = fib_levels = None
                            tp_sl = None

                        # Append debug object used for exporting debug files
                        valid_debugs.append({
                            "symbol": symbol_val,
                            "tf": tf_val,
                            "bias": bias,
                            "filter_weights_long": getattr(sf15, 'filter_weights_long', []),
                            "filter_weights_short": getattr(sf15, 'filter_weights_short', []),
                            "gatekeepers": getattr(sf15, 'gatekeepers', []),
                            "results_long": res15.get("results_long", {}),
                            "results_short": res15.get("results_short", {}),
                            "caption": f"Signal debug log for {symbol_val} {tf_val}",
                            "orderbook_result": orderbook_result,
                            "density_result": density_result,
                            "entry_price": entry_price,
                            "fired_time_utc": fired_time_utc,
                            "early_breakout_15m": early_breakout_15m,
                            "tp": tp,
                            "sl": sl,
                            "fib_levels": fib_levels
                        })

                        # Log fired signal (tracking) and then send telegram
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

                        regime = sf15._market_regime() if hasattr(sf15, "_market_regime") else None

                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                print(f"[DEBUG] send_telegram_alert about to send: symbol={symbol_val} price={entry_price} (type={type(entry_price)}) tp={tp} sl={sl} tp_source={(tp_sl.get('source') if isinstance(tp_sl, dict) else None)} chosen_ratio={(tp_sl.get('chosen_ratio') if isinstance(tp_sl, dict) else None)} achieved_rr={(tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None)}", flush=True)
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
                                    reversal_side=res15.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_15m=early_breakout_15m,
                                    tp=tp,
                                    sl=sl,
                                    tp_sl=tp_sl,
                                    chosen_ratio=(tp_sl.get('chosen_ratio') if isinstance(tp_sl, dict) else None),
                                    achieved_rr=(tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None)
                                )
                                print(f"[DEBUG] send_telegram_alert returned: {sent_ok}", flush=True)
                                if sent_ok:
                                    last_sent[key15] = now
                                else:
                                    print(f"[ERROR] Telegram send failed for {symbol_val} (15min). Not setting cooldown.", flush=True)
                            except Exception as e:
                                print(f"[ERROR] Exception during Telegram send for {symbol_val} (15min): {e}", flush=True)
                                traceback.print_exc()
                        else:
                            print(f"[INFO] DRY_RUN enabled - simulated send for {symbol_val} (15min). Not setting cooldown.", flush=True)
                else:
                    print(f"[INFO] No valid 15min signal for {symbol}.", flush=True)
            except Exception as e:
                print(f"[ERROR] Exception in processing 15min for {symbol}: {e}", flush=True)
                traceback.print_exc()

            # --- 30min TF block (mirror of 15min with identical safe flow) ---
            try:
                key30 = f"{symbol}_30min"
                sf30 = SmartFilter(symbol, df30, df3m=df15, df5m=df1h, tf="30min")
                regime30 = sf30._market_regime()
                res30 = sf30.analyze()

                if isinstance(res30, dict) and res30.get("filters_ok") is True:
                    last30 = last_sent.get(key30, 0)
                    if now - last30 >= COOLDOWN["30min"]:
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

                        bias = res30.get("bias", "NEUTRAL")
                        sf30.bias = bias

                        # RE-ENABLED: SuperGK validation for all signals (FIX: 2026-02-22)
                        print("[SuperGK][MAIN] RE-ENABLED - validating SuperGK for ALL signals (LONG and SHORT)", flush=True)
                        try:
                            super_gk_ok = super_gk_aligned(bias, orderbook_result, density_result)
                        except Exception as e:
                            print(f"[SuperGK][MAIN] Error: {e}, defaulting to True for safety", flush=True)
                            super_gk_ok = True
                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok}", flush=True)

                        if not super_gk_ok:
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                            valid_debugs.append({
                                "symbol": symbol,
                                "tf": "30min",
                                "bias": bias,
                                "filter_weights_long": getattr(sf30, 'filter_weights_long', []),
                                "filter_weights_short": getattr(sf30, 'filter_weights_short', []),
                                "gatekeepers": getattr(sf30, 'gatekeepers', []),
                                "results_long": res30.get("results_long", {}),
                                "results_short": res30.get("results_short", {}),
                                "caption": f"Blocked signal debug for {symbol} 30min",
                                "orderbook_result": orderbook_result,
                                "density_result": density_result,
                                "entry_price": res30.get("price"),
                                "fired_time_utc": datetime.utcnow(),
                                "early_breakout_30m": early_breakout_30m
                            })
                            continue

                        print(f"[LOG] Sending 30min alert for {res30.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()

                        entry_price_raw = None
                        try:
                            entry_price_raw = get_live_entry_price(
                                res30.get("symbol"),
                                bias,
                                tf=res30.get("tf"),
                                slippage=DEFAULT_SLIPPAGE
                            ) or res30.get("price", 0.0)
                        except Exception as e:
                            print(f"[WARN] get_live_entry_price raised: {e} -- falling back to analyzer price", flush=True)
                            entry_price_raw = res30.get("price", 0.0)

                        try:
                            entry_price = float(entry_price_raw)
                        except Exception:
                            try:
                                entry_price = float(str(entry_price_raw))
                                print(f"[WARN] Coerced entry_price from {entry_price_raw} to {entry_price}", flush=True)
                            except Exception:
                                print(f"[WARN] Failed to coerce entry_price ({entry_price_raw}); defaulting to 0.0", flush=True)
                                entry_price = 0.0

                        print(f"[DEBUG] entry_price_final={entry_price:.8f} analyzer_price={res30.get('price')}", flush=True)

                        score = res30.get("score", 0)
                        score_max = res30.get("score_max", 0)
                        passes = res30.get("passes", 0)
                        gatekeepers_total = res30.get("gatekeepers_total", 0)
                        passed_weight = res30.get("passed_weight", 0.0)
                        total_weight = res30.get("total_weight", 0.0)
                        Route = res30.get("Route", None)
                        signal_type = bias
                        tf_val = res30.get("tf", "30min")
                        symbol_val = res30.get("symbol", symbol)
                        try:
                            confidence = round((passed_weight / total_weight) * 100, 1) if total_weight else 0.0
                        except Exception:
                            confidence = 0.0
                        entry_idx = df30.index.get_loc(df30.index[-1])

                        tp_sl = None
                        tp = None
                        sl = None
                        fib_levels = None
                        try:
                            tp_sl = calculate_tp_sl(df30, entry_price, signal_type)
                            tp = tp_sl.get('tp')
                            sl = tp_sl.get('sl')
                            fib_levels = tp_sl.get('fib_levels')
                            print(f"[DEBUG] calculate_tp_sl -> tp={tp} sl={sl} chosen_ratio={tp_sl.get('chosen_ratio')} achieved_rr={tp_sl.get('achieved_rr')}", flush=True)
                            print(f"[DEBUG] tp_sl full dict for {symbol_val}: {tp_sl}", flush=True)
                        except Exception as e:
                            print(f"[WARN] calculate_tp_sl failed: {e}", flush=True)
                            tp = sl = fib_levels = None
                            tp_sl = None

                        valid_debugs.append({
                            "symbol": symbol_val,
                            "tf": tf_val,
                            "bias": bias,
                            "filter_weights_long": getattr(sf30, 'filter_weights_long', []),
                            "filter_weights_short": getattr(sf30, 'filter_weights_short', []),
                            "gatekeepers": getattr(sf30, 'gatekeepers', []),
                            "results_long": res30.get("results_long", {}),
                            "results_short": res30.get("results_short", {}),
                            "caption": f"Signal debug log for {symbol_val} {tf_val}",
                            "orderbook_result": orderbook_result,
                            "density_result": density_result,
                            "entry_price": entry_price,
                            "fired_time_utc": fired_time_utc,
                            "early_breakout_30m": early_breakout_30m,
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

                        regime = sf30._market_regime() if hasattr(sf30, "_market_regime") else None

                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                print(f"[DEBUG] send_telegram_alert about to send: symbol={symbol_val} price={entry_price} (type={type(entry_price)}) tp={tp} sl={sl} tp_source={(tp_sl.get('source') if isinstance(tp_sl, dict) else None)} chosen_ratio={(tp_sl.get('chosen_ratio') if isinstance(tp_sl, dict) else None)} achieved_rr={(tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None)}", flush=True)
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
                                    reversal_side=res30.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_30m=early_breakout_30m,
                                    tp=tp,
                                    sl=sl,
                                    tp_sl=tp_sl,
                                    chosen_ratio=(tp_sl.get('chosen_ratio') if isinstance(tp_sl, dict) else None),
                                    achieved_rr=(tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None)
                                )
                                print(f"[DEBUG] send_telegram_alert returned: {sent_ok}", flush=True)
                                if sent_ok:
                                    last_sent[key30] = now
                                else:
                                    print(f"[ERROR] Telegram send failed for {symbol_val} (30min). Not setting cooldown.", flush=True)
                            except Exception as e:
                                print(f"[ERROR] Exception during Telegram send for {symbol_val} (30min): {e}", flush=True)
                                traceback.print_exc()
                        else:
                            print(f"[INFO] DRY_RUN enabled - simulated send for {symbol_val} (30min). Not setting cooldown.", flush=True)
                else:
                    print(f"[INFO] No valid 30min signal for {symbol}.", flush=True)
            except Exception as e:
                print(f"[ERROR] Exception in processing 30min for {symbol}: {e}", flush=True)
                traceback.print_exc()

            # --- 1h TF block (mirror of 15min with identical safe flow) ---
            try:
                key1h = f"{symbol}_1h"
                sf1h = SmartFilter(symbol, df1h, df3m=df30, df5m=None, tf="1h")
                regime1h = sf1h._market_regime()
                res1h = sf1h.analyze()

                if isinstance(res1h, dict) and res1h.get("filters_ok") is True:
                    last1h = last_sent.get(key1h, 0)
                    if now - last1h >= COOLDOWN["1h"]:
                        numbered_signal = f"{idx}.C"

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

                        bias = res1h.get("bias", "NEUTRAL")
                        sf1h.bias = bias

                        # RE-ENABLED: SuperGK validation for all signals (FIX: 2026-02-22)
                        print("[SuperGK][MAIN] RE-ENABLED - validating SuperGK for ALL signals (LONG and SHORT)", flush=True)
                        try:
                            super_gk_ok = super_gk_aligned(bias, orderbook_result, density_result)
                        except Exception as e:
                            print(f"[SuperGK][MAIN] Error: {e}, defaulting to True for safety", flush=True)
                            super_gk_ok = True
                        print(f"[SuperGK][MAIN] Result -> bias={bias} super_gk_ok={super_gk_ok}", flush=True)

                        if not super_gk_ok:
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                            valid_debugs.append({
                                "symbol": symbol,
                                "tf": "1h",
                                "bias": bias,
                                "filter_weights_long": getattr(sf1h, 'filter_weights_long', []),
                                "filter_weights_short": getattr(sf1h, 'filter_weights_short', []),
                                "gatekeepers": getattr(sf1h, 'gatekeepers', []),
                                "results_long": res1h.get("results_long", {}),
                                "results_short": res1h.get("results_short", {}),
                                "caption": f"Blocked signal debug for {symbol} 1h",
                                "orderbook_result": orderbook_result,
                                "density_result": density_result,
                                "entry_price": res1h.get("price"),
                                "fired_time_utc": datetime.utcnow(),
                                "early_breakout_1h": early_breakout_1h
                            })
                            continue

                        print(f"[LOG] Sending 1h alert for {res1h.get('symbol')}", flush=True)
                        fired_time_utc = datetime.utcnow()

                        entry_price_raw = None
                        try:
                            entry_price_raw = get_live_entry_price(
                                res1h.get("symbol"),
                                bias,
                                tf=res1h.get("tf"),
                                slippage=DEFAULT_SLIPPAGE
                            ) or res1h.get("price", 0.0)
                        except Exception as e:
                            print(f"[WARN] get_live_entry_price raised: {e} -- falling back to analyzer price", flush=True)
                            entry_price_raw = res1h.get("price", 0.0)

                        try:
                            entry_price = float(entry_price_raw)
                        except Exception:
                            try:
                                entry_price = float(str(entry_price_raw))
                                print(f"[WARN] Coerced entry_price from {entry_price_raw} to {entry_price}", flush=True)
                            except Exception:
                                print(f"[WARN] Failed to coerce entry_price ({entry_price_raw}); defaulting to 0.0", flush=True)
                                entry_price = 0.0

                        print(f"[DEBUG] entry_price_final={entry_price:.8f} analyzer_price={res1h.get('price')}", flush=True)

                        score = res1h.get("score", 0)
                        score_max = res1h.get("score_max", 0)
                        passes = res1h.get("passes", 0)
                        gatekeepers_total = res1h.get("gatekeepers_total", 0)
                        passed_weight = res1h.get("passed_weight", 0.0)
                        total_weight = res1h.get("total_weight", 0.0)
                        Route = res1h.get("Route", None)
                        signal_type = bias
                        tf_val = res1h.get("tf", "1h")
                        symbol_val = res1h.get("symbol", symbol)
                        try:
                            confidence = round((passed_weight / total_weight) * 100, 1) if total_weight else 0.0
                        except Exception:
                            confidence = 0.0
                        entry_idx = df1h.index.get_loc(df1h.index[-1])

                        tp_sl = None
                        tp = None
                        sl = None
                        fib_levels = None
                        try:
                            tp_sl = calculate_tp_sl(df1h, entry_price, signal_type)
                            tp = tp_sl.get('tp')
                            sl = tp_sl.get('sl')
                            fib_levels = tp_sl.get('fib_levels')
                            print(f"[DEBUG] calculate_tp_sl -> tp={tp} sl={sl} chosen_ratio={tp_sl.get('chosen_ratio')} achieved_rr={tp_sl.get('achieved_rr')}", flush=True)
                            print(f"[DEBUG] tp_sl full dict for {symbol_val}: {tp_sl}", flush=True)
                        except Exception as e:
                            print(f"[WARN] calculate_tp_sl failed: {e}", flush=True)
                            tp = sl = fib_levels = None
                            tp_sl = None

                        valid_debugs.append({
                            "symbol": symbol_val,
                            "tf": tf_val,
                            "bias": bias,
                            "filter_weights_long": getattr(sf1h, 'filter_weights_long', []),
                            "filter_weights_short": getattr(sf1h, 'filter_weights_short', []),
                            "gatekeepers": getattr(sf1h, 'gatekeepers', []),
                            "results_long": res1h.get("results_long", {}),
                            "results_short": res1h.get("results_short", {}),
                            "caption": f"Signal debug log for {symbol_val} {tf_val}",
                            "orderbook_result": orderbook_result,
                            "density_result": density_result,
                            "entry_price": entry_price,
                            "fired_time_utc": fired_time_utc,
                            "early_breakout_1h": early_breakout_1h,
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

                        regime = sf1h._market_regime() if hasattr(sf1h, "_market_regime") else None

                        if os.getenv("DRY_RUN", "false").lower() != "true":
                            try:
                                print(f"[DEBUG] send_telegram_alert about to send: symbol={symbol_val} price={entry_price} (type={type(entry_price)}) tp={tp} sl={sl} tp_source={(tp_sl.get('source') if isinstance(tp_sl, dict) else None)} chosen_ratio={(tp_sl.get('chosen_ratio') if isinstance(tp_sl, dict) else None)} achieved_rr={(tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None)}", flush=True)
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
                                    reversal_side=res1h.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_1h=early_breakout_1h,
                                    tp=tp,
                                    sl=sl,
                                    tp_sl=tp_sl,
                                    chosen_ratio=(tp_sl.get('chosen_ratio') if isinstance(tp_sl, dict) else None),
                                    achieved_rr=(tp_sl.get('achieved_rr') if isinstance(tp_sl, dict) else None)
                                )
                                print(f"[DEBUG] send_telegram_alert returned: {sent_ok}", flush=True)
                                if sent_ok:
                                    last_sent[key1h] = now
                                else:
                                    print(f"[ERROR] Telegram send failed for {symbol_val} (1h). Not setting cooldown.", flush=True)
                            except Exception as e:
                                print(f"[ERROR] Exception during Telegram send for {symbol_val} (1h): {e}", flush=True)
                                traceback.print_exc()
                        else:
                            print(f"[INFO] DRY_RUN enabled - simulated send for {symbol_val} (1h). Not setting cooldown.", flush=True)
                else:
                    print(f"[INFO] No valid 1h signal for {symbol}.", flush=True)
            except Exception as e:
                print(f"[ERROR] Exception in processing 1h for {symbol}: {e}", flush=True)
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
    print(">>> ENTERED main.py - LIVE SIGNAL MODE", flush=True)
    print(">>> For PEC backtesting, run: python3 run_pec_backtest.py", flush=True)
    print(">>> For historical backtest, run: python3 backtest_real.py", flush=True)
    run()
