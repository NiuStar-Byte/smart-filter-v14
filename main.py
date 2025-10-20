# main.py

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

# Run SmartFilter diagnostics before starting main signal loop
# run_all_filter_tests()  # <-- Add this line

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
    "AVNT-USDT", "ORDER-USDT"
 ]

# TOKENS = [
#    "BMT-USDT", "ZORA-USDT", "X-USDT", "EPT-USDT", "ELDE-USDT",
#    "ACTSOL-USDT", "CROSS-USDT", "KNC-USDT", "AIN-USDT", "ARK-USDT",
#    "PORTAL-USDT", "ICNT-USDT", "OMNI-USDT", "ENA-USDT", "ARB-USDT", 
#    "AUCTION-USDT", "ROAM-USDT", "ERA-USDT", "FUEL-USDT", "TUT-USDT", 
#    "SKATE-USDT", "LA-USDT", "HIPPO-USDT", "VOXEL-USDT", "DUCK-USDT",
#    "GALA-USDT", "FUN-USDT"
# ]

COOLDOWN = {"3min": 60, "5min": 60}
last_sent = {}

PEC_BARS = 5
PEC_WINDOW_MINUTES = 720
OHLCV_LIMIT = 1000

def get_local_wib(dt):
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)
    return dt.tz_localize('UTC').tz_convert('Asia/Jakarta').replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')

def get_resting_order_density(symbol, depth=100, band_pct=0.01):  # PATCHED: band_pct default is now 0.01
    try:
        from kucoin_orderbook import fetch_orderbook
        bids, asks = fetch_orderbook(symbol, depth)
        if bids is None or asks is None or len(bids) == 0 or len(asks) == 0:
            return {'bid_density': 0.0, 'ask_density': 0.0, 'bid_levels': 0, 'ask_levels': 0, 'midprice': None}
        best_bid = bids['price'].iloc[0]
        best_ask = asks['price'].iloc[0]
        midprice = (best_bid + best_ask) / 2
        low, high = midprice * (1 - band_pct), midprice * (1 + band_pct)
        bids_in_band = bids[bids['price'] >= low]
        asks_in_band = asks[asks['price'] <= high]
        bid_density = bids_in_band['size'].sum() / max(len(bids_in_band), 1)
        ask_density = asks_in_band['size'].sum() / max(len(asks_in_band), 1)
        return {'bid_density': float(bid_density), 'ask_density': float(ask_density),
                'bid_levels': len(bids_in_band), 'ask_levels': len(asks_in_band), 'midprice': float(midprice)}
    except Exception:
        return {'bid_density': 0.0, 'ask_density': 0.0, 'bid_levels': 0, 'ask_levels': 0, 'midprice': None}

def candle_color(open, close):
    return 'green' if close > open else 'red'

def early_breakout(df, lookback=3):
    if len(df) < lookback:
        return {'valid_signal': False, 'bias': None, 'price': None}
    
    # Check if last 'lookback' candles are all the same color
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
        result = get_order_wall_delta(symbol)
        print(
            f"[OrderBookDeltaLog] {symbol} | "
            f"buy_wall={result.get('buy_wall',0)} | "
            f"sell_wall={result.get('sell_wall',0)} | "
            f"wall_delta={result.get('wall_delta',0)} | "
            f"midprice={result.get('midprice','N/A')}",
            flush=True
        )
    except Exception as e:
        print(f"[OrderBookDeltaLog] {symbol} ERROR: {e}", flush=True)
    try:
        dens = get_resting_order_density(symbol)
        print(
            f"[RestingOrderDensityLog] {symbol} | "
            f"bid_density={dens.get('bid_density',0):.2f} | ask_density={dens.get('ask_density',0):.2f} | "
            f"bid_levels={dens.get('bid_levels',0)} | ask_levels={dens.get('ask_levels',0)} | midprice={dens.get('midprice','N/A')}",
            flush=True
        )
    except Exception as e:
        print(f"[RestingOrderDensityLog] {symbol} ERROR: {e}", flush=True)

def super_gk_aligned(bias, orderbook_result, density_result):
    wall_delta = orderbook_result.get('wall_delta', 0) if orderbook_result else 0
    orderbook_bias = "LONG" if wall_delta > 0 else "SHORT" if wall_delta < 0 else "NEUTRAL"
    bid_density = density_result.get('bid_density', 0) if density_result else 0
    ask_density = density_result.get('ask_density', 0) if density_result else 0
    density_bias = "LONG" if bid_density > ask_density else "SHORT" if ask_density > bid_density else "NEUTRAL"
    if (orderbook_bias != "NEUTRAL" and bias != orderbook_bias): return False
    if (density_bias != "NEUTRAL" and bias != density_bias): return False
    if orderbook_bias == "NEUTRAL" or density_bias == "NEUTRAL": return False
    return True

def run():
    print("[INFO] Starting Smart Filter engine (LIVE MODE)...\n", flush=True)
    while True:
        try:
            # Run filter diagnostics every cycle (VWAP Divergence debug will appear each time)
            # run_all_filter_tests()
            now = time.time()
            valid_debugs = []
            pec_candidates = []

            for idx, symbol in enumerate(TOKENS, start=1):
                print(f"[INFO] Checking {symbol}...\n", flush=True)
                df3 = get_ohlcv(symbol, interval="3min", limit=OHLCV_LIMIT)
                df5 = get_ohlcv(symbol, interval="5min", limit=OHLCV_LIMIT)
                if df3 is None or df3.empty or df5 is None or df5.empty:
                    continue

                # --- Early Breakout Checks ---
                early_breakout_3m = early_breakout(df3, lookback=3)
                early_breakout_5m = early_breakout(df5, lookback=3)

                # --- 3min TF ---
                try:
                    key3 = f"{symbol}_3min"
                    sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=df5, tf="3min")
                    regime3 = sf3._market_regime()
                    res3 = sf3.analyze()
                    if isinstance(res3, dict) and res3.get("valid_signal") is True:
                        last3 = last_sent.get(key3, 0)
                        if now - last3 >= COOLDOWN["3min"]:
                            numbered_signal = f"{idx}.A"
                            log_orderbook_and_density(symbol)
                            orderbook_result = get_order_wall_delta(symbol)
                            density_result = get_resting_order_density(symbol)
                            bias = res3.get("bias", "NEUTRAL")
                            sf3.bias = bias
                            if not super_gk_aligned(bias, orderbook_result, density_result):
                                print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                                continue
                            print(f"[LOG] Sending 3min alert for {res3.get('symbol')}", flush=True)

                            fired_time_utc = datetime.utcnow()
                            entry_price = get_live_entry_price(
                                res3.get("symbol"),
                                bias,
                                tf=res3.get("tf"),
                                slippage=DEFAULT_SLIPPAGE
                            ) or res3.get("price", 0.0)

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

                            # --- NEW: Calculate TP/SL ---
                            tp_sl = calculate_tp_sl(df3, entry_price, signal_type)
                            tp = tp_sl.get('tp')
                            sl = tp_sl.get('sl')
                            fib_levels = tp_sl.get('fib_levels')  # optional, for debug/log
                            
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
                                "early_breakout_3m": early_breakout_3m,  # <-- Add this line
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

                            # --- Regime is defined here using sf3 ---
                            regime = sf3._market_regime() if hasattr(sf3, "_market_regime") else None
        
                            if os.getenv("DRY_RUN", "false").lower() != "true":
                                send_telegram_alert(
                                    numbered_signal=numbered_signal,
                                    symbol=symbol_val,
                                    signal_type=signal_type,
                                    Route=Route,
                                    price=entry_price,
                                    tf=tf_val,
                                    score=score,
                                    score_max=score_max,
                                    passed=passes,
                                    gatekeepers_total=gatekeepers_total,
                                    confidence=confidence,
                                    weighted=passed_weight,
                                    total_weight=total_weight,
                                    reversal_side=res3.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_3m=early_breakout_3m,  # <-- Pass to alert (if supported)
                                    tp=tp,       # NEW ARGUMENT
                                    sl=sl        # NEW ARGUMENT
                                )
                            last_sent[key3] = now
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
                    if isinstance(res5, dict) and res5.get("valid_signal") is True:
                        last5 = last_sent.get(key5, 0)
                        if now - last5 >= COOLDOWN["5min"]:
                            numbered_signal = f"{idx}.B"
                            log_orderbook_and_density(symbol)
                            orderbook_result = get_order_wall_delta(symbol)
                            density_result = get_resting_order_density(symbol)
                            bias = res5.get("bias", "NEUTRAL")
                            sf5.bias = bias
                            if not super_gk_aligned(bias, orderbook_result, density_result):
                                print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
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

                            # --- NEW: Calculate TP/SL ---
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
                                "early_breakout_5m": early_breakout_5m,  # <-- Add this line
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

                            # --- Regime is defined here using sf5 ---
                            regime = sf5._market_regime() if hasattr(sf5, "_market_regime") else None

                            if os.getenv("DRY_RUN", "false").lower() != "true":
                                send_telegram_alert(
                                    numbered_signal=numbered_signal,
                                    symbol=symbol_val,
                                    signal_type=signal_type,
                                    Route=Route,
                                    price=entry_price,
                                    tf=tf_val,
                                    score=score,
                                    score_max=score_max,
                                    passed=passes,
                                    gatekeepers_total=gatekeepers_total,
                                    confidence=confidence,
                                    weighted=passed_weight,
                                    total_weight=total_weight,
                                    reversal_side=res5.get("reversal_side"),
                                    regime=regime,
                                    early_breakout_5m=early_breakout_5m,  # <-- Pass to alert (if supported)
                                    tp=tp,       # NEW ARGUMENT
                                    sl=sl        # NEW ARGUMENT
                                )
                            last_sent[key5] = now
                    else:
                        print(f"[INFO] No valid 5min signal for {symbol}.", flush=True)
                except Exception as e:
                    print(f"[ERROR] Exception in processing 5min for {symbol}: {e}", flush=True)

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
                else:
                    print("[FIRED] valid_debugs is empty — no debug files to send to Telegram.", flush=True)
            except Exception as e:
                print(f"[FATAL] Exception in debug sending block: {e}", flush=True)

            if valid_debugs:
                print(f"[FIRED] Processed {len(valid_debugs)} valid signals this cycle", flush=True)
            else:
                print("[FIRED] No valid signals processed this cycle", flush=True)

            print("[INFO] ✅ Cycle complete. Sleeping 60 seconds...\n", flush=True)
            time.sleep(60)

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
        # run_all_filter_tests()  # <--- THIS RUNS DIAGNOSTICS ONCE BEFORE LIVE LOOP
        run()
