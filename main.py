# main.py

print("[INFO] main.py script started.", flush=True)
from signal_debug_log import export_signal_debug_txt, log_fired_signal
import os
import time
import pandas as pd
import random
import pytz
from datetime import datetime

from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert, send_telegram_file
from kucoin_orderbook import get_order_wall_delta
from pec_engine import run_pec_check, export_pec_log

TOKENS = [
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT",
    "CROSS-USDT", "KNC-USDT", "AIN-USDT", "ARK-USDT", "PORTAL-USDT",
    "BANANAS31-USDT", "ICNT-USDT", "OMNI-USDT", "PARTI-USDT", "VINE-USDT",
    "DUCK-USDT", "AUCTION-USDT", "ROAM-USDT", "FUEL-USDT", "TUT-USDT",
    "VOXEL-USDT", "ALU-USDT", "TURBO-USDT", "PROMPT-USDT", "HIPPO-USDT"
]
COOLDOWN = {"3min": 540, "5min": 900}
last_sent = {}

PEC_BARS = 5
PEC_WINDOW_MINUTES = 720
OHLCV_LIMIT = 1000

def get_local_wib(dt):
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)
    return dt.tz_localize('UTC').tz_convert('Asia/Jakarta').replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')

def get_resting_order_density(symbol, depth=100, band_pct=0.005):
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

def log_orderbook_and_density(symbol):
    try:
        result = get_order_wall_delta(symbol)
        print(
            f"[OrderBookDeltaLog] {symbol} | "
            f"buy_wall={result['buy_wall']} | "
            f"sell_wall={result['sell_wall']} | "
            f"wall_delta={result['wall_delta']} | "
            f"midprice={result['midprice']}",
            flush=True
        )
    except Exception as e:
        print(f"[OrderBookDeltaLog] {symbol} ERROR: {e}", flush=True)
    try:
        dens = get_resting_order_density(symbol)
        print(
            f"[RestingOrderDensityLog] {symbol} | "
            f"bid_density={dens['bid_density']:.2f} | ask_density={dens['ask_density']:.2f} | "
            f"bid_levels={dens['bid_levels']} | ask_levels={dens['ask_levels']} | midprice={dens['midprice']}",
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
            now = time.time()
            valid_debugs = []
            pec_candidates = []

            for idx, symbol in enumerate(TOKENS, start=1):
                print(f"[INFO] Checking {symbol}...\n", flush=True)
                df3 = get_ohlcv(symbol, interval="3min", limit=OHLCV_LIMIT)
                df5 = get_ohlcv(symbol, interval="5min", limit=OHLCV_LIMIT)
                if df3 is None or df3.empty or df5 is None or df5.empty:
                    continue

                # --- 3min TF ---
                try:
                    key3 = f"{symbol}_3min"
                    sf3 = SmartFilter(symbol, df3, df3m=df3, df5m=df5, tf="3min")
                    res3 = sf3.analyze()
                    if isinstance(res3, dict) and res3.get("valid_signal") is True:
                        last3 = last_sent.get(key3, 0)
                        if now - last3 >= COOLDOWN["3min"]:
                            numbered_signal = f"{idx}.A"
                            log_orderbook_and_density(symbol)
                            orderbook_result = get_order_wall_delta(symbol)
                            density_result = get_resting_order_density(symbol)
                            bias = res3.get("bias", "NEUTRAL")
                            sf3.bias = bias  # Set bias for property
                            if not super_gk_aligned(bias, orderbook_result, density_result):
                                print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                                continue
                            print(f"[LOG] Sending 3min alert for {res3['symbol']}", flush=True)

                            # Get the current UTC time as the fired time
                            fired_time_utc = datetime.utcnow()
                            
                            valid_debugs.append({
                                "symbol": res3["symbol"],
                                "tf": res3["tf"],
                                "bias": res3["bias"],
                                "filter_weights_long": sf3.filter_weights_long,
                                "filter_weights_short": sf3.filter_weights_short,
                                "gatekeepers": sf3.gatekeepers,
                                "results_long": res3.get("filter_results_long", {}),
                                "results_short": res3.get("filter_results_short", {}),
                                "caption": f"Signal debug log for {res3.get('symbol')} {res3.get('tf')}",
                                "orderbook_result": orderbook_result,
                                "density_result": density_result,
                                "entry_price": res3.get("price"),
                                "fired_time_utc": fired_time_utc
                            })
                            
                            # DEPRECATED: Still calculate entry_idx for backward compatibility
                            entry_idx = df3.index.get_loc(df3.index[-1])
                            
                            pec_candidates.append(
                                ("3min", symbol, res3.get("price"), bias, df3, fired_time_utc)  # Use fired_time instead of entry_idx
                            )
                            
                            # Log with both fired_time (primary) and entry_idx (deprecated compatibility)
                            log_fired_signal(
                                symbol=symbol,
                                tf="3min",
                                signal_type=res3.get("bias"),
                                entry_idx=entry_idx,
                                fired_time=fired_time_utc,
                                score=res3.get("score"),
                                max_score=res3.get("score_max"),
                                passed=res3.get("passes"),
                                max_passed=res3.get("gatekeepers_total"),
                                weights=res3.get("passed_weight"),
                                max_weights=res3.get("total_weight"),
                                confidence_rate=res3.get("confidence")
                            )
                            if os.getenv("DRY_RUN", "false").lower() != "true":
                                send_telegram_alert(
                                    numbered_signal=numbered_signal,
                                    symbol=res3.get("symbol"),
                                    signal_type=res3.get("bias"),
                                    price=res3.get("price"),
                                    tf=res3.get("tf"),
                                    score=res3.get("score"),
                                    score_max=res3.get("score_max"),
                                    passed=res3.get("passes"),
                                    gatekeepers_total=res3.get("gatekeepers_total"),
                                    confidence=res3.get("confidence"),
                                    weighted=res3.get("passed_weight"),
                                    total_weight=res3.get("total_weight")
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
                    res5 = sf5.analyze()
                    if isinstance(res5, dict) and res5.get("valid_signal") is True:
                        last5 = last_sent.get(key5, 0)
                        if now - last5 >= COOLDOWN["5min"]:
                            numbered_signal = f"{idx}.B"
                            log_orderbook_and_density(symbol)
                            orderbook_result = get_order_wall_delta(symbol)
                            density_result = get_resting_order_density(symbol)
                            bias = res5.get("bias", "NEUTRAL")
                            sf5.bias = bias  # Set bias for property
                            if not super_gk_aligned(bias, orderbook_result, density_result):
                                print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT", flush=True)
                                continue
                            print(f"[LOG] Sending 5min alert for {res5['symbol']}", flush=True)

                            # Get the current UTC time as the fired time
                            fired_time_utc = datetime.utcnow()
                            
                            valid_debugs.append({
                                "symbol": res5["symbol"],
                                "tf": res5["tf"],
                                "bias": res5["bias"],
                                "filter_weights_long": sf5.filter_weights_long,
                                "filter_weights_short": sf5.filter_weights_short,
                                "gatekeepers": sf5.gatekeepers,
                                "results_long": res5.get("filter_results_long", {}),
                                "results_short": res5.get("filter_results_short", {}),
                                "caption": f"Signal debug log for {res5.get('symbol')} {res5.get('tf')}",
                                "orderbook_result": orderbook_result,
                                "density_result": density_result,
                                "entry_price": res5.get("price"),
                                "fired_time_utc": fired_time_utc
                            })
                            
                            # DEPRECATED: Still calculate entry_idx for backward compatibility
                            entry_idx = df5.index.get_loc(df5.index[-1])
                            
                            pec_candidates.append(
                                ("5min", symbol, res5.get("price"), bias, df5, fired_time_utc)  # Use fired_time instead of entry_idx
                            )
                            
                            # Log with both fired_time (primary) and entry_idx (deprecated compatibility)
                            log_fired_signal(
                                symbol=symbol,
                                tf="5min",
                                signal_type=res5.get("bias"),
                                entry_idx=entry_idx,  # Kept for backward compatibility
                                fired_time=fired_time_utc,  # Primary timestamp-based approach
                                score=res5.get("score"),
                                max_score=res5.get("score_max"),
                                passed=res5.get("passes"),
                                max_passed=res5.get("gatekeepers_total"),
                                weights=res5.get("passed_weight"),
                                max_weights=res5.get("total_weight"),
                                confidence_rate=res5.get("confidence")
                            )
                            if os.getenv("DRY_RUN", "false").lower() != "true":
                                send_telegram_alert(
                                    numbered_signal=numbered_signal,
                                    symbol=res5.get("symbol"),
                                    signal_type=res5.get("bias"),
                                    price=res5.get("price"),
                                    tf=res5.get("tf"),
                                    score=res5.get("score"),
                                    score_max=res5.get("score_max"),
                                    passed=res5.get("passes"),
                                    gatekeepers_total=res5.get("gatekeepers_total"),
                                    confidence=res5.get("confidence"),
                                    weighted=res5.get("passed_weight"),
                                    total_weight=res5.get("total_weight")
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
                            # Print filter weights for transparency
                            print("[FIRED] LONG filter weights:", debug_info["filter_weights_long"], flush=True)
                            print("[FIRED] SHORT filter weights:", debug_info["filter_weights_short"], flush=True)

                            # Use only the new debug txt export function
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
                                # filename="signal_debug_temp.txt"
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

            # Print summary of fired signals (log-based, no CSV dependency)
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


print(">>> ENTERED main.py", flush=True)
if __name__ == "__main__":
    print(">>> __main__ block executing", flush=True)
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

