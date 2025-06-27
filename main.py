import os
# REMOVE or COMMENT OUT this line if present:
# os.environ["PEC_BACKTEST_ONLY"] = "false"  # <--- LIVE MODE for all runs

import time
import pandas as pd
import random
import pytz
from datetime import datetime

from kucoin_data import get_ohlcv
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert, send_telegram_file
from signal_debug_log import dump_signal_debug_txt
from kucoin_orderbook import get_order_wall_delta
from pec_engine import run_pec_check, export_pec_log

# PEC backtest fires ONLY for RAY-USDT; forward mode uses all tokens as usual.
TOKENS = [
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]
PEC_ONLY_TOKEN = "RAY-USDT"  # Only RAY-USDT for PEC logs
COOLDOWN = {"3min": 720, "5min": 900}
last_sent = {}

PEC_BARS = 5
PEC_WINDOW_MINUTES = 500
OHLCV_LIMIT = 1000

def get_local_wib(dt):
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)
    return dt.tz_localize('UTC').tz_convert('Asia/Jakarta').strftime('%H:%M WIB')

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
            f"midprice={result['midprice']}"
        )
    except Exception as e:
        print(f"[OrderBookDeltaLog] {symbol} ERROR: {e}")
    try:
        dens = get_resting_order_density(symbol)
        print(
            f"[RestingOrderDensityLog] {symbol} | "
            f"bid_density={dens['bid_density']:.2f} | ask_density={dens['ask_density']:.2f} | "
            f"bid_levels={dens['bid_levels']} | ask_levels={dens['ask_levels']} | midprice={dens['midprice']}"
        )
    except Exception as e:
        print(f"[RestingOrderDensityLog] {symbol} ERROR: {e}")

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

def backtest_pec_simulation():
    """
    Runs PEC backtest on RAY-USDT only.
    Numbered, WIB time, grouped by LONG/SHORT with custom export header.
    """
    print(f"[BACKTEST PEC] Running PEC simulation for last {PEC_WINDOW_MINUTES} minutes on {PEC_ONLY_TOKEN}...")
    pec_counter = 1
    long_results = []
    short_results = []

    for tf, tf_minutes in [("3min", 3), ("5min", 5)]:
        symbol = PEC_ONLY_TOKEN
        print(f"[BACKTEST PEC] {symbol} {tf} ...")
        df = get_ohlcv(symbol, interval=tf, limit=OHLCV_LIMIT)
        if df is None or df.empty or len(df) < PEC_BARS + 2:
            print(f"[BACKTEST PEC] No data for {symbol} {tf}. Skipping.")
            continue

        times = pd.to_datetime(df.index)
        window_start = times[-1] - pd.Timedelta(minutes=PEC_WINDOW_MINUTES)
        candidate_indices = [i for i in range(len(df) - PEC_BARS) if times[i] >= window_start]

        for i in candidate_indices:
            df_slice = df.iloc[:i+1]
            sf = SmartFilter(symbol, df_slice, df3m=df_slice, df5m=df_slice, tf=tf)
            res = sf.analyze()
            if isinstance(res, dict) and res.get("valid_signal") is True:
                if i + PEC_BARS >= len(df):
                    continue
                entry_idx = i
                entry_price = df["close"].iloc[i]
                signal_type = res.get("bias", "LONG")
                fired_dt = times[i]
                local_time_str = get_local_wib(fired_dt)
                pec_header = f"#{pec_counter} PEC Result Export [BTST:#{pec_counter} {symbol}-{tf}: {local_time_str}]"
                pec_result = run_pec_check(
                    symbol=symbol,
                    entry_idx=entry_idx,
                    tf=tf,
                    signal_type=signal_type,
                    entry_price=entry_price,
                    ohlcv_df=df,
                    pec_bars=PEC_BARS
                )
                # Compose one block: header + result as string
                from io import StringIO
                temp_io = StringIO()
                if pec_header:
                    temp_io.write("\n" + pec_header + "\n")
                summary = pec_result.get("summary", str(pec_result))
                temp_io.write(summary)
                temp_io.write("\n" + "="*32 + "\n")
                pec_block = temp_io.getvalue()

                # Group by LONG/SHORT
                if signal_type.upper() == "LONG":
                    long_results.append(pec_block)
                else:
                    short_results.append(pec_block)

                # --- INDIVIDUAL PEC DEBUG TXT DISABLED IN BACKTEST MODE ---
                # with open("pec_debug_temp.txt", "w") as f:
                #     f.write(pec_block)
                # send_telegram_file("pec_debug_temp.txt", caption=f"PEC result log for {symbol} {tf} [BACKTEST]")

                pec_counter += 1
        print(f"[BACKTEST PEC] Done for {symbol} {tf}.")

    # Dump grouped blocks for review
    with open("pec_long_results.txt", "w") as f:
        f.write("\n================================\n".join(long_results))
    with open("pec_short_results.txt", "w") as f:
        f.write("\n================================\n".join(short_results))
    send_telegram_file("pec_long_results.txt", caption=f"All PEC LONG results for {PEC_ONLY_TOKEN}")
    send_telegram_file("pec_short_results.txt", caption=f"All PEC SHORT results for {PEC_ONLY_TOKEN}")

    print("[BACKTEST PEC] All done. PEC logs grouped in pec_long_results.txt and pec_short_results.txt")

def run():
    # Forward mode: all tokens, all timeframes, all alerts/debugs
    if os.getenv("PEC_BACKTEST_ONLY", "false").lower() == "true":
        backtest_pec_simulation()
        return

    print("[INFO] Starting Smart Filter engine...\n")
    while True:
        now = time.time()
        valid_debugs = []
        pec_candidates = []

        for idx, symbol in enumerate(TOKENS, start=1):
            print(f"[INFO] Checking {symbol}...\n")
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
                        if not super_gk_aligned(bias, orderbook_result, density_result):
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT")
                            continue
                        print(f"[LOG] Sending 3min alert for {res3['symbol']}")
                        valid_debugs.append({
                            "symbol": res3["symbol"],
                            "tf": res3["tf"],
                            "bias": res3["bias"],
                            "filter_weights": sf3.filter_weights,
                            "gatekeepers": sf3.gatekeepers,
                            "results": res3["filter_results"],
                            "caption": f"Signal debug log for {res3.get('symbol')} {res3.get('tf')}",
                            "orderbook_result": orderbook_result,
                            "density_result": density_result,
                            "entry_price": res3.get("price")
                        })
                        entry_idx = df3.index.get_loc(df3.index[-1])
                        pec_candidates.append(
                            ("3min", symbol, res3.get("price"), bias, df3, entry_idx)
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
                    print(f"[INFO] No valid 3min signal for {symbol}.")
            except Exception as e:
                print(f"[ERROR] Exception in processing 3min for {symbol}: {e}")

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
                        if not super_gk_aligned(bias, orderbook_result, density_result):
                            print(f"[BLOCKED] SuperGK not aligned: Signal={bias}, OrderBook={orderbook_result}, Density={density_result} — NO SIGNAL SENT")
                            continue
                        print(f"[LOG] Sending 5min alert for {res5['symbol']}")
                        valid_debugs.append({
                            "symbol": res5["symbol"],
                            "tf": res5["tf"],
                            "bias": res5["bias"],
                            "filter_weights": sf5.filter_weights,
                            "gatekeepers": sf5.gatekeepers,
                            "results": res5["filter_results"],
                            "caption": f"Signal debug log for {res5.get('symbol')} {res5.get('tf')}",
                            "orderbook_result": orderbook_result,
                            "density_result": density_result,
                            "entry_price": res5.get("price")
                        })
                        entry_idx = df5.index.get_loc(df5.index[-1])
                        pec_candidates.append(
                            ("5min", symbol, res5.get("price"), bias, df5, entry_idx)
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
                    print(f"[INFO] No valid 5min signal for {symbol}.")
            except Exception as e:
                print(f"[ERROR] Exception in processing 5min for {symbol}: {e}")

        # --- Send up to 2 debug files to Telegram (Signal Debug txt sampling) ---
        if valid_debugs:
            num = min(len(valid_debugs), 2)
            for debug_info in random.sample(valid_debugs, num):
                dump_signal_debug_txt(
                    symbol=debug_info["symbol"],
                    tf=debug_info["tf"],
                    bias=debug_info["bias"],
                    filter_weights=debug_info["filter_weights"],
                    gatekeepers=debug_info["gatekeepers"],
                    results=debug_info["results"],
                    orderbook_result=debug_info.get("orderbook_result"),
                    density_result=debug_info.get("density_result")
                )
                send_telegram_file(
                    "signal_debug_temp.txt",
                    caption=debug_info["caption"]
                )

        # --- INDIVIDUAL PEC DEBUG TXT DISABLED IN LIVE MODE ---
        # if pec_candidates:
        #     for tf, symbol, entry_price, signal_type, ohlcv_df, entry_idx in pec_candidates[:2]:
        #         try:
        #             pec_result = run_pec_check(
        #                 symbol=symbol,
        #                 entry_idx=entry_idx,
        #                 tf=tf,
        #                 signal_type=signal_type,
        #                 entry_price=entry_price,
        #                 ohlcv_df=ohlcv_df,
        #                 pec_bars=PEC_BARS
        #             )
        #             export_pec_log(pec_result, filename="pec_debug_temp.txt")
        #             send_telegram_file("pec_debug_temp.txt", caption=f"PEC result log for {symbol} {tf}")
        #         except Exception as e:
        #             print(f"[PEC] Error running post-entry check for {symbol} {tf}: {e}")

        print("[INFO] ✅ Cycle complete. Sleeping 60 seconds...\n")
        time.sleep(60)

if __name__ == "__main__":
    # To run ONLY backtest, set env: PEC_BACKTEST_ONLY=true
    if os.getenv("PEC_BACKTEST_ONLY", "false").lower() == "true":
        backtest_pec_simulation()
    else:
        run()
