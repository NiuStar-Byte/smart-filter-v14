# --- Function_ID_00_v1: Import Necessary Libraries ---
import os
import time
import pandas as pd
import random
import pytz
from datetime import datetime

# Importing required functions for kucoin data and orderbook
from kucoin_data import get_ohlcv
from kucoin_orderbook import get_order_wall_delta, get_resting_order_density

# Importing SmartFilter and alert functions
from smart_filter import SmartFilter
from telegram_alert import send_telegram_alert, send_telegram_file

# Importing debug functions
from signal_debug_log import dump_signal_debug_txt

# Importing functions for PEC engine
from pec_engine import run_pec_check, export_pec_log

# Importing indicators from the 'indicators.py' module
from indicators import (
    calculate_rsi_04,
    calculate_bollinger_bands_05,
    calculate_stochastic_oscillator_06,
    calculate_supertrend_07,
    calculate_atr_08,
    calculate_parabolic_sar_09,
    calculate_adx_10,
    calculate_market_structure_11,
    calculate_support_resistance_12,
    calculate_pivot_points_13,
    calculate_composite_trend_indicator_14
)

# PEC backtest fires ONLY when backtest mode is enabled.
TOKENS = [
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT"
]
COOLDOWN = {"3min": 720, "5min": 900}
last_sent = {}

PEC_BARS = 5
PEC_WINDOW_MINUTES = 500
OHLCV_LIMIT = 1000

# --- Function_ID_01_v1: Get Local Time in WIB ---
def get_local_wib_01(dt):  # Function_ID_01_v1
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)
    return dt.tz_localize('UTC').tz_convert('Asia/Jakarta').strftime('%H:%M WIB')


# --- Function_ID_02_v1: Get Resting Order Density ---
def get_resting_order_density_02(symbol, depth=100, band_pct=0.005):  # Function_ID_02_v1
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

# --- Function_ID_03_v1: Log Orderbook and Density ---
def log_orderbook_and_density(symbol):  # Function_ID_03_v1
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

    # Log the indicators for the last row (latest data point)
    try:
        df = get_ohlcv(symbol, interval="3min", limit=1)  # Retrieve the latest 3min data point
        if df is not None and not df.empty:
            print(f"[IndicatorsLog] {symbol} | Latest Indicators:")
            print(f"  RSI: {calculate_rsi_04(df)['RSI'].iloc[-1]:.2f}")
            print(f"  Bollinger Bands: Upper: {df['upper_band'].iloc[-1]:.2f}, Lower: {df['lower_band'].iloc[-1]:.2f}")
            print(f"  Stochastic Oscillator: {df['stochastic'].iloc[-1]:.2f}")
            print(f"  Supertrend: Upper: {df['upper_band'].iloc[-1]:.2f}, Lower: {df['lower_band'].iloc[-1]:.2f}")
            print(f"  ATR: {df['ATR'].iloc[-1]:.2f}")
            print(f"  Parabolic SAR: {df['sar'].iloc[-1]:.2f}")
            print(f"  ADX: {df['ADX'].iloc[-1]:.2f}")
            print(f"  Market Structure: {df['market_structure'].iloc[-1]}")
            print(f"  Support: {df['support'].iloc[-1]:.2f}")
            print(f"  Resistance: {df['resistance'].iloc[-1]:.2f}")
            print(f"  Pivot Points: Pivot: {df['pivot'].iloc[-1]:.2f}, Support_1: {df['support_1'].iloc[-1]:.2f}, Resistance_1: {df['resistance_1'].iloc[-1]:.2f}")
            print(f"  Composite Trend Indicator: {df['CTI'].iloc[-1]:.2f}")
        else:
            print(f"[ERROR] No data available for {symbol} to log indicators.")
    except Exception as e:
        print(f"[IndicatorsLog] {symbol} ERROR: {e}")

# --- Function_ID_04_v1: Calculate RSI ---
def calculate_rsi_04(df, period=14):  # Function_ID_04_v1
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Function_ID_05_v1: Calculate Bollinger Bands ---
def calculate_bollinger_bands_05(df, window=20):  # Function_ID_05_v1
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * 2)
    df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * 2)
    return df

# --- Function_ID_06_v1: Calculate Stochastic Oscillator ---
def calculate_stochastic_oscillator_06(df, window=14):  # Function_ID_06_v1
    df['stochastic'] = ((df['close'] - df['low'].rolling(window=window).min()) /
                        (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min())) * 100
    return df

# --- Function_ID_07_v1: Calculate Supertrend ---
def calculate_supertrend_07(df, period=7, multiplier=3):  # Function_ID_07_v1
    df['ATR'] = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
    df['upper_band'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['lower_band'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']
    return df

# --- Function_ID_08_v1: Calculate ATR ---
def calculate_atr_08(df, period=14):  # Function_ID_08_v1
    df['ATR'] = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()
    return df

# --- Function_ID_09_v1: Calculate Parabolic SAR ---
def calculate_parabolic_sar_09(df, acceleration=0.02, maximum=0.2):  # Function_ID_09_v1
    df['sar'] = df['close'].copy()
    up_trend = True
    ep = df['high'][0]
    af = acceleration
    sar = df['sar'][0]
    
    for i in range(1, len(df)):
        if up_trend:
            sar = sar + af * (ep - sar)
            if df['low'][i] < sar:
                up_trend = False
                sar = ep
                ep = df['low'][i]
                af = acceleration
        else:
            sar = sar + af * (ep - sar)
            if df['high'][i] > sar:
                up_trend = True
                sar = ep
                ep = df['high'][i]
                af = acceleration
        
        df['sar'][i] = sar
    return df

# --- Function_ID_10_v1: Calculate ADX ---
def calculate_adx_10(df, period=14):  # Function_ID_10_v1
    df['+DI'] = df['high'].diff()
    df['-DI'] = df['low'].diff()
    df['ADX'] = abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    return df

# --- Function_ID_11_v1: Calculate Market Structure ---
def calculate_market_structure_11(df):  # Function_ID_11_v1
    df['market_structure'] = 'None'
    for i in range(2, len(df)):
        if df['high'][i] > df['high'][i-1] and df['low'][i] > df['low'][i-1]:
            df['market_structure'][i] = 'Uptrend'
        elif df['high'][i] < df['high'][i-1] and df['low'][i] < df['low'][i-1]:
            df['market_structure'][i] = 'Downtrend'
        else:
            df['market_structure'][i] = 'Sideways'
    return df

# --- Function_ID_12_v1: Calculate Support and Resistance ---
def calculate_support_resistance_12(df, period=20):  # Function_ID_12_v1
    df['support'] = df['low'].rolling(window=period).min()
    df['resistance'] = df['high'].rolling(window=period).max()
    return df

# --- Function_ID_13_v1: Calculate Pivot Points ---
def calculate_pivot_points_13(df):  # Function_ID_13_v1
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['support_1'] = (2 * df['pivot']) - df['high']
    df['resistance_1'] = (2 * df['pivot']) - df['low']
    return df

# --- Function_ID_14_v1: Calculate Composite Trend Indicator ---
def calculate_composite_trend_indicator_14(df):  # Function_ID_14_v1
    df['CTI'] = (df['close'] - df['open']) / (df['high'] - df['low']) * 100
    return df

# --- Function_ID_15_v1: SuperGK Alignment Logic ---
def super_gk_aligned_15(bias, orderbook_result, density_result):  # Function_ID_15_v1
    wall_delta = orderbook_result.get('wall_delta', 0) if orderbook_result else 0
    orderbook_bias = "LONG" if wall_delta > 0 else "SHORT" if wall_delta < 0 else "NEUTRAL"
    
    # Extract density values
    bid_density = density_result.get('bid_density', 0) if density_result else 0
    ask_density = density_result.get('ask_density', 0) if density_result else 0
    density_bias = "LONG" if bid_density > ask_density else "SHORT" if ask_density > bid_density else "NEUTRAL"
    
    # Check if the biases align
    if (orderbook_bias != "NEUTRAL" and bias != orderbook_bias):
        return False
    
    if (density_bias != "NEUTRAL" and bias != density_bias):
        return False
    
    # If either orderbook or density is neutral, we don't align
    if orderbook_bias == "NEUTRAL" or density_bias == "NEUTRAL":
        return False
    
    # If all checks pass, biases align
    return True

# --- Function_ID_16_v1: Main Run Logic ---
def run_16():  # Function_ID_16_v1
    print("[INFO] Starting Smart Filter engine (LIVE MODE)...\n")
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

                        # Log the indicators for the last row (latest data point)
                        print(f"[INFO] {symbol} 3min Indicator Values:")
                        print(f"RSI: {df3['RSI'].iloc[-1]}")
                        print(f"Bollinger Bands: Upper: {df3['upper_band'].iloc[-1]}, Lower: {df3['lower_band'].iloc[-1]}")
                        print(f"Stochastic Oscillator: {df3['stochastic'].iloc[-1]}")
                        print(f"SuperTrend: {df3['upper_band'].iloc[-1]} / {df3['lower_band'].iloc[-1]}")
                        print(f"ATR: {df3['ATR'].iloc[-1]}")
                        print(f"Parabolic SAR: {df3['sar'].iloc[-1]}")
                        print(f"ADX: {df3['ADX'].iloc[-1]}")
                        print(f"Market Structure: {df3['market_structure'].iloc[-1]}")
                        print(f"Support: {df3['support'].iloc[-1]} / Resistance: {df3['resistance'].iloc[-1]}")
                        print(f"Pivot: {df3['pivot'].iloc[-1]}")
                        print(f"CTI: {df3['CTI'].iloc[-1]}")

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

                        # Log the indicators for the last row (latest data point)
                        print(f"[INFO] {symbol} 5min Indicator Values:")
                        print(f"RSI: {df5['RSI'].iloc[-1]}")
                        print(f"Bollinger Bands: Upper: {df5['upper_band'].iloc[-1]}, Lower: {df5['lower_band'].iloc[-1]}")
                        print(f"Stochastic Oscillator: {df5['stochastic'].iloc[-1]}")
                        print(f"SuperTrend: {df5['upper_band'].iloc[-1]} / {df5['lower_band'].iloc[-1]}")
                        print(f"ATR: {df5['ATR'].iloc[-1]}")
                        print(f"Parabolic SAR: {df5['sar'].iloc[-1]}")
                        print(f"ADX: {df5['ADX'].iloc[-1]}")
                        print(f"Market Structure: {df5['market_structure'].iloc[-1]}")
                        print(f"Support: {df5['support'].iloc[-1]} / Resistance: {df5['resistance'].iloc[-1]}")
                        print(f"Pivot: {df5['pivot'].iloc[-1]}")
                        print(f"CTI: {df5['CTI'].iloc[-1]}")

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

        print("[INFO] ✅ Cycle complete. Sleeping 60 seconds...\n")
        time.sleep(60)

if __name__ == "__main__":
    # Mode switch based on Railway variable
    if os.getenv("PEC_BACKTEST_ONLY", "false").lower() == "true":
        from pec_backtest import run_pec_backtest
        run_pec_backtest(
            TOKENS, get_ohlcv, get_local_wib,
            PEC_WINDOW_MINUTES, PEC_BARS, OHLCV_LIMIT
        )
    else:
        run_16()  # Call the revised run function
