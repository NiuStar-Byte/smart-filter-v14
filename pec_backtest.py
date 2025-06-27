import pandas as pd
from smart_filter import SmartFilter
from pec_engine import run_pec_check
from telegram_alert import send_telegram_file
import os
import datetime

def run_pec_backtest(
    TOKENS,
    get_ohlcv,
    get_local_wib,
    PEC_WINDOW_MINUTES,
    PEC_BARS,
    OHLCV_LIMIT,
):
    """
    Runs PEC backtest on ALL tokens from TOKENS.
    Uses $100 per trade, 5-bar fixed exit. Exports clean block with all detail.
    Output filenames are timestamped for version tracking.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    long_file = f"pec_long_results_{timestamp}.txt"
    short_file = f"pec_short_results_{timestamp}.txt"

    print(f"[BACKTEST PEC] Running PEC simulation for last {PEC_WINDOW_MINUTES} minutes on ALL tokens: {', '.join(TOKENS)}...")

    pec_counter = 1
    long_blocks = []
    short_blocks = []

    for symbol in TOKENS:
        for tf, tf_minutes in [("3min", 3), ("5min", 5)]:
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
                    entry_idx = i
                    if entry_idx + PEC_BARS >= len(df):
                        continue
                    entry_price = float(df["close"].iloc[entry_idx])
                    exit_idx = entry_idx + PEC_BARS
                    exit_price = float(df["close"].iloc[exit_idx])
                    fired_dt = times[entry_idx]
                    local_time_str = get_local_wib(fired_dt)
                    signal_type = res.get("bias", "LONG").upper()

                    # $100 notional logic
                    if signal_type == "LONG":
                        pnl_abs = 100 * (exit_price - entry_price) / entry_price
                    else:  # SHORT
                        pnl_abs = 100 * (entry_price - exit_price) / entry_price
                    pnl_pct = 100 * pnl_abs / 100
                    win_loss = "WIN" if pnl_abs > 0 else "LOSS"

                    # Compose full output block
                    pec_block = (
                        f"#{pec_counter} PEC Result Export [BTST:#{pec_counter} {symbol}-{tf}: {local_time_str}]\n"
                        f"Symbol: {symbol}\n"
                        f"TF: {tf}\n"
                        f"Entry Time: {local_time_str}\n"
                        f"Signal: {signal_type}\n"
                        f"Entry Price: {entry_price:.6f}\n"
                        f"Exit Price: {exit_price:.6f}\n"
                        f"PnL ($): {pnl_abs:.2f}\n"
                        f"PnL (%): {pnl_pct:.2f}\n"
                        f"Drawdown (%): -\n"
                        f"Volume Pass (%): -\n"
                        f"Result: {win_loss}\n"
                        + "="*40 + "\n"
                    )

                    if signal_type == "LONG":
                        long_blocks.append(pec_block)
                    else:
                        short_blocks.append(pec_block)
                    pec_counter += 1
            print(f"[BACKTEST PEC] Done for {symbol} {tf}.")

    # Write to files
    with open(long_file, "w") as f:
        f.writelines(long_blocks)
    with open(short_file, "w") as f:
        f.writelines(short_blocks)

    print(f"[DEBUG] {long_file} written, {len(long_blocks)} signals, size={os.path.getsize(long_file)} bytes.")
    print(f"[DEBUG] {short_file} written, {len(short_blocks)} signals, size={os.path.getsize(short_file)} bytes.")

    # Send to Telegram
    print("[DEBUG] Sending PEC long file to Telegram...")
    send_telegram_file(long_file, caption=f"All PEC LONG results for ALL tokens [{timestamp}]")
    print("[DEBUG] Sending PEC short file to Telegram...")
    send_telegram_file(short_file, caption=f"All PEC SHORT results for ALL tokens [{timestamp}]")

    print("[BACKTEST PEC] All done. PEC logs grouped in", long_file, "and", short_file)

