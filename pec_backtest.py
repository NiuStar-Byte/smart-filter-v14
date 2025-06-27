import pandas as pd
from smart_filter import SmartFilter
from pec_engine import run_pec_check
from telegram_alert import send_telegram_file

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
    Numbered, WIB time, grouped by LONG/SHORT with custom export header.
    """
    print(f"[BACKTEST PEC] Running PEC simulation for last {PEC_WINDOW_MINUTES} minutes on ALL tokens: {', '.join(TOKENS)}...")

    pec_counter = 1
    long_results = []
    short_results = []

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
                    from io import StringIO
                    temp_io = StringIO()
                    if pec_header:
                        temp_io.write("\n" + pec_header + "\n")
                    summary = pec_result.get("summary", str(pec_result))
                    temp_io.write(summary)
                    temp_io.write("\n" + "="*32 + "\n")
                    pec_block = temp_io.getvalue()
                    if signal_type.upper() == "LONG":
                        long_results.append(pec_block)
                    else:
                        short_results.append(pec_block)
                    pec_counter += 1
            print(f"[BACKTEST PEC] Done for {symbol} {tf}.")

    # Dump grouped blocks for review
    with open("pec_long_results.txt", "w") as f:
        f.write("\n================================\n".join(long_results))
        print(f"[DEBUG] pec_long_results.txt written, {len(long_results)} signals.")
    with open("pec_short_results.txt", "w") as f:
        f.write("\n================================\n".join(short_results))
        print(f"[DEBUG] pec_short_results.txt written, {len(short_results)} signals.")

    print("[DEBUG] Sending PEC long file to Telegram...")
    send_telegram_file("pec_long_results.txt", caption=f"All PEC LONG results for ALL tokens")
    print("[DEBUG] Sending PEC short file to Telegram...")
    send_telegram_file("pec_short_results.txt", caption=f"All PEC SHORT results for ALL tokens")

    print("[BACKTEST PEC] All done. PEC logs grouped in pec_long_results.txt and pec_short_results.txt")
