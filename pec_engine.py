# pec_engine.py

import pandas as pd
from datetime import datetime

def run_pec_check(symbol, entry_idx, tf, signal_type, entry_price, ohlcv_df, pec_bars=5):
    """
    Perform post-entry quality control (PEC) simulation for a fired signal.
    Args:
        symbol: str, e.g. "SPK-USDT"
        entry_idx: int, index of entry bar in ohlcv_df
        tf: str, e.g. "3min"
        signal_type: "LONG" or "SHORT"
        entry_price: float, actual entry price
        ohlcv_df: pd.DataFrame with columns: ["open", "high", "low", "close", ...]
        pec_bars: int, how many bars ahead to check (default 5)
    Returns:
        result: dict with key stats & verdicts
    """
    try:
        pec_data = ohlcv_df.iloc[entry_idx : entry_idx + pec_bars + 1].copy()
        if pec_data.shape[0] < pec_bars + 1:
            return {"status": "not enough data for PEC"}

        # Calculate Max Favorable/Adverse Excursion (MFE/MAE)
        if signal_type == "LONG":
            max_up = (pec_data["high"].max() - entry_price) / entry_price * 100
            max_dn = (pec_data["low"].min() - entry_price) / entry_price * 100
            final_ret = (pec_data["close"].iloc[-1] - entry_price) / entry_price * 100
        else:
            max_up = (entry_price - pec_data["low"].min()) / entry_price * 100
            max_dn = (entry_price - pec_data["high"].max()) / entry_price * 100
            final_ret = (entry_price - pec_data["close"].iloc[-1]) / entry_price * 100

        # Entry Follow-Through: Did it move at least +0.5% in your favor?
        follow_through = max_up >= 0.5

        # Trailing Stop Survival (0.5% from entry)
        stop_width = 0.5 / 100 * entry_price
        survived = True
        for bar in pec_data.itertuples():
            if signal_type == "LONG" and bar.low < entry_price - stop_width:
                survived = False
                break
            if signal_type == "SHORT" and bar.high > entry_price + stop_width:
                survived = False
                break

        # Signal Persistence (How many closes in same direction?)
        up_bars = ((pec_data["close"] > entry_price) if signal_type == "LONG" else (pec_data["close"] < entry_price)).sum()

        # Volume Confirmation (at least 3/5 bars have above-average volume)
        avg_vol = ohlcv_df["volume"].iloc[max(0, entry_idx-30):entry_idx].mean()
        vol_pass = (pec_data["volume"].iloc[1:] > avg_vol).sum() >= 3

        # Prepare summary for logging/telegram/txt
        summary = f"""# PEC for {symbol} {tf} {signal_type} @ {entry_price:.5f} (Fired: {ohlcv_df.index[entry_idx]})
- Follow-Through: {'✅' if follow_through else '❌'} (MaxFavorable={max_up:.2f}%)
- Max Adverse Excursion: {max_dn:.2f}%
- Final Return: {final_ret:.2f}%
- Trailing Stop (0.5%): {'Survived' if survived else 'Stopped'}
- Volume Confirmation: {'PASS' if vol_pass else 'FAIL'}
- Signal Persistence: {up_bars}/{pec_bars} bars favorable
"""

        result = {
            "symbol": symbol,
            "tf": tf,
            "signal_type": signal_type,
            "entry_price": entry_price,
            "entry_time": str(ohlcv_df.index[entry_idx]),
            "max_favorable": max_up,
            "max_adverse": max_dn,
            "final_return": final_ret,
            "follow_through": follow_through,
            "trailing_stop_survived": survived,
            "volume_confirmation": vol_pass,
            "signal_persistence": up_bars,
            "summary": summary
        }
        return result

    except Exception as e:
        # Golden Rule: Always return a dict, even on error
        return {
            "status": "error",
            "error": str(e),
            "summary": f"# PEC ERROR: {symbol} {tf}: {str(e)}"
        }

def export_pec_log(result, filename="pec_debug_temp.txt", custom_header=None):
    """
    Export a single PEC result (dict) to the given txt file.
    If custom_header is supplied, prepend it as the first line (for numbered, apple-to-apple log).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = result.get("summary", str(result))
    with open(filename, "a") as f:
        if custom_header:
            f.write(f"\n{custom_header}\n")
        else:
            f.write(f"\n# PEC Result Export (created: {now})\n")
        f.write(summary)
        f.write("\n" + "="*32 + "\n")
