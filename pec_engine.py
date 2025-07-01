def run_pec_check(
    symbol,
    entry_idx,
    tf,
    signal_type,
    entry_price,
    ohlcv_df,
    pec_bars=5,  # Check the next 5 bars after entry
    filter_result=None,          # Expect a dict of filter pass/fail per signal
    score=None,                  # Total score (int or float)
    confidence=None,             # Confidence (float or pct)
    passed_gk_count=None,        # Count of passed GK filters
):
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
        filter_result: dict, key=filter name, value=True/False (pass/fail)
        score: int/float, total SmartFilter score
        confidence: float, SmartFilter confidence
        passed_gk_count: int, number of passed GK filters
    Returns:
        result: dict with key stats & verdicts, PLUS diagnostics
    """
    try:
        print(f"[PEC_ENGINE] Starting PEC check for {symbol} at index {entry_idx}.")
        
        # Get the relevant data for backtest (only the next 5 bars)
        pec_data = ohlcv_df.iloc[entry_idx: entry_idx + pec_bars + 1].copy()
        if pec_data.shape[0] < pec_bars + 1:
            print(f"[PEC_ENGINE] Not enough data for PEC: {symbol} at index {entry_idx}.")
            return {"status": "not enough data for PEC"}
        
        # Calculate MFE and MAE for the trade
        if signal_type == "LONG":
            max_up = (pec_data["high"].max() - entry_price) / entry_price * 100
            max_dn = (pec_data["low"].min() - entry_price) / entry_price * 100
            final_ret = (pec_data["close"].iloc[-1] - entry_price) / entry_price * 100
        else:
            max_up = (entry_price - pec_data["low"].min()) / entry_price * 100
            max_dn = (entry_price - pec_data["high"].max()) / entry_price * 100
            final_ret = (entry_price - pec_data["close"].iloc[-1]) / entry_price * 100

        print(f"[PEC_ENGINE] MFE: {max_up:.2f}%, MAE: {max_dn:.2f}%, Final Return: {final_ret:.2f}%")
    
        # The next 5 bars logic for Exit Time and Exit Price:
        exit_bar = entry_idx + pec_bars  # Exit after specified number of bars
        exit_price = pec_data["close"].iloc[-1]  # The exit price after 5 bars
        exit_time = ohlcv_df.index[exit_bar]  # The exit time based on the exit bar
        
        print(f"[PEC_ENGINE] Exit Time: {exit_time}, Exit Price: {exit_price}")

        # Log the exit conditions (for debug)
        logging.debug(f"Exit Time: {exit_time}, Exit Price: {exit_price}")
        
        # Simulate Win/Loss based on the exit price and entry price
        win_loss = "WIN" if final_ret > 0 else "LOSS"
        pnl = final_ret  # Percentage of Profit/Loss

        # Return results
        result = {
            "symbol": symbol,
            "tf": tf,
            "signal_type": signal_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_time": ohlcv_df.index[entry_idx],
            "exit_time": exit_time,
            "win_loss": win_loss,
            "pnl": pnl,
            "final_return": final_ret,
            "max_favorable": max_up,
            "max_adverse": max_dn
        }
        
        return result

    except Exception as e:
        print(f"[PEC_ENGINE] Error in run_pec_check: {e}")
        return {
            "status": "error",
            "error": str(e),
            "summary": f"# PEC ERROR: {symbol} {tf}: {str(e)}"
        }
