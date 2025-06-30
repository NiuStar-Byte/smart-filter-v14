import pandas as pd
import csv
from smart_filter import SmartFilter
from pec_engine import run_pec_check
from telegram_alert import send_telegram_file
import os
from datetime import datetime
from pec_scheduler import TOKENS

def save_to_csv(results, filename="pec_results.csv"):
    """
    Save the backtest results to a CSV file, including Exit Bar # and Exit Price.
    """
    headers = ["Signal Type", "Symbol", "TF", "Entry Time", "Entry Price", "Exit Price", 
               "PnL ($)", "PnL (%)", "Score", "Max Score", "Confidence", "Weighted Confidence", 
               "Gatekeepers Passed", "Filter Results", "GK Flags", "Result", "Exit Bar #"]
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if file.tell() == 0:  # Write headers only if the file is empty
            writer.writerow(headers)

        for result in results:
            writer.writerow([
                result['signal_type'],
                result['symbol'],
                result['tf'],
                result['entry_time'],
                result['entry_price'],
                result['exit_price'],
                result['pnl_abs'],
                result['pnl_pct'],
                result['score'],
                result['score_max'],
                result['confidence'],
                result['weighted_confidence'],
                result['gatekeepers_passed'],
                result['filter_results'],
                result['gk_flags'],
                result['win_loss'],
                result['exit_bar']  # Exit Bar # included here
            ])
    print(f"[{datetime.datetime.now()}] [SCHEDULER] PEC results saved to {filename}")


def evaluate_signal(symbol, df, entry_idx, pec_bars):
    """
    Evaluates a single signal: calculates entry price, exit price, and PnL.
    Args:
        symbol (str): The symbol to backtest.
        df (DataFrame): The OHLCV data for the symbol.
        entry_idx (int): The index of the entry bar.
        pec_bars (int): The number of bars to evaluate after the entry.

    Returns:
        dict: A dictionary containing the signal data for export to CSV.
    """
    entry_price = df["close"].iloc[entry_idx]
    exit_idx = entry_idx + pec_bars
    exit_price = df["close"].iloc[exit_idx]

    # Calculate PnL for the signal (using $100 notional value)
    pnl_abs = 100 * (exit_price - entry_price) / entry_price
    pnl_pct = 100 * pnl_abs / 100

    signal_type = "LONG"  # Assuming the signal is LONG (can be adjusted for SHORT signals)
    
    # Collect results for CSV export
    result = {
        'signal_type': signal_type,
        'symbol': symbol,
        'tf': '5min',  # Assuming 5-minute timeframe, can adjust as needed
        'entry_time': str(df.index[entry_idx]),
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl_abs': pnl_abs,
        'pnl_pct': pnl_pct,
        'score': 'N/A',  # Placeholder for score
        'score_max': 'N/A',  # Placeholder for max score
        'confidence': 'N/A',  # Placeholder for confidence
        'weighted_confidence': 'N/A',  # Placeholder for weighted confidence
        'gatekeepers_passed': 'N/A',  # Placeholder for gatekeepers
        'filter_results': 'N/A',  # Placeholder for filter results
        'gk_flags': 'N/A',  # Placeholder for GK flags
        'win_loss': "WIN" if pnl_abs > 0 else "LOSS",  # Determine if the signal was a win or loss
        'exit_bar': exit_idx,  # Exit Bar # added here
    }
    return result


def run_backtest():
    """
    Run the backtest for all tokens and write the results to CSV.
    """
    print(f"[{datetime.now()}] [SCHEDULER] Running PEC backtest for all tokens...")

    results = []
    for symbol in TOKENS:
        # Fetch OHLCV data for each token
        ohlcv_data = fetch_ohlcv_data(symbol, "5min")  # Adjust timeframe as needed
        
        # Evaluate each signal for the token
        for entry_idx in range(len(ohlcv_data) - PEC_BARS):
            try:
                result = evaluate_signal(symbol, ohlcv_data, entry_idx, PEC_BARS)
                results.append(result)
                print(f"[{datetime.now()}] [SCHEDULER] PEC backtest for {symbol} completed successfully.")
            except Exception as e:
                print(f"[{datetime.now()}] [SCHEDULER] Error in backtest for {symbol}: {e}")
    
    # Save results to CSV
    save_to_csv(results, "pec_results.csv")

    # Send the CSV files to Telegram (kept intact as requested)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    long_file = f"pec_long_results_{timestamp}.csv"
    short_file = f"pec_short_results_{timestamp}.csv"
    send_telegram_file(long_file, caption=f"All PEC LONG results for ALL tokens [{timestamp}]")
    send_telegram_file(short_file, caption=f"All PEC SHORT results for ALL tokens [{timestamp}]")

# Example usage:
# Run the backtest for the configured tokens and timeframe
run_backtest()
