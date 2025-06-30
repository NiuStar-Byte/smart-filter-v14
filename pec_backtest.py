# pec_backtest.py

import os
import csv
import pandas as pd
from telegram_alert import send_telegram_file
from smart_filter import SmartFilter
from datetime import datetime
from pec_scheduler import TOKENS  # Ensure the tokens are imported properly

def save_to_csv(results, filename="pec_results.csv"):
    """
    Save the backtest results to a CSV file.
    """
    if not results:
        print("[ERROR] No data to save to CSV.")
        return  # Exit early if no data

    headers = ["Signal Type", "Symbol", "TF", "Entry Time", "Entry Price", 
               "Exit Price", "PnL ($)", "PnL (%)", "Score", "Max Score", 
               "Confidence", "Weighted Confidence", "Gatekeepers Passed", 
               "Filter Results", "GK Flags", "Result", "Exit Bar #"]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers

        for result in results:
            writer.writerow([
                result.get('signal_type', ''),
                result.get('symbol', ''),
                result.get('tf', ''),
                result.get('entry_time', ''),
                result.get('entry_price', ''),
                result.get('exit_price', ''),
                result.get('pnl_abs', ''),
                result.get('pnl_pct', ''),
                result.get('score', ''),
                result.get('score_max', ''),
                result.get('confidence', ''),
                result.get('weighted_confidence', ''),
                result.get('gatekeepers_passed', ''),
                result.get('filter_results', ''),
                result.get('gk_flags', ''),
                result.get('win_loss', ''),
                result.get('exit_bar', '')
            ])
    print(f"[{datetime.now()}] [SCHEDULER] PEC results saved to {filename}")


def run_pec_backtest(symbols, timeframe, pec_window_minutes, pec_bars, ohlcv_data):
    """
    Run the backtest for the symbols, and save to a CSV file.
    """
    results = []
    for symbol in symbols:
        print(f"[{datetime.now()}] [SCHEDULER] Running PEC for {symbol}...")

        # Evaluate signal for each symbol
        for entry_idx in range(len(ohlcv_data) - pec_bars):
            try:
                result = evaluate_signal(symbol, ohlcv_data, entry_idx, pec_bars)
                results.append(result)
                print(f"[{datetime.now()}] [SCHEDULER] PEC for {symbol} completed successfully.")
            except Exception as e:
                print(f"[{datetime.now()}] [SCHEDULER] Error in backtest for {symbol}: {e}")
        
        # Save results to CSV
        save_to_csv(results, "pec_results.csv")

        # Send the PEC files to Telegram
        long_file = "pec_long_results.csv"
        short_file = "pec_short_results.csv"
        send_telegram_file(long_file, caption=f"Long Signal Results for {symbol}")
        send_telegram_file(short_file, caption=f"Short Signal Results for {symbol}")


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

    pnl_abs = 100 * (exit_price - entry_price) / entry_price
    pnl_pct = 100 * pnl_abs / 100

    signal_type = "LONG"  # Assuming the signal is LONG (can be adjusted for SHORT signals)

    result = {
        'signal_type': signal_type,
        'symbol': symbol,
        'tf': '5min',  # Assuming 5-minute timeframe
        'entry_time': str(df.index[entry_idx]),
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl_abs': pnl_abs,
        'pnl_pct': pnl_pct,
        'score': 'N/A',
        'score_max': 'N/A',
        'confidence': 'N/A',
        'weighted_confidence': 'N/A',
        'gatekeepers_passed': 'N/A',
        'filter_results': 'N/A',
        'gk_flags': 'N/A',
        'win_loss': "WIN" if pnl_abs > 0 else "LOSS",
        'exit_bar': exit_idx
    }
    return result

