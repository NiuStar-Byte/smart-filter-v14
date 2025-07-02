# pecnew_backtest.py

from pecnew_logic import process_PECNew_backtest

def run_backtest(data):
    """
    Run backtest logic for PECNew system.
    Simulates the trades and tracks results based on historical data.
    """
    print("Starting backtest for PECNew...")
    result = process_PECNew_backtest(data)
    print(f"Backtest Result: {result}")
    return result
