# pecnew_execution.py

from pecnew_config import use_PECNew, backtest_mode, historical_data
from pecnew_logic import process_PEC, process_PECNew, process_PECNew_backtest

def main_process(data):
    """
    Main function to switch between PEC and PECNew based on the flag.
    """
    if use_PECNew:
        if backtest_mode:  # In backtest mode, simulate PECNew trades
            result = process_PECNew_backtest(data)
        else:  # Real-time PECNew processing
            result = process_PECNew(data)
    else:  # Use current PEC logic
        result = process_PEC(data)

    print(f"Result: {result}")
    return result

# Example data structure (simulating a trade)
data = {
    'entry_price': 100,  # Example entry price
    'current_price': 105,  # Current market price
    'signal_type': 'LONG',  # Signal type: LONG or SHORT
    'time_elapsed': 17  # Time since entry in minutes (for 3mTF, should be between 0 and 16)
}

# Run the main process
if __name__ == "__main__":
    main_process(data)
