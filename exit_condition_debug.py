
import logging
from exit_logs_tele import send_logs_to_telegram  # Import the function from exit_logs_tele.py

# Set up logging configuration
logging.basicConfig(filename="exit_condition_debug.log", level=logging.DEBUG, format="%(asctime)s - %(message)s")

def log_exit_conditions(exit_time, exit_price, follow_through, stop_survival, volume_condition, condition_met):
    # Log the exit condition status and track when EXIT TIME and EXIT PRICE are populated.
    logging.debug("Evaluating Exit Conditions:")
    logging.debug(f"Exit Time: {exit_time}, Exit Price: {exit_price}")
    log_message = f"Exit Time: {exit_time}, Exit Price: {exit_price}, Condition Met: {condition_met}"
    # Log the message to a file (if necessary)
    logging.info(log_message)
    # Send the log message to Telegram
    send_logs_to_telegram(log_message)  # Send the log to Telegram
    log_exit_conditions("2025-07-01 12:00:00", "123.45", True, True, True, True)
    
    if exit_time == 'N/A' and exit_price == 'N/A':
        logging.debug("Exit Time and Exit Price: Both are N/A (No exit condition met).")
    else:
        logging.debug("Exit Condition Met. Populating Exit Time and Exit Price.")
        logging.debug(f"Follow-Through: {follow_through}, Trailing Stop Survival: {stop_survival}, Volume Condition: {volume_condition}")
        logging.debug(f"Exit Price populated as {exit_price} and Exit Time populated as {exit_time}.")

    logging.debug(f"Exit condition met: {condition_met}")

# Example of tracking when exit conditions are met and how the values are populated
# Use this function inside your existing `pec_engine.py` script where you calculate exit conditions

def track_exit_conditions_example(pec_data, follow_through, stop_survival, volume_condition):
    # Track when the exit price and time are populated based on conditions in pec_data.
    exit_time = None
    exit_price = None
    condition_met = False

    # If follow-through condition is met
    if follow_through:
        exit_price = pec_data["close"].max()
        exit_time = pec_data.index[-1]  # Assuming pec_data has datetime index

    # If trailing stop condition is met
    if stop_survival:
        # Example of stop survival condition
        if pec_data["close"].iloc[-1] < pec_data["close"].iloc[-2] * 0.995:  # Stop condition at 0.5% decrease
            exit_price = pec_data["close"].iloc[-1]
            exit_time = pec_data.index[-1] 

    # If volume condition is met
    if volume_condition:
        # Example of volume condition logic
        if pec_data["volume"].iloc[-1] > pec_data["volume"].mean():
            exit_price = pec_data["close"].iloc[-1]
            exit_time = pec_data.index[-1]

    # Call the logging function to record the exit conditions
    log_exit_conditions(exit_time, exit_price, follow_through, stop_survival, volume_condition, condition_met)

# Example use of the debug function
# Assuming you have pec_data available with follow-through and stop-survival conditions
# track_exit_conditions_example(pec_data, follow_through=True, stop_survival=False, volume_condition=True)

