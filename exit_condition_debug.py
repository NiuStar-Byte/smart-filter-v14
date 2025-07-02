# exit_condition_debug.py

import logging
from exit_logs_tele import send_logs_to_telegram  # Import the function from exit_logs_tele.py
import asyncio  # Ensure asyncio is imported

# Set up logging configuration
logging.basicConfig(filename="exit_condition_debug.log", level=logging.DEBUG, format="%(asctime)s - %(message)s")

# Make the log_exit_conditions function asynchronous
async def log_exit_conditions(exit_time, exit_price, follow_through, stop_survival, volume_condition, condition_met):
    """
    Logs exit conditions both to a file and sends them to Telegram.
    """
    logging.debug(f"Preparing to log exit conditions for Exit Time: {exit_time} and Exit Price: {exit_price}")
    
    # Log the exit condition status and track when EXIT TIME and EXIT PRICE are populated
    logging.debug("Evaluating Exit Conditions:")
    logging.debug(f"Exit Time: {exit_time}, Exit Price: {exit_price}")
    
    # Create the log message
    log_message = f"Exit Time: {exit_time}, Exit Price: {exit_price}, Condition Met: {condition_met}"
    
    # Log the message to a file
    logging.info(log_message)

    # Await the log message to be sent to Telegram
    await send_logs_to_telegram(log_message)  # Await the message being sent to Telegram

    # Debug log example after sending the message
    logging.debug(f"Message sent to Telegram successfully for Exit Time: {exit_time} and Exit Price: {exit_price}")

    # Handling case where exit time and price are N/A (No exit condition met)
    if exit_time == 'N/A' and exit_price == 'N/A':
        logging.debug("Exit Time and Exit Price: Both are N/A (No exit condition met).")
    
    # Optional: You could add more checks for different conditions based on your strategy here

# Example of the function that will call log_exit_conditions (ensure this is async)
async def process_exit_conditions():
    """
    Example function to simulate exit conditions and log them.
    """
    # Example values
    exit_time = "2025-07-02 12:30:00"
    exit_price = 3500
    follow_through = True
    stop_survival = False
    volume_condition = "High"
    condition_met = "True"

    # Call log_exit_conditions and await it
    await log_exit_conditions(exit_time, exit_price, follow_through, stop_survival, volume_condition, condition_met)

# Run the process_exit_conditions function as part of an async event loop
async def main():
    """
    Main function that runs the exit condition logging process.
    """
    await process_exit_conditions()

# Start the event loop
if __name__ == "__main__":
    asyncio.run(main())  # Run the main function that starts the process
