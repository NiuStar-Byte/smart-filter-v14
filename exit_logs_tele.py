# exit_logs_tele.py

import logging
from telegram import Bot  # Ensure Bot class is imported
import asyncio

# Set up logging to a text file
logging.basicConfig(
    filename="exit_logs_tele.txt",  # Change the filename to exit_logs_tele.txt
    level=logging.DEBUG, 
    format="%(asctime)s - %(message)s"
)

# Telegram Bot configuration (Replace with your actual token and chat_id)
BOT_TOKEN = "7100609549:AAHmeFe0RondzYyPKNuGTTp8HNAuT0PbNJs"  # Your bot token
CHAT_ID = "-1002857433223"  # Your chat ID

# Function to send logs to Telegram (async function)
async def send_logs_to_telegram(message):
    """
    Sends the message to a specified Telegram group using the bot.
    """
    try:
        bot = Bot(token=BOT_TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=message)
        
        # Log the success of sending the message to Telegram
        logging.debug(f"Log message sent to Telegram: {message}")  # Log successful send
    except Exception as e:
        logging.error(f"Failed to send message to Telegram: {e}")  # Log error in case of failure

# Function to create the blank log file and send it
def send_blank_log_to_telegram():
    # Create the file, it's empty for now
    with open("exit_logs_tele.txt", "w") as f:
        f.write("")  # Blank file

    # Send the blank file to Telegram
    send_logs_to_telegram("Sending blank exit_logs_tele.txt file to Telegram.")
    logging.debug("Sent blank exit_logs_tele.txt to Telegram.")

# The function to log exit conditions and send to Telegram
def log_exit_conditions(exit_time, exit_price, follow_through, stop_survival, volume_condition, condition_met):
    """
    Logs exit conditions both to a file and to Telegram.
    """
    # Create the log message
    message = f"Exit Time: {exit_time}, Exit Price: {exit_price}\n"
    message += f"Follow-Through: {follow_through}, Trailing Stop Survival: {stop_survival}, Volume Condition: {volume_condition}\n"
    message += f"Exit Condition met: {condition_met}"
    
    # Log to file
    logging.info(message)

    # Send to Telegram (ensure this is awaited properly within the event loop)
    loop = asyncio.get_event_loop()
    loop.create_task(send_logs_to_telegram(message))  # Schedule coroutine for execution
    
    # Send the log message to Telegram
    send_logs_to_telegram(message)
