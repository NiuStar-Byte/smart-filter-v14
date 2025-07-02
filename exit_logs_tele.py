import logging
from telegram import Bot

# Set up logging to a text file
logging.basicConfig(
    filename="exit_logs_tele.txt",  # Change the filename to exit_logs_tele.txt
    level=logging.DEBUG, 
    format="%(asctime)s - %(message)s"
)

# Telegram Bot configuration (Replace with your actual token and chat_id)
BOT_TOKEN = "7100609549:AAHmeFe0RondzYyPKNuGTTp8HNAuT0PbNJs"  # Your bot token
CHAT_ID = "-1002857433223"  # Your chat ID

def send_logs_to_telegram(message):
    """
    Sends the message to a specified Telegram group using the bot.
    """
    try:
        bot = Bot(token=BOT_TOKEN)
        bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"Failed to send message to Telegram: {e}")

def log_exit_conditions(exit_time, exit_price, follow_through, stop_survival, volume_condition, condition_met):
    """
    Logs exit conditions both to a file and to Telegram.
    """
    # Create the log message
    message = f"Exit Time: {exit_time}, Exit Price: {exit_price}\n"
    message += f"Follow-Through: {follow_through}, Trailing Stop Survival: {stop_survival}, Volume Condition: {volume_condition}\n"
    message += f"Exit Condition met: {condition_met}"

    # Log to file
    logging.debug(message)

    # Send the log to Telegram
    send_logs_to_telegram(message)
