import logging
from telegram import Bot

# Import Telegram config
from tg_config import BOT_TOKEN, CHAT_ID

# Check if Telegram is properly configured (not placeholder values)
TELEGRAM_ENABLED = (
    BOT_TOKEN and 
    CHAT_ID and 
    "YOUR_BOT_TOKEN" not in BOT_TOKEN and 
    "YOUR_CHAT_ID" not in CHAT_ID
)

# Set up logging to a text file
logging.basicConfig(
    filename="exit_logs_tele.txt",  # Change the filename to exit_logs_tele.txt
    level=logging.DEBUG, 
    format="%(asctime)s - %(message)s"
)

def send_logs_to_telegram(message):
    """
    Sends the message to a specified Telegram group using the bot.
    Gracefully skips if Telegram not configured.
    """
    # Skip if Telegram not enabled or not configured
    if not TELEGRAM_ENABLED:
        return  # Silently skip if not configured
    
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
