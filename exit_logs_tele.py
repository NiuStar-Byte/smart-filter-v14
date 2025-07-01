import logging
import requests
from telegram import Bot  # Ensure Bot class is imported

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
        bot.send_message(chat_id=CHAT_ID, text=message)  # Sending text message
        logging.debug(f"Log message sent to Telegram: {message}")
    except Exception as e:
        logging.error(f"Failed to send message to Telegram: {e}")

# Function to create the blank log file and send it
def send_blank_log_to_telegram():
    # Create the file, it's empty for now
    with open("exit_logs_tele.txt", "w") as f:
        f.write("")  # Blank file

    # Send the blank file to Telegram
    send_logs_to_telegram("Sending blank exit_logs_tele.txt file to Telegram.")
    logging.debug("Sent blank exit_logs_tele.txt to Telegram.")

# Call the function to send the blank file
send_blank_log_to_telegram()

