import requests
import os

BOT_TOKEN = "7100609549:AAHmeFe0RondzYyPKNuGTTp8HNAuT0PbNJs"  # Use your actual BOT token
CHAT_ID = "-1002857433223"  # Use your actual chat ID

def send_telegram_alert(numbered_signal, symbol, signal_type, price, tf, score, passed):
    try:
        message = (
            f"{numbered_signal} 📊 <b>{symbol} ({tf})</b>\n"
            f"📈 <b>{signal_type} Signal</b>\n"
            f"💰 <code>{price}</code>\n"
            f"✅ <b>Score</b>: {score}\n"
            f"📌 <b>Passed</b>: {passed}"
        )

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }

        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Telegram Error: {response.text}")

    except Exception as e:
        print(f"❌ Telegram alert error: {e}")
