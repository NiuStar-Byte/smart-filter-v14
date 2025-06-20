import requests
import os

BOT_TOKEN = os.getenv("BOT_TOKEN", "7100609549:AAHmeFe0RondzYyPKNuGTTp8HNAuT0PbNJs")
CHAT_ID = os.getenv("CHAT_ID", "-1002857433223")

def send_telegram_alert(symbol, signal_type, price, tf, score, passed):
    try:
        emoji = "📈" if signal_type == "LONG" else "📉"
        message = (
            f"📊 <b>{symbol} ({tf})</b>\n"
            f"{emoji} <b>{signal_type} Signal</b>\n"
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
