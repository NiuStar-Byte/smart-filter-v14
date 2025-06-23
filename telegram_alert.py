import os
import requests

# ——— CONFIG ——————————————————————————
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7100609549:AAHmeFe0RondzYyPKNuGTTp8HNAuT0PbNJs")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "-1002857433223")
SEND_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


def send_telegram_alert(
    numbered_signal: str,
    symbol: str,
    signal_type: str,
    price: float,
    tf: str,
    score: str,
    passed: str
) -> None:
    """
    Sends a formatted Telegram message to your channel/group.
    """
    # Build message with HTML formatting
    message = (
        f"{numbered_signal} 📊 <b>{symbol} ({tf})</b>\n"
        f"📈 <b>{signal_type} Signal</b>\n"
        f"💰 <code>{price}</code>\n"
        f"✅ <b>Score</b>: {score}\n"
        f"📌 <b>Passed</b>: {passed}"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        resp = requests.post(SEND_URL, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"📨 Telegram alert sent: {symbol} {signal_type} @ {price}")
    except requests.RequestException as e:
        print(f"❗ Telegram send error: {e} — response: {getattr(resp, 'text', '')}")
