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
    score_str: str,
    passed_str: str
) -> None:
    """
    Sends a formatted Telegram message to your channel/group and logs full request/response.
    """
    message = (
        f"{numbered_signal} 📊 <b>{symbol} ({tf})</b>\n"
        f"📈 <b>{signal_type} Signal</b>\n"
        f"💰 <code>{price}</code>\n"
        f"✅ <b>Score</b>: {score_str}\n"
        f"📌 <b>Passed</b>: {passed_str}"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    # Debug: log HTTP request
    print(f"[TELE-DEBUG] POST {SEND_URL} payload={payload}")
    try:
        resp = requests.post(SEND_URL, json=payload, timeout=10)
        # Debug: log HTTP response
        print(f"[TELE-DEBUG] RESPONSE status={resp.status_code}, body={resp.text}")
        resp.raise_for_status()
        print(f"📨 Telegram alert sent: {symbol} {signal_type} @ {price}")
    except requests.RequestException as e:
        print(f"❗ Telegram send error: {e}")
        if 'resp' in locals():
            print(f"❗ Response body: {resp.text}")
