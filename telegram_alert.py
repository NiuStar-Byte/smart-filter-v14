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
    passed: str,
    confidence: float = None,
    weighted: float = None,
) -> None:
    """
    Sends a formatted Telegram message to your channel/group.
    Format varies by timeframe (3min includes "[V19 Confirmed]")
    """
    confirmed_tag = " [V19 Confirmed]" if tf == "3min" else ""
    confidence_icon = (
        "🟢" if confidence >= 75 else
        "🟡" if confidence >= 60 else
        "🔴"
    )

    message = (
        f"{numbered_signal}. {symbol} ({tf}){confirmed_tag}\n"
        f"📈 {signal_type} Signal\n"
        f"💰 {price}\n"
        f"✅ Score: {score}\n"
        f"📌 Passed: {passed}\n"
        f"{confidence_icon} Confidence: {confidence}% (Weighted: {weighted})"
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
