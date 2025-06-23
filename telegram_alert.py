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

    # Safely assign confidence icon
    if confidence is not None:
        if confidence >= 75:
            confidence_icon = "🟢"
        elif confidence >= 60:
            confidence_icon = "🟡"
        else:
            confidence_icon = "🔴"
        confidence_line = f"{confidence_icon} <b>Confidence</b>: {confidence:.1f}% (Weighted: {weighted:.1f})"
    else:
        confidence_line = ""

    # Format message
    message = (
        f"{numbered_signal}. <b>{symbol} ({tf}){confirmed_tag}</b>\n"
        f"📈 <b>{signal_type} Signal</b>\n"
        f"💰 <code>{price}</code>\n"
        f"✅ <b>Score</b>: {score}\n"
        f"📌 <b>Passed</b>: {passed}\n"
        f"{confidence_line}"
    ).strip()

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
