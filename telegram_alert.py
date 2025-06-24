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
    score: int,
    passed: int,
    confidence: float = None,
    weighted: float = None,
    total_score: int = 21,
    total_passed: int = 13,
    max_weight: float = 48.8
) -> None:
    """
    Sends a formatted Telegram message to your channel/group.
    """
    confirmed_tag = " [V19 Confirmed]" if tf == "3min" else ""

    # Auto compute confidence
    if confidence is None and weighted is not None:
        confidence = (weighted / max_weight) * 100
    confidence = round(confidence, 1) if confidence is not None else 0.0

    # Format strings
    score_str = f"{score}/{total_score}"
    passed_str = f"{passed}/{total_passed}"
    weighted_str = f"{weighted:.1f}/{max_weight:.1f}" if weighted is not None else f"0.0/{max_weight:.1f}"

    # Confidence icon
    confidence_icon = (
        "🟢" if confidence >= 75 else
        "🟡" if confidence >= 60 else
        "🔴"
    )

    # Final message
    message = (
        f"{numbered_signal}. {symbol} ({tf}){confirmed_tag}\n"
        f"📈 {signal_type} Signal\n"
        f"💰 {price:.6f}\n"
        f"✅ Score: {score_str}\n"
        f"📌 Passed: {passed_str}\n"
        f"{confidence_icon} Confidence: {confidence:.1f}% (Weighted: {weighted_str})"
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
