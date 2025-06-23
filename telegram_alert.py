import os
import requests

# ——— CONFIG ——————————————————————————
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7100609549:AAHmeFe0RondzYyPKNuGTTp8HNAuT0PbNJs")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "-1002857433223")
SEND_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

MAX_WEIGHT = 48.8

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
    Format varies by timeframe (3min includes "[V19 Confirmed]").
    """
    confirmed_tag = " [V19 Confirmed]" if tf == "3min" else ""

    # --- Fix 1: Auto compute confidence if not passed ---
    if confidence is None and weighted is not None:
        confidence = (weighted / MAX_WEIGHT) * 100
    confidence = round(confidence, 1) if confidence is not None else 0.0

    # --- Fix 2: Format weighted properly ---
    weighted_str = f"{weighted:.1f}/{MAX_WEIGHT}" if weighted is not None else f"0.0/{MAX_WEIGHT}"

    # --- Assign confidence icon ---
    confidence_icon = (
        "🟢" if confidence >= 75 else
        "🟡" if confidence >= 60 else
        "🔴"
    )

    # --- Build final message ---
    message = (
        f"{numbered_signal}. {symbol} ({tf}){confirmed_tag}\n"
        f"📈 {signal_type} Signal\n"
        f"💰 {price:.6f}\n"
        f"✅ Score: {score}\n"
        f"📌 Passed: {passed}\n"
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
