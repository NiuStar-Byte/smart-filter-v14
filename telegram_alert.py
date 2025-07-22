import os
import requests
from tg_config import BOT_TOKEN, CHAT_ID

# ——— CONFIG ——————————————————————————
SEND_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
SEND_FILE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"

def send_telegram_alert(
    numbered_signal: str,
    symbol: str,
    signal_type: str,      # 'REVERSAL' or 'CONTINUATION'
    direction: str,        # 'LONG' or 'SHORT'
    price: float,
    tf: str,
    score: int,
    passed: int,
    confidence: float,
    weighted: float,
    score_max: int,
    gatekeepers_total: int,
    total_weight: float,
) -> None:
    print(f"[DEBUG] signal_type received in send_telegram_alert: '{signal_type}'")
    """
    Sends a formatted Telegram message to your channel/group.
    Format varies by timeframe (3min includes "[Confirmed]").
    'signal_type' should be 'REVERSAL' or 'CONTINUATION'.
    """
    confirmed_tag = " [Confirmed]" if tf == "3min" else ""

    # Fix values if accidentally tuple/list
    for vname in ['score', 'passed', 'score_max', 'gatekeepers_total', 'weighted', 'total_weight']:
        v = locals()[vname]
        if isinstance(v, (tuple, list)):
            locals()[vname] = v[0]

    # --- Fix values if accidentally tuple/str ---
    if isinstance(score, (tuple, list)): score = score[0]
    if isinstance(passed, (tuple, list)): passed = passed[0]
    if isinstance(score_max, (tuple, list)): score_max = score_max[0]
    if isinstance(gatekeepers_total, (tuple, list)): gatekeepers_total = gatekeepers_total[0]
    if isinstance(weighted, (tuple, list)): weighted = weighted[0]
    if isinstance(total_weight, (tuple, list)): total_weight = total_weight[0]

    # --- Recalculate confidence (for safety) ---
    try:
        confidence = round((weighted / total_weight) * 100, 1) if total_weight else 0.0
    except Exception:
        confidence = 0.0

    # --- Format weighted display ---
    weighted_str = f"{weighted:.1f}/{total_weight:.1f}" if total_weight else "0.0/0.0"

    # --- Confidence icon ---
    confidence_icon = (
        "🟢" if confidence >= 80 else
        "🟡" if confidence >= 70 else
        "🔴"
    )

    # --- Signal type icon and string logic (already present) ---
    if str(signal_type).upper() == "REVERSAL":
        trend_type = "REVERSAL"
        trend_icon = "🔄"
    elif str(signal_type).upper() == "CONTINUATION":
        trend_type = "TREND CONTINUATION"
        trend_icon = "➡️"
    else:
        trend_type = str(signal_type).upper() if signal_type else "UNKNOWN"
        trend_icon = "❓"

    # --- Direction icon and string logic (existing block for LONG/SHORT) ---
    if str(direction).upper() == "LONG":
        direction_icon = "📈"
        direction_str = "LONG"
    elif str(direction).upper() == "SHORT":
        direction_icon = "📉"
        direction_str = "SHORT"
    else:
        direction_icon = "❓"
        direction_str = str(direction).upper() if direction else "UNKNOWN"

    # Place this line here
    signal_type_str = signal_type_str if signal_type_str else "UNKNOWN"

    # --- Final message format ---
    message = (
        f"{numbered_signal}. {symbol} ({tf}){confirmed_tag}\n"
        f"{direction_icon} {direction_str} Signal\n"
        f"{trend_icon} <b>{trend_type}</b>\n"  # <<-- New line for REVERSAL or TREND CONTINUATION
        f"💰 <b>{price:.6f}</b>\n"
        f"📊 Score: {score}/{score_max}\n"
        f"🎯 Passed: {passed}/{gatekeepers_total}\n"
        f"{confidence_icon} Confidence: {confidence:.1f}%\n"
        f"🏋️‍♀️ Weighted: {weighted_str}"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        resp = requests.post(SEND_URL, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"📨 Telegram alert sent: {symbol} {signal_type_str} @ {price}")
    except requests.RequestException as e:
        print(f"❗ Telegram send error: {e} — response: {getattr(resp, 'text', '')}")

def send_telegram_file(filepath, caption=None):
    """
    Send a local file to the Telegram group as a document.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return
    with open(filepath, 'rb') as f:
        files = {'document': f}
        data = {
            'chat_id': CHAT_ID,
            'caption': caption or "Signal debug log"
        }
        try:
            resp = requests.post(SEND_FILE_URL, data=data, files=files, timeout=20)
            resp.raise_for_status()
            print(f"📄 File sent to Telegram: {filepath}")
        except requests.RequestException as e:
            print(f"❗ Telegram file send error: {e} — response: {getattr(resp, 'text', '')}")
