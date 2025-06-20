import requests
import os

BOT_TOKEN = os.getenv("BOT_TOKEN", "7100609549:AAHmeFe0RondzYyPKNuGTTp8HNAuT0PbNJs")
CHAT_ID = os.getenv("CHAT_ID", "-1002857433223")

def send_telegram_alert_grouped(signals, chat_id=CHAT_ID, bot_token=BOT_TOKEN):
    """
    Send grouped, numbered signals to Telegram.

    signals: list of tuples (symbol, signal_type, price, timeframe, score, passed)
    """

    from collections import defaultdict

    grouped = defaultdict(list)
    for sig in signals:
        symbol = sig[0]
        grouped[symbol].append(sig)

    message_lines = []
    num = 1
    for symbol, sigs in grouped.items():
        # Sort by timeframe alphabetically (e.g., "3min", "5min")
        sigs_sorted = sorted(sigs, key=lambda x: x[3])
        for i, sig in enumerate(sigs_sorted):
            letter = chr(ord('a') + i)
            symbol, signal_type, price, timeframe, score, passed = sig
            line = (
                f"{num}.{letter} {symbol} ({timeframe}) {signal_type} Signal\n"
                f"💰 Price: {price}\n"
                f"✅ Score: {score}/18\n"
                f"📌 Passed: {passed}/12\n"
            )
            message_lines.append(line)
        num += 1

    message = "\n".join(message_lines)

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    response = requests.post(url, data=data)
    if response.status_code != 200:
        print(f"Failed to send Telegram message: {response.text}")
    else:
        print("Grouped Telegram alert sent successfully.")
