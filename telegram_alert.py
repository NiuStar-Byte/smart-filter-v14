import os
import requests
from tg_config import BOT_TOKEN, CHAT_ID

# ——— CONFIG ——————————————————————————
SEND_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
SEND_FILE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"

def send_telegram_alert(
    numbered_signal: str,
    symbol: str,
    signal_type: str,
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
    """
    Sends a formatted Telegram message to your channel/group.
    Format varies by timeframe (3min includes "[V19 Confirmed]").
    """
    confirmed_tag = " [V19 Confirmed]" if tf == "3min" else ""

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
        "🟢" if confidence >= 75 else
        "🟡" if confidence >= 60 else
        "🔴"
    )

    # --- Final message format (ALWAYS English, clear X/Y only) ---
    message = (
        f"{numbered_signal}. {symbol} ({tf}){confirmed_tag}\n"
        f"📈 {signal_type} Signal\n"
        f"💰 {price:.6f}\n"
        f"✅ Score: {score}/{score_max}\n"
        f"📌 Passed: {passed}/{gatekeepers_total}\n"
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

def send_telegram_file(filepath, caption=None):
    """
    Send a local file to the Telegram group as a document.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return
    files = {'document': open(filepath, 'rb')}
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

def send_csv_to_telegram(csv_path):
    url = SEND_FILE_URL
    try:
        with open(csv_path, "rb") as csvfile:
            files = {"document": csvfile}
            data = {"chat_id": CHAT_ID, "caption": "Fired Signals CSV"}
            response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            print("CSV sent successfully!")
        else:
            print(f"Failed to send CSV: {response.text}")
    except FileNotFoundError:
        print(f"File {csv_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def log_fired_signal_txt(symbol, tf, signal_type, entry_idx, txt_path="fired_signals_temp.txt"):
    """
    Log fired signal to TXT file in CSV format.
    Generates UUID and fired_time, writes header only once, appends CSV row as string.
    """
    import uuid, os
    from datetime import datetime
    
    print(f"[DEBUG] log_fired_signal_txt called: {symbol}, {tf}, {signal_type}, {entry_idx}")
    
    # Generate UUID and timestamp
    fired_uuid = str(uuid.uuid4())
    fired_time = datetime.utcnow().isoformat()
    
    # Prepare header and row
    header = "uuid,symbol,tf,signal_type,fired_time,entry_idx"
    row = f"{fired_uuid},{symbol},{tf},{signal_type},{fired_time},{entry_idx}"
    
    # Check if we need to write header (file is empty or missing)
    write_header = False
    try:
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                content = f.read().strip()
                if content == "":
                    write_header = True
        else:
            write_header = True
    except Exception:
        write_header = True
    
    # Write to file
    try:
        with open(txt_path, "a") as f:
            if write_header:
                print(f"[DEBUG] Writing header to {txt_path}")
                f.write(header + "\n")
            print(f"[DEBUG] Writing row to {txt_path}: {row}")
            f.write(row + "\n")
        
        # Print to log as required
        print(f"[FIRED] Logged to TXT: {fired_uuid}, {symbol}, {tf}, {signal_type}, {fired_time}, {entry_idx}")
        
    except Exception as e:
        print(f"[ERROR] log_fired_signal_txt failed: {e}")

def send_txt_to_telegram(txt_path="fired_signals_temp.txt", caption="Signals as TXT"):
    """
    Send TXT file to Telegram using existing file sending logic.
    """
    if not os.path.exists(txt_path):
        print(f"[ERROR] TXT file not found: {txt_path}")
        return
        
    try:
        send_telegram_file(txt_path, caption)
        print(f"[INFO] TXT file sent to Telegram: {txt_path}")
    except Exception as e:
        print(f"[ERROR] Failed to send TXT file to Telegram: {e}")

if __name__ == "__main__":
    # send_csv_to_telegram("fired_signals_temp.csv")  # Commented out - now using TXT files
    pass
