import os
import requests
from tg_config import BOT_TOKEN, CHAT_ID
from check_symbols import get_token_blockchain_info

SEND_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
SEND_FILE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"

def send_telegram_alert(
    numbered_signal: str,
    symbol: str,
    signal_type: str,      # 'LONG' or 'SHORT'
    Route: str,        # 'REVERSAL' or 'CONTINUATION'
    price: float,
    tf: str,
    score: int,
    passed: int,
    confidence: float,
    weighted: float,
    score_max: int,
    gatekeepers_total: int,
    total_weight: float,
    reversal_side=None,
    regime=None,
    early_breakout_3m=None,   # <-- NEW PARAM
    early_breakout_5m=None,    # <-- NEW PARAM
    tp=None,               # <-- NEW ARG
    sl=None                # <-- NEW ARG
) -> None:
    print(f"📨 Telegram alert sent: {symbol} {signal_type} {Route} @ {price}")
    confirmed_tag = " [Confirmed]" if tf == "3min" else ""

    if isinstance(score, (tuple, list)): score = score[0]
    if isinstance(passed, (tuple, list)): passed = passed[0]
    if isinstance(score_max, (tuple, list)): score_max = score_max[0]
    if isinstance(gatekeepers_total, (tuple, list)): gatekeepers_total = gatekeepers_total[0]
    if isinstance(weighted, (tuple, list)): weighted = weighted[0]
    if isinstance(total_weight, (tuple, list)): total_weight = total_weight[0]

    try:
        confidence = round((weighted / total_weight) * 100, 1) if total_weight else 0.0
    except Exception:
        confidence = 0.0

    weighted_str = f"{weighted:.1f}/{total_weight:.1f}" if total_weight else "0.0/0.0"

    confidence_icon = (
        "🟢" if confidence >= 76 else
        "🟡" if confidence >= 51 else
        "🔴"
    )

    if str(signal_type).upper() == "LONG":
        signal_icon = "✈️"
        signal_str = "LONG"
    elif str(signal_type).upper() == "SHORT":
        signal_icon = "🛩️"
        signal_str = "SHORT"
    else:
        signal_icon = "❓"
        signal_str = str(signal_type).upper()

    route_upper = str(Route).upper() if Route is not None else "NO ROUTE"
    route_icon = "❓"
    route_str = "NO ROUTE"
    
    if route_upper == "REVERSAL":
        if isinstance(reversal_side, (list, set)) and "BULLISH" in reversal_side and "BEARISH" in reversal_side:
            route_icon = "🔃🔄"
            route_str = "Ambiguous Reversal Trend"
        elif reversal_side == "BULLISH":
            route_icon = "↗️🔄"
            route_str = "Bullish Reversal Trend"
        elif reversal_side == "BEARISH":
            route_icon = "↘️🔄"
            route_str = "Bearish Reversal Trend"
        else:
            route_icon = "🔄🔄"
            route_str = "Reversal Trend"
    elif route_upper == "AMBIGUOUS":
        route_icon = "🔃🔄"
        route_str = "Ambiguous Reversal Trend"
    elif route_upper == "TREND CONTINUATION":
        if isinstance(reversal_side, str):
            if "BULLISH" in reversal_side:
                route_icon = "↗️➡️"
                route_str = "Bullish Continuation"
            elif "BEARISH" in reversal_side:
                route_icon = "↘️➡️"
                route_str = "Bearish Continuation"
            else:
                route_icon = "➡️➡️"
                route_str = "Continuation Trend"
        else:
            route_icon = "➡️➡️"
            route_str = "Continuation Trend"
    elif route_upper in ["CONTINUATION"]:
        route_icon = "➡️➡️"
        route_str = "Continuation Trend"
    elif route_upper in ["NONE", "NO ROUTE", "?", "", None]:
        route_icon = "🚫"
        route_str = "NO ROUTE"
    else:
        route_icon = "❓"
        route_str = f"{Route if Route else 'NO ROUTE'}"

    if regime == "BULL":
        regime_str = "📈 Regime: <b>BULL</b>\n"
    elif regime == "BEAR":
        regime_str = "📉 Regime: <b>BEAR</b>\n"
    elif regime == "RANGE":
        regime_str = "🚧 Regime: <b>RANGING/SIDEWAYS</b>\n"
    else:
        regime_str = "💢 Regime: <b>NO REGIME</b>\n"
        
    try:
        price_float = float(price)
        price_str = f"{price_float:.6f}"
    except Exception:
        price_float = None
        price_str = str(price)

    early_breakout_msg = ""
    if early_breakout_3m and isinstance(early_breakout_3m, dict) and early_breakout_3m.get("valid_signal"):
        eb_bias = early_breakout_3m.get("bias", "N/A")
        eb_price = early_breakout_3m.get("price", "N/A")
        early_breakout_msg += f"\n⚡ <b>3min Early Breakout</b>: {eb_bias} @ {eb_price}"
    if early_breakout_5m and isinstance(early_breakout_5m, dict) and early_breakout_5m.get("valid_signal"):
        eb_bias = early_breakout_5m.get("bias", "N/A")
        eb_price = early_breakout_5m.get("price", "N/A")
        early_breakout_msg += f"\n⚡ <b>5min Early Breakout</b>: {eb_bias} @ {eb_price}"

    print(f"[DEBUG] price={price}, tp={tp}, sl={sl}, signal_type={signal_type}")
    tp_sl_msg = ""
    tp_pct_str = ""
    sl_pct_str = ""
    
    def is_valid_number(val):
        try:
            float(val)
            return True
        except (TypeError, ValueError):
            return False
    
    if is_valid_number(tp) and is_valid_number(sl) and is_valid_number(price):
        tp_float = float(tp)
        sl_float = float(sl)
        price_float = float(price)
    
        tp_str = f"{tp_float:.6f}"
        sl_str = f"{sl_float:.6f}"
    
        signal_type_test = str(signal_type).strip().upper()
    
        if signal_type_test == "LONG":
            tp_pct = ((tp_float - price_float) / price_float) * 100
            sl_pct = ((sl_float - price_float) / price_float) * 100
        elif signal_type_test == "SHORT":
            tp_pct = ((price_float - tp_float) / price_float) * 100
            sl_pct = ((price_float - sl_float) / price_float) * 100
        else:
            tp_pct = None
            sl_pct = None
    
        tp_pct_str = f" ({tp_pct:+.2f}%)" if tp_pct is not None else ""
        sl_pct_str = f" ({sl_pct:+.2f}%)" if sl_pct is not None else ""
    
        print(f"[DEBUG] Calculated TP_pct={tp_pct_str}, SL_pct={sl_pct_str}")
    
    else:
        tp_str = str(tp) if tp is not None else "-"
        sl_str = str(sl) if sl is not None else "-"
        print(f"[DEBUG] TP/SL not valid numbers, using raw strings")
    
    tp_sl_msg = (
        f"🏁 <b>TP:</b> <code>{tp_str}</code>{tp_pct_str}\n"
        f"⛔ <b>SL:</b> <code>{sl_str}</code>{sl_pct_str}\n"
    )

    token_info = get_token_blockchain_info(symbol)
    if token_info:
        consensus_str = f"\n🔗 Consensus: <b>{token_info['consensus']}</b> ({token_info['blockchain']})"
    else:
        consensus_str = "\n🔗 Consensus: <b>Unknown</b>"
        
    message = (
        f"{numbered_signal}. {symbol} ({tf}){confirmed_tag}\n"
        f"{regime_str}"
        f"{signal_icon} {signal_str} Signal\n"
        f"{route_icon} <b>{route_str}</b>\n"
        f"💰 <b>{price_str}</b>\n"
        f"{tp_sl_msg}"
        f"📊 Score: {score}/{score_max}\n"
        f"🎯 Passed: {passed}/{gatekeepers_total}\n"
        f"{confidence_icon} Confidence: {confidence:.1f}%\n"
        f"🏋️‍♀️ Weighted: {weighted_str}"
        f"{early_breakout_msg}"
        f"{consensus_str}"
    )
    
    print("Signal type:", signal_type, "Route:", Route, "reversal_side:", reversal_side)
    
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    resp = None
    try:
        resp = requests.post(SEND_URL, json=payload, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        resp_text = resp.text if resp is not None else ''
        print(f"❗ Telegram send error: {e} — response: {resp_text}")

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
        resp = None
        try:
            resp = requests.post(SEND_FILE_URL, data=data, files=files, timeout=20)
            resp.raise_for_status()
            print(f"📄 File sent to Telegram: {filepath}")
        except requests.RequestException as e:
            resp_text = resp.text if resp is not None else ''
            print(f"❗ Telegram file send error: {e} — response: {resp_text}")
