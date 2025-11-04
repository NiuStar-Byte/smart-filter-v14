import os
import requests
from tg_config import BOT_TOKEN, CHAT_ID
from check_symbols import get_token_blockchain_info

SEND_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
SEND_FILE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"


def _safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        try:
            return float(str(v))
        except Exception:
            return default


def _is_number(v):
    return _safe_float(v, None) is not None


def _compute_fallback_tp_sl(entry_price, signal_type):
    """
    Compute conservative fallback TP/SL percentages when TP/SL are missing.
    Environment variables:
      - FALLBACK_TP_PCT (default 0.02 -> 2%)
      - FALLBACK_SL_PCT (default 0.01 -> 1%)
    Returns: (tp, sl, tp_pct, sl_pct)
    """
    try:
        tp_pct = float(os.getenv("FALLBACK_TP_PCT", "0.02"))
    except Exception:
        tp_pct = 0.02
    try:
        sl_pct = float(os.getenv("FALLBACK_SL_PCT", "0.01"))
    except Exception:
        sl_pct = 0.01

    entry = _safe_float(entry_price, 0.0)
    if signal_type and str(signal_type).strip().upper() == "SHORT":
        tp = entry * (1.0 - tp_pct)
        sl = entry * (1.0 + sl_pct)
    else:
        tp = entry * (1.0 + tp_pct)
        sl = entry * (1.0 - sl_pct)

    return tp, sl, tp_pct, sl_pct


def _format_price_pct(target, base, signal_type):
    """
    Format price and percent relative to base as in the existing style.
    For LONG:
      pct = (target - base)/base * 100
    For SHORT:
      tp_pct = (base - tp)/base * 100  (positive indicates favorable move)
      sl_pct = (base - sl)/base * 100  (negative/sl above entry => negative)
    The function returns string: "<price> (±X.XX%)"
    """
    try:
        base_f = _safe_float(base, None)
        tgt_f = _safe_float(target, None)
        if base_f is None or tgt_f is None or base_f == 0:
            return f"{tgt_f if tgt_f is not None else target} (0.00%)"
        stype = str(signal_type).strip().upper() if signal_type is not None else ""
        if stype == "SHORT":
            # Profit for SHORT when target (tp) is below entry: compute (entry - tp)/entry
            pct = (base_f - tgt_f) / base_f * 100.0
            # For SL (which is > entry), pct will be negative
        else:
            pct = (tgt_f - base_f) / base_f * 100.0
        sign = "+" if pct >= 0 else ""
        return f"{tgt_f:.6f} ({sign}{pct:.2f}%)"
    except Exception:
        return f"{target} (0.00%)"


def send_telegram_alert(
    numbered_signal: str,
    symbol: str,
    signal_type: str,      # 'LONG' or 'SHORT'
    Route: str,        # route description
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
    early_breakout_3m=None,
    early_breakout_5m=None,
    tp=None,
    sl=None
) -> bool:
    """
    Send Telegram alert. Returns True on success, False on failure.

    Ensures TP/SL are always displayed for both LONG and SHORT:
     - If tp or sl missing, compute fallback from FALLBACK_TP_PCT / FALLBACK_SL_PCT env vars.
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("[Telegram] BOT_TOKEN or CHAT_ID not configured (tg_config).", flush=True)
        return False

    # Defensive: normalize numeric-like inputs
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    try:
        weighted_val = float(weighted)
    except Exception:
        weighted_val = _safe_float(weighted, 0.0)

    try:
        total_weight_val = float(total_weight)
    except Exception:
        total_weight_val = _safe_float(total_weight, 0.0)

    weighted_str = f"{weighted_val:.1f}/{total_weight_val:.1f}" if total_weight_val else f"{weighted_val:.1f}/0.0"

    confidence_icon = (
        "🟢" if confidence >= 76 else
        "🟡" if confidence >= 51 else
        "🔴"
    )

    stype = str(signal_type).upper() if signal_type is not None else "UNKNOWN"
    if stype == "LONG":
        signal_icon = "✈️"
        signal_str = "LONG"
    elif stype == "SHORT":
        signal_icon = "🛩️"
        signal_str = "SHORT"
    else:
        signal_icon = "❓"
        signal_str = stype

    # Route / reversal formatting (preserve existing semantics)
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
    elif route_upper == "TREND CONTINUATION" or route_upper == "CONTINUATION":
        if isinstance(reversal_side, str) and "BULLISH" in reversal_side:
            route_icon = "↗️➡️"
            route_str = "Bullish Continuation"
        elif isinstance(reversal_side, str) and "BEARISH" in reversal_side:
            route_icon = "↘️➡️"
            route_str = "Bearish Continuation"
        else:
            route_icon = "➡️➡️"
            route_str = "Continuation Trend"
    elif route_upper in ["NONE", "NO ROUTE", "?", "", None]:
        route_icon = "🚫"
        route_str = "NO ROUTE"
    else:
        route_icon = "❓"
        route_str = f"{Route if Route else 'NO ROUTE'}"

    # Regime string
    if regime == "BULL":
        regime_str = "📈 Regime: <b>BULL</b>\n"
    elif regime == "BEAR":
        regime_str = "📉 Regime: <b>BEAR</b>\n"
    elif regime == "RANGE":
        regime_str = "🚧 Regime: <b>RANGING/SIDEWAYS</b>\n"
    else:
        regime_str = "💢 Regime: <b>NO REGIME</b>\n"

    # Price formatting
    price_float = _safe_float(price, None)
    price_str = f"{price_float:.6f}" if price_float is not None else str(price)

    # Early breakout info
    early_breakout_msg = ""
    if early_breakout_3m and isinstance(early_breakout_3m, dict) and early_breakout_3m.get("valid_signal"):
        eb_bias = early_breakout_3m.get("bias", "N/A")
        eb_price = early_breakout_3m.get("price", "N/A")
        early_breakout_msg += f"\n⚡ <b>3min Early Breakout</b>: {eb_bias} @ {eb_price}"
    if early_breakout_5m and isinstance(early_breakout_5m, dict) and early_breakout_5m.get("valid_signal"):
        eb_bias = early_breakout_5m.get("bias", "N/A")
        eb_price = early_breakout_5m.get("price", "N/A")
        early_breakout_msg += f"\n⚡ <b>5min Early Breakout</b>: {eb_bias} @ {eb_price}"

    # Ensure TP/SL present; compute fallback if missing or non-numeric
    tp_val = _safe_float(tp, None)
    sl_val = _safe_float(sl, None)
    fallback_used = False
    if tp_val is None or sl_val is None:
        fb_tp, fb_sl, fb_tp_pct, fb_sl_pct = _compute_fallback_tp_sl(price_float if price_float is not None else 0.0, stype)
        if tp_val is None:
            tp_val = fb_tp
        if sl_val is None:
            sl_val = fb_sl
        fallback_used = True
        print(f"[Telegram] TP/SL missing or invalid for {symbol}. Using fallback TP/SL (tp_pct={fb_tp_pct}, sl_pct={fb_sl_pct}).", flush=True)

    # Format TP/SL displays with percent relative to entry and sign conventions
    tp_display = _format_price_pct(tp_val, price_float if price_float is not None else 0.0, stype)
    sl_display = _format_price_pct(sl_val, price_float if price_float is not None else 0.0, stype)

    # Token consensus info
    token_info = None
    try:
        token_info = get_token_blockchain_info(symbol)
    except Exception:
        token_info = None

    if token_info:
        consensus_str = f"\n🔗 Consensus: <b>{token_info.get('consensus', 'Unknown')}</b> ({token_info.get('blockchain', 'Unknown')})"
    else:
        consensus_str = "\n🔗 Consensus: <b>Unknown</b>"

    # Build message (keeps previous HTML formatting style)
    lines = []
    lines.append(f"{numbered_signal}. {symbol} ({tf})")
    lines.append(f"{regime_str}")
    lines.append(f"{signal_icon} {signal_str} Signal")
    lines.append(f"{route_icon} <b>{route_str}</b>")
    lines.append(f"💰 <b>{price_str}</b>")
    lines.append(f"🏁 <b>TP:</b> <code>{tp_display}</code>")
    lines.append(f"⛔ <b>SL:</b> <code>{sl_display}</code>")
    lines.append(f"📊 Score: {score}/{score_max}")
    lines.append(f"🎯 Passed: {passed}/{gatekeepers_total}")
    lines.append(f"{confidence_icon} Confidence: {confidence:.1f}%")
    lines.append(f"🏋️‍♀️ Weighted: {weighted_str}")
    if early_breakout_msg:
        lines.append(early_breakout_msg)
    lines.append(consensus_str)
    if fallback_used:
        lines.append(f"\n⚠️ Note: TP/SL fallback used")
    message = "\n".join(lines)

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    resp = None
    try:
        resp = requests.post(SEND_URL, json=payload, timeout=10)
        resp.raise_for_status()
        # Try to show some response detail if available
        try:
            rj = resp.json()
            if rj.get("ok"):
                mid = rj.get("result", {}).get("message_id")
                print(f"📨 Telegram alert sent: {symbol} {signal_type} {Route} message_id={mid}", flush=True)
            else:
                print(f"📨 Telegram responded but ok=False: {rj}", flush=True)
        except Exception:
            print(f"📨 Telegram alert sent (no JSON result) for: {symbol} {signal_type} {Route}", flush=True)
        return True
    except requests.RequestException as e:
        resp_text = ""
        try:
            resp_text = resp.text if resp is not None else ""
        except Exception:
            resp_text = str(e)
        print(f"❗ Telegram send error: {e} — response: {resp_text}", flush=True)
        return False


def send_telegram_file(filepath, caption=None) -> bool:
    """
    Send a local file to the Telegram group as a document. Returns True on success.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}", flush=True)
        return False
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
            print(f"📄 File sent to Telegram: {filepath}", flush=True)
            return True
        except requests.RequestException as e:
            resp_text = resp.text if resp is not None else ''
            print(f"❗ Telegram file send error: {e} — response: {resp_text}", flush=True)
            return False
