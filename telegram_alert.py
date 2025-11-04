import os
import requests
from tg_config import BOT_TOKEN, CHAT_ID
from check_symbols import get_token_blockchain_info
from typing import Any, Optional

SEND_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
SEND_FILE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        try:
            return float(str(v))
        except Exception:
            return default


def _fmt_price(v: Any, precision: int = 6) -> str:
    try:
        f = float(v)
        if f >= 100:
            return f"{f:,.2f}"
        if f >= 1:
            return f"{f:,.4f}"
        return f"{f:.{precision}f}"
    except Exception:
        return str(v)


def _pct_for_tp(entry: float, tp: float, signal_type: str) -> str:
    """
    Percent shown for TP:
      - SHORT: (entry - tp) / entry * 100 => positive if profitable
      - LONG:  (tp - entry) / entry * 100 => positive if profitable
    Returns formatted string like "+1.23%"
    """
    try:
        if entry == 0:
            return "0.00%"
        st = str(signal_type).strip().upper()
        if st == "SHORT":
            pct = (entry - tp) / entry * 100.0
        else:
            pct = (tp - entry) / entry * 100.0
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.2f}%"
    except Exception:
        return "N/A"


def _pct_for_sl(entry: float, sl: float, signal_type: str) -> str:
    """
    Percent shown for SL:
      - SHORT: (entry - sl) / entry * 100 -> negative when SL is above entry (since entry - sl < 0)
      - LONG:  (sl - entry) / entry * 100 -> negative when SL is below entry
    Returns formatted string with sign, e.g. "-1.23%"
    """
    try:
        if entry == 0:
            return "0.00%"
        st = str(signal_type).strip().upper()
        if st == "SHORT":
            pct = (entry - sl) / entry * 100.0
        else:
            pct = (sl - entry) / entry * 100.0
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.2f}%"
    except Exception:
        return "N/A"


def send_telegram_alert(
    numbered_signal: str,
    symbol: str,
    signal_type: str,      # 'LONG' or 'SHORT'
    Route: str,            # route description
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
    sl=None,
    # optional rich data: either pass a dict via tp_sl or pass chosen_ratio/achieved_rr directly
    tp_sl: Optional[dict] = None,
    chosen_ratio: Optional[float] = None,
    achieved_rr: Optional[float] = None,
) -> bool:
    """
    Send a Telegram formatted alert message without extra blank lines.
    Accepts optional tp_sl dict (as returned from calculate_tp_sl) or chosen_ratio/achieved_rr params.
    """

    if not BOT_TOKEN or not CHAT_ID:
        print("[Telegram] BOT_TOKEN or CHAT_ID not configured (tg_config).", flush=True)
        return False

    # Normalize numeric-like inputs
    try:
        confidence_val = float(confidence)
    except Exception:
        confidence_val = 0.0

    try:
        weighted_val = float(weighted)
    except Exception:
        weighted_val = _safe_float(weighted, 0.0) or 0.0

    try:
        total_weight_val = float(total_weight)
    except Exception:
        total_weight_val = _safe_float(total_weight, 0.0) or 0.0

    weighted_str = f"{weighted_val:.1f}/{total_weight_val:.1f}" if total_weight_val else f"{weighted_val:.1f}/0.0"

    confidence_icon = "🟢" if confidence_val >= 76 else ("🟡" if confidence_val >= 51 else "🔴")

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

    # Route / reversal formatting (preserve semantics)
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
    elif route_upper in ["TREND CONTINUATION", "CONTINUATION"]:
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

    # Regime line (no trailing newline)
    regime_display = regime if regime else "N/A"

    # Price formatting
    price_float = _safe_float(price, None)
    price_display = _fmt_price(price_float) if price_float is not None else str(price)

    # Early breakout info (single-line entries)
    early_break_lines = []
    if early_breakout_3m and isinstance(early_breakout_3m, dict) and early_breakout_3m.get("valid_signal"):
        eb_bias = early_breakout_3m.get("bias", "N/A")
        eb_price = early_breakout_3m.get("price", "N/A")
        early_break_lines.append(f"⚡ {tf} Early Breakout: {eb_bias} @ {eb_price}")
    if early_breakout_5m and isinstance(early_breakout_5m, dict) and early_breakout_5m.get("valid_signal"):
        eb_bias = early_breakout_5m.get("bias", "N/A")
        eb_price = early_breakout_5m.get("price", "N/A")
        early_break_lines.append(f"⚡ 5min Early Breakout: {eb_bias} @ {eb_price}")

    # Ensure TP/SL present and numeric; compute fallback if missing
    tp_val = _safe_float(tp, None)
    sl_val = _safe_float(sl, None)
    fallback_used = False
    if tp_sl and isinstance(tp_sl, dict):
        # Prefer structured tp_sl if provided
        tp_val = _safe_float(tp_sl.get("tp"), tp_val)
        sl_val = _safe_float(tp_sl.get("sl"), sl_val)
        if chosen_ratio is None:
            chosen_ratio = tp_sl.get("chosen_ratio")
        if achieved_rr is None:
            achieved_rr = tp_sl.get("achieved_rr")

    if tp_val is None or sl_val is None:
        # compute conservative fallback; read env vars
        try:
            tp_pct = float(os.getenv("FALLBACK_TP_PCT", "0.02"))
        except Exception:
            tp_pct = 0.02
        try:
            sl_pct = float(os.getenv("FALLBACK_SL_PCT", "0.01"))
        except Exception:
            sl_pct = 0.01

        entry = price_float if price_float is not None else 0.0
        if stype == "SHORT":
            if tp_val is None:
                tp_val = entry * (1.0 - tp_pct)
            if sl_val is None:
                sl_val = entry * (1.0 + sl_pct)
        else:
            if tp_val is None:
                tp_val = entry * (1.0 + tp_pct)
            if sl_val is None:
                sl_val = entry * (1.0 - sl_pct)
        fallback_used = True
        print(f"[Telegram] TP/SL missing or invalid for {symbol}. Using fallback TP/SL (tp_pct={tp_pct}, sl_pct={sl_pct}).", flush=True)

    # Format TP/SL displays
    tp_pct_display = _pct_for_tp(price_float if price_float is not None else 0.0, tp_val, stype)
    sl_pct_display = _pct_for_sl(price_float if price_float is not None else 0.0, sl_val, stype)
    tp_display = f"{_fmt_price(tp_val)} ({tp_pct_display})"
    sl_display = f"{_fmt_price(sl_val)} ({sl_pct_display})"

    # Token consensus info (single-line)
    token_info = None
    try:
        token_info = get_token_blockchain_info(symbol)
    except Exception:
        token_info = None
    if token_info:
        consensus_display = f"🔗 Consensus: {token_info.get('consensus', 'Unknown')} ({token_info.get('blockchain', 'Unknown')})"
    else:
        consensus_display = "🔗 Consensus: Unknown"

    # Fib ratio and achieved_rr line (if provided)
    fib_rr_line = ""
    if chosen_ratio is not None or achieved_rr is not None:
        ratio_str = f"{chosen_ratio}" if chosen_ratio is not None else "N/A"
        rr_str = f"{achieved_rr:.2f}" if (achieved_rr is not None and isinstance(achieved_rr, (int, float))) else "N/A"
        fib_rr_line = f"🎯 Fib: {ratio_str} | R:R: {rr_str}"

    # Build message lines WITHOUT any blank empty lines
    lines = []
    lines.append(f"{numbered_signal}. {symbol} ({tf})")
    lines.append(f"{'📉' if regime_display == 'BEAR' else ('📈' if regime_display == 'BULL' else '🔎')} Regime: {regime_display}")
    lines.append(f"{signal_icon} {signal_str} Signal")
    lines.append(f"{route_icon} {route_str}")
    lines.append(f"💰 {price_display}")
    lines.append(f"🏁 TP: {tp_display}")
    lines.append(f"⛔ SL: {sl_display}")
    lines.append(f"📊 Score: {score}/{score_max}")
    lines.append(f"🎯 Passed: {passed}/{gatekeepers_total}")
    lines.append(f"{confidence_icon} Confidence: {confidence_val:.1f}%")
    lines.append(f"🏋️‍♀️ Weighted: {weighted_str}")
    # append early breakout lines (each on its own single line)
    for eb in early_break_lines:
        lines.append(eb)
    # consensus & fib/rr & fallback note
    lines.append(consensus_display)
    if fib_rr_line:
        lines.append(fib_rr_line)
    if fallback_used:
        lines.append("⚠️ Note: TP/SL fallback used")

    # Join with single newline (no extra blank lines)
    message = "\n".join(lines)

    # Print the message locally for debugging
    print(f"[Telegram][MSG] {numbered_signal}. {symbol}\n{message}", flush=True)

    # Send to Telegram
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        resp = requests.post(SEND_URL, json=payload, timeout=10)
        resp.raise_for_status()
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
            resp_text = resp.text if resp is not None else str(e)
        except Exception:
            resp_text = str(e)
        print(f"❗ Telegram send error: {e} — response: {resp_text}", flush=True)
        return False


def send_telegram_file(filepath: str, caption: Optional[str] = None) -> bool:
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
