import pandas as pd
from datetime import datetime
import csv
import uuid
import os
import re

def parse_fired_log_line(line):
    """
    Parses a [FIRED] log line and returns a dict with all required fields.
    Backward-compatible parser for previous log formats.
    """
    result = {}

    match = re.match(
        r'\[FIRED\] Logged: ([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+)',
        line)
    if not match:
        return None
    result['uuid'] = match.group(1)
    result['symbol'] = match.group(2)
    result['tf'] = match.group(3)
    result['signal_type'] = match.group(4)
    result['entry_time'] = match.group(5)
    result['entry_price'] = match.group(6)

    def extract(pattern, line, default=''):
        m = re.search(pattern, line)
        return m.group(1) if m else default

    result['score'] = extract(r'SCORE:\s*([0-9\.]+)', line)
    result['max_score'] = extract(r'MAX SCORE:\s*([0-9\.]+)', line)
    result['passed'] = extract(r'PASSED:\s*([0-9\.]+)', line)
    result['max_passed'] = extract(r'MAX PASSED:\s*([0-9\.]+)', line)
    result['weights'] = extract(r'WEIGHTS:\s*([0-9\.]+)', line)
    result['max_weights'] = extract(r'MAX WEIGHTS:\s*([0-9\.]+)', line)
    result['confidence_rate'] = extract(r'CONFIDENCE RATE:\s*([0-9\.]+%?)', line)

    for field in ['score', 'max_score', 'passed', 'max_passed', 'weights', 'max_weights', 'confidence_rate']:
        if field not in result or result[field] is None:
            result[field] = ''
    return result

def export_signal_debug_txt(symbol, tf, bias, filter_weights_long, filter_weights_short, gatekeepers,
                           results_long=None, results_short=None,
                           orderbook_result=None, density_result=None,
                           results=None):
    """
    Export a human-readable debug file for a fired (or blocked) signal.
    This variant intentionally marks SuperGK as globally disabled in the verdict section.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("signal_debug_temp.txt", "w") as f:
        f.write(f"# Signal Debug Export (created: {timestamp})\n")

        # LONG results
        if results_long and len(results_long) > 0:
            rows_long = []
            for fname, res in results_long.items():
                rows_long.append({
                    "Symbol": symbol,
                    "Timeframe": tf,
                    "SignalType": "LONG",
                    "Filter Name": fname,
                    "Weight": filter_weights_long.get(fname, 0),
                    "GateKeeper": fname in gatekeepers,
                    "Result": res,
                    "PASSES": "PASS" if fname in gatekeepers and res else ""
                })
            df_long = pd.DataFrame(rows_long)
            df_long = df_long.sort_values("Weight", ascending=False)
            f.write("\n## LONG Filter Results\n")
            df_long.to_csv(f, sep="\t", index=False)
        else:
            f.write("\n(No LONG filter results)\n")

        # SHORT results
        if results_short and len(results_short) > 0:
            rows_short = []
            for fname, res in results_short.items():
                rows_short.append({
                    "Symbol": symbol,
                    "Timeframe": tf,
                    "SignalType": "SHORT",
                    "Filter Name": fname,
                    "Weight": filter_weights_short.get(fname, 0),
                    "GateKeeper": fname in gatekeepers,
                    "Result": res,
                    "PASSES": "PASS" if fname in gatekeepers and res else ""
                })
            df_short = pd.DataFrame(rows_short)
            df_short = df_short.sort_values("Weight", ascending=False)
            f.write("\n## SHORT Filter Results\n")
            df_short.to_csv(f, sep="\t", index=False)
        else:
            f.write("\n(No SHORT filter results)\n")

        # Legacy block (if results provided in older format)
        if (not (results_long and any(results_long.values())) and
            not (results_short and any(results_short.values())) and
            results is not None and len(results) > 0):
            rows = []
            for fname, res in results.items():
                rows.append({
                    "Symbol": symbol,
                    "Timeframe": tf,
                    "SignalType": bias,
                    "Filter Name": fname,
                    "Weight": filter_weights_long.get(fname, 0),
                    "GateKeeper": fname in gatekeepers,
                    "Result": res,
                    "PASSES": "PASS" if fname in gatekeepers and res else ""
                })
            df_legacy = pd.DataFrame(rows)
            df_legacy = df_legacy.sort_values("Weight", ascending=False)
            f.write("\n## (Legacy results)\n")
            df_legacy.to_csv(f, sep="\t", index=False)

    # ---- Validation Verdict: reflect GLOBAL SuperGK DISABLED ----
    verdict_lines = []
    verdict_lines.append(f"\n==== VALIDATION VERDICT ({timestamp}) ====")
    verdict_lines.append(f"Signal: {bias} on {symbol} @ {tf}")

    def _safe_float(d, k, default=0.0):
        try:
            return float(d.get(k, default)) if d is not None else float(default)
        except Exception:
            return float(default)

    buy_wall = _safe_float(orderbook_result, "buy_wall", 0.0)
    sell_wall = _safe_float(orderbook_result, "sell_wall", 0.0)
    wall_delta = _safe_float(orderbook_result, "wall_delta", buy_wall - sell_wall)

    bid_density = _safe_float(density_result, "bid_density", 0.0)
    ask_density = _safe_float(density_result, "ask_density", 0.0)

    total_wall = max(buy_wall + sell_wall, 1e-9)
    wall_pct = (abs(wall_delta) / total_wall) * 100.0
    wall_sign = "LONG" if wall_delta > 0 else "SHORT" if wall_delta < 0 else "NEUTRAL"

    density_diff = bid_density - ask_density
    density_pct = abs(density_diff)
    density_sign = "LONG" if density_diff > 0 else "SHORT" if density_diff < 0 else "NEUTRAL"

    # Show raw metrics but clearly mark SuperGK as disabled
    orderbook_verdict = (
        f"OrderBook Wall:     (bypassed - SuperGK disabled)  "
        f"(wall_delta={wall_delta:.4f}, wall_pct={wall_pct:.3f}%, raw_favors={wall_sign})"
    )

    density_verdict = (
        f"Resting Density:    (bypassed - SuperGK disabled)  "
        f"(bid_density={bid_density:.2f}, ask_density={ask_density:.2f}, density_pct={density_pct:.3f}%, raw_favors={density_sign})"
    )

    # Provide composite info for transparency but clearly mark it's not used
    try:
        # Attempt to compute composite using defaults for transparency (not used for blocking)
        wall_weight = float(os.getenv("SUPERGK_WALL_WEIGHT") or 0.25)
        density_weight = float(os.getenv("SUPERGK_DENSITY_WEIGHT") or 0.75)
        composite_threshold = float(os.getenv("SUPERGK_COMPOSITE_THRESHOLD") or 0.01)
    except Exception:
        wall_weight = 0.25
        density_weight = 0.75
        composite_threshold = 0.01

    wall_score = wall_pct / 100.0
    density_score = density_pct / 100.0

    def _sign_to_factor(sign, desired):
        if sign == desired:
            return 1.0
        elif sign == "NEUTRAL":
            return 0.0
        else:
            return -1.0

    wall_factor = _sign_to_factor(wall_sign if wall_pct >= float(os.getenv("SUPERGK_WALL_PCT_THRESHOLD") or 1.0) else "NEUTRAL", bias)
    density_factor = _sign_to_factor(density_sign if density_pct >= float(os.getenv("SUPERGK_DENSITY_THRESHOLD") or 0.5) else "NEUTRAL", bias)

    composite = wall_weight * wall_factor * wall_score + density_weight * density_factor * density_score
    wall_component = wall_weight * wall_factor * wall_score
    density_component = density_weight * density_factor * density_score

    final_verdict = "BYPASSED (GLOBAL) ✅"
    composite_info = f"(SuperGK disabled in runtime/main.py — not used to block sends) (composite={composite:.6f}, wall_component={wall_component:.6f}, density_component={density_component:.6f}, comp_th={composite_threshold})"

    verdict_lines.append(orderbook_verdict)
    verdict_lines.append(density_verdict)
    verdict_lines.append(f"FINAL VERDICT:      {final_verdict} {composite_info}")
    verdict_lines.append("==== END ====")

    with open("signal_debug_temp.txt", "a") as f:
        for line in verdict_lines:
            f.write("\n" + line)

def log_fired_signal(
    symbol,
    tf,
    signal_type,
    entry_idx=None,
    fired_time=None,
    score=None,
    max_score=None,
    passed=None,
    max_passed=None,
    weights=None,
    max_weights=None,
    confidence_rate=None,
    entry_price=None
):
    """
    Log fired signal information to console and append to a tracking CSV.
    """
    fired_uuid = str(uuid.uuid4())
    fired_time_str = (
        fired_time if isinstance(fired_time, str)
        else fired_time.isoformat() if fired_time is not None and hasattr(fired_time, 'isoformat')
        else datetime.utcnow().isoformat()
    )
    entry_idx_compat = entry_idx if entry_idx is not None else -1

    log_line = (
        f"[FIRED] Logged: {fired_uuid}, {symbol}, {tf}, {signal_type}, {fired_time_str}, {entry_idx_compat}, "
        f"SCORE: {score if score is not None else 'None'}, "
        f"MAX SCORE: {max_score if max_score is not None else 'None'}, "
        f"PASSED: {passed if passed is not None else 'None'}, "
        f"MAX PASSED: {max_passed if max_passed is not None else 'None'}, "
        f"WEIGHTS: {weights if weights is not None else 'None'}, "
        f"MAX WEIGHTS: {max_weights if max_weights is not None else 'None'}, "
        f"CONFIDENCE RATE: {confidence_rate if confidence_rate is not None else 'None'}%"
    )
    print(log_line)

    try:
        with open("signal_tracking.txt", "a") as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write to signal_tracking.txt: {e}")

    log_file = "fired_signals_temp.csv"
    header = [
        "uuid", "symbol", "tf", "signal_type", "fired_time", "entry_idx",
        "score", "max_score", "passed", "max_passed", "weights", "max_weights", "confidence_rate"
    ]
    row = [
        fired_uuid, symbol, tf, signal_type, fired_time_str, entry_idx_compat,
        score, max_score, passed, max_passed, weights, max_weights, confidence_rate
    ]

    try:
        write_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
        with open(log_file, "a", newline='') as f:
            writer = csv.writer(f, delimiter=",")
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print(f"[ERROR] log_fired_signal failed: {e}")

def print_fired_signals_csv():
    print("\n[DEBUG] === Contents of fired_signals_temp.csv ===")
    try:
        with open("fired_signals_temp.csv") as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"[ERROR] Could not read fired_signals_temp.csv: {e}")

if __name__ == "__main__":
    # quick smoke test for the log function
    log_fired_signal(
        symbol="HIPPO-USDT",
        tf="3min",
        signal_type="LONG",
        entry_idx=99,
        fired_time=str(datetime.now()),
        score=16,
        max_score=23,
        passed=13,
        max_passed=17,
        weights=50.1,
        max_weights=65.8,
        confidence_rate=76.1,
        entry_price=0.0
    )
