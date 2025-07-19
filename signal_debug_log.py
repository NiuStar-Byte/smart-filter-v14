import pandas as pd
from datetime import datetime
import csv
import uuid
import os
import re

def parse_fired_log_line(line):
    """
    Parses a [FIRED] log line and returns a dict with all required fields.
    Supports fields: SCORE, MAX SCORE, PASSED, MAX PASSED, WEIGHTS, MAX WEIGHTS, CONFIDENCE RATE.
    Returns empty string for missing values (backward compatibility).
    Example line:
    [FIRED] Logged: 86ff64e8-ad63-4961-820b-d6ac9afaef46, AERO-USDT, 5min, LONG, 2025-07-16T11:44:16.217760, 99, SCORE: 17, MAX SCORE: 23, PASSED: 14, MAX PASSED: 17, WEIGHTS: 54.7, MAX WEIGHTS: 65.8, CONFIDENCE RATE: 83.1%
    """
    result = {}

    # First part: id, symbol, tf, signal_type, entry_time, entry_price
    match = re.match(
        r'\[FIRED\] Logged: ([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+)',
        line)
    if not match:
        return None  # Not a valid [FIRED] line
    result['uuid'] = match.group(1)
    result['symbol'] = match.group(2)
    result['tf'] = match.group(3)
    result['signal_type'] = match.group(4)
    result['entry_time'] = match.group(5)
    result['entry_price'] = match.group(6)

    # Extract additional fields using regex
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

    # Backward compatibility: if any missing, set to ''
    for field in ['score', 'max_score', 'passed', 'max_passed', 'weights', 'max_weights', 'confidence_rate']:
        if field not in result or result[field] is None:
            result[field] = ''
    return result

import pandas as pd
from datetime import datetime

def export_signal_debug_txt(symbol, tf, bias, filter_weights_long, filter_weights_short, gatekeepers,
                           results_long=None, results_short=None,
                           orderbook_result=None, density_result=None,
                           results=None, filename="signal_debug_temp.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "w") as f:
        f.write(f"# Signal Debug Export (created: {timestamp})\n")

        # --- Write LONG results (with weight map check) ---
        rows_long = []
        if results_long is not None:
            for fname, res in results_long.items():
                weight = filter_weights_long.get(fname)
                if weight is None:
                    print(f"[WARNING] Filter '{fname}' not found in filter_weights_long map!")
                rows_long.append({
                    "Symbol": symbol,
                    "Timeframe": tf,
                    "SignalType": "LONG",
                    "Filter Name": fname,
                    "Weight": weight if weight is not None else 0,
                    "WeightMissing": weight is None,
                    "GateKeeper": fname in gatekeepers,
                    "Result": res,
                    "PASSES": "PASS" if fname in gatekeepers and res else ""
                })
            df_long = pd.DataFrame(rows_long)
            df_long = df_long.sort_values("Weight", ascending=False)
            f.write("\n## LONG Filter Results\n")
            df_long.to_csv(f, sep="\t", index=False)

        # --- Write SHORT results (with weight map check) ---
        rows_short = []
        if results_short is not None:
            for fname, res in results_short.items():
                weight = filter_weights_short.get(fname)
                if weight is None:
                    print(f"[WARNING] Filter '{fname}' not found in filter_weights_short map!")
                rows_short.append({
                    "Symbol": symbol,
                    "Timeframe": tf,
                    "SignalType": "SHORT",
                    "Filter Name": fname,
                    "Weight": weight if weight is not None else 0,
                    "WeightMissing": weight is None,
                    "GateKeeper": fname in gatekeepers,
                    "Result": res,
                    "PASSES": "PASS" if fname in gatekeepers and res else ""
                })
            df_short = pd.DataFrame(rows_short)
            df_short = df_short.sort_values("Weight", ascending=False)
            f.write("\n## SHORT Filter Results\n")
            df_short.to_csv(f, sep="\t", index=False)


        # --- Legacy results, for backward compatibility ---
        if (not (results_long and any(results_long.values())) and 
            not (results_short and any(results_short.values())) and 
            results is not None and len(results) > 0):
            rows = []
            for fname, res in results.items():
                if bias == "LONG":
                    weight = filter_weights_long.get(fname, 0)
                elif bias == "SHORT":
                    weight = filter_weights_short.get(fname, 0)
                else:
                    weight = 0
                rows.append({
                    "Symbol": symbol,
                    "Timeframe": tf,
                    "SignalType": bias,
                    "Filter Name": fname,
                    "Weight": weight,
                    "GateKeeper": fname in gatekeepers,
                    "Result": res,
                    "PASSES": "PASS" if fname in gatekeepers and res else ""
                })
            df_legacy = pd.DataFrame(rows)
            df_legacy = df_legacy.sort_values("Weight", ascending=False)
            f.write("\n## (Legacy results)\n")
            df_legacy.to_csv(f, sep="\t", index=False)

    # ---- Automated Validation Verdict Section ----
    verdict_lines = []
    verdict_lines.append(f"\n==== VALIDATION VERDICT ({timestamp}) ====")
    verdict_lines.append(f"Signal: {bias} on {symbol} @ {tf}")

    # OrderBook logic
    orderbook_verdict = ""
    if orderbook_result is not None:
        wall_delta = orderbook_result.get('wall_delta', 0)
        wall_bias = "LONG" if wall_delta > 0 else "SHORT" if wall_delta < 0 else "NEUTRAL"
        signal_conflict = (bias != wall_bias and wall_bias != "NEUTRAL")
        orderbook_verdict = (
            f"OrderBook Wall:     {'Conflict' if signal_conflict else 'Aligned'}  "
            f"(wall_delta={wall_delta:.4f}, favors {wall_bias})"
        )
    else:
        orderbook_verdict = "OrderBook Wall:     N/A"

    # Density logic
    density_verdict = ""
    if density_result is not None:
        bid_density = density_result.get("bid_density", 0)
        ask_density = density_result.get("ask_density", 0)
        density_bias = "LONG" if bid_density > ask_density else "SHORT" if ask_density > bid_density else "NEUTRAL"
        density_conflict = (bias != density_bias and density_bias != "NEUTRAL")
        density_verdict = (
            f"Resting Density:    {'Conflict' if density_conflict else 'Aligned'}  "
            f"(bid_density={bid_density:.2f}, ask_density={ask_density:.2f}, favors {density_bias})"
        )
    else:
        density_verdict = "Resting Density:    N/A"

    # Final overall verdict
    final_verdict = "ALIGNED ✅"
    if (
        (orderbook_result is not None and bias != ("LONG" if orderbook_result.get('wall_delta', 0) > 0 else "SHORT"))
        or
        (density_result is not None and bias != ("LONG" if density_result.get("bid_density", 0) > density_result.get("ask_density", 0) else "SHORT"))
    ):
        final_verdict = "CONTRADICTORY ❌"

    verdict_lines.append(orderbook_verdict)
    verdict_lines.append(density_verdict)
    verdict_lines.append(f"FINAL VERDICT:      {final_verdict}")
    verdict_lines.append("==== END ====")

    # Append verdict section to file
    with open(filename, "a") as f:
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
    confidence_rate=None
):
    """
    Log fired signal information to console for log-based parsing,
    with enhanced tracking of additional signal metrics.

    Args:
        symbol: Trading symbol (e.g., "BTC-USDT")
        tf: Timeframe (e.g., "3min", "5min")
        signal_type: Signal direction ("LONG" or "SHORT")
        entry_idx: Index position in DataFrame (optional)
        fired_time: UTC timestamp when signal was fired (optional)
        score: Score value for the signal (optional)
        max_score: Maximum possible score (optional)
        passed: Number of passed checks (optional)
        max_passed: Maximum passed checks (optional)
        weights: Weight value (optional)
        max_weights: Maximum possible weights (optional)
        confidence_rate: Confidence rate as a percentage (optional)
    """
    print(f"[DEBUG] log_fired_signal called: {symbol}, {tf}, {signal_type}, entry_idx={entry_idx}, fired_time={fired_time}")

    fired_uuid = str(uuid.uuid4())
    fired_time_str = (
        fired_time if isinstance(fired_time, str)
        else fired_time.isoformat() if fired_time is not None and hasattr(fired_time, 'isoformat')
        else datetime.utcnow().isoformat()
    )
    entry_idx_compat = entry_idx if entry_idx is not None else -1

    # Enhanced log output
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

    # DEPRECATED: CSV logging kept for backward compatibility
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

    return fired_uuid

def print_fired_signals_csv():
    print("\n[DEBUG] === Contents of fired_signals_temp.csv ===")
    try:
        with open("fired_signals_temp.csv") as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"[ERROR] Could not read fired_signals_temp.csv: {e}")

# Example usage for enhanced log output
if __name__ == "__main__":
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
        confidence_rate=76.1
    )
