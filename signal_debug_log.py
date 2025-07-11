import pandas as pd
from datetime import datetime, timedelta
import pytz

def dump_signal_debug_txt(*args, **kwargs):
    pass

def dump_signal_debug_txt(symbol, tf, bias, filter_weights_long, filter_weights_short, gatekeepers,
                         results_long=None, results_short=None,
                         orderbook_result=None, density_result=None,
                         results=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("signal_debug_temp.txt", "w") as f:
        f.write(f"# Signal Debug Export (created: {timestamp})\n")

        # --- Write LONG results if available ---
        if results_long is not None and len(results_long) > 0:
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

        # --- Write SHORT results if available ---
        if results_short is not None and len(results_short) > 0:
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

        # --- For backward compatibility: write 'results' if no long/short provided ---
        if (not results_long and not results_short) and results is not None:
            rows = []
            for fname, res in results.items():
                rows.append({
                    "Symbol": symbol,
                    "Timeframe": tf,
                    "SignalType": bias,
                    "Filter Name": fname,
                    "Weight": filter_weights_long.get(fname, 0),  # or whichever is appropriate
                    "GateKeeper": fname in gatekeepers,
                    "Result": res,
                    "PASSES": "PASS" if fname in gatekeepers and res else ""
                })
            df = pd.DataFrame(rows)
            df = df.sort_values("Weight", ascending=False)
            f.write("\n## (Legacy results)\n")
            df.to_csv(f, sep="\t", index=False)

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
    with open("signal_debug_temp.txt", "a") as f:
        for line in verdict_lines:
            f.write("\n" + line)

def log_fired_signal(symbol, tf, signal_type, entry_idx):
    import csv, uuid, os
    from datetime import datetime

    print(f"[DEBUG] log_fired_signal called: {symbol}, {tf}, {signal_type}, {entry_idx}")

    log_file = "fired_signals_temp.csv"
    header = ["uuid", "symbol", "tf", "signal_type", "fired_time", "entry_idx"]
    fired_time = datetime.now(pytz.UTC).isoformat()
    row = [
        str(uuid.uuid4()),
        symbol,
        tf,
        signal_type,
        fired_time,
        entry_idx
    ]

    print("[DEBUG] Current working directory:", os.getcwd())
    print("[DEBUG] Contents of cwd:", os.listdir())

    try:
        write_header = False
        try:
            if os.path.exists(log_file):
                print(f"[DEBUG] {log_file} exists.")
            else:
                print(f"[DEBUG] {log_file} does NOT exist. Will create.")
            with open(log_file, "r", newline='') as f:
                content = f.read().strip()
                print(f"[DEBUG] Existing content length: {len(content)}")
                if content == "":
                    write_header = True
        except FileNotFoundError:
            print("[DEBUG] FileNotFoundError - header will be written.")
            write_header = True

        with open(log_file, "a", newline='') as f:
            writer = csv.writer(f, delimiter=",")
            if write_header:
                print("[DEBUG] Writing header:", header)
                writer.writerow(header)
            print("[DEBUG] Writing row:", row)
            writer.writerow(row)
        print("[DEBUG] CSV write completed successfully.")
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

def get_recent_signals_wib(minutes=720):
    """
    Filter signals from fired_signals_temp.csv based on WIB time reference.
    
    Args:
        minutes (int): Number of minutes to look back from current WIB time (default 720)
    
    Returns:
        list: List of signal records within the specified time window
    """
    import csv
    
    # Define WIB timezone (UTC+07:00)
    wib_tz = pytz.timezone('Asia/Jakarta')
    utc_tz = pytz.UTC
    
    # Get current WIB time and calculate threshold
    wib_now = datetime.now(wib_tz)
    threshold_wib = wib_now - timedelta(minutes=minutes)
    
    # Convert threshold to UTC for comparison with fired_time
    threshold_utc = threshold_wib.astimezone(utc_tz)
    
    print(f"[DEBUG] WIB Now: {wib_now.isoformat()}")
    print(f"[DEBUG] Filtering signals fired after: {threshold_wib.isoformat()} (WIB)")
    print(f"[DEBUG] UTC threshold for comparison: {threshold_utc.isoformat()}")
    
    recent_signals = []
    
    try:
        with open("fired_signals_temp.csv", "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Parse fired_time (stored in UTC)
                    fired_time_str = row.get('fired_time', '')
                    if not fired_time_str:
                        continue
                    
                    # Handle both ISO format with and without timezone info
                    if fired_time_str.endswith('Z'):
                        fired_time_str = fired_time_str[:-1] + '+00:00'
                    elif '+' not in fired_time_str and 'T' in fired_time_str:
                        # Assume UTC if no timezone specified
                        fired_time_utc = datetime.fromisoformat(fired_time_str).replace(tzinfo=utc_tz)
                    else:
                        fired_time_utc = datetime.fromisoformat(fired_time_str)
                        if fired_time_utc.tzinfo is None:
                            fired_time_utc = fired_time_utc.replace(tzinfo=utc_tz)
                    
                    # Compare with threshold
                    if fired_time_utc >= threshold_utc:
                        recent_signals.append(dict(row))
                        print(f"[DEBUG] INCLUDED: {row.get('symbol')} {row.get('signal_type')} at {fired_time_str}")
                    else:
                        print(f"[DEBUG] EXCLUDED: {row.get('symbol')} {row.get('signal_type')} at {fired_time_str} (too old)")
                        
                except Exception as e:
                    print(f"[WARNING] Could not parse signal record: {row}, error: {e}")
                    continue
                    
    except FileNotFoundError:
        print("[DEBUG] fired_signals_temp.csv not found")
    except Exception as e:
        print(f"[ERROR] Error reading fired_signals_temp.csv: {e}")
    
    print(f"[DEBUG] Found {len(recent_signals)} signals within last {minutes} minutes (WIB time)")
    return recent_signals

def print_recent_signals_csv(minutes=720):
    """
    Print signals from fired_signals_temp.csv that are within the specified time window
    based on WIB time reference.
    
    Args:
        minutes (int): Number of minutes to look back from current WIB time (default 720)
    """
    print(f"\n[DEBUG] === Recent signals within last {minutes} minutes (WIB time reference) ===")
    recent_signals = get_recent_signals_wib(minutes)
    
    if not recent_signals:
        print("No recent signals found.")
        return
    
    # Print header
    if recent_signals:
        headers = list(recent_signals[0].keys())
        print(",".join(headers))
        
        # Print each signal
        for signal in recent_signals:
            values = [str(signal.get(h, '')) for h in headers]
            print(",".join(values))
