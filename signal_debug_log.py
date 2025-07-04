import pandas as pd

def dump_signal_debug_txt(*args, **kwargs):
    pass

def log_fired_signal(symbol, tf, signal_type, entry_idx):
    import csv, uuid, os
    from datetime import datetime

    print(f"[DEBUG] log_fired_signal called: {symbol}, {tf}, {signal_type}, {entry_idx}")

    log_file = "fired_signals_temp.csv"
    header = ["uuid", "symbol", "tf", "signal_type", "fired_time", "entry_idx"]
    fired_time = datetime.utcnow().isoformat()
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

def dump_signal_debug_txt(symbol, tf, bias, filter_weights, gatekeepers, results,
                         orderbook_result=None, density_result=None):
    """
    Dumps per-filter debug info for a single signal, sorted by weight descending,
    to 'signal_debug_temp.txt' (tab-separated), and appends validation verdict & timestamp.
    """
    # --- Timestamp section ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for fname, res in results.items():
        rows.append({
            "Symbol": symbol,
            "Timeframe": tf,
            "SignalType": bias,
            "Filter Name": fname,
            "Weight": filter_weights.get(fname, 0),
            "GateKeeper": fname in gatekeepers,
            "Result": res,
            "PASSES": "PASS" if fname in gatekeepers and res else ""
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Weight", ascending=False)

    # --- Write DataFrame + timestamp at the top ---
    with open("signal_debug_temp.txt", "w") as f:
        f.write(f"# Signal Debug Export (created: {timestamp})\n")
    df.to_csv("signal_debug_temp.txt", sep="\t", index=False, mode="a")

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
