import pandas as pd
from datetime import datetime

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
