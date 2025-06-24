def dump_signal_debug_txt(symbol, tf, bias, filter_weights, gatekeepers, results):
    rows = []
    for fname, res in results.items():
        rows.append({
            "Symbol": symbol,
            "Timeframe": tf,
            "SignalType": bias,
            "Filter Name": fname,
            "Weight": float(filter_weights.get(fname, 0)),   # ensure float for sorting
            "GateKeeper": fname in gatekeepers,
            "Result": res,
            "PASSES": "PASS" if fname in gatekeepers and res else ""
        })
    df = pd.DataFrame(rows)
    # Robust sort: if Weight missing, treat as -inf so they go last
    df = df.sort_values("Weight", ascending=False, na_position="last")
    df.to_csv("signal_debug_temp.txt", sep="\t", index=False)
