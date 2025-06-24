import pandas as pd

def dump_signal_debug_txt(symbol, tf, bias, filter_weights, gatekeepers, results):
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
    # Sort by Weight descending before saving
    df = df.sort_values("Weight", ascending=False)
    df.to_csv("signal_debug_temp.txt", sep="\t", index=False)
    
