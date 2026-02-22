# pec_backtest_v2.py
# NEW VERSION: Loads from signals_fired.jsonl instead of logs.txt
# Uses PEC_CONFIG parameters (MIN_ACCEPTED_RR, MAX_BARS_BY_TF)

import json
import os
import pandas as pd
import datetime
from collections import defaultdict
from pec_config import MIN_ACCEPTED_RR, MAX_BARS_BY_TF, SIGNALS_JSONL_PATH
from pec_engine import find_realistic_exit

def load_signals_from_jsonl(jsonl_path=None):
    """
    Load fired signals from signals_fired.jsonl (JSONL format).
    
    Args:
        jsonl_path: Path to signals_fired.jsonl file
    
    Returns:
        list: List of signal dicts loaded from JSONL
    """
    if jsonl_path is None:
        jsonl_path = SIGNALS_JSONL_PATH
    
    signals = []
    
    if not os.path.exists(jsonl_path):
        print(f"[JSONL_LOADER] Signal file not found: {jsonl_path}")
        return signals
    
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    signal = json.loads(line)
                    # Validate signal has required fields
                    if all(k in signal for k in ['symbol', 'timeframe', 'entry_price', 'tp_target', 'sl_target', 'achieved_rr']):
                        signals.append(signal)
                    else:
                        print(f"[JSONL_LOADER] Skipping incomplete signal: {signal}")
                except json.JSONDecodeError as e:
                    print(f"[JSONL_LOADER] Failed to parse JSON line: {line[:50]}... Error: {e}")
                    continue
        
        print(f"[JSONL_LOADER] Loaded {len(signals)} signals from {jsonl_path}")
    
    except Exception as e:
        print(f"[JSONL_LOADER] Error reading JSONL file: {e}")
    
    return signals

def run_pec_backtest_v2(
    get_ohlcv,
    get_local_wib,
    jsonl_path=None,
    output_csv="pec_batch_results.csv"
):
    """
    Run PEC backtest using signals from signals_fired.jsonl.
    
    Uses:
    - MIN_ACCEPTED_RR from pec_config (filter cutoff)
    - MAX_BARS_BY_TF from pec_config (per-timeframe exit bars)
    - Signal's stored TP/SL prices
    
    Args:
        get_ohlcv: Function to fetch OHLCV data for symbol/timeframe
        get_local_wib: Function to convert UTC to WIB (for logging)
        jsonl_path: Path to signals_fired.jsonl (default: from pec_config)
        output_csv: Output CSV filename for results
    
    Returns:
        dict: Backtest summary stats
    """
    
    print("[PEC_BACKTEST_V2] Starting batch backtest from JSONL...")
    print(f"[PEC_BACKTEST_V2] Config: MIN_ACCEPTED_RR={MIN_ACCEPTED_RR}, MAX_BARS_BY_TF={MAX_BARS_BY_TF}")
    
    # 1. Load signals from JSONL
    signals = load_signals_from_jsonl(jsonl_path)
    if not signals:
        print("[PEC_BACKTEST_V2] No signals to backtest. Exiting.")
        return {"total_signals": 0, "processed": 0, "error": "No signals loaded"}
    
    # 2. Filter signals by MIN_ACCEPTED_RR
    filtered_signals = [s for s in signals if s.get('achieved_rr', 0) >= MIN_ACCEPTED_RR]
    print(f"[PEC_BACKTEST_V2] Filtered {len(filtered_signals)}/{len(signals)} signals with RR >= {MIN_ACCEPTED_RR}")
    
    # 3. Group by symbol + timeframe
    signals_by_pair = defaultdict(list)
    for sig in filtered_signals:
        symbol = sig.get('symbol')
        tf = sig.get('timeframe')
        signals_by_pair[(symbol, tf)].append(sig)
    
    print(f"[PEC_BACKTEST_V2] Processing {len(signals_by_pair)} unique symbol/timeframe pairs...")
    
    # 4. Process each signal
    results = []
    stats = {
        "total_signals": len(filtered_signals),
        "processed": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "by_timeframe": {}
    }
    
    for (symbol, tf), pair_signals in sorted(signals_by_pair.items()):
        print(f"\n[PEC_BACKTEST_V2] Processing {symbol} {tf} ({len(pair_signals)} signals)...")
        
        # Get max_bars for this timeframe
        max_bars = MAX_BARS_BY_TF.get(tf, 20)  # Default to 20 if TF not in config
        
        for signal in pair_signals:
            try:
                # Fetch OHLCV data
                ohlcv_df = get_ohlcv(symbol, interval=tf, limit=500)
                if ohlcv_df is None or ohlcv_df.empty:
                    print(f"  [SKIP] {symbol} {tf}: No OHLCV data available")
                    continue
                
                # Find entry bar by timestamp
                fired_time_utc = pd.to_datetime(signal.get('fired_time_utc'))
                
                # Simple timestamp matching: find closest bar
                if ohlcv_df.index.tz is None:
                    ohlcv_times = pd.to_datetime(ohlcv_df.index)
                else:
                    ohlcv_times = pd.to_datetime(ohlcv_df.index).tz_convert(None)
                
                time_diffs = (ohlcv_times - fired_time_utc).abs()
                entry_idx = time_diffs.argmin()
                
                if entry_idx + max_bars >= len(ohlcv_df):
                    print(f"  [SKIP] {symbol} {tf}: Not enough future bars (need {max_bars})")
                    continue
                
                entry_price = signal.get('entry_price')
                tp_price = signal.get('tp_target')
                sl_price = signal.get('sl_target')
                signal_type = signal.get('signal_type', 'LONG')
                
                # Calculate exit using dynamic max_bars and stored TP/SL
                exit_result = find_realistic_exit(
                    ohlcv_df, entry_idx, entry_price, signal_type,
                    max_bars=max_bars,
                    tp_price=tp_price,
                    sl_price=sl_price
                )
                
                if exit_result is None:
                    print(f"  [SKIP] {symbol} {tf}: Could not determine exit")
                    continue
                
                exit_price = exit_result['exit_price']
                exit_reason = exit_result['exit_reason']
                
                # Calculate PnL
                if signal_type == "LONG":
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                
                is_win = pnl_pct > 0
                
                # Record result
                result = {
                    'symbol': symbol,
                    'timeframe': tf,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'tp_target': tp_price,
                    'sl_target': sl_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct,
                    'result': 'WIN' if is_win else 'LOSS',
                    'fired_time': signal.get('fired_time_utc'),
                    'achieved_rr': signal.get('achieved_rr'),
                    'score': signal.get('score'),
                    'confidence': signal.get('confidence')
                }
                results.append(result)
                
                # Update stats
                stats["processed"] += 1
                if is_win:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                stats["total_pnl"] += pnl_pct
                
                if tf not in stats["by_timeframe"]:
                    stats["by_timeframe"][tf] = {"wins": 0, "losses": 0, "count": 0}
                stats["by_timeframe"][tf]["count"] += 1
                if is_win:
                    stats["by_timeframe"][tf]["wins"] += 1
                else:
                    stats["by_timeframe"][tf]["losses"] += 1
                
                print(f"  ✓ {symbol} {tf} {signal_type}: {entry_price:.6f} → {exit_price:.6f} = {pnl_pct:+.2f}% ({exit_reason})")
            
            except Exception as e:
                print(f"  [ERROR] {symbol} {tf}: {str(e)}")
                continue
    
    # 5. Save results to CSV
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        print(f"\n[PEC_BACKTEST_V2] Results saved to {output_csv}")
    
    # 6. Print summary
    win_rate = (stats["wins"] / stats["processed"] * 100) if stats["processed"] > 0 else 0
    print(f"\n[PEC_BACKTEST_V2] SUMMARY:")
    print(f"  Total Processed: {stats['processed']}/{stats['total_signals']}")
    print(f"  Wins: {stats['wins']} | Losses: {stats['losses']} | Win Rate: {win_rate:.1f}%")
    print(f"  Total PnL: {stats['total_pnl']:+.2f}%")
    print(f"  Average PnL: {(stats['total_pnl']/stats['processed'] if stats['processed'] > 0 else 0):+.2f}%")
    
    if stats["by_timeframe"]:
        print(f"\n  By Timeframe:")
        for tf, tf_stats in sorted(stats["by_timeframe"].items()):
            tf_wr = (tf_stats["wins"] / tf_stats["count"] * 100) if tf_stats["count"] > 0 else 0
            print(f"    {tf}: {tf_stats['wins']}/{tf_stats['count']} wins ({tf_wr:.1f}%)")
    
    stats["output_file"] = output_csv
    return stats

if __name__ == "__main__":
    print("[PEC_BACKTEST_V2] This module is meant to be imported. Use run_pec_backtest_v2() function.")
