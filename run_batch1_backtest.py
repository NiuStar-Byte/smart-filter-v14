#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main')

from pec_backtest_v2 import run_pec_backtest_v2
from kucoin_data import get_ohlcv
from main import get_local_wib

print("[BATCH_1] Starting PEC backtest on signals_fired.jsonl...")
results = run_pec_backtest_v2(
    get_ohlcv=get_ohlcv,
    get_local_wib=get_local_wib,
    output_csv="pec_batch1_results.csv"
)
print(f"\n[BATCH_1] Backtest complete!")
print(f"[BATCH_1] Summary:")
print(f"  - Total signals: {results.get('total_signals', 0)}")
print(f"  - Filtered signals (RR >= 1.25): {results.get('filtered_signals', 0)}")
print(f"  - Winning signals: {results.get('winning_signals', 0)}")
print(f"  - Losing signals: {results.get('losing_signals', 0)}")
print(f"  - Win rate: {results.get('win_rate', 0):.2f}%")
print(f"  - Total P&L: ${results.get('total_pnl', 0):,.2f}")
print(f"\n[BATCH_1] Results saved to: pec_batch1_results.csv")
