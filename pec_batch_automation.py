#!/usr/bin/env python3
"""
PEC Batch Automation Script - Zero Confirmation Workflow

Automatically:
1. Monitors signal accumulation
2. Triggers backtest at 50+ signals
3. Generates comprehensive report bundle
4. Saves to ~/Desktop/PEC Batch[N]/

No manual intervention needed. Just run this in background.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
SIGNALS_JSONL = '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/signals_fired.jsonl'
WORKSPACE = '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main'
DESKTOP = os.path.expanduser('~/Desktop')
BATCH_TRIGGER_COUNT = 50  # Trigger backtest at 50 signals

def count_signals():
    """Count signals in JSONL file."""
    if not os.path.exists(SIGNALS_JSONL):
        return 0
    try:
        with open(SIGNALS_JSONL, 'r') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def get_batch_number():
    """Determine next batch number based on existing Desktop folders."""
    batch_num = 1
    while os.path.exists(os.path.join(DESKTOP, f'PEC Batch{batch_num}')):
        batch_num += 1
    return batch_num

def run_backtest():
    """Run PEC backtest on accumulated signals."""
    print(f"\n{'='*70}")
    print(f"[PEC_AUTOMATION] TRIGGERING BACKTEST")
    print(f"{'='*70}\n")
    
    os.chdir(WORKSPACE)
    result = subprocess.run(
        ['python3', 'run_batch1_backtest.py'],
        env={**os.environ, 'SIGNALS_JSONL_PATH': 'signals_fired.jsonl'},
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"[ERROR] Backtest failed: {result.stderr}")
        return False
    
    print(f"[SUCCESS] Backtest completed")
    return True

def generate_reports():
    """Generate comprehensive report bundle."""
    print(f"\n[PEC_AUTOMATION] GENERATING REPORTS")
    
    batch_num = get_batch_number()
    batch_dir = os.path.join(DESKTOP, f'PEC Batch{batch_num}')
    os.makedirs(batch_dir, exist_ok=True)
    
    # Run report generation script
    script = f"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

csv_path = '{WORKSPACE}/pec_batch1_results.csv'
output_dir = '{batch_dir}'

# Load CSV
df = pd.read_csv(csv_path)

# === EXCEL 1: RAW RESULTS ===
excel1 = f'{batch_dir}/1_BATCH{batch_num}_RAW_RESULTS.xlsx'
with pd.ExcelWriter(excel1, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='All Trades', index=False)
print(f"✅ {{excel1}}")

# === EXCEL 2: EXIT DISTRIBUTION ===
exit_dist_data = []
for reason in ['TP', 'SL', 'TIMEOUT']:
    mask = df['exit_reason'] == reason
    trades = df[mask]
    wins = (trades['result'] == 'WIN').sum()
    losses = (trades['result'] == 'LOSS').sum()
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    avg_pnl = trades['pnl_pct'].mean() if total > 0 else 0
    
    exit_dist_data.append({{
        'Exit Reason': reason,
        'Total Trades': total,
        'Wins': wins,
        'Losses': losses,
        'Win Rate %': round(win_rate, 2),
        'Avg PnL %': round(avg_pnl, 2),
        'Total PnL %': round(trades['pnl_pct'].sum(), 2),
        'Min PnL %': round(trades['pnl_pct'].min(), 2),
        'Max PnL %': round(trades['pnl_pct'].max(), 2)
    }})

exit_df = pd.DataFrame(exit_dist_data)
excel2 = f'{batch_dir}/2_EXIT_DISTRIBUTION_ANALYSIS.xlsx'
with pd.ExcelWriter(excel2, engine='openpyxl') as writer:
    exit_df.to_excel(writer, sheet_name='Exit Reasons', index=False)
print(f"✅ {{excel2}}")

# === EXCEL 3: TIMEFRAME ANALYSIS ===
tf_data = []
for tf in ['15min', '30min', '1h']:
    mask = df['timeframe'] == tf
    trades = df[mask]
    if len(trades) == 0:
        continue
    
    wins = (trades['result'] == 'WIN').sum()
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    
    tp_count = (trades['exit_reason'] == 'TP').sum()
    sl_count = (trades['exit_reason'] == 'SL').sum()
    timeout_count = (trades['exit_reason'] == 'TIMEOUT').sum()
    
    tf_data.append({{
        'Timeframe': tf,
        'Total Trades': total,
        'Wins': wins,
        'Losses': total - wins,
        'Win Rate %': round(win_rate, 2),
        'Avg PnL %': round(trades['pnl_pct'].mean(), 2),
        'Total PnL %': round(trades['pnl_pct'].sum(), 2),
        'TP Exits': tp_count,
        'SL Exits': sl_count,
        'TIMEOUT Exits': timeout_count,
        'Avg Hold Bars': round(trades['hold_bars'].mean(), 1)
    }})

tf_df = pd.DataFrame(tf_data)
excel3 = f'{batch_dir}/3_TIMEFRAME_ANALYSIS.xlsx'
with pd.ExcelWriter(excel3, engine='openpyxl') as writer:
    tf_df.to_excel(writer, sheet_name='By Timeframe', index=False)
print(f"✅ {{excel3}}")

# === EXCEL 4: SYMBOL ANALYSIS ===
symbol_data = []
for symbol in sorted(df['symbol'].unique()):
    mask = df['symbol'] == symbol
    trades = df[mask]
    wins = (trades['result'] == 'WIN').sum()
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    
    symbol_data.append({{
        'Symbol': symbol,
        'Total Trades': total,
        'Wins': wins,
        'Losses': total - wins,
        'Win Rate %': round(win_rate, 2),
        'Avg PnL %': round(trades['pnl_pct'].mean(), 2),
        'Total PnL %': round(trades['pnl_pct'].sum(), 2),
        'Best Trade %': round(trades['pnl_pct'].max(), 2),
        'Worst Trade %': round(trades['pnl_pct'].min(), 2),
        'Avg RR': round(trades['achieved_rr'].mean(), 2)
    }})

symbol_df = pd.DataFrame(symbol_data)
excel4 = f'{batch_dir}/4_SYMBOL_ANALYSIS.xlsx'
with pd.ExcelWriter(excel4, engine='openpyxl') as writer:
    symbol_df.to_excel(writer, sheet_name='By Symbol', index=False)
print(f"✅ {{excel4}}")

# === SUMMARY REPORT ===
total_trades = len(df)
total_wins = (df['result'] == 'WIN').sum()
win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
total_pnl = df['pnl_pct'].sum()
avg_pnl = df['pnl_pct'].mean()

summary = f'''================================================================================
BATCH {batch_num} PEC BACKTEST - EXECUTIVE SUMMARY
Generated: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}}
================================================================================

OVERALL METRICS
------------------------------------------------------------------------
Total Signals: {{total_trades}}
Winning Trades: {{total_wins}}
Losing Trades: {{total_trades - total_wins}}
Win Rate: {{win_rate:.2f}}%
Total P&L: {{total_pnl:.2f}}%
Average P&L/Trade: {{avg_pnl:.2f}}%

EXIT DISTRIBUTION
------------------------------------------------------------------------
'''

for _, row in exit_df.iterrows():
    summary += f'''
{{row['Exit Reason']}} Exits: {{int(row['Total Trades'])}} trades ({{row['Total Trades']/total_trades*100:.1f}}%)
  Wins: {{int(row['Wins'])}} | Losses: {{int(row['Losses'])}} | Win Rate: {{row['Win Rate %']:.2f}}%
  Avg PnL: {{row['Avg PnL %']:.2f}}% | Total PnL: {{row['Total PnL %']:.2f}}%
'''

summary += f'''
================================================================================
VERDICT
================================================================================
Win Rate: {{win_rate:.2f}}% (Target: 55%+ for profitability)
'''

if win_rate >= 55:
    summary += f"✓ PASS: Method is WORKING - Ready for live trading\\n"
elif win_rate >= 50:
    summary += f"⚠ MARGINAL: Win rate acceptable but optimize for consistency\\n"
else:
    summary += f"✗ FAIL: Method needs refinement - Continue optimization\\n"

summary_path = f'{batch_dir}/0_EXECUTIVE_SUMMARY.txt'
with open(summary_path, 'w') as f:
    f.write(summary)
print(f"✅ {{summary_path}}")

print(f"\\n✅ ALL REPORTS GENERATED - {{batch_dir}}/")
"""
    
    try:
        exec(script)
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        return False
    
    return True

def monitor_and_trigger():
    """Continuously monitor signals and trigger backtest when threshold reached."""
    print(f"\n{'='*70}")
    print(f"[PEC_AUTOMATION] STARTING BATCH MONITOR")
    print(f"{'='*70}\n")
    
    while True:
        try:
            signal_count = count_signals()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if signal_count >= BATCH_TRIGGER_COUNT:
                print(f"[{timestamp}] ✅ Signals accumulated: {signal_count}/{BATCH_TRIGGER_COUNT}")
                print(f"[{timestamp}] TRIGGERING BACKTEST & REPORT GENERATION")
                
                # Run backtest
                if not run_backtest():
                    print(f"[ERROR] Backtest failed, retrying...")
                    continue
                
                # Generate reports
                if not generate_reports():
                    print(f"[ERROR] Report generation failed, retrying...")
                    continue
                
                batch_num = get_batch_number() - 1
                print(f"\n{'='*70}")
                print(f"✅ BATCH {batch_num} COMPLETE")
                print(f"📁 Reports saved to: ~/Desktop/PEC Batch{batch_num}/")
                print(f"{'='*70}\n")
                
                # Reset for next batch (clear old signals)
                print(f"[INFO] Ready for Batch {batch_num + 1}")
                print(f"[INFO] Monitoring signals...")
            else:
                print(f"[{timestamp}] Signals: {signal_count}/{BATCH_TRIGGER_COUNT}")
            
            # Wait 60 seconds before next check
            import time
            time.sleep(60)
        
        except KeyboardInterrupt:
            print(f"\n[INFO] Automation stopped by user")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] {e}")
            import time
            time.sleep(60)

if __name__ == '__main__':
    monitor_and_trigger()
