#!/usr/bin/env python3
"""
batch1_analysis.py

Analyze Batch 1 results with focus on:
- Win rate vs targets
- Exit reason distribution (TP/SL/TIMEOUT)
- Profitability by timeframe and exit reason
- Validation against hybrid exit mechanism goals
"""

import pandas as pd
import sys
from collections import defaultdict

def analyze_batch1(csv_file='batch1_results.csv'):
    """
    Comprehensive Batch 1 analysis.
    
    Shows:
    - Overall metrics (win rate, avg PnL, total PnL)
    - Exit reason breakdown with profitability
    - Timeframe breakdown
    - Verdict on TP/SL mechanism quality
    """
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"\n❌ Results file not found: {csv_file}")
        print(f"   Expected at: {csv_file}")
        return False
    
    total = len(df)
    if total == 0:
        print(f"\n❌ No results in {csv_file}")
        return False
    
    # Overall metrics
    wins = (df['result'] == 'WIN').sum()
    losses = (df['result'] == 'LOSS').sum()
    win_rate = wins / total * 100 if total > 0 else 0
    avg_pnl = df['pnl_pct'].mean()
    total_pnl = df['pnl_pct'].sum()
    std_pnl = df['pnl_pct'].std()
    
    # Exit reason breakdown
    exit_counts = df['exit_reason'].value_counts() if 'exit_reason' in df.columns else None
    
    print("\n" + "="*75)
    print("BATCH 1 RESULTS ANALYSIS")
    print("="*75)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Signals Processed: {total}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg PnL per trade: {avg_pnl:+.2f}%")
    print(f"  Total PnL: {total_pnl:+.2f}%")
    print(f"  Std Dev: {std_pnl:.2f}%")
    
    # Exit reason analysis (if available)
    if exit_counts is not None and len(exit_counts) > 0:
        print(f"\n🎯 EXIT REASON DISTRIBUTION:")
        
        exit_stats = {}
        for reason in ['TP', 'SL', 'TIMEOUT']:
            if reason not in df['exit_reason'].values:
                continue
            
            reason_trades = df[df['exit_reason'] == reason]
            count = len(reason_trades)
            pct = count / total * 100
            reason_wr = (reason_trades['result'] == 'WIN').sum() / count * 100 if count > 0 else 0
            reason_avg = reason_trades['pnl_pct'].mean()
            
            exit_stats[reason] = {
                'count': count,
                'pct': pct,
                'wr': reason_wr,
                'avg_pnl': reason_avg
            }
            
            status = ""
            if reason == 'TP' and 60 <= pct <= 70:
                status = "✓"
            elif reason == 'SL' and 20 <= pct <= 30:
                status = "✓"
            elif reason == 'TIMEOUT' and 5 <= pct <= 10:
                status = "✓"
            
            print(f"  {reason:8s}: {count:3d} trades ({pct:5.1f}%) {status}")
            print(f"             Win Rate: {reason_wr:5.1f}% | Avg PnL: {reason_avg:+.2f}%")
    
    # Target validation
    if exit_stats:
        print(f"\n✅ TARGET METRICS:")
        tp_check = exit_stats.get('TP', {}).get('pct', 0)
        sl_check = exit_stats.get('SL', {}).get('pct', 0)
        timeout_check = exit_stats.get('TIMEOUT', {}).get('pct', 0)
        
        tp_ok = 60 <= tp_check <= 70
        sl_ok = 20 <= sl_check <= 30
        timeout_ok = 5 <= timeout_check <= 10
        
        print(f"  TP exits:    {tp_check:5.1f}% (target: 60-70%) {'✓' if tp_ok else '✗'}")
        print(f"  SL exits:    {sl_check:5.1f}% (target: 20-30%) {'✓' if sl_ok else '✗'}")
        print(f"  TIMEOUT:     {timeout_check:5.1f}% (target: 5-10%) {'✓' if timeout_ok else '✗'}")
    
    # By timeframe
    if 'timeframe' in df.columns:
        print(f"\n📈 BY TIMEFRAME:")
        timeframes = sorted(df['timeframe'].unique())
        
        tf_stats = {}
        for tf in timeframes:
            tf_trades = df[df['timeframe'] == tf]
            tf_wins = (tf_trades['result'] == 'WIN').sum()
            tf_wr = tf_wins / len(tf_trades) * 100 if len(tf_trades) > 0 else 0
            tf_avg = tf_trades['pnl_pct'].mean()
            
            tf_stats[tf] = {'count': len(tf_trades), 'wr': tf_wr, 'avg': tf_avg}
            
            print(f"  {tf:6s}: {len(tf_trades):3d} trades | WR: {tf_wr:5.1f}% | Avg: {tf_avg:+.2f}%")
    
    # By signal type
    if 'signal_type' in df.columns:
        print(f"\n🔄 BY SIGNAL TYPE:")
        for stype in ['LONG', 'SHORT']:
            if stype not in df['signal_type'].values:
                continue
            
            stype_trades = df[df['signal_type'] == stype]
            stype_wins = (stype_trades['result'] == 'WIN').sum()
            stype_wr = stype_wins / len(stype_trades) * 100 if len(stype_trades) > 0 else 0
            stype_avg = stype_trades['pnl_pct'].mean()
            
            print(f"  {stype:6s}: {len(stype_trades):3d} trades | WR: {stype_wr:5.1f}% | Avg: {stype_avg:+.2f}%")
    
    # Verdict
    print(f"\n" + "="*75)
    print("VERDICT:")
    print("="*75)
    
    if win_rate >= 55:
        verdict = "✅ BATCH 1 PASSED"
        detail = f"Win rate {win_rate:.1f}% exceeds 55% target."
        action = "→ Fibonacci TP/SL method is working. Proceed to Batch 2 (150 signals)."
    elif win_rate >= 50:
        verdict = "⚠️  BATCH 1 MARGINAL"
        detail = f"Win rate {win_rate:.1f}% is borderline (50-55% range)."
        action = "→ Review exit distribution. Consider ATR-based TP/SL optimization."
    else:
        verdict = "❌ BATCH 1 FAILED"
        detail = f"Win rate {win_rate:.1f}% below 50% threshold."
        action = "→ Switch to ATR-based TP/SL (2:1 RR), retry Batch 1."
    
    print(f"\n{verdict}")
    print(f"{detail}")
    print(f"{action}")
    
    # Recommendations
    print(f"\n💡 INSIGHTS:")
    if exit_stats:
        tp_pct = exit_stats.get('TP', {}).get('pct', 0)
        if tp_pct > 70:
            print(f"  • High TP hit rate ({tp_pct:.0f}%) - TP targets may be too generous")
        elif tp_pct < 60:
            print(f"  • Low TP hit rate ({tp_pct:.0f}%) - TP targets may be too tight")
        else:
            print(f"  • TP hit rate optimal ({tp_pct:.1f}%)")
        
        sl_pct = exit_stats.get('SL', {}).get('pct', 0)
        if sl_pct > 30:
            print(f"  • High SL hit rate ({sl_pct:.0f}%) - SL too wide, losing trades drag profits")
        elif sl_pct < 20:
            print(f"  • Low SL hit rate ({sl_pct:.0f}%) - SL very tight, consider widening")
        else:
            print(f"  • SL hit rate balanced ({sl_pct:.1f}%)")
    
    if avg_pnl > 0.5:
        print(f"  • Excellent average trade PnL ({avg_pnl:.2f}%)")
    elif avg_pnl > 0:
        print(f"  • Marginal average trade PnL ({avg_pnl:.2f}%) - optimize TP/SL")
    else:
        print(f"  • Negative average PnL ({avg_pnl:.2f}%) - TP/SL needs major revision")
    
    print("\n" + "="*75 + "\n")
    
    return True

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'batch1_results.csv'
    success = analyze_batch1(csv_file)
    sys.exit(0 if success else 1)
