#!/usr/bin/env python3
"""
Track 1.25:1 TP/SL Ratio Fix (2026-03-18)
Measures: Win Rate, Avg Win/Loss, P&L per signal
Baseline: 46.02% WR, avg loss $22.37 > avg win $14.81
Target: Tighter losses, improved P&L
"""

import json
from datetime import datetime
from pathlib import Path

# Load signals
signal_file = Path("/Users/geniustarigan/.openclaw/workspace/SENT_SIGNALS.jsonl")
if not signal_file.exists():
    print("❌ Signal file not found")
    exit(1)

signals = []
try:
    with open(signal_file) as f:
        for line in f:
            try:
                signals.append(json.loads(line))
            except:
                pass
except Exception as e:
    print(f"Error reading signals: {e}")
    exit(1)

print(f"\n📊 1.25:1 RATIO FIX TRACKING")
print(f"{'='*80}")
print(f"Total signals: {len(signals)}")

# Filter by NEW (Mar 16+)
cutoff_date = datetime(2026, 3, 16)
new_signals = [s for s in signals if datetime.fromisoformat(s.get('fired_time_utc', '').replace('Z', '+00:00')) >= cutoff_date]
print(f"NEW signals (Mar 16+): {len(new_signals)}")

if not new_signals:
    print("⚠️  No new signals yet. Waiting for post-deployment generation...")
    exit(0)

# Count by exit
tp_count = sum(1 for s in new_signals if s.get('exit_type') == 'TP_HIT')
sl_count = sum(1 for s in new_signals if s.get('exit_type') == 'SL_HIT')
timeout_count = sum(1 for s in new_signals if s.get('exit_type') == 'TIMEOUT')
open_count = sum(1 for s in new_signals if s.get('exit_type') == 'OPEN')

print(f"\n✅ RESULTS (NEW signals only):")
print(f"  TP_HIT:    {tp_count}")
print(f"  SL_HIT:    {sl_count}")
print(f"  TIMEOUT:   {timeout_count}")
print(f"  OPEN:      {open_count}")

# Win rate
closed = tp_count + sl_count + timeout_count
timeout_wins = sum(1 for s in new_signals if s.get('exit_type') == 'TIMEOUT' and s.get('pnl_usd', 0) > 0)
wins = tp_count + timeout_wins

if closed > 0:
    wr = (wins / closed) * 100
    print(f"\n📈 WIN RATE: {wr:.2f}% ({wins}/{closed})")
else:
    print(f"\n⚠️  No closed trades yet")
    exit(0)

# P&L metrics
closed_trades = [s for s in new_signals if s.get('exit_type') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']]

tp_pnl = [s.get('pnl_usd', 0) for s in closed_trades if s.get('exit_type') == 'TP_HIT']
sl_pnl = [s.get('pnl_usd', 0) for s in closed_trades if s.get('exit_type') == 'SL_HIT']
timeout_pnl = [s.get('pnl_usd', 0) for s in closed_trades if s.get('exit_type') == 'TIMEOUT']

total_pnl = sum(tp_pnl) + sum(sl_pnl) + sum(timeout_pnl)
avg_tp = sum(tp_pnl) / len(tp_pnl) if tp_pnl else 0
avg_sl = sum(sl_pnl) / len(sl_pnl) if sl_pnl else 0
avg_timeout = sum(timeout_pnl) / len(timeout_pnl) if timeout_pnl else 0

print(f"\n💰 P&L BREAKDOWN:")
print(f"  Avg Win (TP):     ${avg_tp:+.2f}")
print(f"  Avg Loss (SL):    ${avg_sl:+.2f}")
print(f"  Avg Timeout:      ${avg_timeout:+.2f}")
print(f"  Total P&L:        ${total_pnl:+.2f}")
print(f"  Avg per Signal:   ${(total_pnl / len(closed_trades) if closed_trades else 0):+.2f}")

# Compare to baseline
baseline_avg_win = 14.81
baseline_avg_loss = -22.37
current_gap = abs(avg_tp) - abs(avg_sl)
baseline_gap = abs(baseline_avg_win) - abs(baseline_avg_loss)

print(f"\n📊 VS BASELINE (46.02% WR sample):")
print(f"  Baseline avg loss: ${baseline_avg_loss:+.2f}")
print(f"  Current avg loss:  ${avg_sl:+.2f}")
print(f"  Improvement:       {((abs(avg_sl) - abs(baseline_avg_loss)) / abs(baseline_avg_loss) * 100):.1f}%")
print(f"  Loss gap reduction: {(current_gap - baseline_gap):.2f}")

print(f"\n✅ 1.25:1 ratio fix is WORKING" if abs(avg_sl) < abs(baseline_avg_loss) else f"\n⚠️  Monitor - losses still higher than baseline")
print()
