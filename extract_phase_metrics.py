#!/usr/bin/env python3
"""
Extract Phase 2 Metrics - A/B Test Comparison

Usage:
  python3 extract_phase_metrics.py > PHASE2_METRICS_DAY_X.txt
  
Compares PHASE 1 (archived baseline) vs PHASE 2 (live data)
and shows improvement metrics.
"""

import json
import pandas as pd
from datetime import datetime
from collections import defaultdict

PHASE1_CUTOFF = datetime(2026, 3, 2, 18, 4, 0)  # Mar 02 18:04 GMT+7
PHASE1_FILE = "SENT_SIGNALS_PHASE1_BASELINE.jsonl"
PHASE2_FILE = "SENT_SIGNALS.jsonl"

def load_signals(filepath):
    """Load signals from JSONL file"""
    signals = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    sig = json.loads(line.strip())
                    if sig:
                        signals.append(sig)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")
        return []
    
    return signals

def extract_by_phase(signals, cutoff=PHASE1_CUTOFF):
    """Split signals into Phase 1 and Phase 2"""
    phase1 = []
    phase2 = []
    
    for sig in signals:
        try:
            # Try both field names (fired_at from new signals, fired_time_utc from old)
            fired_str = sig.get('fired_at') or sig.get('fired_time_utc') or sig.get('sent_time_utc', '')
            if not fired_str:
                continue
            
            # Parse ISO format timestamp
            fired = datetime.fromisoformat(fired_str.replace('Z', '+00:00'))
            
            if fired < cutoff:
                phase1.append(sig)
            else:
                phase2.append(sig)
        except Exception as e:
            continue
    
    return phase1, phase2

def calculate_metrics(signals):
    """Calculate performance metrics from signals"""
    if not signals:
        return None
    
    total = len(signals)
    
    # Filter to closed trades
    closed = [s for s in signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']]
    open_sigs = [s for s in signals if s.get('status') == 'OPEN']
    
    if not closed:
        return {
            "total_signals": total,
            "open_signals": len(open_sigs),
            "closed_trades": 0,
            "win_rate_pct": None,
            "p_and_l": None,
            "sharpe_ratio": None,
        }
    
    # Win rate
    wins = len([s for s in closed if s.get('status') == 'TP_HIT'])
    losses = len([s for s in closed if s.get('status') == 'SL_HIT'])
    timeouts = len([s for s in closed if s.get('status') == 'TIMEOUT'])
    
    wr = (wins + timeouts) / len(closed) if closed else 0
    
    # P&L (sum of pnl_usd or p_and_l field if available)
    pnl = 0
    for s in closed:
        pnl_val = s.get('pnl_usd') or s.get('p_and_l') or 0
        try:
            pnl += float(pnl_val)
        except:
            pass
    
    # Direction breakdown (signal_type contains LONG/SHORT)
    long_signals = [s for s in closed if s.get('signal_type') == 'LONG']
    short_signals = [s for s in closed if s.get('signal_type') == 'SHORT']
    
    long_wr = (len([s for s in long_signals if s.get('status') == 'TP_HIT']) + 
               len([s for s in long_signals if s.get('status') == 'TIMEOUT'])) / len(long_signals) if long_signals else 0
    short_wr = (len([s for s in short_signals if s.get('status') == 'TP_HIT']) + 
                len([s for s in short_signals if s.get('status') == 'TIMEOUT'])) / len(short_signals) if short_signals else 0
    
    metrics = {
        "total_signals": total,
        "open_signals": len(open_sigs),
        "closed_trades": len(closed),
        "wins_tp": wins,
        "losses_sl": losses,
        "timeouts": timeouts,
        "win_rate_pct": wr * 100,
        "long_trades": len(long_signals),
        "long_wr_pct": long_wr * 100,
        "short_trades": len(short_signals),
        "short_wr_pct": short_wr * 100,
        "p_and_l_usd": pnl,
        "profit_factor": (wins + (timeouts * 0.5)) / (losses + (timeouts * 0.5)) if (losses + timeouts) > 0 else 0,
    }
    
    return metrics

def print_comparison():
    """Print Phase 1 vs Phase 2 comparison"""
    
    # Load all signals
    all_signals = load_signals(PHASE2_FILE)
    phase1_signals, phase2_signals = extract_by_phase(all_signals)
    
    phase1_metrics = calculate_metrics(phase1_signals)
    phase2_metrics = calculate_metrics(phase2_signals)
    
    print("=" * 100)
    print("PHASE 2 A/B TEST METRICS")
    print("=" * 100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()
    
    # Phase 1
    print("PHASE 1 (Before Hard Gates) - Feb 27 to Mar 02 18:04 GMT+7")
    print("-" * 100)
    if phase1_metrics:
        print(f"  Total Signals:      {phase1_metrics.get('total_signals', 'N/A')}")
        print(f"  Closed Trades:      {phase1_metrics.get('closed_trades', 'N/A')}")
        print(f"  Win Rate:           {phase1_metrics.get('win_rate_pct', 'N/A'):.1f}%")
        print(f"  LONG WR:            {phase1_metrics.get('long_wr_pct', 'N/A'):.1f}% ({phase1_metrics.get('long_trades', 0)} trades)")
        print(f"  SHORT WR:           {phase1_metrics.get('short_wr_pct', 'N/A'):.1f}% ({phase1_metrics.get('short_trades', 0)} trades)")
        print(f"  P&L:                ${phase1_metrics.get('p_and_l_usd', 'N/A'):.2f}")
        print(f"  Profit Factor:      {phase1_metrics.get('profit_factor', 'N/A'):.2f}")
    else:
        print("  No Phase 1 data available")
    print()
    
    # Phase 2
    print("PHASE 2 (After Hard Gates + Regime Adjustments) - Mar 02 18:04 onwards")
    print("-" * 100)
    if phase2_metrics and phase2_metrics.get('closed_trades', 0) > 0:
        print(f"  Total Signals:      {phase2_metrics.get('total_signals', 'N/A')}")
        print(f"  Open Signals:       {phase2_metrics.get('open_signals', 'N/A')}")
        print(f"  Closed Trades:      {phase2_metrics.get('closed_trades', 'N/A')}")
        print(f"  Win Rate:           {phase2_metrics.get('win_rate_pct', 'N/A'):.1f}%")
        print(f"  LONG WR:            {phase2_metrics.get('long_wr_pct', 'N/A'):.1f}% ({phase2_metrics.get('long_trades', 0)} trades)")
        print(f"  SHORT WR:           {phase2_metrics.get('short_wr_pct', 'N/A'):.1f}% ({phase2_metrics.get('short_trades', 0)} trades)")
        print(f"  P&L:                ${phase2_metrics.get('p_and_l_usd', 'N/A'):.2f}")
        print(f"  Profit Factor:      {phase2_metrics.get('profit_factor', 'N/A'):.2f}")
    else:
        print(f"  Collecting... {phase2_metrics.get('closed_trades', 0) if phase2_metrics else 0} closed trades so far")
    print()
    
    # Comparison
    if phase1_metrics and phase2_metrics and phase2_metrics.get('closed_trades', 0) > 5:
        print("IMPROVEMENT")
        print("-" * 100)
        wr_imp = (phase2_metrics.get('win_rate_pct', 0) - phase1_metrics.get('win_rate_pct', 0))
        long_imp = (phase2_metrics.get('long_wr_pct', 0) - phase1_metrics.get('long_wr_pct', 0))
        short_imp = (phase2_metrics.get('short_wr_pct', 0) - phase1_metrics.get('short_wr_pct', 0))
        pnl_imp = (phase2_metrics.get('p_and_l_usd', 0) - phase1_metrics.get('p_and_l_usd', 0))
        
        status_wr = "✅" if wr_imp > 0 else "❌"
        status_long = "✅" if long_imp > 0 else "⚠️"
        status_short = "✅" if short_imp > 0 else "⚠️"
        status_pnl = "✅" if pnl_imp > 0 else "⚠️"
        
        print(f"  Win Rate:           {status_wr} {wr_imp:+.1f}% ({phase1_metrics.get('win_rate_pct', 0):.1f}% → {phase2_metrics.get('win_rate_pct', 0):.1f}%)")
        print(f"  LONG WR:            {status_long} {long_imp:+.1f}% ({phase1_metrics.get('long_wr_pct', 0):.1f}% → {phase2_metrics.get('long_wr_pct', 0):.1f}%)")
        print(f"  SHORT WR:           {status_short} {short_imp:+.1f}% ({phase1_metrics.get('short_wr_pct', 0):.1f}% → {phase2_metrics.get('short_wr_pct', 0):.1f}%)")
        print(f"  P&L:                {status_pnl} ${pnl_imp:+.2f} ({phase1_metrics.get('p_and_l_usd', 0):.2f} → {phase2_metrics.get('p_and_l_usd', 0):.2f})")
        print()
        
        # Decision
        if wr_imp >= 3.5:
            print("  ✅ POSITIVE RESULT: Proceed to Phase 3")
        elif wr_imp >= 0:
            print("  ⚠️  MINOR IMPROVEMENT: Sufficient data collected, needs review")
        else:
            print("  ❌ NEGATIVE RESULT: Phase 2 made things worse, revert to Phase 1")
    else:
        print("IMPROVEMENT: Collecting data... (need at least 5-10 closed trades for meaningful comparison)")
    
    print("=" * 100)

if __name__ == "__main__":
    print_comparison()
