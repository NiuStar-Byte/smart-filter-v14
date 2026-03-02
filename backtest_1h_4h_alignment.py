#!/usr/bin/env python3
"""
backtest_1h_4h_alignment.py

Test if adding 1h+4h alignment confirmation to 1h signals improves results.

This is an EXPLORATORY test to see if ultra-premium 1h signals (with 4h confirmation)
would be worth the filtering cost.

Author: Nox
Date: 2026-03-03
"""

import json
import os
from collections import defaultdict

def load_sent_signals():
    """Load SENT_SIGNALS.jsonl"""
    signals = []
    filepath = 'SENT_SIGNALS.jsonl'
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: {filepath} not found")
        return []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    signals.append(json.loads(line))
        print(f"✅ Loaded {len(signals)} signals from SENT_SIGNALS.jsonl")
        return signals
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_1h_only_scenario(signals):
    """
    Scenario A: Current Phase 4A (30min+1h filter applies to ALL signals)
    No special handling for 1h signals
    """
    # Just count 1h signals
    signals_1h = [s for s in signals if s.get('timeframe') == '1h']
    closed_1h = [s for s in signals_1h if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']]
    winning_1h = [s for s in closed_1h if s.get('status') == 'TP_HIT']
    
    wr_1h = (len(winning_1h) / len(closed_1h) * 100) if closed_1h else 0
    total_pnl = sum(s.get('pnl_usd', 0) for s in signals_1h if s.get('pnl_usd'))
    
    return {
        'scenario': 'Scenario A: 1h signals (current, no 4h check)',
        'total_signals': len(signals_1h),
        'closed_trades': len(closed_1h),
        'winning_trades': len(winning_1h),
        'wr': wr_1h,
        'total_pnl': total_pnl,
        'avg_pnl_per_trade': (total_pnl / len(closed_1h)) if closed_1h else 0
    }

def detect_trend(candles):
    """Simple trend: close > MA20 = LONG, else SHORT"""
    if not candles or len(candles) < 20:
        return 'NONE'
    
    closes = [c.get('close', 0) for c in candles if 'close' in c]
    if not closes:
        return 'NONE'
    
    ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else sum(closes) / len(closes)
    current_close = closes[-1]
    
    return 'LONG' if current_close > ma20 else 'SHORT'

def load_synthetic_candles():
    """Load candles from candle_cache directory"""
    cache_dir = 'candle_cache'
    candle_data = {}
    
    if not os.path.exists(cache_dir):
        print(f"⚠️  {cache_dir} not found")
        return candle_data
    
    try:
        for filename in os.listdir(cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(cache_dir, filename)
                with open(filepath, 'r') as f:
                    candle_data[filename] = json.load(f)
        
        print(f"✅ Loaded candle data for {len(candle_data)} symbol/TF pairs")
        return candle_data
    except Exception as e:
        print(f"⚠️  Error loading candles: {e}")
        return candle_data

def test_1h_with_4h_confirmation(signals, candle_data):
    """
    Scenario B: Ultra-premium 1h signals
    1h signals only sent if 1h trend = 4h trend (synthetic 4h proxy)
    
    Since we don't have real 4h data, we simulate it by checking
    if 1h trend is stable (use 1h candles from 2-3 periods ago vs now)
    """
    signals_1h = [s for s in signals if s.get('timeframe') == '1h']
    
    # Filter: try to find which 1h signals would have 4h confirmation
    # Since no real 4h data, estimate by checking 1h trend consistency
    confirmed_signals = []
    
    for sig in signals_1h:
        symbol = sig.get('symbol')
        signal_type = sig.get('signal_type')
        
        # Get 1h candles
        key_1h = f"{symbol}_1h.json"
        if key_1h not in candle_data or not candle_data[key_1h]:
            confirmed_signals.append(sig)  # No data, assume confirmed
            continue
        
        # Check if trend is consistent (proxy for 4h alignment)
        candles_1h = candle_data[key_1h]
        if len(candles_1h) < 5:
            confirmed_signals.append(sig)
            continue
        
        # Estimate "4h trend" as average of last 4 hours of 1h candles
        trend_1h_now = detect_trend(candles_1h[-1:])
        trend_1h_avg = detect_trend(candles_1h[-4:])  # Last 4 1h candles = pseudo 4h
        
        # Only confirm if both agree
        if trend_1h_now == trend_1h_avg == signal_type:
            confirmed_signals.append(sig)
    
    # Calculate metrics
    closed_confirmed = [s for s in confirmed_signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT']]
    winning_confirmed = [s for s in closed_confirmed if s.get('status') == 'TP_HIT']
    
    wr_confirmed = (len(winning_confirmed) / len(closed_confirmed) * 100) if closed_confirmed else 0
    total_pnl = sum(s.get('pnl_usd', 0) for s in confirmed_signals if s.get('pnl_usd'))
    
    return {
        'scenario': 'Scenario B: 1h signals with 1h+4h (pseudo) confirmation',
        'total_signals': len(signals_1h),
        'confirmed_signals': len(confirmed_signals),
        'confirmation_rate': (len(confirmed_signals) / len(signals_1h) * 100) if signals_1h else 0,
        'closed_trades': len(closed_confirmed),
        'winning_trades': len(winning_confirmed),
        'wr': wr_confirmed,
        'total_pnl': total_pnl,
        'avg_pnl_per_trade': (total_pnl / len(closed_confirmed)) if closed_confirmed else 0
    }

def run_comparison():
    """Run both scenarios and compare"""
    
    print("=" * 80)
    print("1H SIGNAL CONFIRMATION TEST: Should 1h signals check 4h alignment?")
    print("=" * 80)
    print()
    
    # Load data
    signals = load_sent_signals()
    if not signals:
        print("❌ No signals loaded. Aborting.")
        return
    
    print(f"📊 Testing {len(signals)} total signals\n")
    
    candle_data = load_synthetic_candles()
    print()
    
    # Run scenarios
    scenario_a = test_1h_only_scenario(signals)
    scenario_b = test_1h_with_4h_confirmation(signals, candle_data)
    
    print()
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    print(f"📊 {scenario_a['scenario']}")
    print(f"   Total 1h signals: {scenario_a['total_signals']}")
    print(f"   Closed trades: {scenario_a['closed_trades']}")
    print(f"   WR: {scenario_a['wr']:.1f}%")
    print(f"   P&L: ${scenario_a['total_pnl']:.2f}")
    print(f"   Avg P&L/trade: ${scenario_a['avg_pnl_per_trade']:.2f}")
    print()
    
    print(f"📊 {scenario_b['scenario']}")
    print(f"   Total 1h signals: {scenario_b['total_signals']}")
    print(f"   Confirmed signals: {scenario_b['confirmed_signals']} ({scenario_b['confirmation_rate']:.1f}%)")
    print(f"   Closed trades: {scenario_b['closed_trades']}")
    print(f"   WR: {scenario_b['wr']:.1f}%")
    print(f"   P&L: ${scenario_b['total_pnl']:.2f}")
    print(f"   Avg P&L/trade: ${scenario_b['avg_pnl_per_trade']:.2f}")
    print()
    
    # Delta
    wr_delta = scenario_b['wr'] - scenario_a['wr']
    pnl_delta = scenario_b['total_pnl'] - scenario_a['total_pnl']
    signal_reduction = (scenario_a['total_signals'] - scenario_b['confirmed_signals']) / scenario_a['total_signals'] * 100 if scenario_a['total_signals'] > 0 else 0
    
    print("=" * 80)
    print("DELTA (Scenario B vs A)")
    print("=" * 80)
    print()
    print(f"WR Delta: {wr_delta:+.1f}% (now {scenario_b['wr']:.1f}%)")
    print(f"P&L Delta: ${pnl_delta:+.2f}")
    print(f"Signals Filtered: {signal_reduction:.1f}%")
    print()
    
    # Recommendation
    print("=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)
    print()
    
    if scenario_b['wr'] > scenario_a['wr'] and scenario_b['confirmed_signals'] >= 5:
        print(f"✅ ADD 1h+4h CONFIRMATION")
        print(f"   WR improves {wr_delta:+.1f}% with acceptable signal reduction ({signal_reduction:.1f}%)")
        print(f"   Filters out {scenario_a['total_signals'] - scenario_b['confirmed_signals']} weak 1h signals")
        print(f"   Creates ultra-premium 1h signal tier")
    elif scenario_b['wr'] == scenario_a['wr']:
        print(f"⚠️  NEUTRAL")
        print(f"   WR unchanged, but filters {signal_reduction:.1f}% of signals")
        print(f"   Depends on risk preference:")
        print(f"   - Higher quality but fewer signals")
        print(f"   - May help during choppy markets")
    elif scenario_b['confirmed_signals'] < 5:
        print(f"❌ SKIP 1h+4h CONFIRMATION")
        print(f"   Too few confirmed signals ({scenario_b['confirmed_signals']})")
        print(f"   Filters {signal_reduction:.1f}% of signals - too aggressive")
    else:
        print(f"❌ SKIP 1h+4h CONFIRMATION")
        print(f"   WR declines {wr_delta:.1f}% - not worth the filtering cost")
    
    print()
    print("=" * 80)
    print("📝 NEXT STEPS")
    print("=" * 80)
    print()
    print("This test uses synthetic candle data (not real 4h data)")
    print("For production, we'd need:")
    print("  1. Real 4h candles from KuCoin API")
    print("  2. Real 1d candles from KuCoin API")
    print("  3. Proper trend detection on larger TFs")
    print()
    print("If recommendation is YES, we can implement as Phase 4A-Extended")
    print()

if __name__ == '__main__':
    run_comparison()
