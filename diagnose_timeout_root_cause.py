#!/usr/bin/env python3
"""
ROOT CAUSE DIAGNOSTIC: Why do TIMEOUT signals deteriorate?

Analyzes TIMEOUT signals grouped by:
- Route (signal type: TREND CONTINUATION, REVERSAL, etc.)
- Regime (BULL, BEAR, RANGE)
- TimeFrame (15min, 30min, 1h)
- Confidence (HIGH ≥76%, MID 51-75%, LOW ≤50%)

Identifies worst-performing combinations that are killing P&L.
"""

import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os

workspace = "/Users/geniustarigan/.openclaw/workspace"
master_file = os.path.join(workspace, "SIGNALS_MASTER.jsonl")

def load_timeout_signals():
    """Load TIMEOUT signals (exclude STALE_TIMEOUT)"""
    timeout_signals = []
    
    if not os.path.exists(master_file):
        print(f"[ERROR] {master_file} not found")
        return []
    
    try:
        with open(master_file, 'r') as f:
            for line in f:
                if line.strip():
                    sig = json.loads(line)
                    
                    # Only TIMEOUT (not STALE_TIMEOUT)
                    if sig.get('status') != 'TIMEOUT':
                        continue
                    
                    # Skip stale (data quality issues)
                    if sig.get('data_quality_flag') and 'STALE_TIMEOUT' in sig.get('data_quality_flag'):
                        continue
                    
                    timeout_signals.append(sig)
        
        print(f"[INFO] Loaded {len(timeout_signals)} TIMEOUT signals")
        return timeout_signals
    
    except Exception as e:
        print(f"[ERROR] Loading signals: {e}")
        return []

def get_confidence_level(confidence):
    """Bin confidence into HIGH/MID/LOW"""
    conf_val = float(confidence or 0)
    if conf_val >= 76:
        return "HIGH"
    elif conf_val >= 51:
        return "MID"
    else:
        return "LOW"

def analyze_by_dimension(timeout_signals, dimension):
    """Analyze TIMEOUT outcomes by dimension"""
    stats = defaultdict(lambda: {
        'count': 0,
        'wins': 0,
        'losses': 0,
        'pnl': 0.0,
        'signals': []
    })
    
    for sig in timeout_signals:
        if dimension == 'route':
            key = sig.get('route', 'UNKNOWN')
        elif dimension == 'regime':
            key = sig.get('regime', 'UNKNOWN')
        elif dimension == 'timeframe':
            key = sig.get('timeframe', 'UNKNOWN')
        elif dimension == 'confidence':
            key = get_confidence_level(sig.get('confidence'))
        else:
            continue
        
        stats[key]['count'] += 1
        
        pnl = sig.get('pnl_usd', 0)
        if pnl > 0:
            stats[key]['wins'] += 1
        elif pnl < 0:
            stats[key]['losses'] += 1
        
        stats[key]['pnl'] += float(pnl)
        stats[key]['signals'].append(sig)
    
    return stats

def analyze_by_2d(timeout_signals, dim1, dim2):
    """Analyze TIMEOUT by 2D combination (e.g., ROUTE × REGIME)"""
    stats = defaultdict(lambda: {
        'count': 0,
        'wins': 0,
        'losses': 0,
        'pnl': 0.0,
        'signals': []
    })
    
    for sig in timeout_signals:
        # Get first dimension
        if dim1 == 'route':
            val1 = sig.get('route', 'UNKNOWN')
        elif dim1 == 'regime':
            val1 = sig.get('regime', 'UNKNOWN')
        elif dim1 == 'timeframe':
            val1 = sig.get('timeframe', 'UNKNOWN')
        elif dim1 == 'confidence':
            val1 = get_confidence_level(sig.get('confidence'))
        else:
            continue
        
        # Get second dimension
        if dim2 == 'route':
            val2 = sig.get('route', 'UNKNOWN')
        elif dim2 == 'regime':
            val2 = sig.get('regime', 'UNKNOWN')
        elif dim2 == 'timeframe':
            val2 = sig.get('timeframe', 'UNKNOWN')
        elif dim2 == 'confidence':
            val2 = get_confidence_level(sig.get('confidence'))
        else:
            continue
        
        key = (val1, val2)
        
        stats[key]['count'] += 1
        
        pnl = sig.get('pnl_usd', 0)
        if pnl > 0:
            stats[key]['wins'] += 1
        elif pnl < 0:
            stats[key]['losses'] += 1
        
        stats[key]['pnl'] += float(pnl)
        stats[key]['signals'].append(sig)
    
    return stats

def print_1d_analysis(stats, dimension):
    """Print 1D analysis (ranked by total loss)"""
    print(f"\n{'='*160}")
    print(f"🔍 TIMEOUT ANALYSIS BY {dimension.upper()}")
    print(f"{'='*160}")
    print(f"{'Key':<25} | {'Count':<6} | {'Wins':<6} | {'Loss':<6} | {'WR%':<8} | {'Total P&L':<12} | {'Avg P&L':<12} | {'Alert':<20}")
    print(f"{'-'*160}")
    
    # Sort by total loss (most negative)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['pnl'])
    
    for key, stat in sorted_stats:
        if stat['count'] == 0:
            continue
        
        total_closed = stat['wins'] + stat['losses']
        wr = (stat['wins'] / total_closed * 100) if total_closed > 0 else 0
        avg_pnl = stat['pnl'] / stat['count']
        
        # Alert logic
        alert = ""
        if stat['pnl'] < -500:
            alert = "🔴 CRITICAL LOSS"
        elif stat['pnl'] < -200:
            alert = "🟡 HIGH LOSS"
        elif wr < 20:
            alert = "⚠️  LOW WR"
        elif avg_pnl < -15:
            alert = "🟠 BAD AVG"
        else:
            alert = "✓ OK"
        
        print(f"{str(key):<25} | {stat['count']:<6} | {stat['wins']:<6} | {stat['losses']:<6} | {wr:>6.1f}% | ${stat['pnl']:>+10.2f} | ${avg_pnl:>+10.2f} | {alert:<20}")

def print_2d_analysis(stats, dim1, dim2):
    """Print 2D analysis (ranked by total loss)"""
    print(f"\n{'='*180}")
    print(f"🔍 TIMEOUT ANALYSIS BY {dim1.upper()} × {dim2.upper()}")
    print(f"{'='*180}")
    print(f"{dim1.upper():<15} | {dim2.upper():<15} | {'Count':<6} | {'Wins':<6} | {'Loss':<6} | {'WR%':<8} | {'Total P&L':<12} | {'Avg P&L':<12} | {'Alert':<20}")
    print(f"{'-'*180}")
    
    # Sort by total loss
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['pnl'])
    
    for (key1, key2), stat in sorted_stats:
        if stat['count'] < 3:  # Skip combos with <3 signals
            continue
        
        total_closed = stat['wins'] + stat['losses']
        wr = (stat['wins'] / total_closed * 100) if total_closed > 0 else 0
        avg_pnl = stat['pnl'] / stat['count']
        
        # Alert logic
        alert = ""
        if stat['pnl'] < -300:
            alert = "🔴 CRITICAL"
        elif stat['pnl'] < -100:
            alert = "🟡 HIGH LOSS"
        elif wr < 20:
            alert = "⚠️  LOW WR"
        elif avg_pnl < -15:
            alert = "🟠 BAD AVG"
        else:
            alert = "✓ OK"
        
        print(f"{str(key1):<15} | {str(key2):<15} | {stat['count']:<6} | {stat['wins']:<6} | {stat['losses']:<6} | {wr:>6.1f}% | ${stat['pnl']:>+10.2f} | ${avg_pnl:>+10.2f} | {alert:<20}")

def main():
    print("\n" + "="*160)
    print("🔍 ROOT CAUSE DIAGNOSTIC: TIMEOUT SIGNAL DETERIORATION")
    print("="*160)
    
    timeout_signals = load_timeout_signals()
    if not timeout_signals:
        print("No TIMEOUT signals found")
        return
    
    # 1D ANALYSES
    print("\n" + "█"*160)
    print("1️⃣  ONE-DIMENSIONAL ANALYSIS")
    print("█"*160)
    
    # By Route
    route_stats = analyze_by_dimension(timeout_signals, 'route')
    print_1d_analysis(route_stats, 'route')
    
    # By Regime
    regime_stats = analyze_by_dimension(timeout_signals, 'regime')
    print_1d_analysis(regime_stats, 'regime')
    
    # By TimeFrame
    tf_stats = analyze_by_dimension(timeout_signals, 'timeframe')
    print_1d_analysis(tf_stats, 'timeframe')
    
    # By Confidence
    conf_stats = analyze_by_dimension(timeout_signals, 'confidence')
    print_1d_analysis(conf_stats, 'confidence')
    
    # 2D ANALYSES
    print("\n" + "█"*180)
    print("2️⃣  TWO-DIMENSIONAL ANALYSIS (Worst Combos)")
    print("█"*180)
    
    # Route × Regime
    route_regime = analyze_by_2d(timeout_signals, 'route', 'regime')
    print_2d_analysis(route_regime, 'route', 'regime')
    
    # Route × TimeFrame
    route_tf = analyze_by_2d(timeout_signals, 'route', 'timeframe')
    print_2d_analysis(route_tf, 'route', 'timeframe')
    
    # Route × Confidence
    route_conf = analyze_by_2d(timeout_signals, 'route', 'confidence')
    print_2d_analysis(route_conf, 'route', 'confidence')
    
    # Regime × TimeFrame
    regime_tf = analyze_by_2d(timeout_signals, 'regime', 'timeframe')
    print_2d_analysis(regime_tf, 'regime', 'timeframe')
    
    # Regime × Confidence
    regime_conf = analyze_by_2d(timeout_signals, 'regime', 'confidence')
    print_2d_analysis(regime_conf, 'regime', 'confidence')
    
    # SUMMARY & RECOMMENDATIONS
    print("\n" + "="*160)
    print("💡 SUMMARY & RECOMMENDATIONS")
    print("="*160)
    
    # Find worst combos
    print("\n🔴 WORST COMBOS (Total P&L < -300):")
    worst_combos = []
    
    for (key1, key2), stat in sorted(route_regime.items(), key=lambda x: x[1]['pnl']):
        if stat['pnl'] < -300 and stat['count'] >= 3:
            worst_combos.append((f"ROUTE={key1} × REGIME={key2}", stat))
    
    for (key1, key2), stat in sorted(route_tf.items(), key=lambda x: x[1]['pnl']):
        if stat['pnl'] < -300 and stat['count'] >= 3:
            worst_combos.append((f"ROUTE={key1} × TF={key2}", stat))
    
    worst_combos = sorted(worst_combos, key=lambda x: x[1]['pnl'])[:5]
    
    for combo_name, stat in worst_combos:
        total_closed = stat['wins'] + stat['losses']
        wr = (stat['wins'] / total_closed * 100) if total_closed > 0 else 0
        print(f"  ❌ {combo_name}: {stat['count']} signals, WR={wr:.1f}%, P&L=${stat['pnl']:+.2f}")
    
    # Best combos
    print("\n✅ BEST COMBOS (Total P&L > +200):")
    best_combos = []
    
    for (key1, key2), stat in sorted(route_regime.items(), key=lambda x: -x[1]['pnl']):
        if stat['pnl'] > 200 and stat['count'] >= 3:
            best_combos.append((f"ROUTE={key1} × REGIME={key2}", stat))
    
    for (key1, key2), stat in sorted(route_tf.items(), key=lambda x: -x[1]['pnl']):
        if stat['pnl'] > 200 and stat['count'] >= 3:
            best_combos.append((f"ROUTE={key1} × TF={key2}", stat))
    
    best_combos = sorted(best_combos, key=lambda x: -x[1]['pnl'])[:5]
    
    for combo_name, stat in best_combos:
        total_closed = stat['wins'] + stat['losses']
        wr = (stat['wins'] / total_closed * 100) if total_closed > 0 else 0
        print(f"  ✅ {combo_name}: {stat['count']} signals, WR={wr:.1f}%, P&L=${stat['pnl']:+.2f}")
    
    # Actionable recommendations
    print("\n" + "="*160)
    print("🎯 ACTIONABLE RECOMMENDATIONS")
    print("="*160)
    
    print("\n1. DISABLE (Block from TIMEOUT):")
    print("   Routes/Regimes that LOSE on TIMEOUT should not fire TIMEOUT trades")
    print("   Or: Increase TP target / Decrease SL target (tighter risk)")
    
    print("\n2. IMPROVE (Signal Filters):")
    print("   Which filters are accepting bad signals?")
    print("   - Trend detection? (TREND CONTINUATION failing?)")
    print("   - Regime detection? (BULL/BEAR mismatch?)")
    print("   - Entry timing? (Too late in candle?)")
    
    print("\n3. VALIDATE (Confidence Levels):")
    if conf_stats['LOW']['pnl'] < conf_stats['HIGH']['pnl']:
        print("   ⚠️  LOW confidence signals have worse P&L on TIMEOUT")
        print("   → Consider: Don't fire LOW confidence signals at all")
    
    print("\n" + "="*160)

if __name__ == '__main__':
    main()
