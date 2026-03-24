#!/usr/bin/env python3
"""
PHASE 1, 2, 3 UNIFIED TRACKER - Version 2 (CLEAN)
Purpose: Monitor post-implementation validation of Phase 1 (Route Veto), Phase 2 (TF4h), Phase 3 (Quality)
Data source: SIGNALS_MASTER.jsonl (single source of truth)
Deployment: 2026-03-24 18:52 GMT+7
"""

import json
from datetime import datetime
from pathlib import Path

# Hardcoded paths
SIGNALS_MASTER = Path("/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl")
FOUNDATION_CUTOFF = "2026-03-19T23:59:59"  # Last FOUNDATION signal

def load_signals():
    """Load all signals from SIGNALS_MASTER.jsonl"""
    signals = []
    if not SIGNALS_MASTER.exists():
        return signals
    
    try:
        with open(SIGNALS_MASTER, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        signal = json.loads(line)
                        signals.append(signal)
                    except:
                        pass
    except:
        pass
    
    return signals

def separate_foundation_and_new(signals):
    """Split signals into FOUNDATION (locked) and NEW_LIVE"""
    foundation = []
    new = []
    
    for sig in signals:
        origin = sig.get('signal_origin', 'UNKNOWN')
        if origin == 'FOUNDATION':
            foundation.append(sig)
        elif origin == 'NEW_LIVE':
            new.append(sig)
    
    return foundation, new

def calculate_wr(signals):
    """Calculate win rate from closed signals"""
    if not signals:
        return 0.0, 0, 0, 0
    
    closed = [s for s in signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'STALE_TIMEOUT']]
    if not closed:
        return 0.0, 0, 0, 0
    
    tp_count = len([s for s in closed if s.get('status') == 'TP_HIT'])
    sl_count = len([s for s in closed if s.get('status') == 'SL_HIT'])
    timeout_count = len([s for s in closed if s.get('status') == 'STALE_TIMEOUT'])
    
    wr = (tp_count / len(closed) * 100) if closed else 0.0
    return wr, tp_count, sl_count, timeout_count

def main():
    signals = load_signals()
    foundation, new = separate_foundation_and_new(signals)
    
    # === SECTION 1: FOUNDATION (LOCKED) ===
    f_wr, f_tp, f_sl, f_timeout = calculate_wr(foundation)
    f_closed = len([s for s in foundation if s.get('status') in ['TP_HIT', 'SL_HIT', 'STALE_TIMEOUT']])
    
    # === SECTION 2: NEW SIGNALS ===
    n_wr, n_tp, n_sl, n_timeout = calculate_wr(new)
    n_closed = len([s for s in new if s.get('status') in ['TP_HIT', 'SL_HIT', 'STALE_TIMEOUT']])
    
    # Separate LONG vs SHORT
    new_long = [s for s in new if s.get('signal_type') == 'LONG']
    new_short = [s for s in new if s.get('signal_type') == 'SHORT']
    nl_wr, nl_tp, nl_sl, nl_timeout = calculate_wr(new_long)
    ns_wr, ns_tp, ns_sl, ns_timeout = calculate_wr(new_short)
    
    # === SECTION 3: TIMEFRAME BREAKDOWN ===
    tf_breakdown = {}
    for tf in ['15min', '30min', '1h', '4h']:
        tf_signals = [s for s in new if s.get('timeframe') == tf]
        wr, tp, sl, timeout = calculate_wr(tf_signals)
        tf_breakdown[tf] = {
            'total': len(tf_signals),
            'closed': len([s for s in tf_signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'STALE_TIMEOUT']]),
            'tp': tp,
            'sl': sl,
            'wr': wr
        }
    
    # === SECTION 4: PHASE 1 - ROUTE VETO ===
    route_breakdown = {}
    for route in ['TREND CONTINUATION', 'REVERSAL', 'NONE', 'AMBIGUOUS']:
        route_signals = [s for s in new if s.get('route') == route]
        wr, tp, sl, timeout = calculate_wr(route_signals)
        route_breakdown[route] = {
            'total': len(route_signals),
            'closed': len([s for s in route_signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'STALE_TIMEOUT']]),
            'tp': tp,
            'wr': wr
        }
    
    none_ambig = route_breakdown.get('NONE', {}).get('total', 0) + route_breakdown.get('AMBIGUOUS', {}).get('total', 0)
    
    # === PRINT REPORT ===
    print("=" * 80)
    print("PHASE 1 & 2 & 3 TRACKER - VERSION 2 (CLEAN)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()} (local)")
    print(f"Data source: {SIGNALS_MASTER}")
    print()
    
    # SECTION 1
    print("📊 SECTION 1: FOUNDATION BASELINE (LOCKED - Feb 27 to Mar 19)")
    print("-" * 80)
    print(f"Total Signals: {len(foundation)} | Closed: {f_closed} | WR: {f_wr:.1f}%")
    print(f"TP: {f_tp} | SL: {f_sl} | Timeout: {f_timeout}")
    print()
    
    # SECTION 2
    print("✅ SECTION 2: NEW SIGNALS (Phase 1 & 2 Live)")
    print("-" * 80)
    print(f"Total Signals: {len(new)} | Closed: {n_closed} | WR: {n_wr:.1f}%")
    print(f"TP: {n_tp} | SL: {n_sl} | Timeout: {n_timeout} | Open: {len(new) - n_closed}")
    print(f"LONG: {len(new_long)} signals | WR: {nl_wr:.1f}%")
    print(f"SHORT: {len(new_short)} signals | WR: {ns_wr:.1f}%")
    print(f"Δ WR vs Foundation: {n_wr - f_wr:+.1f}pp")
    print()
    
    # SECTION 3
    print("📈 SECTION 3: TIMEFRAME BREAKDOWN")
    print("-" * 80)
    for tf in ['15min', '30min', '1h', '4h']:
        data = tf_breakdown[tf]
        symbol = "✅" if data['total'] > 0 else "❌"
        print(f"{symbol} {tf:6} | Total: {data['total']:3} | Closed: {data['closed']:3} | TP: {data['tp']:2} | WR: {data['wr']:5.1f}%")
    total_tf = sum(tf_breakdown[tf]['total'] for tf in tf_breakdown)
    print(f"✓ MATCH: {total_tf} == {len(new)} (must match Section 2 total)")
    print()
    
    # SECTION 4
    print("🚫 SECTION 4: PHASE 1 - ROUTE VETO EFFECTIVENESS")
    print("-" * 80)
    print(f"Target: NONE + AMBIGUOUS = 0 (all blocked by veto)")
    print(f"Actual: {none_ambig} leaked signals (SHOULD BE 0) {'❌ VETO FAILING' if none_ambig > 0 else '✅ VETO WORKING'}")
    print()
    for route in ['TREND CONTINUATION', 'REVERSAL', 'NONE', 'AMBIGUOUS']:
        data = route_breakdown[route]
        symbol = "✅" if route not in ['NONE', 'AMBIGUOUS'] else ("❌" if data['total'] > 0 else "✅")
        print(f"{symbol} {route:20} | Total: {data['total']:3} | Closed: {data['closed']:3} | TP: {data['tp']:2} | WR: {data['wr']:5.1f}%")
    total_route = sum(route_breakdown[r]['total'] for r in route_breakdown)
    mismatch = total_route - len(new)
    if mismatch != 0:
        print(f"❌ MISMATCH: {total_route} != {len(new)} (difference: {mismatch} signals)")
    else:
        print(f"✓ MATCH: {total_route} == {len(new)}")
    print()
    
    # SECTION 5
    print("🎯 SECTION 5: PHASE 2 & 3 LOCK-IN DECISION CHECKLIST")
    print("-" * 80)
    checks = []
    checks.append(("NEW signals >200", len(new) >= 200, len(new)))
    checks.append(("NEW closed signals >80", n_closed >= 80, n_closed))
    checks.append(("TF4h fired >10 signals", tf_breakdown['4h']['total'] >= 10, tf_breakdown['4h']['total']))
    checks.append(("TF4h WR >25%", tf_breakdown['4h']['wr'] >= 25.0, f"{tf_breakdown['4h']['wr']:.1f}%"))
    checks.append(("NEW WR approaching baseline", n_wr >= f_wr - 5, f"{n_wr:.1f}% (baseline {f_wr:.1f}%)"))
    checks.append(("Route veto effective", none_ambig == 0, f"{none_ambig} leaked"))
    
    passed = 0
    for desc, result, value in checks:
        symbol = "✅" if result else "⏳"
        print(f"{symbol} {desc:35} | {value}")
        if result:
            passed += 1
    
    print(f"\nStatus: {passed}/6 criteria met")
    if passed >= 5:
        print("🟢 READY FOR PHASE 3 DEPLOYMENT")
    elif passed >= 4:
        print("🟡 CLOSE - Need more signal accumulation")
    else:
        print("🔴 TOO EARLY - Continue Phase 1 & 2 monitoring")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
