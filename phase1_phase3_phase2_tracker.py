#!/usr/bin/env python3
"""
PHASE 1 & PHASE 3 & PHASE 2 UNIFIED TRACKER
Real-time monitoring of Phase 1 (route veto), Phase 3 (audit), and Phase 2 (TF4h) implementation
Updated: 2026-03-24 21:28 GMT+7

Metrics tracked:
- FOUNDATION baseline (locked Feb 27 - Mar 19) - from PEC_ENHANCED_REPORT.txt
- NEW signals (Phase 1+2 live: Mar 24 18:52+ GMT+7) - all TFs with Phase 1 & 2 logic
- Route veto effectiveness (NONE/AMBIGUOUS blocking)
- Timeframe breakdown (15min, 30min, 1h, 4h)
- Win rate trending
- Direction/Regime breakdown
"""

import json
import os
from collections import defaultdict
from datetime import datetime

# Configuration
SIGNALS_FILE = '/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl'
FOUNDATION_BOUNDARY = '2026-03-19T18:03:17'
PHASE1_DEPLOY_UTC = '2026-03-24T17:00:00'  # Fresh tracking start (2026-03-25 00:00 GMT+7)

# FOUNDATION BASELINE - from PEC_ENHANCED_REPORT.txt (locked, immutable)
FOUNDATION_LOCKED = {
    'total': 2224,
    'closed': 1339,
    'tp': 348,
    'sl': 746,
    'timeout': 245,
    'open': 22,
    'wr': 32.7,  # From PEC reporter
    'long_wr': 28.6,
    'short_wr': 46.4,
    'pnl': -4427.61
}

def load_signals():
    """Load all signals from MASTER file"""
    signals = []
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, 'r') as f:
            for line in f:
                try:
                    signal = json.loads(line.strip())
                    # Normalize field names: convert 'direction' → 'signal_type' if needed
                    # (pec_enhanced_reporter does this - mirror for consistency)
                    if 'direction' in signal and 'signal_type' not in signal:
                        signal['signal_type'] = signal['direction']
                    signals.append(signal)
                except:
                    pass
    return signals

def analyze_signals(signals):
    """Analyze signals by period and dimension"""
    
    # Separate: Foundation vs NEW (after Phase 1 deployment)
    foundation = [s for s in signals if s.get('fired_time_utc', '') <= FOUNDATION_BOUNDARY]
    new_all = [s for s in signals if s.get('fired_time_utc', '') > FOUNDATION_BOUNDARY]
    
    # NEW signals split by deployment phases
    new_phase1_2 = [s for s in new_all if s.get('fired_time_utc', '') >= PHASE1_DEPLOY_UTC]
    
    def calc_stats(sig_list):
        """Calculate WR and counts"""
        if not sig_list:
            return {'total': 0, 'closed': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'open': 0, 'wr': 0, 'long': [], 'short': []}
        
        total = len(sig_list)
        tp = sum(1 for s in sig_list if s.get('status') == 'TP_HIT')
        sl = sum(1 for s in sig_list if s.get('status') == 'SL_HIT')
        timeout = sum(1 for s in sig_list if s.get('status') == 'TIMEOUT')
        open_count = sum(1 for s in sig_list if s.get('status') == 'OPEN')
        closed = tp + sl + timeout
        wr = 100 * tp / closed if closed > 0 else 0
        
        # Read from signal_type (actual field in data), not direction
        long_sigs = [s for s in sig_list if s.get('signal_type') == 'LONG']
        short_sigs = [s for s in sig_list if s.get('signal_type') == 'SHORT']
        
        return {
            'total': total,
            'closed': closed,
            'tp': tp,
            'sl': sl,
            'timeout': timeout,
            'open': open_count,
            'wr': wr,
            'long': long_sigs,
            'short': short_sigs
        }
    
    # Calculate by timeframe
    timeframes = defaultdict(list)
    for s in new_phase1_2:
        tf = s.get('timeframe', 'UNKNOWN')
        timeframes[tf].append(s)
    
    # Calculate by route (normalize underscore/space inconsistency)
    routes = defaultdict(list)
    for s in new_phase1_2:
        route = s.get('route', 'UNKNOWN')
        # Normalize: TREND_CONTINUATION → TREND CONTINUATION (standardize to space format)
        route = route.replace('_', ' ') if route else 'UNKNOWN'
        routes[route].append(s)
    
    # Calculate by direction (use signal_type, not direction field)
    directions = defaultdict(list)
    for s in new_phase1_2:
        direction = s.get('signal_type', 'UNKNOWN')  # READ FROM signal_type, not direction
        directions[direction].append(s)
    
    # Calculate by regime
    regimes = defaultdict(list)
    for s in new_phase1_2:
        regime = s.get('regime', 'UNKNOWN')
        regimes[regime].append(s)
    
    return {
        'foundation': FOUNDATION_LOCKED,
        'new_all': calc_stats(new_all),
        'new_phase1_2': calc_stats(new_phase1_2),
        'timeframes': {tf: calc_stats(sigs) for tf, sigs in timeframes.items()},
        'routes': {r: calc_stats(sigs) for r, sigs in routes.items()},
        'directions': {d: calc_stats(sigs) for d, sigs in directions.items()},
        'regimes': {reg: calc_stats(sigs) for reg, sigs in regimes.items()},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'deploy_time': '2026-03-24 18:52 GMT+7'
    }

def get_dynamic_end_time():
    """Get current time in GMT+7 format"""
    now = datetime.now()
    return now.strftime('%Y-%m-%d %H:%M GMT+7')

def print_tracker(stats):
    """Print formatted tracker output"""
    
    print("\n" + "="*80)
    print("PHASE 1 & PHASE 3 & PHASE 2 UNIFIED TRACKER")
    print("="*80)
    print(f"Last Updated: {stats['timestamp']} (local)")
    print(f"Phase 1 & 2 Deployed: {stats['deploy_time']} | Now: {get_dynamic_end_time()}")
    print()
    
    # SECTION 1: FOUNDATION BASELINE (from PEC reporter)
    print("📊 SECTION 1: FOUNDATION BASELINE (LOCKED - Feb 27 to Mar 19)")
    print("-" * 80)
    f = stats['foundation']
    print(f"  Total Signals: {f['total']} | Closed: {f['closed']} | WR: {f['wr']:.1f}%")
    print(f"  TP: {f['tp']} | SL: {f['sl']} | Timeout: {f['timeout']} | Open: {f['open']}")
    print(f"  LONG WR: {f['long_wr']:.1f}% | SHORT WR: {f['short_wr']:.1f}%")
    print(f"  P&L: ${f['pnl']:,.2f}")
    print(f"  🔒 (This baseline is IMMUTABLE - NEW signals calculated as TOTAL - FOUNDATION)")
    print()
    
    # SECTION 2: NEW SIGNALS (Phase 1 & 2 combined - dynamic time range)
    print(f"✅ SECTION 2: NEW SIGNALS (Phase 1 & 2 Live - {stats['deploy_time']} to {get_dynamic_end_time()})")
    print("-" * 80)
    n = stats['new_phase1_2']
    print(f"  Total Signals: {n['total']} | Closed: {n['closed']} | WR: {n['wr']:.2f}%")
    print(f"  TP: {n['tp']} | SL: {n['sl']} | Timeout: {n['timeout']} | Open: {n['open']}")
    
    # Calculate LONG/SHORT WR
    long_stats = calc_stats(n['long']) if n['long'] else {'closed': 0, 'tp': 0, 'wr': 0}
    short_stats = calc_stats(n['short']) if n['short'] else {'closed': 0, 'tp': 0, 'wr': 0}
    
    print(f"  LONG: {len(n['long'])} signals | WR: {long_stats['wr']:.2f}%")
    print(f"  SHORT: {len(n['short'])} signals | WR: {short_stats['wr']:.2f}%")
    
    delta_wr = n['wr'] - f['wr']
    delta_str = f"({delta_wr:+.2f}pp vs Foundation)" if delta_wr != 0 else ""
    print(f"  {delta_str}")
    print()
    
    # SECTION 3: TIMEFRAME BREAKDOWN (15min, 30min, 1h, 4h)
    print("📈 SECTION 3: TIMEFRAME BREAKDOWN (All TFs with Phase 1 & 2 logic)")
    print("-" * 80)
    print("  Timeframe Comparison:")
    print()
    
    tf_order = ['15min', '30min', '1h', '2h', '4h']
    total_tf_check = 0
    
    for tf_name in tf_order:
        if tf_name in stats['timeframes']:
            tf = stats['timeframes'][tf_name]
            status = "🆕" if tf_name in ['2h', '4h'] else "✅"
            if tf['total'] > 0:
                print(f"  {status} {tf_name:10} | Total: {tf['total']:4} | Closed: {tf['closed']:4} | TP: {tf['tp']:3} | SL: {tf['sl']:3} | WR: {tf['wr']:6.2f}%")
                total_tf_check += tf['total']
            else:
                print(f"  {status} {tf_name:10} | Total:    0 | Closed:    0 | TP:   0 | SL:   0 | WR:   0.00%")
        else:
            status = "🆕" if tf_name in ['2h', '4h'] else "✅"
            print(f"  {status} {tf_name:10} | Total:    0 | Closed:    0 | TP:   0 | SL:   0 | WR:   0.00%")
    
    # Verify total matches
    match_status = "✅ MATCH" if total_tf_check == n['total'] else f"❌ MISMATCH ({total_tf_check} != {n['total']})"
    print(f"  {match_status} (TF total must equal Section 2 total)")
    print()
    
    # SECTION 4: ROUTE VETO EFFECTIVENESS
    print("🚫 SECTION 4: ROUTE VETO EFFECTIVENESS (Phase 1)")
    print("-" * 80)
    print("  Target: NONE/AMBIGUOUS signals = 0 (all blocked by veto)")
    print()
    
    # Count veto violations
    none_count = stats['routes'].get('NONE', {}).get('total', 0)
    ambiguous_count = stats['routes'].get('AMBIGUOUS', {}).get('total', 0)
    veto_failed = none_count + ambiguous_count
    
    if veto_failed > 0:
        print(f"  🚨 VETO FAILURE: {veto_failed} signals leaked through (should be 0)")
        print()
    
    # Show all routes
    route_order = ['TREND CONTINUATION', 'REVERSAL', 'NONE', 'AMBIGUOUS', 'UNKNOWN']
    total_routes_check = 0
    
    for route_name in route_order:
        if route_name in stats['routes']:
            r = stats['routes'][route_name]
            if r['total'] > 0:
                status = "✅ ALLOWED" if route_name not in ['NONE', 'AMBIGUOUS'] else "❌ SHOULD BE 0"
                print(f"  {route_name:20} | Total: {r['total']:4} | Closed: {r['closed']:4} | TP: {r['tp']:3} | WR: {r['wr']:6.2f}% {status}")
                total_routes_check += r['total']
            else:
                if route_name in ['NONE', 'AMBIGUOUS']:
                    print(f"  {route_name:20} | Total:    0 | Closed:    0 | TP:   0 | WR:   0.00% ✅ BLOCKED")
    
    # Verify route total matches
    match_status = "✅ MATCH" if total_routes_check == n['total'] else f"❌ MISMATCH ({total_routes_check} != {n['total']})"
    print(f"  {match_status} (Route total must equal Section 2 total)")
    print()
    
    # SECTION 5: DECISION CRITERIA
    print("🎯 SECTION 5: PHASE 2 LOCK-IN DECISION CHECKLIST")
    print("-" * 80)
    
    checks = [
        ("NEW signals >100 total", n['total'] > 100),
        ("NEW closed signals >50", n['closed'] > 50),
        ("TF2h fired at least 10 signals", stats['timeframes'].get('2h', {}).get('total', 0) >= 10),
        ("TF2h WR stable >35%", stats['timeframes'].get('2h', {}).get('wr', 0) > 35 if stats['timeframes'].get('2h', {}).get('total', 0) > 0 else False),
        ("TF4h fired at least 5 signals", stats['timeframes'].get('4h', {}).get('total', 0) >= 5),
        ("TF4h WR stable >25%", stats['timeframes'].get('4h', {}).get('wr', 0) > 25 if stats['timeframes'].get('4h', {}).get('total', 0) > 0 else False),
        ("NEW WR approaching baseline", n['wr'] >= f['wr'] * 0.95),
        ("Route veto effective (NONE/AMBIGUOUS = 0)", veto_failed == 0),
    ]
    
    passed = 0
    for check_name, result in checks:
        status = "✅" if result else "⏳"
        print(f"  {status} {check_name}")
        if result:
            passed += 1
    
    print()
    print(f"  Status: {passed}/{len(checks)} criteria met")
    if passed == len(checks):
        print("  🟢 READY TO LOCK PHASE 2 - Proceed with confidence")
    elif passed >= 5:
        print("  🟡 ALMOST READY - Wait for TF4h data or veto confirmation")
    elif passed >= 3:
        print("  🟡 IN PROGRESS - Monitor accumulation (24-48h window)")
    else:
        print("  🔴 TOO EARLY - Need more signal accumulation")
    
    print()
    print("="*80)
    print("⏱️  Recommendation: Re-run tracker every 2-4 hours (Phase 2 moves faster)")
    print("📌 Target: 24-48 hours of Phase 2 signals for TF4h validation")
    print("🔄 Lock Phase 2 when: All 6 criteria ✅ + TF4h WR stable >25%")
    print("="*80)
    print()

def calc_stats(sig_list):
    """Helper to calculate stats for a signal list"""
    if not sig_list:
        return {'closed': 0, 'tp': 0, 'wr': 0}
    
    tp = sum(1 for s in sig_list if s.get('status') == 'TP_HIT')
    sl = sum(1 for s in sig_list if s.get('status') == 'SL_HIT')
    timeout = sum(1 for s in sig_list if s.get('status') == 'TIMEOUT')
    closed = tp + sl + timeout
    wr = 100 * tp / closed if closed > 0 else 0
    
    return {'closed': closed, 'tp': tp, 'wr': wr}

if __name__ == '__main__':
    signals = load_signals()
    stats = analyze_signals(signals)
    print_tracker(stats)
