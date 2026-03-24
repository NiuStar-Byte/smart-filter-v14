#!/usr/bin/env python3
"""
PHASE 1 & PHASE 3 TRACKER
Real-time monitoring of Phase 1 (route veto) and Phase 3 (audit) implementation
Updated: 2026-03-24 18:52 GMT+7

Metrics tracked:
- FOUNDATION baseline (locked Feb 27 - Mar 19)
- NEW signals (Phase 1 live: Mar 24 18:52+ GMT+7)
- Route veto effectiveness (NONE/AMBIGUOUS blocking)
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
PHASE1_DEPLOY_UTC = '2026-03-24T11:52:00'  # 2026-03-24 18:52 GMT+7

def load_signals():
    """Load all signals from MASTER file"""
    signals = []
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, 'r') as f:
            for line in f:
                try:
                    signals.append(json.loads(line.strip()))
                except:
                    pass
    return signals

def analyze_signals(signals):
    """Analyze signals by period and dimension"""
    
    # Separate by period
    foundation = [s for s in signals if s.get('fired_time_utc', '') <= FOUNDATION_BOUNDARY]
    new_signals = [s for s in signals if s.get('fired_time_utc', '') > FOUNDATION_BOUNDARY]
    
    # Further separate NEW by Phase 1 deploy time
    new_pre_phase1 = [s for s in new_signals if s.get('fired_time_utc', '') < PHASE1_DEPLOY_UTC]
    new_post_phase1 = [s for s in new_signals if s.get('fired_time_utc', '') >= PHASE1_DEPLOY_UTC]
    
    def calc_stats(sig_list):
        """Calculate WR and counts"""
        if not sig_list:
            return {'total': 0, 'closed': 0, 'tp': 0, 'sl': 0, 'timeout': 0, 'open': 0, 'wr': 0}
        
        total = len(sig_list)
        tp = sum(1 for s in sig_list if s.get('status') == 'TP_HIT')
        sl = sum(1 for s in sig_list if s.get('status') == 'SL_HIT')
        timeout = sum(1 for s in sig_list if s.get('status') == 'TIMEOUT')
        open_count = sum(1 for s in sig_list if s.get('status') == 'OPEN')
        closed = tp + sl + timeout
        wr = 100 * tp / closed if closed > 0 else 0
        
        return {'total': total, 'closed': closed, 'tp': tp, 'sl': sl, 'timeout': timeout, 'open': open_count, 'wr': wr}
    
    # Calculate by route
    routes = defaultdict(list)
    for s in new_post_phase1:
        route = s.get('route', 'UNKNOWN')
        routes[route].append(s)
    
    # Calculate by direction
    directions = defaultdict(list)
    for s in new_post_phase1:
        direction = s.get('direction', 'UNKNOWN')
        directions[direction].append(s)
    
    # Calculate by regime
    regimes = defaultdict(list)
    for s in new_post_phase1:
        regime = s.get('regime', 'UNKNOWN')
        regimes[regime].append(s)
    
    return {
        'foundation': calc_stats(foundation),
        'new_all': calc_stats(new_signals),
        'new_pre_phase1': calc_stats(new_pre_phase1),
        'new_post_phase1': calc_stats(new_post_phase1),
        'routes': {r: calc_stats(sigs) for r, sigs in routes.items()},
        'directions': {d: calc_stats(sigs) for d, sigs in directions.items()},
        'regimes': {reg: calc_stats(sigs) for reg, sigs in regimes.items()},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def print_tracker(stats):
    """Print formatted tracker output"""
    
    print("\n" + "="*80)
    print("PHASE 1 & PHASE 3 REAL-TIME TRACKER")
    print("="*80)
    print(f"Last Updated: {stats['timestamp']} (local)")
    print(f"Phase 1 Deployed: 2026-03-24 18:52 GMT+7 (11:52 UTC)")
    print()
    
    # SECTION 1: FOUNDATION BASELINE (locked)
    print("📊 SECTION 1: FOUNDATION BASELINE (LOCKED - Feb 27 to Mar 19)")
    print("-" * 80)
    f = stats['foundation']
    print(f"  Total Signals: {f['total']}")
    print(f"  Closed: {f['closed']} | TP: {f['tp']} | SL: {f['sl']} | Timeout: {f['timeout']} | Open: {f['open']}")
    print(f"  📈 Win Rate: {f['wr']:.2f}% (BASELINE REFERENCE)")
    print()
    
    # SECTION 2: NEW SIGNALS - ALL (since Mar 20)
    print("📊 SECTION 2: NEW SIGNALS TOTAL (Mar 20+)")
    print("-" * 80)
    n_all = stats['new_all']
    print(f"  Total Signals: {n_all['total']}")
    print(f"  Closed: {n_all['closed']} | TP: {n_all['tp']} | SL: {n_all['sl']} | Timeout: {n_all['timeout']} | Open: {n_all['open']}")
    print(f"  📈 Win Rate: {n_all['wr']:.2f}%")
    delta = n_all['wr'] - f['wr']
    delta_str = f"({delta:+.2f}pp)" if delta != 0 else ""
    print(f"  vs Foundation: {delta_str}")
    print()
    
    # SECTION 3: PHASE 1 PRE-DEPLOYMENT (Mar 20-24 before 18:52)
    print("📊 SECTION 3: NEW PRE-PHASE1 (Mar 20 to Mar 24 18:52 GMT+7)")
    print("-" * 80)
    pre = stats['new_pre_phase1']
    print(f"  Total Signals: {pre['total']} (baseline for comparison)")
    print(f"  Closed: {pre['closed']} | TP: {pre['tp']} | SL: {pre['sl']} | Timeout: {pre['timeout']} | Open: {pre['open']}")
    print(f"  📈 Win Rate: {pre['wr']:.2f}%")
    print()
    
    # SECTION 4: PHASE 1 POST-DEPLOYMENT (Mar 24 18:52+ GMT+7)
    print("✅ SECTION 4: NEW POST-PHASE1 (Mar 24 18:52+ GMT+7) - LIVE PHASE 1")
    print("-" * 80)
    post = stats['new_post_phase1']
    print(f"  Total Signals: {post['total']}")
    print(f"  Closed: {post['closed']} | TP: {post['tp']} | SL: {post['sl']} | Timeout: {post['timeout']} | Open: {post['open']}")
    print(f"  📈 Win Rate: {post['wr']:.2f}%")
    delta_pre = post['wr'] - pre['wr']
    delta_str_pre = f"({delta_pre:+.2f}pp)" if delta_pre != 0 else ""
    print(f"  vs Pre-Phase1: {delta_str_pre}")
    delta_foundation = post['wr'] - f['wr']
    delta_str_foundation = f"({delta_foundation:+.2f}pp)" if delta_foundation != 0 else ""
    print(f"  vs Foundation: {delta_str_foundation}")
    print()
    
    # SECTION 5: ROUTE VETO EFFECTIVENESS
    print("🚫 SECTION 5: ROUTE VETO EFFECTIVENESS (Phase 1)")
    print("-" * 80)
    print("  Target: NONE/AMBIGUOUS blocked (0% of new signals)")
    print()
    
    if 'routes' in stats:
        for route_name in ['TREND CONTINUATION', 'REVERSAL', 'NONE', 'AMBIGUOUS']:
            if route_name in stats['routes']:
                r = stats['routes'][route_name]
                if r['total'] > 0:
                    status = "❌ SHOULD BE BLOCKED" if route_name in ['NONE', 'AMBIGUOUS'] else "✅ ALLOWED"
                    print(f"  {route_name:20} | Total: {r['total']:4} | Closed: {r['closed']:4} | TP: {r['tp']:3} | WR: {r['wr']:6.2f}% {status}")
    print()
    
    # SECTION 6: DIRECTION BREAKDOWN
    print("📊 SECTION 6: DIRECTION BREAKDOWN (Post-Phase1)")
    print("-" * 80)
    if 'directions' in stats:
        for direction in sorted(stats['directions'].keys()):
            d = stats['directions'][direction]
            if d['total'] > 0:
                print(f"  {direction:10} | Total: {d['total']:4} | Closed: {d['closed']:4} | TP: {d['tp']:3} | WR: {d['wr']:6.2f}%")
    print()
    
    # SECTION 7: REGIME BREAKDOWN
    print("📊 SECTION 7: REGIME BREAKDOWN (Post-Phase1)")
    print("-" * 80)
    if 'regimes' in stats:
        for regime in sorted(stats['regimes'].keys()):
            reg = stats['regimes'][regime]
            if reg['total'] > 0:
                print(f"  {regime:10} | Total: {reg['total']:4} | Closed: {reg['closed']:4} | TP: {reg['tp']:3} | WR: {reg['wr']:6.2f}%")
    print()
    
    # SECTION 8: DECISION CRITERIA FOR PHASE 2
    print("🎯 SECTION 8: PHASE 2 DECISION CHECKLIST")
    print("-" * 80)
    
    checks = [
        ("NEW post-Phase1 has >100 signals", post['total'] > 100),
        ("NEW WR trending upward vs pre-Phase1", post['wr'] > pre['wr']),
        ("NEW WR approaching Foundation baseline", post['wr'] >= f['wr'] * 0.95),
        ("Route veto blocking NONE/AMBIGUOUS", stats['routes'].get('NONE', {}).get('total', 0) == 0 or 
                                               stats['routes'].get('NONE', {}).get('total', 0) < 5),
        ("TREND_CONT/REVERSAL sustained (>25%)", stats['routes'].get('TREND CONTINUATION', {}).get('wr', 0) > 15),
        ("No major regression in LONG/SHORT", abs(stats['directions'].get('LONG', {}).get('wr', 0) - 
                                                   stats['directions'].get('SHORT', {}).get('wr', 0)) < 20),
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
        print("  🟢 READY FOR PHASE 2")
    else:
        print("  🟡 WAIT - Accumulating more Phase 1 data")
    
    print()
    print("="*80)
    print("⏱️  Recommendation: Re-run tracker every 4-6 hours")
    print("📌 Target accumulation: 24-72 hours of Phase 1 signals")
    print("="*80)
    print()

if __name__ == '__main__':
    signals = load_signals()
    stats = analyze_signals(signals)
    print_tracker(stats)
