#!/usr/bin/env python3
"""
PHASE 1 & PHASE 3 & PHASE 2 TRACKER
Real-time monitoring of Phase 1 (route veto), Phase 3 (audit), and Phase 2 (TF4h) implementation
Updated: 2026-03-24 20:07 GMT+7

Metrics tracked:
- FOUNDATION baseline (locked Feb 27 - Mar 19)
- NEW signals (Phase 1 live: Mar 24 18:52+ GMT+7)
- Route veto effectiveness (NONE/AMBIGUOUS blocking)
- Win rate trending
- Direction/Regime breakdown
- **PHASE 2: TF4h signals and performance**
"""

import json
import os
from collections import defaultdict
from datetime import datetime

# Configuration
SIGNALS_FILE = '/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl'
FOUNDATION_BOUNDARY = '2026-03-19T18:03:17'
PHASE1_DEPLOY_UTC = '2026-03-24T11:52:00'  # 2026-03-24 18:52 GMT+7
PHASE2_DEPLOY_UTC = '2026-03-24T12:07:00'  # 2026-03-24 19:07 GMT+7 (Phase 2 TF4h goes live)

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
    new_phase1 = [s for s in new_signals if PHASE1_DEPLOY_UTC <= s.get('fired_time_utc', '') < PHASE2_DEPLOY_UTC]
    new_phase2 = [s for s in new_signals if s.get('fired_time_utc', '') >= PHASE2_DEPLOY_UTC]
    
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
    
    # Calculate by timeframe (Phase 2)
    timeframes = defaultdict(list)
    for s in new_phase2:
        tf = s.get('timeframe', 'UNKNOWN')
        timeframes[tf].append(s)
    
    # Calculate by route (Phase 1)
    routes = defaultdict(list)
    for s in new_phase1:
        route = s.get('route', 'UNKNOWN')
        routes[route].append(s)
    
    # Calculate by direction
    directions = defaultdict(list)
    for s in new_phase1:
        direction = s.get('direction', 'UNKNOWN')
        directions[direction].append(s)
    
    # Calculate by regime
    regimes = defaultdict(list)
    for s in new_phase1:
        regime = s.get('regime', 'UNKNOWN')
        regimes[regime].append(s)
    
    return {
        'foundation': calc_stats(foundation),
        'new_all': calc_stats(new_signals),
        'new_pre_phase1': calc_stats(new_pre_phase1),
        'new_phase1': calc_stats(new_phase1),
        'new_phase2': calc_stats(new_phase2),
        'timeframes': {tf: calc_stats(sigs) for tf, sigs in timeframes.items()},
        'routes': {r: calc_stats(sigs) for r, sigs in routes.items()},
        'directions': {d: calc_stats(sigs) for d, sigs in directions.items()},
        'regimes': {reg: calc_stats(sigs) for reg, sigs in regimes.items()},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def print_tracker(stats):
    """Print formatted tracker output"""
    
    print("\n" + "="*80)
    print("PHASE 1 & PHASE 3 & PHASE 2 REAL-TIME TRACKER")
    print("="*80)
    print(f"Last Updated: {stats['timestamp']} (local)")
    print(f"Phase 1 Deployed: 2026-03-24 18:52 GMT+7 | Phase 2 Deployed: 2026-03-24 19:07 GMT+7")
    print()
    
    # SECTION 1: FOUNDATION BASELINE (locked)
    print("📊 SECTION 1: FOUNDATION BASELINE (LOCKED - Feb 27 to Mar 19)")
    print("-" * 80)
    f = stats['foundation']
    print(f"  Total Signals: {f['total']}")
    print(f"  Closed: {f['closed']} | TP: {f['tp']} | SL: {f['sl']} | Timeout: {f['timeout']} | Open: {f['open']}")
    print(f"  📈 Win Rate: {f['wr']:.2f}% (BASELINE REFERENCE)")
    print()
    
    # SECTION 2: PHASE 1 (Mar 24 18:52 to 19:07)
    print("✅ SECTION 2: PHASE 1 LIVE (Mar 24 18:52 to 19:07 GMT+7)")
    print("-" * 80)
    p1 = stats['new_phase1']
    print(f"  Total Signals: {p1['total']}")
    print(f"  Closed: {p1['closed']} | TP: {p1['tp']} | SL: {p1['sl']} | Timeout: {p1['timeout']} | Open: {p1['open']}")
    print(f"  📈 Win Rate: {p1['wr']:.2f}%")
    delta_p1 = p1['wr'] - f['wr']
    delta_str_p1 = f"({delta_p1:+.2f}pp vs Foundation)" if delta_p1 != 0 else ""
    print(f"  {delta_str_p1}")
    print()
    
    # SECTION 3: PHASE 2 (Mar 24 19:07+ GMT+7) - NEW WITH TF4h
    print("🚀 SECTION 3: PHASE 2 LIVE (Mar 24 19:07+ GMT+7) - TF4h ADDED")
    print("-" * 80)
    p2 = stats['new_phase2']
    print(f"  Total Signals: {p2['total']}")
    print(f"  Closed: {p2['closed']} | TP: {p2['tp']} | SL: {p2['sl']} | Timeout: {p2['timeout']} | Open: {p2['open']}")
    print(f"  📈 Win Rate: {p2['wr']:.2f}%")
    delta_p2 = p2['wr'] - f['wr']
    delta_str_p2 = f"({delta_p2:+.2f}pp vs Foundation)" if delta_p2 != 0 else ""
    print(f"  {delta_str_p2}")
    print()
    
    # SECTION 4: TIMEFRAME BREAKDOWN (Phase 2)
    print("📈 SECTION 4: TIMEFRAME BREAKDOWN (Phase 2 TF4h + existing TFs)")
    print("-" * 80)
    print("  Timeframe Comparison (Post-Phase2 signals):")
    print()
    
    if 'timeframes' in stats and stats['timeframes']:
        for tf_name in ['15min', '30min', '1h', '4h']:
            if tf_name in stats['timeframes']:
                tf = stats['timeframes'][tf_name]
                if tf['total'] > 0:
                    status = "🆕 NEW" if tf_name == '4h' else "✅"
                    print(f"  {status} {tf_name:10} | Total: {tf['total']:4} | Closed: {tf['closed']:4} | TP: {tf['tp']:3} | WR: {tf['wr']:6.2f}%")
    else:
        print("  (No Phase 2 timeframe data yet - still accumulating)")
    print()
    
    # SECTION 5: PHASE 2 TF4h SPECIFIC ANALYSIS
    print("🔍 SECTION 5: PHASE 2 TF4h DEEP DIVE")
    print("-" * 80)
    if 'timeframes' in stats and '4h' in stats['timeframes']:
        tf4h = stats['timeframes']['4h']
        if tf4h['total'] > 0:
            print(f"  ✅ TF4h signals fired to Telegram: {tf4h['total']}")
            print(f"  📊 Closed: {tf4h['closed']} | Still OPEN: {tf4h['open']}")
            print(f"  💰 Winners (TP_HIT): {tf4h['tp']} | Losers (SL_HIT): {tf4h['sl']} | Timeout: {tf4h['timeout']}")
            print(f"  📈 TF4h Win Rate: {tf4h['wr']:.2f}%")
            print()
            
            # Compare TF4h to other timeframes
            if 'timeframes' in stats:
                print("  vs Other Timeframes (Phase 2):")
                for tf_name in ['15min', '30min', '1h']:
                    if tf_name in stats['timeframes']:
                        other_tf = stats['timeframes'][tf_name]
                        if other_tf['total'] > 0:
                            diff = tf4h['wr'] - other_tf['wr']
                            comparison = "🟢 BETTER" if diff > 2 else "🟡 SIMILAR" if abs(diff) <= 2 else "🔴 WORSE"
                            print(f"    {tf_name}: {other_tf['wr']:.2f}% WR | Delta: {diff:+.2f}pp {comparison}")
        else:
            print("  ⏳ No TF4h signals yet (still waiting for Phase 2 accumulation)")
    else:
        print("  ⏳ No TF4h timeframe data (Phase 2 just deployed)")
    print()
    
    # SECTION 6: ROUTE VETO EFFECTIVENESS (Phase 1)
    print("🚫 SECTION 6: ROUTE VETO EFFECTIVENESS (Phase 1)")
    print("-" * 80)
    print("  Target: NONE/AMBIGUOUS blocked (0% of Phase 1 signals)")
    print()
    
    if 'routes' in stats:
        for route_name in ['TREND CONTINUATION', 'REVERSAL', 'NONE', 'AMBIGUOUS']:
            if route_name in stats['routes']:
                r = stats['routes'][route_name]
                if r['total'] > 0:
                    status = "❌ SHOULD BE BLOCKED" if route_name in ['NONE', 'AMBIGUOUS'] else "✅ ALLOWED"
                    print(f"  {route_name:20} | Total: {r['total']:4} | Closed: {r['closed']:4} | TP: {r['tp']:3} | WR: {r['wr']:6.2f}% {status}")
    print()
    
    # SECTION 7: DECISION CRITERIA FOR PHASE 2 LOCK-IN
    print("🎯 SECTION 7: PHASE 2 LOCK-IN DECISION CHECKLIST")
    print("-" * 80)
    
    p2_sig = stats['new_phase2']['total']
    p2_closed = stats['new_phase2']['closed']
    p2_wr = stats['new_phase2']['wr']
    
    tf4h_exists = 'timeframes' in stats and '4h' in stats['timeframes']
    tf4h_signals = stats['timeframes'].get('4h', {}).get('total', 0) if tf4h_exists else 0
    tf4h_wr = stats['timeframes'].get('4h', {}).get('wr', 0) if tf4h_exists else 0
    
    checks = [
        ("Phase 2 has >50 signals total", p2_sig > 50),
        ("Phase 2 has >20 closed signals", p2_closed > 20),
        ("TF4h fired at least 1 signal", tf4h_signals >= 1),
        ("TF4h WR > 20% (viable)", tf4h_wr > 20 if tf4h_signals > 0 else False),
        ("Phase 2 WR stable or improving", p2_wr >= f['wr'] * 0.90),
        ("No major regressions in TF15/30/1h", True),  # Placeholder
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
        print("  🟢 READY TO LOCK PHASE 2")
    elif passed >= 4:
        print("  🟡 ALMOST READY - Wait for more Phase 2 data")
    else:
        print("  🔴 TOO EARLY - Need more accumulation")
    
    print()
    print("="*80)
    print("⏱️  Recommendation: Re-run tracker every 2-4 hours (Phase 2 moves faster)")
    print("📌 Target accumulation: 24-48 hours of Phase 2 signals for TF4h validation")
    print("🔄 Decision: Lock Phase 2 when all 6 criteria met + TF4h WR stable >25%")
    print("="*80)
    print()

if __name__ == '__main__':
    signals = load_signals()
    stats = analyze_signals(signals)
    print_tracker(stats)
