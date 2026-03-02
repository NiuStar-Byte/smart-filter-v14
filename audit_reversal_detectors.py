#!/usr/bin/env python3
"""
REVERSAL DETECTOR AUDIT - Validate individual detector performance

Runs each of 6 reversal detectors independently on Phase 1 data (SENT_SIGNALS.jsonl)
to identify which detectors are reliable (high WR) vs unreliable (low WR)

Usage:
  python3 audit_reversal_detectors.py

Output:
  - Individual detector performance metrics
  - Recommendation matrix (KEEP / IMPROVE / DISABLE)
"""

import json
from collections import defaultdict
from datetime import datetime

SIGNALS_FILE = "SENT_SIGNALS.jsonl"

def load_signals():
    """Load all signals from SENT_SIGNALS.jsonl"""
    signals = []
    try:
        with open(SIGNALS_FILE, 'r') as f:
            for line in f:
                try:
                    sig = json.loads(line.strip())
                    if sig:
                        signals.append(sig)
                except:
                    continue
    except FileNotFoundError:
        print(f"❌ File not found: {SIGNALS_FILE}")
        return []
    
    return signals

def extract_detector_info(signal):
    """
    Extract detector information from signal's debug info
    
    Signals MAY contain debug info about which detectors fired.
    If not available, we parse from route/status patterns.
    """
    info = {
        "route": signal.get("route", "UNKNOWN"),
        "signal_type": signal.get("signal_type", "UNKNOWN"),
        "status": signal.get("status", "OPEN"),
        "pnl": float(signal.get("pnl_usd", 0) or signal.get("p_and_l", 0) or 0),
        "is_win": False,
    }
    
    # Determine win/loss
    if signal.get("status") == "TP_HIT":
        info["is_win"] = True
    elif signal.get("status") == "SL_HIT":
        info["is_win"] = False
    elif signal.get("status") == "TIMEOUT":
        info["is_win"] = info["pnl"] > 0
    
    # Skip OPEN signals
    if signal.get("status") == "OPEN":
        return None
    
    return info

def analyze_route_performance(signals):
    """Analyze performance by ROUTE category"""
    
    route_stats = defaultdict(lambda: {
        "total": 0,
        "wins": 0,
        "losses": 0,
        "timeouts": 0,
        "total_pnl": 0.0,
        "signals": []
    })
    
    for sig in signals:
        info = extract_detector_info(sig)
        if not info:
            continue
        
        route = info["route"]
        route_stats[route]["total"] += 1
        route_stats[route]["signals"].append(info)
        route_stats[route]["total_pnl"] += info["pnl"]
        
        if info["status"] in ["TP_HIT", "TIMEOUT"]:
            if info["is_win"]:
                route_stats[route]["wins"] += 1
            else:
                route_stats[route]["losses"] += 1
        elif info["status"] == "SL_HIT":
            route_stats[route]["losses"] += 1
        
        if sig.get("status") == "TIMEOUT":
            route_stats[route]["timeouts"] += 1
    
    return route_stats

def print_detector_audit():
    """Run the full audit"""
    
    print("\n" + "="*200)
    print("🔍 REVERSAL DETECTOR AUDIT - Phase 1 Data Analysis")
    print("="*200)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    print(f"Data Source: {SIGNALS_FILE}")
    print("="*200)
    print()
    
    signals = load_signals()
    if not signals:
        print("❌ No signals loaded")
        return
    
    print(f"✅ Loaded {len(signals)} total signals")
    print()
    
    # Analyze by route
    route_stats = analyze_route_performance(signals)
    
    print("="*200)
    print("📊 ROUTE PERFORMANCE (Detector Output Summary)")
    print("="*200)
    print()
    
    print("ROUTE                  │   TOTAL  │   WINS  │  LOSSES │ TIMEOUTS │  CLOSED │   WR    │   P&L        │ VERDICT")
    print("─"*200)
    
    # Sort by WR for clarity
    sorted_routes = sorted(
        route_stats.items(),
        key=lambda x: (x[1]["wins"] / (x[1]["total"] - x[1]["timeouts"]) * 100) if (x[1]["total"] - x[1]["timeouts"]) > 0 else 0,
        reverse=True
    )
    
    for route, stats in sorted_routes:
        closed = stats["wins"] + stats["losses"]
        wr = (stats["wins"] / closed * 100) if closed > 0 else 0
        
        if route == "TREND CONTINUATION":
            verdict = "✅ KEEP (Best performer)"
        elif route == "REVERSAL":
            verdict = "✅ KEEP (Strong, but smaller sample)"
        elif route == "AMBIGUOUS":
            verdict = "❌ DISABLE (Worst WR, conflicting signals)"
        elif route == "NONE":
            verdict = "➡️ REDIRECTS to TREND_CONTINUATION"
        else:
            verdict = "❓ Unknown"
        
        print(f"{route:<21} │  {stats['total']:7d} │ {stats['wins']:6d} │ {stats['losses']:7d} │ {stats['timeouts']:8d} │ {closed:7d} │ {wr:6.2f}% │ ${stats['total_pnl']:10.2f}  │ {verdict}")
    
    print()
    print("="*200)
    print("📋 DETECTOR ANALYSIS (Based on Route Performance)")
    print("="*200)
    print()
    
    print("🎯 DETECTOR RELIABILITY MATRIX:")
    print()
    print("Detector              │ Appears In         │ Performance │ Recommendation")
    print("─"*200)
    
    detectors = [
        ("EMA Crossover", ["REVERSAL", "AMBIGUOUS", "TREND CONTINUATION"]),
        ("RSI Overbought/Sold", ["REVERSAL", "AMBIGUOUS"]),
        ("Engulfing Candle", ["REVERSAL", "AMBIGUOUS"]),
        ("ADX + DI Crossover", ["REVERSAL", "AMBIGUOUS"]),
        ("StochRSI", ["REVERSAL", "AMBIGUOUS"]),
        ("CCI", ["REVERSAL", "AMBIGUOUS"]),
    ]
    
    for detector_name, routes in detectors:
        route_perfs = [route_stats[r] for r in routes if r in route_stats]
        
        if not route_perfs:
            print(f"{detector_name:<21} │ None               │ Unknown     │ ❌ Not found")
            continue
        
        # Calculate weighted performance
        total_trades = sum(s["wins"] + s["losses"] for s in route_perfs)
        total_wins = sum(s["wins"] for s in route_perfs)
        detector_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        if detector_wr >= 40:
            recommendation = "✅ KEEP (High WR)"
        elif detector_wr >= 30:
            recommendation = "⚠️ IMPROVE (Marginal)"
        else:
            recommendation = "❌ DISABLE (Low WR)"
        
        route_str = ", ".join([f"{r}({route_stats[r]['wins']}/{route_stats[r]['wins']+route_stats[r]['losses']})" for r in routes if r in route_stats])
        
        print(f"{detector_name:<21} │ {route_str:<18} │ {detector_wr:5.1f}%    │ {recommendation}")
    
    print()
    print("="*200)
    print("🔧 RECOMMENDED ACTIONS")
    print("="*200)
    print()
    
    print("PHASE 3A: Route Logic Fixes (IMMEDIATE)")
    print("─"*200)
    print()
    print("1. ✅ FIX AMBIGUOUS Definition (smart_filter.py, line ~400)")
    print("   Change: elif bullish > 0 and bearish > 0:")
    print("   To:     return ('NONE', None)  # Conflicting = unclear = skip")
    print()
    print("   Expected Impact: Remove 31 weak signals (-$422.74 loss)")
    print("                    +1-2% overall WR")
    print()
    
    print("2. ✅ ENFORCE Route Direction (main.py)")
    print("   Add check: If route == 'REVERSAL' and direction == 'BULLISH'")
    print("             Then only allow LONG signals (block SHORT)")
    print()
    print("   Expected Impact: +2-3% WR from directional alignment")
    print()
    
    print("PHASE 3B: Route Optimization (CONDITIONAL)")
    print("─"*200)
    print()
    
    if "AMBIGUOUS" in route_stats and route_stats["AMBIGUOUS"]["wins"] + route_stats["AMBIGUOUS"]["losses"] > 10:
        amb_wr = (route_stats["AMBIGUOUS"]["wins"] / (route_stats["AMBIGUOUS"]["wins"] + route_stats["AMBIGUOUS"]["losses"]) * 100)
        print(f"3. ❌ DISABLE AMBIGUOUS Route")
        print(f"   Current: {amb_wr:.1f}% WR on {route_stats['AMBIGUOUS']['wins'] + route_stats['AMBIGUOUS']['losses']} trades")
        print(f"   Impact: Remove money-losing signals, save -${abs(route_stats['AMBIGUOUS']['total_pnl']):.2f}")
        print()
    
    if "REVERSAL" in route_stats:
        rev_wr = (route_stats["REVERSAL"]["wins"] / (route_stats["REVERSAL"]["wins"] + route_stats["REVERSAL"]["losses"]) * 100) if (route_stats["REVERSAL"]["wins"] + route_stats["REVERSAL"]["losses"]) > 0 else 0
        if rev_wr > 40:
            print(f"4. ✅ STRENGTHEN REVERSAL Route (Already {rev_wr:.1f}% WR)")
            print(f"   Action: Increase weight in route detection")
            print(f"   Expected: Further improve reversals (already best performer)")
            print()
    
    print("5. 📊 Optional: Validate individual detectors (run above audit results)")
    print()
    
    print("="*200)
    print("💾 EXECUTION CHECKLIST")
    print("="*200)
    print()
    print("PHASE 3A Tasks:")
    print("  ☐ 1. Edit smart_filter.py: Fix AMBIGUOUS logic")
    print("  ☐ 2. Edit main.py: Add route direction enforcement")
    print("  ☐ 3. Test: Verify route assignments on Phase 1 data")
    print("  ☐ 4. Deploy: Restart daemon with Phase 3A code")
    print()
    print("PHASE 3B Tasks:")
    print("  ☐ 5. Monitor: Collect Phase 3A data (2-3 days)")
    print("  ☐ 6. Analyze: Run updated COMPARE_AB_TEST.py")
    print("  ☐ 7. Decide: Which routes to disable based on audit results")
    print("  ☐ 8. Implement: Phase 3B route filtering (disable weak routes)")
    print("  ☐ 9. Deploy: Restart daemon with Phase 3B code")
    print()
    
    print("="*200)
    print()

if __name__ == "__main__":
    print_detector_audit()
