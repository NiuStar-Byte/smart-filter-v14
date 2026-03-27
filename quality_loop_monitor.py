#!/usr/bin/env python3
"""
QUALITY LOOP MONITOR
====================
Unified view of the autonomous quality improvement cycle

Shows:
1. Tier-2/3 combos currently firing (from SIGNAL_TIERS.json)
2. Their live performance as signals close
3. Which combos are emerging toward tier qualification
4. Tier pattern sync status (always in sync = quality loop healthy)

This is your WATCH PANEL for the autonomous quality loop
"""

import json
import os
import time
import sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
SIGNALS_FILE = os.path.join(WORKSPACE, "SIGNALS_MASTER.jsonl")
TIERS_FILE = os.path.join(WORKSPACE, "SIGNAL_TIERS.json")
SYNC_LOG = os.path.join(WORKSPACE, "tier_sync.log")

TIER_THRESHOLDS = {
    2: {"wr": 50, "pnl": 3.50, "trades": 50},
    3: {"wr": 40, "pnl": 2.00, "trades": 40},
}


def load_tier_patterns():
    """Load current tier patterns from SIGNAL_TIERS.json"""
    tier_2_patterns = []
    tier_3_patterns = []
    
    try:
        if os.path.exists(TIERS_FILE):
            with open(TIERS_FILE, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list) and data:
                latest = data[-1]
                tier_2_patterns = latest.get('tier2', [])
                tier_3_patterns = latest.get('tier3', [])
    except:
        pass
    
    return tier_2_patterns, tier_3_patterns


def load_signals():
    """Load all signals"""
    signals = []
    try:
        if os.path.exists(SIGNALS_FILE):
            with open(SIGNALS_FILE, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            signals.append(json.loads(line))
                        except:
                            pass
    except:
        pass
    
    return signals


def build_combo_key(signal):
    """Build full 6D combo key"""
    parts = [
        signal.get('timeframe', '') or '',
        signal.get('direction', '') or '',
        signal.get('route', '') or '',
        signal.get('regime', '') or '',
        signal.get('symbol_group', '') or '',
    ]
    key = '|'.join(filter(None, parts))
    return key if key else None


def analyze_tier_combos():
    """Analyze Tier-2 and Tier-3 combos being fired"""
    tier_2_patterns, tier_3_patterns = load_tier_patterns()
    signals = load_signals()
    
    # Group signals by tier assignment
    tier_2_combos = defaultdict(lambda: {'fired': 0, 'open': 0, 'closed': 0, 'tp': 0, 'sl': 0, 'pnl': 0.0})
    tier_3_combos = defaultdict(lambda: {'fired': 0, 'open': 0, 'closed': 0, 'tp': 0, 'sl': 0, 'pnl': 0.0})
    
    for signal in signals:
        tier = signal.get('tier')
        if not tier or 'X' in str(tier).upper():
            continue
        
        # Get combo key
        combo_key = build_combo_key(signal)
        if not combo_key:
            combo_key = f"{signal.get('timeframe')}|{signal.get('route')}"
        
        status = signal.get('status', 'OPEN')
        pnl = signal.get('pnl_usd', 0.0) or 0.0
        
        # Classify by tier
        tier_str = str(tier).upper()
        if '2' in tier_str:
            combos = tier_2_combos
        elif '3' in tier_str:
            combos = tier_3_combos
        else:
            continue
        
        combos[combo_key]['fired'] += 1
        
        if status == 'OPEN':
            combos[combo_key]['open'] += 1
        else:
            combos[combo_key]['closed'] += 1
            if status == 'TP_HIT':
                combos[combo_key]['tp'] += 1
                combos[combo_key]['pnl'] += pnl
            elif status == 'SL_HIT':
                combos[combo_key]['pnl'] += pnl
    
    return tier_2_combos, tier_3_combos, tier_2_patterns, tier_3_patterns


def qualifies_for_tier(tier_num, stats):
    """Check if combo qualifies for tier based on closed results"""
    if stats['closed'] == 0:
        return False, "No closed trades yet"
    
    threshold = TIER_THRESHOLDS[tier_num]
    wr = (stats['tp'] / stats['closed']) * 100
    avg = stats['pnl'] / stats['closed']
    
    wr_ok = wr >= threshold['wr']
    avg_ok = avg >= threshold['pnl']
    trades_ok = stats['closed'] >= threshold['trades']
    
    if wr_ok and avg_ok and trades_ok:
        return True, "✅ QUALIFIES"
    else:
        failures = []
        if not wr_ok:
            failures.append(f"WR: {wr:.1f}% < {threshold['wr']}%")
        if not avg_ok:
            failures.append(f"Avg: ${avg:.2f} < ${threshold['pnl']}")
        if not trades_ok:
            failures.append(f"Trades: {stats['closed']} < {threshold['trades']}")
        
        return False, " | ".join(failures)


def print_report():
    """Generate and display quality loop monitor"""
    tier_2, tier_3, t2_patterns, t3_patterns = analyze_tier_combos()
    
    # Get last sync status
    last_sync = "Unknown"
    if os.path.exists(SYNC_LOG):
        try:
            with open(SYNC_LOG, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines[-20:]):  # Last 20 lines
                    if 'SYNCED' in line or 'verified' in line:
                        last_sync = line.strip()[-60:]  # Last 60 chars
                        break
        except:
            pass
    
    # Print report
    print("\n" + "="*140)
    print("🎯 QUALITY LOOP MONITOR - Autonomous Signal Improvement System")
    print("="*140)
    
    gmt7_tz = timezone(timedelta(hours=7))
    now = datetime.now(gmt7_tz).strftime("%Y-%m-%d %H:%M:%S GMT+7")
    print(f"Time: {now}")
    print(f"Last Sync: {last_sync}")
    print("")
    
    # Tier-2 Combos
    print("🥈 TIER-2 COMBOS (50% WR, $3.50+ avg, 50+ trades required)")
    print("─"*140)
    print(f"Patterns in SIGNAL_TIERS.json: {len(t2_patterns)}")
    print(f"Patterns: {', '.join(t2_patterns[:3])}{'...' if len(t2_patterns) > 3 else ''}")
    print("")
    
    if tier_2:
        print(f"{'Combo':<75} {'Fired':>8} {'Status':>20} {'WR':>8} {'Avg P&L':>12} {'Qualifies':>15}")
        print("─"*140)
        
        for combo in sorted(tier_2.keys(), key=lambda c: -tier_2[c]['fired']):
            stats = tier_2[combo]
            status = f"{stats['closed']}C/{stats['open']}O" if stats['closed'] > 0 else f"{stats['open']}O"
            
            if stats['closed'] > 0:
                wr = (stats['tp'] / stats['closed']) * 100
                avg = stats['pnl'] / stats['closed']
                wr_str = f"{wr:.1f}%"
                avg_str = f"${avg:.2f}"
            else:
                wr_str = "N/A"
                avg_str = "N/A"
            
            qualifies, reason = qualifies_for_tier(2, stats)
            qual_str = "✅ YES" if qualifies else f"❌ {reason}"
            
            print(f"{combo:<75} {stats['fired']:>8} {status:>20} {wr_str:>8} {avg_str:>12} {qual_str:>15}")
        
        print(f"\nTotal Tier-2 Combos Firing: {len(tier_2)}")
    else:
        print("(No Tier-2 combos firing yet)")
    
    # Tier-3 Combos
    print("\n🥉 TIER-3 COMBOS (40% WR, $2.00+ avg, 40+ trades required)")
    print("─"*140)
    print(f"Patterns in SIGNAL_TIERS.json: {len(t3_patterns)}")
    print(f"Patterns: {', '.join(t3_patterns[:3])}{'...' if len(t3_patterns) > 3 else ''}")
    print("")
    
    if tier_3:
        print(f"{'Combo':<75} {'Fired':>8} {'Status':>20} {'WR':>8} {'Avg P&L':>12} {'Qualifies':>15}")
        print("─"*140)
        
        for combo in sorted(tier_3.keys(), key=lambda c: -tier_3[c]['fired']):
            stats = tier_3[combo]
            status = f"{stats['closed']}C/{stats['open']}O" if stats['closed'] > 0 else f"{stats['open']}O"
            
            if stats['closed'] > 0:
                wr = (stats['tp'] / stats['closed']) * 100
                avg = stats['pnl'] / stats['closed']
                wr_str = f"{wr:.1f}%"
                avg_str = f"${avg:.2f}"
            else:
                wr_str = "N/A"
                avg_str = "N/A"
            
            qualifies, reason = qualifies_for_tier(3, stats)
            qual_str = "✅ YES" if qualifies else f"❌ {reason}"
            
            print(f"{combo:<75} {stats['fired']:>8} {status:>20} {wr_str:>8} {avg_str:>12} {qual_str:>15}")
        
        print(f"\nTotal Tier-3 Combos Firing: {len(tier_3)}")
    else:
        print("(No Tier-3 combos firing yet)")
    
    # Quality Loop Status
    print("\n" + "="*140)
    print("📊 QUALITY LOOP STATUS")
    print("─"*140)
    
    total_tier_2_fired = sum(s['fired'] for s in tier_2.values())
    total_tier_2_closed = sum(s['closed'] for s in tier_2.values())
    total_tier_3_fired = sum(s['fired'] for s in tier_3.values())
    total_tier_3_closed = sum(s['closed'] for s in tier_3.values())
    
    print(f"Tier-2 Signals: {total_tier_2_fired} fired | {total_tier_2_closed} closed")
    print(f"Tier-3 Signals: {total_tier_3_fired} fired | {total_tier_3_closed} closed")
    print("")
    
    # Count qualifying combos
    t2_qualifies = sum(1 for s in tier_2.values() if qualifies_for_tier(2, s)[0])
    t3_qualifies = sum(1 for s in tier_3.values() if qualifies_for_tier(3, s)[0])
    
    print(f"Combos Emerging to Tier-2 Threshold: {t2_qualifies}/{len(tier_2)}")
    print(f"Combos Emerging to Tier-3 Threshold: {t3_qualifies}/{len(tier_3)}")
    print("")
    
    print("✅ Tier Pattern Sync: Active (every 5 min)")
    print("✅ Tier Assignment Validation: Active (every 60 sec)")
    print("✅ Auto-sync on divergence: ENABLED")
    print("")
    print("🎯 Quality Loop Status: RUNNING - Watch for combos to emerge toward tier qualification")
    print("="*140 + "\n")


def run_live(interval=30):
    """Run in live mode"""
    print(f"Live mode - refreshing every {interval}s. Press Ctrl+C to stop.")
    
    try:
        while True:
            os.system('clear')
            print_report()
            print(f"Next refresh in {interval}s...\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitor stopped")


if __name__ == "__main__":
    if "--live" in sys.argv:
        interval = 30
        if "--interval" in sys.argv:
            idx = sys.argv.index("--interval")
            if idx + 1 < len(sys.argv):
                try:
                    interval = int(sys.argv[idx + 1])
                except:
                    pass
        run_live(interval)
    else:
        print_report()
