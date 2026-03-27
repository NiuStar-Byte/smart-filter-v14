#!/usr/bin/env python3
"""
TIER PERFORMANCE TRACKER - FOCUSED VERSION
===========================================
Shows Tier-1, Tier-2, Tier-3 combos with detailed performance
Reads tier field from signals (assigned by main.py)
"""

import json
import os
import time
import sys
from datetime import datetime, timezone, timedelta

# ============================================================================
# CONSTANTS
# ============================================================================

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
SIGNALS_FILE = os.path.join(WORKSPACE, "SIGNALS_MASTER.jsonl")
OUTPUT_FILE = os.path.join(WORKSPACE, "TIER_PERFORMANCE_FOCUSED.txt")
ARCHIVE_DIR = os.path.join(WORKSPACE, "tier_performance_reports")

# Ensure archive directory exists
os.makedirs(ARCHIVE_DIR, exist_ok=True)


# ============================================================================
# COMBO METRICS
# ============================================================================

class ComboMetrics:
    def __init__(self, combo_key, tier):
        self.combo_key = combo_key
        self.tier = tier
        self.signals = []
        self.total_signals = 0
        self.open_count = 0
        self.closed_count = 0
        self.tp_hit = 0
        self.sl_hit = 0
        self.timeout_win = 0
        self.timeout_loss = 0
        self.total_pnl = 0.0
    
    def add_signal(self, signal):
        self.signals.append(signal)
        self.total_signals += 1
        
        status = signal.get("status", "OPEN")
        pnl = signal.get("pnl_usd", 0.0) or 0.0
        
        if status == "OPEN":
            self.open_count += 1
        else:
            self.closed_count += 1
            
            if status == "TP_HIT":
                self.tp_hit += 1
                self.total_pnl += pnl
            elif status == "SL_HIT":
                self.sl_hit += 1
                self.total_pnl += pnl
            elif status == "TIMEOUT_WIN":
                self.timeout_win += 1
                self.total_pnl += pnl
            elif status == "TIMEOUT_LOSS":
                self.timeout_loss += 1
                self.total_pnl += pnl
    
    @property
    def wr_percent(self):
        if self.closed_count == 0:
            return 0.0
        wins = self.tp_hit + self.timeout_win
        return (wins / self.closed_count) * 100
    
    @property
    def avg_pnl(self):
        if self.closed_count == 0:
            return 0.0
        return self.total_pnl / self.closed_count


# ============================================================================
# LOAD DATA
# ============================================================================

def load_signals():
    signals = []
    if not os.path.exists(SIGNALS_FILE):
        print(f"ERROR: {SIGNALS_FILE} not found")
        return signals
    
    with open(SIGNALS_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                signal = json.loads(line)
                signals.append(signal)
            except json.JSONDecodeError:
                pass
    
    return signals


# ============================================================================
# ANALYZE & GENERATE REPORT
# ============================================================================

def generate_report():
    print("[TIER TRACKER] Loading signals...")
    signals = load_signals()
    print(f"[TIER TRACKER] Loaded {len(signals)} signals")
    
    print("[TIER TRACKER] Analyzing by tier...")
    
    # Group by combo + tier
    tier_combos = {}  # (combo_key, tier) → ComboMetrics
    tier_1_combos = []
    tier_2_combos = []
    tier_3_combos = []
    tier_x_signals = 0
    
    for signal in signals:
        tier = signal.get('tier')  # Read tier field assigned by main.py
        
        # Skip if no tier or Tier-X
        if tier is None or str(tier).upper() in ['TIER-X', 'NONE', 'NULL']:
            tier_x_signals += 1
            continue
        
        # Build combo key
        tf = signal.get("timeframe", "")
        direction = signal.get("direction", "")
        route = signal.get("route", "")
        
        route_parts = route.split("|") if route else []
        combo_parts = [tf, direction] + route_parts if route_parts else [tf, direction]
        combo_key = "|".join(filter(None, combo_parts))
        
        if not combo_key:
            tier_x_signals += 1
            continue
        
        # Normalize tier names (handles "Tier-1", "Tier-2", "Tier-3", 1, 2, 3, etc.)
        tier_str = str(tier).upper()
        tier_num = None
        
        if '1' in tier_str:
            tier_num = 1
        elif '2' in tier_str:
            tier_num = 2
        elif '3' in tier_str:
            tier_num = 3
        else:
            tier_x_signals += 1
            continue
        
        key = (combo_key, tier_num)
        if key not in tier_combos:
            tier_combos[key] = ComboMetrics(combo_key, tier_num)
        
        tier_combos[key].add_signal(signal)
    
    # Organize by tier (show combos with ANY signals, not just closed trades)
    for (combo_key, tier_num), combo in tier_combos.items():
        if combo.total_signals > 0:  # Show combos that have fired at least 1 signal
            if tier_num == 1:
                tier_1_combos.append(combo)
            elif tier_num == 2:
                tier_2_combos.append(combo)
            elif tier_num == 3:
                tier_3_combos.append(combo)
    
    # Sort by WR descending
    tier_1_combos.sort(key=lambda c: -c.wr_percent)
    tier_2_combos.sort(key=lambda c: -c.wr_percent)
    tier_3_combos.sort(key=lambda c: -c.wr_percent)
    
    # Generate report
    report_lines = []
    report_lines.append("═" * 120)
    report_lines.append("🎯 TIER PERFORMANCE - ACTIVE COMBOS (Tier-1/2/3)")
    report_lines.append("═" * 120)
    
    gmt7_tz = timezone(timedelta(hours=7))
    now_gmt7 = datetime.now(timezone.utc).astimezone(gmt7_tz)
    report_lines.append(f"Report Generated: {now_gmt7.strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    report_lines.append("")
    
    # Tier-1
    report_lines.append("")
    report_lines.append("🥇 TIER-1: ELITE (Active Signals)")
    report_lines.append("─" * 120)
    
    if tier_1_combos:
        report_lines.append(f"{'Combo Key':<50} {'Fired':>8} {'Open':>8} {'Closed':>8} {'WR':>8} {'P&L':>12}")
        report_lines.append("─" * 120)
        for combo in tier_1_combos:
            wr_str = f"{combo.wr_percent:.1f}%" if combo.closed_count > 0 else "N/A"
            pnl_str = f"${combo.total_pnl:,.2f}" if combo.closed_count > 0 else "N/A"
            report_lines.append(
                f"{combo.combo_key:<50} {combo.total_signals:>8} {combo.open_count:>8} {combo.closed_count:>8} {wr_str:>8} {pnl_str:>12}"
            )
        
        # Summary stats
        total_t1_signals = sum(c.total_signals for c in tier_1_combos)
        total_t1_closed = sum(c.closed_count for c in tier_1_combos)
        total_t1_pnl = sum(c.total_pnl for c in tier_1_combos)
        avg_wr = sum(c.wr_percent for c in tier_1_combos) / len(tier_1_combos) if tier_1_combos else 0
        
        report_lines.append("─" * 120)
        report_lines.append(f"{'TIER-1 TOTAL':<50} {avg_wr:>7.2f}% {total_t1_signals:>15} {total_t1_closed:>8} ${total_t1_pnl:>11,.2f}")
        report_lines.append(f"Combos: {len(tier_1_combos)}")
    else:
        report_lines.append("(None yet - watch for signals with Tier-1 assignment)")
    
    # Tier-2
    report_lines.append("")
    report_lines.append("🥈 TIER-2: GOOD (Active Signals)")
    report_lines.append("─" * 120)
    
    if tier_2_combos:
        report_lines.append(f"{'Combo Key':<50} {'Fired':>8} {'Open':>8} {'Closed':>8} {'WR':>8} {'P&L':>12}")
        report_lines.append("─" * 120)
        for combo in tier_2_combos:
            wr_str = f"{combo.wr_percent:.1f}%" if combo.closed_count > 0 else "N/A"
            pnl_str = f"${combo.total_pnl:,.2f}" if combo.closed_count > 0 else "N/A"
            report_lines.append(
                f"{combo.combo_key:<50} {combo.total_signals:>8} {combo.open_count:>8} {combo.closed_count:>8} {wr_str:>8} {pnl_str:>12}"
            )
        
        # Summary stats
        total_t2_signals = sum(c.total_signals for c in tier_2_combos)
        total_t2_closed = sum(c.closed_count for c in tier_2_combos)
        total_t2_pnl = sum(c.total_pnl for c in tier_2_combos)
        avg_wr = sum(c.wr_percent for c in tier_2_combos) / len(tier_2_combos) if tier_2_combos else 0
        
        report_lines.append("─" * 120)
        report_lines.append(f"{'TIER-2 TOTAL':<50} {avg_wr:>7.2f}% {total_t2_signals:>15} {total_t2_closed:>8} ${total_t2_pnl:>11,.2f}")
        report_lines.append(f"Combos: {len(tier_2_combos)}")
    else:
        report_lines.append("(None yet - watch for signals with Tier-2 assignment)")
    
    # Tier-3
    report_lines.append("")
    report_lines.append("🥉 TIER-3: ACCEPTABLE (Active Signals)")
    report_lines.append("─" * 120)
    
    if tier_3_combos:
        report_lines.append(f"{'Combo Key':<50} {'Fired':>8} {'Open':>8} {'Closed':>8} {'WR':>8} {'P&L':>12}")
        report_lines.append("─" * 120)
        for combo in tier_3_combos:
            wr_str = f"{combo.wr_percent:.1f}%" if combo.closed_count > 0 else "N/A"
            pnl_str = f"${combo.total_pnl:,.2f}" if combo.closed_count > 0 else "N/A"
            report_lines.append(
                f"{combo.combo_key:<50} {combo.total_signals:>8} {combo.open_count:>8} {combo.closed_count:>8} {wr_str:>8} {pnl_str:>12}"
            )
        
        # Summary stats
        total_t3_signals = sum(c.total_signals for c in tier_3_combos)
        total_t3_closed = sum(c.closed_count for c in tier_3_combos)
        total_t3_pnl = sum(c.total_pnl for c in tier_3_combos)
        avg_wr = sum(c.wr_percent for c in tier_3_combos) / len(tier_3_combos) if tier_3_combos else 0
        
        report_lines.append("─" * 120)
        report_lines.append(f"{'TIER-3 TOTAL':<50} {avg_wr:>7.2f}% {total_t3_signals:>15} {total_t3_closed:>8} ${total_t3_pnl:>11,.2f}")
        report_lines.append(f"Combos: {len(tier_3_combos)}")
    else:
        report_lines.append("(None yet - watch for signals with Tier-3 assignment)")
    
    # Summary
    report_lines.append("")
    report_lines.append("═" * 120)
    report_lines.append("📊 SUMMARY")
    report_lines.append("─" * 120)
    report_lines.append(f"Tier-1 Combos: {len(tier_1_combos)} (Active - Elite performers)")
    report_lines.append(f"Tier-2 Combos: {len(tier_2_combos)} (Active - Good performers)")
    report_lines.append(f"Tier-3 Combos: {len(tier_3_combos)} (Active - Acceptable performers)")
    report_lines.append(f"Tier-X Signals: {tier_x_signals} (In-Training)")
    report_lines.append("")
    
    # Project 4 Readiness
    total_fired_combos = len(tier_1_combos) + len(tier_2_combos) + len(tier_3_combos)
    if len(tier_1_combos) >= 3 and len(tier_2_combos) >= 5:
        readiness = "🟢 HIGH - Multiple Tier-1/2 combos firing consistently"
    elif len(tier_2_combos) >= 3:
        readiness = "🟡 MEDIUM - Tier-2 combos emerging"
    elif total_fired_combos > 0:
        readiness = "🟡 MEDIUM - Tier-3/2 combos starting to fire"
    else:
        readiness = "🔴 LOW - Waiting for first tiered combos"
    
    report_lines.append(f"Project 4 Readiness: {readiness}")
    report_lines.append("═" * 120)
    
    report_text = "\n".join(report_lines)
    
    # Write files
    with open(OUTPUT_FILE, 'w') as f:
        f.write(report_text)
    print(f"[TIER TRACKER] Report written to {OUTPUT_FILE}")
    
    # Archive
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_file = os.path.join(ARCHIVE_DIR, f"TIER_FOCUSED_{timestamp}.txt")
    with open(archive_file, 'w') as f:
        f.write(report_text)
    print(f"[TIER TRACKER] Archived to {archive_file}")
    
    return report_text


def run_live_mode(refresh_interval=30):
    """Live refresh mode"""
    print(f"[TIER TRACKER] Starting LIVE mode (refresh every {refresh_interval}s)")
    print("[TIER TRACKER] Press Ctrl+C to stop\n")
    
    try:
        while True:
            os.system('clear')
            report = generate_report()
            print("\n" + report)
            print(f"\n[TIER TRACKER] Next refresh in {refresh_interval}s... (Ctrl+C to stop)")
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\n[TIER TRACKER] Live mode stopped")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if "--live" in sys.argv:
        refresh_interval = 30
        if "--interval" in sys.argv:
            idx = sys.argv.index("--interval")
            if idx + 1 < len(sys.argv):
                try:
                    refresh_interval = int(sys.argv[idx + 1])
                except ValueError:
                    pass
        run_live_mode(refresh_interval)
    else:
        report = generate_report()
        print("\n" + report)
