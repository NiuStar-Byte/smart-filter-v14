#!/usr/bin/env python3
"""
TIER PERFORMANCE COMPARISON TRACKER
=====================================
Empirical comparison: Tiered vs Non-Tiered signals across pre/post 3/4 normalization

Groups:
- GROUP A: Pre-3/4 norm (< 2026-03-25T17:54:00Z) - NON-TIERED
- GROUP B: Post-3/4 norm (>= 2026-03-25T17:54:00Z) - NON-TIERED
- GROUP C: Pre-3/4 norm (< 2026-03-25T17:54:00Z) - TIERED (1,2,3)
- GROUP D: Post-3/4 norm (>= 2026-03-25T17:54:00Z) - TIERED (1,2,3)

Purpose: Validate hypothesis that "Tiered signals have better performance than non-tiered"
Output: TIER_COMPARISON_REPORT.txt + timestamped archive
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ============================================================================
# CONSTANTS
# ============================================================================

CUTOFF_TIMESTAMP = datetime.fromisoformat("2026-03-25T17:54:00+00:00")
WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
SIGNALS_FILE = os.path.join(WORKSPACE, "SIGNALS_MASTER.jsonl")
TIERS_FILE = os.path.join(WORKSPACE, "SIGNAL_TIERS.json")
OUTPUT_FILE = os.path.join(WORKSPACE, "TIER_COMPARISON_REPORT.txt")
ARCHIVE_DIR = os.path.join(WORKSPACE, "tier_comparison_reports")

# Ensure archive directory exists
os.makedirs(ARCHIVE_DIR, exist_ok=True)


# ============================================================================
# TIER LOOKUP
# ============================================================================

def load_tier_mappings():
    """Load tier patterns from SIGNAL_TIERS.json and build combo → tier lookup"""
    tier_map = {}  # pattern → tier number
    
    if not os.path.exists(TIERS_FILE):
        print(f"[TIER COMPARISON] Warning: {TIERS_FILE} not found")
        return tier_map
    
    try:
        with open(TIERS_FILE, 'r') as f:
            data = json.load(f)
        
        # Get the latest entry (last in list)
        if isinstance(data, list) and data:
            latest = data[-1]
            for tier_num, tier_name in [(1, 'tier1'), (2, 'tier2'), (3, 'tier3')]:
                if tier_name in latest:
                    for pattern in latest[tier_name]:
                        tier_map[pattern] = tier_num
        elif isinstance(data, dict) and data:
            # Fallback for dict format
            latest = data.get(str(max(map(int, data.keys()))))
            if latest:
                for tier_num, tier_name in [(1, 'tier1'), (2, 'tier2'), (3, 'tier3')]:
                    if tier_name in latest:
                        for pattern in latest[tier_name]:
                            tier_map[pattern] = tier_num
    except Exception as e:
        print(f"[TIER COMPARISON] Error loading tier mappings: {e}")
    
    return tier_map


def get_signal_tier(signal, tier_map):
    """Determine signal tier by matching combo pattern against tier_map"""
    # Extract signal fields
    timeframe = signal.get("timeframe", "").lower() if signal.get("timeframe") else ""
    direction = signal.get("direction", "").upper() if signal.get("direction") else ""
    regime = signal.get("regime", "").upper() if signal.get("regime") else ""
    route = signal.get("route", "").upper() if signal.get("route") else ""
    
    # Normalize timeframe (remove 'min' suffix, convert to number)
    tf_short = timeframe.replace("min", "") if timeframe else ""
    
    # Generate all possible pattern combinations to try (in order of specificity)
    patterns_to_try = [
        # Full form with all fields
        f"{tf_short}_{direction}_{route}_{regime}",
        f"{timeframe}_{direction}_{route}_{regime}",
        
        # Shorthand prefix + values (most common in SIGNAL_TIERS.json)
        f"TF_DIR_ROUTE_REGIME_{tf_short}_{direction}_{route}_{regime}",
        f"TF_DIR_ROUTE_{tf_short}_{direction}_{route}",
        f"TF_DIR_REGIME_{tf_short}_{direction}_{regime}",
        f"TF_ROUTE_REGIME_{tf_short}_{route}_{regime}",
        f"DIR_ROUTE_REGIME_{direction}_{route}_{regime}",
        f"TF_DIR_{tf_short}_{direction}",
        f"TF_REGIME_{tf_short}_{regime}",
        f"TF_ROUTE_{tf_short}_{route}",
        f"DIR_REGIME_{direction}_{regime}",
        f"DIR_ROUTE_{direction}_{route}",
        f"ROUTE_REGIME_{route}_{regime}",
        
        # Values only (partial patterns)
        f"{tf_short}_{direction}_{regime}",
        f"{tf_short}_{direction}_{route}",
        f"{direction}_{regime}",
        f"{direction}_{route}",
    ]
    
    for pattern in patterns_to_try:
        if pattern in tier_map:
            return tier_map[pattern]
    
    return None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class GroupMetrics:
    def __init__(self, name):
        self.name = name
        self.signals = []
        self.signal_count = 0
        self.closed_trades = 0
        self.tp_hit = 0
        self.sl_hit = 0
        self.timeout_win = 0
        self.timeout_loss = 0
        self.open_count = 0
        self.total_pnl = 0.0
        self.tp_pnl = 0.0
        self.sl_pnl = 0.0
        self.timeout_pnl = 0.0
        self.rr_values = []
        self.tier_distribution = defaultdict(int)

    def calculate(self):
        """Calculate derived metrics"""
        self.signal_count = len(self.signals)
        
        # Calculate P&L totals by outcome
        for signal in self.signals:
            status = signal.get("status", "OPEN")
            pnl = signal.get("pnl_usd", 0.0)
            if pnl is None:
                pnl = 0.0
            
            if status == "TP_HIT":
                self.tp_hit += 1
                self.tp_pnl += pnl
            elif status == "SL_HIT":
                self.sl_hit += 1
                self.sl_pnl += pnl
            elif status == "TIMEOUT_WIN":
                self.timeout_win += 1
                self.timeout_pnl += pnl
            elif status == "TIMEOUT_LOSS":
                self.timeout_loss += 1
                self.timeout_pnl += pnl
            elif status == "OPEN":
                self.open_count += 1
            
            self.total_pnl += pnl
            
            # Track RR if available
            if "rr" in signal and signal["rr"] is not None:
                self.rr_values.append(signal["rr"])
        
        # Now calculate closed_trades after the loop
        self.closed_trades = self.tp_hit + self.sl_hit + self.timeout_win + self.timeout_loss

    @property
    def wr_percent(self):
        """Overall Win Rate: (TP_HIT + TIMEOUT_WIN) / Closed * 100"""
        if self.closed_trades == 0:
            return 0.0
        return ((self.tp_hit + self.timeout_win) / self.closed_trades) * 100

    @property
    def wr_tp_sl_only(self):
        """Win Rate based on TP & SL ONLY: TP_HIT / (TP_HIT + SL_HIT) * 100"""
        tp_sl_total = self.tp_hit + self.sl_hit
        if tp_sl_total == 0:
            return 0.0
        return (self.tp_hit / tp_sl_total) * 100

    @property
    def avg_per_signal(self):
        """Average P&L per signal"""
        if self.signal_count == 0:
            return 0.0
        return self.total_pnl / self.signal_count

    @property
    def avg_per_closed_trade(self):
        """Average P&L per closed trade"""
        if self.closed_trades == 0:
            return 0.0
        return self.total_pnl / self.closed_trades

    @property
    def avg_tp_pnl(self):
        """Average P&L per TP_HIT"""
        if self.tp_hit == 0:
            return 0.0
        return self.tp_pnl / self.tp_hit

    @property
    def avg_sl_pnl(self):
        """Average P&L per SL_HIT"""
        if self.sl_hit == 0:
            return 0.0
        return self.sl_pnl / self.sl_hit

    @property
    def avg_rr(self):
        """Average Risk:Reward ratio"""
        if not self.rr_values:
            return 0.0
        return sum(self.rr_values) / len(self.rr_values)

    @property
    def highest_rr(self):
        """Highest RR in group"""
        return max(self.rr_values) if self.rr_values else 0.0

    @property
    def lowest_rr(self):
        """Lowest RR in group"""
        return min(self.rr_values) if self.rr_values else 0.0


# ============================================================================
# LOAD & FILTER DATA
# ============================================================================

def load_signals():
    """Load signals from SIGNALS_MASTER.jsonl"""
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
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} JSON error: {e}")
                continue
    
    return signals


def classify_signals(signals, tier_map):
    """Classify signals into 4 groups"""
    groups = {
        'A': GroupMetrics("GROUP A: FOUNDATION (Pre-3/4 Norm) - NON-TIERED"),
        'B': GroupMetrics("GROUP B: FRESH (Post-3/4 Norm) - NON-TIERED"),
        'C': GroupMetrics("GROUP C: FOUNDATION (Pre-3/4 Norm) - TIERED (1,2,3)"),
        'D': GroupMetrics("GROUP D: FRESH (Post-3/4 Norm) - TIERED (1,2,3)"),
    }
    
    skipped_count = 0
    for signal in signals:
        # Parse fired_time_utc (the actual entry time field in SIGNALS_MASTER)
        fired_time_str = signal.get("fired_time_utc", "")
        if not fired_time_str or not fired_time_str.strip():
            skipped_count += 1
            continue
        try:
            # Parse as UTC, then strip timezone for naive comparison
            entry_time_aware = datetime.fromisoformat(fired_time_str.replace('Z', '+00:00'))
            entry_time = entry_time_aware.replace(tzinfo=None)
            cutoff_naive = CUTOFF_TIMESTAMP.replace(tzinfo=None)
        except (ValueError, AttributeError):
            skipped_count += 1
            continue
        
        # Get tier by matching signal pattern against tier_map
        tier = get_signal_tier(signal, tier_map)
        is_tiered = tier in [1, 2, 3]
        is_pre_cutoff = entry_time < cutoff_naive
        
        # Classify
        if is_pre_cutoff and not is_tiered:
            groups['A'].signals.append(signal)
        elif not is_pre_cutoff and not is_tiered:
            groups['B'].signals.append(signal)
        elif is_pre_cutoff and is_tiered:
            groups['C'].signals.append(signal)
        elif not is_pre_cutoff and is_tiered:
            groups['D'].signals.append(signal)
    
    if skipped_count > 0:
        print(f"[TIER COMPARISON] Skipped {skipped_count} signals with missing/invalid fired_time_utc")
    
    return groups


# ============================================================================
# GENERATE REPORT
# ============================================================================

def format_group_section(group, is_tiered=False):
    """Format a group's metrics section"""
    lines = []
    lines.append("")
    lines.append(group.name)
    lines.append("─" * 80)
    
    if is_tiered and group.tier_distribution:
        tier_str = " | ".join([f"Tier-{t}: {count}" for t, count in sorted(group.tier_distribution.items())])
        lines.append(f"Tier Distribution: {tier_str}")
    
    lines.append(f"Total Signals: {group.signal_count}")
    lines.append("")
    
    lines.append("SIGNAL BREAKDOWN:")
    lines.append(f"  • TP_HIT: {group.tp_hit}")
    lines.append(f"  • SL_HIT: {group.sl_hit}")
    lines.append(f"  • TIMEOUT_WIN: {group.timeout_win}")
    lines.append(f"  • TIMEOUT_LOSS: {group.timeout_loss}")
    lines.append(f"  • OPEN: {group.open_count}")
    lines.append("")
    
    lines.append("CLOSED TRADES ANALYSIS:")
    lines.append(f"  Closed Trades: {group.closed_trades}")
    lines.append(f"  Overall WR: {group.wr_percent:.2f}%")
    lines.append(f"  Calculation: ({group.tp_hit} TP + {group.timeout_win} TIMEOUT_WIN) / {group.closed_trades} Closed = {group.tp_hit + group.timeout_win} / {group.closed_trades}")
    lines.append(f"  WR (TP & SL ONLY): {group.wr_tp_sl_only:.2f}%")
    lines.append(f"  Calculation: ({group.tp_hit} TP) / ({group.tp_hit} TP + {group.sl_hit} SL) = {group.tp_hit} / {group.tp_hit + group.sl_hit}")
    lines.append("")
    
    lines.append("P&L ANALYSIS:")
    lines.append(f"  Total P&L: ${group.total_pnl:,.2f}")
    lines.append(f"  Avg per Signal: ${group.avg_per_signal:,.2f}")
    lines.append(f"  Avg per Closed Trade: ${group.avg_per_closed_trade:,.2f}")
    lines.append(f"  Avg per TP: ${group.avg_tp_pnl:,.2f}")
    lines.append(f"  Avg per SL: ${group.avg_sl_pnl:,.2f}")
    lines.append("")
    
    lines.append("P&L BREAKDOWN:")
    lines.append(f"  • TP_HIT: ${group.tp_pnl:,.2f}")
    lines.append(f"  • SL_HIT: ${group.sl_pnl:,.2f}")
    lines.append(f"  • TIMEOUT: ${group.timeout_pnl:,.2f}")
    lines.append("")
    
    lines.append("RISK:REWARD (RR) METRICS:")
    lines.append(f"  Highest RR: {group.highest_rr:.2f}" if group.rr_values else "  Highest RR: N/A")
    lines.append(f"  Avg RR: {group.avg_rr:.2f}" if group.rr_values else "  Avg RR: N/A")
    lines.append(f"  Lowest RR: {group.lowest_rr:.2f}" if group.rr_values else "  Lowest RR: N/A")
    
    return "\n".join(lines)


def format_delta_analysis(groups):
    """Format performance delta analysis"""
    lines = []
    lines.append("")
    lines.append("═" * 80)
    lines.append("📈 PERFORMANCE DELTA ANALYSIS")
    lines.append("═" * 80)
    
    # Tiering Effect Before Normalization: C vs A
    lines.append("")
    lines.append("1️⃣ TIERING EFFECT (Before 3/4 Normalization): GROUP C vs GROUP A")
    lines.append("─" * 80)
    wr_delta_pre = groups['C'].wr_percent - groups['A'].wr_percent
    pnl_delta_pre = groups['C'].avg_per_signal - groups['A'].avg_per_signal
    lines.append(f"  WR Delta: {wr_delta_pre:+.2f}% (C: {groups['C'].wr_percent:.2f}% vs A: {groups['A'].wr_percent:.2f}%)")
    lines.append(f"  Avg P&L per Signal Delta: ${pnl_delta_pre:+,.2f} (C: ${groups['C'].avg_per_signal:,.2f} vs A: ${groups['A'].avg_per_signal:,.2f})")
    conclusion = "✅ Tiering improved performance (pre-3/4)" if wr_delta_pre > 0 and pnl_delta_pre > 0 else "❌ Tiering worsened or mixed results (pre-3/4)"
    lines.append(f"  {conclusion}")
    
    # Tiering Effect After Normalization: D vs B
    lines.append("")
    lines.append("2️⃣ TIERING EFFECT (After 3/4 Normalization): GROUP D vs GROUP B")
    lines.append("─" * 80)
    wr_delta_post = groups['D'].wr_percent - groups['B'].wr_percent
    pnl_delta_post = groups['D'].avg_per_signal - groups['B'].avg_per_signal
    lines.append(f"  WR Delta: {wr_delta_post:+.2f}% (D: {groups['D'].wr_percent:.2f}% vs B: {groups['B'].wr_percent:.2f}%)")
    lines.append(f"  Avg P&L per Signal Delta: ${pnl_delta_post:+,.2f} (D: ${groups['D'].avg_per_signal:,.2f} vs B: ${groups['B'].avg_per_signal:,.2f})")
    conclusion = "✅ Tiering improved performance (post-3/4)" if wr_delta_post > 0 and pnl_delta_post > 0 else "❌ Tiering worsened or mixed results (post-3/4)"
    lines.append(f"  {conclusion}")
    
    # 3/4 Normalization Effect on Non-Tiered: B vs A
    lines.append("")
    lines.append("3️⃣ 3/4 NORMALIZATION EFFECT (Non-Tiered): GROUP B vs GROUP A")
    lines.append("─" * 80)
    wr_delta_norm_nt = groups['B'].wr_percent - groups['A'].wr_percent
    pnl_delta_norm_nt = groups['B'].avg_per_signal - groups['A'].avg_per_signal
    lines.append(f"  WR Delta: {wr_delta_norm_nt:+.2f}% (B: {groups['B'].wr_percent:.2f}% vs A: {groups['A'].wr_percent:.2f}%)")
    lines.append(f"  Avg P&L per Signal Delta: ${pnl_delta_norm_nt:+,.2f} (B: ${groups['B'].avg_per_signal:,.2f} vs A: ${groups['A'].avg_per_signal:,.2f})")
    conclusion = "✅ 3/4 normalization improved non-tiered" if wr_delta_norm_nt > 0 and pnl_delta_norm_nt > 0 else "❌ 3/4 normalization worsened or mixed non-tiered"
    lines.append(f"  {conclusion}")
    
    # 3/4 Normalization Effect on Tiered: D vs C
    lines.append("")
    lines.append("4️⃣ 3/4 NORMALIZATION EFFECT (Tiered): GROUP D vs GROUP C")
    lines.append("─" * 80)
    wr_delta_norm_t = groups['D'].wr_percent - groups['C'].wr_percent
    pnl_delta_norm_t = groups['D'].avg_per_signal - groups['C'].avg_per_signal
    lines.append(f"  WR Delta: {wr_delta_norm_t:+.2f}% (D: {groups['D'].wr_percent:.2f}% vs C: {groups['C'].wr_percent:.2f}%)")
    lines.append(f"  Avg P&L per Signal Delta: ${pnl_delta_norm_t:+,.2f} (D: ${groups['D'].avg_per_signal:,.2f} vs C: ${groups['C'].avg_per_signal:,.2f})")
    conclusion = "✅ 3/4 normalization improved tiered" if wr_delta_norm_t > 0 and pnl_delta_norm_t > 0 else "❌ 3/4 normalization worsened or mixed tiered"
    lines.append(f"  {conclusion}")
    
    return "\n".join(lines)


def format_hypothesis_validation(groups):
    """Format hypothesis validation section"""
    lines = []
    lines.append("")
    lines.append("═" * 80)
    lines.append("🎯 HYPOTHESIS VALIDATION")
    lines.append("═" * 80)
    lines.append("")
    lines.append("THEORY: 'Tiered signals have better performance than non-tiered'")
    lines.append("")
    
    # Check all four conditions
    c_better_a_wr = groups['C'].wr_percent > groups['A'].wr_percent
    c_better_a_pnl = groups['C'].avg_per_signal > groups['A'].avg_per_signal
    d_better_b_wr = groups['D'].wr_percent > groups['B'].wr_percent
    d_better_b_pnl = groups['D'].avg_per_signal > groups['B'].avg_per_signal
    
    lines.append("EVIDENCE (Both WR and Avg P&L must improve for ✅):")
    lines.append(f"  ✅ C > A in WR? {'YES' if c_better_a_wr else 'NO'} ({groups['C'].wr_percent:.2f}% vs {groups['A'].wr_percent:.2f}%)")
    lines.append(f"  ✅ C > A in Avg P&L? {'YES' if c_better_a_pnl else 'NO'} (${groups['C'].avg_per_signal:,.2f} vs ${groups['A'].avg_per_signal:,.2f})")
    lines.append(f"  ✅ D > B in WR? {'YES' if d_better_b_wr else 'NO'} ({groups['D'].wr_percent:.2f}% vs {groups['B'].wr_percent:.2f}%)")
    lines.append(f"  ✅ D > B in Avg P&L? {'YES' if d_better_b_pnl else 'NO'} (${groups['D'].avg_per_signal:,.2f} vs ${groups['B'].avg_per_signal:,.2f})")
    lines.append("")
    
    # Overall verdict
    both_pre_better = c_better_a_wr and c_better_a_pnl
    both_post_better = d_better_b_wr and d_better_b_pnl
    
    if both_pre_better and both_post_better:
        verdict = "✅ THEORY STRONGLY SUPPORTED"
        reason = "Tiering shows consistent improvement in both pre and post 3/4 normalization"
    elif both_pre_better or both_post_better:
        verdict = "⚠️ THEORY PARTIALLY SUPPORTED"
        reason = "Tiering shows improvement in one phase but not the other"
    else:
        verdict = "❌ THEORY REFUTED"
        reason = "Tiering shows no consistent improvement over non-tiered signals"
    
    lines.append(f"VERDICT: {verdict}")
    lines.append(f"REASON: {reason}")
    lines.append("")
    
    # Recommendation
    lines.append("RECOMMENDATION FOR TIER CONFIG UPDATE:")
    if both_pre_better and both_post_better:
        lines.append("  ✅ Aggressive tier config is JUSTIFIED - tiering demonstrably improves performance")
        lines.append("  → Consider implementing the proposed Tier-1: ≥61%, Tier-2: ≥56%, Tier-3: ≥51%")
    elif both_pre_better or both_post_better:
        lines.append("  ⚠️ Mixed results - tiering works but with caveats")
        lines.append("  → Consider moderate tier tightening (e.g., +3-4% WR increase per tier)")
    else:
        lines.append("  ❌ No evidence supports tiering - maintain or relax current tier config")
        lines.append("  → Investigate why tiered signals don't outperform")
    
    return "\n".join(lines)


def generate_report():
    """Generate complete comparison report"""
    print("[TIER COMPARISON] Loading signals...")
    signals = load_signals()
    print(f"[TIER COMPARISON] Loaded {len(signals)} signals")
    
    print("[TIER COMPARISON] Loading tier mappings...")
    tier_map = load_tier_mappings()
    print(f"[TIER COMPARISON] Loaded {len(tier_map)} tier patterns")
    
    print("[TIER COMPARISON] Classifying into 4 groups...")
    groups = classify_signals(signals, tier_map)
    
    print("[TIER COMPARISON] Calculating metrics...")
    for group in groups.values():
        group.calculate()
    
    # Generate report content
    report_lines = []
    report_lines.append("═" * 80)
    report_lines.append("📊 TIER PERFORMANCE COMPARISON ANALYSIS")
    report_lines.append("═" * 80)
    gmt7_tz = timezone(timedelta(hours=7))
    now_gmt7 = datetime.now(timezone.utc).astimezone(gmt7_tz)
    report_lines.append(f"Report Generated: {now_gmt7.strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
    report_lines.append(f"Cutoff Timestamp: 2026-03-25 17:54:00 UTC (3/4 Normalization Implementation)")
    report_lines.append("")
    
    # Group sections
    report_lines.append(format_group_section(groups['A'], is_tiered=False))
    report_lines.append(format_group_section(groups['B'], is_tiered=False))
    report_lines.append(format_group_section(groups['C'], is_tiered=True))
    report_lines.append(format_group_section(groups['D'], is_tiered=True))
    
    # Delta analysis
    report_lines.append(format_delta_analysis(groups))
    
    # Hypothesis validation
    report_lines.append(format_hypothesis_validation(groups))
    
    # Footer
    report_lines.append("")
    report_lines.append("═" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Write to output file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(report_text)
    print(f"[TIER COMPARISON] Report written to {OUTPUT_FILE}")
    
    # Archive with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_file = os.path.join(ARCHIVE_DIR, f"TIER_COMPARISON_{timestamp}.txt")
    with open(archive_file, 'w') as f:
        f.write(report_text)
    print(f"[TIER COMPARISON] Archived to {archive_file}")
    
    return report_text


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    report = generate_report()
    print("\n" + report)
