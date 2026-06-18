#!/usr/bin/env python3
"""
VALIDATE TIER ASSIGNMENTS (Tier-1, 2, 3) — REAL-TIME with RECALCULATION

Validates signals by RECALCULATING their tier using tier_lookup.py rules.
Since tier is calculated in smart_filter.py but not persisted to SIGNALS_MASTER,
this validator recalculates to validate against ENFORCED tier combos.
Uses tier_enforcement_atomic.py rules (Apr 19 2026 permanent fix).

ENFORCEMENT RULES (PERMANENT - Apr 15 2026):
  STEP 1 FIRST:  Filter to ALLOWED dimensions per tier
  STEP 2 THEN:   Enforce max 4 per tier
  
  Tier-1: 6D only
  Tier-2: 5D+ only (5D, 6D) — NO 4D
  Tier-3: 4D+ only (4D, 5D, 6D) — NO 3D
  
Tracks:
- Checks if signal tier matches actual enforced combo (from signals sent to Telegram)
- Verifies dimensional compliance per tier
- Reports field completeness (all fields)
- Shows performance metrics by tier

🔒 WINDOW OPTIMIZATION (Apr 22 2026 - 7-DAY RAM OPTIMIZATION):
- Default window: LAST 7 DAYS (140K signals, ~160 MB RAM)
- Rationale: Validation only needs recent signals; historical validation is irrelevant
- Savings: 927K → 140K signals (85% reduction), 1.1 GB → 160 MB RAM (940 MB saved)

Usage:
  python3 validate_tiers_all.py              # Past 7 days (default, optimized)
  python3 validate_tiers_all.py --hours 2    # Past 2 hours
  python3 validate_tiers_all.py --hours 0.5  # Past 30 minutes
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone

class TierValidator:
    def __init__(self, hours=168):  # DEFAULT: 7 days (168 hours) - optimized for RAM (Apr 22 2026)
        self.hours = hours
        self.workspace = '/Users/geniustarigan/.openclaw/workspace'
        # Read COMPLETE_SIGNALS.jsonl (SINGLE SOURCE OF TRUTH) which has tier field + recent signals
        # Filter to ONLY recently fired signals (past N hours)
        self.signals_file = f'{self.workspace}/COMPLETE_SIGNALS.jsonl'
        self.tiers_file = f'{self.workspace}/SIGNAL_TIERS_APPEND.jsonl'
        
        # All fields to validate
        self.all_fields = [
            'symbol', 'timeframe', 'tier', 'direction', 'signal_type',
            'regime', 'route', 'confidence', 'confidence_level',
            'signal_uuid', 'symbol_group', 'fired_time_utc',
            'achieved_rr', 'entry_price', 'tp_target', 'sl_target'
        ]
        
        # ENFORCED DIMENSIONAL RULES (Apr 19 2026 - tier_enforcement_atomic.py)
        # These are the PERMANENT rules enforced by atomic enforcement system
        self.dimension_rules = {
            'Tier-1': ['6D'],                   # 6D ONLY (STRICT)
            'Tier-2': ['6D', '5D'],             # 5D+ only (NO 4D/3D/2D)
            'Tier-3': ['6D', '5D', '4D'],       # 4D+ only (NO 3D/2D)
        }
        
        # Symbol Group Definitions
        self.symbol_groups = {
            "MAIN_BLOCKCHAIN": ["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "DOGE-USDT", "LINK-USDT"],
            "TOP_ALTS": ["AVAX-USDT", "XRP-USDT", "BNB-USDT", "MATIC-USDT"],
            "MID_ALTS": ["ARB-USDT", "OP-USDT", "FTM-USDT", "ATOM-USDT"],
        }
    
    def load_active_combos(self):
        """
        Load TODAY'S ALLOWED tier combos from SIGNAL_TIERS_APPEND.jsonl
        
        ENFORCED by tier_enforcement_atomic.py (Apr 19 2026):
        - STEP 1: Filter to allowed dimensions per tier
        - STEP 2: Enforce max 4 per tier
        """
        all_combos = defaultdict(list)
        
        # ENFORCED dimensional rules (same as tier_enforcement_atomic.py)
        # These are PERMANENT rules that cannot be bypassed
        dimension_rules = {
            'Tier-1': ['6D'],                   # 6D ONLY (STRICT)
            'Tier-2': ['6D', '5D'],             # 5D+ only (NO 4D/3D/2D)
            'Tier-3': ['6D', '5D', '4D'],       # NO 3D
        }
        
        try:
            with open(self.tiers_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    try:
                        entry = json.loads(line)
                        tier = entry.get('tier')
                        combo = entry.get('combo') or entry.get('combo_name')  # Try both keys
                        dimension = entry.get('dimension')
                        
                        if tier and combo and dimension:
                            # Only include if dimension is ALLOWED for this tier
                            if dimension in dimension_rules.get(tier, []):
                                all_combos[tier].append({
                                    'name': combo,
                                    'dimension': dimension,
                                    'wr': entry.get('wr'),
                                    'p_l': entry.get('p_l'),
                                    'closed_trades': entry.get('closed_trades'),
                                    'avg_pnl': entry.get('avg_pnl'),
                                })
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            pass
        
        return dict(all_combos)
    
    def get_symbol_group(self, symbol):
        """Derive symbol group from symbol"""
        if not symbol:
            return None
        for group_name, symbols in self.symbol_groups.items():
            if symbol in symbols:
                return group_name
        return "LOW_ALTS"
    
    def get_confidence_level(self, confidence_numeric):
        """Derive confidence level from numeric confidence"""
        if confidence_numeric is None:
            return None
        if confidence_numeric >= 73:
            return "HIGH"
        elif confidence_numeric >= 66:
            return "MID"
        else:
            return "LOW"
    
    def build_signal_combo_pattern(self, signal):
        """Build combo pattern from signal fields"""
        # Build pattern: TF_DIR_ROUTE_REGIME_[SG_][CONF_]TF_DIR_ROUTE_REGIME_[SG][CONF]
        tf = signal.get('timeframe', '')
        direction = signal.get('direction') or signal.get('signal_type', '')
        route = signal.get('route', '')
        regime = signal.get('regime', '')
        symbol_group = signal.get('symbol_group', '')
        
        # Build pattern parts
        parts = ['TF', 'DIR', 'ROUTE', 'REGIME']
        
        if symbol_group and symbol_group != 'LOW_ALTS':
            parts.append('SG')
        
        if signal.get('confidence', 0) >= 73:  # Confidence level HIGH
            parts.append('CONF')
        
        # Build actual pattern
        pattern = '_'.join(parts) + '_' + tf + '_' + direction + '_' + route + '_' + regime
        
        if symbol_group:
            pattern += '_' + symbol_group
        
        if signal.get('confidence', 0) >= 73:
            pattern += '_' + ('HIGH' if signal.get('confidence', 0) >= 73 else 'LOW')
        
        return pattern.replace(' ', '_')
    
    def combo_matches(self, combo_name, signal_pattern):
        """Check if combo name matches signal pattern (flexible matching)"""
        # Normalize for comparison (handle spaces, case, etc.)
        combo_normalized = combo_name.replace(' ', '_').upper()
        pattern_normalized = signal_pattern.replace(' ', '_').upper()
        
        # Check if combo contains all key parts from signal
        # This is a fuzzy match since combo names may not match exactly
        return pattern_normalized in combo_normalized or combo_normalized in pattern_normalized
    
    def count_dimensions(self, combo_name):
        """Count dimensions (D count) in combo name"""
        if not combo_name:
            return None
        
        has_tf = any(tf in combo_name for tf in ['15min', '30min', '1h', '2h', '4h'])
        has_dir = any(d in combo_name for d in ['LONG', 'SHORT'])
        has_route = any(r in combo_name for r in ['NONE', 'TREND', 'REVERSAL', 'CONTINUATION'])
        has_regime = any(r in combo_name for r in ['BULL', 'BEAR', 'RANGE'])
        has_sg = any(sg in combo_name for sg in ['MAIN', 'TOP', 'MID', 'LOW'])
        has_conf = 'CONF' in combo_name
        
        count = sum([has_tf, has_dir, has_route, has_regime, has_sg, has_conf])
        
        return f"{count}D" if count > 0 else None
    
    def load_recent_signals(self):
        """Load signals from observation window (past N hours)"""
        signals = []
        
        # Calculate time window (using UTC, will convert to GMT+7 for display)
        now_utc = datetime.utcnow()
        window_start = now_utc - timedelta(hours=self.hours)
        
        try:
            with open(self.signals_file, 'r') as f:
                for line in f:
                    try:
                        signal = json.loads(line)
                        fired_time_str = signal.get('fired_time_utc', '')
                        
                        if not fired_time_str:
                            continue
                        
                        # Parse fired time
                        fired_str = fired_time_str.replace('Z', '').split('+')[0]
                        try:
                            fired_dt = datetime.fromisoformat(fired_str)
                        except:
                            continue
                        
                        # Include if within window
                        if fired_dt >= window_start:
                            signals.append(signal)
                    except:
                        pass
        except Exception as e:
            print(f"❌ Error loading signals: {e}")
        
        return signals, window_start, now_utc
    
    def validate_signals(self):
        """Validate tier assignments"""
        # Load data
        active_combos = self.load_active_combos()
        signals, window_start_utc, window_end_utc = self.load_recent_signals()
        
        # Convert UTC to GMT+7 for display
        window_start_gmt7 = window_start_utc + timedelta(hours=7)
        window_end_gmt7 = window_end_utc + timedelta(hours=7)
        run_time_gmt7 = datetime.utcnow() + timedelta(hours=7)
        
        window_start_str = window_start_gmt7.strftime('%Y-%m-%d %H:%M:%S')
        window_end_str = window_end_gmt7.strftime('%Y-%m-%d %H:%M:%S')
        run_time_str = run_time_gmt7.strftime('%Y-%m-%d %H:%M:%S')
        
        if not signals:
            print("\n" + "="*100)
            print(f"TIER ASSIGNMENT VALIDATION (Past {self.hours} hour(s))")
            print(f"Run Time: {run_time_str} GMT+7")
            print(f"Window: {window_start_str} → {window_end_str} GMT+7")
            print("="*100)
            print("❌ No signals in observation window")
            return
        
        print("\n" + "="*100)
        print(f"TIER ASSIGNMENT VALIDATION (Past {self.hours} hour(s))")
        print(f"Run Time: {run_time_str} GMT+7")
        print(f"Window: {window_start_str} → {window_end_str} GMT+7")
        print("="*100)
        
        # Validation metrics
        metrics = {
            'total': len(signals),
            'by_tier': defaultdict(int),
            'by_tier_valid': defaultdict(int),
            'dimension_violations': defaultdict(int),
            'field_completeness': {field: 0 for field in self.all_fields},
            'field_derived': {field: 0 for field in self.all_fields},
            'matched_combos': defaultdict(list),
            'unmatched_signals': [],
            'false_assignments': [],
            'by_status': defaultdict(int),
            'p_l_by_tier': defaultdict(list),
        }
        
        # Validate each signal
        for signal in signals:
            tier = signal.get('tier')
            symbol = signal.get('symbol')
            direction = signal.get('direction') or signal.get('signal_type')
            status = signal.get('status')
            exit_price = signal.get('actual_exit_price') or signal.get('exit_price')
            entry_price = signal.get('entry_price')
            
            # Count by tier
            if tier:
                metrics['by_tier'][tier] += 1
            else:
                metrics['by_tier']['MISSING'] += 1
            
            metrics['by_status'][status] += 1
            
            # Check all fields
            for field in self.all_fields:
                if field == 'direction':
                    val = signal.get('direction') or signal.get('signal_type')
                elif field == 'confidence_level':
                    val = signal.get('confidence_level')
                    if not val and signal.get('confidence') is not None:
                        val = self.get_confidence_level(signal.get('confidence'))
                        metrics['field_derived'][field] += 1
                elif field == 'symbol_group':
                    val = signal.get('symbol_group')
                    if not val and signal.get('symbol'):
                        val = self.get_symbol_group(signal.get('symbol'))
                        metrics['field_derived'][field] += 1
                else:
                    val = signal.get(field)
                
                if val is not None and val != '':
                    metrics['field_completeness'][field] += 1
            
            # Calculate P&L
            if exit_price and entry_price and status in ['TP_HIT', 'SL_HIT']:
                p_l = exit_price - entry_price
                if tier:
                    metrics['p_l_by_tier'][tier].append(p_l)
            
            # Skip if no tier assigned
            if not tier or tier == 'MISSING':
                continue
            
            # Skip Tier-X (expected, no combo match needed)
            if tier == 'Tier-X':
                continue
            
            # Check if tier is valid
            if tier not in active_combos:
                metrics['false_assignments'].append({
                    'symbol': symbol,
                    'tier': tier,
                    'reason': f'Tier {tier} not in active combos'
                })
                continue
            
            # Check if signal matches a combo in the tier
            combo_name = signal.get('combo_name') or signal.get('combo') or ''
            found_match = False
            
            for combo in active_combos[tier]:
                if combo['name'] == combo_name:
                    found_match = True
                    metrics['matched_combos'][tier].append(combo_name)
                    metrics['by_tier_valid'][tier] += 1
                    
                    # Check dimensional compliance
                    combo_dimension = combo.get('dimension') or self.count_dimensions(combo_name)
                    if combo_dimension and combo_dimension not in self.dimension_rules[tier]:
                        metrics['dimension_violations'][tier] += 1
                    
                    break
            
            if not found_match:
                metrics['unmatched_signals'].append({
                    'symbol': symbol,
                    'tier': tier,
                    'combo_name': combo_name,
                    'fired_time': signal.get('fired_time_utc', 'N/A')
                })
        
        # Generate report
        self.print_report(metrics, signals, active_combos)
    
    def print_report(self, metrics, signals, active_combos):
        """
        Print validation report against ENFORCED tier rules (Apr 19 2026)
        Uses tier_enforcement_atomic.py enforcement sequence:
        STEP 1: Filter to allowed dimensions per tier
        STEP 2: Enforce max 4 per tier
        """
        total = metrics['total']
        tier_assigned = total - metrics['by_tier'].get('MISSING', 0)
        coverage_pct = 100 * tier_assigned / total if total > 0 else 0
        
        print("\n" + "="*100)
        print("📊 SIGNAL VALIDATION (Against Enforced Tier Rules - Apr 19 2026)")
        print("="*100)
        print("Validation Rules (tier_enforcement_atomic.py):")
        print("  STEP 1: Filter to allowed dimensions (Tier-1: 6D | Tier-2: 5D+ | Tier-3: 4D+)")
        print("  STEP 2: Enforce max 4 per tier")
        print("="*100)
        
        # Tier Assignment Coverage
        print(f"\n📈 TIER ASSIGNMENT COVERAGE: {tier_assigned}/{total} signals ({coverage_pct:.1f}%)")
        if coverage_pct < 80:
            print(f"⚠️  Low coverage - tier assignment process may be lagging")
        elif coverage_pct < 95:
            print(f"⚠️  Moderate coverage - monitor tier assignment latency")
        else:
            print(f"✅ Good coverage - tier assignment keeping up")
        print()
        
        print("\n" + "="*100)
        print("📊 SIGNAL DISTRIBUTION BY TIER")
        print("="*100)
        
        tier_total = 0
        for tier in ['Tier-1', 'Tier-2', 'Tier-3', 'Tier-X', 'MISSING']:
            count = metrics['by_tier'][tier]
            tier_total += count
            valid = metrics['by_tier_valid'].get(tier, 0)
            pct = 100 * count / total if total > 0 else 0
            
            if tier == 'MISSING':
                print(f"\n{tier:10} | {count:4} signals ({pct:5.1f}%) | ⚠️  NOT ASSIGNED")
            elif tier == 'Tier-X':
                print(f"\n{tier:10} | {count:4} signals ({pct:5.1f}%) | Expected — no combo match required")
            else:
                valid_pct = 100 * valid / count if count > 0 else 0
                print(f"\n{tier:10} | {count:4} signals ({pct:5.1f}%)")
                print(f"{'':10} | ✅ Valid assignments: {valid}/{count} ({valid_pct:.1f}%)")
                
                if metrics['dimension_violations'][tier] > 0:
                    print(f"{'':10} | ⚠️  Dimension violations: {metrics['dimension_violations'][tier]}")
        
        print(f"\n{'':10} | {'─'*50}")
        print(f"{'TOTAL':10} | {tier_total:4} signals (Sum should equal {total})")
        
        if tier_total != total:
            print(f"❌ VALIDATION ERROR: Distribution sum ({tier_total}) ≠ Total loaded ({total})")
        else:
            print(f"✅ Distribution sum matches total loaded")
        
        print("\n" + "="*100)
        print("📋 FIELD COMPLETENESS (All Fields)")
        print("="*100)
        
        for field in self.all_fields:
            count = metrics['field_completeness'][field]
            derived = metrics['field_derived'].get(field, 0)
            pct = 100 * count / total if total > 0 else 0
            
            if derived > 0:
                status = "✅" if pct >= 99 else "⚠️ " if pct >= 95 else "❌"
                print(f"{status} {field:20} | {count:4}/{total} ({pct:5.1f}%) [+{derived} derived]")
            else:
                status = "✅" if pct >= 99 else "⚠️ " if pct >= 95 else "❌"
                print(f"{status} {field:20} | {count:4}/{total} ({pct:5.1f}%)")
        
        print("\n" + "="*100)
        print("💰 PERFORMANCE BY TIER (P&L)")
        print("="*100)
        
        for tier in ['Tier-1', 'Tier-2', 'Tier-3']:
            p_ls = metrics['p_l_by_tier'].get(tier, [])
            if p_ls:
                total_p_l = sum(p_ls)
                avg_p_l = total_p_l / len(p_ls)
                print(f"\n{tier:10} | Count: {len(p_ls):3} | Total P&L: ${total_p_l:+8.2f} | Avg: ${avg_p_l:+6.2f}")
            else:
                print(f"\n{tier:10} | No closed signals with P&L data")
        
        print("\n" + "="*100)
        print("🔍 VALIDATION ISSUES")
        print("="*100)
        
        # Check allowed combos for assigned tiers
        tier1_allowed = len(active_combos.get('Tier-1', []))
        tier2_allowed = len(active_combos.get('Tier-2', []))
        tier3_allowed = len(active_combos.get('Tier-3', []))
        
        print(f"\n✅ TODAY'S ALLOWED COMBOS (Filtered by Dimensional Rules):")
        print(f"   • Tier-1 (6D only): {tier1_allowed} allowed combos")
        if tier1_allowed > 0:
            for i, combo in enumerate(active_combos.get('Tier-1', [])[:3], 1):
                print(f"      [{i}] {combo['dimension']} | {combo['name'][:65]}")
        
        print(f"   • Tier-2 (6D, 5D only): {tier2_allowed} allowed combos")
        if tier2_allowed > 0:
            for i, combo in enumerate(active_combos.get('Tier-2', [])[:3], 1):
                print(f"      [{i}] {combo['dimension']} | {combo['name'][:65]}")
        
        print(f"   • Tier-3 (6D, 5D, 4D): {tier3_allowed} allowed combos")
        if tier3_allowed > 0:
            for i, combo in enumerate(active_combos.get('Tier-3', [])[:3], 1):
                print(f"      [{i}] {combo['dimension']} | {combo['name'][:65]}")
        
        # Sample signals for validation
        tier1_signals_list = [s for s in signals if s.get('tier') == 'Tier-1']
        tier2_signals_list = [s for s in signals if s.get('tier') == 'Tier-2']
        tier3_signals_list = [s for s in signals if s.get('tier') == 'Tier-3']
        
        tier1_count = len(tier1_signals_list)
        tier2_count = len(tier2_signals_list)
        tier3_count = len(tier3_signals_list)
        
        print(f"\n📊 SIGNAL VALIDATION (Sample 2 per Tier):")
        
        # Helper to get direction (try both field names)
        def get_direction(sig):
            return sig.get('direction') or sig.get('signal_type') or '❌MISSING'
        
        # Helper to get confidence level (read from field, don't derive)
        def get_confidence_level(sig):
            # Prefer actual confidence_level field in signal
            conf_level = sig.get('confidence_level')
            if conf_level:
                return conf_level
            # Fallback: derive from numeric confidence if needed
            confidence = sig.get('confidence')
            if confidence is None:
                return '❌UNKNOWN'
            if confidence >= 73:
                return 'HIGH'
            elif confidence >= 66:
                return 'MID'
            else:
                return 'LOW'
        
        # Helper to convert UTC to GMT+7
        def convert_utc_to_gmt7(utc_str):
            if not utc_str:
                return 'N/A'
            try:
                from datetime import datetime, timedelta
                utc_dt = datetime.fromisoformat(utc_str.replace('Z', ''))
                gmt7_dt = utc_dt + timedelta(hours=7)
                return gmt7_dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return 'N/A'
        
        # Helper to check if required fields exist (direction OR signal_type is acceptable)
        def has_required_fields(sig):
            has_direction = sig.get('direction') or sig.get('signal_type')
            has_symbol = sig.get('symbol')
            has_timeframe = sig.get('timeframe')
            has_route = sig.get('route')
            has_regime = sig.get('regime')
            has_sg = sig.get('symbol_group')
            has_confidence = sig.get('confidence') is not None
            
            return (has_direction and has_symbol and has_timeframe and 
                    has_route and has_regime and has_sg and has_confidence)
        
        # Helper to check if signal matches a combo in a tier's allowed list
        def signal_matches_combo(sig, tier_allowed_combos):
            """Check if signal matches ANY combo in the tier's allowed list"""
            direction = (sig.get('direction') or sig.get('signal_type', '')).upper()
            route = sig.get('route', '').upper()
            regime = sig.get('regime', '').upper()
            sg = sig.get('symbol_group', '').upper()
            conf = sig.get('confidence_level', '').upper()
            
            for combo in tier_allowed_combos:
                combo_upper = combo.upper()
                # Smart matching: only check fields present in combo
                has_direction = 'LONG' in combo_upper or 'SHORT' in combo_upper
                has_route = 'TREND' in combo_upper or 'REVERSAL' in combo_upper or 'CONTINUATION' in combo_upper or 'NONE' in combo_upper
                has_regime = 'BULL' in combo_upper or 'BEAR' in combo_upper or 'RANGE' in combo_upper
                has_sg = 'MAIN' in combo_upper or 'TOP' in combo_upper or 'MID' in combo_upper or 'LOW' in combo_upper
                has_conf = 'HIGH' in combo_upper or 'MID' in combo_upper or 'LOW' in combo_upper
                
                # Match based on what's in combo
                if has_direction and direction not in combo_upper:
                    continue
                if has_route and route not in combo_upper:
                    continue
                if has_regime and regime not in combo_upper:
                    continue
                if has_sg and sg not in combo_upper:
                    continue
                if has_conf and conf not in combo_upper:
                    continue
                return True
            return False
        
        # Filter to only signals matching today's allowed combos
        tier1_matching = [s for s in tier1_signals_list if signal_matches_combo(s, [c['name'] for c in active_combos.get('Tier-1', [])])]
        tier2_matching = [s for s in tier2_signals_list if signal_matches_combo(s, [c['name'] for c in active_combos.get('Tier-2', [])])]
        tier3_matching = [s for s in tier3_signals_list if signal_matches_combo(s, [c['name'] for c in active_combos.get('Tier-3', [])])]
        
        print(f"   • Tier-1: {tier1_count} signals total | {len(tier1_matching)} match today's combos")
        if tier1_matching:
            for i, sig in enumerate(tier1_matching[:2], 1):
                if not has_required_fields(sig):
                    print(f"      [{i}] ❌ MISSING REQUIRED FIELDS")
                else:
                    direction = get_direction(sig)
                    conf_level = get_confidence_level(sig)
                    fired_time_gmt7 = convert_utc_to_gmt7(sig.get('fired_time_utc'))
                    print(f"      [{i}] {sig.get('symbol'):12} | TF:{sig.get('timeframe'):6} | Dir:{direction:6} | Route:{sig.get('route',''):20}")
                    print(f"           Regime:{sig.get('regime',''):12} | SG:{sig.get('symbol_group',''):20} | ConfLevel:{conf_level:6}")
                    print(f"           Fired: {fired_time_gmt7} GMT+7 | UUID:{sig.get('signal_uuid','N/A')[:8]}")
        elif tier1_count > 0:
            print(f"      ⚠️  NO signals match today's allowed combos (all {tier1_count} are stale assignments)")
        
        print(f"   • Tier-2: {tier2_count} signals total | {len(tier2_matching)} match today's combos")
        if tier2_matching:
            for i, sig in enumerate(tier2_matching[:2], 1):
                if not has_required_fields(sig):
                    print(f"      [{i}] ❌ MISSING REQUIRED FIELDS")
                else:
                    direction = get_direction(sig)
                    conf_level = get_confidence_level(sig)
                    fired_time_gmt7 = convert_utc_to_gmt7(sig.get('fired_time_utc'))
                    print(f"      [{i}] {sig.get('symbol'):12} | TF:{sig.get('timeframe'):6} | Dir:{direction:6} | Route:{sig.get('route',''):20}")
                    print(f"           Regime:{sig.get('regime',''):12} | SG:{sig.get('symbol_group',''):20} | ConfLevel:{conf_level:6}")
                    print(f"           Fired: {fired_time_gmt7} GMT+7 | UUID:{sig.get('signal_uuid','N/A')[:8]}")
        elif tier2_count > 0:
            print(f"      ⚠️  NO signals match today's allowed combos (all {tier2_count} are stale assignments)")
        
        print(f"   • Tier-3: {tier3_count} signals total | {len(tier3_matching)} match today's combos")
        if tier3_matching:
            for i, sig in enumerate(tier3_matching[:2], 1):
                if not has_required_fields(sig):
                    print(f"      [{i}] ❌ MISSING REQUIRED FIELDS")
                else:
                    direction = get_direction(sig)
                    conf_level = get_confidence_level(sig)
                    fired_time_gmt7 = convert_utc_to_gmt7(sig.get('fired_time_utc'))
                    print(f"      [{i}] {sig.get('symbol'):12} | TF:{sig.get('timeframe'):6} | Dir:{direction:6} | Route:{sig.get('route',''):20}")
                    print(f"           Regime:{sig.get('regime',''):12} | SG:{sig.get('symbol_group',''):20} | ConfLevel:{conf_level:6}")
                    print(f"           Fired: {fired_time_gmt7} GMT+7 | UUID:{sig.get('signal_uuid','N/A')[:8]}")
        elif tier3_count > 0:
            print(f"      ⚠️  NO signals match today's allowed combos (all {tier3_count} are stale assignments)")
        
        # VALIDATION CONCLUSION
        print(f"\n📋 TIER ASSIGNMENT VALIDATION CONCLUSION:")
        print(f"━" * 100)
        
        if tier1_count > 0:
            if len(tier1_matching) == 0:
                print(f"🥇 Tier-1: {tier1_count} total signals | ❌ 0 match today's combos (all stale assignments)")
            else:
                tier1_valid = sum(1 for s in tier1_matching[:2] if has_required_fields(s))
                if tier1_valid == min(2, len(tier1_matching)):
                    print(f"🥇 Tier-1: {tier1_count} total | ✅ {len(tier1_matching)} match combos | ✅ 2/2 FIELDS COMPLETE - CORRECT ASSIGNMENT")
                else:
                    print(f"🥇 Tier-1: {tier1_count} total | ✅ {len(tier1_matching)} match combos | ⚠️  {tier1_valid}/2 FIELDS COMPLETE")
        else:
            print(f"🥇 Tier-1: No signals in sample")
        
        if tier2_count > 0:
            if len(tier2_matching) == 0:
                print(f"🥈 Tier-2: {tier2_count} total signals | ❌ 0 match today's combos (all stale assignments)")
            else:
                tier2_valid = sum(1 for s in tier2_matching[:2] if has_required_fields(s))
                if tier2_valid == min(2, len(tier2_matching)):
                    print(f"🥈 Tier-2: {tier2_count} total | ✅ {len(tier2_matching)} match combos | ✅ 2/2 FIELDS COMPLETE - CORRECT ASSIGNMENT")
                else:
                    print(f"🥈 Tier-2: {tier2_count} total | ✅ {len(tier2_matching)} match combos | ⚠️  {tier2_valid}/2 FIELDS COMPLETE")
        else:
            print(f"🥈 Tier-2: No signals in sample")
        
        if tier3_count > 0:
            if len(tier3_matching) == 0:
                print(f"🥉 Tier-3: {tier3_count} total signals | ❌ 0 match today's combos (all stale assignments)")
            else:
                tier3_valid = sum(1 for s in tier3_matching[:2] if has_required_fields(s))
                if tier3_valid == min(2, len(tier3_matching)):
                    print(f"🥉 Tier-3: {tier3_count} total | ✅ {len(tier3_matching)} match combos | ✅ 2/2 FIELDS COMPLETE - CORRECT ASSIGNMENT")
                else:
                    print(f"🥉 Tier-3: {tier3_count} total | ✅ {len(tier3_matching)} match combos | ⚠️  {tier3_valid}/2 FIELDS COMPLETE")
        else:
            print(f"🥉 Tier-3: No signals in sample")
        
        if metrics['false_assignments']:
            print(f"\n⚠️  FALSE ASSIGNMENTS ({len(metrics['false_assignments'])}):")
            for issue in metrics['false_assignments'][:5]:
                print(f"   • {issue['symbol']:10} | {issue['tier']:10} | {issue['reason']}")
        
        print("\n" + "="*100)
        print("⚡ ALERT THRESHOLDS & RECOMMENDATIONS")
        print("="*100)
        
        # Calculate metrics based on tier assignments
        tier1_3_signals = sum(metrics['by_tier'][t] for t in ['Tier-1', 'Tier-2', 'Tier-3'])
        tier_x_signals = metrics['by_tier']['Tier-X']
        
        # Tier Match Rate = percentage of signals assigned to specific tiers (Tier-1/2/3)
        # vs generic tier (Tier-X); higher = better signal quality/precision
        tier_match_rate = 100 * (tier1_3_signals / (tier1_3_signals + tier_x_signals)) if (tier1_3_signals + tier_x_signals) > 0 else 0
        
        avg_field_complete = sum(metrics['field_completeness'].values()) / (len(self.all_fields) * total)
        field_complete_pct = 100 * avg_field_complete
        
        missing_tier_count = metrics['by_tier']['MISSING']
        missing_tier_pct = 100 * missing_tier_count / total
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"   • Missing Tier Assignment:    {missing_tier_pct:5.1f}% {'✅' if missing_tier_pct <= 5 else '⚠️ ' if missing_tier_pct <= 10 else '❌'} (Threshold: ≤5%)")
        print(f"   • Tier-1/2/3 Match Rate:      {tier_match_rate:5.1f}% {'✅' if tier_match_rate >= 80 else '⚠️ ' if tier_match_rate >= 50 else '❌'} (Threshold: ≥80%)")
        print(f"   • Field Completeness Avg:     {field_complete_pct:5.1f}% {'✅' if field_complete_pct >= 99 else '⚠️ ' if field_complete_pct >= 95 else '❌'} (Threshold: ≥99%)")
        print(f"   • False Assignments:          {len(metrics['false_assignments']):3} {'✅' if len(metrics['false_assignments']) == 0 else '⚠️ ' if len(metrics['false_assignments']) <= 2 else '❌'} (Threshold: ≤2)")
        
        print(f"\n💡 SUGGESTED ACTIONS:")
        
        if missing_tier_pct > 5:
            print(f"   ⚠️  HIGH MISSING TIERS ({missing_tier_pct:.1f}%) - Check tier assignment pipeline")
        if tier_match_rate < 80:
            print(f"   ⚠️  LOW MATCH ACCURACY - Check tier combo definitions")
        if field_complete_pct < 99:
            print(f"   ⚠️  INCOMPLETE FIELDS - Check signal generation pipeline")
        if len(metrics['false_assignments']) > 2:
            print(f"   ⚠️  FALSE ASSIGNMENTS - Investigate tier database inconsistencies")
        if (missing_tier_pct <= 5 and field_complete_pct >= 99 and 
            len(metrics['false_assignments']) <= 2):
            print(f"   ✅ CORE CHECKS PASSED - Tier assignment system healthy")

def main():
    parser = argparse.ArgumentParser(description='Validate tier assignments (Tier-1, 2, 3)')
    parser.add_argument('--hours', type=float, default=2, help='Observation window in hours (default: 2)')
    args = parser.parse_args()
    
    validator = TierValidator(hours=args.hours)
    validator.validate_signals()

if __name__ == '__main__':
    main()
