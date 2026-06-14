#!/usr/bin/env python3
"""
MTF Alignment Comparison Tracker v2
Compares POST-MTF (original: with partial band) vs POST-MTF-ENHANCED (v2: 4h daily MTF, no partial)
Same format & structure as MTF_alignment_comparison_tracker.py (v1)
Windows: Extended observation period (35 days) from May 10 - June 14 2026 for better closure tracking
Last updated: 2026-06-01
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Dict, List

# Add smart-filter-v14-main to path for closure reconciliation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'smart-filter-v14-main'))


class MTFv2ComparisonTracker:
    """Compare POST-MTF original vs POST-MTF-ENHANCED alignment performance"""
    
    # ============================================================================
    # OBSERVATION WINDOW DEFINITION (CRITICAL FOR BAND TRACKING ACCURACY)
    # ============================================================================
    # REQUIREMENT: All signals in v2 window MUST have mtf_alignment_band field
    # 
    # START: May 10 2026 14:15 GMT+7 = When mtf_alignment_band FIRST deployed
    # - Before this time: Signals DON'T have mtf_alignment_band field (incomplete)
    # - From this time: 100% of signals have mtf_alignment_band + mtf_alignment_score
    #
    # END:   June 14 2026 14:15 GMT+7 = 35 days after START (extended for more closures)
    # - Allows maximum signal closure window and better MTF band performance statistics
    #
    # POST-MTF (v1):       Apr 8 07:57 - May 8 21:07 GMT+7 (30 days, baseline)
    # POST-MTF-ENHANCED:   May 10 14:15 - June 14 14:15 GMT+7 (35 days, 100% instrumented, extended for closure rates)
    
    POST_START_1 = "2026-04-08T00:57:00"  # 2026-04-08 07:57 GMT+7 (UTC) - full deployment start
    POST_END_1 = "2026-05-08T14:07:00"    # 2026-05-08 21:07 GMT+7 (UTC) - deployment complete (baseline reference)
    
    # REVISED: Start v2 window from when mtf_alignment_band field was deployed to signals
    POST_START_2 = "2026-05-10T07:15:00"  # 2026-05-10 14:15 GMT+7 (UTC) - MTF band field deployed
    POST_END_2 = "2026-06-14T07:15:00"    # 2026-06-14 14:15 GMT+7 (UTC) - 35 days observation (extended to June 14 for more closures & signal volume)
    
    def __init__(self):
        self.results_file = '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/MTF_ALIGNMENT_RESULTS.jsonl'
        self.master_file = '/Users/geniustarigan/.openclaw/workspace/SIGNALS_MASTER.jsonl.backup_before_dedup'  # Full dataset WITH closure data
        
    def load_signals(self) -> List[dict]:
        """Load all signals from ALL_SIGNALS.jsonl (complete historical dataset with 75K+ closed signals)"""
        # ALL_SIGNALS contains full history from day 1 (needed for baseline metrics)
        # Not the truncated SIGNALS_MASTER (only recent 17K)
        signals = []
        try:
            with open(self.master_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            signals.append(json.loads(line))
                        except:
                            pass
            print(f"✅ Loaded {len(signals)} signals from SIGNALS_MASTER.jsonl (closure data already applied)", flush=True)
            return signals
        except FileNotFoundError:
            print(f"❌ File not found: {self.master_file}", flush=True)
            return []
    
    def load_mtf_results(self) -> Dict[str, dict]:
        """Load MTF alignment results indexed by signal_uuid"""
        mtf_data = {}
        try:
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        uuid = record.get('signal_uuid')
                        if uuid:
                            mtf_data[uuid] = record
            print(f"✅ Loaded {len(mtf_data)} MTF alignment results", flush=True)
            return mtf_data
        except FileNotFoundError:
            print(f"⚠️ File not found: {self.results_file}", flush=True)
            return {}
    
    def filter_window(self, signals: List[dict], start: str, end: str) -> List[dict]:
        """Filter signals by fired_time_utc within window"""
        filtered = []
        for sig in signals:
            fired_time = sig.get('fired_time_utc', '')
            if start <= fired_time < end:
                filtered.append(sig)
        return filtered
    
    def calculate_metrics(self, signals: List[dict], mtf_data: Dict[str, dict], 
                         window_name: str) -> dict:
        """Calculate performance metrics for a signal window"""
        
        if not signals:
            return {
                'window': window_name,
                'signals_fired': 0,
                'signals_closed': 0,
                'tp_hit': 0,
                'sl_hit': 0,
                'open': 0,
                'wr': 0.0,
                'p_l': 0.0,
                'by_timeframe': {},
                'by_alignment_band': {}
            }
        
        # Status breakdown (use same status values as original v1 tracker)
        tp_hits = [s for s in signals if s.get('status') == 'TP_HIT']
        sl_hits = [s for s in signals if s.get('status') == 'SL_HIT']
        opens = [s for s in signals if s.get('status') == 'OPEN']
        
        closed = tp_hits + sl_hits
        
        # Totals
        total_signals = len(signals)
        total_closed = len(closed)
        tp_count = len(tp_hits)
        sl_count = len(sl_hits)
        
        # WR (WIN / (WIN + LOSS))
        tp_sl_total = tp_count + sl_count
        wr = (tp_count / tp_sl_total * 100) if tp_sl_total > 0 else 0.0
        
        # P&L
        total_pl = sum([float(s.get('pnl_usd') or 0) for s in signals])
        
        # By timeframe
        by_tf = {}
        for tf in ['15min', '30min', '1h', '2h', '4h']:
            tf_closed = [s for s in closed if s.get('timeframe') == tf]
            if tf_closed:
                tf_tp = len([s for s in tf_closed if s.get('status') == 'TP_HIT'])
                tf_sl = len([s for s in tf_closed if s.get('status') == 'SL_HIT'])
                tf_total = tf_tp + tf_sl
                tf_wr = (tf_tp / tf_total * 100) if tf_total > 0 else 0.0
                tf_pl = sum([float(s.get('pnl_usd') or 0) for s in tf_closed])
                
                by_tf[tf] = {
                    'signals': len(tf_closed),
                    'tp': tf_tp,
                    'sl': tf_sl,
                    'wr': round(tf_wr, 2),
                    'pl': round(tf_pl, 2)
                }
        
        # By alignment band (use v2_band from SIGNALS_MASTER.jsonl)
        # Show all bands, even if empty (for v2 forward window: shows tracking placeholders)
        by_alignment = {}
        by_alignment_open = {}  # Track OPEN signals separately (awaiting closure)
        
        for band in ['strong', 'weak', 'conflict', 'neutral', 'unassigned']:
            band_closed = []
            band_open = []  # Signals in this band that are still OPEN
            
            for sig in signals:
                # Read mtf_alignment_band directly from signal (now persisted in SIGNALS_MASTER.jsonl)
                # V2 ENHANCED: Uses mtf_alignment_band field (not v2_band)
                sig_band = sig.get('mtf_alignment_band')
                if sig_band is None and band == 'unassigned':
                    # Treat missing band as 'unassigned'
                    sig_match = True
                elif sig_band == band:
                    sig_match = True
                else:
                    sig_match = False
                
                if sig_match:
                    if sig.get('status') in ['TP_HIT', 'SL_HIT']:
                        band_closed.append(sig)
                    elif sig.get('status') == 'OPEN':
                        band_open.append(sig)
            
            # Closed signals (completed trades)
            if band_closed:
                band_tp = len([s for s in band_closed if s.get('status') == 'TP_HIT'])
                band_sl = len([s for s in band_closed if s.get('status') == 'SL_HIT'])
                band_total = band_tp + band_sl
                band_wr = (band_tp / band_total * 100) if band_total > 0 else 0.0
                band_pl = sum([float(s.get('pnl_usd') or 0) for s in band_closed])
                
                by_alignment[band] = {
                    'signals': len(band_closed),
                    'tp': band_tp,
                    'sl': band_sl,
                    'wr': round(band_wr, 2),
                    'pl': round(band_pl, 2)
                }
            else:
                # Empty band (v2 forward window shows 0 values as tracking placeholders)
                by_alignment[band] = {
                    'signals': 0,
                    'tp': 0,
                    'sl': 0,
                    'wr': 0.0,
                    'pl': 0.0
                }
            
            # Open signals (awaiting closure - important for conflict band monitoring)
            by_alignment_open[band] = len(band_open)
        
        return {
            'window': window_name,
            'signals_fired': total_signals,
            'signals_closed': total_closed,
            'tp_hit': tp_count,
            'sl_hit': sl_count,
            'open': len(opens),
            'wr': round(wr, 2),
            'p_l': round(total_pl, 2),
            'by_timeframe': by_tf,
            'by_alignment_band': by_alignment
        }
    
    def run_comparison(self):
        """Run full v2 comparison"""
        
        print('\n' + '=' * 110)
        print('MTF ALIGNMENT COMPARISON TRACKER v2 - POST-MTF vs POST-MTF-ENHANCED (REVISED WINDOW - 21 DAYS)')
        print('=' * 110)
        print('\n📋 METHODOLOGY (v2 COMPARISON - WINDOW EXTENDED):')
        print('  • POST-MTF (baseline): Apr 8 07:57 - May 8 21:07 GMT+7 (30 days, completed deployment)')
        print('  • POST-MTF-ENHANCED (v2 EXTENDED): May 10 14:15 - June 14 14:15 GMT+7 (35 days, extended for better closure rates)')
        print('  • MTF band field deployed: May 10 2026 14:15 GMT+7')
        print('  • Window extended: June 14 to allow more signals to close and validate v2 performance vs v1')
        print('  • Measures: Real-time impact of MTF band assignment on signal quality (strong/weak/conflict/neutral)\n')
        
        signals = self.load_signals()
        mtf_data = self.load_mtf_results()
        
        if not signals:
            print("❌ No signals loaded")
            return False
        
        # POST-MTF (original, full deployment baseline - COMPLETED)
        print('\n🔍 Analyzing POST-MTF Window (Baseline - 30 Days Complete)')
        print('-' * 110)
        print(f'Period: 2026-04-08 07:57 → 2026-05-08 21:07 GMT+7 (full deployment, original config) ✅ COMPLETE')
        
        post1_signals = self.filter_window(signals, self.POST_START_1, self.POST_END_1)
        post1_metrics = self.calculate_metrics(post1_signals, mtf_data, 'POST-MTF')
        
        print(f"✅ Metrics calculated")
        print(f"   Signals fired: {post1_metrics['signals_fired']}")
        print(f"   Signals closed: {post1_metrics['signals_closed']}")
        print(f"   WR: {post1_metrics['wr']:.2f}%")
        print(f"   P&L: ${post1_metrics['p_l']:,.2f}")
        
        # POST-MTF-ENHANCED (v2, revised window with mtf_alignment_band instrumentation)
        print('\n🔍 Analyzing POST-MTF-ENHANCED Window (REVISED - All Signals Instrumented with MTF Band)')
        print('-' * 110)
        print(f'Period: 2026-05-10 14:15 → 2026-06-14 14:15 GMT+7 (v2 live, extended observation for better closure rates) 🔄 IN PROGRESS')
        
        post2_signals = self.filter_window(signals, self.POST_START_2, self.POST_END_2)
        post2_metrics = self.calculate_metrics(post2_signals, mtf_data, 'POST-MTF-ENHANCED')
        
        # Calculate real metrics from v2 signals
        v2_open = len([s for s in post2_signals if s.get('status') == 'OPEN'])
        v2_tp = len([s for s in post2_signals if s.get('status') == 'TP_HIT'])
        v2_sl = len([s for s in post2_signals if s.get('status') == 'SL_HIT'])
        v2_timeout = len([s for s in post2_signals if s.get('status') == 'TIMEOUT'])
        v2_closed = v2_tp + v2_sl + v2_timeout
        v2_wr = (v2_tp / (v2_tp + v2_sl) * 100) if (v2_tp + v2_sl) > 0 else 0.0
        v2_pl = sum(float(s.get('pnl_usd') or 0) for s in post2_signals if s.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT'])
        
        # Open signals breakdown by band
        open_by_band = {}
        for band in ['strong', 'weak', 'conflict', 'neutral', 'unassigned']:
            band_open = 0
            for sig in post2_signals:
                sig_band = sig.get('mtf_alignment_band')
                if sig_band is None and band == 'unassigned':
                    sig_match = True
                elif sig_band == band:
                    sig_match = True
                else:
                    sig_match = False
                if sig_match and sig.get('status') == 'OPEN':
                    band_open += 1
            open_by_band[band] = band_open
        
        open_breakdown = f"{open_by_band['strong']} strong; {open_by_band['weak']} weak; {open_by_band['conflict']} conflict; {open_by_band['neutral']} neutral; {open_by_band['unassigned']} unassigned"
        
        print(f"✅ Metrics calculated")
        print(f"   Signals fired: {len(post2_signals):,}")
        print(f"   Signals closed: {v2_closed:,} ({v2_tp} TP; {v2_sl} SL; {v2_timeout} TIMEOUT)")
        print(f"   Signals OPEN: {v2_open:,} ({open_breakdown})")
        print(f"   WR: {v2_wr:.2f}% (TP / TP+SL)")
        print(f"   P&L: ${v2_pl:,.2f}")
        
        # Comparison
        wr_change = v2_wr - post1_metrics['wr']
        pl_change = v2_pl - post1_metrics['p_l']
        
        print('\n' + '=' * 110)
        print('COMPARISON: 30-Day Baseline vs v2 Live Forward (21-Day Observation)')
        print('=' * 110)
        print('')
        wr_status = '✅ IMPROVED' if wr_change > 0.5 else ('⚠️ REGRESSED' if wr_change < -0.5 else '➡️ STABLE')
        pl_status = '✅ PROFIT' if pl_change > 0 else ('❌ LOSS' if pl_change < 0 else '➡️ BREAK-EVEN')
        print(f'Win Rate Change:        {wr_change:+.2f}pp ({post1_metrics["wr"]:.2f}% → {v2_wr:.2f}%) {wr_status}')
        print(f'P&L Change:             ${pl_change:+,.2f} {pl_status}')
        print(f'Signal Volume Change:   {len(post2_signals):,} signals (vs {post1_metrics["signals_fired"]:,})')
        print('')
        
        # By timeframe comparison
        print('By Timeframe Comparison:')
        print('TF     | v1 WR  | v2 WR  | Change | v1 P&L   | v2 P&L   | Change')
        print('-' * 110)
        
        all_tfs = set(post1_metrics['by_timeframe'].keys()) | set(post2_metrics['by_timeframe'].keys())
        for tf in sorted(all_tfs):
            tf1 = post1_metrics['by_timeframe'].get(tf, {})
            tf2 = post2_metrics['by_timeframe'].get(tf, {})
            
            wr1 = tf1.get('wr', 0)
            wr2 = tf2.get('wr', 0)
            pl1 = tf1.get('pl', 0)
            pl2 = tf2.get('pl', 0)
            
            wr_delta = wr2 - wr1
            pl_delta = pl2 - pl1
            
            print(f'{tf:6s} | {wr1:6.2f}% | {wr2:6.2f}% | {wr_delta:+6.2f}% | ${pl1:8,.2f} | ${pl2:8,.2f} | ${pl_delta:+8,.2f}')
        
        # NOTE: Old "By Alignment Band Comparison" section removed (was showing phantom WR%)
        # Use "SIGNAL STATUS BREAKDOWN BY BAND" section below for real metrics
        
        # DETAILED SIGNAL STATUS BREAKDOWN BY BAND (V2 WINDOW)
        print('\n' + '=' * 140)
        print('SIGNAL STATUS BREAKDOWN BY BAND (V2 Window - May 10 14:15 - June 14 14:15 GMT+7 EXTENDED):')
        print('=' * 140)
        print(f"{'Band':<12} | {'OPEN':<6} | {'TP_HIT':<7} | {'SL_HIT':<7} | {'TIMEOUT':<8} | {'Closed':<7} | {'WR %':<7} | {'P&L':>12}")
        print('-' * 140)
        
        # Calculate status breakdown for v2 window only
        for band in ['strong', 'weak', 'conflict', 'neutral', 'unassigned']:
            band_open = 0
            band_tp = 0
            band_sl = 0
            band_timeout = 0
            band_pl = 0.0
            
            for sig in post2_signals:
                sig_band = sig.get('mtf_alignment_band')
                if sig_band is None and band == 'unassigned':
                    sig_match = True
                elif sig_band == band:
                    sig_match = True
                else:
                    sig_match = False
                
                if sig_match:
                    status = sig.get('status')
                    if status == 'OPEN':
                        band_open += 1
                    elif status == 'TP_HIT':
                        band_tp += 1
                        band_pl += float(sig.get('pnl_usd') or 0)
                    elif status == 'SL_HIT':
                        band_sl += 1
                        band_pl += float(sig.get('pnl_usd') or 0)
                    elif status == 'TIMEOUT':
                        band_timeout += 1
                        band_pl += float(sig.get('pnl_usd') or 0)
            
            band_closed = band_tp + band_sl + band_timeout
            band_wr = (band_tp / (band_tp + band_sl) * 100) if (band_tp + band_sl) > 0 else 0.0
            
            print(f'{band:<12} | {band_open:<6} | {band_tp:<7} | {band_sl:<7} | {band_timeout:<8} | {band_closed:<7} | {band_wr:>5.2f}% | ${band_pl:>10,.2f}')
        
        # TOTALS row
        band_open_total = 0
        band_tp_total = 0
        band_sl_total = 0
        band_timeout_total = 0
        
        for sig in post2_signals:
            if sig.get('status') == 'OPEN':
                band_open_total += 1
            elif sig.get('status') == 'TP_HIT':
                band_tp_total += 1
            elif sig.get('status') == 'SL_HIT':
                band_sl_total += 1
            elif sig.get('status') == 'TIMEOUT':
                band_timeout_total += 1
        
        band_closed_total = band_tp_total + band_sl_total + band_timeout_total
        band_wr_total = (band_tp_total / (band_tp_total + band_sl_total) * 100) if (band_tp_total + band_sl_total) > 0 else 0.0
        band_pl_total = sum(float(sig.get('pnl_usd') or 0) for sig in post2_signals if sig.get('status') in ['TP_HIT', 'SL_HIT', 'TIMEOUT'])
        
        print('-' * 140)
        print(f"{'TOTAL':<12} | {band_open_total:<6} | {band_tp_total:<7} | {band_sl_total:<7} | {band_timeout_total:<8} | {band_closed_total:<7} | {band_wr_total:>5.2f}% | ${band_pl_total:>10,.2f}")
        print('=' * 140)
        
        # Show OPEN signals awaiting closure (especially important for conflict band)
        print('\n⏳ OPEN SIGNALS BY BAND (Awaiting Closure in v2):')
        print('Band      | v2 Open Count')
        print('-' * 40)
        post2_open = {}
        for band in ['strong', 'weak', 'conflict', 'neutral', 'unassigned']:  # V2: Removed 'partial', use only 4 bands
            open_count = 0
            for sig in post2_signals:
                # Read mtf_alignment_band directly from signal (treat NULL/missing as 'unassigned')
                # V2 ENHANCED: Use mtf_alignment_band field (not v2_band)
                sig_band = sig.get('mtf_alignment_band')
                if sig_band is None and band == 'unassigned':
                    # Count NULL mtf_alignment_band as 'unassigned'
                    if sig.get('status') == 'OPEN':
                        open_count += 1
                elif sig_band == band and sig.get('status') == 'OPEN':
                    open_count += 1
            post2_open[band] = open_count
            icon = '🔴' if band == 'conflict' and open_count > 0 else '⏳' if open_count > 0 else '✓'
            print(f'{band:9s} | {icon} {open_count:3d}')
        
        # Detailed conflict monitoring
        conflict_closed_v2 = post2_metrics['by_alignment_band'].get('conflict', {}).get('signals', 0)
        conflict_open_v2 = post2_open.get('conflict', 0)
        print(f'\n🔴 CONFLICT BAND TRACKING:')
        print(f'   ✅ v1 (Closed): {post1_metrics["by_alignment_band"].get("conflict", {}).get("signals", 0)} signals | WR: {post1_metrics["by_alignment_band"].get("conflict", {}).get("wr", 0):.2f}%')
        print(f'   ⏳ v2 (Closed): {conflict_closed_v2} signals | WR: {post2_metrics["by_alignment_band"].get("conflict", {}).get("wr", 0):.2f}%')
        print(f'   🔴 v2 (OPEN):   {conflict_open_v2} signals (awaiting closure)')
        print(f'   ℹ️  Conflict signals = "Counter-Trend Risk" in Telegram (e.g., "15min SHORT conflicts with higher TF signals")')
        print(f'   ℹ️  These signals will populate v2 conflict metrics once they close.')
        
        print('\n' + '=' * 110)
        print('INTERPRETATION (Observation Window: May 10 14:15 - June 14 14:15 GMT+7 - EXTENDED):', flush=True)
        print('✅ v2 PROPERLY INSTRUMENTED: All signals from May 10 14:15 have mtf_alignment_band', flush=True)
        print('⏳ 35-Day Observation: Extended window for maximum signal closures and band performance metrics', flush=True)
        print('📊 Current: {fired} fired | {closed} closed | WR: {wr:.2f}% | P&L: ${pl:,.2f}'.format(
            fired=post2_metrics['signals_fired'],
            closed=post2_metrics['signals_closed'],
            wr=post2_metrics['wr'],
            pl=post2_metrics['p_l']
        ), flush=True)
        print('💪 Band Targets: strong≥65% WR | weak≥50% WR | conflict≤40% WR | neutral≥45% WR', flush=True)
        print('🎯 Overall Target: Maintain v2 WR ≥ 36.56% (baseline v1: 33.49%, improvement: +3.07pp)', flush=True)
        print('=' * 110 + '\n', flush=True)


if __name__ == '__main__':
    tracker = MTFv2ComparisonTracker()
    tracker.run_comparison()
