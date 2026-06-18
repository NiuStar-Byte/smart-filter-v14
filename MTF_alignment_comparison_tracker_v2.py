#!/usr/bin/env python3
"""
MTF Alignment Comparison Tracker v2 - FRESH START (2026-06-18)
Observation Window: June 18 2026 00:00 GMT+7 onwards (fresh baseline, no historical comparison)
Last updated: 2026-06-18
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
    # OBSERVATION WINDOW DEFINITION (FRESH START - 2026-06-18)
    # ============================================================================
    # Observation Window: June 18 2026 00:00 GMT+7 onwards (FRESH START - NO HISTORICAL COMPARISON)
    
    POST_START_2 = "2026-06-17T17:00:00"  # 2026-06-18 00:00 GMT+7 (UTC) - FRESH START WINDOW
    POST_END_2 = None  # Ongoing (no end date - continuous observation)
    
    def __init__(self):
        self.results_file = '/Users/geniustarigan/.openclaw/workspace/smart-filter-v14-main/MTF_ALIGNMENT_RESULTS.jsonl'
        self.master_file = '/Users/geniustarigan/.openclaw/workspace/COMPLETE_SIGNALS.jsonl'  # SINGLE SOURCE OF TRUTH - WITH closure data
        
    def load_signals(self) -> List[dict]:
        """Load all signals from COMPLETE_SIGNALS.jsonl (SINGLE SOURCE OF TRUTH with closure data)"""
        # COMPLETE_SIGNALS.jsonl contains current active signals + all closed signals with full metrics
        signals = []
        try:
            with open(self.master_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            signals.append(json.loads(line))
                        except:
                            pass
            print(f"✅ Loaded {len(signals)} signals from COMPLETE_SIGNALS.jsonl (SINGLE SOURCE OF TRUTH)", flush=True)
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
    
    def filter_window(self, signals: List[dict], start: str, end: str = None) -> List[dict]:
        """Filter signals by fired_time_utc within window. If end is None, include all signals from start onwards."""
        filtered = []
        for sig in signals:
            fired_time = sig.get('fired_time_utc', '')
            if end is None:
                # Open-ended window (from start onwards)
                if start <= fired_time:
                    filtered.append(sig)
            else:
                # Closed window (from start to end)
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
        """Run fresh start observation (2026-06-18 onwards)"""
        
        print('\n' + '=' * 110)
        print('MTF ALIGNMENT TRACKER v2 - FRESH START OBSERVATION (2026-06-18 00:00 GMT+7)')
        print('=' * 110)
        print('\n📋 OBSERVATION WINDOW:')
        print('  • Fresh Start: June 18 2026 00:00 GMT+7 onwards')
        print('  • No historical comparison\n')
        
        signals = self.load_signals()
        mtf_data = self.load_mtf_results()
        
        if not signals:
            print("❌ No signals loaded")
            return False
        
        # FRESH START WINDOW (v2, 2026-06-18 onwards with mtf_alignment_band instrumentation)
        print('\n🔍 Analyzing Fresh Start Window (2026-06-18 00:00 GMT+7 onwards)')
        print('-' * 110)
        print(f'Period: 2026-06-18 00:00 onwards (fresh start, all signals instrumented with MTF band) 🔄 IN PROGRESS')
        
        post2_signals = self.filter_window(signals, self.POST_START_2, self.POST_END_2)
        post2_metrics = self.calculate_metrics(post2_signals, mtf_data, 'FRESH_START')
        
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
        
        # REMOVED: Old comparison logic (no historical baseline for fresh start)
        
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
        
        # Fresh start summary
        conflict_open_v2 = post2_open.get('conflict', 0)
        print(f'\n🔴 CONFLICT BAND MONITORING:')
        print(f'   ⏳ v2 (OPEN):   {conflict_open_v2} signals (awaiting closure)')
        print(f'   ℹ️  Conflict signals = "Counter-Trend Risk" (e.g., "15min SHORT conflicts with higher TF signals")')
        
        print('\n' + '=' * 110)
        print('FRESH START OBSERVATION (2026-06-18 00:00 GMT+7 ONWARDS):', flush=True)
        print('✅ BASELINE ESTABLISHED: 648 signals fired from fresh start window', flush=True)
        print('⏳ OBSERVATION: Ongoing accumulation of signal closures and band performance metrics', flush=True)
        print('📊 Current: {fired} fired | {closed} closed | WR: {wr:.2f}% | P&L: ${pl:,.2f}'.format(
            fired=len(post2_signals),
            closed=v2_closed,
            wr=v2_wr,
            pl=v2_pl
        ), flush=True)
        print('💪 Band Distribution: {strong} strong | {weak} weak | {conflict} conflict | {neutral} neutral'.format(
            strong=open_by_band['strong'],
            weak=open_by_band['weak'],
            conflict=open_by_band['conflict'],
            neutral=open_by_band['neutral']
        ), flush=True)
        print('=' * 110 + '\n', flush=True)


if __name__ == '__main__':
    tracker = MTFv2ComparisonTracker()
    tracker.run_comparison()
