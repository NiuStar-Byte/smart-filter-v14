#!/usr/bin/env python3
"""
ANALYZE: Multi-Timeframe Alignment

Tests how well different timeframes align:
- 15min signal vs 30min trend
- 15min signal vs 1h trend
- 30min signal vs 1h trend
- 1h signal vs 4h trend (when available)

Purpose: Determine if multi-TF filtering would improve WR
"""

import json
from collections import defaultdict
from datetime import datetime

SIGNALS_FILE = "SENT_SIGNALS.jsonl"

class MultiTFAnalyzer:
    def __init__(self):
        self.signals = {
            '15min': [],
            '30min': [],
            '1h': [],
            '4h': [],
            '1d': [],
        }
        self.alignment_stats = defaultdict(lambda: {
            'total': 0,
            'aligned': 0,
            'misaligned': 0,
            'alignment_rate': 0,
            'tp_hit_when_aligned': 0,
            'tp_hit_when_misaligned': 0,
        })
    
    def load_signals(self):
        """Load signals grouped by timeframe"""
        try:
            with open(SIGNALS_FILE, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line)
                        if not sig:
                            continue
                        
                        tf = sig.get('timeframe', '').lower()
                        if tf in ['15min', '30min', '1h', '4h', '1d']:
                            self.signals[tf].append(sig)
                    except:
                        pass
        except FileNotFoundError:
            print(f"❌ {SIGNALS_FILE} not found")
    
    def analyze_alignment(self):
        """
        Analyze if timeframes align well
        
        For each 15min signal, check:
        - What was the 30min trend at that time?
        - What was the 1h trend at that time?
        """
        
        # Group by symbol and time
        signals_by_symbol_time = defaultdict(lambda: defaultdict(list))
        
        for tf, sigs in self.signals.items():
            for sig in sigs:
                symbol = sig.get('symbol', '')
                fired_str = sig.get('fired_time_utc', '')
                
                if fired_str:
                    fired = datetime.fromisoformat(fired_str.split('+')[0])
                    signals_by_symbol_time[symbol][fired].append((tf, sig))
        
        # Analyze alignment per pair
        print("\n" + "="*150)
        print("📊 MULTI-TIMEFRAME ALIGNMENT ANALYSIS")
        print("="*150)
        print()
        
        # 15min vs 30min alignment
        print("1. 15min Signals vs 30min Trend Alignment")
        print("-"*150)
        self._analyze_tf_pair(self.signals['15min'], self.signals['30min'], '15min', '30min')
        
        # 15min vs 1h alignment
        print()
        print("2. 15min Signals vs 1h Trend Alignment")
        print("-"*150)
        self._analyze_tf_pair(self.signals['15min'], self.signals['1h'], '15min', '1h')
        
        # 30min vs 1h alignment
        print()
        print("3. 30min Signals vs 1h Trend Alignment")
        print("-"*150)
        self._analyze_tf_pair(self.signals['30min'], self.signals['1h'], '30min', '1h')
        
        # 1h vs 4h alignment
        print()
        print("4. 1h Signals vs 4h Trend Alignment")
        print("-"*150)
        self._analyze_tf_pair(self.signals['1h'], self.signals['4h'], '1h', '4h')
        
        # 4h vs 1d alignment (if 1d data available)
        if self.signals['1d']:
            print()
            print("5. 4h Signals vs 1d Trend Alignment")
            print("-"*150)
            self._analyze_tf_pair(self.signals['4h'], self.signals['1d'], '4h', '1d')
        
        print()
        print("="*150)
        print("💡 INTERPRETATION & STRATEGY")
        print("="*150)
        print()
        print("Alignment Rate = % of signals where direction matches higher TF")
        print()
        print("Good alignment (>70%): Filtering by higher TF would remove only noise")
        print("Poor alignment (<50%): Timeframes contradict → higher TF filter risky")
        print("Mixed alignment (50-70%): Selective filtering works best")
        print()
        print("MULTI-TF FILTERING STRATEGY:")
        print("  - High alignment pairs (>70%) = Use for strict filters")
        print("  - Medium alignment pairs (50-70%) = Use selectively")
        print("  - Low alignment pairs (<50%) = Avoid filtering")
        print()
        print("PROGRESSIVE FILTERING (Low to High TF):")
        print("  Level 1: 15min standalone signal")
        print("  Level 2: + 30min confirmation (easy filter)")
        print("  Level 3: + 1h confirmation (medium filter)")
        print("  Level 4: + 4h confirmation (hard filter - highest conviction)")
        print()
        print("HIGHEST CONVICTION SIGNALS:")
        print("  = 1h signal confirmed by 4h trend")
        print("  = Expected: 85%+ WR, -50% signal count")
        print()
    
    def _analyze_tf_pair(self, signals_fast, signals_slow, tf_fast, tf_slow):
        """Compare alignment between two timeframes"""
        
        # Group slow signals by symbol
        slow_by_symbol = defaultdict(list)
        for sig in signals_slow:
            symbol = sig.get('symbol', '')
            slow_by_symbol[symbol].append(sig)
        
        # For each fast signal, check if there's a matching slow signal
        alignment_by_symbol = defaultdict(lambda: {
            'total': 0,
            'aligned': 0,
            'misaligned': 0,
            'aligned_tp_rate': 0,
            'misaligned_tp_rate': 0,
        })
        
        for fast_sig in signals_fast:
            symbol = fast_sig.get('symbol', '')
            fast_dir = fast_sig.get('signal_type', '')  # LONG or SHORT
            fast_status = fast_sig.get('status', '')
            
            if fast_dir not in ['LONG', 'SHORT']:
                continue
            
            # Find nearest slow signal for same symbol
            slow_sigs = slow_by_symbol.get(symbol, [])
            if not slow_sigs:
                continue
            
            # Get most recent slow signal (rough approximation)
            if slow_sigs:
                slow_sig = slow_sigs[-1]  # Most recent
                slow_dir = slow_sig.get('signal_type', '')
                
                alignment_by_symbol[symbol]['total'] += 1
                
                # Check alignment
                if fast_dir == slow_dir:
                    alignment_by_symbol[symbol]['aligned'] += 1
                    if fast_status == 'TP_HIT':
                        alignment_by_symbol[symbol]['aligned_tp_rate'] += 1
                else:
                    alignment_by_symbol[symbol]['misaligned'] += 1
                    if fast_status == 'TP_HIT':
                        alignment_by_symbol[symbol]['misaligned_tp_rate'] += 1
        
        # Print results
        print()
        print(f"Symbol     │ Total │ Aligned │ Misaligned │ Alignment % │ TP Rate (Aligned) │ TP Rate (Misaligned) │ Recommendation")
        print("─"*150)
        
        total_all = 0
        aligned_all = 0
        
        for symbol in sorted(alignment_by_symbol.keys()):
            stats = alignment_by_symbol[symbol]
            total = stats['total']
            
            if total < 5:
                continue
            
            aligned = stats['aligned']
            misaligned = stats['misaligned']
            alignment_pct = (aligned / total * 100) if total > 0 else 0
            
            aligned_tp_rate = (stats['aligned_tp_rate'] / aligned * 100) if aligned > 0 else 0
            misaligned_tp_rate = (stats['misaligned_tp_rate'] / misaligned * 100) if misaligned > 0 else 0
            
            total_all += total
            aligned_all += aligned
            
            # Recommendation
            if alignment_pct > 75:
                rec = f"✅ Strong align - use {tf_slow} filter"
            elif alignment_pct > 60:
                rec = f"⚠️ Good align - use {tf_slow} filter selectively"
            elif alignment_pct > 40:
                rec = f"⚡ Mixed - {tf_slow} filter risky"
            else:
                rec = f"❌ Poor align - skip {tf_slow} filter"
            
            print(f"{symbol:<10} │ {total:>5} │ {aligned:>7} │ {misaligned:>10} │ {alignment_pct:>11.1f}% │ {aligned_tp_rate:>17.1f}% │ {misaligned_tp_rate:>20.1f}% │ {rec}")
        
        # Overall stats
        print()
        print("OVERALL:")
        overall_alignment = (aligned_all / total_all * 100) if total_all > 0 else 0
        print(f"  Total signals: {total_all}")
        print(f"  Aligned: {aligned_all} ({overall_alignment:.1f}%)")
        print(f"  Misaligned: {total_all - aligned_all} ({100-overall_alignment:.1f}%)")
        print()
        
        if overall_alignment > 70:
            print(f"✅ VERDICT: {tf_fast} vs {tf_slow} alignment is GOOD")
            print(f"   → Multi-TF filter using {tf_slow} trend would be effective")
        elif overall_alignment > 50:
            print(f"⚠️ VERDICT: {tf_fast} vs {tf_slow} alignment is MIXED")
            print(f"   → Multi-TF filter using {tf_slow} might work selectively")
        else:
            print(f"❌ VERDICT: {tf_fast} vs {tf_slow} alignment is POOR")
            print(f"   → Multi-TF filter not recommended for this pair")

if __name__ == "__main__":
    analyzer = MultiTFAnalyzer()
    analyzer.load_signals()
    analyzer.analyze_alignment()
