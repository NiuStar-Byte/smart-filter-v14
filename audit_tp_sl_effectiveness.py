#!/usr/bin/env python3
"""
AUDIT: TP/SL Effectiveness Analysis

Analyzes current TP/SL targets:
- Current hit rates by symbol
- Distance to TP vs actual close
- Average hold time per symbol
- Optimal TP/SL recommendations

Purpose: Find which symbols need dynamic TP/SL vs fixed
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta

SIGNALS_FILE = "SENT_SIGNALS.jsonl"

class TPSLAudit:
    def __init__(self):
        self.symbol_stats = defaultdict(lambda: {
            'closed': [],
            'tp_hits': 0,
            'sl_hits': 0,
            'timeouts': 0,
            'tp_distance_avg': 0,
            'sl_distance_avg': 0,
            'hold_time_avg': 0,
        })
    
    def load_signals(self):
        """Load and analyze all signals"""
        try:
            with open(SIGNALS_FILE, 'r') as f:
                for line in f:
                    try:
                        sig = json.loads(line)
                        if not sig or sig.get('status') == 'OPEN':
                            continue
                        
                        if sig.get('status') not in ['TP_HIT', 'SL_HIT', 'TIMEOUT']:
                            continue
                        
                        symbol = sig.get('symbol', 'UNKNOWN')
                        self.symbol_stats[symbol]['closed'].append(sig)
                        
                        # Count hits
                        if sig.get('status') == 'TP_HIT':
                            self.symbol_stats[symbol]['tp_hits'] += 1
                        elif sig.get('status') == 'SL_HIT':
                            self.symbol_stats[symbol]['sl_hits'] += 1
                        elif sig.get('status') == 'TIMEOUT':
                            self.symbol_stats[symbol]['timeouts'] += 1
                    except:
                        pass
        except FileNotFoundError:
            print(f"❌ {SIGNALS_FILE} not found")
    
    def analyze_tp_sl_distance(self):
        """Calculate average distance to TP/SL hit"""
        for symbol, stats in self.symbol_stats.items():
            tp_distances = []
            sl_distances = []
            hold_times = []
            
            for sig in stats['closed']:
                entry = float(sig.get('entry_price', 0))
                tp = float(sig.get('tp_target', 0))
                sl = float(sig.get('sl_target', 0))
                exit_price = float(sig.get('actual_exit_price', 0))
                
                if entry > 0:
                    if sig.get('status') == 'TP_HIT' and tp > 0:
                        distance = abs(tp - entry) / entry * 100
                        tp_distances.append(distance)
                    elif sig.get('status') == 'SL_HIT' and sl > 0:
                        distance = abs(entry - sl) / entry * 100
                        sl_distances.append(distance)
                
                # Calculate hold time
                fired_str = sig.get('fired_time_utc', '')
                closed_str = sig.get('closed_at', '')
                if fired_str and closed_str:
                    try:
                        fired = datetime.fromisoformat(fired_str.split('+')[0])
                        closed = datetime.fromisoformat(closed_str.split('+')[0])
                        hold_mins = (closed - fired).total_seconds() / 60
                        hold_times.append(hold_mins)
                    except:
                        pass
            
            # Calculate averages
            if tp_distances:
                stats['tp_distance_avg'] = sum(tp_distances) / len(tp_distances)
            if sl_distances:
                stats['sl_distance_avg'] = sum(sl_distances) / len(sl_distances)
            if hold_times:
                stats['hold_time_avg'] = sum(hold_times) / len(hold_times)
    
    def recommend_tp_sl(self, symbol):
        """Recommend optimal TP/SL for symbol"""
        stats = self.symbol_stats[symbol]
        closed = len(stats['closed'])
        
        if closed < 5:
            return None
        
        tp_hit_rate = stats['tp_hits'] / closed * 100 if closed > 0 else 0
        sl_hit_rate = stats['sl_hits'] / closed * 100 if closed > 0 else 0
        
        # Recommendations based on hit rates
        if tp_hit_rate > 60:
            tp_action = "INCREASE TP"  # Aim higher, still winning
            tp_multiplier = 1.025 if stats['tp_distance_avg'] > 0 else 1.02
        elif tp_hit_rate > 40:
            tp_action = "KEEP TP"
            tp_multiplier = 1.015
        else:
            tp_action = "DECREASE TP"  # TP too aggressive
            tp_multiplier = 1.01
        
        if sl_hit_rate > 50:
            sl_action = "TIGHTEN SL"  # SL too wide
            sl_multiplier = 0.01
        elif sl_hit_rate > 30:
            sl_action = "KEEP SL"
            sl_multiplier = 0.015
        else:
            sl_action = "WIDEN SL"  # SL too tight
            sl_multiplier = 0.02
        
        return {
            'symbol': symbol,
            'closed_trades': closed,
            'tp_hit_rate': tp_hit_rate,
            'sl_hit_rate': sl_hit_rate,
            'timeout_rate': (stats['timeouts'] / closed * 100) if closed > 0 else 0,
            'tp_action': tp_action,
            'tp_multiplier': tp_multiplier,
            'sl_action': sl_action,
            'sl_multiplier': sl_multiplier,
            'avg_hold_mins': stats['hold_time_avg'],
        }
    
    def print_report(self):
        """Generate audit report"""
        print("\n" + "="*150)
        print("🎯 TP/SL EFFECTIVENESS AUDIT")
        print("="*150)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+7')}")
        print("="*150)
        print()
        
        # Summary
        total_symbols = len(self.symbol_stats)
        total_closed = sum(len(s['closed']) for s in self.symbol_stats.values())
        
        print(f"SUMMARY:")
        print(f"  Total Symbols:    {total_symbols}")
        print(f"  Total Closed:     {total_closed}")
        print()
        
        # Per-symbol breakdown
        print("SYMBOL BREAKDOWN:")
        print()
        
        # Sort by closed trades
        sorted_symbols = sorted(
            self.symbol_stats.items(),
            key=lambda x: len(x[1]['closed']),
            reverse=True
        )
        
        print("Symbol     │ Closed │ TP Hit │ SL Hit │ Timeout │ Avg Hold │ TP Action     │ SL Action")
        print("─"*110)
        
        for symbol, stats in sorted_symbols[:30]:  # Top 30 symbols
            if len(stats['closed']) < 3:
                continue
            
            closed = len(stats['closed'])
            tp_rate = (stats['tp_hits'] / closed * 100) if closed > 0 else 0
            sl_rate = (stats['sl_hits'] / closed * 100) if closed > 0 else 0
            timeout_rate = (stats['timeouts'] / closed * 100) if closed > 0 else 0
            
            rec = self.recommend_tp_sl(symbol)
            if not rec:
                continue
            
            hold_time_str = f"{rec['avg_hold_mins']:.0f}m" if rec['avg_hold_mins'] > 0 else "N/A"
            
            print(f"{symbol:<10} │ {closed:>6} │ {tp_rate:>5.1f}% │ {sl_rate:>5.1f}% │ {timeout_rate:>6.1f}% │ {hold_time_str:>8} │ {rec['tp_action']:<13} │ {rec['sl_action']}")
        
        print()
        print("="*150)
        print("📊 RECOMMENDATIONS FOR PHASE 4A")
        print("="*150)
        print()
        
        # Group by recommendation
        need_increase_tp = []
        need_decrease_tp = []
        need_tighten_sl = []
        need_widen_sl = []
        
        for symbol, stats in sorted_symbols:
            if len(stats['closed']) < 5:
                continue
            
            rec = self.recommend_tp_sl(symbol)
            if not rec:
                continue
            
            if rec['tp_action'] == "INCREASE TP":
                need_increase_tp.append(rec)
            elif rec['tp_action'] == "DECREASE TP":
                need_decrease_tp.append(rec)
            
            if rec['sl_action'] == "TIGHTEN SL":
                need_tighten_sl.append(rec)
            elif rec['sl_action'] == "WIDEN SL":
                need_widen_sl.append(rec)
        
        if need_increase_tp:
            print(f"✅ INCREASE TP ({len(need_increase_tp)} symbols):")
            print("  Symbols have >60% TP hit rate. Can aim higher.")
            for rec in need_increase_tp[:5]:
                print(f"    {rec['symbol']}: {rec['tp_hit_rate']:.1f}% TP rate → use {rec['tp_multiplier']:.3f}x")
            print()
        
        if need_decrease_tp:
            print(f"⚠️ DECREASE TP ({len(need_decrease_tp)} symbols):")
            print("  Symbols have <40% TP hit rate. Targets too aggressive.")
            for rec in need_decrease_tp[:5]:
                print(f"    {rec['symbol']}: {rec['tp_hit_rate']:.1f}% TP rate → use {rec['tp_multiplier']:.3f}x")
            print()
        
        if need_tighten_sl:
            print(f"🔒 TIGHTEN SL ({len(need_tighten_sl)} symbols):")
            print("  Symbols have >50% SL hit rate. Stops too wide.")
            for rec in need_tighten_sl[:5]:
                print(f"    {rec['symbol']}: {rec['sl_hit_rate']:.1f}% SL rate → use {rec['sl_multiplier']:.3f}x")
            print()
        
        if need_widen_sl:
            print(f"📈 WIDEN SL ({len(need_widen_sl)} symbols):")
            print("  Symbols have <30% SL hit rate. Stops too tight.")
            for rec in need_widen_sl[:5]:
                print(f"    {rec['symbol']}: {rec['sl_hit_rate']:.1f}% SL rate → use {rec['sl_multiplier']:.3f}x")
            print()
        
        print("="*150)
        print()
        print("💡 PHASE 4A ACTION:")
        print("  1. Group symbols by recommendation")
        print("  2. Test TP/SL multipliers in backtest")
        print("  3. Deploy symbol-specific TP/SL")
        print()

if __name__ == "__main__":
    audit = TPSLAudit()
    audit.load_signals()
    audit.analyze_tp_sl_distance()
    audit.print_report()
