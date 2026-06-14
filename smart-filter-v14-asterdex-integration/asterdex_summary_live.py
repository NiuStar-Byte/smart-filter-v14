#!/usr/bin/env python3
"""
ASTERDEX SUMMARY REPORT - LIVE CONTINUOUS UPDATES
Auto-refreshes summary report every N seconds
Press Ctrl+C to stop
"""
import json
import time
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

class AsterdexSummaryLive:
    def __init__(self):
        self.open_file = Path('ASTERDEX_POSITIONS_LIVE_FINAL.jsonl')
        self.closed_file = Path('ASTERDEX_CLOSED_POSITIONS_LIVE.jsonl')
        self.last_update = None
    
    def _read_positions(self, filepath, is_open=False):
        """Read positions from file"""
        positions_by_symbol = {}
        
        if not filepath.exists():
            return []
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        pos = json.loads(line)
                        symbol = pos.get('symbol')
                        
                        if is_open:
                            positions_by_symbol[symbol] = pos
                        else:
                            if symbol not in positions_by_symbol:
                                positions_by_symbol[symbol] = []
                            positions_by_symbol[symbol].append(pos)
            
            if is_open:
                return list(positions_by_symbol.values())
            else:
                all_closed = []
                for symbol, poses in positions_by_symbol.items():
                    all_closed.extend(poses)
                return all_closed
        except:
            return []
    
    def _clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def display_report(self):
        """Display current summary report"""
        self._clear_screen()
        
        open_pos = self._read_positions(self.open_file, is_open=True)
        closed_pos = self._read_positions(self.closed_file, is_open=False)
        
        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        print("\n" + "="*80)
        print("🎯 ASTERDEX LIVE SUMMARY REPORT")
        print("="*80)
        print(f"Last Update: {now}")
        print()
        
        # ===================== OVERVIEW =====================
        total_open = len(open_pos)
        total_closed = len(closed_pos)
        total = total_open + total_closed
        
        print(f"📊 OVERVIEW: {total} total | {total_open} open | {total_closed} closed")
        print()
        
        # ===================== OPEN SUMMARY =====================
        if open_pos:
            total_unrealized = 0
            long_count = 0
            short_count = 0
            
            for pos in open_pos:
                if pos.get('side') == 'LONG':
                    long_count += 1
                else:
                    short_count += 1
                total_unrealized += pos.get('unrealized_pnl_usd', 0)
            
            print("🟢 OPEN POSITIONS ({}) | LONG: {} | SHORT: {}".format(total_open, long_count, short_count))
            print(f"   Total Unrealized: ${total_unrealized:+.2f}")
            print()
            
            # Top gainers/losers
            sorted_open = sorted(open_pos, key=lambda p: p.get('unrealized_pnl_usd', 0), reverse=True)
            
            gainers = [p for p in sorted_open if p.get('unrealized_pnl_usd', 0) > 0][:3]
            losers = [p for p in sorted_open if p.get('unrealized_pnl_usd', 0) < 0][-3:][::-1]
            
            if gainers:
                print("   ✅ Top Gainers:")
                for pos in gainers:
                    print(f"      {pos.get('symbol'):12} +${pos.get('unrealized_pnl_usd', 0):>7.2f} (+{pos.get('unrealized_pnl_pct', 0):>6.2f}%)")
            
            if losers:
                print("   ❌ Top Losers:")
                for pos in losers:
                    print(f"      {pos.get('symbol'):12} -${abs(pos.get('unrealized_pnl_usd', 0)):>7.2f} ({pos.get('unrealized_pnl_pct', 0):>6.2f}%)")
            
            print()
        
        # ===================== CLOSED SUMMARY =====================
        if closed_pos:
            wins = [p for p in closed_pos if p.get('pnl_usd', 0) > 0]
            losses = [p for p in closed_pos if p.get('pnl_usd', 0) < 0]
            breakeven = [p for p in closed_pos if p.get('pnl_usd', 0) == 0]
            
            win_rate = len(wins) / total_closed * 100 if total_closed > 0 else 0
            total_pnl = sum(p.get('pnl_usd', 0) for p in closed_pos)
            avg_pnl = total_pnl / total_closed if total_closed > 0 else 0
            avg_duration = sum(p.get('duration_hours', 0) for p in closed_pos) / total_closed if total_closed > 0 else 0
            
            print("⚫ CLOSED POSITIONS ({})".format(total_closed))
            print(f"   Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L / {len(breakeven)}BE)")
            print(f"   Total P&L: ${total_pnl:+.2f} | Avg: ${avg_pnl:+.2f} | Duration: {avg_duration:.2f}h avg")
            print()
        
        # ===================== BY SYMBOL =====================
        if closed_pos:
            print("📊 TOP PERFORMERS (CLOSED)")
            print("-" * 80)
            
            symbol_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
            
            for pos in closed_pos:
                symbol = pos.get('symbol')
                pnl = pos.get('pnl_usd', 0)
                
                symbol_stats[symbol]['count'] += 1
                symbol_stats[symbol]['pnl'] += pnl
                if pnl > 0:
                    symbol_stats[symbol]['wins'] += 1
            
            # Calculate stats and sort
            for symbol in symbol_stats:
                stats = symbol_stats[symbol]
                stats['avg_pnl'] = stats['pnl'] / stats['count']
                stats['wr'] = stats['wins'] / stats['count'] * 100
            
            sorted_by_pnl = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)[:5]
            
            print(f"  {'Symbol':12} {'Count':>6} {'WR%':>6} {'Total P&L':>12} {'Avg P&L':>10}")
            print("  " + "-" * 76)
            
            for symbol, stats in sorted_by_pnl:
                status = "✅" if stats['pnl'] > 0 else "❌"
                print(f"  {status} {symbol:10} {stats['count']:>6} {stats['wr']:>6.1f}% ${stats['pnl']:>10.2f} ${stats['avg_pnl']:>9.2f}")
            
            print()
        
        print("="*80)
        print("(Auto-refreshing... Press Ctrl+C to stop)")
        print()
    
    def run(self, interval_sec=5):
        """Run continuous live report"""
        print("🚀 Starting live summary report...")
        print(f"Refresh interval: {interval_sec} seconds")
        
        try:
            while True:
                self.display_report()
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n⏹️ Stopped")

if __name__ == "__main__":
    import sys
    
    # Parse interval from command line (default 5 seconds)
    interval = 5
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except:
            pass
    
    live = AsterdexSummaryLive()
    live.run(interval_sec=interval)
