#!/usr/bin/env python3
"""
ASTERDEX SUMMARY REPORT - ONE-TIME SNAPSHOT
Reads open and closed position files, generates comprehensive report
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

class AsterdexSummaryReport:
    def __init__(self):
        self.open_file = Path('ASTERDEX_POSITIONS_LIVE_FINAL.jsonl')
        self.closed_file = Path('ASTERDEX_CLOSED_POSITIONS_LIVE.jsonl')
    
    def _read_positions(self, filepath, is_open=False):
        """Read positions from file (get latest if multiple entries)"""
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
                            # For open, always use latest
                            positions_by_symbol[symbol] = pos
                        else:
                            # For closed, collect all
                            if symbol not in positions_by_symbol:
                                positions_by_symbol[symbol] = []
                            positions_by_symbol[symbol].append(pos)
            
            if is_open:
                return list(positions_by_symbol.values())
            else:
                # Flatten closed positions
                all_closed = []
                for symbol, poses in positions_by_symbol.items():
                    all_closed.extend(poses)
                return all_closed
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return []
    
    def generate_report(self):
        """Generate comprehensive summary report"""
        open_pos = self._read_positions(self.open_file, is_open=True)
        closed_pos = self._read_positions(self.closed_file, is_open=False)
        
        print("\n" + "="*80)
        print("🎯 ASTERDEX POSITION SUMMARY REPORT")
        print("="*80)
        print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()
        
        # ===================== OVERVIEW =====================
        print("📊 OVERVIEW")
        print("-" * 80)
        total_open = len(open_pos)
        total_closed = len(closed_pos)
        total = total_open + total_closed
        
        print(f"  Total Positions: {total}")
        print(f"    • Open: {total_open}")
        print(f"    • Closed: {total_closed}")
        print()
        
        # ===================== OPEN POSITIONS =====================
        if open_pos:
            print("🟢 OPEN POSITIONS ({})".format(total_open))
            print("-" * 80)
            
            total_unrealized = 0
            long_count = 0
            short_count = 0
            
            for pos in sorted(open_pos, key=lambda p: abs(p.get('unrealized_pnl_usd', 0)), reverse=True):
                symbol = pos.get('symbol')
                side = pos.get('side')
                qty = pos.get('quantity', 0)
                entry = pos.get('entry_price', 0)
                mark = pos.get('current_mark_price', 0)
                unrealized = pos.get('unrealized_pnl_usd', 0)
                unrealized_pct = pos.get('unrealized_pnl_pct', 0)
                
                if side == 'LONG':
                    long_count += 1
                else:
                    short_count += 1
                
                total_unrealized += unrealized
                
                status = "📈" if unrealized > 0 else "📉" if unrealized < 0 else "➡️"
                
                print(f"  {status} {symbol:12} {side:5} {qty:>8.2f} @ {entry:>10.8f} | Mark: {mark:>10.8f} | {unrealized:>+7.2f} ({unrealized_pct:>+6.2f}%)")
            
            print()
            print(f"  LONG: {long_count} | SHORT: {short_count}")
            print(f"  Total Unrealized P&L: ${total_unrealized:+.2f}")
            print()
        
        # ===================== CLOSED POSITIONS =====================
        if closed_pos:
            print("⚫ CLOSED POSITIONS ({})".format(total_closed))
            print("-" * 80)
            
            wins = [p for p in closed_pos if p.get('pnl_usd', 0) > 0]
            losses = [p for p in closed_pos if p.get('pnl_usd', 0) < 0]
            breakeven = [p for p in closed_pos if p.get('pnl_usd', 0) == 0]
            
            win_rate = len(wins) / total_closed * 100 if total_closed > 0 else 0
            total_pnl = sum(p.get('pnl_usd', 0) for p in closed_pos)
            avg_pnl = total_pnl / total_closed if total_closed > 0 else 0
            avg_duration = sum(p.get('duration_hours', 0) for p in closed_pos) / total_closed if total_closed > 0 else 0
            
            print(f"  Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L / {len(breakeven)}BE)")
            print(f"  Total P&L: ${total_pnl:+.2f}")
            print(f"  Avg P&L/trade: ${avg_pnl:+.2f}")
            print(f"  Avg Duration: {avg_duration:.2f} hours")
            print()
            
            # Top winners
            print("  🏆 TOP 5 WINNERS:")
            for pos in sorted(wins, key=lambda p: p.get('pnl_usd', 0), reverse=True)[:5]:
                symbol = pos.get('symbol')
                pnl = pos.get('pnl_usd', 0)
                pnl_pct = pos.get('pnl_pct', 0)
                duration = pos.get('duration_hours', 0)
                print(f"     {symbol:12} +${pnl:>7.2f} (+{pnl_pct:>6.2f}%) in {duration:>5.2f}h")
            
            print()
            
            # Top losers
            if losses:
                print("  🔴 TOP 5 LOSERS:")
                for pos in sorted(losses, key=lambda p: p.get('pnl_usd', 0))[:5]:
                    symbol = pos.get('symbol')
                    pnl = pos.get('pnl_usd', 0)
                    pnl_pct = pos.get('pnl_pct', 0)
                    duration = pos.get('duration_hours', 0)
                    print(f"     {symbol:12} -${abs(pnl):>7.2f} ({pnl_pct:>6.2f}%) in {duration:>5.2f}h")
                print()
        
        # ===================== BY SYMBOL =====================
        if closed_pos:
            print("📈 PERFORMANCE BY SYMBOL (CLOSED ONLY)")
            print("-" * 80)
            
            symbol_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0, 'avg_pnl': 0})
            
            for pos in closed_pos:
                symbol = pos.get('symbol')
                pnl = pos.get('pnl_usd', 0)
                
                symbol_stats[symbol]['count'] += 1
                symbol_stats[symbol]['pnl'] += pnl
                if pnl > 0:
                    symbol_stats[symbol]['wins'] += 1
            
            # Calculate avg and sort
            for symbol in symbol_stats:
                symbol_stats[symbol]['avg_pnl'] = symbol_stats[symbol]['pnl'] / symbol_stats[symbol]['count']
            
            sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
            
            print(f"  {'Symbol':12} {'Count':>6} {'Wins':>6} {'WR%':>6} {'Total P&L':>12} {'Avg P&L':>10}")
            print("  " + "-" * 76)
            
            for symbol, stats in sorted_symbols[:20]:
                wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
                status = "✅" if stats['pnl'] > 0 else "❌"
                print(f"  {status} {symbol:10} {stats['count']:>6} {stats['wins']:>6} {wr:>6.1f}% ${stats['pnl']:>10.2f} ${stats['avg_pnl']:>9.2f}")
            
            print()
        
        print("="*80)
        print()

if __name__ == "__main__":
    report = AsterdexSummaryReport()
    report.generate_report()
